from __future__ import annotations

import hashlib
import json
import math
import os
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final, Literal, Self, override

from pydantic import Field, ValidationError, model_validator

from worldmm_smvqa.config import REMOTE_ENV_FLAG
from worldmm_smvqa.schema import FrameMetadata, FrozenModel, SourceStreamExample

SENSOR_FRAME_MANIFEST_ENV: Final = "WORLDMM_SENSOR_FRAME_MANIFEST"
SENSOR_FRAME_MANIFEST_NAME: Final = "sensor_frames.jsonl"
SENSOR_FRAME_MANIFEST_VERSION: Final = "sensor-frame-manifest-v1"
SENSOR_RATE_HZ: Final = 1.0
SENSOR_PERIOD_SECONDS: Final = 1.0
SAMPLING_POLICY: Final = "first-frame-per-1s-window"
TIMESTAMP_EPSILON: Final = 1e-9


@dataclass(frozen=True, slots=True)
class SensorFrameManifestError(Exception):
    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"SensorFrameManifestError: {self.path}: {self.detail}"


@dataclass(frozen=True, slots=True)
class SensorFrameManifestSummary:
    path: Path
    source_count: int
    source_frame_count: int
    selected_frame_count: int
    sensor_rate_hz: float


class SensorFrameSelection(FrozenModel):
    sample_index: int = Field(ge=0)
    frame_ref: str
    timestamp: float


class SensorFrameManifestRecord(FrozenModel):
    manifest_version: Literal["sensor-frame-manifest-v1"] = (
        SENSOR_FRAME_MANIFEST_VERSION
    )
    video_id: str
    sensor_rate_hz: float = SENSOR_RATE_HZ
    cadence_origin: float
    sampling_policy: Literal["first-frame-per-1s-window"] = SAMPLING_POLICY
    source_frame_count: int = Field(ge=0)
    source_frame_sha256: str
    selected_frames: tuple[SensorFrameSelection, ...]

    @model_validator(mode="after")
    def _require_1hz_contract(self) -> Self:
        if not math.isclose(self.sensor_rate_hz, SENSOR_RATE_HZ):
            msg = f"sensor_rate_hz must be {SENSOR_RATE_HZ}"
            raise ValueError(msg)
        if not math.isfinite(self.cadence_origin):
            msg = "cadence_origin must be finite"
            raise ValueError(msg)
        indices = tuple(frame.sample_index for frame in self.selected_frames)
        if indices != tuple(sorted(set(indices))):
            msg = "selected frame sample_index values must be sorted and unique"
            raise ValueError(msg)
        refs = tuple(frame.frame_ref for frame in self.selected_frames)
        if any(not frame_ref for frame_ref in refs) or len(refs) != len(set(refs)):
            msg = "selected frame_ref values must be non-empty and unique"
            raise ValueError(msg)
        if any(not math.isfinite(frame.timestamp) for frame in self.selected_frames):
            msg = "selected frame timestamps must be finite"
            raise ValueError(msg)
        return self


def build_sensor_frame_manifest(
    sources: Sequence[SourceStreamExample],
) -> tuple[SensorFrameManifestRecord, ...]:
    if not sources:
        raise SensorFrameManifestError(
            path=Path("sources.jsonl"),
            detail="source inventory has no records",
        )
    video_ids = tuple(source.video_id for source in sources)
    if len(video_ids) != len(set(video_ids)):
        raise SensorFrameManifestError(
            path=Path("sources.jsonl"),
            detail="source video_id values must be unique",
        )
    return tuple(_manifest_record(source) for source in sources)


def write_sensor_frame_manifest(
    sources: Sequence[SourceStreamExample],
    path: Path,
) -> SensorFrameManifestSummary:
    records = build_sensor_frame_manifest(sources)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            for record in records:
                _ = output.write(f"{record.model_dump_json()}\n")
        _ = temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return SensorFrameManifestSummary(
        path=path,
        source_count=len(records),
        source_frame_count=sum(record.source_frame_count for record in records),
        selected_frame_count=sum(len(record.selected_frames) for record in records),
        sensor_rate_hz=SENSOR_RATE_HZ,
    )


def read_sensor_frame_manifest(
    path: Path,
) -> tuple[SensorFrameManifestRecord, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SensorFrameManifestError(path=path, detail=str(exc)) from exc
    records: list[SensorFrameManifestRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(SensorFrameManifestRecord.model_validate_json(line))
        except ValidationError as exc:
            raise SensorFrameManifestError(
                path=path,
                detail=f"line {line_number}: {exc}",
            ) from exc
    if not records:
        raise SensorFrameManifestError(path=path, detail="manifest has no records")
    return tuple(records)


def apply_sensor_frame_manifest(
    sources: Sequence[SourceStreamExample],
    records: Sequence[SensorFrameManifestRecord],
    *,
    path: Path,
) -> tuple[SourceStreamExample, ...]:
    expected = {
        record.video_id: record for record in build_sensor_frame_manifest(sources)
    }
    actual = _records_by_video(records, path)
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise SensorFrameManifestError(
            path=path,
            detail=f"video IDs differ; missing={missing} extra={extra}",
        )
    for video_id, expected_record in expected.items():
        if actual[video_id] != expected_record:
            raise SensorFrameManifestError(
                path=path,
                detail=f"stale or modified source frame inventory: {video_id}",
            )
    return tuple(_sensed_source(source, actual[source.video_id]) for source in sources)


def configured_sensor_frame_manifest(
    fixture_dir: Path,
    *,
    explicit: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    if explicit is not None:
        return _require_manifest(explicit)
    values = os.environ if env is None else env
    if configured := values.get(SENSOR_FRAME_MANIFEST_ENV):
        return _require_manifest(Path(configured))
    local = fixture_dir / SENSOR_FRAME_MANIFEST_NAME
    if local.is_file():
        return local
    if values.get(REMOTE_ENV_FLAG) == "1":
        raise SensorFrameManifestError(
            path=local,
            detail=f"remote runs require {SENSOR_FRAME_MANIFEST_ENV}",
        )
    return None


def _manifest_record(source: SourceStreamExample) -> SensorFrameManifestRecord:
    _validate_source_frames(source)
    selected: list[SensorFrameSelection] = []
    previous_index: int | None = None
    for frame in source.frame_metadata:
        sample_index = _sample_index(frame.timestamp, source.start_time)
        if sample_index == previous_index:
            continue
        selected.append(
            SensorFrameSelection(
                sample_index=sample_index,
                frame_ref=frame.frame_ref,
                timestamp=frame.timestamp,
            ),
        )
        previous_index = sample_index
    return SensorFrameManifestRecord(
        video_id=source.video_id,
        cadence_origin=source.start_time,
        source_frame_count=len(source.frame_metadata),
        source_frame_sha256=_source_frame_sha256(source.frame_metadata),
        selected_frames=tuple(selected),
    )


def _validate_source_frames(source: SourceStreamExample) -> None:
    if not math.isfinite(source.start_time) or not math.isfinite(source.end_time):
        raise _source_error(source, "source interval must be finite")
    previous_timestamp: float | None = None
    refs: set[str] = set()
    for frame in source.frame_metadata:
        if not frame.frame_ref:
            raise _source_error(source, "frame_ref must be non-empty")
        if frame.frame_ref in refs:
            raise _source_error(source, f"duplicate frame_ref: {frame.frame_ref}")
        if not math.isfinite(frame.timestamp):
            raise _source_error(source, "frame timestamps must be finite")
        if (
            frame.timestamp < source.start_time - TIMESTAMP_EPSILON
            or frame.timestamp > source.end_time + TIMESTAMP_EPSILON
        ):
            raise _source_error(
                source,
                f"frame timestamp outside source interval: {frame.timestamp}",
            )
        if previous_timestamp is not None and frame.timestamp < previous_timestamp:
            raise _source_error(source, "frame timestamps must be sorted")
        refs.add(frame.frame_ref)
        previous_timestamp = frame.timestamp


def _source_error(source: SourceStreamExample, detail: str) -> SensorFrameManifestError:
    return SensorFrameManifestError(
        path=Path("sources.jsonl"),
        detail=f"{source.video_id}: {detail}",
    )


def _sample_index(timestamp: float, cadence_origin: float) -> int:
    relative = (timestamp - cadence_origin) / SENSOR_PERIOD_SECONDS
    return max(0, math.floor(relative + TIMESTAMP_EPSILON))


def _source_frame_sha256(frames: Sequence[FrameMetadata]) -> str:
    payload = tuple((frame.frame_ref, frame.timestamp) for frame in frames)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _records_by_video(
    records: Sequence[SensorFrameManifestRecord],
    path: Path,
) -> dict[str, SensorFrameManifestRecord]:
    result: dict[str, SensorFrameManifestRecord] = {}
    for record in records:
        if record.video_id in result:
            raise SensorFrameManifestError(
                path=path,
                detail=f"duplicate video_id: {record.video_id}",
            )
        result[record.video_id] = record
    return result


def _sensed_source(
    source: SourceStreamExample,
    record: SensorFrameManifestRecord,
) -> SourceStreamExample:
    selected_refs = frozenset(frame.frame_ref for frame in record.selected_frames)
    frame_metadata = tuple(
        frame for frame in source.frame_metadata if frame.frame_ref in selected_refs
    )
    selected_timestamps = tuple(frame.timestamp for frame in frame_metadata)
    ocr_entries = tuple(
        entry for entry in source.ocr_entries if entry.frame_ref in selected_refs
    )
    object_detections = tuple(
        detection
        for detection in source.object_detections
        if _interval_contains_timestamp(
            selected_timestamps,
            start_time=detection.start_time,
            end_time=detection.end_time,
        )
    )
    return source.model_copy(
        update={
            # Legacy captions have no timestamp/frame provenance and cannot prove
            # that they came from the sensed RGB inventory.
            "captions": (),
            "ocr": tuple(dict.fromkeys(entry.text for entry in ocr_entries)),
            "ocr_entries": ocr_entries,
            "objects": tuple(
                dict.fromkeys(detection.label for detection in object_detections),
            ),
            "object_detections": object_detections,
            "frame_refs": tuple(frame.frame_ref for frame in frame_metadata),
            "frame_metadata": frame_metadata,
        },
    )


def _interval_contains_timestamp(
    timestamps: Sequence[float],
    *,
    start_time: float,
    end_time: float,
) -> bool:
    index = bisect_left(timestamps, start_time)
    return index < len(timestamps) and timestamps[index] <= end_time


def _require_manifest(path: Path) -> Path:
    if not path.is_file():
        raise SensorFrameManifestError(path=path, detail="manifest does not exist")
    return path
