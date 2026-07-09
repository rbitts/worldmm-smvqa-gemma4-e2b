from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

from worldmm_smvqa.chunking import read_source_streams
from worldmm_smvqa.schema import (
    FrameMetadata,
    FrozenModel,
    MemoryBuilderCandidate,
    ObjectMetadata,
    OCRMetadata,
    SourceStreamExample,
    ensure_memory_builder_input,
)


@dataclass(frozen=True, slots=True)
class MissingGroundingError(Exception):
    video_id: str
    frame_ref: str

    @override
    def __str__(self) -> str:
        return f"MissingGroundingError: {self.video_id}: {self.frame_ref}"


@dataclass(frozen=True, slots=True)
class VisualBuildSummary:
    path: Path
    records: int


class VisualMemoryRecord(FrozenModel):
    record_type: Literal["visual"] = "visual"
    memory_id: str
    store: Literal["visual"] = "visual"
    video_id: str
    frame_ref: str
    timestamp: float
    start_time: float
    end_time: float
    embedding_ref: str
    ocr_refs: tuple[str, ...]
    object_refs: tuple[str, ...]
    timestamp_grounding: str
    source_frame_description: str


def build_visual_memory(
    candidates: Sequence[MemoryBuilderCandidate],
) -> tuple[VisualMemoryRecord, ...]:
    sources = tuple(_source(candidate) for candidate in candidates)
    return tuple(
        record
        for source in sorted(sources, key=lambda item: item.video_id)
        for record in _visual_records(source)
    )


def write_fixture_visual_memory(
    fixture_dir: Path,
    output: Path,
) -> VisualBuildSummary:
    records = build_visual_memory(read_source_streams(fixture_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in records),
        encoding="utf-8",
    )
    return VisualBuildSummary(path=output, records=len(records))


def _source(candidate: MemoryBuilderCandidate) -> SourceStreamExample:
    value = ensure_memory_builder_input(candidate)
    return SourceStreamExample(
        video_id=value.video_id,
        start_time=value.start_time,
        end_time=value.end_time,
        transcript=value.transcript,
        transcript_spans=value.transcript_spans,
        captions=value.captions,
        ocr=value.ocr,
        ocr_entries=value.ocr_entries,
        objects=value.objects,
        object_detections=value.object_detections,
        frame_refs=value.frame_refs,
        frame_metadata=value.frame_metadata,
    )


def _visual_records(source: SourceStreamExample) -> tuple[VisualMemoryRecord, ...]:
    metadata_by_ref = {frame.frame_ref: frame for frame in source.frame_metadata}
    records: list[VisualMemoryRecord] = []
    for frame_ref in source.frame_refs:
        frame = metadata_by_ref.get(frame_ref)
        if frame is None:
            raise MissingGroundingError(video_id=source.video_id, frame_ref=frame_ref)
        records.append(_visual_record(source, frame))
    return tuple(records)


def _visual_record(
    source: SourceStreamExample,
    frame: FrameMetadata,
) -> VisualMemoryRecord:
    timestamp_token = _format_seconds(frame.timestamp)
    memory_id = f"visual:{source.video_id}:{frame.frame_ref}:{timestamp_token}"
    return VisualMemoryRecord(
        memory_id=memory_id,
        video_id=source.video_id,
        frame_ref=frame.frame_ref,
        timestamp=frame.timestamp,
        start_time=source.start_time,
        end_time=source.end_time,
        embedding_ref=f"fixture-embedding:{frame.frame_ref}:{timestamp_token}",
        ocr_refs=_ocr_refs(frame.timestamp, source.ocr_entries),
        object_refs=_object_refs(frame.timestamp, source.object_detections),
        timestamp_grounding=f"{source.video_id}@{timestamp_token}",
        source_frame_description=frame.description,
    )


def _ocr_refs(
    timestamp: float,
    entries: Sequence[OCRMetadata],
) -> tuple[str, ...]:
    return tuple(
        entry.text
        for entry in entries
        if entry.start_time <= timestamp <= entry.end_time
    )


def _object_refs(
    timestamp: float,
    detections: Sequence[ObjectMetadata],
) -> tuple[str, ...]:
    return tuple(
        item.label
        for item in detections
        if item.start_time <= timestamp <= item.end_time
    )


def _format_seconds(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")
