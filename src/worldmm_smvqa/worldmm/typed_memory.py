from __future__ import annotations

import json
import os
from collections.abc import Sequence
from math import floor, isclose, isfinite, sqrt
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Annotated, Final, Literal, Self

from pydantic import Field, FiniteFloat, TypeAdapter, model_validator

from worldmm_smvqa.schema import FrozenModel

if TYPE_CHECKING:
    from worldmm_smvqa.schema import SourceStreamExample
    from worldmm_smvqa.sensor_frames import SensorFrameManifestRecord

type NonEmptyStr = Annotated[str, Field(min_length=1)]
type NonNegativeFiniteFloat = Annotated[FiniteFloat, Field(ge=0.0)]
type PositiveFiniteFloat = Annotated[FiniteFloat, Field(gt=0.0)]
type Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
type Quaternion = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
type Covariance3 = tuple[Vec3, Vec3, Vec3]
type MemoryProvenance = Literal[
    "observed",
    "multi_view_fused",
    "model_inferred",
    "relation_inferred",
    "human_confirmed",
]
type EventKind = Literal[
    "appeared",
    "disappeared",
    "moved",
    "opened",
    "closed",
    "interaction",
]
type WritableRecordType = Literal[
    "object",
    "plane",
    "portal",
    "free_space",
    "landmark",
    "event",
]

_COVARIANCE_TOLERANCE = 1e-9
_EVIDENCE_TIMESTAMP_TOLERANCE = 1e-9
DEFAULT_TYPED_MEMORY_WINDOW_SECONDS: Final = 30.0
MAX_TYPED_MEMORY_RECORD_BYTES: Final = 1024 * 1024


class ValidityInterval(FrozenModel):
    start_time: FiniteFloat
    end_time: FiniteFloat

    @model_validator(mode="after")
    def _require_forward_or_point_time(self) -> Self:
        if self.end_time < self.start_time:
            msg = "end_time must be greater than or equal to start_time"
            raise ValueError(msg)
        return self


class SpatialUncertainty(FrozenModel):
    covariance_xyz: Covariance3
    standard_deviation_m: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def _require_positive_semidefinite_covariance(self) -> Self:
        covariance = self.covariance_xyz
        if any(
            not isclose(
                covariance[row][column],
                covariance[column][row],
                rel_tol=_COVARIANCE_TOLERANCE,
                abs_tol=_COVARIANCE_TOLERANCE,
            )
            for row in range(3)
            for column in range(row + 1, 3)
        ):
            msg = "covariance must be symmetric"
            raise ValueError(msg)

        scale = max(1.0, *(abs(value) for row in covariance for value in row))
        normalized = tuple(
            tuple(value / scale for value in row) for row in covariance
        )
        if any(
            normalized[index][index] < -_COVARIANCE_TOLERANCE
            for index in range(3)
        ):
            msg = "covariance must be positive semidefinite"
            raise ValueError(msg)
        if any(
            (
                normalized[first][first] * normalized[second][second]
                - normalized[first][second] * normalized[second][first]
            )
            < -_COVARIANCE_TOLERANCE
            for first, second in ((0, 1), (0, 2), (1, 2))
        ):
            msg = "covariance must be positive semidefinite"
            raise ValueError(msg)
        a, b, c = normalized[0]
        _d, e, f = normalized[1]
        _g, _h, i = normalized[2]
        determinant = (
            (a * e * i)
            + (b * f * _g)
            + (c * _d * _h)
            - (c * e * _g)
            - (b * _d * i)
            - (a * f * _h)
        )
        if determinant < -_COVARIANCE_TOLERANCE:
            msg = "covariance must be positive semidefinite"
            raise ValueError(msg)
        max_variance = max(covariance[index][index] for index in range(3))
        required_deviation = sqrt(max(0.0, max_variance))
        if self.standard_deviation_m < required_deviation - (
            _COVARIANCE_TOLERANCE * max(1.0, required_deviation)
        ):
            msg = "standard_deviation_m must cover covariance diagonal"
            raise ValueError(msg)
        return self


class ObjectGeometry(FrozenModel):
    centroid: Vec3
    extent: tuple[PositiveFiniteFloat, PositiveFiniteFloat, PositiveFiniteFloat]
    orientation_xyzw: Quaternion = (0.0, 0.0, 0.0, 1.0)


class PlaneGeometry(FrozenModel):
    normal: Vec3
    offset_m: FiniteFloat
    boundary: tuple[Vec3, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def _require_nonzero_normal(self) -> Self:
        if self.normal == (0.0, 0.0, 0.0):
            msg = "plane normal must be non-zero"
            raise ValueError(msg)
        return self


class PortalGeometry(FrozenModel):
    centroid: Vec3
    normal: Vec3
    width_m: PositiveFiniteFloat
    height_m: PositiveFiniteFloat

    @model_validator(mode="after")
    def _require_nonzero_normal(self) -> Self:
        if self.normal == (0.0, 0.0, 0.0):
            msg = "portal normal must be non-zero"
            raise ValueError(msg)
        return self


class FreeSpaceGeometry(FrozenModel):
    floor_polygon: tuple[Vec3, ...] = Field(min_length=3)
    height_m: PositiveFiniteFloat


class LandmarkGeometry(FrozenModel):
    position: Vec3
    ray_direction: Vec3
    view_cone_degrees: Annotated[FiniteFloat, Field(gt=0.0, le=180.0)]

    @model_validator(mode="after")
    def _require_nonzero_ray(self) -> Self:
        if self.ray_direction == (0.0, 0.0, 0.0):
            msg = "ray_direction must be non-zero"
            raise ValueError(msg)
        return self


class EventGeometry(FrozenModel):
    before_position: Vec3 | None = None
    after_position: Vec3 | None = None


class TypedMemoryRecordBase(FrozenModel):
    memory_id: NonEmptyStr
    source_video_id: NonEmptyStr
    entity_id: NonEmptyStr
    instance_id: NonEmptyStr
    local_frame_id: NonEmptyStr
    geometry_uncertainty: SpatialUncertainty
    validity: ValidityInterval
    first_seen_time: FiniteFloat
    last_seen_time: FiniteFloat
    observation_count: int = Field(ge=1)
    confidence: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    provenance: MemoryProvenance
    evidence_refs: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _require_observations_within_validity(self) -> Self:
        if not (
            self.validity.start_time
            <= self.first_seen_time
            <= self.last_seen_time
            <= self.validity.end_time
        ):
            msg = "first_seen_time and last_seen_time must be within validity"
            raise ValueError(msg)
        return self


class ObjectMemoryRecord(TypedMemoryRecordBase):
    record_type: Literal["object"] = "object"
    geometry: ObjectGeometry
    semantic_label: NonEmptyStr
    place_label: NonEmptyStr | None = None


class PlaneMemoryRecord(TypedMemoryRecordBase):
    record_type: Literal["plane"] = "plane"
    geometry: PlaneGeometry


class PortalMemoryRecord(TypedMemoryRecordBase):
    record_type: Literal["portal"] = "portal"
    geometry: PortalGeometry
    connects_frame_ids: tuple[NonEmptyStr, NonEmptyStr]


class FreeSpaceMemoryRecord(TypedMemoryRecordBase):
    record_type: Literal["free_space"] = "free_space"
    geometry: FreeSpaceGeometry


class LandmarkMemoryRecord(TypedMemoryRecordBase):
    record_type: Literal["landmark"] = "landmark"
    geometry: LandmarkGeometry
    descriptor_ref: NonEmptyStr | None = None


class EventMemoryRecord(TypedMemoryRecordBase):
    record_type: Literal["event"] = "event"
    geometry: EventGeometry
    event_kind: EventKind
    involved_entity_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_event_positions(self) -> Self:
        before = self.geometry.before_position
        after = self.geometry.after_position
        if self.event_kind == "moved" and (before is None or after is None):
            msg = "moved event requires before_position and after_position"
            raise ValueError(msg)
        if self.event_kind == "appeared" and (before is not None or after is None):
            msg = "appeared event requires only after_position"
            raise ValueError(msg)
        if self.event_kind == "disappeared" and (before is None or after is not None):
            msg = "disappeared event requires only before_position"
            raise ValueError(msg)
        return self


class NoWriteMemoryRecord(TypedMemoryRecordBase):
    record_type: Literal["no_write"] = "no_write"
    geometry: None = None
    candidate_type: WritableRecordType
    reason: NonEmptyStr


type TypedMemoryRecord = Annotated[
    ObjectMemoryRecord
    | PlaneMemoryRecord
    | PortalMemoryRecord
    | FreeSpaceMemoryRecord
    | LandmarkMemoryRecord
    | EventMemoryRecord
    | NoWriteMemoryRecord,
    Field(discriminator="record_type"),
]


def canonical_jsonl_bytes(record: TypedMemoryRecordBase) -> bytes:
    payload = json.dumps(
        record.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{payload}\n".encode()


def serialized_byte_cost(record: TypedMemoryRecordBase) -> int:
    if isinstance(record, NoWriteMemoryRecord):
        return 0
    return len(canonical_jsonl_bytes(record))


class ScoredMemoryCandidate(FrozenModel):
    record: TypedMemoryRecord
    score: FiniteFloat


class TypedMemoryWriteSummary(FrozenModel):
    output_path: Path
    byte_budget: int = Field(gt=0)
    candidate_count: int = Field(ge=0)
    writable_candidate_count: int = Field(ge=0)
    no_write_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    skipped_for_budget_count: int = Field(ge=0)
    actual_bytes: int = Field(ge=0)
    selected_memory_ids: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def _require_consistent_counts(self) -> Self:
        if self.writable_candidate_count + self.no_write_count != self.candidate_count:
            msg = "writable_candidate_count + no_write_count must equal candidate_count"
            raise ValueError(msg)
        if (
            self.selected_count + self.skipped_for_budget_count
            != self.writable_candidate_count
        ):
            msg = "selected_count + skipped_for_budget_count must equal writable count"
            raise ValueError(msg)
        if self.selected_count != len(self.selected_memory_ids):
            msg = "selected_count must equal selected_memory_ids length"
            raise ValueError(msg)
        if self.actual_bytes > self.byte_budget:
            msg = "actual_bytes must not exceed byte_budget"
            raise ValueError(msg)
        return self


class TypedMemoryArtifactSummary(FrozenModel):
    record_count: int = Field(ge=0)
    actual_bytes: int = Field(ge=0)
    window_count: int = Field(ge=0)
    max_window_bytes: int = Field(ge=0)
    window_seconds: PositiveFiniteFloat


class TypedMemoryWriterError(ValueError):
    """Invalid writer input or persistent artifact verification failure."""


_TYPED_MEMORY_ADAPTER: Final[TypeAdapter[TypedMemoryRecord]] = TypeAdapter(
    TypedMemoryRecord
)
_EVIDENCE_REQUIRED_PROVENANCE: Final = frozenset(
    {"observed", "multi_view_fused", "model_inferred", "human_confirmed"}
)

type SourceBounds = dict[str, tuple[float, float]]
type SelectedFrameTimestamps = dict[str, dict[str, float]]
type GroundingContext = tuple[SourceBounds, SelectedFrameTimestamps]


def write_typed_memory_artifact(
    candidates: Sequence[ScoredMemoryCandidate],
    *,
    output: Path,
    byte_budget: int,
) -> TypedMemoryWriteSummary:
    """Greedily persist decoded records by stable score-per-actual-byte rank."""
    if byte_budget <= 0:
        msg = "byte_budget must be positive"
        raise TypedMemoryWriterError(msg)

    seen_ids: set[str] = set()
    writable: list[tuple[ScoredMemoryCandidate, int]] = []
    no_write_count = 0
    for candidate in candidates:
        memory_id = candidate.record.memory_id
        if memory_id in seen_ids:
            msg = f"duplicate memory_id: {memory_id}"
            raise TypedMemoryWriterError(msg)
        seen_ids.add(memory_id)
        if not isfinite(candidate.score):
            msg = f"{memory_id}: score must be finite"
            raise TypedMemoryWriterError(msg)
        if isinstance(candidate.record, NoWriteMemoryRecord):
            no_write_count += 1
            continue
        writable.append((candidate, serialized_byte_cost(candidate.record)))

    # Python's stable sort preserves decoder order for equal value-per-byte scores.
    ranked = sorted(writable, key=lambda item: -(item[0].score / item[1]))
    selected: list[TypedMemoryRecordBase] = []
    actual_bytes = 0
    for candidate, record_bytes in ranked:
        if actual_bytes + record_bytes > byte_budget:
            continue
        selected.append(candidate.record)
        actual_bytes += record_bytes

    payload = b"".join(canonical_jsonl_bytes(record) for record in selected)
    if len(payload) != actual_bytes:
        msg = "selected record byte recount mismatch"
        raise TypedMemoryWriterError(msg)
    _write_verified_artifact_atomic(output, payload, selected)

    return TypedMemoryWriteSummary(
        output_path=output,
        byte_budget=byte_budget,
        candidate_count=len(candidates),
        writable_candidate_count=len(writable),
        no_write_count=no_write_count,
        selected_count=len(selected),
        skipped_for_budget_count=len(writable) - len(selected),
        actual_bytes=actual_bytes,
        selected_memory_ids=tuple(record.memory_id for record in selected),
    )


def validate_typed_memory_artifact(  # noqa: PLR0912
    path: Path,
    *,
    byte_budget_per_window: int | None = None,
    window_seconds: float = DEFAULT_TYPED_MEMORY_WINDOW_SECONDS,
    sources: Sequence[SourceStreamExample] | None = None,
    sensor_records: Sequence[SensorFrameManifestRecord] | None = None,
) -> TypedMemoryArtifactSummary:
    """Verify persisted typed records and recount canonical bytes per time window."""
    if byte_budget_per_window is not None and byte_budget_per_window <= 0:
        msg = "byte_budget_per_window must be positive"
        raise TypedMemoryWriterError(msg)
    if not isfinite(window_seconds) or window_seconds <= 0.0:
        msg = "window_seconds must be positive and finite"
        raise TypedMemoryWriterError(msg)
    grounding_context = _build_grounding_context(sources, sensor_records)
    seen_ids: set[str] = set()
    window_bytes: dict[tuple[str, int], int] = {}
    actual_bytes = 0
    try:
        with path.open("rb") as stream:
            for line_number, raw_line in enumerate(
                iter(lambda: stream.readline(MAX_TYPED_MEMORY_RECORD_BYTES + 1), b""),
                start=1,
            ):
                actual_bytes += len(raw_line)
                if len(raw_line) > MAX_TYPED_MEMORY_RECORD_BYTES:
                    msg = f"typed memory row exceeds 1 MiB: {line_number}"
                    raise TypedMemoryWriterError(msg)
                if not raw_line.endswith(b"\n"):
                    msg = "JSONL artifact must end with a newline"
                    raise TypedMemoryWriterError(msg)
                line = raw_line[:-1]
                if not line:
                    msg = f"blank JSONL row: {line_number}"
                    raise TypedMemoryWriterError(msg)
                try:
                    record: TypedMemoryRecord = _TYPED_MEMORY_ADAPTER.validate_json(
                        line,
                    )
                except ValueError as exc:
                    msg = f"invalid typed memory row {line_number}: {exc}"
                    raise TypedMemoryWriterError(msg) from exc
                if isinstance(record, NoWriteMemoryRecord):
                    msg = "no_write record cannot be persisted"
                    raise TypedMemoryWriterError(msg)
                if record.memory_id in seen_ids:
                    msg = f"duplicate memory_id: {record.memory_id}"
                    raise TypedMemoryWriterError(msg)
                seen_ids.add(record.memory_id)
                canonical = canonical_jsonl_bytes(record)
                if canonical != raw_line:
                    msg = f"typed memory row is not canonical JSON: {line_number}"
                    raise TypedMemoryWriterError(msg)
                if grounding_context is not None:
                    _validate_record_grounding(record, line_number, grounding_context)
                window_key = (
                    record.source_video_id,
                    floor(float(record.first_seen_time) / window_seconds),
                )
                window_bytes[window_key] = window_bytes.get(window_key, 0) + len(
                    canonical,
                )
                if (
                    byte_budget_per_window is not None
                    and window_bytes[window_key] > byte_budget_per_window
                ):
                    msg = (
                        "typed memory window exceeds byte budget: "
                        f"{window_key[0]}:{window_key[1]}"
                    )
                    raise TypedMemoryWriterError(msg)
    except OSError as exc:
        msg = f"cannot read typed memory artifact: {exc}"
        raise TypedMemoryWriterError(msg) from exc

    return TypedMemoryArtifactSummary(
        record_count=len(seen_ids),
        actual_bytes=actual_bytes,
        window_count=len(window_bytes),
        max_window_bytes=max(window_bytes.values(), default=0),
        window_seconds=window_seconds,
    )


def _build_grounding_context(
    sources: Sequence[SourceStreamExample] | None,
    sensor_records: Sequence[SensorFrameManifestRecord] | None,
) -> GroundingContext | None:
    if sources is None and sensor_records is None:
        return None
    if sources is None or sensor_records is None:
        msg = "sources and sensor_records must be provided together"
        raise TypedMemoryWriterError(msg)

    source_bounds: SourceBounds = {}
    for source in sources:
        if source.video_id in source_bounds:
            msg = f"duplicate contextual source video_id: {source.video_id}"
            raise TypedMemoryWriterError(msg)
        source_bounds[source.video_id] = (source.start_time, source.end_time)

    selected_frames: SelectedFrameTimestamps = {}
    for sensor_record in sensor_records:
        if sensor_record.video_id in selected_frames:
            msg = f"duplicate contextual sensor video_id: {sensor_record.video_id}"
            raise TypedMemoryWriterError(msg)
        selected_frames[sensor_record.video_id] = {
            frame.frame_ref: frame.timestamp for frame in sensor_record.selected_frames
        }
    if source_bounds.keys() != selected_frames.keys():
        missing = sorted(source_bounds.keys() - selected_frames.keys())
        extra = sorted(selected_frames.keys() - source_bounds.keys())
        msg = (
            "contextual source and sensor video IDs differ: "
            f"missing={missing} extra={extra}"
        )
        raise TypedMemoryWriterError(msg)
    return source_bounds, selected_frames


def _validate_record_grounding(
    record: TypedMemoryRecordBase,
    line_number: int,
    context: GroundingContext,
) -> None:
    source_bounds, selected_frames = context
    bounds = source_bounds.get(record.source_video_id)
    if bounds is None:
        msg = (
            f"unknown source_video_id at typed memory row {line_number}: "
            f"{record.source_video_id}"
        )
        raise TypedMemoryWriterError(msg)
    if (
        record.validity.start_time < bounds[0]
        or record.validity.end_time > bounds[1]
        or record.first_seen_time < bounds[0]
        or record.last_seen_time > bounds[1]
    ):
        msg = (
            f"typed memory row {line_number} times are outside source bounds: "
            f"{record.source_video_id}"
        )
        raise TypedMemoryWriterError(msg)
    if (
        record.provenance in _EVIDENCE_REQUIRED_PROVENANCE
        and not record.evidence_refs
    ):
        msg = (
            f"typed memory row {line_number} provenance "
            f"{record.provenance} requires evidence_refs"
        )
        raise TypedMemoryWriterError(msg)
    invalid_refs = tuple(
        ref
        for ref in record.evidence_refs
        if ref not in selected_frames[record.source_video_id]
    )
    if invalid_refs:
        msg = (
            f"typed memory row {line_number} evidence_refs are not selected sensor "
            f"frames for {record.source_video_id}: {invalid_refs}"
        )
        raise TypedMemoryWriterError(msg)
    invalid_times = tuple(
        ref
        for ref in record.evidence_refs
        if (
            selected_frames[record.source_video_id][ref]
            < record.first_seen_time - _EVIDENCE_TIMESTAMP_TOLERANCE
            or selected_frames[record.source_video_id][ref]
            > record.last_seen_time + _EVIDENCE_TIMESTAMP_TOLERANCE
        )
    )
    if invalid_times:
        msg = (
            f"typed memory row {line_number} evidence frame timestamps are outside "
            f"record observation interval: {invalid_times}"
        )
        raise TypedMemoryWriterError(msg)
    if record.provenance in _EVIDENCE_REQUIRED_PROVENANCE:
        evidence_times = tuple(
            selected_frames[record.source_video_id][ref]
            for ref in record.evidence_refs
        )
        if (
            not isclose(
                min(evidence_times),
                record.first_seen_time,
                rel_tol=0.0,
                abs_tol=_EVIDENCE_TIMESTAMP_TOLERANCE,
            )
            or not isclose(
                max(evidence_times),
                record.last_seen_time,
                rel_tol=0.0,
                abs_tol=_EVIDENCE_TIMESTAMP_TOLERANCE,
            )
            or record.observation_count != len(set(record.evidence_refs))
        ):
            msg = (
                f"typed memory row {line_number} observation interval/count do not "
                "match evidence_refs"
            )
            raise TypedMemoryWriterError(msg)


def _write_verified_artifact_atomic(
    output: Path,
    payload: bytes,
    expected_records: Sequence[TypedMemoryRecordBase],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(mode="wb", dir=output.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
            _ = handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_persistent_artifact(temporary_path, expected_records)
        _ = temporary_path.replace(output)
        temporary_path = None
        _verify_persistent_artifact(output, expected_records)
    except (OSError, TypeError, ValueError) as error:
        msg = f"artifact verification failed: {error}"
        raise TypedMemoryWriterError(msg) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _verify_persistent_artifact(
    path: Path,
    expected_records: Sequence[TypedMemoryRecordBase],
) -> None:
    payload = path.read_bytes()
    if path.stat().st_size != len(payload):
        msg = "file-size verification failed"
        raise ValueError(msg)
    if payload and not payload.endswith(b"\n"):
        msg = "JSONL artifact must end with a newline"
        raise ValueError(msg)

    parsed: list[TypedMemoryRecordBase] = []
    for line in payload.splitlines():
        record: TypedMemoryRecord = _TYPED_MEMORY_ADAPTER.validate_python(
            json.loads(line)
        )
        if isinstance(record, NoWriteMemoryRecord):
            msg = "no_write record cannot be persisted"
            raise TypeError(msg)
        parsed.append(record)

    if tuple(record.memory_id for record in parsed) != tuple(
        record.memory_id for record in expected_records
    ):
        msg = "parsed memory IDs do not match selected records"
        raise ValueError(msg)
    recounted_bytes = sum(serialized_byte_cost(record) for record in parsed)
    if recounted_bytes != len(payload):
        msg = "parsed record byte recount mismatch"
        raise ValueError(msg)
