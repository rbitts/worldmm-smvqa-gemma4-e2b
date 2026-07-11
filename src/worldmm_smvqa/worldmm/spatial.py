from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

from worldmm_smvqa.schema import (
    GazeSample,
    ObjectMetadata,
    PoseSample,
    SourceStreamExample,
    StreamChunk,
)
from worldmm_smvqa.worldmm.geometry_binding import (
    DeterministicGeometryBinder,
    EmptyGeometryPrimitivesError,
    build_geometry_primitives,
)
from worldmm_smvqa.worldmm.spatial_types import (
    InvalidSpatialInputError,
    ObjectStateSnapshotRecord,
    SpatialAnchorRecord,
    SpatialProvenance,
    SpatialRelationKind,
    SpatialRelationRecord,
    WearerTrajectorySummaryRecord,
    ZoneRecord,
)

DEFAULT_CELL_SIZE: Final = 2.0
DEFAULT_NEAR_THRESHOLD: Final = 1.5
DEFAULT_DIRECTION_THRESHOLD: Final = 0.25
DEFAULT_INSTANCE_ASSOCIATION_THRESHOLD: Final = 0.75
DEFAULT_MOVEMENT_THRESHOLD: Final = 0.25
OBJECT_STATE_SNAPSHOT_SNIPPET: Final = (
    "as of t={chunk_end}s, {object_label} last seen in {zone_id} "
    "at t={last_seen}s near ({x},{y})"
)
WEARER_TRAJECTORY_SNIPPET: Final = "wearer in {zone_ids} during [{start},{end}]s"


@dataclass(frozen=True, slots=True)
class _Position:
    x: float
    y: float
    z: float
    provenance: SpatialProvenance
    coordinate_frame: str


def build_zones(
    source: SourceStreamExample,
    *,
    cell_size: float = DEFAULT_CELL_SIZE,
) -> tuple[ZoneRecord, ...]:
    _require_positive(value=cell_size, name="cell_size", video_id=source.video_id)
    _require_sorted_pose_samples(source.video_id, source.pose_samples)
    if not source.pose_samples:
        return ()
    _require_compatible_pose_frames(source.video_id, source.pose_samples)
    cells = tuple(
        _cell(sample.x, sample.y, cell_size) for sample in source.pose_samples
    )
    visits = _visit_intervals(source.pose_samples, cells)
    return tuple(
        _zone_record(source.video_id, cell, samples, visits[cell])
        for cell, samples in sorted(
            _samples_by_cell(source.pose_samples, cells).items(),
            key=lambda item: item[0],
        )
    )


def build_object_anchors(
    source: SourceStreamExample,
) -> tuple[SpatialAnchorRecord, ...]:
    _require_sorted_pose_samples(source.video_id, source.pose_samples)
    anchors: list[SpatialAnchorRecord] = []
    for detection_index, detection in sorted(
        enumerate(source.object_detections),
        key=lambda item: (item[1].start_time, item[1].end_time, item[0]),
    ):
        position = _anchor_position(source, detection)
        instance_id = _instance_id(
            detection,
            detection_index=detection_index,
            position=position,
            anchors=anchors,
        )
        previous_index = _latest_instance_index(anchors, instance_id)
        previous = anchors[previous_index] if previous_index is not None else None
        moved = (
            previous is not None
            and previous.provenance == "object_geometry"
            and position.provenance == "object_geometry"
            and math.dist(
                (previous.x, previous.y, previous.z),
                (position.x, position.y, position.z),
            )
            > DEFAULT_MOVEMENT_THRESHOLD
        )
        if moved and previous_index is not None and previous is not None:
            previous = previous.model_copy(
                update={"valid_to": detection.start_time},
            )
            anchors[previous_index] = previous
        anchors.append(
            _anchor_record(
                source,
                detection,
                position=position,
                instance_id=instance_id,
                memory_id=_unique_anchor_memory_id(
                    source.video_id,
                    detection,
                    instance_id,
                    anchors,
                ),
                previous=previous,
                moved=moved,
            ),
        )
    return tuple(anchors)


def derive_relations(
    anchors: Sequence[SpatialAnchorRecord],
    *,
    near_threshold: float = DEFAULT_NEAR_THRESHOLD,
    direction_threshold: float = DEFAULT_DIRECTION_THRESHOLD,
) -> tuple[SpatialRelationRecord, ...]:
    _require_positive(value=near_threshold, name="near_threshold", video_id="anchors")
    _require_positive(
        value=direction_threshold,
        name="direction_threshold",
        video_id="anchors",
    )
    relations: list[SpatialRelationRecord] = []
    for left_index, left in enumerate(anchors):
        for right in anchors[left_index + 1 :]:
            overlap_start = max(left.start_time, right.start_time)
            overlap_end = min(left.end_time, right.end_time)
            if (
                left.video_id == right.video_id
                and left.zone_id == right.zone_id
                and _anchor_identity(left) != _anchor_identity(right)
                and overlap_start <= overlap_end
                and _object_geometry_grounded(left, right)
                and math.hypot(left.x - right.x, left.y - right.y) <= near_threshold
            ):
                subject, target = sorted((left, right), key=_anchor_identity_key)
                relations.append(
                    _relation_record(
                        subject=subject,
                        relation="near",
                        target=target,
                        start_time=overlap_start,
                        end_time=overlap_end,
                    ),
                )
            if (
                left.video_id == right.video_id
                and _anchor_identity(left) != _anchor_identity(right)
                and overlap_start <= overlap_end
                and _object_geometry_grounded(left, right)
            ):
                relations.extend(
                    _directional_relations(
                        left,
                        right,
                        start_time=overlap_start,
                        end_time=overlap_end,
                        threshold=direction_threshold,
                    ),
                )
    return tuple(
        sorted(
            _unique_relation_ids(relations),
            key=lambda item: (
                item.video_id,
                item.subject,
                item.relation,
                item.object,
                item.start_time,
                item.end_time,
            ),
        ),
    )


def _unique_relation_ids(
    relations: Sequence[SpatialRelationRecord],
) -> tuple[SpatialRelationRecord, ...]:
    seen: set[str] = set()
    unique: list[SpatialRelationRecord] = []
    for relation in relations:
        if relation.memory_id in seen:
            suffix = (
                f"{relation.subject_instance_id or relation.subject}:"
                f"{relation.object_instance_id or relation.object}"
            )
            memory_id = f"{relation.memory_id}:{suffix}"
            duplicate_index = 2
            while memory_id in seen:
                memory_id = f"{relation.memory_id}:{suffix}:{duplicate_index}"
                duplicate_index += 1
            unique.append(
                relation.model_copy(
                    update={"memory_id": memory_id},
                ),
            )
        else:
            unique.append(relation)
        seen.add(unique[-1].memory_id)
    return tuple(unique)


def build_object_state_snapshots(
    clip_chunks: Sequence[StreamChunk],
    anchors: Sequence[SpatialAnchorRecord],
) -> tuple[ObjectStateSnapshotRecord, ...]:
    _require_sorted_clip_chunks(clip_chunks)
    stream_end_times = _stream_end_times(clip_chunks)
    records: list[ObjectStateSnapshotRecord] = []
    for chunk in clip_chunks:
        seen = _latest_anchor_by_object(chunk, anchors)
        records.extend(
            _object_snapshot(
                chunk=chunk,
                anchor=anchor,
                stream_end_time=stream_end_times[chunk.video_id],
            )
            for anchor in sorted(
                seen.values(),
                key=lambda item: (-item.end_time, item.object_label),
            )
        )
    return tuple(records)


def build_trajectory_summaries(
    clip_chunks: Sequence[StreamChunk],
    zones: Sequence[ZoneRecord],
) -> tuple[WearerTrajectorySummaryRecord, ...]:
    _require_sorted_clip_chunks(clip_chunks)
    stream_end_times = _stream_end_times(clip_chunks)
    summaries: list[WearerTrajectorySummaryRecord] = []
    for chunk in clip_chunks:
        summary = _trajectory_summary(
            chunk=chunk,
            zones=_overlapping_zones(chunk, zones),
            stream_end_time=stream_end_times[chunk.video_id],
        )
        if (
            summaries
            and summaries[-1].video_id == summary.video_id
            and summaries[-1].zone_ids == summary.zone_ids
        ):
            continue
        summaries.append(summary)
    return tuple(summaries)


def _require_positive(*, value: float, name: str, video_id: str) -> None:
    if value <= 0.0:
        raise InvalidSpatialInputError(
            video_id=video_id,
            detail=f"{name} must be positive",
        )


def _require_sorted_pose_samples(
    video_id: str,
    samples: Sequence[PoseSample],
) -> None:
    previous: float | None = None
    for sample in samples:
        if previous is not None and sample.timestamp < previous:
            raise InvalidSpatialInputError(
                video_id=video_id,
                detail="pose_samples must be sorted by timestamp",
            )
        previous = sample.timestamp


def _require_compatible_pose_frames(
    video_id: str,
    samples: Sequence[PoseSample],
) -> None:
    frames = {_pose_coordinate_frame(sample) for sample in samples}
    if len(frames) > 1:
        raise InvalidSpatialInputError(
            video_id=video_id,
            detail="pose_samples use incompatible coordinate frames",
        )


def _require_sorted_clip_chunks(chunks: Sequence[StreamChunk]) -> None:
    previous_key: tuple[str, float, float] | None = None
    for chunk in chunks:
        if chunk.granularity != "clip_30s":
            raise InvalidSpatialInputError(
                video_id=chunk.video_id,
                detail="clip_chunks must have clip_30s granularity",
            )
        key = (chunk.video_id, chunk.start_time, chunk.end_time)
        if previous_key is not None and key < previous_key:
            raise InvalidSpatialInputError(
                video_id=chunk.video_id,
                detail="clip_chunks must be sorted by video_id and time",
            )
        previous_key = key


def _stream_end_times(chunks: Sequence[StreamChunk]) -> dict[str, float]:
    end_times: dict[str, float] = {}
    for chunk in chunks:
        end_times[chunk.video_id] = max(
            chunk.end_time,
            end_times.get(chunk.video_id, chunk.end_time),
        )
    return end_times


def _latest_anchor_by_object(
    chunk: StreamChunk,
    anchors: Sequence[SpatialAnchorRecord],
) -> dict[str, SpatialAnchorRecord]:
    seen: dict[str, SpatialAnchorRecord] = {}
    for anchor in anchors:
        if anchor.video_id != chunk.video_id or anchor.end_time > chunk.end_time:
            continue
        identity = _anchor_identity(anchor)
        current = seen.get(identity)
        if current is None or anchor.end_time > current.end_time:
            seen[identity] = anchor
    return seen


def _object_snapshot(
    *,
    chunk: StreamChunk,
    anchor: SpatialAnchorRecord,
    stream_end_time: float,
) -> ObjectStateSnapshotRecord:
    chunk_end = _format_seconds(chunk.end_time)
    last_seen = _format_seconds(anchor.end_time)
    x = _format_seconds(anchor.x)
    y = _format_seconds(anchor.y)
    identity_suffix = f":{anchor.instance_id}" if anchor.instance_id else ""
    return ObjectStateSnapshotRecord(
        memory_id=(
            f"spatial_snapshot:{chunk.video_id}:{anchor.object_label}"
            f"{identity_suffix}:{chunk_end}"
        ),
        video_id=chunk.video_id,
        object_label=anchor.object_label,
        zone_id=anchor.zone_id,
        last_seen_time=anchor.end_time,
        x=anchor.x,
        y=anchor.y,
        z=anchor.z,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
        snippet=OBJECT_STATE_SNAPSHOT_SNIPPET.format(
            chunk_end=chunk_end,
            object_label=anchor.object_label,
            zone_id=anchor.zone_id,
            last_seen=last_seen,
            x=x,
            y=y,
        ),
        base_score=_base_score(chunk.end_time, stream_end_time),
        instance_id=anchor.instance_id,
    )


def _trajectory_summary(
    *,
    chunk: StreamChunk,
    zones: tuple[ZoneRecord, ...],
    stream_end_time: float,
) -> WearerTrajectorySummaryRecord:
    if not zones:
        raise InvalidSpatialInputError(
            video_id=chunk.video_id,
            detail=(
                "no zone visit overlaps "
                f"{_format_seconds(chunk.start_time)}-{_format_seconds(chunk.end_time)}"
            ),
        )
    zone_ids = tuple(zone.zone_id for zone in zones)
    chunk_end = _format_seconds(chunk.end_time)
    return WearerTrajectorySummaryRecord(
        memory_id=f"spatial_trajectory:{chunk.video_id}:{chunk_end}",
        video_id=chunk.video_id,
        zone_ids=zone_ids,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
        snippet=WEARER_TRAJECTORY_SNIPPET.format(
            zone_ids=", ".join(zone_ids),
            start=_format_seconds(chunk.start_time),
            end=chunk_end,
        ),
        base_score=_base_score(chunk.end_time, stream_end_time),
    )


def _overlapping_zones(
    chunk: StreamChunk,
    zones: Sequence[ZoneRecord],
) -> tuple[ZoneRecord, ...]:
    return tuple(
        sorted(
            (
                zone
                for zone in zones
                if zone.video_id == chunk.video_id
                and any(
                    _intervals_overlap(
                        left_start=chunk.start_time,
                        left_end=chunk.end_time,
                        right_start=start_time,
                        right_end=end_time,
                    )
                    for start_time, end_time in zone.visit_intervals
                )
            ),
            key=lambda item: item.zone_id,
        ),
    )


def _intervals_overlap(
    *,
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> bool:
    return left_start <= right_end and right_start <= left_end


def _base_score(end_time: float, stream_end_time: float) -> float:
    raw_score = end_time / max(1.0, stream_end_time)
    return min(1.0, max(0.0, raw_score))


def _cell(x: float, y: float, cell_size: float) -> tuple[int, int]:
    return (math.floor(x / cell_size), math.floor(y / cell_size))


def _zone_id(video_id: str, x: float, y: float, cell_size: float) -> str:
    cell_x, cell_y = _cell(x, y, cell_size)
    return f"zone_{video_id}_{cell_x}_{cell_y}"


def _samples_by_cell(
    samples: Sequence[PoseSample],
    cells: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], tuple[PoseSample, ...]]:
    grouped: dict[tuple[int, int], list[PoseSample]] = {}
    for sample, cell in zip(samples, cells, strict=True):
        grouped.setdefault(cell, []).append(sample)
    return {cell: tuple(cell_samples) for cell, cell_samples in grouped.items()}


def _visit_intervals(
    samples: Sequence[PoseSample],
    cells: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], tuple[tuple[float, float], ...]]:
    visits: dict[tuple[int, int], list[tuple[float, float]]] = {}
    current_cell = cells[0]
    start_time = samples[0].timestamp
    end_time = start_time
    for sample, cell in zip(samples[1:], cells[1:], strict=True):
        if cell == current_cell:
            end_time = sample.timestamp
        else:
            visits.setdefault(current_cell, []).append((start_time, end_time))
            current_cell = cell
            start_time = sample.timestamp
            end_time = sample.timestamp
    visits.setdefault(current_cell, []).append((start_time, end_time))
    return {cell: tuple(cell_visits) for cell, cell_visits in visits.items()}


def _zone_record(
    video_id: str,
    cell: tuple[int, int],
    samples: Sequence[PoseSample],
    visit_intervals: tuple[tuple[float, float], ...],
) -> ZoneRecord:
    sample_count = len(samples)
    cell_x, cell_y = cell
    return ZoneRecord(
        zone_id=f"zone_{video_id}_{cell_x}_{cell_y}",
        video_id=video_id,
        centroid_x=sum(sample.x for sample in samples) / sample_count,
        centroid_y=sum(sample.y for sample in samples) / sample_count,
        centroid_z=sum(sample.z for sample in samples) / sample_count,
        cell=cell,
        visit_intervals=visit_intervals,
        visit_centroids=tuple(
            _pose_centroid(
                tuple(
                    sample
                    for sample in samples
                    if start_time <= sample.timestamp <= end_time
                ),
                video_id=video_id,
            )
            for start_time, end_time in visit_intervals
        ),
        coordinate_frame=_pose_coordinate_frame(samples[0]),
    )


def _pose_centroid(
    samples: Sequence[PoseSample],
    *,
    video_id: str,
) -> tuple[float, float, float]:
    sample_count = len(samples)
    if sample_count == 0:
        raise InvalidSpatialInputError(
            video_id=video_id,
            detail="zone visit has no pose samples",
        )
    first = samples[0]
    return (first.x, first.y, first.z)


def _anchor_record(  # noqa: PLR0913
    source: SourceStreamExample,
    detection: ObjectMetadata,
    *,
    position: _Position,
    instance_id: str,
    memory_id: str,
    previous: SpatialAnchorRecord | None,
    moved: bool,
) -> SpatialAnchorRecord:
    first_seen_time = (
        previous.first_seen_time
        if previous is not None and previous.first_seen_time is not None
        else (previous.start_time if previous is not None else detection.start_time)
    )
    valid_from = (
        detection.start_time
        if previous is None or moved
        else (previous.valid_from or previous.start_time)
    )
    anchor = SpatialAnchorRecord(
        memory_id=memory_id,
        video_id=source.video_id,
        object_label=detection.label,
        x=position.x,
        y=position.y,
        z=position.z,
        zone_id=_zone_id(source.video_id, position.x, position.y, DEFAULT_CELL_SIZE),
        start_time=detection.start_time,
        end_time=detection.end_time,
        frame_refs=tuple(
            frame.frame_ref
            for frame in source.frame_metadata
            if detection.start_time <= frame.timestamp <= detection.end_time
        ),
        confidence=detection.confidence,
        provenance=position.provenance,
        instance_id=instance_id,
        coordinate_frame=position.coordinate_frame,
        uncertainty_m=0.0 if position.provenance == "object_geometry" else None,
        first_seen_time=first_seen_time,
        last_seen_time=detection.end_time,
        valid_from=valid_from,
        change_type=(
            "appeared" if previous is None else ("moved" if moved else "observed")
        ),
    )
    try:
        bound = DeterministicGeometryBinder().bind(
            anchor,
            build_geometry_primitives(_causal_source(source, detection)),
        )
    except EmptyGeometryPrimitivesError:
        return anchor
    return anchor.model_copy(
        update={
            "geometry_frame_ref": bound.primitive.frame_ref,
            "geometry_source": bound.primitive.source,
            "geometry_distance_m": bound.distance_m,
            "geometry_embedding_ref": bound.embedding_ref,
        },
    )


def _unique_anchor_memory_id(
    video_id: str,
    detection: ObjectMetadata,
    instance_id: str,
    anchors: Sequence[SpatialAnchorRecord],
) -> str:
    base = (
        f"spatial_anchor:{video_id}:{detection.label}:"
        f"{_format_seconds(detection.start_time)}"
    )
    if all(anchor.memory_id != base for anchor in anchors):
        return base
    return f"{base}:{instance_id}"


def _causal_source(
    source: SourceStreamExample,
    detection: ObjectMetadata,
) -> SourceStreamExample:
    cutoff = detection.end_time
    return source.model_copy(
        update={
            "object_detections": tuple(
                item for item in source.object_detections if item.end_time <= cutoff
            ),
            "pose_samples": tuple(
                sample for sample in source.pose_samples if sample.timestamp <= cutoff
            ),
            "gaze_samples": tuple(
                sample for sample in source.gaze_samples if sample.timestamp <= cutoff
            ),
            "frame_metadata": tuple(
                frame for frame in source.frame_metadata if frame.timestamp <= cutoff
            ),
        },
    )


def _instance_id(
    detection: ObjectMetadata,
    *,
    detection_index: int,
    position: _Position,
    anchors: Sequence[SpatialAnchorRecord],
) -> str:
    if detection.instance_id:
        return detection.instance_id
    if position.provenance == "object_geometry":
        unavailable_instance_ids = {
            anchor.instance_id
            for anchor in anchors
            if anchor.object_label == detection.label
            and anchor.instance_id is not None
            and (
                anchor.start_time == detection.start_time
                or anchor.end_time > detection.start_time
            )
        }
        previous = tuple(
            anchor
            for anchor in anchors
            if anchor.object_label == detection.label
            and anchor.instance_id is not None
            and anchor.instance_id.startswith("local:")
            and anchor.instance_id not in unavailable_instance_ids
            and anchor.provenance == "object_geometry"
            and anchor.end_time <= detection.start_time
            and math.dist(
                (anchor.x, anchor.y, anchor.z),
                (position.x, position.y, position.z),
            )
            <= DEFAULT_INSTANCE_ASSOCIATION_THRESHOLD
        )
        if previous:
            return (
                min(
                    previous,
                    key=lambda anchor: (
                        math.dist(
                            (anchor.x, anchor.y, anchor.z),
                            (position.x, position.y, position.z),
                        ),
                        -anchor.end_time,
                        _anchor_identity(anchor),
                    ),
                ).instance_id
                or ""
            )
    # ponytail: proxy-only observations stay distinct; merge only with a real ID.
    return (
        f"local:{detection.label}:{_format_seconds(detection.start_time)}:"
        f"{detection_index}"
    )


def _latest_instance_index(
    anchors: Sequence[SpatialAnchorRecord],
    instance_id: str,
) -> int | None:
    for index in range(len(anchors) - 1, -1, -1):
        if anchors[index].instance_id == instance_id:
            return index
    return None


def _anchor_position(
    source: SourceStreamExample,
    detection: ObjectMetadata,
) -> _Position:
    if detection.x is not None and detection.y is not None and detection.z is not None:
        return _Position(
            x=detection.x,
            y=detection.y,
            z=detection.z,
            provenance="object_geometry",
            coordinate_frame="source_world",
        )
    gaze = _gaze_target(source.gaze_samples, detection)
    if gaze is not None:
        return _Position(
            x=gaze.x,
            y=gaze.y,
            z=gaze.z,
            provenance="gaze",
            coordinate_frame="source_world",
        )
    pose = _interpolated_pose(
        video_id=source.video_id,
        samples=tuple(
            sample
            for sample in source.pose_samples
            if sample.timestamp <= detection.end_time
        ),
        timestamp=(detection.start_time + detection.end_time) / 2.0,
    )
    # ponytail: viewer-position anchor; upgrade path = v3 depth/geometry binding
    return _Position(
        x=pose.x,
        y=pose.y,
        z=pose.z,
        provenance="pose",
        coordinate_frame=_pose_coordinate_frame(pose),
    )


def _gaze_target(
    samples: Sequence[GazeSample],
    detection: ObjectMetadata,
) -> GazeSample | None:
    matching = tuple(
        sample
        for sample in samples
        if detection.start_time <= sample.timestamp <= detection.end_time
    )
    if not matching:
        return None
    return min(matching, key=lambda sample: sample.timestamp)


def _interpolated_pose(
    *,
    video_id: str,
    samples: Sequence[PoseSample],
    timestamp: float,
) -> PoseSample:
    if not samples:
        raise InvalidSpatialInputError(
            video_id=video_id,
            detail="pose_samples required for pose-provenance anchor",
        )
    for sample in samples:
        if timestamp == sample.timestamp:
            return sample
    if timestamp <= samples[0].timestamp:
        return samples[0]
    if timestamp >= samples[-1].timestamp:
        return samples[-1]
    for left, right in pairwise(samples):
        if left.timestamp <= timestamp <= right.timestamp:
            left_frame = _pose_coordinate_frame(left)
            right_frame = _pose_coordinate_frame(right)
            if left_frame != right_frame:
                raise InvalidSpatialInputError(
                    video_id=video_id,
                    detail="cannot interpolate across incompatible pose frames",
                )
            span = right.timestamp - left.timestamp
            ratio = 0.0 if span == 0.0 else (timestamp - left.timestamp) / span
            return PoseSample(
                timestamp=timestamp,
                x=left.x + (right.x - left.x) * ratio,
                y=left.y + (right.y - left.y) * ratio,
                z=left.z + (right.z - left.z) * ratio,
                yaw=_interpolate_optional(left.yaw, right.yaw, ratio),
                coordinate_frame=left_frame,
            )
    raise InvalidSpatialInputError(
        video_id=video_id,
        detail="pose_samples do not cover interpolation timestamp",
    )


def _pose_coordinate_frame(sample: PoseSample) -> str:
    return sample.coordinate_frame or "source_world"


def _interpolate_optional(
    left: float | None,
    right: float | None,
    ratio: float,
) -> float | None:
    if left is None or right is None:
        return None
    return left + ((right - left) * ratio)


def _format_seconds(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _object_geometry_grounded(
    left: SpatialAnchorRecord,
    right: SpatialAnchorRecord,
) -> bool:
    return (
        left.provenance == "object_geometry"
        and right.provenance == "object_geometry"
        and left.coordinate_frame == right.coordinate_frame
    )


def _anchor_identity(anchor: SpatialAnchorRecord) -> str:
    return anchor.instance_id or anchor.memory_id


def _anchor_identity_key(anchor: SpatialAnchorRecord) -> tuple[str, str]:
    return (anchor.object_label, _anchor_identity(anchor))


def _directional_relations(
    left: SpatialAnchorRecord,
    right: SpatialAnchorRecord,
    *,
    start_time: float,
    end_time: float,
    threshold: float,
) -> tuple[SpatialRelationRecord, ...]:
    relations: list[SpatialRelationRecord] = []
    if abs(left.x - right.x) >= threshold:
        relations.append(
            _relation_record(
                subject=left if left.x < right.x else right,
                relation="left_of",
                target=right if left.x < right.x else left,
                start_time=start_time,
                end_time=end_time,
            ),
        )
        relations.append(
            _relation_record(
                subject=right if left.x < right.x else left,
                relation="right_of",
                target=left if left.x < right.x else right,
                start_time=start_time,
                end_time=end_time,
            ),
        )
    if abs(left.y - right.y) >= threshold:
        relations.append(
            _relation_record(
                subject=left if left.y > right.y else right,
                relation="in_front_of",
                target=right if left.y > right.y else left,
                start_time=start_time,
                end_time=end_time,
            ),
        )
        relations.append(
            _relation_record(
                subject=right if left.y > right.y else left,
                relation="behind",
                target=left if left.y > right.y else right,
                start_time=start_time,
                end_time=end_time,
            ),
        )
    if abs(left.z - right.z) >= threshold:
        relations.append(
            _relation_record(
                subject=left if left.z > right.z else right,
                relation="above",
                target=right if left.z > right.z else left,
                start_time=start_time,
                end_time=end_time,
            ),
        )
        relations.append(
            _relation_record(
                subject=right if left.z > right.z else left,
                relation="below",
                target=left if left.z > right.z else right,
                start_time=start_time,
                end_time=end_time,
            ),
        )
    return tuple(relations)


def _relation_record(
    *,
    subject: SpatialAnchorRecord,
    relation: SpatialRelationKind,
    target: SpatialAnchorRecord,
    start_time: float,
    end_time: float,
) -> SpatialRelationRecord:
    available_time = max(subject.end_time, target.end_time)
    return SpatialRelationRecord(
        memory_id=(
            f"spatial_relation:{subject.video_id}:{subject.object_label}:"
            f"{relation}:{target.object_label}:{_format_seconds(start_time)}"
        ),
        video_id=subject.video_id,
        subject=subject.object_label,
        relation=relation,
        object=target.object_label,
        zone_id=subject.zone_id,
        start_time=start_time,
        # Both endpoint observations must be complete before this fact exists.
        end_time=available_time,
        distance_m=math.dist(
            (subject.x, subject.y, subject.z),
            (target.x, target.y, target.z),
        ),
        delta_x=target.x - subject.x,
        delta_y=target.y - subject.y,
        delta_z=target.z - subject.z,
        subject_instance_id=subject.instance_id,
        object_instance_id=target.instance_id,
        coordinate_frame=subject.coordinate_frame,
        valid_from=start_time,
        valid_to=end_time,
    )
