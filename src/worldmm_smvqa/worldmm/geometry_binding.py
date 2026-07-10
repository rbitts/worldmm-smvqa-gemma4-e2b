from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol, override, runtime_checkable

from worldmm_smvqa.schema import FrozenModel

if TYPE_CHECKING:
    from worldmm_smvqa.schema import SourceStreamExample
    from worldmm_smvqa.worldmm.spatial_types import SpatialAnchorRecord

type GeometryPrimitiveSource = Literal["object_geometry", "slam_pose", "gaze"]

_FRAME_TIME_RE: Final[re.Pattern[str]] = re.compile(r"(\d+)(?!.*\d)")
_SOURCE_PRIORITY: Final[dict[GeometryPrimitiveSource, int]] = {
    "object_geometry": 0,
    "gaze": 1,
    "slam_pose": 2,
}


@dataclass(frozen=True, slots=True)
class EmptyGeometryPrimitivesError(Exception):
    anchor_memory_id: str

    @override
    def __str__(self) -> str:
        return f"EmptyGeometryPrimitivesError: {self.anchor_memory_id}"


class GeometryPrimitive(FrozenModel):
    frame_ref: str
    x: float
    y: float
    z: float
    source: GeometryPrimitiveSource
    timestamp: float | None = None
    object_label: str | None = None


class BoundAnchor(FrozenModel):
    anchor_memory_id: str
    embedding_ref: str | None
    primitive: GeometryPrimitive
    distance_m: float


@runtime_checkable
class SemanticGeometryBinder(Protocol):
    def bind(
        self,
        anchor: SpatialAnchorRecord,
        primitives: Sequence[GeometryPrimitive],
    ) -> BoundAnchor:
        """Bind one spatial anchor to one geometry primitive."""
        ...


class DeterministicGeometryBinder:
    def bind(
        self,
        anchor: SpatialAnchorRecord,
        primitives: Sequence[GeometryPrimitive],
    ) -> BoundAnchor:
        """Bind anchor to the closest typed geometry primitive."""
        if not primitives:
            raise EmptyGeometryPrimitivesError(anchor_memory_id=anchor.memory_id)
        primitive = min(primitives, key=lambda item: _primitive_sort_key(anchor, item))
        return BoundAnchor(
            anchor_memory_id=anchor.memory_id,
            embedding_ref=f"geometry:{primitive.source}:{primitive.frame_ref}",
            primitive=primitive,
            distance_m=_distance(anchor, primitive),
        )


NoopBinder = DeterministicGeometryBinder


def build_geometry_primitives(
    source: SourceStreamExample,
) -> tuple[GeometryPrimitive, ...]:
    primitives: list[GeometryPrimitive] = []
    for detection in source.object_detections:
        if detection.x is None or detection.y is None or detection.z is None:
            continue
        frame_refs = _frame_refs_for_interval(
            source,
            start_time=detection.start_time,
            end_time=detection.end_time,
        )
        fallback = _synthetic_frame_ref(
            source.video_id,
            detection.label,
            detection.start_time,
        )
        primitives.extend(
            GeometryPrimitive(
                frame_ref=frame_ref,
                x=detection.x,
                y=detection.y,
                z=detection.z,
                source="object_geometry",
                timestamp=_frame_timestamp(source, frame_ref),
                object_label=detection.label,
            )
            for frame_ref in frame_refs or (fallback,)
        )
    primitives.extend(
        GeometryPrimitive(
            frame_ref=_nearest_frame_ref(source, sample.timestamp),
            x=sample.x,
            y=sample.y,
            z=sample.z,
            source="gaze",
            timestamp=sample.timestamp,
        )
        for sample in source.gaze_samples
    )
    primitives.extend(
        GeometryPrimitive(
            frame_ref=_nearest_frame_ref(source, sample.timestamp),
            x=sample.x,
            y=sample.y,
            z=sample.z,
            source="slam_pose",
            timestamp=sample.timestamp,
        )
        for sample in source.pose_samples
    )
    return tuple(primitives)


def _primitive_sort_key(
    anchor: SpatialAnchorRecord,
    primitive: GeometryPrimitive,
) -> tuple[float, int, float, float, str, float, float, float]:
    label_penalty = (
        0.0 if primitive.object_label in {None, anchor.object_label} else 1.0
    )
    return (
        label_penalty,
        _SOURCE_PRIORITY[primitive.source],
        _distance(anchor, primitive),
        abs(_primitive_time(primitive) - _anchor_midpoint(anchor)),
        primitive.frame_ref,
        primitive.x,
        primitive.y,
        primitive.z,
    )


def _distance(anchor: SpatialAnchorRecord, primitive: GeometryPrimitive) -> float:
    return math.dist(
        (anchor.x, anchor.y, anchor.z),
        (primitive.x, primitive.y, primitive.z),
    )


def _anchor_midpoint(anchor: SpatialAnchorRecord) -> float:
    return (anchor.start_time + anchor.end_time) / 2.0


def _primitive_time(primitive: GeometryPrimitive) -> float:
    return primitive.timestamp if primitive.timestamp is not None else _frame_time(
        primitive.frame_ref,
    )


def _frame_time(frame_ref: str) -> float:
    match = _FRAME_TIME_RE.search(frame_ref)
    if match is None:
        return math.inf
    return float(match.group(1))


def _frame_refs_for_interval(
    source: SourceStreamExample,
    *,
    start_time: float,
    end_time: float,
) -> tuple[str, ...]:
    return tuple(
        frame.frame_ref
        for frame in source.frame_metadata
        if start_time <= frame.timestamp <= end_time
    )


def _frame_timestamp(source: SourceStreamExample, frame_ref: str) -> float | None:
    for frame in source.frame_metadata:
        if frame.frame_ref == frame_ref:
            return frame.timestamp
    return None


def _nearest_frame_ref(source: SourceStreamExample, timestamp: float) -> str:
    if not source.frame_metadata:
        return _synthetic_frame_ref(source.video_id, "geometry", timestamp)
    return min(
        source.frame_metadata,
        key=lambda frame: (abs(frame.timestamp - timestamp), frame.frame_ref),
    ).frame_ref


def _synthetic_frame_ref(video_id: str, label: str, timestamp: float) -> str:
    token = str(int(timestamp)) if timestamp.is_integer() else f"{timestamp:.6f}"
    return f"{video_id}_{label}_{token}"
