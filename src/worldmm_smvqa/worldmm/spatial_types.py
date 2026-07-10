from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

from pydantic import Field

from worldmm_smvqa.schema import FrozenModel

type SpatialProvenance = Literal["object_geometry", "pose", "slam_pose", "gaze"]
type SpatialRelationKind = Literal[
    "near",
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "above",
    "below",
]


@dataclass(frozen=True, slots=True)
class InvalidSpatialInputError(Exception):
    video_id: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"InvalidSpatialInputError: {self.video_id}: {self.detail}"


@dataclass(frozen=True, slots=True)
class SpatialBuildSummary:
    path: Path
    zones: int
    anchors: int
    relations: int


class ZoneRecord(FrozenModel):
    record_type: Literal["zone"] = "zone"
    zone_id: str
    video_id: str
    centroid_x: float
    centroid_y: float
    centroid_z: float
    cell: tuple[int, int]
    visit_intervals: tuple[tuple[float, float], ...]


class SpatialAnchorRecord(FrozenModel):
    record_type: Literal["spatial_anchor"] = "spatial_anchor"
    memory_id: str
    store: Literal["spatial"] = "spatial"
    video_id: str
    object_label: str
    x: float
    y: float
    z: float
    zone_id: str
    start_time: float
    end_time: float
    frame_refs: tuple[str, ...]
    confidence: float
    provenance: SpatialProvenance
    geometry_frame_ref: str | None = None
    geometry_source: SpatialProvenance | None = None
    geometry_distance_m: float | None = None
    geometry_embedding_ref: str | None = None


class SpatialRelationRecord(FrozenModel):
    record_type: Literal["spatial_relation"] = "spatial_relation"
    memory_id: str
    store: Literal["spatial"] = "spatial"
    video_id: str
    subject: str
    relation: SpatialRelationKind = "near"
    object: str
    zone_id: str
    start_time: float
    end_time: float
    distance_m: float | None = None
    delta_x: float | None = None
    delta_y: float | None = None
    delta_z: float | None = None


class SpatialTokenRecord(FrozenModel):
    record_type: Literal["spatial_token"] = "spatial_token"
    memory_id: str
    store: Literal["spatial"] = "spatial"
    video_id: str
    encoder: str = "structured-v1"
    projection_head: str = "identity-v1"
    token_decoder: str = "delta-topk-v1"  # noqa: S105
    codec: str = "compact-json-v1"
    start_time: float
    end_time: float
    token: str
    importance: float = Field(ge=0.0, le=1.0)
    frame_refs: tuple[str, ...] = ()


class ObjectStateSnapshotRecord(FrozenModel):
    record_type: Literal["object_state_snapshot"] = "object_state_snapshot"
    memory_id: str
    store: Literal["spatial"] = "spatial"
    video_id: str
    object_label: str
    zone_id: str
    last_seen_time: float
    x: float
    y: float
    z: float
    start_time: float
    end_time: float
    snippet: str
    base_score: float


class WearerTrajectorySummaryRecord(FrozenModel):
    record_type: Literal["wearer_trajectory_summary"] = "wearer_trajectory_summary"
    memory_id: str
    store: Literal["spatial"] = "spatial"
    video_id: str
    zone_ids: tuple[str, ...]
    start_time: float
    end_time: float
    snippet: str
    base_score: float
