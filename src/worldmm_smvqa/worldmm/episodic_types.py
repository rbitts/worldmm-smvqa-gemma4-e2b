from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

from worldmm_smvqa.schema import ChunkGranularity, FrozenModel

type EpisodicEdgeType = Literal["contains", "temporal_next"]


@dataclass(frozen=True, slots=True)
class InvalidTemporalGraphError(Exception):
    video_id: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"InvalidTemporalGraphError: {self.video_id}: {self.detail}"


@dataclass(frozen=True, slots=True)
class EpisodicBuildSummary:
    path: Path
    nodes: int
    edges: int
    contains_edges: int


class EpisodicNodeRecord(FrozenModel):
    record_type: Literal["node"] = "node"
    node_id: str
    video_id: str
    granularity: ChunkGranularity
    start_time: float
    end_time: float
    source_chunk_id: str
    source_memory_ids: tuple[str, ...]
    source_modality_refs: tuple[str, ...]
    source_modalities: tuple[str, ...]
    frame_refs: tuple[str, ...]
    confidence: float
    text_embedding_id: str


class EpisodicEdgeRecord(FrozenModel):
    record_type: Literal["edge"] = "edge"
    edge_id: str
    edge_type: EpisodicEdgeType
    video_id: str
    source_node_id: str
    target_node_id: str
    start_time: float
    end_time: float
    source_start_time: float
    source_end_time: float
    target_start_time: float
    target_end_time: float


type EpisodicRecord = EpisodicNodeRecord | EpisodicEdgeRecord


def contains_node(parent: EpisodicNodeRecord, child: EpisodicNodeRecord) -> bool:
    return (
        parent.video_id == child.video_id
        and parent.start_time <= child.start_time
        and child.end_time <= parent.end_time
        and parent.granularity == "shard_30m"
        and child.granularity == "clip_30s"
    )
