from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, assert_never

from worldmm_smvqa.worldmm.episodic_types import (
    EpisodicEdgeType,
    EpisodicNodeRecord,
    InvalidTemporalGraphError,
    contains_node,
)

if TYPE_CHECKING:
    from worldmm_smvqa.schema import ChunkGranularity, MemoryRecord, StreamChunk


def require_edge_valid(
    edge_type: EpisodicEdgeType,
    source: EpisodicNodeRecord,
    target: EpisodicNodeRecord,
) -> None:
    if source.video_id != target.video_id:
        raise InvalidTemporalGraphError(source.video_id, "cross-video edge")
    match edge_type:
        case "contains":
            if not contains_node(source, target):
                raise InvalidTemporalGraphError(
                    source.video_id,
                    "contains edge escapes span",
                )
            return
        case "temporal_next":
            if source.granularity != target.granularity:
                raise InvalidTemporalGraphError(
                    source.video_id,
                    "cross-scale temporal edge",
                )
            if source.end_time > target.start_time:
                raise InvalidTemporalGraphError(
                    source.video_id,
                    "temporal edge overlap",
                )
            return
    assert_never(edge_type)


def require_chunk_spans_valid(chunks: Sequence[StreamChunk]) -> None:
    grouped: dict[tuple[str, ChunkGranularity], list[StreamChunk]] = {}
    for chunk in chunks:
        key = (chunk.video_id, chunk.granularity)
        grouped.setdefault(key, []).append(chunk)
    for (video_id, granularity), group in grouped.items():
        ordered = tuple(sorted(group, key=lambda chunk: chunk.start_time))
        previous: StreamChunk | None = None
        for chunk in ordered:
            if previous is not None and chunk.start_time < previous.end_time:
                raise InvalidTemporalGraphError(
                    video_id=video_id,
                    detail=(
                        f"overlap in {granularity}: "
                        f"{previous.chunk_id} -> {chunk.chunk_id}"
                    ),
                )
            previous = chunk


def require_memory_spans_valid(source_memories: Sequence[MemoryRecord]) -> None:
    transcript_memories = tuple(
        memory for memory in source_memories if memory.store == "transcript"
    )
    grouped: dict[str, list[MemoryRecord]] = {}
    for memory in transcript_memories:
        chunk_id = memory.source_chunk_id
        if chunk_id is None:
            raise InvalidTemporalGraphError(memory.video_id, "missing source chunk")
        grouped.setdefault(chunk_id, []).append(memory)
    for group in grouped.values():
        _require_no_memory_overlap(group)


def _require_no_memory_overlap(group: Sequence[MemoryRecord]) -> None:
    ordered = tuple(sorted(group, key=lambda memory: memory.start_time))
    previous: MemoryRecord | None = None
    for memory in ordered:
        if previous is not None and memory.start_time < previous.end_time:
            raise InvalidTemporalGraphError(
                video_id=memory.video_id,
                detail=(
                    "overlap in transcript events: "
                    f"{previous.memory_id} -> {memory.memory_id}"
                ),
            )
        previous = memory
