from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, assert_never

from worldmm_smvqa.chunking import build_chunks, read_source_streams
from worldmm_smvqa.memory_sources import build_source_memories
from worldmm_smvqa.worldmm.episodic_types import (
    EpisodicBuildSummary,
    EpisodicEdgeRecord,
    EpisodicEdgeType,
    EpisodicNodeRecord,
    EpisodicRecord,
    InvalidTemporalGraphError,
    contains_node,
)
from worldmm_smvqa.worldmm.episodic_validation import (
    require_chunk_spans_valid,
    require_edge_valid,
    require_memory_spans_valid,
)

if TYPE_CHECKING:
    from worldmm_smvqa.schema import ChunkGranularity, MemoryRecord, StreamChunk

__all__ = [
    "EpisodicBuildSummary",
    "EpisodicEdgeRecord",
    "EpisodicNodeRecord",
    "EpisodicRecord",
    "InvalidTemporalGraphError",
    "build_episodic_graph",
    "write_fixture_episodic_memory",
]

EPISODIC_STORE: Final = "episodic"
MODALITY_ORDER: Final = ("caption", "transcript", "ocr", "object", "frame")


def build_episodic_graph(
    chunks: Sequence[StreamChunk],
    source_memories: Sequence[MemoryRecord],
) -> tuple[EpisodicRecord, ...]:
    ordered_chunks = tuple(sorted(chunks, key=_chunk_sort_key))
    require_chunk_spans_valid(ordered_chunks)
    require_memory_spans_valid(source_memories)
    memory_index = _memories_by_chunk(source_memories)
    nodes = tuple(_node_from_chunk(chunk, memory_index) for chunk in ordered_chunks)
    edges = (*_contains_edges(nodes), *_temporal_edges(nodes))
    return (*nodes, *edges)


def write_fixture_episodic_memory(
    fixture_dir: Path,
    output: Path,
) -> EpisodicBuildSummary:
    sources = read_source_streams(fixture_dir)
    chunks = build_chunks(sources)
    clip_chunks = tuple(
        chunk for chunk in chunks if chunk.granularity == "clip_30s"
    )
    memories = build_source_memories(clip_chunks)
    records = build_episodic_graph(chunks, memories)
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in records),
        encoding="utf-8",
    )
    edges = tuple(record for record in records if record.record_type == "edge")
    return EpisodicBuildSummary(
        path=output,
        nodes=sum(1 for record in records if record.record_type == "node"),
        edges=len(edges),
        contains_edges=sum(1 for edge in edges if edge.edge_type == "contains"),
    )


def _node_from_chunk(
    chunk: StreamChunk,
    memory_index: dict[str, tuple[MemoryRecord, ...]],
) -> EpisodicNodeRecord:
    memories = _chunk_memories(chunk, memory_index)
    node_id = _node_id(chunk.chunk_id)
    return EpisodicNodeRecord(
        node_id=node_id,
        video_id=chunk.video_id,
        granularity=chunk.granularity,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
        source_chunk_id=chunk.chunk_id,
        source_memory_ids=tuple(memory.memory_id for memory in memories),
        source_modality_refs=tuple(
            f"{memory.store}:{memory.memory_id}" for memory in memories
        ),
        source_modalities=_source_modalities(memories),
        frame_refs=_frame_refs(chunk, memories),
        confidence=1.0 if memories else 0.0,
        text_embedding_id=f"embedding:{node_id}:text",
    )


def _chunk_memories(
    chunk: StreamChunk,
    memory_index: dict[str, tuple[MemoryRecord, ...]],
) -> tuple[MemoryRecord, ...]:
    match chunk.granularity:
        case "clip_30s":
            return memory_index.get(chunk.chunk_id, ())
        case "shard_30m":
            return tuple(
                memory
                for memories in memory_index.values()
                for memory in memories
                if memory.video_id == chunk.video_id
                and chunk.start_time <= memory.start_time
                and memory.end_time <= chunk.end_time
            )
    assert_never(chunk.granularity)


def _contains_edges(
    nodes: Sequence[EpisodicNodeRecord],
) -> tuple[EpisodicEdgeRecord, ...]:
    shards = tuple(node for node in nodes if node.granularity == "shard_30m")
    clips = tuple(node for node in nodes if node.granularity == "clip_30s")
    return tuple(
        _edge("contains", shard, clip)
        for shard in shards
        for clip in clips
        if contains_node(shard, clip)
    )


def _temporal_edges(
    nodes: Sequence[EpisodicNodeRecord],
) -> tuple[EpisodicEdgeRecord, ...]:
    edges: list[EpisodicEdgeRecord] = []
    shards = tuple(node for node in nodes if node.granularity == "shard_30m")
    clips = tuple(node for node in nodes if node.granularity == "clip_30s")
    for shard in shards:
        ordered = tuple(
            sorted(
                (clip for clip in clips if contains_node(shard, clip)),
                key=lambda node: node.start_time,
            ),
        )
        for index, source in enumerate(ordered[:-1]):
            target = ordered[index + 1]
            edges.append(_temporal_edge(source, target))
    return tuple(edges)


def _edge(
    edge_type: EpisodicEdgeType,
    source: EpisodicNodeRecord,
    target: EpisodicNodeRecord,
) -> EpisodicEdgeRecord:
    require_edge_valid(edge_type, source, target)
    return EpisodicEdgeRecord(
        edge_id=f"episodic-edge:{edge_type}:{source.node_id}->{target.node_id}",
        edge_type=edge_type,
        video_id=source.video_id,
        source_node_id=source.node_id,
        target_node_id=target.node_id,
        start_time=min(source.start_time, target.start_time),
        end_time=max(source.end_time, target.end_time),
        source_start_time=source.start_time,
        source_end_time=source.end_time,
        target_start_time=target.start_time,
        target_end_time=target.end_time,
    )


def _temporal_edge(
    source: EpisodicNodeRecord,
    target: EpisodicNodeRecord,
) -> EpisodicEdgeRecord:
    require_edge_valid("temporal_next", source, target)
    return EpisodicEdgeRecord(
        edge_id=f"episodic-edge:temporal_next:{source.node_id}->{target.node_id}",
        edge_type="temporal_next",
        video_id=source.video_id,
        source_node_id=source.node_id,
        target_node_id=target.node_id,
        start_time=source.start_time,
        end_time=source.end_time,
        source_start_time=source.start_time,
        source_end_time=source.end_time,
        target_start_time=target.start_time,
        target_end_time=target.end_time,
    )


def _memories_by_chunk(
    source_memories: Sequence[MemoryRecord],
) -> dict[str, tuple[MemoryRecord, ...]]:
    grouped: dict[str, list[MemoryRecord]] = {}
    for memory in sorted(source_memories, key=_memory_sort_key):
        chunk_id = memory.source_chunk_id
        if chunk_id is not None:
            grouped.setdefault(chunk_id, []).append(memory)
    return {chunk_id: tuple(memories) for chunk_id, memories in grouped.items()}

def _source_modalities(memories: Sequence[MemoryRecord]) -> tuple[str, ...]:
    stores = {memory.store for memory in memories}
    ordered = tuple(store for store in MODALITY_ORDER if store in stores)
    extras = tuple(sorted(stores - set(MODALITY_ORDER)))
    return (*ordered, *extras)


def _frame_refs(
    chunk: StreamChunk,
    memories: Sequence[MemoryRecord],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *chunk.frame_refs,
                *(ref for memory in memories for ref in memory.frame_refs),
            ),
        ),
    )

def _node_id(chunk_id: str) -> str:
    return f"{EPISODIC_STORE}:{chunk_id}"


def _chunk_sort_key(chunk: StreamChunk) -> tuple[str, int, float, float, str]:
    return (
        chunk.video_id,
        _granularity_rank(chunk.granularity),
        chunk.start_time,
        chunk.end_time,
        chunk.chunk_id,
    )


def _memory_sort_key(memory: MemoryRecord) -> tuple[str, float, float, str, str]:
    return (
        memory.video_id,
        memory.start_time,
        memory.end_time,
        memory.store,
        memory.memory_id,
    )


def _granularity_rank(granularity: ChunkGranularity) -> int:
    match granularity:
        case "shard_30m":
            return 0
        case "clip_30s":
            return 1
    assert_never(granularity)
