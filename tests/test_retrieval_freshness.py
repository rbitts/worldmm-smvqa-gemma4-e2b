from __future__ import annotations

from worldmm_smvqa.retrieval_protocols import (
    build_egobutler_hierarchy,
    coarse_to_fine_candidates,
)
from worldmm_smvqa.retrieval_types import RetrievalMemoryRecord
from worldmm_smvqa.schema import ChunkGranularity, QuestionRequest, StreamChunk


def test_equal_lexical_scores_select_latest_causal_clip_after_move() -> None:
    # Given: the same entity relation appears before and after a move.
    chunks = (
        _chunk("v1:0:1800:shard_30m", 0.0, 1800.0, "shard_30m"),
        _chunk("v1:0:30:clip_30s", 0.0, 30.0, "clip_30s"),
        _chunk("v1:1800:3600:shard_30m", 1800.0, 3600.0, "shard_30m"),
        _chunk("v1:1800:1830:clip_30s", 1800.0, 1830.0, "clip_30s"),
    )
    records = (
        _relation("relation-old", 5.0, 10.0, x=0.0),
        _relation("relation-new", 1805.0, 1810.0, x=2.0),
    )
    hierarchy = build_egobutler_hierarchy(chunks, records)
    question = QuestionRequest(
        question_id="q-after-move",
        video_id="v1",
        question="Where was the mug after it moved?",
        question_time=1840.0,
        answer_choices=(),
    )

    # When: both causal clips have the same lexical score under a one-clip cap.
    result = coarse_to_fine_candidates(
        question,
        hierarchy,
        records,
        max_clips=1,
    )

    # Then: freshness wins the tie, avoiding the stale pre-move relation.
    assert result.selected_clip_ids == ("v1:1800:1830:clip_30s",)
    assert tuple(record.memory_id for record in result.records) == ("relation-new",)


def _chunk(
    chunk_id: str,
    start_time: float,
    end_time: float,
    granularity: ChunkGranularity,
) -> StreamChunk:
    return StreamChunk(
        chunk_id=chunk_id,
        video_id="v1",
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
    )


def _relation(
    memory_id: str,
    start_time: float,
    end_time: float,
    *,
    x: float,
) -> RetrievalMemoryRecord:
    return RetrievalMemoryRecord(
        memory_id=memory_id,
        source_store="spatial",
        video_id="v1",
        start_time=start_time,
        end_time=end_time,
        snippet="mug left of notebook",
        frame_refs=(),
        geometry={
            "subject": "mug",
            "relation": "left_of",
            "object": "notebook",
            "x": x,
        },
    )
