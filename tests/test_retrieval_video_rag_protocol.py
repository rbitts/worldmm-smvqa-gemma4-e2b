from __future__ import annotations

import pytest
from pydantic import ValidationError

from worldmm_smvqa.retrieval_protocols import (
    cap_frame_refs,
    eligible_video_rag_shards,
    filter_records_to_shards,
)
from worldmm_smvqa.retrieval_types import (
    RETRIEVAL_FRAME_REF_CAP,
    RetrievalMemoryRecord,
    RetrievalTrace,
)
from worldmm_smvqa.schema import AnswerChoice, QuestionRequest, StreamChunk


def _question_at_45s() -> QuestionRequest:
    return QuestionRequest(
        question_id="q-video-rag",
        video_id="video-a",
        question="Where is the mug?",
        question_time=45.0,
        answer_choices=(
            AnswerChoice(choice_id="a", text="desk", choice_ltype="place"),
        ),
    )


def _shard(chunk_id: str, start_time: float, end_time: float) -> StreamChunk:
    return StreamChunk(
        chunk_id=chunk_id,
        video_id="video-a",
        start_time=start_time,
        end_time=end_time,
        granularity="shard_30m",
    )


def _memory(
    memory_id: str,
    start_time: float,
    end_time: float,
) -> RetrievalMemoryRecord:
    return RetrievalMemoryRecord(
        memory_id=memory_id,
        source_store="semantic",
        video_id="video-a",
        start_time=start_time,
        end_time=end_time,
        snippet="mug desk future perfect high score",
        frame_refs=(),
        base_score=100.0,
    )


def test_eligible_video_rag_shards_when_question_at_45s() -> None:
    # Given: past/future shards, a same-window clip, and a cross-video shard.
    question = _question_at_45s()
    chunks = (
        _shard("video-a:0:30:shard_30m", 0.0, 30.0),
        _shard("video-a:30:60:shard_30m", 30.0, 60.0),
        StreamChunk(
            chunk_id="video-a:0:30:clip_30s",
            video_id="video-a",
            start_time=0.0,
            end_time=30.0,
            granularity="clip_30s",
        ),
        StreamChunk(
            chunk_id="video-b:0:30:shard_30m",
            video_id="video-b",
            start_time=0.0,
            end_time=30.0,
            granularity="shard_30m",
        ),
    )

    # When: Video-RAG shard eligibility is computed.
    eligible = eligible_video_rag_shards(question, chunks)

    # Then: only same-video shard windows fully before question time remain.
    assert tuple(shard.chunk_id for shard in eligible) == (
        "video-a:0:30:shard_30m",
    )
    trace = RetrievalTrace(
        protocols=("smvqa-video-rag",),
        eligible_shard_ids=tuple(shard.chunk_id for shard in eligible),
        selected_clip_ids=(),
        policy_route="video-rag-shard-eligibility",
        store_order=("semantic",),
        candidate_counts=(),
        causal_filtered_count=0,
        frame_ref_count=0,
    )
    assert trace.eligible_shard_ids == ("video-a:0:30:shard_30m",)


def test_filter_records_to_shards_excludes_future_high_score_before_sorting() -> None:
    # Given: one eligible shard and a future high-score record in the next shard.
    eligible_shards = (_shard("video-a:0:30:shard_30m", 0.0, 30.0),)
    records = (
        _memory("past-memory", 10.0, 20.0),
        _memory("future-high-score-memory", 35.0, 40.0),
        RetrievalMemoryRecord(
            memory_id="cross-video-memory",
            source_store="semantic",
            video_id="video-b",
            start_time=10.0,
            end_time=20.0,
            snippet="mug desk",
            frame_refs=(),
            base_score=200.0,
        ),
    )

    # When: candidate records are scoped to eligible shard spans.
    filtered = filter_records_to_shards(records, eligible_shards)

    # Then: no future or cross-video memory can reach score sorting.
    assert tuple(record.memory_id for record in filtered) == ("past-memory",)


def test_cap_frame_refs_limits_evidence_pack_frame_count() -> None:
    # Given: more frame refs than Video-RAG evidence packing allows.
    frame_refs = tuple(f"frame-{index:04d}" for index in range(40))

    # When: refs are capped for an evidence pack.
    capped = cap_frame_refs(frame_refs)

    # Then: only the first 32 refs remain and the trace accepts that count.
    assert len(capped) == RETRIEVAL_FRAME_REF_CAP
    assert capped == frame_refs[:RETRIEVAL_FRAME_REF_CAP]
    trace = RetrievalTrace(
        protocols=("smvqa-video-rag",),
        eligible_shard_ids=("video-a:0:30:shard_30m",),
        selected_clip_ids=(),
        policy_route="video-rag-shard-eligibility",
        store_order=("visual",),
        candidate_counts=(),
        causal_filtered_count=0,
        frame_ref_count=len(capped),
    )
    assert trace.frame_ref_count == RETRIEVAL_FRAME_REF_CAP
    with pytest.raises(ValidationError, match="frame_ref_count must be <= 32"):
        _ = RetrievalTrace(
            protocols=("smvqa-video-rag",),
            eligible_shard_ids=("video-a:0:30:shard_30m",),
            selected_clip_ids=(),
            policy_route="video-rag-shard-eligibility",
            store_order=("visual",),
            candidate_counts=(),
            causal_filtered_count=0,
            frame_ref_count=RETRIEVAL_FRAME_REF_CAP + 1,
        )
