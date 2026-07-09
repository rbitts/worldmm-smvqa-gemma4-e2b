from __future__ import annotations

import pytest
from pydantic import ValidationError

from worldmm_smvqa.retrieval_types import (
    EvidenceItem,
    EvidencePack,
    RetrievalCandidateCount,
    RetrievalTrace,
)


def _evidence_item() -> EvidenceItem:
    return EvidenceItem(
        memory_id="memory-001",
        snippet="mug last seen beside the notebook",
        frame_refs=("frame-001",),
        source_store="semantic",
        start_time=1.0,
        end_time=2.0,
        retrieval_score=0.75,
    )


def _trace(frame_ref_count: int = 1) -> RetrievalTrace:
    return RetrievalTrace(
        protocols=("smvqa-video-rag", "egobutler", "worldmm"),
        eligible_shard_ids=("fake_video_001_shard_0000",),
        selected_clip_ids=("fake_video_001_clip_0000",),
        policy_route="balanced",
        store_order=("episodic", "semantic", "visual"),
        candidate_counts=(
            RetrievalCandidateCount(
                source_store="episodic",
                before_causal_filter=2,
                after_causal_filter=1,
            ),
            RetrievalCandidateCount(
                source_store="semantic",
                before_causal_filter=3,
                after_causal_filter=2,
            ),
            RetrievalCandidateCount(
                source_store="visual",
                before_causal_filter=1,
                after_causal_filter=1,
            ),
        ),
        causal_filtered_count=4,
        frame_ref_count=frame_ref_count,
    )


def _pack(trace: RetrievalTrace | None) -> EvidencePack:
    if trace is None:
        return EvidencePack(
            question_id="q_fake_001",
            video_id="fake_video_001",
            requested_stores=("episodic", "semantic", "visual"),
            selected_stores=("semantic",),
            evidence_budget=1,
            evidence=(_evidence_item(),),
            causal_filtered_count=4,
        )
    return EvidencePack(
        question_id="q_fake_001",
        video_id="fake_video_001",
        requested_stores=("episodic", "semantic", "visual"),
        selected_stores=("semantic",),
        evidence_budget=1,
        evidence=(_evidence_item(),),
        causal_filtered_count=4,
        retrieval_trace=trace,
    )


def test_evidence_pack_round_trips_retrieval_trace() -> None:
    # Given: a new EvidencePack carrying the protocol trace contract.
    pack = _pack(_trace())

    # When: the pack crosses the JSON boundary.
    parsed = EvidencePack.model_validate_json(pack.model_dump_json())

    # Then: trace fields survive with typed protocol and store data.
    assert parsed.retrieval_trace.protocols == (
        "smvqa-video-rag",
        "egobutler",
        "worldmm",
    )
    assert parsed.retrieval_trace.eligible_shard_ids == (
        "fake_video_001_shard_0000",
    )
    assert parsed.retrieval_trace.selected_clip_ids == (
        "fake_video_001_clip_0000",
    )
    assert parsed.retrieval_trace.policy_route == "balanced"
    assert parsed.retrieval_trace.store_order == (
        "episodic",
        "semantic",
        "visual",
    )
    assert parsed.retrieval_trace.candidate_counts[1] == RetrievalCandidateCount(
        source_store="semantic",
        before_causal_filter=3,
        after_causal_filter=2,
    )
    assert parsed.retrieval_trace.causal_filtered_count == 4
    assert parsed.retrieval_trace.frame_ref_count == 1


def test_evidence_pack_uses_documented_legacy_trace_default() -> None:
    # Given: a legacy fixture-like EvidencePack payload with no trace.
    legacy_pack = _pack(trace=None)

    # When: the pack is parsed.
    parsed = EvidencePack.model_validate_json(legacy_pack.model_dump_json())

    # Then: the compatibility trace is explicit and empty.
    assert parsed.retrieval_trace.protocols == ()
    assert parsed.retrieval_trace.eligible_shard_ids == ()
    assert parsed.retrieval_trace.selected_clip_ids == ()
    assert parsed.retrieval_trace.policy_route == "legacy-missing-trace"
    assert parsed.retrieval_trace.store_order == ()
    assert parsed.retrieval_trace.candidate_counts == ()
    assert parsed.retrieval_trace.causal_filtered_count == 0
    assert parsed.retrieval_trace.frame_ref_count == 0


def test_retrieval_trace_rejects_frame_ref_count_over_cap() -> None:
    # Given / When / Then: the schema cap rejects impossible frame counts.
    with pytest.raises(ValidationError, match="frame_ref_count must be <= 32"):
        _ = _trace(frame_ref_count=33)


def test_retrieval_trace_schema_has_no_label_fields() -> None:
    # Given: the trace model fields.
    field_names = set(RetrievalTrace.model_fields) | set(
        RetrievalCandidateCount.model_fields,
    )

    # When: checking leakage-sensitive names.
    forbidden_fragments = (
        "answer",
        "choice",
        "label",
        "evidence_list",
        "verification",
    )

    # Then: trace fields contain no label or answer surface.
    assert not {
        field
        for field in field_names
        for fragment in forbidden_fragments
        if fragment in field
    }
