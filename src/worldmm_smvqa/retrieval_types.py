from __future__ import annotations

from typing import Final, Literal, Self, cast

from pydantic import Field, model_validator

from worldmm_smvqa.schema import FrozenModel

type RetrievalStore = Literal["episodic", "semantic", "visual", "spatial"]
type RetrievalProtocol = Literal["smvqa-video-rag", "egobutler", "worldmm"]

RETRIEVAL_FRAME_REF_CAP: Final = 32


class RetrievalMemoryRecord(FrozenModel):
    memory_id: str
    source_store: RetrievalStore
    video_id: str
    start_time: float
    end_time: float
    snippet: str
    frame_refs: tuple[str, ...]
    base_score: float = 0.0
    geometry: dict[str, float | str] | None = None


class EvidenceItem(FrozenModel):
    memory_id: str
    video_id: str
    snippet: str
    frame_refs: tuple[str, ...]
    source_store: RetrievalStore
    start_time: float
    end_time: float
    retrieval_score: float
    geometry: dict[str, float | str] | None = None


class RetrievalCandidateCount(FrozenModel):
    source_store: RetrievalStore
    before_causal_filter: int
    after_causal_filter: int

    @model_validator(mode="after")
    def _require_valid_counts(self) -> Self:
        if self.before_causal_filter < 0:
            msg = "before_causal_filter must be >= 0"
            raise ValueError(msg)
        if self.after_causal_filter < 0:
            msg = "after_causal_filter must be >= 0"
            raise ValueError(msg)
        if self.after_causal_filter > self.before_causal_filter:
            msg = "after_causal_filter must be <= before_causal_filter"
            raise ValueError(msg)
        return self


class RetrievalTrace(FrozenModel):
    protocols: tuple[RetrievalProtocol, ...]
    eligible_shard_ids: tuple[str, ...]
    selected_clip_ids: tuple[str, ...]
    policy_route: str
    store_order: tuple[RetrievalStore, ...]
    candidate_counts: tuple[RetrievalCandidateCount, ...]
    causal_filtered_count: int
    frame_ref_count: int

    @model_validator(mode="after")
    def _require_valid_trace_counts(self) -> Self:
        if self.causal_filtered_count < 0:
            msg = "causal_filtered_count must be >= 0"
            raise ValueError(msg)
        if self.frame_ref_count < 0:
            msg = "frame_ref_count must be >= 0"
            raise ValueError(msg)
        if self.frame_ref_count > RETRIEVAL_FRAME_REF_CAP:
            msg = f"frame_ref_count must be <= {RETRIEVAL_FRAME_REF_CAP}"
            raise ValueError(msg)
        return self


def legacy_missing_retrieval_trace() -> RetrievalTrace:
    """Compatibility factory for legacy fixture JSON written before traces."""
    return RetrievalTrace(
        protocols=(),
        eligible_shard_ids=(),
        selected_clip_ids=(),
        policy_route="legacy-missing-trace",
        store_order=(),
        candidate_counts=(),
        causal_filtered_count=0,
        frame_ref_count=0,
    )


class EvidencePack(FrozenModel):
    question_id: str
    video_id: str
    requested_stores: tuple[RetrievalStore, ...]
    selected_stores: tuple[RetrievalStore, ...]
    evidence_budget: int
    evidence: tuple[EvidenceItem, ...]
    causal_filtered_count: int
    retrieval_trace: RetrievalTrace = Field(
        default_factory=legacy_missing_retrieval_trace,
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_legacy_evidence_video_ids(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = cast("dict[str, object]", value)
        video_id = payload.get("video_id")
        evidence = payload.get("evidence")
        if not isinstance(video_id, str) or not isinstance(evidence, (list, tuple)):
            return payload
        evidence_items = cast("list[object] | tuple[object, ...]", evidence)
        updated: list[object] = []
        for item in evidence_items:
            if isinstance(item, dict):
                evidence_item = cast("dict[str, object]", item)
                if "video_id" not in evidence_item:
                    evidence_item = {**evidence_item, "video_id": video_id}
                updated.append(evidence_item)
            else:
                updated.append(item)
        return {**payload, "evidence": tuple(updated)}
