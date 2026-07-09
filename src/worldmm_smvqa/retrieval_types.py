from __future__ import annotations

from typing import Literal

from worldmm_smvqa.schema import FrozenModel

type RetrievalStore = Literal["episodic", "semantic", "visual"]


class RetrievalMemoryRecord(FrozenModel):
    memory_id: str
    source_store: RetrievalStore
    video_id: str
    start_time: float
    end_time: float
    snippet: str
    frame_refs: tuple[str, ...]
    base_score: float = 0.0


class EvidenceItem(FrozenModel):
    memory_id: str
    snippet: str
    frame_refs: tuple[str, ...]
    source_store: RetrievalStore
    start_time: float
    end_time: float
    retrieval_score: float


class EvidencePack(FrozenModel):
    question_id: str
    video_id: str
    requested_stores: tuple[RetrievalStore, ...]
    selected_stores: tuple[RetrievalStore, ...]
    evidence_budget: int
    evidence: tuple[EvidenceItem, ...]
    causal_filtered_count: int
