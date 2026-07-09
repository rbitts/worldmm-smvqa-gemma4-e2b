from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, override

from worldmm_smvqa.chunking import build_chunks, read_source_streams
from worldmm_smvqa.memory_sources import build_source_memories
from worldmm_smvqa.retrieval_types import (
    EvidenceItem,
    EvidencePack,
    RetrievalMemoryRecord,
    RetrievalStore,
)
from worldmm_smvqa.worldmm.episodic import build_episodic_graph
from worldmm_smvqa.worldmm.episodic_types import EpisodicNodeRecord, EpisodicRecord
from worldmm_smvqa.worldmm.semantic import (
    SemanticTripleRecord,
    build_semantic_memory,
)
from worldmm_smvqa.worldmm.visual import VisualMemoryRecord, build_visual_memory

if TYPE_CHECKING:
    from worldmm_smvqa.schema import QuestionRequest

STORE_ORDER: Final[tuple[RetrievalStore, ...]] = ("episodic", "semantic", "visual")
DEFAULT_EVIDENCE_BUDGET: Final = 6
STOP_WORDS: Final = frozenset(
    {"a", "and", "is", "on", "the", "what", "where", "which", "with"},
)


@dataclass(frozen=True, slots=True)
class InvalidRetrievalStoreError(Exception):
    store: str

    @override
    def __str__(self) -> str:
        return f"InvalidRetrievalStoreError: {self.store}"


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    record: RetrievalMemoryRecord
    score: float


def retrieve_evidence(
    question: QuestionRequest,
    memory_records: Sequence[RetrievalMemoryRecord],
    *,
    enabled_stores: frozenset[RetrievalStore],
    evidence_budget: int = DEFAULT_EVIDENCE_BUDGET,
) -> EvidencePack:
    requested_stores = _ordered_stores(enabled_stores)
    scoped = tuple(
        record
        for record in memory_records
        if record.video_id == question.video_id
        and record.source_store in enabled_stores
    )
    causal = tuple(
        record for record in scoped if record.end_time <= question.question_time
    )
    scored = tuple(
        sorted(
            (_score_candidate(question, record) for record in causal),
            key=_score_sort_key,
        ),
    )
    evidence = _adaptive_evidence(scored, requested_stores, evidence_budget)
    return EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=requested_stores,
        selected_stores=tuple(dict.fromkeys(item.source_store for item in evidence)),
        evidence_budget=evidence_budget,
        evidence=evidence,
        causal_filtered_count=len(scoped) - len(causal),
    )


def build_fixture_retrieval_stores(
    fixture_dir: Path,
) -> tuple[RetrievalMemoryRecord, ...]:
    sources = read_source_streams(fixture_dir)
    chunks = build_chunks(sources)
    clip_chunks = tuple(chunk for chunk in chunks if chunk.granularity == "clip_30s")
    source_memories = build_source_memories(clip_chunks)
    return build_retrieval_records(
        build_episodic_graph(chunks, source_memories),
        build_semantic_memory(sources),
        build_visual_memory(sources),
    )


def build_retrieval_records(
    episodic: Sequence[EpisodicRecord],
    semantic: Sequence[SemanticTripleRecord],
    visual: Sequence[VisualMemoryRecord],
) -> tuple[RetrievalMemoryRecord, ...]:
    records: list[RetrievalMemoryRecord] = []
    records.extend(
        _episodic_candidate(record)
        for record in episodic
        if isinstance(record, EpisodicNodeRecord)
    )
    records.extend(_semantic_candidate(record) for record in semantic)
    records.extend(_visual_candidate(record) for record in visual)
    return tuple(records)


def parse_retrieval_stores(value: str) -> frozenset[RetrievalStore]:
    stores: list[RetrievalStore] = []
    for part in value.split(","):
        store = part.strip()
        if store:
            stores.append(_parse_retrieval_store(store))
    return frozenset(stores)


def injected_future_memory(question: QuestionRequest) -> RetrievalMemoryRecord:
    return RetrievalMemoryRecord(
        memory_id="injected-post-question-high-score",
        source_store="semantic",
        video_id=question.video_id,
        start_time=question.question_time + 1.0,
        end_time=question.question_time + 2.0,
        snippet=f"{question.question} future perfect high score",
        frame_refs=(),
        base_score=100.0,
    )


def _episodic_candidate(record: EpisodicNodeRecord) -> RetrievalMemoryRecord:
    return RetrievalMemoryRecord(
        memory_id=record.node_id,
        source_store="episodic",
        video_id=record.video_id,
        start_time=record.start_time,
        end_time=record.end_time,
        snippet=_episodic_snippet(record),
        frame_refs=record.frame_refs,
        base_score=record.confidence,
    )


def _semantic_candidate(record: SemanticTripleRecord) -> RetrievalMemoryRecord:
    return RetrievalMemoryRecord(
        memory_id=record.memory_id,
        source_store="semantic",
        video_id=record.video_id,
        start_time=record.start_time,
        end_time=record.end_time,
        snippet=record.text,
        frame_refs=(),
        base_score=record.confidence,
    )


def _visual_candidate(record: VisualMemoryRecord) -> RetrievalMemoryRecord:
    return RetrievalMemoryRecord(
        memory_id=record.memory_id,
        source_store="visual",
        video_id=record.video_id,
        start_time=record.start_time,
        end_time=record.timestamp,
        snippet=" ".join(
            (
                record.source_frame_description,
                *record.ocr_refs,
                *record.object_refs,
            ),
        ),
        frame_refs=(record.frame_ref,),
        base_score=1.0,
    )


def _parse_retrieval_store(store: str) -> RetrievalStore:
    match store:
        case "episodic":
            return "episodic"
        case "semantic":
            return "semantic"
        case "visual":
            return "visual"
        case other:
            raise InvalidRetrievalStoreError(store=other)


def _score_candidate(
    question: QuestionRequest,
    record: RetrievalMemoryRecord,
) -> ScoredCandidate:
    query_terms = _query_terms(question)
    snippet_terms = _tokens(record.snippet)
    overlap = len(query_terms & snippet_terms)
    normalized_overlap = overlap / max(len(query_terms), 1)
    return ScoredCandidate(
        record=record,
        score=round(normalized_overlap + (record.base_score * 0.01), 6),
    )


def _adaptive_evidence(
    scored: Sequence[ScoredCandidate],
    stores: Sequence[RetrievalStore],
    evidence_budget: int,
) -> tuple[EvidenceItem, ...]:
    selected: list[EvidenceItem] = []
    used_ids: set[str] = set()
    while len(selected) < evidence_budget:
        added = False
        for store in stores:
            candidate = _next_store_candidate(scored, store, used_ids)
            if candidate is None:
                continue
            selected.append(_evidence_item(candidate))
            used_ids.add(candidate.record.memory_id)
            added = True
            if len(selected) >= evidence_budget:
                break
        if not added:
            break
    return tuple(selected)


def _next_store_candidate(
    scored: Sequence[ScoredCandidate],
    store: RetrievalStore,
    used_ids: set[str],
) -> ScoredCandidate | None:
    for candidate in scored:
        if (
            candidate.record.source_store == store
            and candidate.record.memory_id not in used_ids
        ):
            return candidate
    return None


def _evidence_item(candidate: ScoredCandidate) -> EvidenceItem:
    record = candidate.record
    return EvidenceItem(
        memory_id=record.memory_id,
        snippet=record.snippet,
        frame_refs=record.frame_refs,
        source_store=record.source_store,
        start_time=record.start_time,
        end_time=record.end_time,
        retrieval_score=candidate.score,
    )


def _query_terms(question: QuestionRequest) -> frozenset[str]:
    return _tokens(
        " ".join(
            (
                question.question,
                *(choice.text for choice in question.answer_choices),
            ),
        ),
    )


def _tokens(text: str) -> frozenset[str]:
    cleaned = "".join(char if char.isalnum() else " " for char in text.lower())
    return frozenset(token for token in cleaned.split() if token not in STOP_WORDS)


def _score_sort_key(candidate: ScoredCandidate) -> tuple[float, str, float, str]:
    record = candidate.record
    return (-candidate.score, record.source_store, record.end_time, record.memory_id)


def _ordered_stores(stores: frozenset[RetrievalStore]) -> tuple[RetrievalStore, ...]:
    return tuple(store for store in STORE_ORDER if store in stores)


def _episodic_snippet(record: EpisodicNodeRecord) -> str:
    return " ".join(
        (
            record.granularity,
            *record.source_modalities,
            *record.source_modality_refs,
            *record.frame_refs,
        ),
    )
