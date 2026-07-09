from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import TYPE_CHECKING, Final, override

from worldmm_smvqa.chunking import build_chunks, read_source_streams
from worldmm_smvqa.memory_sources import build_source_memories
from worldmm_smvqa.retrieval_protocols import (
    WorldMMRetrievalPolicy,
    build_egobutler_hierarchy,
    cap_frame_refs,
    coarse_to_fine_candidates,
    eligible_video_rag_shards,
    filter_records_to_shards,
)
from worldmm_smvqa.retrieval_types import (
    RETRIEVAL_FRAME_REF_CAP,
    EvidenceItem,
    EvidencePack,
    RetrievalCandidateCount,
    RetrievalMemoryRecord,
    RetrievalStore,
    RetrievalTrace,
)
from worldmm_smvqa.worldmm.episodic import build_episodic_graph
from worldmm_smvqa.worldmm.episodic_types import EpisodicNodeRecord, EpisodicRecord
from worldmm_smvqa.worldmm.semantic import (
    SemanticTripleRecord,
    build_semantic_memory,
)
from worldmm_smvqa.worldmm.spatial import (
    build_object_anchors,
    build_object_state_snapshots,
    build_trajectory_summaries,
    build_zones,
    derive_relations,
)
from worldmm_smvqa.worldmm.spatial_types import (
    ObjectStateSnapshotRecord,
    SpatialAnchorRecord,
    SpatialRelationRecord,
    WearerTrajectorySummaryRecord,
    ZoneRecord,
)
from worldmm_smvqa.worldmm.visual import VisualMemoryRecord, build_visual_memory

if TYPE_CHECKING:
    from worldmm_smvqa.schema import QuestionRequest, StreamChunk

STORE_ORDER: Final[tuple[RetrievalStore, ...]] = (
    "episodic",
    "semantic",
    "visual",
    "spatial",
)
DEFAULT_EVIDENCE_BUDGET: Final = 6
CLIP_SECONDS: Final = 30.0
SHARD_SECONDS: Final = 1800.0
STOP_WORDS: Final = frozenset(
    {"a", "and", "is", "on", "the", "what", "where", "which", "with"},
)

type SpatialRetrievalRecord = (
    ZoneRecord
    | SpatialAnchorRecord
    | SpatialRelationRecord
    | ObjectStateSnapshotRecord
    | WearerTrajectorySummaryRecord
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


@dataclass(frozen=True, slots=True)
class ProtocolSelection:
    records: tuple[RetrievalMemoryRecord, ...]
    eligible_shard_ids: tuple[str, ...]
    selected_clip_ids: tuple[str, ...]
    causal_filtered_count: int
    candidate_counts: tuple[RetrievalCandidateCount, ...]


@dataclass(frozen=True, slots=True)
class EvidenceSelection:
    items: tuple[EvidenceItem, ...]
    frame_ref_count: int


@dataclass(frozen=True, slots=True)
class RetrievalOptions:
    evidence_budget: int = DEFAULT_EVIDENCE_BUDGET
    chunks: Sequence[StreamChunk] | None = None
    max_frame_refs: int = RETRIEVAL_FRAME_REF_CAP


DEFAULT_RETRIEVAL_OPTIONS: Final = RetrievalOptions()


def retrieve_evidence(
    question: QuestionRequest,
    memory_records: Sequence[RetrievalMemoryRecord],
    *,
    enabled_stores: frozenset[RetrievalStore],
    options: RetrievalOptions = DEFAULT_RETRIEVAL_OPTIONS,
) -> EvidencePack:
    requested_stores = _ordered_stores(enabled_stores)
    scoped = tuple(
        record
        for record in memory_records
        if record.video_id == question.video_id
        and record.source_store in enabled_stores
    )
    route = WorldMMRetrievalPolicy().route(
        question,
        available_stores=requested_stores,
    )
    selected = _protocol_records(
        question,
        scoped,
        requested_stores,
        chunks=options.chunks,
    )
    scored = tuple(
        sorted(
            (_score_candidate(question, record) for record in selected.records),
            key=_score_sort_key,
        ),
    )
    evidence = _policy_evidence(
        scored,
        route.store_order,
        options.evidence_budget,
        max_frame_refs=options.max_frame_refs,
    )
    return EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=requested_stores,
        selected_stores=tuple(
            dict.fromkeys(item.source_store for item in evidence.items),
        ),
        evidence_budget=options.evidence_budget,
        evidence=evidence.items,
        causal_filtered_count=selected.causal_filtered_count,
        retrieval_trace=RetrievalTrace(
            protocols=("smvqa-video-rag", "egobutler", "worldmm"),
            eligible_shard_ids=selected.eligible_shard_ids,
            selected_clip_ids=selected.selected_clip_ids,
            policy_route=route.reason,
            store_order=tuple(route.store_order),
            candidate_counts=selected.candidate_counts,
            causal_filtered_count=selected.causal_filtered_count,
            frame_ref_count=evidence.frame_ref_count,
        ),
    )


def build_fixture_retrieval_stores(
    fixture_dir: Path,
) -> tuple[RetrievalMemoryRecord, ...]:
    sources = read_source_streams(fixture_dir)
    chunks = build_chunks(sources)
    clip_chunks = tuple(chunk for chunk in chunks if chunk.granularity == "clip_30s")
    source_memories = build_source_memories(clip_chunks)
    zones = tuple(record for source in sources for record in build_zones(source))
    anchors = tuple(
        record for source in sources for record in build_object_anchors(source)
    )
    trajectory_chunks = tuple(
        chunk for chunk in clip_chunks if _has_zone_overlap(chunk, zones)
    )
    return build_retrieval_records(
        build_episodic_graph(chunks, source_memories),
        build_semantic_memory(sources),
        build_visual_memory(sources),
        (
            *zones,
            *anchors,
            *tuple(derive_relations(anchors)),
            *build_object_state_snapshots(clip_chunks, anchors),
            *build_trajectory_summaries(trajectory_chunks, zones),
        ),
    )


def build_retrieval_records(
    episodic: Sequence[EpisodicRecord],
    semantic: Sequence[SemanticTripleRecord],
    visual: Sequence[VisualMemoryRecord],
    spatial: Sequence[SpatialRetrievalRecord] = (),
) -> tuple[RetrievalMemoryRecord, ...]:
    records: list[RetrievalMemoryRecord] = []
    records.extend(
        _episodic_candidate(record)
        for record in episodic
        if isinstance(record, EpisodicNodeRecord)
    )
    records.extend(_semantic_candidate(record) for record in semantic)
    records.extend(_visual_candidate(record) for record in visual)
    records.extend(_spatial_candidate(record) for record in spatial)
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


def _spatial_candidate(record: SpatialRetrievalRecord) -> RetrievalMemoryRecord:
    match record:
        case ZoneRecord():
            start_time, end_time = _zone_time_span(record)
            return RetrievalMemoryRecord(
                memory_id=f"spatial_zone:{record.zone_id}",
                source_store="spatial",
                video_id=record.video_id,
                start_time=start_time,
                end_time=end_time,
                snippet=(
                    f"zone {record.zone_id} centered near "
                    f"({_format_float(record.centroid_x)},"
                    f"{_format_float(record.centroid_y)})"
                ),
                frame_refs=(),
            )
        case SpatialAnchorRecord():
            return RetrievalMemoryRecord(
                memory_id=record.memory_id,
                source_store="spatial",
                video_id=record.video_id,
                start_time=record.start_time,
                end_time=record.end_time,
                snippet=(
                    f"{record.object_label} anchored in {record.zone_id} "
                    f"during [{_format_float(record.start_time)},"
                    f"{_format_float(record.end_time)}] near "
                    f"({_format_float(record.x)},{_format_float(record.y)})"
                ),
                frame_refs=record.frame_refs,
                base_score=record.confidence,
            )
        case SpatialRelationRecord():
            return RetrievalMemoryRecord(
                memory_id=record.memory_id,
                source_store="spatial",
                video_id=record.video_id,
                start_time=record.start_time,
                end_time=record.end_time,
                snippet=(
                    f"{record.subject} {record.relation} {record.object} "
                    f"in {record.zone_id} during "
                    f"[{_format_float(record.start_time)},"
                    f"{_format_float(record.end_time)}]"
                ),
                frame_refs=(),
                base_score=1.0,
            )
        case ObjectStateSnapshotRecord() | WearerTrajectorySummaryRecord():
            return RetrievalMemoryRecord(
                memory_id=record.memory_id,
                source_store="spatial",
                video_id=record.video_id,
                start_time=record.start_time,
                end_time=record.end_time,
                snippet=record.snippet,
                frame_refs=(),
                base_score=record.base_score,
            )


def _parse_retrieval_store(store: str) -> RetrievalStore:
    match store:
        case "episodic":
            return "episodic"
        case "semantic":
            return "semantic"
        case "visual":
            return "visual"
        case "spatial":
            return "spatial"
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


def _protocol_records(
    question: QuestionRequest,
    scoped: Sequence[RetrievalMemoryRecord],
    requested_stores: Sequence[RetrievalStore],
    *,
    chunks: Sequence[StreamChunk] | None,
) -> ProtocolSelection:
    if chunks is None:
        return _legacy_protocol_records(question, scoped, requested_stores)

    causal = tuple(
        record for record in scoped if record.end_time <= question.question_time
    )
    eligible_shards = eligible_video_rag_shards(question, chunks)
    shard_scoped = filter_records_to_shards(causal, eligible_shards)
    hierarchy = build_egobutler_hierarchy(chunks, scoped)
    coarse_selection = coarse_to_fine_candidates(question, hierarchy, scoped)
    selected_memory_ids = {
        record.memory_id for record in coarse_selection.records
    }
    selected_records = tuple(
        record for record in shard_scoped if record.memory_id in selected_memory_ids
    )
    return ProtocolSelection(
        records=selected_records,
        eligible_shard_ids=tuple(shard.chunk_id for shard in eligible_shards),
        selected_clip_ids=coarse_selection.selected_clip_ids,
        causal_filtered_count=len(scoped) - len(causal),
        candidate_counts=_candidate_counts(scoped, causal, requested_stores),
    )


def _legacy_protocol_records(
    question: QuestionRequest,
    scoped: Sequence[RetrievalMemoryRecord],
    requested_stores: Sequence[RetrievalStore],
) -> ProtocolSelection:
    causal = tuple(
        record for record in scoped if record.end_time <= question.question_time
    )
    shard_ids = _eligible_shard_ids(question, causal)
    shard_scoped = tuple(
        record for record in causal if _record_shard_id(record) in shard_ids
    )
    selected_clip_ids = _selected_clip_ids(question, shard_scoped)
    selected_records = tuple(
        record
        for record in shard_scoped
        if not selected_clip_ids or _record_clip_id(record) in selected_clip_ids
    )
    return ProtocolSelection(
        records=selected_records,
        eligible_shard_ids=shard_ids,
        selected_clip_ids=selected_clip_ids,
        causal_filtered_count=len(scoped) - len(causal),
        candidate_counts=_candidate_counts(scoped, causal, requested_stores),
    )


def _policy_evidence(
    scored: Sequence[ScoredCandidate],
    stores: Sequence[RetrievalStore],
    evidence_budget: int,
    *,
    max_frame_refs: int,
) -> EvidenceSelection:
    selected: list[EvidenceItem] = []
    used_ids: set[str] = set()
    frame_refs: list[str] = []
    for store in stores:
        for candidate in _store_candidates(scored, store):
            if candidate.record.memory_id in used_ids:
                continue
            item, item_frame_refs = _evidence_item(
                candidate,
                remaining_frame_refs=max_frame_refs - len(frame_refs),
            )
            selected.append(item)
            frame_refs.extend(item_frame_refs)
            used_ids.add(candidate.record.memory_id)
            if len(selected) >= evidence_budget:
                return EvidenceSelection(
                    items=tuple(selected),
                    frame_ref_count=len(frame_refs),
                )
    return EvidenceSelection(
        items=tuple(selected),
        frame_ref_count=len(frame_refs),
    )


def _store_candidates(
    scored: Sequence[ScoredCandidate],
    store: RetrievalStore,
) -> tuple[ScoredCandidate, ...]:
    return tuple(
        candidate for candidate in scored if candidate.record.source_store == store
    )


def _evidence_item(
    candidate: ScoredCandidate,
    *,
    remaining_frame_refs: int,
) -> tuple[EvidenceItem, tuple[str, ...]]:
    record = candidate.record
    frame_refs = cap_frame_refs(record.frame_refs, remaining_frame_refs)
    return (
        EvidenceItem(
            memory_id=record.memory_id,
            snippet=record.snippet,
            frame_refs=frame_refs,
            source_store=record.source_store,
            start_time=record.start_time,
            end_time=record.end_time,
            retrieval_score=candidate.score,
        ),
        frame_refs,
    )


def _eligible_shard_ids(
    question: QuestionRequest,
    causal_records: Sequence[RetrievalMemoryRecord],
) -> tuple[str, ...]:
    shard_ids = tuple(
        dict.fromkeys(
            _record_shard_id(record)
            for record in sorted(causal_records, key=_record_time_key)
            if _record_shard_end(record) <= question.question_time
        ),
    )
    if shard_ids or not causal_records:
        return shard_ids
    first_record = min(causal_records, key=_record_time_key)
    return (_record_shard_id(first_record),)


def _selected_clip_ids(
    question: QuestionRequest,
    records: Sequence[RetrievalMemoryRecord],
) -> tuple[str, ...]:
    clip_ids = tuple(
        dict.fromkeys(
            _record_clip_id(record)
            for record in sorted(records, key=_record_time_key)
        ),
    )
    if not clip_ids:
        return ()
    ranked = sorted(
        clip_ids,
        key=lambda clip_id: _clip_sort_key(question, clip_id, records),
    )
    return (ranked[0],)


def _candidate_counts(
    scoped: Sequence[RetrievalMemoryRecord],
    causal: Sequence[RetrievalMemoryRecord],
    stores: Sequence[RetrievalStore],
) -> tuple[RetrievalCandidateCount, ...]:
    return tuple(
        RetrievalCandidateCount(
            source_store=store,
            before_causal_filter=sum(
                1 for record in scoped if record.source_store == store
            ),
            after_causal_filter=sum(
                1 for record in causal if record.source_store == store
            ),
        )
        for store in stores
    )


def _clip_sort_key(
    question: QuestionRequest,
    clip_id: str,
    records: Sequence[RetrievalMemoryRecord],
) -> tuple[float, float, str]:
    snippet = " ".join(
        record.snippet for record in records if _record_clip_id(record) == clip_id
    )
    query_terms = _query_terms(question)
    score = 0.0
    if query_terms:
        score = len(query_terms & _tokens(snippet)) / len(query_terms)
    return (-score, _window_start_from_id(clip_id), clip_id)


def _record_shard_id(record: RetrievalMemoryRecord) -> str:
    start = floor(record.start_time / SHARD_SECONDS) * SHARD_SECONDS
    return _window_id(
        record.video_id,
        start,
        start + SHARD_SECONDS,
        "shard_30m",
    )


def _record_shard_end(record: RetrievalMemoryRecord) -> float:
    start = floor(record.start_time / SHARD_SECONDS) * SHARD_SECONDS
    return start + SHARD_SECONDS


def _record_clip_id(record: RetrievalMemoryRecord) -> str:
    start = floor(record.start_time / CLIP_SECONDS) * CLIP_SECONDS
    return _window_id(
        record.video_id,
        start,
        start + CLIP_SECONDS,
        "clip_30s",
    )


def _window_id(
    video_id: str,
    start_time: float,
    end_time: float,
    granularity: str,
) -> str:
    return (
        f"{video_id}:{_format_float(start_time)}:"
        f"{_format_float(end_time)}:{granularity}"
    )


def _window_start_from_id(window_id: str) -> float:
    return float(window_id.split(":")[1])


def _record_time_key(record: RetrievalMemoryRecord) -> tuple[float, float, str]:
    return (record.start_time, record.end_time, record.memory_id)


def _zone_time_span(record: ZoneRecord) -> tuple[float, float]:
    starts = tuple(start for start, _end in record.visit_intervals)
    ends = tuple(end for _start, end in record.visit_intervals)
    return min(starts), max(ends)


def _has_zone_overlap(
    chunk: StreamChunk,
    zones: Sequence[ZoneRecord],
) -> bool:
    return any(
        zone.video_id == chunk.video_id
        and any(
            start < chunk.end_time and chunk.start_time < end
            for start, end in zone.visit_intervals
        )
        for zone in zones
    )


def _format_float(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


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
