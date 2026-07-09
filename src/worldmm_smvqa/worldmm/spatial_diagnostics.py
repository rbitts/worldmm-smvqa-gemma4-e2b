from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, override

from worldmm_smvqa.schema import FrozenModel
from worldmm_smvqa.worldmm.spatial_types import SpatialRelationRecord

if TYPE_CHECKING:
    from worldmm_smvqa.retrieval_types import EvidenceItem, EvidencePack, RetrievalStore
    from worldmm_smvqa.schema import QALabelExample

type DiagnosticStore = Literal[
    "transcript",
    "ocr",
    "episodic",
    "semantic",
    "visual",
    "spatial",
]
type DiagnosticProtocol = Literal["smvqa-video-rag", "egobutler", "worldmm"]

EVIDENCE_SPAN_PARTS: Final = 4
STORES: tuple[DiagnosticStore, ...] = (
    "transcript",
    "ocr",
    "episodic",
    "semantic",
    "visual",
    "spatial",
)
PROTOCOLS: tuple[DiagnosticProtocol, ...] = (
    "smvqa-video-rag",
    "egobutler",
    "worldmm",
)


@dataclass(frozen=True, slots=True)
class InvalidEvidenceSpanError(Exception):
    raw_span: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"InvalidEvidenceSpanError: {self.raw_span}: {self.detail}"


class EvidenceSpan(NamedTuple):
    video_id: str
    start: float
    end: float
    store: DiagnosticStore


class RelationTuple(NamedTuple):
    subject: str
    relation: Literal["near"]
    object: str
    zone_id: str


class ExpectedSpatialRelation(FrozenModel):
    video_id: str
    subject: str
    relation: Literal["near"]
    object: str
    zone_id: str


class RelationAccuracy(FrozenModel):
    precision: float
    recall: float
    f1: float
    true_positive: int
    predicted: int
    expected: int


class MemoryRecallAtK(FrozenModel):
    k: int
    recall_at_k: dict[DiagnosticStore, float]
    protocol_recall_at_k: dict[DiagnosticProtocol, float]


type RelationInput = SpatialRelationRecord | ExpectedSpatialRelation | RelationTuple


def parse_evidence_span(raw_span: str) -> EvidenceSpan:
    parts = raw_span.split(":")
    if len(parts) != EVIDENCE_SPAN_PARTS:
        raise InvalidEvidenceSpanError(
            raw_span=raw_span,
            detail="expected video:start:end:store",
        )
    video_id, raw_start, raw_end, raw_store = parts
    try:
        start = float(raw_start)
        end = float(raw_end)
    except ValueError as exc:
        raise InvalidEvidenceSpanError(
            raw_span=raw_span,
            detail="start and end must be numbers",
        ) from exc
    if end <= start:
        raise InvalidEvidenceSpanError(
            raw_span=raw_span,
            detail="end must be greater than start",
        )
    return EvidenceSpan(
        video_id=video_id,
        start=start,
        end=end,
        store=_parse_store(raw_span, raw_store),
    )


def relation_accuracy(
    predicted_relations: Sequence[RelationInput],
    expected: Sequence[RelationInput],
) -> RelationAccuracy:
    predicted_set = {_relation_tuple(relation) for relation in predicted_relations}
    expected_set = {_relation_tuple(relation) for relation in expected}
    true_positive = len(predicted_set & expected_set)
    precision = _ratio(true_positive, len(predicted_set))
    recall = _ratio(true_positive, len(expected_set))
    return RelationAccuracy(
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        true_positive=true_positive,
        predicted=len(predicted_set),
        expected=len(expected_set),
    )


def memory_recall_at_k(
    evidence_packs: Sequence[EvidencePack],
    labels: Sequence[QALabelExample],
    k: int,
) -> MemoryRecallAtK:
    packs_by_id = {pack.question_id: pack for pack in evidence_packs}
    store_hits = dict.fromkeys(STORES, 0)
    store_totals = dict.fromkeys(STORES, 0)
    protocol_hits = dict.fromkeys(PROTOCOLS, 0)
    protocol_totals = dict.fromkeys(PROTOCOLS, 0)
    for label in labels:
        pack = packs_by_id.get(label.question_id)
        for span in tuple(parse_evidence_span(raw) for raw in label.evidence_list):
            store_totals[span.store] += 1
            if pack is None:
                continue
            evidence = pack.evidence[:k]
            retrieval_store = _retrieval_store_for_label(span.store)
            if _has_span_hit(pack, evidence, span, retrieval_store):
                store_hits[span.store] += 1
            for protocol in pack.retrieval_trace.protocols:
                protocol_totals[protocol] += 1
                if _has_span_hit(pack, evidence, span, None):
                    protocol_hits[protocol] += 1
    return MemoryRecallAtK(
        k=k,
        recall_at_k={
            store: _ratio(store_hits[store], store_totals[store])
            for store in STORES
        },
        protocol_recall_at_k={
            protocol: _ratio(protocol_hits[protocol], protocol_totals[protocol])
            for protocol in PROTOCOLS
        },
    )


def _parse_store(raw_span: str, raw_store: str) -> DiagnosticStore:
    match raw_store:
        case "transcript":
            return "transcript"
        case "ocr":
            return "ocr"
        case "episodic":
            return "episodic"
        case "semantic":
            return "semantic"
        case "visual":
            return "visual"
        case "spatial":
            return "spatial"
        case other:
            raise InvalidEvidenceSpanError(
                raw_span=raw_span,
                detail=f"unknown store: {other}",
            )


def _retrieval_store_for_label(store: DiagnosticStore) -> RetrievalStore:
    match store:
        case "transcript" | "episodic":
            return "episodic"
        case "ocr" | "visual":
            return "visual"
        case "semantic":
            return "semantic"
        case "spatial":
            return "spatial"


def _relation_tuple(relation: RelationInput) -> RelationTuple:
    match relation:
        case SpatialRelationRecord(
            subject=subject,
            relation=raw_relation,
            object=target,
            zone_id=zone_id,
        ):
            return RelationTuple(
                subject=subject,
                relation=raw_relation,
                object=target,
                zone_id=zone_id,
            )
        case ExpectedSpatialRelation(
            subject=subject,
            relation=raw_relation,
            object=target,
            zone_id=zone_id,
        ):
            return RelationTuple(
                subject=subject,
                relation=raw_relation,
                object=target,
                zone_id=zone_id,
            )
        case RelationTuple():
            return relation


def _has_span_hit(
    pack: EvidencePack,
    evidence: Sequence[EvidenceItem],
    span: EvidenceSpan,
    store: RetrievalStore | None,
) -> bool:
    if pack.video_id != span.video_id:
        return False
    return any(_item_hits_span(item, span, store) for item in evidence)


def _item_hits_span(
    item: EvidenceItem,
    span: EvidenceSpan,
    store: RetrievalStore | None,
) -> bool:
    if store is not None and item.source_store != store:
        return False
    return item.start_time < span.end and span.start < item.end_time


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    if denominator == 0.0:
        return 0.0
    return (2.0 * precision * recall) / denominator
