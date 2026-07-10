from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, override

from pydantic import ValidationError

from worldmm_smvqa.retrieval_types import EvidencePack
from worldmm_smvqa.schema import FrozenModel, QALabelExample
from worldmm_smvqa.worldmm.spatial_types import (
    SpatialRelationKind,
    SpatialRelationRecord,
)

if TYPE_CHECKING:
    from worldmm_smvqa.retrieval_types import EvidenceItem, RetrievalStore

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


@dataclass(frozen=True, slots=True)
class SpatialDiagnosticsError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"SpatialDiagnosticsError: {self.detail}"


class EvidenceSpan(NamedTuple):
    video_id: str
    start: float
    end: float
    store: DiagnosticStore


class RelationTuple(NamedTuple):
    subject: str
    relation: SpatialRelationKind
    object: str
    zone_id: str


class ExpectedSpatialRelation(FrozenModel):
    video_id: str
    subject: str
    relation: SpatialRelationKind
    object: str
    zone_id: str
    distance_m: float | None = None
    delta_x: float | None = None
    delta_y: float | None = None
    delta_z: float | None = None


class RelationAccuracy(FrozenModel):
    precision: float
    recall: float
    f1: float
    true_positive: int
    predicted: int
    expected: int


class RelationMetricAccuracy(FrozenModel):
    precision: float
    recall: float
    f1: float
    true_positive: int
    predicted: int
    expected: int
    distance_tolerance_m: float
    delta_tolerance_m: float


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


def relation_metric_accuracy(
    predicted_relations: Sequence[SpatialRelationRecord],
    expected: Sequence[ExpectedSpatialRelation],
    *,
    distance_tolerance_m: float = 0.25,
    delta_tolerance_m: float = 0.25,
) -> RelationMetricAccuracy:
    predicted = tuple(predicted_relations)
    expected_relations = tuple(expected)
    matched: set[int] = set()
    true_positive = 0
    for expected_relation in expected_relations:
        for index, predicted_relation in enumerate(predicted):
            if index in matched:
                continue
            if _metric_relation_matches(
                predicted_relation,
                expected_relation,
                distance_tolerance_m=distance_tolerance_m,
                delta_tolerance_m=delta_tolerance_m,
            ):
                matched.add(index)
                true_positive += 1
                break
    precision = _ratio(true_positive, len(predicted))
    recall = _ratio(true_positive, len(expected_relations))
    return RelationMetricAccuracy(
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        true_positive=true_positive,
        predicted=len(predicted),
        expected=len(expected_relations),
        distance_tolerance_m=distance_tolerance_m,
        delta_tolerance_m=delta_tolerance_m,
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
            if _has_span_hit(evidence, span, retrieval_store):
                store_hits[span.store] += 1
            for protocol in pack.retrieval_trace.protocols:
                protocol_totals[protocol] += 1
                if _has_span_hit(evidence, span, None):
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


def _metric_relation_matches(
    predicted: SpatialRelationRecord,
    expected: ExpectedSpatialRelation,
    *,
    distance_tolerance_m: float,
    delta_tolerance_m: float,
) -> bool:
    if _relation_tuple(predicted) != _relation_tuple(expected):
        return False
    if expected.distance_m is not None and not _within_tolerance(
        predicted.distance_m,
        expected.distance_m,
        distance_tolerance_m,
    ):
        return False
    return all(
        _within_tolerance(predicted_value, expected_value, delta_tolerance_m)
        for predicted_value, expected_value in (
            (predicted.delta_x, expected.delta_x),
            (predicted.delta_y, expected.delta_y),
            (predicted.delta_z, expected.delta_z),
        )
        if expected_value is not None
    )


def _within_tolerance(
    predicted: float | None,
    expected: float,
    tolerance: float,
) -> bool:
    return predicted is not None and abs(predicted - expected) <= tolerance


def _has_span_hit(
    evidence: Sequence[EvidenceItem],
    span: EvidenceSpan,
    store: RetrievalStore | None,
) -> bool:
    return any(_item_hits_span(item, span, store) for item in evidence)


def _item_hits_span(
    item: EvidenceItem,
    span: EvidenceSpan,
    store: RetrievalStore | None,
) -> bool:
    if store is not None and item.source_store != store:
        return False
    if item.video_id != span.video_id:
        return False
    if item.start_time == item.end_time:
        return span.start <= item.start_time <= span.end
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


def write_spatial_retrieval_diagnostics(
    evidence_path: Path,
    label_path: Path,
    output: Path,
) -> None:
    packs = _read_packs(evidence_path)
    labels = _read_labels(label_path)
    labels_by_id = {label.question_id: label for label in labels}
    recall = {
        str(k): memory_recall_at_k(packs, labels, k).model_dump()
        for k in (1, 3, 5)
    }
    evidence = tuple(item for pack in packs for item in pack.evidence)
    payload = {
        "packs": len(packs),
        "spatial_selected_packs": sum(
            1 for pack in packs if "spatial" in pack.selected_stores
        ),
        "spatial_evidence_items": sum(
            1 for item in evidence if item.source_store == "spatial"
        ),
        "geometry_evidence_items": sum(
            1 for item in evidence if item.geometry is not None
        ),
        "cross_video_evidence_items": sum(
            1
            for pack in packs
            for item in pack.evidence
            if item.video_id != pack.video_id
        ),
        "causal_violation_count": sum(
            1
            for pack in packs
            for item in pack.evidence
            if (label := labels_by_id.get(pack.question_id)) is not None
            and item.end_time > label.question_time
        ),
        "memory_recall": recall,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            _ = stream.write("\n")
        _ = temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_packs(path: Path) -> tuple[EvidencePack, ...]:
    try:
        return tuple(
            EvidencePack.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, ValidationError) as exc:
        raise SpatialDiagnosticsError(detail=f"{path}: {exc}") from exc


def _read_labels(path: Path) -> tuple[QALabelExample, ...]:
    try:
        return tuple(
            QALabelExample.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, ValidationError) as exc:
        raise SpatialDiagnosticsError(detail=f"{path}: {exc}") from exc
