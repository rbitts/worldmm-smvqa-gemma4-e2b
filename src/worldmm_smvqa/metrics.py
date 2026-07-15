from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Final, override

from pydantic import BaseModel, Field, ValidationError

from worldmm_smvqa.schema import (
    ANSWER_CHOICE_COUNT,
    FrozenModel,
    PredictionRecord,
    QALabelExample,
    is_unanswerable_choice,
)

EVIDENCE_SPAN_PARTS: Final = 4
PERCENT_SCALE: Final = 100.0
EMPTY_EVALUATION_SOURCE: Final = "labels"
EMPTY_EVALUATION_DETAIL: Final = "evaluation set is empty"


@dataclass(frozen=True, slots=True)
class InvalidPredictionError(Exception):
    question_id: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"InvalidPredictionError: {self.question_id}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ScoredPrediction:
    label: QALabelExample
    prediction: PredictionRecord
    correct_rank: int | None


class SummaryStats(FrozenModel):
    count: int
    total: int
    min: int
    max: int
    mean: float


class MetricDiagnostics(FrozenModel):
    causal_violation_count: int
    prompt_tokens: SummaryStats
    memory_size: SummaryStats


class EvaluationMetrics(FrozenModel):
    ans_f1: float = Field(ge=0.0, le=100.0, serialization_alias="Ans-F1")
    qa_acc: float = Field(ge=0.0, le=100.0, serialization_alias="QA-Acc")
    qa_mrr: float = Field(ge=0.0, le=100.0, serialization_alias="QA-MRR")
    memory_recall_at_1: float = Field(
        ge=0.0,
        le=1.0,
        serialization_alias="Memory-Recall@1",
    )
    memory_recall_at_3: float = Field(
        ge=0.0,
        le=1.0,
        serialization_alias="Memory-Recall@3",
    )
    memory_recall_at_5: float = Field(
        ge=0.0,
        le=1.0,
        serialization_alias="Memory-Recall@5",
    )
    diagnostics: MetricDiagnostics


def evaluate_prediction_files(
    prediction_path: Path,
    label_path: Path,
) -> EvaluationMetrics:
    labels = _read_jsonl_models(label_path, QALabelExample, "labels")
    predictions = _read_jsonl_models(prediction_path, PredictionRecord, "predictions")
    return evaluate_predictions(labels, predictions)


def evaluate_predictions(
    labels: Sequence[QALabelExample],
    predictions: Sequence[PredictionRecord],
) -> EvaluationMetrics:
    rows = _scored_predictions(labels, predictions)
    return EvaluationMetrics(
        ans_f1=PERCENT_SCALE * _answer_f1(rows),
        qa_acc=PERCENT_SCALE * _qa_accuracy(rows),
        qa_mrr=PERCENT_SCALE * _qa_mrr(rows),
        memory_recall_at_1=_memory_recall(rows, 1),
        memory_recall_at_3=_memory_recall(rows, 3),
        memory_recall_at_5=_memory_recall(rows, 5),
        diagnostics=MetricDiagnostics(
            causal_violation_count=_causal_violation_count(rows),
            prompt_tokens=_summary(
                tuple(row.prediction.prompt_token_count for row in rows),
            ),
            memory_size=_summary(
                tuple(_retrieved_memory_count(row.prediction) for row in rows),
            ),
        ),
    )


def write_metrics(metrics: EvaluationMetrics, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(
        f"{metrics.model_dump_json(by_alias=True)}\n",
        encoding="utf-8",
    )


def metrics_stdout(metrics: EvaluationMetrics, output: Path) -> str:
    return (
        f"wrote {output}\n"
        f"Ans-F1={metrics.ans_f1:.6f}\n"
        f"QA-Acc={metrics.qa_acc:.6f}\n"
        f"QA-MRR={metrics.qa_mrr:.6f}\n"
        f"Memory-Recall@1={metrics.memory_recall_at_1:.6f}\n"
        f"Memory-Recall@3={metrics.memory_recall_at_3:.6f}\n"
        f"Memory-Recall@5={metrics.memory_recall_at_5:.6f}\n"
        "diagnostics="
        f"causal_violation_count:{metrics.diagnostics.causal_violation_count},"
        f"prompt_tokens_total:{metrics.diagnostics.prompt_tokens.total},"
        f"memory_size_total:{metrics.diagnostics.memory_size.total}\n"
    )


def _read_jsonl_models[ModelT: BaseModel](
    path: Path,
    model: type[ModelT],
    source_name: str,
) -> tuple[ModelT, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InvalidPredictionError(question_id=source_name, detail=str(exc)) from exc

    records: list[ModelT] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except ValidationError as exc:
            detail = f"line {line_number}: {exc}"
            raise InvalidPredictionError(source_name, detail) from exc
    return tuple(records)


def _scored_predictions(
    labels: Sequence[QALabelExample],
    predictions: Sequence[PredictionRecord],
) -> tuple[ScoredPrediction, ...]:
    if not labels:
        raise InvalidPredictionError(
            EMPTY_EVALUATION_SOURCE,
            EMPTY_EVALUATION_DETAIL,
        )
    labels_by_id = _unique_labels(labels)
    predictions_by_id = _unique_predictions(predictions)
    _require_matching_questions(labels_by_id, predictions_by_id)
    rows: list[ScoredPrediction] = []
    for label in labels:
        prediction = predictions_by_id[label.question_id]
        _require_valid_choices(label, prediction)
        rows.append(
            ScoredPrediction(label, prediction, _correct_rank(label, prediction)),
        )
    return tuple(rows)


def _unique_labels(labels: Sequence[QALabelExample]) -> dict[str, QALabelExample]:
    result: dict[str, QALabelExample] = {}
    for label in labels:
        if label.question_id in result:
            raise InvalidPredictionError(
                label.question_id,
                "duplicate label question_id",
            )
        result[label.question_id] = label
    return result


def _unique_predictions(
    predictions: Sequence[PredictionRecord],
) -> dict[str, PredictionRecord]:
    result: dict[str, PredictionRecord] = {}
    for prediction in predictions:
        if prediction.question_id in result:
            raise InvalidPredictionError(
                prediction.question_id,
                "duplicate prediction question_id",
            )
        result[prediction.question_id] = prediction
    return result


def _require_matching_questions(
    labels_by_id: dict[str, QALabelExample],
    predictions_by_id: dict[str, PredictionRecord],
) -> None:
    missing = set(labels_by_id) - set(predictions_by_id)
    extra = set(predictions_by_id) - set(labels_by_id)
    if missing:
        raise InvalidPredictionError(sorted(missing)[0], "missing prediction")
    if extra:
        raise InvalidPredictionError(sorted(extra)[0], "prediction has no label")


def _require_valid_choices(
    label: QALabelExample,
    prediction: PredictionRecord,
) -> None:
    seen: set[str] = set()
    choice_ids = tuple(choice.choice_id for choice in label.answer_choices)
    if len(choice_ids) != ANSWER_CHOICE_COUNT:
        raise InvalidPredictionError(
            prediction.question_id,
            "label must contain exactly four answer choices",
        )
    if len(choice_ids) != len(set(choice_ids)):
        raise InvalidPredictionError(
            prediction.question_id,
            "label contains duplicate answer choice IDs",
        )
    if label.answer not in choice_ids:
        raise InvalidPredictionError(
            prediction.question_id,
            "label answer is not a choice ID",
        )
    unanswerable_ids = tuple(
        choice.choice_id
        for choice in label.answer_choices
        if is_unanswerable_choice(choice)
    )
    if len(unanswerable_ids) != 1:
        raise InvalidPredictionError(
            prediction.question_id,
            "label must contain exactly one unanswerable choice",
        )
    unanswerable_id = unanswerable_ids[0]
    if label.is_answerable == (label.answer == unanswerable_id):
        raise InvalidPredictionError(
            prediction.question_id,
            "label answerability disagrees with its gold choice",
        )
    for choice_id in prediction.ranked_choices:
        if choice_id in seen:
            raise InvalidPredictionError(
                prediction.question_id,
                f"duplicate ranked choice: {choice_id}",
            )
        if choice_id not in choice_ids:
            raise InvalidPredictionError(
                prediction.question_id,
                f"unknown ranked choice: {choice_id}",
            )
        seen.add(choice_id)
    if set(prediction.ranked_choices) != set(choice_ids):
        missing = sorted(set(choice_ids) - set(prediction.ranked_choices))
        raise InvalidPredictionError(
            prediction.question_id,
            f"ranked choices must contain every answer choice; missing={missing}",
        )
    if prediction.answerable == (prediction.ranked_choices[0] == unanswerable_id):
        raise InvalidPredictionError(
            prediction.question_id,
            "prediction answerability disagrees with its top-ranked choice",
        )


def _correct_rank(label: QALabelExample, prediction: PredictionRecord) -> int | None:
    for index, choice_id in enumerate(prediction.ranked_choices, start=1):
        if choice_id == label.answer:
            return index
    return None


def _answer_f1(rows: Sequence[ScoredPrediction]) -> float:
    true_positive = sum(
        1 for row in rows if row.prediction.answerable and row.label.is_answerable
    )
    false_positive = sum(
        1 for row in rows if row.prediction.answerable and not row.label.is_answerable
    )
    false_negative = sum(
        1 for row in rows if not row.prediction.answerable and row.label.is_answerable
    )
    denominator = (2 * true_positive) + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return (2 * true_positive) / denominator


def _qa_accuracy(rows: Sequence[ScoredPrediction]) -> float:
    if not rows:
        return 0.0
    correct = sum(
        1 for row in rows if row.prediction.ranked_choices[0] == row.label.answer
    )
    return correct / len(rows)


def _qa_mrr(rows: Sequence[ScoredPrediction]) -> float:
    if not rows:
        return 0.0
    reciprocal_rank = sum(
        1.0 / row.correct_rank for row in rows if row.correct_rank is not None
    )
    return reciprocal_rank / len(rows)


def _memory_recall(rows: Sequence[ScoredPrediction], k: int) -> float:
    recall_values: list[float] = []
    for row in rows:
        if not row.label.evidence_list:
            continue
        expected = tuple(
            _label_evidence_span(row.label, raw_span)
            for raw_span in row.label.evidence_list
        )
        retrieved = _retrieved_spans(row.prediction, k)
        hits = sum(
            any(
                _spans_overlap(expected_span, retrieved_span)
                for retrieved_span in retrieved
            )
            for expected_span in expected
        )
        recall_values.append(hits / len(expected))
    if not recall_values:
        return 0.0
    return sum(recall_values) / len(recall_values)


def _causal_violation_count(rows: Sequence[ScoredPrediction]) -> int:
    violations = 0
    for row in rows:
        for _video_id, _start_time, end_time in _retrieved_spans(row.prediction):
            if end_time > row.label.question_time:
                violations += 1
    return violations


def _retrieved_spans(
    prediction: PredictionRecord,
    k: int | None = None,
) -> tuple[tuple[str, float, float], ...]:
    limit = (
        len(prediction.retrieved_evidence or prediction.supporting_memory_ids)
        if k is None
        else k
    )
    if prediction.retrieved_evidence:
        return tuple(
            (item.video_id, item.start_time, item.end_time)
            for item in prediction.retrieved_evidence[:limit]
        )
    if prediction.supporting_evidence:
        return tuple(
            (item.video_id, item.start_time, item.end_time)
            for item in prediction.supporting_evidence[:limit]
        )
    return tuple(
        span
        for memory_id in prediction.supporting_memory_ids[:limit]
        if (span := _parse_legacy_span(memory_id)) is not None
    )


def _retrieved_memory_count(prediction: PredictionRecord) -> int:
    if prediction.retrieved_evidence:
        return len(prediction.retrieved_evidence)
    return len(prediction.supporting_memory_ids)


def _label_evidence_span(
    label: QALabelExample,
    raw_span: str,
) -> tuple[str, float, float]:
    span = _parse_legacy_span(raw_span)
    if span is None:
        raise InvalidPredictionError(
            label.question_id,
            f"invalid label evidence span: {raw_span}",
        )
    return span


def _parse_legacy_span(raw_span: str) -> tuple[str, float, float] | None:
    parts = raw_span.split(":")
    if len(parts) != EVIDENCE_SPAN_PARTS:
        return None
    video_id, raw_start, raw_end, store = parts
    if not video_id or not store:
        return None
    try:
        start_time = float(raw_start)
        end_time = float(raw_end)
    except ValueError:
        return None
    if not isfinite(start_time) or not isfinite(end_time) or end_time <= start_time:
        return None
    return video_id, start_time, end_time


def _spans_overlap(
    left: tuple[str, float, float],
    right: tuple[str, float, float],
) -> bool:
    left_video, left_start, left_end = left
    right_video, right_start, right_end = right
    if left_video != right_video:
        return False
    if left_start == left_end:
        return right_start <= left_start <= right_end
    if right_start == right_end:
        return left_start <= right_start <= left_end
    return left_start < right_end and right_start < left_end


def _summary(values: Sequence[int]) -> SummaryStats:
    if not values:
        return SummaryStats(count=0, total=0, min=0, max=0, mean=0.0)
    total = sum(values)
    return SummaryStats(
        count=len(values),
        total=total,
        min=min(values),
        max=max(values),
        mean=total / len(values),
    )
