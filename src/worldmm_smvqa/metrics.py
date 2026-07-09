from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, override

from pydantic import BaseModel, Field, ValidationError

from worldmm_smvqa.schema import FrozenModel, PredictionRecord, QALabelExample

MEMORY_ID_MIN_PARTS: Final = 3


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
    ans_f1: float = Field(serialization_alias="Ans-F1")
    qa_acc: float = Field(serialization_alias="QA-Acc")
    qa_mrr: float = Field(serialization_alias="QA-MRR")
    memory_recall_at_1: float = Field(serialization_alias="Memory-Recall@1")
    memory_recall_at_3: float = Field(serialization_alias="Memory-Recall@3")
    memory_recall_at_5: float = Field(serialization_alias="Memory-Recall@5")
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
    answerable_rows = tuple(row for row in rows if row.label.is_answerable)
    return EvaluationMetrics(
        ans_f1=_answer_f1(rows),
        qa_acc=_qa_accuracy(answerable_rows),
        qa_mrr=_qa_mrr(answerable_rows),
        memory_recall_at_1=_memory_recall(rows, 1),
        memory_recall_at_3=_memory_recall(rows, 3),
        memory_recall_at_5=_memory_recall(rows, 5),
        diagnostics=MetricDiagnostics(
            causal_violation_count=_causal_violation_count(rows),
            prompt_tokens=_summary(
                tuple(row.prediction.prompt_token_count for row in rows),
            ),
            memory_size=_summary(
                tuple(len(row.prediction.supporting_memory_ids) for row in rows),
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
    choice_ids = {choice.choice_id for choice in label.answer_choices}
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
    if not prediction.ranked_choices:
        raise InvalidPredictionError(prediction.question_id, "ranked choices are empty")


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
        1
        for row in rows
        if row.prediction.answerable
        and row.prediction.ranked_choices[0] == row.label.answer
    )
    return correct / len(rows)


def _qa_mrr(rows: Sequence[ScoredPrediction]) -> float:
    if not rows:
        return 0.0
    reciprocal_rank = sum(
        1.0 / row.correct_rank
        for row in rows
        if row.prediction.answerable and row.correct_rank is not None
    )
    return reciprocal_rank / len(rows)


def _memory_recall(rows: Sequence[ScoredPrediction], k: int) -> float:
    recall_values: list[float] = []
    for row in rows:
        if not row.label.evidence_list:
            continue
        expected = set(row.label.evidence_list)
        retrieved = set(row.prediction.supporting_memory_ids[:k])
        recall_values.append(len(expected & retrieved) / len(expected))
    if not recall_values:
        return 0.0
    return sum(recall_values) / len(recall_values)


def _causal_violation_count(rows: Sequence[ScoredPrediction]) -> int:
    violations = 0
    for row in rows:
        for memory_id in row.prediction.supporting_memory_ids:
            memory_end = _memory_end_time(memory_id)
            if memory_end is not None and memory_end > row.label.question_time:
                violations += 1
    return violations


def _memory_end_time(memory_id: str) -> float | None:
    parts = memory_id.split(":")
    if len(parts) < MEMORY_ID_MIN_PARTS:
        return None
    try:
        return float(parts[2])
    except ValueError:
        return None


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
