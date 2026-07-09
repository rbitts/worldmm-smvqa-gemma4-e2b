from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from worldmm_smvqa.metrics import (
    InvalidPredictionError,
    evaluate_prediction_files,
    evaluate_predictions,
)
from worldmm_smvqa.schema import AnswerChoice, PredictionRecord, QALabelExample

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    _ = env.pop("WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST", None)
    return subprocess.run(
        ["uv", "run", "worldmm-smvqa", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_evaluate_fixture_metrics_when_ranked_answers_vary() -> None:
    # Given: checked-in labels with hand-ranked predictions.
    labels = ROOT / "tests/fixtures/tiny_smvqa/labels.jsonl"
    predictions = ROOT / "tests/fixtures/tiny_smvqa/predictions.good.jsonl"

    # When: metrics are computed from JSONL files.
    metrics = evaluate_prediction_files(predictions, labels)

    # Then: answer F1, top-1 accuracy, MRR, recall, and diagnostics match math.
    assert metrics.ans_f1 == pytest.approx(1.0)
    assert metrics.qa_acc == pytest.approx(0.5)
    assert metrics.qa_mrr == pytest.approx(7.0 / 12.0)
    assert metrics.memory_recall_at_1 == pytest.approx(0.5)
    assert metrics.memory_recall_at_3 == pytest.approx(0.75)
    assert metrics.memory_recall_at_5 == pytest.approx(0.75)
    assert metrics.diagnostics.causal_violation_count == 1
    assert metrics.diagnostics.prompt_tokens.total == 100
    assert metrics.diagnostics.memory_size.total == 4


def test_answer_f1_counts_false_positive_when_label_is_unanswerable() -> None:
    # Given: one answerable label and one unanswerable label.
    labels = (
        label("q_answerable", answer="A", is_answerable=True),
        label("q_unanswerable", answer="", is_answerable=False),
    )
    predictions = (
        prediction("q_answerable", ranked_choices=("A", "B")),
        prediction("q_unanswerable", ranked_choices=("A", "B")),
    )

    # When: answerability F1 is computed.
    metrics = evaluate_predictions(labels, predictions)

    # Then: the unanswerable positive prediction is a false positive.
    assert metrics.ans_f1 == pytest.approx(2.0 / 3.0)
    assert metrics.qa_acc == pytest.approx(1.0)


def test_missing_correct_choice_scores_zero_reciprocal_rank() -> None:
    # Given: an answerable label whose correct choice is absent from the ranking.
    labels = (label("q_missing", answer="C", is_answerable=True),)
    predictions = (prediction("q_missing", ranked_choices=("A", "B")),)

    # When: metrics are computed.
    metrics = evaluate_predictions(labels, predictions)

    # Then: top-1 accuracy and reciprocal rank are both zero.
    assert metrics.qa_acc == pytest.approx(0.0)
    assert metrics.qa_mrr == pytest.approx(0.0)


def test_duplicate_ranked_choices_fail_when_prediction_is_malformed() -> None:
    # Given: a prediction that repeats the same ranked choice.
    labels = (label("q_duplicate", answer="A", is_answerable=True),)
    predictions = (prediction("q_duplicate", ranked_choices=("A", "A")),)

    # When / Then: validation rejects the malformed ranked choices.
    with pytest.raises(InvalidPredictionError, match="duplicate ranked choice"):
        _ = evaluate_predictions(labels, predictions)


def test_evaluate_cli_writes_metrics_json(tmp_path: Path) -> None:
    # Given: fixture labels and predictions plus an output file.
    output = tmp_path / "metrics.json"

    # When: evaluate runs through the CLI.
    result = run_cli(
        "evaluate",
        "--pred",
        "tests/fixtures/tiny_smvqa/predictions.good.jsonl",
        "--labels",
        "tests/fixtures/tiny_smvqa/labels.jsonl",
        "--out",
        str(output),
    )

    # Then: it exits successfully and exposes metric names plus diagnostics.
    assert result.returncode == 0
    assert output.is_file()
    assert "Ans-F1" in result.stdout
    assert "QA-Acc" in result.stdout
    assert "QA-MRR" in result.stdout
    assert "diagnostics" in result.stdout
    metrics_text = output.read_text(encoding="utf-8")
    assert '"Ans-F1":1.0' in metrics_text
    assert '"causal_violation_count":1' in metrics_text


def label(
    question_id: str,
    *,
    answer: str,
    is_answerable: bool,
) -> QALabelExample:
    return QALabelExample(
        question_id=question_id,
        video_id="fake_video_001",
        question="Synthetic question?",
        question_time=45.0,
        answer_choices=(
            AnswerChoice(choice_id="A", text="alpha", choice_ltype="synthetic"),
            AnswerChoice(choice_id="B", text="bravo", choice_ltype="synthetic"),
            AnswerChoice(choice_id="C", text="charlie", choice_ltype="synthetic"),
        ),
        answer=answer,
        is_answerable=is_answerable,
        evidence_list=("fake_video_001:5:12:transcript",) if is_answerable else (),
        verification_score=1.0,
    )


def prediction(
    question_id: str,
    ranked_choices: tuple[str, ...],
) -> PredictionRecord:
    return PredictionRecord(
        question_id=question_id,
        answerable=True,
        ranked_choices=ranked_choices,
        answer=ranked_choices[0] if ranked_choices else None,
        confidence=0.9,
        supporting_memory_ids=("fake_video_001:5:12:transcript",),
        prompt_token_count=10,
        raw_model_output_path=None,
    )
