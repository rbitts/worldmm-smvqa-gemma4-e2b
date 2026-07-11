from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from worldmm_smvqa.metrics import (
    InvalidPredictionError,
    evaluate_prediction_files,
    evaluate_predictions,
)
from worldmm_smvqa.schema import (
    AnswerChoice,
    PredictionRecord,
    QALabelExample,
    SupportingEvidence,
)

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
    assert metrics.ans_f1 == pytest.approx(100.0)
    assert metrics.qa_acc == pytest.approx(100.0 * 4.0 / 6.0)
    assert metrics.qa_mrr == pytest.approx(100.0 * 55.0 / 72.0)
    assert metrics.memory_recall_at_1 == pytest.approx(4.0 / 5.0)
    assert metrics.memory_recall_at_3 == pytest.approx(4.0 / 5.0)
    assert metrics.memory_recall_at_5 == pytest.approx(4.0 / 5.0)
    assert metrics.diagnostics.causal_violation_count == 0
    assert metrics.diagnostics.prompt_tokens.total == 210
    assert metrics.diagnostics.memory_size.total == 4


def test_answer_f1_counts_false_positive_when_label_is_unanswerable() -> None:
    # Given: one answerable label and one unanswerable label.
    labels = (
        label("q_answerable", answer="A", is_answerable=True),
        label("q_unanswerable", answer="D", is_answerable=False),
    )
    predictions = (
        prediction("q_answerable", ranked_choices=("A", "B", "C", "D")),
        prediction("q_unanswerable", ranked_choices=("A", "B", "C", "D")),
    )

    # When: answerability F1 is computed.
    metrics = evaluate_predictions(labels, predictions)

    # Then: the unanswerable positive prediction is a false positive.
    assert metrics.ans_f1 == pytest.approx(100.0 * 2.0 / 3.0)
    assert metrics.qa_acc == pytest.approx(50.0)
    assert metrics.qa_mrr == pytest.approx(62.5)


def test_unanswerable_choice_counts_for_accuracy_and_mrr() -> None:
    # Given: correct answerable and N/A rankings with separate binary decisions.
    labels = (
        label("q_answerable", answer="A", is_answerable=True),
        label("q_unanswerable", answer="D", is_answerable=False),
    )
    predictions = (
        prediction("q_answerable", ranked_choices=("A", "B", "C", "D")),
        PredictionRecord(
            question_id="q_unanswerable",
            answerable=False,
            ranked_choices=("D", "A", "B", "C"),
            answer=None,
            confidence=0.9,
            supporting_memory_ids=("fake_video_001:5:12:transcript",),
            prompt_token_count=10,
            raw_model_output_path=None,
        ),
    )

    # When: official four-way metrics are computed over every question.
    metrics = evaluate_predictions(labels, predictions)

    # Then: N/A receives MCQ credit while Ans-F1 remains a separate metric.
    assert metrics.ans_f1 == pytest.approx(100.0)
    assert metrics.qa_acc == pytest.approx(100.0)
    assert metrics.qa_mrr == pytest.approx(100.0)


def test_empty_evaluation_and_inconsistent_abstention_are_rejected() -> None:
    with pytest.raises(InvalidPredictionError, match="evaluation set is empty"):
        _ = evaluate_predictions((), ())

    labels = (label("q_bad_abstention", answer="D", is_answerable=False),)
    predictions = (
        PredictionRecord(
            question_id="q_bad_abstention",
            answerable=False,
            ranked_choices=("A", "D", "B", "C"),
            answer=None,
            confidence=0.9,
            supporting_memory_ids=(),
            prompt_token_count=0,
            raw_model_output_path=None,
        ),
    )
    with pytest.raises(InvalidPredictionError, match="top-ranked choice"):
        _ = evaluate_predictions(labels, predictions)


def test_partial_choice_ranking_is_rejected() -> None:
    # Given: a prediction that omits one of the four answer choices.
    labels = (label("q_missing", answer="C", is_answerable=True),)
    predictions = (prediction("q_missing", ranked_choices=("A", "B", "C")),)

    # When / Then: MRR cannot be computed from an incomplete ranking.
    with pytest.raises(InvalidPredictionError, match="every answer choice"):
        _ = evaluate_predictions(labels, predictions)


def test_duplicate_ranked_choices_fail_when_prediction_is_malformed() -> None:
    # Given: a prediction that repeats the same ranked choice.
    # When / Then: prediction schema rejects it before evaluation.
    with pytest.raises(ValidationError, match="duplicate choice IDs"):
        _ = prediction(
            "q_duplicate",
            ranked_choices=("A", "A", "B", "C"),
        )


@pytest.mark.parametrize(
    "values",
    [
        (True, None, ("A", "B", "C", "D"), 0.5, "requires answer"),
        (True, "B", ("A", "B", "C", "D"), 0.5, "top-ranked"),
        (False, "A", ("A", "B", "C", "D"), 0.5, "null answer"),
        (True, "A", ("A", "B", "C", "D"), -0.1, "greater than"),
        (True, "A", ("A", "B", "C", "D"), 1.1, "less than"),
    ],
)
def test_prediction_contract_rejects_inconsistent_output(
    values: tuple[bool, str | None, tuple[str, ...], float, str],
) -> None:
    answerable, answer, ranking, confidence, message = values
    with pytest.raises(ValidationError, match=message):
        _ = PredictionRecord(
            question_id="q_invalid",
            answerable=answerable,
            ranked_choices=ranking,
            answer=answer,
            confidence=confidence,
            supporting_memory_ids=(),
            prompt_token_count=0,
            raw_model_output_path=None,
        )


def test_prediction_contract_keeps_matching_geometry_proofs() -> None:
    prediction_record = PredictionRecord(
        question_id="q-proof",
        answerable=True,
        ranked_choices=("A", "B", "C", "D"),
        answer="A",
        confidence=0.8,
        supporting_memory_ids=(),
        geometry_proof_ids=("proof-1",),
        geometry_proofs=({"proof_id": "proof-1", "value": 1.5},),
        prompt_token_count=0,
        raw_model_output_path=None,
    )

    assert prediction_record.geometry_proof_ids == ("proof-1",)
    assert prediction_record.geometry_proofs[0]["value"] == 1.5


def test_prediction_contract_rejects_mismatched_geometry_proof() -> None:
    with pytest.raises(ValidationError, match="match geometry_proof_ids"):
        _ = PredictionRecord(
            question_id="q-proof",
            answerable=True,
            ranked_choices=("A", "B", "C", "D"),
            answer="A",
            confidence=0.8,
            supporting_memory_ids=(),
            geometry_proof_ids=("proof-1",),
            geometry_proofs=({"proof_id": "proof-2"},),
            prompt_token_count=0,
            raw_model_output_path=None,
        )


def test_memory_recall_uses_structured_temporal_overlap() -> None:
    # Given: an opaque memory ID whose structured span overlaps label evidence.
    labels = (label("q_overlap", answer="A", is_answerable=True),)
    predictions = (
        prediction(
            "q_overlap",
            ranked_choices=("A", "B", "C", "D"),
            supporting_memory_ids=("semantic:opaque",),
            supporting_evidence=(
                SupportingEvidence(
                    memory_id="semantic:opaque",
                    store="semantic",
                    video_id="fake_video_001",
                    start_time=10.0,
                    end_time=15.0,
                ),
            ),
        ),
    )

    # When: retrieval metrics are computed.
    metrics = evaluate_predictions(labels, predictions)

    # Then: positive temporal overlap counts despite different ID formats.
    assert metrics.memory_recall_at_1 == pytest.approx(1.0)


def test_memory_recall_accepts_visual_point_inside_label_span() -> None:
    # Given: frame evidence represented by one timestamp inside label evidence.
    labels = (label("q_frame", answer="A", is_answerable=True),)
    predictions = (
        prediction(
            "q_frame",
            ranked_choices=("A", "B", "C", "D"),
            supporting_memory_ids=("visual:frame",),
            supporting_evidence=(
                SupportingEvidence(
                    memory_id="visual:frame",
                    store="visual",
                    video_id="fake_video_001",
                    start_time=8.0,
                    end_time=8.0,
                ),
            ),
        ),
    )

    # When: retrieval metrics compare temporal support.
    metrics = evaluate_predictions(labels, predictions)

    # Then: point evidence counts as overlapping [5, 12].
    assert metrics.memory_recall_at_1 == pytest.approx(1.0)


def test_memory_recall_uses_full_retrieval_not_model_support_subset() -> None:
    # Given: model supports one miss, while retrieved rank 2 overlaps evidence.
    labels = (label("q_retrieval", answer="A", is_answerable=True),)
    miss = SupportingEvidence(
        memory_id="miss",
        store="semantic",
        video_id="fake_video_001",
        start_time=20.0,
        end_time=25.0,
    )
    hit = SupportingEvidence(
        memory_id="hit",
        store="episodic",
        video_id="fake_video_001",
        start_time=6.0,
        end_time=10.0,
    )
    predictions = (
        prediction(
            "q_retrieval",
            ranked_choices=("A", "B", "C", "D"),
            supporting_memory_ids=("miss",),
            supporting_evidence=(miss,),
            retrieved_evidence=(miss, hit),
        ),
    )

    # When: retrieval recall is computed.
    metrics = evaluate_predictions(labels, predictions)

    # Then: @1 misses and @3 sees rank-2 retrieval independently of QA support.
    assert metrics.memory_recall_at_1 == 0.0
    assert metrics.memory_recall_at_3 == pytest.approx(1.0)


def test_causal_violations_use_structured_end_time() -> None:
    # Given: an opaque memory ID with structured evidence after question time.
    labels = (label("q_future", answer="A", is_answerable=True),)
    predictions = (
        prediction(
            "q_future",
            ranked_choices=("A", "B", "C", "D"),
            supporting_memory_ids=("opaque",),
            supporting_evidence=(
                SupportingEvidence(
                    memory_id="opaque",
                    store="visual",
                    video_id="fake_video_001",
                    start_time=44.0,
                    end_time=46.0,
                ),
            ),
        ),
    )

    # When: diagnostics are computed.
    metrics = evaluate_predictions(labels, predictions)

    # Then: structured end_time detects the violation.
    assert metrics.diagnostics.causal_violation_count == 1


def test_invalid_legacy_memory_id_is_not_used_as_temporal_fallback() -> None:
    # Given: a legacy prediction without structured metadata and an invalid ID.
    labels = (label("q_invalid_legacy", answer="A", is_answerable=True),)
    predictions = (
        prediction(
            "q_invalid_legacy",
            ranked_choices=("A", "B", "C", "D"),
            supporting_memory_ids=("fake_video_001:5:inf:transcript",),
        ),
    )

    # When: retrieval metrics are computed.
    metrics = evaluate_predictions(labels, predictions)

    # Then: malformed fallback metadata produces no recall or causal violation.
    assert metrics.memory_recall_at_1 == 0.0
    assert metrics.diagnostics.causal_violation_count == 0


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
    assert '"Ans-F1":100.0' in metrics_text
    assert '"causal_violation_count":0' in metrics_text


def test_evaluate_cli_rejects_causal_evidence_violation(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "metrics.json"
    label_row = label("q_future", answer="A", is_answerable=True)
    prediction_row = prediction(
        "q_future",
        ranked_choices=("A", "B", "C", "D"),
        supporting_memory_ids=("future",),
        supporting_evidence=(
            SupportingEvidence(
                memory_id="future",
                store="spatial",
                video_id="fake_video_001",
                start_time=46.0,
                end_time=47.0,
            ),
        ),
    )
    _ = labels.write_text(label_row.model_dump_json() + "\n", encoding="utf-8")
    _ = predictions.write_text(
        prediction_row.model_dump_json() + "\n",
        encoding="utf-8",
    )

    result = run_cli(
        "evaluate",
        "--pred",
        str(predictions),
        "--labels",
        str(labels),
        "--out",
        str(output),
    )

    assert result.returncode != 0
    assert "causal evidence violations: 1" in result.stderr
    assert not output.exists()


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
            AnswerChoice(
                choice_id="D",
                text="This question cannot be answered.",
                choice_ltype="unanswerable",
            ),
        ),
        answer=answer,
        is_answerable=is_answerable,
        evidence_list=("fake_video_001:5:12:transcript",) if is_answerable else (),
        verification_score=1.0,
    )


def prediction(
    question_id: str,
    ranked_choices: tuple[str, ...],
    *,
    supporting_memory_ids: tuple[str, ...] = (
        "fake_video_001:5:12:transcript",
    ),
    supporting_evidence: tuple[SupportingEvidence, ...] = (),
    retrieved_evidence: tuple[SupportingEvidence, ...] = (),
) -> PredictionRecord:
    return PredictionRecord(
        question_id=question_id,
        answerable=True,
        ranked_choices=ranked_choices,
        answer=ranked_choices[0] if ranked_choices else None,
        confidence=0.9,
        supporting_memory_ids=supporting_memory_ids,
        supporting_evidence=supporting_evidence,
        retrieved_evidence=retrieved_evidence,
        prompt_token_count=10,
        raw_model_output_path=None,
    )
