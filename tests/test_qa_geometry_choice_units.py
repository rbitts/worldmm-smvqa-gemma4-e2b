from __future__ import annotations

import json
from typing import Literal

import pytest

from worldmm_smvqa.qa import QAParseError, parse_qa_output
from worldmm_smvqa.schema import AnswerChoice, QuestionRequest
from worldmm_smvqa.worldmm.geometry_executor import GeometryProof

Operation = Literal["distance", "last_seen", "last_location", "count"]
Unit = Literal["meters", "seconds", "count"]


def _parse(
    operation: Operation,
    unit: Unit | None,
    value: float | str,
    choices: tuple[str, str, str],
    answer: str,
) -> None:
    question = QuestionRequest(
        question_id=f"q-{operation}",
        video_id="video-1",
        question="Geometry question?",
        question_time=10.0,
        answer_choices=(
            *(
                AnswerChoice(choice_id=choice_id, text=text, choice_ltype="answer")
                for choice_id, text in zip("ABC", choices, strict=True)
            ),
            AnswerChoice(
                choice_id="D",
                text="This question cannot be answered.",
                choice_ltype="unanswerable",
            ),
        ),
    )
    proof = GeometryProof(
        proof_id=f"proof-{operation}",
        answerable=True,
        value=value,
        entity_ids=("entity-1",),
        operation=operation,
        coordinate_frame="source_world",
        uncertainty=0.0,
        uncertainty_unit=unit,
        provenance=("observed",),
        evidence_refs=("frame-1",),
    )
    raw = json.dumps(
        {
            "answerable": True,
            "ranked_choices": [answer, *(item for item in "ABCD" if item != answer)],
            "answer": answer,
            "confidence": 0.9,
            "supporting_memory_ids": [],
            "geometry_proof_ids": [proof.proof_id],
        },
    )
    _ = parse_qa_output(
        question=question,
        raw_outputs=(raw,),
        prompt_token_count=1,
        raw_model_output_path=None,
        geometry_proofs=(proof,),
    )


@pytest.mark.parametrize(
    ("operation", "unit", "value", "choices", "answer"),
    [
        ("distance", "meters", 2.0, ("2 ft", "200 cm", "3 m"), "B"),
        ("last_seen", "seconds", 3_600.0, ("60 minutes", "2 hours", "00:30:00"), "A"),
        ("count", "count", 2, ("one", "two", "3"), "B"),
    ],
)
def test_geometry_choice_matching_normalizes_supported_units(
    operation: Operation,
    unit: Unit,
    value: float,
    choices: tuple[str, str, str],
    answer: str,
) -> None:
    _parse(operation, unit, value, choices, answer)


def test_last_location_proof_matches_one_semantic_place_choice() -> None:
    _parse(
        "last_location",
        None,
        "kitchen counter",
        ("on the kitchen counter", "in the bedroom", "near the door"),
        "A",
    )


def test_last_location_proof_rejects_different_selected_place() -> None:
    with pytest.raises(QAParseError, match="contradicts cited geometry proof"):
        _parse(
            "last_location",
            None,
            "kitchen counter",
            ("on the kitchen counter", "in the bedroom", "near the door"),
            "B",
        )


def test_two_meters_does_not_match_two_feet() -> None:
    with pytest.raises(QAParseError, match="contradicts cited geometry proof"):
        _parse("distance", "meters", 2.0, ("2 ft", "2 m", "3 m"), "A")


@pytest.mark.parametrize(
    ("operation", "unit", "value", "choices"),
    [
        ("distance", "meters", 2.0, ("2 yards", "3 yards", "4 yards")),
        ("distance", "meters", 2.0, ("2 m 3 cm", "3 m", "4 m")),
        ("last_seen", "seconds", 8.0, ("earlier", "later", "eventually")),
        ("count", "count", 2, ("a pair", "several", "many")),
    ],
)
def test_unparseable_or_mixed_geometry_choices_fail_closed(
    operation: Operation,
    unit: Unit,
    value: float,
    choices: tuple[str, str, str],
) -> None:
    with pytest.raises(QAParseError, match="explicit, parseable choice units"):
        _parse(operation, unit, value, choices, "A")


def test_geometry_proof_must_match_exactly_one_choice() -> None:
    with pytest.raises(QAParseError, match="exactly one answer choice"):
        _parse("distance", "meters", 2.0, ("2 m", "200 cm", "3 m"), "A")
