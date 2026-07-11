from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from worldmm_smvqa.fixtures import read_fixture_questions
from worldmm_smvqa.qa import QAParseError, parse_qa_output
from worldmm_smvqa.qa_prompt import (
    SPATIAL_EVIDENCE_PLACEHOLDER,
    build_qa_prompt,
)
from worldmm_smvqa.qa_transformers import (
    TransformersCliUsageError,
    validate_external_evidence_packs,
)
from worldmm_smvqa.retrieval_types import EvidenceItem, EvidencePack
from worldmm_smvqa.worldmm.geometry_executor import (
    GeometryQuery,
    execute_geometry,
)

if TYPE_CHECKING:
    from worldmm_smvqa.schema import QuestionRequest

FIXTURE = Path("tests/fixtures/tiny_smvqa")


def test_external_pack_set_rejects_unknown_duplicate_and_missing_questions() -> None:
    questions = read_fixture_questions(FIXTURE)[:2]
    by_id = {question.question_id: question for question in questions}
    first = _pack(questions[0])
    second = _pack(questions[1])

    with pytest.raises(TransformersCliUsageError, match="duplicate evidence pack"):
        validate_external_evidence_packs((first, first, second), by_id)

    unknown = first.model_copy(update={"question_id": "unknown-question"})
    with pytest.raises(
        TransformersCliUsageError,
        match="unknown evidence pack question",
    ):
        validate_external_evidence_packs((unknown, second), by_id)

    with pytest.raises(TransformersCliUsageError, match="missing evidence pack"):
        validate_external_evidence_packs((first,), by_id)


@pytest.mark.parametrize(
    ("pack_update", "item_update", "message"),
    [
        ({"video_id": "off-scope-video"}, {}, "pack video_id.*outside"),
        ({}, {"video_id": "off-scope-video"}, "evidence video_id.*outside"),
        ({}, {"start_time": 3.0, "end_time": 2.0}, "start_time exceeds"),
        ({}, {"end_time": 46.0}, "ends after question_time"),
    ],
)
def test_external_pack_rejects_scope_and_causal_time_violations(
    pack_update: dict[str, object],
    item_update: dict[str, object],
    message: str,
) -> None:
    question = read_fixture_questions(FIXTURE)[0]
    pack = _pack(question)
    if item_update:
        pack = pack.model_copy(
            update={
                "evidence": (pack.evidence[0].model_copy(update=item_update),),
            },
        )
    if pack_update:
        pack = pack.model_copy(update=pack_update)

    with pytest.raises(TransformersCliUsageError, match=message):
        validate_external_evidence_packs(
            (pack,),
            {question.question_id: question},
        )


def test_geometry_answer_requires_answerable_executor_proof_citation() -> None:
    question = read_fixture_questions(FIXTURE)[4]
    proof = execute_geometry(
        (
            {
                "entity_id": "opaque-entity-1",
                "label": "mug",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "coordinate_frame": "source_world",
                "uncertainty_m": 0.0,
                "provenance": "object_geometry",
                "evidence_refs": ["opaque-spatial-1"],
            },
            {
                "entity_id": "opaque-entity-2",
                "label": "notebook",
                "x": 0.5,
                "y": 0.0,
                "z": 0.0,
                "coordinate_frame": "source_world",
                "uncertainty_m": 0.0,
                "provenance": "object_geometry",
                "evidence_refs": ["opaque-spatial-2"],
            },
        ),
        GeometryQuery(
            operation="distance",
            coordinate_frame="source_world",
            subject="opaque-entity-1",
            object="opaque-entity-2",
        ),
    )
    without_citation = (
        '{"answerable":true,"ranked_choices":["B","C","A","D"],'
        '"answer":"B","confidence":0.9,"supporting_memory_ids":[],'
        '"geometry_proof_ids":[]}'
    )

    with pytest.raises(QAParseError, match="requires a geometry proof ID"):
        _ = parse_qa_output(
            question=question,
            raw_outputs=(without_citation,),
            prompt_token_count=10,
            raw_model_output_path=None,
            geometry_proofs=(proof,),
        )

    with_citation = without_citation.replace(
        '"geometry_proof_ids":[]',
        f'"geometry_proof_ids":["{proof.proof_id}"]',
    )
    prediction = parse_qa_output(
        question=question,
        raw_outputs=(with_citation,),
        prompt_token_count=10,
        raw_model_output_path=None,
        geometry_proofs=(proof,),
    )
    assert prediction.geometry_proof_ids == (proof.proof_id,)

    contradictory = with_citation.replace(
        '"ranked_choices":["B","C","A","D"],"answer":"B"',
        '"ranked_choices":["A","B","C","D"],"answer":"A"',
    )
    with pytest.raises(QAParseError, match="contradicts cited geometry proof"):
        _ = parse_qa_output(
            question=question,
            raw_outputs=(contradictory,),
            prompt_token_count=10,
            raw_model_output_path=None,
            geometry_proofs=(proof,),
        )


def test_prompt_withholds_spatial_payload_but_keeps_non_spatial_snippet() -> None:
    question = read_fixture_questions(FIXTURE)[0]
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("spatial", "semantic"),
        selected_stores=("spatial", "semantic"),
        evidence_budget=2,
        evidence=(
            EvidenceItem(
                memory_id="opaque-spatial-id",
                video_id=question.video_id,
                snippet="subject left_of object at x=9.875 distance_m=7.25",
                frame_refs=(),
                source_store="spatial",
                start_time=1.0,
                end_time=2.0,
                retrieval_score=1.0,
                geometry={"relation": "left_of", "distance_m": 7.25, "x": 9.875},
            ),
            EvidenceItem(
                memory_id="semantic-id",
                video_id=question.video_id,
                snippet="the lamp was switched on",
                frame_refs=(),
                source_store="semantic",
                start_time=1.0,
                end_time=2.0,
                retrieval_score=0.9,
            ),
        ),
        causal_filtered_count=0,
    )

    prompt = build_qa_prompt(question, pack)

    assert "opaque-spatial-id" in prompt
    assert SPATIAL_EVIDENCE_PLACEHOLDER in prompt
    assert "left_of" not in prompt
    assert "9.875" not in prompt
    assert "7.25" not in prompt
    assert "the lamp was switched on" in prompt


def test_support_ids_without_trusted_pack_are_rejected() -> None:
    question = read_fixture_questions(FIXTURE)[0]
    raw = (
        '{"answerable":true,"ranked_choices":["A","B","C","D"],'
        '"answer":"A","confidence":0.9,'
        '"supporting_memory_ids":["fabricated"]}'
    )

    with pytest.raises(QAParseError, match="require a trusted evidence pack"):
        _ = parse_qa_output(
            question=question,
            raw_outputs=(raw,),
            prompt_token_count=10,
            raw_model_output_path=None,
        )


def _pack(question: QuestionRequest) -> EvidencePack:
    question_id = question.question_id
    video_id = question.video_id
    return EvidencePack(
        question_id=question_id,
        video_id=video_id,
        requested_stores=("semantic",),
        selected_stores=("semantic",),
        evidence_budget=1,
        evidence=(
            EvidenceItem(
                memory_id=f"memory-{question_id}",
                video_id=video_id,
                snippet="safe semantic evidence",
                frame_refs=(),
                source_store="semantic",
                start_time=1.0,
                end_time=2.0,
                retrieval_score=1.0,
            ),
        ),
        causal_filtered_count=0,
    )
