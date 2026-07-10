from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from worldmm_smvqa.schema import AnswerChoice, QALabelExample, SourceStreamExample


def test_supermemory_all_qa_record_is_not_a_source_stream_example() -> None:
    # Given: representative /tmp/supermemory_vqa_meta/all_qa.json metadata.
    payload = _supermemory_all_qa_record()

    # When / Then: QA metadata must stay out of source-stream memory input.
    with pytest.raises(ValidationError):
        _ = SourceStreamExample.model_validate(payload)


def test_supermemory_qa_label_contract_uses_answer_choice_ids() -> None:
    # Given: SuperMemory choices name answers by text and index.
    payload = _supermemory_all_qa_record()

    # When: a tentative QA label is built for existing A-D model contracts.
    label = _tentative_qa_label(payload)

    # Then: answer uses the A-D choice ID, not the raw answer text.
    assert tuple(choice.choice_id for choice in label.answer_choices) == (
        "A",
        "B",
        "C",
        "D",
    )
    assert label.answer == "B"
    assert label.answer in {choice.choice_id for choice in label.answer_choices}
    assert label.answer != payload["correct_answer"]


def test_supermemory_qa_metadata_preserves_multiple_video_ids() -> None:
    # Given: SuperMemory QA can point at a pool of videos, plus one primary video.
    payload = _supermemory_all_qa_record()

    # When: adapter-side metadata is carried alongside existing single-video label.
    video_ids = tuple(_str_list(payload["video_ids"]))
    label = _tentative_qa_label(payload)

    # Then: multi-video scope remains available and is not collapsed to label.video_id.
    assert len(video_ids) > 1
    assert label.video_id == str(_dict(payload["metadata"])["primary_video_id"])
    assert label.video_id in video_ids
    assert label.video_ids == video_ids


def _supermemory_all_qa_record() -> dict[str, object]:
    return {
        "question_id": 1,
        "question": "I'm thinking about cooking meat. What did B use his pot for?",
        "choices": [
            "This question can not be answered.",
            "He said he uses it to cook beef.",
            "He said he uses it to cook chicken.",
            "He said he uses it to cook meat.",
        ],
        "correct_answer": "He said he uses it to cook beef.",
        "correct_option_index": 1,
        "choice_types": ["incorrect", "correct", "incorrect", "vague"],
        "subject": 1,
        "metadata": {
            "skill": "conversational_memory",
            "primary_video_id": "Person_1_session_8_03102026_glasses_1264",
            "primary_video_start_time": 1773180268,
        },
        "video_ids": [
            "Person_1_session_16_03292026_glasses_1283",
            "Person_1_session_8_03102026_glasses_1264",
        ],
        "start_time": 1773180268,
        "question_evidence": {
            "video_id": "Person_1_session_8_03102026_glasses_1264",
            "time_spans": [
                {
                    "start_time": 600.0,
                    "end_time": 636.0,
                    "video_id": "Person_1_session_8_03102026_glasses_1264",
                },
            ],
        },
        "is_answerable": True,
        "answer_evidence": {
            "evidence_list": [
                {
                    "video_id": "Person_1_session_6_01312026_glasses_1275",
                    "time_span": {"start_time": 600.0, "end_time": 655.0},
                },
            ],
            "is_answerable": True,
            "text": "He said he uses it to cook beef.",
        },
    }


def _tentative_qa_label(payload: dict[str, object]) -> QALabelExample:
    choices = _str_list(payload["choices"])
    choice_types = _str_list(payload["choice_types"])
    answer_index = _int(payload["correct_option_index"])
    metadata = _dict(payload["metadata"])

    return QALabelExample(
        question_id=str(payload["question_id"]),
        video_id=str(metadata["primary_video_id"]),
        video_ids=tuple(_str_list(payload["video_ids"])),
        question=str(payload["question"]),
        question_time=float(_int(payload["start_time"])),
        answer_choices=tuple(
            AnswerChoice(
                choice_id=chr(ord("A") + index),
                text=choice,
                choice_ltype=choice_types[index],
            )
            for index, choice in enumerate(choices)
        ),
        answer=chr(ord("A") + answer_index),
        is_answerable=bool(payload["is_answerable"]),
        evidence_list=_evidence_refs(payload),
        verification_score=1.0,
    )


def _evidence_refs(payload: dict[str, object]) -> tuple[str, ...]:
    evidence = _dict(payload["answer_evidence"])
    refs: list[str] = []
    for item in _dict_list(evidence["evidence_list"]):
        time_span = _dict(item["time_span"])
        video_id = str(item["video_id"])
        start_time = _float(time_span["start_time"])
        end_time = _float(time_span["end_time"])
        refs.append(f"{video_id}:{start_time}:{end_time}:supermemory")
    return tuple(refs)


def _dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _dict_list(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    items = cast("list[object]", value)
    assert all(isinstance(item, dict) for item in items)
    return cast("list[dict[str, object]]", value)


def _str_list(value: object) -> list[str]:
    assert isinstance(value, list)
    items = cast("list[object]", value)
    assert all(isinstance(item, str) for item in items)
    return cast("list[str]", value)


def _int(value: object) -> int:
    assert isinstance(value, int)
    return value


def _float(value: object) -> float:
    assert isinstance(value, int | float)
    return float(value)
