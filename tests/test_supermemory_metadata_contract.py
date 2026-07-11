from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from worldmm_smvqa.schema import (
    AnswerChoice,
    FrameMetadata,
    GazeSample,
    ObjectMetadata,
    PoseSample,
    QALabelExample,
    QuestionRequest,
    SourceStreamExample,
)


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


def test_supermemory_task_skill_and_choice_types_survive_adapter() -> None:
    payload = _supermemory_all_qa_record()

    label = _tentative_qa_label(payload)

    assert label.task == "conversational_memory"
    assert label.skill == "cross_session_recall"
    assert tuple(choice.choice_ltype for choice in label.answer_choices) == tuple(
        _str_list(payload["choice_types"]),
    )


def test_unanswerable_label_keeps_na_choice_id() -> None:
    payload = _supermemory_all_qa_record()
    payload["is_answerable"] = False
    payload["correct_option_index"] = 0
    payload["correct_answer"] = _str_list(payload["choices"])[0]

    label = _tentative_qa_label(payload)

    assert not label.is_answerable
    assert label.answer == "A"
    assert label.answer_choices[0].choice_ltype == "incorrect"


def test_optional_spatial_sensor_contract_round_trips() -> None:
    source = SourceStreamExample(
        video_id="video-1",
        start_time=0.0,
        end_time=1.0,
        object_detections=(
            ObjectMetadata(
                start_time=0.0,
                end_time=1.0,
                label="mug",
                confidence=0.9,
                instance_id="mug-7",
                x=1.0,
                y=2.0,
                z=3.0,
            ),
        ),
        pose_samples=(
            PoseSample(
                timestamp=0.5,
                x=1.0,
                y=2.0,
                z=3.0,
                roll=0.1,
                pitch=0.2,
                yaw=0.3,
                coordinate_frame="slam_world",
                pose_covariance=(0.0,) * 36,
            ),
        ),
    )

    restored = SourceStreamExample.model_validate_json(source.model_dump_json())

    assert restored.object_detections[0].instance_id == "mug-7"
    assert restored.pose_samples[0].coordinate_frame == "slam_world"
    assert restored.pose_samples[0].pose_covariance == (0.0,) * 36


def test_sensor_and_question_numeric_contract_rejects_non_finite_values() -> None:
    with pytest.raises(ValidationError):
        _ = ObjectMetadata(
            start_time=0.0,
            end_time=1.0,
            label="mug",
            confidence=float("nan"),
        )
    with pytest.raises(ValidationError):
        _ = PoseSample(timestamp=float("inf"), x=0.0, y=0.0, z=0.0)
    with pytest.raises(ValidationError):
        _ = GazeSample(timestamp=float("nan"), x=0.0, y=0.0, z=0.0)
    with pytest.raises(ValidationError):
        _ = FrameMetadata(
            frame_ref="frame-1",
            timestamp=float("inf"),
            description="invalid",
        )
    with pytest.raises(ValidationError):
        _ = QuestionRequest(
            question_id="q-1",
            video_id="video-1",
            question="Where?",
            question_time=float("nan"),
            answer_choices=(),
        )


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
            "task": "conversational_memory",
            "skill": "cross_session_recall",
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
        task=str(metadata["task"]),
        skill=str(metadata["skill"]),
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
