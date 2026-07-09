from __future__ import annotations

import pytest
from pydantic import ValidationError

from worldmm_smvqa.schema import (
    PROHIBITED_MEMORY_FIELDS,
    AnswerChoice,
    LeakageError,
    QALabelExample,
    SourceStreamExample,
    StreamChunk,
    ensure_memory_builder_input,
)


def test_source_stream_excludes_labels() -> None:
    # Given: source-stream data with evaluator-only answer labels mixed in.
    payload = {
        "video_id": "video-1",
        "start_time": 0.0,
        "end_time": 30.0,
        "transcript": "left mug on desk",
        "answer": "desk",
        "is_answerable": True,
    }

    # When / Then: boundary parsing rejects the leaked label fields.
    with pytest.raises(ValidationError):
        _ = SourceStreamExample.model_validate(payload)


def test_prohibited_memory_fields_are_explicit() -> None:
    # Given / When / Then: the anti-leakage contract names every forbidden label.
    assert PROHIBITED_MEMORY_FIELDS == (
        "answer",
        "answer_choices.choice_ltype",
        "is_answerable",
        "evidence_list",
        "verification_score",
    )


def test_memory_builder_accepts_source_examples_and_chunks() -> None:
    # Given: allowed memory-builder inputs.
    source = SourceStreamExample(
        video_id="video-1",
        start_time=0.0,
        end_time=60.0,
        transcript="moved the keys",
    )
    chunk = StreamChunk(
        chunk_id="video-1:0:30:clip_30s",
        video_id="video-1",
        start_time=0.0,
        end_time=30.0,
        granularity="clip_30s",
        transcript="moved the keys",
    )

    # When / Then: the guard returns the same typed values for builder use.
    assert ensure_memory_builder_input(source) is source
    assert ensure_memory_builder_input(chunk) is chunk


def test_memory_builder_rejects_qa_label_example() -> None:
    # Given: evaluator-only QA labels.
    label = QALabelExample(
        question_id="q-1",
        video_id="video-1",
        question="Where are the keys?",
        question_time=120.0,
        answer_choices=(
            AnswerChoice(choice_id="A", text="desk", choice_ltype="location"),
            AnswerChoice(choice_id="B", text="sink", choice_ltype="location"),
        ),
        answer="A",
        is_answerable=True,
        evidence_list=("video-1:0:30:clip_30s",),
        verification_score=1.0,
    )

    # When / Then: memory-builder boundary rejects labels with the typed error.
    with pytest.raises(LeakageError, match="QALabelExample"):
        _ = ensure_memory_builder_input(label)
