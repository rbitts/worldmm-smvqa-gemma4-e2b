from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import override

from pydantic import BaseModel, ValidationError

from worldmm_smvqa.fixture_data import tiny_fixture_examples
from worldmm_smvqa.schema import (
    QALabelExample,
    QuestionRequest,
    SourceStreamExample,
)

__all__ = [
    "FixtureCounts",
    "FixtureValidationError",
    "read_fixture_questions",
    "tiny_fixture_examples",
    "validate_fixture",
    "write_tiny_fixture",
]


@dataclass(frozen=True, slots=True)
class FixtureCounts:
    source_examples: int
    qa_examples: int


@dataclass(frozen=True, slots=True)
class FixtureValidationError(Exception):
    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"FixtureValidationError: {self.path}: {self.detail}"


def write_tiny_fixture(output_dir: Path) -> FixtureCounts:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources, labels = tiny_fixture_examples()
    _write_jsonl(output_dir / "sources.jsonl", sources)
    _write_jsonl(output_dir / "questions.jsonl", _questions(labels))
    _write_jsonl(output_dir / "labels.jsonl", labels)
    return FixtureCounts(source_examples=len(sources), qa_examples=len(labels))


def validate_fixture(input_dir: Path) -> FixtureCounts:
    sources = _read_jsonl_models(input_dir / "sources.jsonl", SourceStreamExample)
    questions = _read_jsonl_models(
        input_dir / "questions.jsonl",
        QuestionRequest,
    )
    labels = _read_jsonl_models(input_dir / "labels.jsonl", QALabelExample)
    for path, records in (
        (input_dir / "sources.jsonl", sources),
        (input_dir / "questions.jsonl", questions),
        (input_dir / "labels.jsonl", labels),
    ):
        if not records:
            raise FixtureValidationError(path=path, detail="file has no records")
    _require_unique_question_ids(
        questions,
        input_dir / "questions.jsonl",
    )
    _require_unique_question_ids(labels, input_dir / "labels.jsonl")
    _require_matching_questions(questions, labels, input_dir)
    _require_known_videos(sources, questions, input_dir / "questions.jsonl")
    _require_valid_label_answers(labels, input_dir / "labels.jsonl")
    return FixtureCounts(source_examples=len(sources), qa_examples=len(questions))


def read_fixture_questions(input_dir: Path) -> tuple[QuestionRequest, ...]:
    return _read_jsonl_models(input_dir / "questions.jsonl", QuestionRequest)


def _questions(labels: tuple[QALabelExample, ...]) -> tuple[QuestionRequest, ...]:
    return tuple(
        QuestionRequest(
            question_id=label.question_id,
            video_id=label.video_id,
            video_ids=label.video_ids,
            question=label.question,
            question_time=label.question_time,
            answer_choices=label.answer_choices,
        )
        for label in labels
    )


def _write_jsonl(
    path: Path,
    records: (
        tuple[SourceStreamExample, ...]
        | tuple[QuestionRequest, ...]
        | tuple[QALabelExample, ...]
    ),
) -> None:
    _ = path.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl_models[ModelT: BaseModel](
    path: Path,
    model: type[ModelT],
) -> tuple[ModelT, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FixtureValidationError(path=path, detail=str(exc)) from exc

    records: list[ModelT] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except ValidationError as exc:
            detail = f"line {line_number}: {exc}"
            raise FixtureValidationError(path=path, detail=detail) from exc
    return tuple(records)


def _require_unique_question_ids(
    records: tuple[QuestionRequest, ...] | tuple[QALabelExample, ...],
    path: Path,
) -> None:
    seen: set[str] = set()
    for record in records:
        if record.question_id in seen:
            raise FixtureValidationError(
                path=path,
                detail=f"duplicate question_id: {record.question_id}",
            )
        seen.add(record.question_id)


def _require_matching_questions(
    questions: tuple[QuestionRequest, ...],
    labels: tuple[QALabelExample, ...],
    input_dir: Path,
) -> None:
    expected = {question.question_id: question for question in _questions(labels)}
    actual = {question.question_id: question for question in questions}
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        detail = f"question/label IDs differ; missing={missing} extra={extra}"
        raise FixtureValidationError(
            path=input_dir / "questions.jsonl",
            detail=detail,
        )
    for question_id, question in actual.items():
        if question != expected[question_id]:
            raise FixtureValidationError(
                path=input_dir / "questions.jsonl",
                detail=f"question fields differ from label: {question_id}",
            )


def _require_known_videos(
    sources: tuple[SourceStreamExample, ...],
    questions: tuple[QuestionRequest, ...],
    path: Path,
) -> None:
    source_video_ids = {source.video_id for source in sources}
    for question in questions:
        video_ids = question.video_ids or (question.video_id,)
        missing = sorted(set(video_ids) - source_video_ids)
        if missing:
            raise FixtureValidationError(
                path=path,
                detail=(
                    f"{question.question_id}: unknown video_id(s): "
                    f"{', '.join(missing)}"
                ),
            )
        if question.video_id not in video_ids:
            raise FixtureValidationError(
                path=path,
                detail=(
                    f"{question.question_id}: primary video_id absent from video_ids"
                ),
            )


def _require_valid_label_answers(
    labels: tuple[QALabelExample, ...],
    path: Path,
) -> None:
    for label in labels:
        choice_ids = tuple(choice.choice_id for choice in label.answer_choices)
        if len(choice_ids) != len(set(choice_ids)):
            raise FixtureValidationError(
                path=path,
                detail=f"{label.question_id}: duplicate answer choice ID",
            )
        if label.is_answerable and label.answer not in choice_ids:
            raise FixtureValidationError(
                path=path,
                detail=f"{label.question_id}: answer is not a choice ID",
            )
        if not label.is_answerable and label.answer:
            raise FixtureValidationError(
                path=path,
                detail=f"{label.question_id}: unanswerable row must use empty answer",
            )
