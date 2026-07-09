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
    labels = _read_jsonl_models(input_dir / "labels.jsonl", QALabelExample)
    return FixtureCounts(source_examples=len(sources), qa_examples=len(labels))


def read_fixture_questions(input_dir: Path) -> tuple[QuestionRequest, ...]:
    return _read_jsonl_models(input_dir / "questions.jsonl", QuestionRequest)


def _questions(labels: tuple[QALabelExample, ...]) -> tuple[QuestionRequest, ...]:
    return tuple(
        QuestionRequest(
            question_id=label.question_id,
            video_id=label.video_id,
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
