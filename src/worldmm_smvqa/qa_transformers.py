from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, override

from pydantic import ValidationError

from worldmm_smvqa.config import REMOTE_ENV_FLAG, RemoteOnlyError
from worldmm_smvqa.fixtures import read_fixture_questions
from worldmm_smvqa.qa_shards import (
    QAShardError,
    distributed_env,
    merge_shards,
    packs_for_rank,
    rank_output_path,
    wait_for_shards,
)
from worldmm_smvqa.retrieval_types import EvidencePack
from worldmm_smvqa.transformers_backend import TransformersGenerationError

if TYPE_CHECKING:
    from worldmm_smvqa.qa import QABackend
    from worldmm_smvqa.schema import PredictionRecord, QuestionRequest

type TransformersBackendName = Literal["gemma4", "real", "mock"]


@dataclass(frozen=True, slots=True)
class TransformersCliArgs:
    model: str
    fixture: Path
    evidence: Path
    out: Path
    backend: TransformersBackendName


@dataclass(frozen=True, slots=True)
class TransformersCliResult:
    written: Path
    predictions: int


@dataclass(frozen=True, slots=True)
class TransformersCliUsageError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"UsageError: {self.detail}"


def run_transformers_cli(
    args: TransformersCliArgs,
    env: Mapping[str, str],
) -> TransformersCliResult:
    from worldmm_smvqa.qa import (  # noqa: PLC0415
        Gemma4QABackend,
        MockQABackend,
        QABackendUnavailableError,
        QAParseError,
        build_qa_prompt,
        parse_qa_output,
        write_predictions_jsonl,
    )

    questions = _questions_by_id(args.fixture)
    packs = _read_evidence_packs(args.evidence)
    match args.backend:
        case "mock":
            backend: QABackend = MockQABackend()
        case "gemma4" | "real":
            if env.get(REMOTE_ENV_FLAG) != "1":
                raise RemoteOnlyError(action="qa_transformers real model")
            backend = Gemma4QABackend(model_path=args.model)

    distributed = distributed_env(env)
    rank_packs = packs_for_rank(packs, distributed)
    predictions: list[PredictionRecord] = []
    for pack in rank_packs:
        question = _question_for_pack(pack, questions)
        prompt = build_qa_prompt(question, pack)
        try:
            raw_outputs = backend.raw_outputs(prompt, question, pack)
            predictions.append(
                parse_qa_output(
                    question=question,
                    raw_outputs=raw_outputs,
                    prompt_token_count=len(prompt.split()),
                    raw_model_output_path=None,
                ),
            )
        except (QABackendUnavailableError, QAParseError) as exc:
            raise TransformersCliUsageError(detail=str(exc)) from exc
    written = rank_output_path(args.out, distributed)
    write_predictions_jsonl(tuple(predictions), written)
    if distributed.world_size == 1:
        return TransformersCliResult(written=written, predictions=len(predictions))
    if distributed.rank == 0:
        wait_for_shards(args.out, distributed.world_size, env)
        merge_shards(args.out, packs, distributed.world_size)
        return TransformersCliResult(written=args.out, predictions=len(packs))
    return TransformersCliResult(written=written, predictions=len(predictions))


def parse_cli_args(argv: Sequence[str]) -> TransformersCliArgs:
    model: str | None = None
    fixture: Path | None = None
    evidence: Path | None = None
    out: Path | None = None
    backend: TransformersBackendName = "gemma4"
    index = 0
    while index < len(argv):
        option = argv[index]
        if option in {"--model", "--fixture", "--evidence", "--out", "--backend"}:
            if index + 1 >= len(argv):
                raise TransformersCliUsageError(detail=f"missing value for {option}")
            value = argv[index + 1]
            if option == "--model":
                model = value
            elif option == "--fixture":
                fixture = Path(value)
            elif option == "--evidence":
                evidence = Path(value)
            elif option == "--out":
                out = Path(value)
            elif option == "--backend":
                backend = _parse_backend(value)
            index += 2
            continue
        raise TransformersCliUsageError(detail=f"unknown option: {option}")
    if model is None:
        raise TransformersCliUsageError(detail="qa_transformers requires --model")
    if fixture is None:
        raise TransformersCliUsageError(detail="qa_transformers requires --fixture")
    if evidence is None:
        raise TransformersCliUsageError(detail="qa_transformers requires --evidence")
    if out is None:
        raise TransformersCliUsageError(detail="qa_transformers requires --out")
    return TransformersCliArgs(
        model=model,
        fixture=fixture,
        evidence=evidence,
        out=out,
        backend=backend,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_cli_args(sys.argv[1:] if argv is None else argv)
        result = run_transformers_cli(args, env=os.environ)
    except (
        TransformersCliUsageError,
        RemoteOnlyError,
        QAShardError,
        TransformersGenerationError,
        OSError,
        ValidationError,
    ) as exc:
        _ = sys.stderr.write(f"{exc}\n")
        return 2
    _ = sys.stdout.write(f"wrote {result.written}\npredictions={result.predictions}\n")
    return 0


def _read_evidence_packs(path: Path) -> tuple[EvidencePack, ...]:
    packs: list[EvidencePack] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            packs.append(EvidencePack.model_validate_json(line))
        except ValidationError as exc:
            detail = f"{path}: line {line_number}: {exc}"
            raise TransformersCliUsageError(detail=detail) from exc
    return tuple(packs)


def _questions_by_id(fixture: Path) -> dict[str, QuestionRequest]:
    return {
        question.question_id: question
        for question in read_fixture_questions(fixture)
    }


def _question_for_pack(
    pack: EvidencePack,
    questions: Mapping[str, QuestionRequest],
) -> QuestionRequest:
    question = questions.get(pack.question_id)
    if question is None:
        raise TransformersCliUsageError(detail=f"unknown question: {pack.question_id}")
    return question


def _parse_backend(value: str) -> TransformersBackendName:
    match value:
        case "gemma4" | "real" | "mock":
            return value
        case other:
            raise TransformersCliUsageError(detail=f"unknown backend: {other}")


if __name__ == "__main__":
    raise SystemExit(main())
