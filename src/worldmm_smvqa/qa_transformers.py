from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, override

from pydantic import ValidationError

from worldmm_smvqa.chunking import read_source_streams
from worldmm_smvqa.config import REMOTE_ENV_FLAG, RemoteOnlyError
from worldmm_smvqa.fixtures import read_fixture_questions
from worldmm_smvqa.qa_shards import (
    QAShardError,
    checkpoint_rank,
    complete_rank,
    distributed_env,
    load_rank_progress,
    merge_shards,
    packs_for_rank,
    rank_output_path,
    wait_for_shards,
)
from worldmm_smvqa.retrieval_types import EvidencePack
from worldmm_smvqa.transformers_backend import TransformersGenerationError
from worldmm_smvqa.video_frames import sample_video_frames

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
        parse_qa_output,
    )
    from worldmm_smvqa.qa_prompt import build_qa_prompt  # noqa: PLC0415

    questions = _questions_by_id(args.fixture)
    sources = read_source_streams(args.fixture)
    packs = _read_evidence_packs(args.evidence)
    frame_root = Path(env.get("SMVQA_FRAME_ROOT", args.fixture / "frames"))
    match args.backend:
        case "mock":
            backend: QABackend = MockQABackend()
        case "gemma4" | "real":
            if env.get(REMOTE_ENV_FLAG) != "1":
                raise RemoteOnlyError(action="qa_transformers real model")
            backend = Gemma4QABackend(model_path=args.model)

    distributed = distributed_env(env)
    rank_packs = packs_for_rank(packs, distributed)
    written = rank_output_path(args.out, distributed)
    predictions = list(load_rank_progress(written))
    _validate_rank_progress(rank_packs, predictions, completed=written.exists())
    completed_question_ids = {
        prediction.question_id for prediction in predictions
    }
    for pack in rank_packs:
        if pack.question_id in completed_question_ids:
            continue
        question = _question_for_pack(pack, questions)
        video_frames = sample_video_frames(
            sources,
            question,
            pack,
            frame_root=frame_root,
            max_frames=32,
        )
        prompt = build_qa_prompt(question, pack, video_frames)
        raw_outputs: list[str] = []
        raw_output_path = _raw_output_path(args.out, question.question_id)
        try:
            prediction: PredictionRecord | None = None
            last_parse_error: QAParseError | None = None
            for _attempt in range(2):
                raw_outputs.extend(
                    backend.raw_outputs(prompt, question, pack, video_frames),
                )
                _write_raw_outputs(raw_output_path, raw_outputs)
                try:
                    prediction = parse_qa_output(
                        question=question,
                        raw_outputs=raw_outputs,
                        prompt_token_count=len(prompt.split()),
                        raw_model_output_path=str(raw_output_path),
                        evidence_pack=pack,
                    )
                except QAParseError as exc:
                    last_parse_error = exc
                    continue
                break
            if prediction is None:
                if last_parse_error is None:
                    raise TransformersCliUsageError(
                        detail=f"no QA output for {question.question_id}",
                    )
                raise last_parse_error
            predictions.append(prediction)
            checkpoint_rank(written, predictions)
        except (QABackendUnavailableError, QAParseError) as exc:
            raise TransformersCliUsageError(detail=str(exc)) from exc
    complete_rank(written, predictions)
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


def _validate_rank_progress(
    packs: Sequence[EvidencePack],
    predictions: Sequence[PredictionRecord],
    *,
    completed: bool,
) -> None:
    expected = {pack.question_id for pack in packs}
    actual = [prediction.question_id for prediction in predictions]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for question_id in actual:
        if question_id in seen:
            duplicates.add(question_id)
        seen.add(question_id)
    if duplicates:
        raise QAShardError(
            detail=f"duplicate QA checkpoint question: {sorted(duplicates)[0]}",
        )
    unexpected = set(actual) - expected
    if unexpected:
        raise QAShardError(
            detail=f"unexpected QA checkpoint question: {sorted(unexpected)[0]}",
        )
    if completed and set(actual) != expected:
        missing = sorted(expected - set(actual))
        raise QAShardError(
            detail=f"incomplete final QA shard; missing: {', '.join(missing)}",
        )


def _raw_output_path(out: Path, question_id: str) -> Path:
    digest = hashlib.sha256(question_id.encode()).hexdigest()[:16]
    return out.parent / f"{out.stem}_raw_model_outputs" / f"q_{digest}.json"


def _write_raw_outputs(path: Path, raw_outputs: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _ = temporary.write_text(
            json.dumps(
                {"raw_outputs": tuple(raw_outputs)},
                ensure_ascii=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        _ = temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
