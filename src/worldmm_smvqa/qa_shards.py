from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

from pydantic import ValidationError

if TYPE_CHECKING:
    from worldmm_smvqa.retrieval_types import EvidencePack
    from worldmm_smvqa.schema import PredictionRecord


@dataclass(frozen=True, slots=True)
class DistributedEnv:
    rank: int
    world_size: int


@dataclass(frozen=True, slots=True)
class QAShardError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"QAShardError: {self.detail}"


def distributed_env(env: Mapping[str, str]) -> DistributedEnv:
    rank = _env_int(env, "RANK", default=0)
    world_size = _env_int(env, "WORLD_SIZE", default=1)
    if world_size < 1:
        raise QAShardError(detail="WORLD_SIZE must be positive")
    if rank < 0 or rank >= world_size:
        raise QAShardError(detail="RANK must be in [0, WORLD_SIZE)")
    return DistributedEnv(rank=rank, world_size=world_size)


def packs_for_rank(
    packs: Sequence[EvidencePack],
    distributed: DistributedEnv,
) -> tuple[EvidencePack, ...]:
    return tuple(
        pack
        for index, pack in enumerate(packs)
        if index % distributed.world_size == distributed.rank
    )


def rank_output_path(out: Path, distributed: DistributedEnv) -> Path:
    if distributed.world_size == 1:
        return out
    shard_name = (
        f"{out.stem}.rank{distributed.rank:05d}"
        f"-of{distributed.world_size:05d}{out.suffix}"
    )
    return out.with_name(shard_name)


def wait_for_shards(out: Path, world_size: int, env: Mapping[str, str]) -> None:
    timeout_seconds = _env_int(env, "WORLDMM_QA_SHARD_TIMEOUT_SECONDS", default=3600)
    deadline = time.monotonic() + timeout_seconds
    shard_paths = tuple(
        rank_output_path(out, DistributedEnv(rank=rank, world_size=world_size))
        for rank in range(world_size)
    )
    while True:
        missing = tuple(path for path in shard_paths if not path.exists())
        if not missing:
            return
        if time.monotonic() >= deadline:
            names = ", ".join(str(path) for path in missing)
            raise QAShardError(detail=f"missing QA rank shard(s): {names}")
        time.sleep(0.1)


def merge_shards(out: Path, packs: Sequence[EvidencePack], world_size: int) -> None:
    from worldmm_smvqa.schema import PredictionRecord  # noqa: PLC0415

    predictions_by_question: dict[str, PredictionRecord] = {}
    for rank in range(world_size):
        shard = rank_output_path(out, DistributedEnv(rank=rank, world_size=world_size))
        lines = shard.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                prediction = PredictionRecord.model_validate_json(line)
            except ValidationError as exc:
                detail = f"{shard}: line {line_number}: {exc}"
                raise QAShardError(detail=detail) from exc
            predictions_by_question[prediction.question_id] = prediction
    ordered_predictions = tuple(
        _prediction_for_pack(pack, predictions_by_question) for pack in packs
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    _ = out.write_text(
        "".join(
            f"{prediction.model_dump_json()}\n"
            for prediction in ordered_predictions
        ),
        encoding="utf-8",
    )


def _prediction_for_pack(
    pack: EvidencePack,
    predictions_by_question: Mapping[str, PredictionRecord],
) -> PredictionRecord:
    prediction = predictions_by_question.get(pack.question_id)
    if prediction is None:
        raise QAShardError(detail=f"missing prediction for {pack.question_id}")
    return prediction


def _env_int(env: Mapping[str, str], name: str, *, default: int) -> int:
    raw_value = env.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise QAShardError(detail=f"{name} must be an integer") from exc
