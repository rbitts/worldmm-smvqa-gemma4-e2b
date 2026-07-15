from __future__ import annotations

import fcntl
import hashlib
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, override

from pydantic import Field, ValidationError, model_validator

from worldmm_smvqa.schema import FrozenModel, PredictionRecord


class QuestionIdPack(Protocol):
    """Provides the stable identifier used for shard assignment."""

    @property
    def question_id(self) -> str:
        """Return the stable question identifier."""
        ...


DEFAULT_APPROVED_QA_SHARD_SALT = "worldmm-qa-shards-v1"
FINALIZATION_RECEIPT_MISSING_PREDICTIONS = (
    "finalization receipt exists without final predictions"
)
FINALIZATION_TRANSACTION_MISMATCH = (
    "finalization receipt does not match sealed transaction"
)
FINAL_PREDICTIONS_RECEIPT_MISMATCH = (
    "final predictions do not match immutable finalization receipt"
)
UNRECEIPTED_FINAL_PREDICTIONS = (
    "unreceipted final predictions cannot be recovered without a sealed precommit"
)
PRECOMMIT_TRANSACTION_MISMATCH = (
    "precommitted final predictions do not match sealed transaction"
)


@dataclass(frozen=True, slots=True)
class QAShardError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"QAShardError: {self.detail}"


@dataclass(frozen=True, slots=True)
class DistributedEnv:
    """Describes the active distributed rank and world size."""

    rank: int
    world_size: int

    def __post_init__(self) -> None:
        if self.world_size < 1:
            msg = "world_size must be positive"
            raise QAShardError(msg)
        if self.rank < 0 or self.rank >= self.world_size:
            msg = "rank must be in [0, world_size)"
            raise QAShardError(msg)


@dataclass(frozen=True, slots=True)
class QuestionShardMapError(ValueError):
    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


class QuestionShardAssignment(FrozenModel):
    """Assign one question to a distributed rank."""

    question_id: str = Field(min_length=1)
    rank: int = Field(ge=0)


class QuestionShardMap(FrozenModel):
    """Sealed label-blind assignment, ordered by UTF-8 bytes."""

    approved_salt: str = Field(min_length=1)
    world_size: int = Field(ge=1)
    assignments: tuple[QuestionShardAssignment, ...]

    @model_validator(mode="after")
    def _require_complete_unique_map(self) -> QuestionShardMap:
        ids = tuple(item.question_id for item in self.assignments)
        if len(ids) != len(set(ids)):
            msg = "question shard map contains duplicate question IDs"
            raise QuestionShardMapError(msg)
        if tuple(sorted(ids, key=lambda item: item.encode("utf-8"))) != ids:
            msg = (
                "question shard map assignments must be sorted by UTF-8 question_id "
                "bytes"
            )
            raise QuestionShardMapError(msg)
        if any(item.rank >= self.world_size for item in self.assignments):
            msg = "question shard map rank is outside world_size"
            raise QuestionShardMapError(msg)
        expected = tuple(
            _rank_for_question(item.question_id, self.world_size, self.approved_salt)
            for item in self.assignments
        )
        if tuple(item.rank for item in self.assignments) != expected:
            msg = "question shard map assignment does not match approved salt"
            raise QuestionShardMapError(msg)
        return self

    @property
    def sha256(self) -> str:
        """Return the canonical SHA-256 digest of this assignment map."""
        return _sha256_text(self.model_dump_json())


class QAResumeManifest(FrozenModel):
    """Legacy resume binding retained for non-EXP-0005 callers only."""

    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    question_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class QAShardLineage(FrozenModel):
    """Complete label-blind pre-evaluation contract shared by E0, T0, and T1."""

    approved_salt: str = Field(min_length=1)
    world_size: int = Field(ge=1)
    question_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decoding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int


class QAFinalizationReceipt(FrozenModel):
    """Bind finalized rank predictions to their sealed lineage."""

    receipt_schema: Literal["qa_finalization_receipt_v1"] = "qa_finalization_receipt_v1"
    lineage: QAShardLineage
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_ids: tuple[str, ...]
    rank_receipt_sha256: tuple[str, ...] = ()


class QAPrecommitMarker(FrozenModel):
    """Crash-safe authorization for one exact final prediction publication."""

    marker_schema: Literal["qa_precommit_marker_v1"] = "qa_precommit_marker_v1"
    lineage: QAShardLineage
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_ids: tuple[str, ...]
    rank_receipt_sha256: tuple[str, ...] = ()


class QACheckpointMetadata(FrozenModel):
    """Bind partial progress to the exact sealed transaction lineage."""

    marker_schema: Literal["qa_checkpoint_metadata_v1"] = "qa_checkpoint_metadata_v1"
    lineage: QAShardLineage
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_ids: tuple[str, ...]


def distributed_env(env: Mapping[str, str]) -> DistributedEnv:
    rank = _env_int(env, "RANK", default=0)
    world_size = _env_int(env, "WORLD_SIZE", default=1)
    if world_size < 1:
        msg = "WORLD_SIZE must be positive"
        raise QAShardError(msg)
    if rank < 0 or rank >= world_size:
        msg = "RANK must be in [0, WORLD_SIZE)"
        raise QAShardError(msg)
    return DistributedEnv(rank, world_size)


def build_question_shard_map(
    packs: Sequence[QuestionIdPack],
    world_size: int,
    approved_salt: str,
) -> QuestionShardMap:
    if world_size < 1:
        msg = "world_size must be positive"
        raise QAShardError(msg)
    if not approved_salt:
        msg = "approved shard salt must not be blank"
        raise QAShardError(msg)
    ids = tuple(
        sorted(
            (pack.question_id for pack in packs),
            key=lambda item: item.encode("utf-8"),
        )
    )
    if len(ids) != len(set(ids)):
        msg = "evidence packs contain duplicate question IDs"
        raise QAShardError(msg)
    return QuestionShardMap(
        approved_salt=approved_salt,
        world_size=world_size,
        assignments=tuple(
            QuestionShardAssignment(
                question_id=item,
                rank=_rank_for_question(item, world_size, approved_salt),
            )
            for item in ids
        ),
    )


def packs_for_rank[QuestionIdPackT: QuestionIdPack](
    packs: Sequence[QuestionIdPackT],
    distributed: DistributedEnv,
    approved_salt: str = DEFAULT_APPROVED_QA_SHARD_SALT,
) -> tuple[QuestionIdPackT, ...]:
    shard_map = build_question_shard_map(
        packs,
        distributed.world_size,
        approved_salt,
    )
    return packs_for_rank_from_map(packs, distributed, shard_map)


def packs_for_rank_from_map[QuestionIdPackT: QuestionIdPack](
    packs: Sequence[QuestionIdPackT],
    distributed: DistributedEnv,
    shard_map: QuestionShardMap,
) -> tuple[QuestionIdPackT, ...]:
    if shard_map.world_size != distributed.world_size:
        msg = "question shard map world_size does not match rank"
        raise QAShardError(msg)
    by_id = {pack.question_id: pack for pack in packs}
    expected_ids = {item.question_id for item in shard_map.assignments}
    if len(by_id) != len(packs) or set(by_id) != expected_ids:
        msg = "evidence packs do not match immutable question shard map"
        raise QAShardError(msg)
    return tuple(
        by_id[item.question_id]
        for item in shard_map.assignments
        if item.rank == distributed.rank
    )


def rank_output_path(out: Path, distributed: DistributedEnv) -> Path:
    if distributed.world_size == 1:
        return out
    name = (
        f"{out.stem}.rank{distributed.rank:05d}-of{distributed.world_size:05d}"
        f"{out.suffix}"
    )
    return out.with_name(name)


def partial_output_path(out: Path) -> Path:
    return out.with_name(f"{out.name}.partial")


def resume_manifest_path(out: Path) -> Path:
    return out.with_name(f"{out.name}.resume.json")


def finalization_receipt_path(out: Path) -> Path:
    return out.with_name(f"{out.name}.final.json")


def precommit_marker_path(out: Path) -> Path:
    return out.with_name(f"{out.name}.precommit.json")


def checkpoint_metadata_path(out: Path) -> Path:
    return out.with_name(f"{out.name}.checkpoint.json")


# Explicit compatibility API: it cannot create an EXP-0005 finalization receipt.
def load_rank_progress(
    out: Path,
    expected_resume: QAResumeManifest | None = None,
) -> tuple[PredictionRecord, ...]:
    _reject_legacy_reserved_path(out)
    _validate_resume_manifest(out, expected_resume)
    if out.exists():
        return _read_predictions(out)
    partial = partial_output_path(out)
    return _read_predictions(partial) if partial.exists() else ()


def checkpoint_rank(
    out: Path,
    predictions: Sequence[PredictionRecord],
    resume_manifest: QAResumeManifest | None = None,
) -> None:
    _reject_legacy_reserved_path(out)
    _write_resume_manifest(out, resume_manifest)
    _write_predictions_atomic(partial_output_path(out), predictions)


def complete_rank(
    out: Path,
    predictions: Sequence[PredictionRecord],
    resume_manifest: QAResumeManifest | None = None,
) -> None:
    _reject_legacy_reserved_path(out)
    _write_resume_manifest(out, resume_manifest)
    _write_predictions_atomic(out, predictions)
    partial_output_path(out).unlink(missing_ok=True)


def sealed_load_rank_progress(
    out: Path,
    shard_map: QuestionShardMap,
    distributed: DistributedEnv,
    lineage: QAShardLineage,
) -> tuple[PredictionRecord, ...]:
    _validate_lineage(shard_map, distributed, lineage)
    lock_descriptor = _claim_finalization_lock(out)
    try:
        expected = _rank_question_ids(shard_map, distributed.rank)
        if _recover_finalization_receipt(out, lineage, expected):
            return _read_receipt_bound_predictions(
                out,
                finalization_receipt_path(out),
                None,
                shard_map,
                distributed,
                lineage,
            )
        partial = partial_output_path(out)
        if not partial.exists():
            return ()
        predictions, prediction_digest = _read_prediction_snapshot(partial)
        _validate_checkpoint_metadata(out, lineage, expected, prediction_digest)
        _validate_prediction_prefix(predictions, expected)
        return predictions
    finally:
        os.close(lock_descriptor)


def sealed_checkpoint_rank(
    out: Path,
    predictions: Sequence[PredictionRecord],
    shard_map: QuestionShardMap,
    distributed: DistributedEnv,
    lineage: QAShardLineage,
) -> None:
    _validate_lineage(shard_map, distributed, lineage)
    if (
        out.exists()
        or finalization_receipt_path(out).exists()
        or precommit_marker_path(out).exists()
    ):
        msg = "cannot write after rank finalization"
        raise QAShardError(msg)
    _validate_prediction_prefix(
        predictions, _rank_question_ids(shard_map, distributed.rank)
    )
    _write_checkpoint_metadata(
        out,
        QACheckpointMetadata(
            lineage=lineage,
            predictions_sha256=_prediction_digest(predictions),
            question_ids=_rank_question_ids(shard_map, distributed.rank),
        ),
    )
    _write_predictions_atomic(partial_output_path(out), predictions)


def sealed_complete_rank(
    out: Path,
    predictions: Sequence[PredictionRecord],
    shard_map: QuestionShardMap,
    distributed: DistributedEnv,
    lineage: QAShardLineage,
) -> None:
    _validate_lineage(shard_map, distributed, lineage)
    lock_descriptor = _claim_finalization_lock(out)
    try:
        expected = _rank_question_ids(shard_map, distributed.rank)
        if _recover_finalization_receipt(out, lineage, expected):
            finalized, _ = _read_prediction_snapshot(out)
            if _prediction_digest(predictions) != _prediction_digest(finalized):
                msg = "finalized predictions do not match completion request"
                raise QAShardError(msg)
            return
        if finalization_receipt_path(out).exists():
            msg = "finalization receipt exists without final predictions"
            raise QAShardError(msg)
        _validate_prediction_exact(predictions, expected)
        prediction_digest = _prediction_digest(predictions)
        _write_precommit_marker(
            out,
            QAPrecommitMarker(
                lineage=lineage,
                predictions_sha256=prediction_digest,
                question_ids=expected,
            ),
        )
        _write_predictions_atomic(out, predictions)
        _write_receipt(
            out,
            QAFinalizationReceipt(
                lineage=lineage,
                predictions_sha256=prediction_digest,
                question_ids=expected,
            ),
        )
        precommit_marker_path(out).unlink(missing_ok=True)
        checkpoint_metadata_path(out).unlink(missing_ok=True)
        partial_output_path(out).unlink(missing_ok=True)
    finally:
        os.close(lock_descriptor)


# Explicit compatibility API: legacy shards are committed by their prediction files.
def wait_for_shards(out: Path, world_size: int, env: Mapping[str, str]) -> None:
    timeout = _env_int(env, "WORLDMM_QA_SHARD_TIMEOUT_SECONDS", default=3600)
    deadline = time.monotonic() + timeout
    paths = tuple(
        rank_output_path(out, DistributedEnv(rank, world_size))
        for rank in range(world_size)
    )
    _wait_for_paths(paths, deadline, "missing QA rank shard(s)")


def wait_for_sealed_shards(
    out: Path, world_size: int, env: Mapping[str, str]
) -> dict[Path, str]:
    """Wait for sealed receipt commit markers and snapshot their contents."""
    timeout = _env_int(env, "WORLDMM_QA_SHARD_TIMEOUT_SECONDS", default=3600)
    deadline = time.monotonic() + timeout
    paths = tuple(
        finalization_receipt_path(
            rank_output_path(out, DistributedEnv(rank, world_size))
        )
        for rank in range(world_size)
    )
    _wait_for_paths(paths, deadline, "missing QA rank finalization receipt(s)")
    return {path: _receipt_file_digest(path) for path in paths}


def merge_shards(out: Path, packs: Sequence[QuestionIdPack], world_size: int) -> None:
    """Legacy merge; EXP-0005 must use merge_sealed_shards."""
    if world_size < 1:
        msg = "world_size must be positive"
        raise QAShardError(msg)
    expected = tuple(pack.question_id for pack in packs)
    if len(expected) != len(set(expected)):
        msg = "evidence packs contain duplicate question IDs"
        raise QAShardError(msg)
    found: dict[str, PredictionRecord] = {}
    for rank in range(world_size):
        path = rank_output_path(out, DistributedEnv(rank, world_size))
        for prediction in _read_predictions(path):
            if prediction.question_id in found:
                msg = f"duplicate prediction for {prediction.question_id}"
                raise QAShardError(msg)
            found[prediction.question_id] = prediction
    if unexpected := set(found).difference(expected):
        msg = f"unexpected prediction for {sorted(unexpected)[0]}"
        raise QAShardError(msg)
    _write_predictions_atomic(
        out, tuple(_prediction_for_id(item, found) for item in expected)
    )


def merge_sealed_shards(
    out: Path,
    packs: Sequence[QuestionIdPack],
    shard_map: QuestionShardMap,
    lineage: QAShardLineage,
) -> None:
    """Merge only receipt-bound, exact-rank, canonical-order publications."""
    _validate_lineage(shard_map, DistributedEnv(0, shard_map.world_size), lineage)
    _validate_packs_against_shard_map(packs, shard_map)
    if shard_map.world_size == 1:
        try:
            lock_descriptor = _claim_finalization_lock(out)
        except QAShardError as exc:
            if "finalization is in progress" not in str(exc):
                raise
        else:
            try:
                expected_ids = tuple(item.question_id for item in shard_map.assignments)
                if _recover_finalization_receipt(out, lineage, expected_ids):
                    _ = _read_receipt_bound_predictions(
                        out,
                        finalization_receipt_path(out),
                        None,
                        shard_map,
                        DistributedEnv(0, 1),
                        lineage,
                    )
                    return
            finally:
                os.close(lock_descriptor)
    receipt_snapshots = wait_for_sealed_shards(out, shard_map.world_size, os.environ)
    if shard_map.world_size == 1:
        receipt_path = finalization_receipt_path(out)
        _ = _read_receipt_bound_predictions(
            out,
            receipt_path,
            receipt_snapshots.get(receipt_path),
            shard_map,
            DistributedEnv(0, 1),
            lineage,
        )
        return
    lock_descriptor = _claim_finalization_lock(out)
    try:
        expected_ids = tuple(item.question_id for item in shard_map.assignments)
        rank_receipt_digests = tuple(
            receipt_snapshots[
                finalization_receipt_path(
                    rank_output_path(out, DistributedEnv(rank, shard_map.world_size))
                )
            ]
            for rank in range(shard_map.world_size)
        )
        by_question: dict[str, PredictionRecord] = {}
        for rank in range(shard_map.world_size):
            distributed = DistributedEnv(rank, shard_map.world_size)
            path = rank_output_path(out, distributed)
            receipt_path = finalization_receipt_path(path)
            predictions = _read_receipt_bound_predictions(
                path,
                receipt_path,
                receipt_snapshots.get(receipt_path),
                shard_map,
                distributed,
                lineage,
            )
            by_question.update(
                {prediction.question_id: prediction for prediction in predictions}
            )
        if _recover_finalization_receipt(
            out, lineage, expected_ids, rank_receipt_digests
        ):
            return
        merged = tuple(
            _prediction_for_id(question_id, by_question) for question_id in expected_ids
        )
        _validate_prediction_exact(merged, expected_ids)
        prediction_digest = _prediction_digest(merged)
        _write_precommit_marker(
            out,
            QAPrecommitMarker(
                lineage=lineage,
                predictions_sha256=prediction_digest,
                question_ids=expected_ids,
                rank_receipt_sha256=rank_receipt_digests,
            ),
        )
        _write_predictions_atomic(out, merged)
        _write_receipt(
            out,
            QAFinalizationReceipt(
                lineage=lineage,
                predictions_sha256=prediction_digest,
                question_ids=expected_ids,
                rank_receipt_sha256=rank_receipt_digests,
            ),
        )
        precommit_marker_path(out).unlink(missing_ok=True)
    finally:
        os.close(lock_descriptor)


def _rank_for_question(question_id: str, world_size: int, approved_salt: str) -> int:
    salt = approved_salt.encode("utf-8")
    question = question_id.encode("utf-8")
    preimage = (
        b"worldmm-qa-shard-v1\x00"
        + len(salt).to_bytes(4, "big")
        + salt
        + len(question).to_bytes(4, "big")
        + question
    )
    return int.from_bytes(hashlib.sha256(preimage).digest(), "big") % world_size


def _rank_question_ids(shard_map: QuestionShardMap, rank: int) -> tuple[str, ...]:
    return tuple(
        item.question_id for item in shard_map.assignments if item.rank == rank
    )


def _validate_lineage(
    shard_map: QuestionShardMap,
    distributed: DistributedEnv,
    lineage: QAShardLineage,
) -> None:
    if (
        shard_map.world_size != distributed.world_size
        or shard_map.world_size != lineage.world_size
    ):
        msg = "sealed lineage world_size does not match question shard map"
        raise QAShardError(msg)
    if (
        shard_map.approved_salt != lineage.approved_salt
        or shard_map.sha256 != lineage.question_map_sha256
    ):
        msg = "sealed lineage does not match question shard map"
        raise QAShardError(msg)


def _validate_packs_against_shard_map(
    packs: Sequence[QuestionIdPack], shard_map: QuestionShardMap
) -> None:
    pack_ids = tuple(pack.question_id for pack in packs)
    expected_pack_ids = {item.question_id for item in shard_map.assignments}
    if len(pack_ids) != len(set(pack_ids)) or set(pack_ids) != expected_pack_ids:
        msg = "evidence packs do not match immutable question shard map"
        raise QAShardError(msg)


def _validate_prediction_prefix(
    predictions: Sequence[PredictionRecord],
    expected: Sequence[str],
) -> None:
    ids = tuple(item.question_id for item in predictions)
    if ids != tuple(expected[: len(ids)]):
        msg = "rank predictions are not an exact ordered resume prefix"
        raise QAShardError(msg)


def _validate_prediction_exact(
    predictions: Sequence[PredictionRecord],
    expected: Sequence[str],
) -> None:
    _validate_prediction_prefix(predictions, expected)
    if len(predictions) != len(expected):
        msg = "rank predictions are incomplete"
        raise QAShardError(msg)


def _prediction_for_id(
    question_id: str,
    predictions: Mapping[str, PredictionRecord],
) -> PredictionRecord:
    if (prediction := predictions.get(question_id)) is None:
        msg = f"missing prediction for {question_id}"
        raise QAShardError(msg)
    return prediction


def _prediction_digest(predictions: Sequence[PredictionRecord]) -> str:
    return _sha256_bytes(_prediction_bytes(predictions))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _prediction_bytes(predictions: Sequence[PredictionRecord]) -> bytes:
    payload = "".join(f"{item.model_dump_json()}\n" for item in predictions)
    return payload.encode("utf-8")


def _finalization_lock_path(out: Path) -> Path:
    return out.with_name(f"{out.name}.final.lock")


def _claim_finalization_lock(out: Path) -> int:
    """Acquire a kernel-released, process-lifetime exclusive advisory lock."""
    lock_path = _finalization_lock_path(out)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        msg = "cannot write while finalization is in progress"
        raise QAShardError(msg) from exc
    return descriptor


def _receipt_file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        msg = f"invalid finalization receipt: {path}"
        raise QAShardError(msg) from exc


def _write_receipt(out: Path, receipt: QAFinalizationReceipt) -> None:
    _write_text_atomic(finalization_receipt_path(out), f"{receipt.model_dump_json()}\n")


def _write_precommit_marker(out: Path, marker: QAPrecommitMarker) -> None:
    _write_text_atomic(precommit_marker_path(out), f"{marker.model_dump_json()}\n")


def _write_checkpoint_metadata(out: Path, metadata: QACheckpointMetadata) -> None:
    _write_text_atomic(checkpoint_metadata_path(out), f"{metadata.model_dump_json()}\n")


def _validate_checkpoint_metadata(
    out: Path,
    lineage: QAShardLineage,
    expected_question_ids: Sequence[str],
    predictions_sha256: str,
) -> None:
    try:
        metadata = QACheckpointMetadata.model_validate_json(
            checkpoint_metadata_path(out).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        msg = "partial predictions lack valid sealed checkpoint metadata"
        raise QAShardError(msg) from exc
    if (
        metadata.lineage != lineage
        or metadata.question_ids != tuple(expected_question_ids)
        or metadata.predictions_sha256 != predictions_sha256
    ):
        msg = "partial predictions do not match sealed checkpoint metadata"
        raise QAShardError(msg)


def _read_receipt_snapshot(path: Path) -> tuple[QAFinalizationReceipt, str]:
    try:
        contents = path.read_bytes()
        return (
            QAFinalizationReceipt.model_validate_json(contents),
            _sha256_bytes(contents),
        )
    except (OSError, ValidationError) as exc:
        msg = f"invalid finalization receipt: {path}"
        raise QAShardError(msg) from exc


def _read_receipt_bound_predictions(  # noqa: PLR0913
    out: Path,
    receipt_path: Path,
    receipt_snapshot: str | None,
    shard_map: QuestionShardMap,
    distributed: DistributedEnv,
    lineage: QAShardLineage,
) -> tuple[PredictionRecord, ...]:
    receipt, receipt_digest = _read_receipt_snapshot(receipt_path)
    if receipt_snapshot is not None and receipt_snapshot != receipt_digest:
        msg = "finalization receipt changed after readiness"
        raise QAShardError(msg)
    if receipt.lineage != lineage:
        msg = "finalization receipt lineage does not match sealed lineage"
        raise QAShardError(msg)
    if receipt.rank_receipt_sha256:
        msg = "rank finalization receipt must not contain merged rank receipts"
        raise QAShardError(msg)
    expected = _rank_question_ids(shard_map, distributed.rank)
    if receipt.question_ids != expected:
        msg = "finalization receipt question order does not match canonical assignment"
        raise QAShardError(msg)
    predictions, prediction_digest = _read_prediction_snapshot(out)
    if receipt.predictions_sha256 != prediction_digest:
        msg = "final predictions do not match immutable finalization receipt"
        raise QAShardError(msg)
    _validate_prediction_exact(predictions, expected)
    return predictions


def _recover_finalization_receipt(
    out: Path,
    lineage: QAShardLineage,
    expected_question_ids: Sequence[str],
    rank_receipt_sha256: Sequence[str] = (),
) -> bool:
    """Recover only a matching precommitted sealed publication."""
    receipt_path = finalization_receipt_path(out)
    if receipt_path.exists():
        if not out.exists():
            raise QAShardError(FINALIZATION_RECEIPT_MISSING_PREDICTIONS)
        receipt, _ = _read_receipt_snapshot(receipt_path)
        if (
            receipt.lineage != lineage
            or receipt.question_ids != tuple(expected_question_ids)
            or receipt.rank_receipt_sha256 != tuple(rank_receipt_sha256)
        ):
            raise QAShardError(FINALIZATION_TRANSACTION_MISMATCH)
        predictions, prediction_digest = _read_prediction_snapshot(out)
        if receipt.predictions_sha256 != prediction_digest:
            raise QAShardError(FINAL_PREDICTIONS_RECEIPT_MISMATCH)
        _validate_prediction_exact(predictions, expected_question_ids)
        return True
    if not out.exists():
        if not precommit_marker_path(out).exists():
            return False
        try:
            marker = QAPrecommitMarker.model_validate_json(
                precommit_marker_path(out).read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise QAShardError(UNRECEIPTED_FINAL_PREDICTIONS) from exc
        if (
            marker.lineage != lineage
            or marker.question_ids != tuple(expected_question_ids)
            or marker.rank_receipt_sha256 != tuple(rank_receipt_sha256)
        ):
            raise QAShardError(PRECOMMIT_TRANSACTION_MISMATCH)
        raise QAShardError(UNRECEIPTED_FINAL_PREDICTIONS)
    try:
        marker = QAPrecommitMarker.model_validate_json(
            precommit_marker_path(out).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise QAShardError(UNRECEIPTED_FINAL_PREDICTIONS) from exc
    predictions, prediction_digest = _read_prediction_snapshot(out)
    if (
        marker.lineage != lineage
        or marker.question_ids != tuple(expected_question_ids)
        or marker.predictions_sha256 != prediction_digest
        or marker.rank_receipt_sha256 != tuple(rank_receipt_sha256)
    ):
        raise QAShardError(PRECOMMIT_TRANSACTION_MISMATCH)
    _validate_prediction_exact(predictions, expected_question_ids)
    _write_receipt(
        out,
        QAFinalizationReceipt(
            lineage=lineage,
            predictions_sha256=prediction_digest,
            question_ids=tuple(expected_question_ids),
            rank_receipt_sha256=tuple(rank_receipt_sha256),
        ),
    )
    precommit_marker_path(out).unlink(missing_ok=True)
    checkpoint_metadata_path(out).unlink(missing_ok=True)
    partial_output_path(out).unlink(missing_ok=True)
    return True


def _wait_for_paths(
    paths: Sequence[Path],
    deadline: float,
    missing_prefix: str,
) -> None:
    while any(not path.exists() for path in paths):
        if time.monotonic() >= deadline:
            missing = tuple(str(path) for path in paths if not path.exists())
            msg = f"{missing_prefix}: {', '.join(missing)}"
            raise QAShardError(msg)
        time.sleep(0.1)


def _reject_legacy_reserved_path(out: Path) -> None:
    """Keep compatibility writers out of sealed receipt and lock namespaces."""
    reserved = (finalization_receipt_path(out), _finalization_lock_path(out))
    if out in reserved or out.name.endswith((".final.json", ".final.lock")):
        msg = "legacy compatibility APIs cannot write sealed receipt or lock paths"
        raise QAShardError(msg)


def _write_resume_manifest(out: Path, manifest: QAResumeManifest | None) -> None:
    if manifest is not None:
        _write_text_atomic(resume_manifest_path(out), f"{manifest.model_dump_json()}\n")


def _validate_resume_manifest(out: Path, expected: QAResumeManifest | None) -> None:
    if expected is None:
        return
    path = resume_manifest_path(out)
    if not path.exists():
        if out.exists() or partial_output_path(out).exists():
            msg = "resume predictions lack a resume manifest"
            raise QAShardError(msg)
        return
    try:
        actual = QAResumeManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        msg = f"invalid resume manifest: {exc}"
        raise QAShardError(msg) from exc
    if actual != expected:
        msg = "resume manifest does not match current model/prompt/seed/map/evidence"
        raise QAShardError(msg)


def _env_int(env: Mapping[str, str], name: str, *, default: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        msg = f"{name} must be an integer"
        raise QAShardError(msg) from exc


def _read_prediction_snapshot(path: Path) -> tuple[tuple[PredictionRecord, ...], str]:
    try:
        contents = path.read_bytes()
    except OSError as exc:
        msg = f"invalid predictions: {path}"
        raise QAShardError(msg) from exc
    return _parse_predictions(path, contents.decode("utf-8")), _sha256_bytes(contents)


def _read_predictions(path: Path) -> tuple[PredictionRecord, ...]:
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"invalid predictions: {path}"
        raise QAShardError(msg) from exc
    return _parse_predictions(path, contents)


def _parse_predictions(path: Path, contents: str) -> tuple[PredictionRecord, ...]:
    predictions: list[PredictionRecord] = []
    for line_number, line in enumerate(contents.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            predictions.append(PredictionRecord.model_validate_json(line))
        except ValidationError as exc:
            msg = f"{path}: line {line_number}: {exc}"
            raise QAShardError(msg) from exc
    return tuple(predictions)


def _write_predictions_atomic(
    path: Path,
    predictions: Sequence[PredictionRecord],
) -> None:
    _write_text_atomic(
        path, "".join(f"{item.model_dump_json()}\n" for item in predictions)
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        _ = temporary.write_text(text, encoding="utf-8")
        _ = temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
