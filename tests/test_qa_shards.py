from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from worldmm_smvqa import qa_shards
from worldmm_smvqa.qa_shards import (
    DistributedEnv,
    QAFinalizationReceipt,
    QAResumeManifest,
    QAShardError,
    QAShardLineage,
    QuestionShardMap,
    build_question_shard_map,
    checkpoint_rank,
    complete_rank,
    finalization_receipt_path,
    load_rank_progress,
    merge_sealed_shards,
    merge_shards,
    packs_for_rank_from_map,
    partial_output_path,
    precommit_marker_path,
    rank_output_path,
    sealed_checkpoint_rank,
    sealed_complete_rank,
    sealed_load_rank_progress,
    wait_for_sealed_shards,
    wait_for_shards,
)
from worldmm_smvqa.schema import PredictionRecord

SIMULATED_CRASH_MESSAGE = "simulated crash"


@dataclass(frozen=True)
class _Pack:
    question_id: str


def _prediction(question_id: str, answer: str = "A") -> PredictionRecord:
    return PredictionRecord(
        question_id=question_id,
        answerable=True,
        ranked_choices=(answer,),
        answer=answer,
        confidence=1.0,
        supporting_memory_ids=(),
        prompt_token_count=1,
        raw_model_output_path=None,
    )


def _resume(digest: str = "a" * 64) -> QAResumeManifest:
    return QAResumeManifest(
        model_sha256=digest,
        prompt_sha256=digest,
        seed=7,
        question_map_sha256=digest,
        evidence_sha256=digest,
    )


def _lineage(
    mapping: QuestionShardMap,
    digest: str = "a" * 64,
) -> QAShardLineage:
    return QAShardLineage(
        approved_salt=mapping.approved_salt,
        world_size=mapping.world_size,
        question_map_sha256=mapping.sha256,
        model_sha256=digest,
        prompt_sha256=digest,
        decoding_sha256=digest,
        runtime_sha256=digest,
        evidence_sha256=digest,
        seed=7,
    )


def test_shard_map_rejects_tampered_hash_assignment() -> None:
    mapping = build_question_shard_map((_Pack("q1"),), 2, "approved-salt")
    assignment = mapping.assignments[0]
    with pytest.raises(ValueError, match="does not match approved salt"):
        _ = mapping.model_validate(
            {
                **mapping.model_dump(),
                "assignments": [
                    {
                        "question_id": assignment.question_id,
                        "rank": 1 - assignment.rank,
                    }
                ],
            }
        )


def test_empty_final_is_authoritative_over_stale_partial(tmp_path: Path) -> None:
    out = tmp_path / "predictions.jsonl"
    complete_rank(out, ())
    checkpoint_rank(out, (_prediction("stale"),))
    assert load_rank_progress(out) == ()
    assert partial_output_path(out).exists()


def test_partition_is_hash_stable_across_input_order() -> None:
    packs = tuple(_Pack(question_id) for question_id in ("q3", "q1", "q2"))
    mapping = build_question_shard_map(packs, 3, "approved-salt")
    reversed_mapping = build_question_shard_map(
        tuple(reversed(packs)), 3, "approved-salt"
    )
    assert mapping == reversed_mapping
    assert {
        pack.question_id
        for pack in packs_for_rank_from_map(packs, DistributedEnv(0, 3), mapping)
    } == {item.question_id for item in mapping.assignments if item.rank == 0}


def test_distributed_env_rejects_invalid_direct_construction() -> None:
    with pytest.raises(QAShardError, match="world_size must be positive"):
        _ = DistributedEnv(0, 0)
    with pytest.raises(QAShardError, match="rank must be in"):
        _ = DistributedEnv(1, 1)


def test_resume_rejects_any_bound_input_mismatch(tmp_path: Path) -> None:
    out = tmp_path / "predictions.jsonl"
    checkpoint_rank(out, (_prediction("q1"),), _resume())
    with pytest.raises(QAShardError, match="resume manifest does not match"):
        _ = load_rank_progress(out, _resume("b" * 64))


def test_merge_rejects_duplicate_and_missing_predictions(tmp_path: Path) -> None:
    out = tmp_path / "predictions.jsonl"
    packs = (_Pack("q1"), _Pack("q2"))
    complete_rank(rank_output_path(out, DistributedEnv(0, 2)), (_prediction("q1"),))
    complete_rank(rank_output_path(out, DistributedEnv(1, 2)), (_prediction("q1"),))
    with pytest.raises(QAShardError, match="duplicate prediction"):
        merge_shards(out, packs, 2)

    complete_rank(rank_output_path(out, DistributedEnv(1, 2)), (_prediction("q2"),))
    merge_shards(out, packs, 2)
    assert tuple(line.question_id for line in load_rank_progress(out)) == ("q1", "q2")

    out.unlink()
    complete_rank(rank_output_path(out, DistributedEnv(1, 2)), ())
    with pytest.raises(QAShardError, match="missing prediction"):
        merge_shards(out, packs, 2)


def test_sealed_rank_rejects_prefix_tamper_and_final_mutation(tmp_path: Path) -> None:
    out = tmp_path / "sealed.jsonl"
    packs = (_Pack("β"), _Pack("a"))
    mapping = build_question_shard_map(packs, 1, "salt-雪")
    lineage = _lineage(mapping)
    with pytest.raises(QAShardError, match="ordered resume prefix"):
        sealed_checkpoint_rank(
            out,
            (_prediction("β"),),
            mapping,
            DistributedEnv(0, 1),
            lineage,
        )
    sealed_checkpoint_rank(
        out,
        (_prediction("a"),),
        mapping,
        DistributedEnv(0, 1),
        lineage,
    )
    sealed_complete_rank(
        out,
        (_prediction("a"), _prediction("β")),
        mapping,
        DistributedEnv(0, 1),
        lineage,
    )
    assert finalization_receipt_path(out).exists()
    with pytest.raises(QAShardError, match="cannot write after rank finalization"):
        sealed_checkpoint_rank(
            out,
            (_prediction("a"),),
            mapping,
            DistributedEnv(0, 1),
            lineage,
        )
    merge_sealed_shards(out, packs, mapping, lineage)
    wrong_predictions = (_prediction("a"), _prediction("unexpected"))
    serialized = "".join(
        f"{prediction.model_dump_json()}\n" for prediction in wrong_predictions
    )
    _ = out.write_text(serialized, encoding="utf-8")
    wrong_receipt = QAFinalizationReceipt(
        lineage=lineage,
        predictions_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        question_ids=("a", "β"),
    )
    _ = finalization_receipt_path(out).write_text(
        f"{wrong_receipt.model_dump_json()}\n",
        encoding="utf-8",
    )
    with pytest.raises(QAShardError, match="ordered resume prefix"):
        _ = sealed_load_rank_progress(out, mapping, DistributedEnv(0, 1), lineage)
    with pytest.raises(QAShardError, match="ordered resume prefix"):
        merge_sealed_shards(out, packs, mapping, lineage)


def test_sealed_load_rejects_noncanonical_prediction_bytes(tmp_path: Path) -> None:
    out = tmp_path / "sealed.jsonl"
    packs = (_Pack("q1"),)
    mapping = build_question_shard_map(packs, 1, "salt")
    lineage = _lineage(mapping)
    sealed_complete_rank(
        out, (_prediction("q1"),), mapping, DistributedEnv(0, 1), lineage
    )

    _ = out.write_bytes(out.read_bytes() + b"\n")

    with pytest.raises(
        QAShardError, match="do not match immutable finalization receipt"
    ):
        _ = sealed_load_rank_progress(out, mapping, DistributedEnv(0, 1), lineage)


def test_single_rank_merge_validates_packs_before_return(tmp_path: Path) -> None:
    out = tmp_path / "sealed.jsonl"
    packs = (_Pack("q1"),)
    mapping = build_question_shard_map(packs, 1, "salt")
    lineage = _lineage(mapping)
    sealed_complete_rank(
        out, (_prediction("q1"),), mapping, DistributedEnv(0, 1), lineage
    )

    with pytest.raises(QAShardError, match="evidence packs do not match"):
        merge_sealed_shards(out, (), mapping, lineage)


def test_sealed_waiter_requires_receipt_commit_marker(tmp_path: Path) -> None:
    out = tmp_path / "sealed.jsonl"
    complete_rank(out, (_prediction("q1"),))

    wait_for_shards(out, 1, {"WORLDMM_QA_SHARD_TIMEOUT_SECONDS": "0"})
    with pytest.raises(QAShardError, match="finalization receipt"):
        _ = wait_for_sealed_shards(out, 1, {"WORLDMM_QA_SHARD_TIMEOUT_SECONDS": "0"})


def test_concurrent_sealed_writers_publish_exactly_one_receipt_bound_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "sealed.jsonl"
    mapping = build_question_shard_map((_Pack("q1"),), 1, "salt")
    lineage = _lineage(mapping)
    write_started = threading.Event()
    release_write = threading.Event()
    original_write = qa_shards._write_predictions_atomic  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    writer_errors: list[Exception] = []

    def pause_first_final_write(
        path: Path, predictions: tuple[PredictionRecord, ...]
    ) -> None:
        if path == out and not write_started.is_set():
            write_started.set()
            assert release_write.wait(timeout=1)
        original_write(path, predictions)

    def first_writer() -> None:
        try:
            sealed_complete_rank(
                out,
                (_prediction("q1"),),
                mapping,
                DistributedEnv(0, 1),
                lineage,
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - asserted below
            writer_errors.append(exc)

    monkeypatch.setattr(qa_shards, "_write_predictions_atomic", pause_first_final_write)
    writer = threading.Thread(target=first_writer)
    writer.start()
    assert write_started.wait(timeout=1)

    with pytest.raises(
        QAShardError, match="cannot write while finalization is in progress"
    ):
        sealed_complete_rank(
            out,
            (_prediction("q1", "B"),),
            mapping,
            DistributedEnv(0, 1),
            lineage,
        )

    release_write.set()
    writer.join(timeout=1)
    assert not writer.is_alive()
    assert writer_errors == []
    assert sealed_load_rank_progress(out, mapping, DistributedEnv(0, 1), lineage) == (
        _prediction("q1"),
    )


def test_receipt_is_the_concurrent_commit_marker_for_waiter_and_merger(  # noqa: PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "sealed.jsonl"
    packs = (_Pack("q1"),)
    mapping = build_question_shard_map(packs, 1, "salt")
    lineage = _lineage(mapping)
    receipt_started = threading.Event()
    release_receipt = threading.Event()
    waiter_done = threading.Event()
    merger_done = threading.Event()
    writer_errors: list[Exception] = []
    waiter_errors: list[Exception] = []
    merger_errors: list[Exception] = []
    original_receipt = qa_shards._write_receipt  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    original_wait = qa_shards.wait_for_sealed_shards

    def delay_receipt(path: Path, receipt: QAFinalizationReceipt) -> None:
        receipt_started.set()
        assert release_receipt.wait(timeout=1)
        original_receipt(path, receipt)

    def wait_with_test_timeout(
        wait_out: Path, world_size: int, env: dict[str, str]
    ) -> dict[Path, str]:
        return original_wait(
            wait_out,
            world_size,
            {**env, "WORLDMM_QA_SHARD_TIMEOUT_SECONDS": "1"},
        )

    def writer() -> None:
        try:
            sealed_complete_rank(
                out,
                (_prediction("q1"),),
                mapping,
                DistributedEnv(0, 1),
                lineage,
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - asserted below
            writer_errors.append(exc)

    def waiter() -> None:
        try:
            _ = wait_for_sealed_shards(
                out, 1, {"WORLDMM_QA_SHARD_TIMEOUT_SECONDS": "1"}
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - asserted below
            waiter_errors.append(exc)
        finally:
            waiter_done.set()

    def merger() -> None:
        try:
            merge_sealed_shards(out, packs, mapping, lineage)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - asserted below
            merger_errors.append(exc)
        finally:
            merger_done.set()

    monkeypatch.setattr(qa_shards, "_write_receipt", delay_receipt)
    monkeypatch.setattr(qa_shards, "wait_for_sealed_shards", wait_with_test_timeout)
    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    assert receipt_started.wait(timeout=1)
    waiter_thread = threading.Thread(target=waiter)
    merger_thread = threading.Thread(target=merger)
    waiter_thread.start()
    merger_thread.start()
    assert not waiter_done.wait(timeout=0.05)
    assert not merger_done.wait(timeout=0.05)
    assert out.exists()
    assert not finalization_receipt_path(out).exists()

    release_receipt.set()
    writer_thread.join(timeout=1)
    waiter_thread.join(timeout=1)
    merger_thread.join(timeout=1)
    assert not writer_thread.is_alive()
    assert not waiter_thread.is_alive()
    assert not merger_thread.is_alive()
    assert writer_errors == []
    assert waiter_errors == []
    assert merger_errors == []
    assert sealed_load_rank_progress(out, mapping, DistributedEnv(0, 1), lineage) == (
        _prediction("q1"),
    )


def test_merge_rejects_receipt_and_shard_replacement_after_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "merged.jsonl"
    packs = tuple(_Pack(f"q{index}") for index in range(8))
    mapping = build_question_shard_map(packs, 2, "salt")
    lineage = _lineage(mapping)
    for rank in range(2):
        expected = tuple(
            item.question_id for item in mapping.assignments if item.rank == rank
        )
        sealed_complete_rank(
            rank_output_path(out, DistributedEnv(rank, 2)),
            tuple(_prediction(question_id) for question_id in expected),
            mapping,
            DistributedEnv(rank, 2),
            lineage,
        )
    rank_to_replace = next(
        rank
        for rank in range(2)
        if any(item.rank == rank for item in mapping.assignments)
    )
    path = rank_output_path(out, DistributedEnv(rank_to_replace, 2))
    replacement_ids = tuple(
        item.question_id for item in mapping.assignments if item.rank == rank_to_replace
    )
    replacement = tuple(
        _prediction(question_id, "B") for question_id in replacement_ids
    )
    serialized = "".join(
        f"{prediction.model_dump_json()}\n" for prediction in replacement
    )
    replacement_receipt = QAFinalizationReceipt(
        lineage=lineage,
        predictions_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        question_ids=replacement_ids,
    )
    original_wait = qa_shards.wait_for_sealed_shards

    def replace_after_readiness(
        wait_out: Path, world_size: int, env: dict[str, str]
    ) -> dict[Path, str]:
        snapshots = original_wait(wait_out, world_size, env)
        _ = path.write_text(serialized, encoding="utf-8")
        _ = finalization_receipt_path(path).write_text(
            f"{replacement_receipt.model_dump_json()}\n", encoding="utf-8"
        )
        return snapshots

    monkeypatch.setattr(qa_shards, "wait_for_sealed_shards", replace_after_readiness)
    with pytest.raises(QAShardError, match="receipt changed after readiness"):
        merge_sealed_shards(out, packs, mapping, lineage)
    assert not out.exists()


def test_precommit_without_predictions_is_immutable_on_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "predictions.jsonl"
    packs = (_Pack("q1"),)
    mapping = build_question_shard_map(packs, 1, "salt")
    lineage = _lineage(mapping)

    def crash_after_precommit(path: Path, predictions: object) -> None:
        _ = path, predictions
        raise OSError(SIMULATED_CRASH_MESSAGE)

    monkeypatch.setattr(qa_shards, "_write_predictions_atomic", crash_after_precommit)
    with pytest.raises(OSError, match=SIMULATED_CRASH_MESSAGE):
        sealed_complete_rank(
            out, (_prediction("q1"),), mapping, DistributedEnv(0, 1), lineage
        )
    assert precommit_marker_path(out).exists()
    with pytest.raises(QAShardError, match="unreceipted final predictions"):
        sealed_complete_rank(
            out, (_prediction("q1"),), mapping, DistributedEnv(0, 1), lineage
        )


def test_precommit_rejects_cross_lineage_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "predictions.jsonl"
    packs = (_Pack("q1"),)
    mapping = build_question_shard_map(packs, 1, "salt")
    lineage = _lineage(mapping)

    def crash_after_precommit(path: Path, predictions: object) -> None:
        _ = path, predictions
        raise OSError(SIMULATED_CRASH_MESSAGE)

    monkeypatch.setattr(qa_shards, "_write_predictions_atomic", crash_after_precommit)
    with pytest.raises(OSError, match=SIMULATED_CRASH_MESSAGE):
        sealed_complete_rank(
            out, (_prediction("q1"),), mapping, DistributedEnv(0, 1), lineage
        )
    with pytest.raises(QAShardError, match="precommitted final predictions"):
        _ = sealed_load_rank_progress(
            out, mapping, DistributedEnv(0, 1), _lineage(mapping, "b" * 64)
        )
