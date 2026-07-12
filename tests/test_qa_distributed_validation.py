from __future__ import annotations

import hashlib
import importlib
import json
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, override

import pytest

from worldmm_smvqa import qa_transformers
from worldmm_smvqa.qa_shards import DistributedEnv
from worldmm_smvqa.qa_transformers import (
    TransformersCliUsageError,
    synchronize_rank_memory_read,
    synchronize_typed_memory_postread,
    synchronize_typed_memory_validation,
)
from worldmm_smvqa.retrieval_types import EvidenceLineage

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(slots=True)
class _SharedBroadcast:
    payload: object = None
    calls: list[tuple[str, int]] = field(default_factory=list)
    gathered: dict[int, object] = field(default_factory=dict)


class _FakeDistributed:
    def __init__(
        self,
        rank: int,
        shared: _SharedBroadcast,
        *,
        initialized: bool = True,
    ) -> None:
        self.rank: int = rank
        self.shared: _SharedBroadcast = shared
        self.initialized: bool = initialized

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self.initialized

    def get_rank(self) -> int:
        return self.rank

    def get_world_size(self) -> int:
        return 2

    def barrier(self) -> None:
        self.shared.calls.append(("barrier", self.rank))

    def broadcast_object_list(
        self,
        object_list: list[object],
        src: int = 0,
    ) -> None:
        assert src == 0
        self.shared.calls.append(("broadcast", self.rank))
        if self.rank == src:
            self.shared.payload = object_list[0]
        else:
            object_list[0] = self.shared.payload

    def all_gather_object(
        self,
        object_list: list[object],
        obj: object,
    ) -> None:
        self.shared.gathered[self.rank] = obj
        if len(self.shared.gathered) != self.get_world_size():
            msg = "sequential fake requires preloaded all-gather values"
            raise RuntimeError(msg)
        for rank in range(self.get_world_size()):
            object_list[rank] = self.shared.gathered[rank]


class _ThreadedDistributed(_FakeDistributed):
    def __init__(
        self,
        rank: int,
        shared: _SharedBroadcast,
        barrier: threading.Barrier,
        root_waiting: threading.Event,
        payload_ready: threading.Event,
    ) -> None:
        super().__init__(rank, shared)
        self.thread_barrier: threading.Barrier = barrier
        self.root_waiting: threading.Event = root_waiting
        self.payload_ready: threading.Event = payload_ready

    @override
    def barrier(self) -> None:
        if self.rank == 0:
            self.root_waiting.set()
        _ = self.thread_barrier.wait(timeout=2.0)

    @override
    def broadcast_object_list(
        self,
        object_list: list[object],
        src: int = 0,
    ) -> None:
        if self.rank == src:
            self.shared.payload = object_list[0]
            self.payload_ready.set()
            return
        assert self.payload_ready.wait(timeout=2.0)
        object_list[0] = self.shared.payload

    @override
    def all_gather_object(
        self,
        object_list: list[object],
        obj: object,
    ) -> None:
        self.shared.gathered[self.rank] = obj
        _ = self.thread_barrier.wait(timeout=2.0)
        for rank in range(self.get_world_size()):
            object_list[rank] = self.shared.gathered[rank]
        _ = self.thread_barrier.wait(timeout=2.0)


class _FakeProcessGroup(_FakeDistributed):
    def __init__(self, shared: _SharedBroadcast) -> None:
        super().__init__(0, shared, initialized=False)
        self.init_call: tuple[str, str, timedelta] | None = None

    def init_process_group(
        self,
        backend: str,
        *,
        init_method: str,
        timeout: timedelta,
    ) -> None:
        self.init_call = (backend, init_method, timeout)
        self.initialized: bool = True

    def destroy_process_group(self) -> None:
        self.initialized = False


def _validation_inputs(tmp_path: Path) -> tuple[Path, EvidenceLineage]:
    inference_manifest = tmp_path / "typed-memory.inference.json"
    payload = {
        "record_count": 3,
        "actual_bytes": 1200,
        "window_count": 2,
        "max_window_bytes": 700,
        "window_seconds": 30.0,
    }
    _ = inference_manifest.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(inference_manifest.read_bytes()).hexdigest()
    lineage = EvidenceLineage(
        lane="student",
        producer="spatial-student",
        evidence_sha256="0" * 64,
        checkpoint_sha256="1" * 64,
        typed_memory_sha256="2" * 64,
        inference_manifest_sha256=digest,
        config_sha256="3" * 64,
        sensor_sha256="4" * 64,
        data_sha256="5" * 64,
        memory_manifest_sha256="6" * 64,
        episodic_memory_sha256="7" * 64,
        semantic_memory_sha256="8" * 64,
        visual_memory_sha256="9" * 64,
    )
    return inference_manifest, lineage


def test_distributed_typed_validation_runs_full_validator_only_on_rank_zero(
    tmp_path: Path,
) -> None:
    inference_manifest, lineage = _validation_inputs(tmp_path)
    shared = _SharedBroadcast()
    calls = 0

    def validate() -> EvidenceLineage:
        nonlocal calls
        calls += 1
        return lineage

    rank_zero = synchronize_typed_memory_validation(
        validate,
        inference_manifest,
        DistributedEnv(rank=0, world_size=2),
        distributed_api=_FakeDistributed(0, shared),
    )
    rank_one = synchronize_typed_memory_validation(
        lambda: pytest.fail("nonzero rank performed full validation"),
        inference_manifest,
        DistributedEnv(rank=1, world_size=2),
        distributed_api=_FakeDistributed(1, shared),
    )

    assert calls == 1
    assert rank_one == rank_zero
    assert rank_zero.typed_memory_sha256 == "2" * 64
    assert rank_zero.record_count == 3
    assert shared.calls == [
        ("barrier", 0),
        ("broadcast", 0),
        ("barrier", 1),
        ("broadcast", 1),
    ]


def test_distributed_typed_validation_broadcasts_rank_zero_failure(
    tmp_path: Path,
) -> None:
    inference_manifest, _ = _validation_inputs(tmp_path)
    shared = _SharedBroadcast()

    def reject() -> EvidenceLineage:
        raise TransformersCliUsageError(detail="invalid typed artifact")

    with pytest.raises(TransformersCliUsageError) as root_error:
        _ = synchronize_typed_memory_validation(
            reject,
            inference_manifest,
            DistributedEnv(rank=0, world_size=2),
            distributed_api=_FakeDistributed(0, shared),
        )
    with pytest.raises(TransformersCliUsageError) as peer_error:
        _ = synchronize_typed_memory_validation(
            lambda: pytest.fail("peer validator must not run"),
            inference_manifest,
            DistributedEnv(rank=1, world_size=2),
            distributed_api=_FakeDistributed(1, shared),
        )

    assert root_error.value.detail == "invalid typed artifact"
    assert peer_error.value.detail == root_error.value.detail


def test_uninitialized_distributed_api_preserves_local_validation(
    tmp_path: Path,
) -> None:
    inference_manifest, lineage = _validation_inputs(tmp_path)
    shared = _SharedBroadcast()
    called = False

    def validate() -> EvidenceLineage:
        nonlocal called
        called = True
        return lineage

    seal = synchronize_typed_memory_validation(
        validate,
        inference_manifest,
        DistributedEnv(rank=1, world_size=2),
        distributed_api=_FakeDistributed(1, shared, initialized=False),
    )

    assert called
    assert seal.lineage == lineage
    assert shared.calls == []


def test_rank_zero_validation_waits_for_slow_rank_before_reading_artifact(
    tmp_path: Path,
) -> None:
    inference_manifest, lineage = _validation_inputs(tmp_path)
    artifact = tmp_path / "artifact.txt"
    _ = artifact.write_text("valid", encoding="utf-8")
    shared = _SharedBroadcast()
    root_waiting = threading.Event()
    payload_ready = threading.Event()
    barrier = threading.Barrier(2)
    errors: list[str] = []

    def validate() -> EvidenceLineage:
        if artifact.read_text(encoding="utf-8") != "valid":
            raise TransformersCliUsageError(detail="artifact changed")
        return lineage

    def run_rank(rank: int, validator: Callable[[], EvidenceLineage | None]) -> None:
        try:
            _ = synchronize_typed_memory_validation(
                validator,
                inference_manifest,
                DistributedEnv(rank=rank, world_size=2),
                distributed_api=_ThreadedDistributed(
                    rank,
                    shared,
                    barrier,
                    root_waiting,
                    payload_ready,
                ),
            )
        except TransformersCliUsageError as exc:
            errors.append(exc.detail)

    root = threading.Thread(target=run_rank, args=(0, validate))
    root.start()
    assert root_waiting.wait(timeout=2.0)
    _ = artifact.write_text("changed", encoding="utf-8")
    peer = threading.Thread(
        target=run_rank,
        args=(1, lambda: pytest.fail("peer validator must not run")),
    )
    peer.start()
    root.join(timeout=2.0)
    peer.join(timeout=2.0)

    assert not root.is_alive()
    assert not peer.is_alive()
    assert errors == ["artifact changed", "artifact changed"]


def test_postread_digest_recheck_runs_only_on_rank_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_manifest, lineage = _validation_inputs(tmp_path)
    seal = synchronize_typed_memory_validation(
        lambda: lineage,
        inference_manifest,
        DistributedEnv(rank=0, world_size=1),
    )
    shared = _SharedBroadcast()
    calls = 0

    def validate_memory(_lineage: EvidenceLineage, _path: Path) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        qa_transformers,
        "_validate_memory_artifact_lineage",
        validate_memory,
    )
    synchronize_typed_memory_postread(
        seal,
        tmp_path / "memory-manifest.json",
        inference_manifest,
        DistributedEnv(rank=0, world_size=2),
        distributed_api=_FakeDistributed(0, shared),
    )
    synchronize_typed_memory_postread(
        seal,
        tmp_path / "memory-manifest.json",
        inference_manifest,
        DistributedEnv(rank=1, world_size=2),
        distributed_api=_FakeDistributed(1, shared),
    )

    assert calls == 1


def test_distributed_validation_rejects_malformed_broadcast_payload(
    tmp_path: Path,
) -> None:
    inference_manifest, _ = _validation_inputs(tmp_path)
    shared = _SharedBroadcast(payload='{"schema_version":1,"status":"ok"}')

    with pytest.raises(
        TransformersCliUsageError,
        match="invalid distributed typed-memory validation payload",
    ):
        _ = synchronize_typed_memory_validation(
            lambda: pytest.fail("peer validator must not run"),
            inference_manifest,
            DistributedEnv(rank=1, world_size=2),
            distributed_api=_FakeDistributed(1, shared),
        )


def test_qa_entrypoint_initializes_bounded_gloo_validation_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_group = _FakeProcessGroup(_SharedBroadcast())

    def import_module(name: str) -> object:
        assert name == "torch.distributed"
        return process_group

    monkeypatch.setattr(
        importlib,
        "import_module",
        import_module,
    )

    initialized = qa_transformers.initialize_qa_validation_process_group(
        {
            "RANK": "0",
            "WORLD_SIZE": "2",
            "WORLDMM_QA_SHARD_TIMEOUT_SECONDS": "7200",
        },
    )

    assert initialized is process_group
    assert process_group.init_call == (
        "gloo",
        "env://",
        timedelta(seconds=7200),
    )


def test_rank_memory_failure_is_gathered_and_raised_on_every_rank() -> None:
    shared = _SharedBroadcast()
    barrier = threading.Barrier(2)
    root_waiting = threading.Event()
    payload_ready = threading.Event()
    errors: dict[int, str] = {}

    def run_rank(rank: int, error: Exception | None) -> None:
        try:
            synchronize_rank_memory_read(
                error,
                DistributedEnv(rank=rank, world_size=2),
                distributed_api=_ThreadedDistributed(
                    rank,
                    shared,
                    barrier,
                    root_waiting,
                    payload_ready,
                ),
            )
        except TransformersCliUsageError as exc:
            errors[rank] = exc.detail

    root = threading.Thread(target=run_rank, args=(0, None))
    peer = threading.Thread(
        target=run_rank,
        args=(1, ValueError("bad rank projection")),
    )
    root.start()
    peer.start()
    root.join(timeout=2.0)
    peer.join(timeout=2.0)

    assert not root.is_alive()
    assert not peer.is_alive()
    assert errors == {
        0: "rank 1 memory projection failed: ValueError: bad rank projection",
        1: "rank 1 memory projection failed: ValueError: bad rank projection",
    }
