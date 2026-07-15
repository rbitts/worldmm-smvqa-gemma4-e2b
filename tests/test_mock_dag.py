from __future__ import annotations

import socket
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from worldmm_smvqa.mock_dag import run_local_mock_dag

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/spatial/model_boundary_contract_v1.json"
ARCHITECTURE = ROOT / "configs/spatial/student_architecture_v1.json"


def test_local_mock_dag_runs_one_real_optimizer_step_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_network(*_args: object, **_kwargs: object) -> None:
        detail = "mock DAG attempted network access"
        raise AssertionError(detail)

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket.socket, "connect", reject_network)

    result = run_local_mock_dag(CONTRACT, student_architecture=ARCHITECTURE)

    assert result.observations == 2
    assert all(count > 0 for count in result.qwen_store_counts.values())
    assert "spatial" in result.selected_stores
    assert set(result.selected_stores).intersection({"episodic", "semantic", "visual"})
    assert result.loss >= 0.0
    assert result.gradient_parameter_count > 0
    assert result.optimizer_steps == 1
    assert result.parameter_changed
    assert result.checkpoint_round_trip
    assert result.prediction_answerable
