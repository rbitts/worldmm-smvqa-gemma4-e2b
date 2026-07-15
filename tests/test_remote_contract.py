from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from worldmm_smvqa.remote_plan import ExperimentGraphV1
from worldmm_smvqa.schema import PredictionRecord

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ENV_NAMES = frozenset(
    {
        "SMVQA_DATA_ROOT",
        "GEMMA_MODEL_PATH",
        "WORLDMM_OUTPUT_ROOT",
        "BASTION_HOST",
        "HEAD_NODE",
    },
)


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in REMOTE_ENV_NAMES | {
        "WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST",
        "WORLDMM_SMVQA_REMOTE_APPROVED",
    }:
        _ = env.pop(name, None)
    return subprocess.run(
        ["uv", "run", "--offline", "worldmm-smvqa", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_qa_transformers_mock_cli_writes_predictions_from_evidence(
    tmp_path: Path,
) -> None:
    # Given: evidence packs generated locally without model weights.
    smoke_dir = tmp_path / "smoke"
    smoke = run_cli(
        "smoke",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--out",
        str(smoke_dir),
    )
    assert smoke.returncode == 0, smoke.stderr
    predictions = tmp_path / "predictions.jsonl"

    # When: the remote QA module command is probed in mock mode.
    result = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "python",
            "-m",
            "worldmm_smvqa.qa_transformers",
            "--model",
            "remote-gemma-placeholder",
            "--fixture",
            "tests/fixtures/tiny_smvqa",
            "--evidence",
            str(smoke_dir / "evidence_packs.jsonl"),
            "--evidence-lane",
            "heuristic",
            "--out",
            str(predictions),
            "--backend",
            "mock",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "UV_NO_NETWORK": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    # Then: predictions are actually written.
    assert result.returncode == 0, result.stderr
    rows = [
        PredictionRecord.model_validate_json(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 6
    assert {row.question_id for row in rows} == {
        "q_fake_001",
        "q_fake_002",
        "q_fake_003",
        "q_fake_004",
        "q_fake_005",
        "q_fake_006",
    }


def test_qa_transformers_mock_cli_shards_and_merges_ddp_predictions(
    tmp_path: Path,
) -> None:
    # Given: evidence packs generated locally without model weights.
    smoke_dir = tmp_path / "smoke"
    smoke = run_cli(
        "smoke",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--out",
        str(smoke_dir),
    )
    assert smoke.returncode == 0, smoke.stderr
    predictions = tmp_path / "predictions.jsonl"
    triton_cache_root = tmp_path / "triton-cache"
    command: list[str] = [
        "uv",
        "run",
        "--offline",
        "python",
        "-m",
        "worldmm_smvqa.qa_transformers",
        "--model",
        "remote-gemma-placeholder",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--evidence",
        str(smoke_dir / "evidence_packs.jsonl"),
        "--evidence-lane",
        "heuristic",
        "--out",
        str(predictions),
        "--backend",
        "mock",
    ]

    # When: nonzero rank writes first, then rank zero writes and merges.
    rank_one = subprocess.run(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "RANK": "1",
            "UV_NO_NETWORK": "1",
            "WORLDMM_TRITON_CACHE_ROOT": str(triton_cache_root),
            "WORLD_SIZE": "2",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    rank_zero = subprocess.run(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "RANK": "0",
            "UV_NO_NETWORK": "1",
            "WORLDMM_TRITON_CACHE_ROOT": str(triton_cache_root),
            "WORLD_SIZE": "2",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    # Then: each rank used a distinct shard and rank zero merged final output.
    assert rank_one.returncode == 0, rank_one.stderr
    assert rank_zero.returncode == 0, rank_zero.stderr
    assert (tmp_path / "predictions.rank00001-of00002.jsonl").is_file()
    assert (tmp_path / "predictions.rank00000-of00002.jsonl").is_file()
    assert (triton_cache_root / "rank-00000").is_dir()
    assert (triton_cache_root / "rank-00001").is_dir()
    rows = [
        PredictionRecord.model_validate_json(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    assert [row.question_id for row in rows] == [
        "q_fake_001",
        "q_fake_002",
        "q_fake_003",
        "q_fake_004",
        "q_fake_005",
        "q_fake_006",
    ]


def test_plan_stdout_shell_quotes_script_path(tmp_path: Path) -> None:
    # Given: an out path with quotes and shell metacharacters.
    out_dir = tmp_path / "remote plan' ; touch bad $(uname)"

    # When: launch-remote prints the copy/paste command.
    result = run_cli(
        "launch-remote",
        "--profile",
        "teacher-oracle",
        "--experiment-config",
        "configs/spatial/exp_0005_teacher_oracle.example.json",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    # Then: the dry-run exposes absolute operator artifacts, not execution commands.
    assert result.returncode == 0, result.stderr
    assert f"wrote {out_dir / 'operator_contract.json'}" in result.stdout
    assert f"wrote {out_dir / 'approval_blockers.json'}" in result.stdout
    assert "ssh -J" not in result.stdout
    assert "WORLDMM_DAG_PHASE=" not in result.stdout


def test_checked_in_remote_script_delegates_without_printing_paths() -> None:
    # Given: the checked-in remote helper is user-facing.
    script_text = (ROOT / "scripts/remote/run_worldmm_smvqa.sh").read_text(
        encoding="utf-8"
    )

    # Then: it delegates to the generated plan and does not print sensitive paths.
    assert "worldmm-smvqa launch-remote --dry-run" in script_text
    assert "printf 'WORLDMM_OUTPUT_ROOT=%s" not in script_text
    assert "printf 'GEMMA_MODEL_PATH=%s" not in script_text


def test_teacher_oracle_config_is_one_frozen_experiment_graph() -> None:
    graph = ExperimentGraphV1.model_validate_json(
        (ROOT / "configs/spatial/exp_0005_teacher_oracle.example.json").read_text(
            encoding="utf-8"
        )
    )

    assert graph.model_config.get("frozen") is True
    assert tuple(_stage.name for _stage in graph.stage_specs) == (
        "preflight",
        "geometry",
        "semantic",
        "place",
        "gate",
        "terminal",
        "e0_materialize",
        "e0_retrieve",
        "e0_qa",
        "t0_materialize",
        "t0_retrieve",
        "t0_qa",
        "t1_materialize",
        "t1_retrieve",
        "t1_qa",
        "evaluator",
        "finalizer",
    )
    rendered = json.dumps(graph.model_dump(mode="json"), sort_keys=True)
    for key in graph.manifest_job_keys:
        assert key in rendered
