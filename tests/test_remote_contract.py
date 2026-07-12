from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

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
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "UV_NO_NETWORK": "1"},
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
    command = [
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
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    # Then: sync and launch commands parse to safe argv without eval.
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    plan_sync_argv = shlex.split(lines[-2])
    assert plan_sync_argv[0] == "rsync"
    assert plan_sync_argv[-2] == f"{out_dir}/"
    argv = shlex.split(lines[-1])
    assert argv[:4] == [
        "ssh",
        "-J",
        "$BASTION_HOST",
        "$HEAD_NODE",
    ]
    remote_argv = shlex.split(argv[4])
    assert remote_argv == [
        "cd",
        ("${WORLDMM_REMOTE_REPO:-/repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b}"),
        "&&",
        "mkdir",
        "-p",
        "remote-plan/logs",
        "&&",
        "WORLDMM_DAG_PHASE=preflight",
        "bash",
        "remote-plan/submit_worldmm_smvqa_dag.sh",
    ]
    assert str(out_dir) not in argv[4]


def test_checked_in_remote_script_delegates_without_printing_paths() -> None:
    # Given: the checked-in remote helper is user-facing.
    script_text = (ROOT / "scripts/remote/run_worldmm_smvqa.sh").read_text(
        encoding="utf-8",
    )

    # Then: it delegates to the generated plan and does not print sensitive paths.
    assert "worldmm-smvqa launch-remote --dry-run" in script_text
    assert "printf 'WORLDMM_OUTPUT_ROOT=%s" not in script_text
    assert "printf 'GEMMA_MODEL_PATH=%s" not in script_text
