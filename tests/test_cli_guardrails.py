from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    _ = env.pop("WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST", None)
    return subprocess.run(
        ["uv", "run", "worldmm-smvqa", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_lists_scaffold_commands() -> None:
    # Given: a newly scaffolded benchmark CLI.
    # When: the user asks for top-level help.
    result = run_cli("--help")

    # Then: every planned command surface is discoverable.
    assert result.returncode == 0
    for command in (
        "prepare-fixture",
        "validate-schema",
        "build-memory",
        "retrieve",
        "retrieve-batch",
        "qa",
        "evaluate",
        "diagnose-spatial",
        "report",
        "smoke",
        "launch-remote",
    ):
        assert command in result.stdout


def test_launch_remote_dry_run_prints_commands_without_submit(tmp_path: Path) -> None:
    # Given: remote config that uses environment placeholders.
    out_dir = tmp_path / "remote_plan"

    # When: launch-remote is run in dry-run mode.
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    # Then: it prints the remote command and does not submit work.
    assert result.returncode == 0
    assert 'ssh -J "$BASTION_HOST" "$HEAD_NODE"' in result.stdout
    assert "dry-run" in result.stdout
    assert (out_dir / "expected_outputs.json").is_file()


def test_qa_real_model_is_remote_only_on_local_config() -> None:
    # Given: local config and no explicit remote override.
    # When: real-model QA is requested.
    result = run_cli("qa", "--config", "configs/local.example.yaml", "--real-model")

    # Then: the guardrail refuses before any model work can start.
    assert result.returncode != 0
    assert "remote-only" in result.stderr


def test_missing_config_path_fails_before_command_work() -> None:
    # Given: a config path that does not exist.
    # When: a command tries to load it.
    result = run_cli("smoke", "--config", "configs/missing.yaml")

    # Then: the CLI reports the bad path and exits nonzero.
    assert result.returncode != 0
    assert "ConfigNotFound" in result.stderr


def test_build_memory_invalid_stage_fails_nonzero(tmp_path: Path) -> None:
    # Given: an unsupported build-memory stage and requested artifact path.
    output = tmp_path / "bad.jsonl"

    # When: the command is run.
    result = run_cli("build-memory", "--stage", "bad", "--out", str(output))

    # Then: it fails with a typed usage error and writes no artifact.
    assert result.returncode != 0
    assert "UsageError" in result.stderr
    assert not output.exists()


def test_build_memory_invalid_store_fails_nonzero(tmp_path: Path) -> None:
    # Given: an unsupported build-memory store and requested artifact directory.
    output = tmp_path / "bad"

    # When: the command is run.
    result = run_cli("build-memory", "--store", "bad", "--out", str(output))

    # Then: it fails with a typed usage error and writes no artifact.
    assert result.returncode != 0
    assert "UsageError" in result.stderr
    assert not output.exists()
