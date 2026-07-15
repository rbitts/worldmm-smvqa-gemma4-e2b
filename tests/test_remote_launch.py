from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import pytest


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
    for name in REMOTE_ENV_NAMES | {"WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST"}:
        _ = env.pop(name, None)
    return subprocess.run(
        ["uv", "run", "worldmm-smvqa", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _install_remote_command_sentinels(tmp_path: Path) -> tuple[Path, Path]:
    sentinel_dir = tmp_path / "remote-command-sentinels"
    sentinel_dir.mkdir()
    invocation_log = tmp_path / "remote-command-invocations.log"
    sentinel = (
        '#!/bin/sh\nprintf "%s\\n" "$0" >> "$WORLDMM_REMOTE_SENTINEL_LOG"\nexit 97\n'
    )
    for command in (
        "ssh",
        "rsync",
        "scp",
        "sftp",
        "sbatch",
        "srun",
        "curl",
        "wget",
    ):
        path = sentinel_dir / command
        _ = path.write_text(sentinel, encoding="utf-8")
        path.chmod(0o755)
    return sentinel_dir, invocation_log


def test_launch_remote_teacher_oracle_dry_run_never_invokes_remote_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_dir, invocation_log = _install_remote_command_sentinels(tmp_path)
    monkeypatch.setenv(
        "PATH",
        f"{sentinel_dir}{os.pathsep}{os.environ['PATH']}",
    )
    monkeypatch.setenv("WORLDMM_REMOTE_SENTINEL_LOG", str(invocation_log))
    out_dir = tmp_path / "remote_plan"

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

    assert result.returncode == 0, result.stderr
    assert not invocation_log.exists()
    assert "remote plan mode=dry-run" in result.stdout
    assert (
        "# dry-run/plan only; no ssh, remote shell, or job submission opened locally"
        in (result.stdout)
    )
    assert "WORLDMM_SMVQA_REMOTE_APPROVED=1" not in result.stdout
    assert {
        "approval_blockers.json",
        "copyback_policy.txt",
        "expected_outputs.json",
        "experiment_graph.json",
        "operator_contract.json",
        "run_teacher_oracle_stage.sh",
        "submit_teacher_oracle_downstream.sh",
        "submit_teacher_oracle_preflight.sh",
        "submit_teacher_oracle_provider_gate.sh",
    } == {path.name for path in out_dir.iterdir()}
    assert not (out_dir / "run_worldmm_smvqa.sh").exists()
    blockers = cast(
        "dict[str, object]",
        json.loads((out_dir / "approval_blockers.json").read_text(encoding="utf-8")),
    )
    assert blockers["runnable"] is False
    assert blockers["blockers"]


def test_launch_remote_dry_run_prints_operator_contract_not_a_phase_shortcut(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "remote_plan"

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

    assert result.returncode == 0
    assert "remote plan mode=dry-run" in result.stdout
    assert f"wrote {out_dir / 'operator_contract.json'}" in result.stdout
    assert f"wrote {out_dir / 'approval_blockers.json'}" in result.stdout
    assert "runnable=false" in result.stdout
    assert "WORLDMM_DAG_PHASE=phase-a" not in result.stdout
    assert "bash remote-plan/submit_teacher_oracle_preflight.sh" not in result.stdout
    assert "WORLDMM_SMVQA_REMOTE_APPROVED=1" not in result.stdout
    assert "legacy single-job compatibility" not in result.stdout
    assert (out_dir / "submit_teacher_oracle_preflight.sh").is_file()
    assert not (out_dir / "run_worldmm_smvqa.sh").exists()


def test_launch_remote_dry_run_requires_worldmm_output_root_config(
    tmp_path: Path,
) -> None:
    # Given: WORLDMM_OUTPUT_ROOT is absent from the remote config template.
    config = tmp_path / "remote.yaml"
    _ = config.write_text(
        """runtime:
  location: remote
remote:
  bastion_host: ${BASTION_HOST}
  head_node: ${HEAD_NODE}
  data_root: ${SMVQA_DATA_ROOT}
  model_path: ${GEMMA_MODEL_PATH}
  execution_profile: teacher-oracle
  experiment_config: configs/spatial/exp_0005_teacher_oracle.example.json
""",
        encoding="utf-8",
    )

    # When: the launch dry-run is requested.
    result = run_cli(
        "launch-remote",
        "--profile",
        "teacher-oracle",
        "--experiment-config",
        "configs/spatial/exp_0005_teacher_oracle.example.json",
        "--dry-run",
        "--config",
        str(config),
        "--out",
        str(tmp_path / "remote_plan"),
    )

    # Then: the boundary reports the missing remote config value.
    assert result.returncode != 0
    assert "MissingRemoteConfig: WORLDMM_OUTPUT_ROOT" in result.stderr


def test_remote_example_uses_only_expected_remote_env_placeholders() -> None:
    # Given: the remote example config is the user-facing template.
    # When: the placeholder names are read.
    config_text = (ROOT / "configs/remote.example.yaml").read_text(encoding="utf-8")

    # Then: only Todo 9 env placeholders are present.
    placeholders = {part.split("}", 1)[0] for part in config_text.split("${")[1:]}
    assert placeholders == REMOTE_ENV_NAMES
