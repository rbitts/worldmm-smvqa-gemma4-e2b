from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pydantic import TypeAdapter

from worldmm_smvqa.remote_plan import ExpectedOutputs

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ENV_NAMES = frozenset(
    {
        "SMVQA_DATA_ROOT",
        "GEMMA_MODEL_PATH",
        "WORLDMM_OUTPUT_ROOT",
        "BASTION_HOST",
        "HEAD_NODE",
        "REMOTE_JOB_LAUNCHER",
    },
)
REQUIRED_STAGES = (
    "prepare source manifests",
    "build 30s/30m chunks",
    "generate/load captions OCR object frame refs",
    "build WorldMM episodic semantic visual stores",
    "retrieve per QA under causal cutoff",
    "run Gemma 4 E2B QA",
    "evaluate official metrics",
    "write summary",
)


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in REMOTE_ENV_NAMES | {
        "WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST",
        "WORLDMM_SMVQA_REMOTE_APPROVED",
    }:
        _ = env.pop(name, None)
    return subprocess.run(
        ["uv", "run", "worldmm-smvqa", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launch_remote_dry_run_writes_full_plan_contract(tmp_path: Path) -> None:
    # Given: a remote full benchmark plan output directory.
    out_dir = tmp_path / "remote_plan"

    # When: launch-remote is run as a local dry-run.
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    # Then: deterministic artifacts describe every remote stage and copyback rule.
    assert result.returncode == 0, result.stderr
    script = out_dir / "run_worldmm_smvqa.sh"
    expected = out_dir / "expected_outputs.json"
    policy = out_dir / "copyback_policy.txt"
    assert script.is_file()
    assert expected.is_file()
    assert policy.is_file()

    script_text = script.read_text(encoding="utf-8")
    for stage in REQUIRED_STAGES:
        assert stage in script_text
    assert "ssh " not in script_text
    assert "sbatch " not in script_text
    assert "torchrun " not in script_text
    assert "$WORLDMM_OUTPUT_ROOT" in script_text
    assert "$SMVQA_DATA_ROOT" in script_text
    assert "$GEMMA_MODEL_PATH" in script_text

    manifest = TypeAdapter(ExpectedOutputs).validate_json(
        expected.read_text(encoding="utf-8"),
    )
    assert manifest["remote_job_reference"] == "${REMOTE_JOB_ID_OR_PROCESS_REF}"
    assert manifest["metrics"] == ["Ans-F1", "QA-Acc", "QA-MRR"]
    for value in manifest["outputs"].values():
        assert isinstance(value, str)
        assert value.startswith("$WORLDMM_OUTPUT_ROOT/")

    policy_text = policy.read_text(encoding="utf-8")
    assert "metrics/logs/plots/summaries/small samples" in policy_text
    assert "no full datasets" in policy_text
    assert "no model weights" in policy_text
    assert "no checkpoints" in policy_text


def test_launch_remote_default_ddp_launcher_tokenizes_as_argv(
    tmp_path: Path,
) -> None:
    # Given: the generated remote script uses the default multi-word launcher.
    out_dir = tmp_path / "remote_plan"
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )
    assert result.returncode == 0, result.stderr
    script_text = (out_dir / "run_worldmm_smvqa.sh").read_text(encoding="utf-8")

    # When: Bash expands the launcher setup without WORLDMM_DDP_LAUNCHER set.
    launcher_setup = (
        "set -euo pipefail\n"
        ': "${WORLDMM_DDP_LAUNCHER:=python -m torch.distributed.run}"\n'
        'read -r -a worldmm_ddp_launcher_argv <<< "$WORLDMM_DDP_LAUNCHER"\n'
        'printf "%s\\n" "${worldmm_ddp_launcher_argv[@]}"'
    )
    proof = subprocess.run(
        ["bash", "-c", launcher_setup],
        env={},
        text=True,
        capture_output=True,
        check=False,
    )

    # Then: the generated script calls argv entries, not one quoted executable.
    assert proof.returncode == 0, proof.stderr
    assert proof.stdout.splitlines() == ["python", "-m", "torch.distributed.run"]
    assert (
        'read -r -a worldmm_ddp_launcher_argv <<< "$WORLDMM_DDP_LAUNCHER"'
        in script_text
    )
    assert '"${worldmm_ddp_launcher_argv[@]}" --nnodes' in script_text
    assert '"$WORLDMM_DDP_LAUNCHER" --nnodes' not in script_text
    assert "eval " not in script_text


def test_launch_remote_submit_requires_explicit_env_approval(tmp_path: Path) -> None:
    # Given: submit was requested without the required approval env value.
    out_dir = tmp_path / "remote_plan"

    # When: launch-remote tries to submit.
    result = run_cli(
        "launch-remote",
        "--submit",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    # Then: the typed approval guard fails before any plan or remote process starts.
    assert result.returncode != 0
    assert "ExplicitApprovalRequired" in result.stderr
    assert not out_dir.exists()
    combined = f"{result.stdout}\n{result.stderr}"
    assert "ssh " not in combined
    assert "REMOTE_JOB_LAUNCHER" not in combined


def test_launch_remote_dry_run_replaces_stale_artifacts(tmp_path: Path) -> None:
    # Given: a stale artifact from an interrupted earlier run.
    out_dir = tmp_path / "remote_plan"
    out_dir.mkdir()
    stale_script = out_dir / "run_worldmm_smvqa.sh"
    _ = stale_script.write_text("stale\n", encoding="utf-8")

    # When: the dry-run is repeated.
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    # Then: generated files are replaced by current deterministic content.
    assert result.returncode == 0, result.stderr
    assert stale_script.read_text(encoding="utf-8") != "stale\n"
    assert "prepare source manifests" in stale_script.read_text(encoding="utf-8")


def test_launch_remote_config_requires_remote_placeholders(tmp_path: Path) -> None:
    # Given: a malformed remote config missing WORLDMM_OUTPUT_ROOT.
    config = tmp_path / "missing-output-root.yaml"
    _ = config.write_text(
        """runtime:
  location: remote
remote:
  bastion_host: ${BASTION_HOST}
  head_node: ${HEAD_NODE}
  job_launcher: ${REMOTE_JOB_LAUNCHER}
  data_root: ${SMVQA_DATA_ROOT}
  model_path: ${GEMMA_MODEL_PATH}
""",
        encoding="utf-8",
    )

    # When: a dry-run plan is requested.
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        str(config),
        "--out",
        str(tmp_path / "remote_plan"),
    )

    # Then: the boundary reports a typed missing remote config error.
    assert result.returncode != 0
    assert "MissingRemoteConfig: WORLDMM_OUTPUT_ROOT" in result.stderr
