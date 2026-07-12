from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import cast

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
        ["uv", "run", "worldmm-smvqa", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launch_remote_writes_only_phased_dag(tmp_path: Path) -> None:
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
    submitter = out_dir / "submit_worldmm_smvqa_dag.sh"
    runner = out_dir / "run_worldmm_smvqa_stage.sh"
    assert submitter.is_file()
    assert runner.is_file()
    assert not (out_dir / "run_worldmm_smvqa.sh").exists()
    for script in (submitter, runner):
        proof = subprocess.run(
            ["bash", "-n", str(script)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert proof.returncode == 0, proof.stderr
    expected = cast(
        "dict[str, object]",
        json.loads((out_dir / "expected_outputs.json").read_text()),
    )
    outputs = cast("dict[str, str]", expected["outputs"])
    conditional_outputs = cast(
        "dict[str, str]",
        expected["conditional_outputs"],
    )
    assert outputs["final_report"].endswith("/summary/final_report.md")
    assert outputs["spatial_infer_contract"].endswith(
        "/diagnostics/spatial_infer_contract.txt",
    )
    assert {
        "deployed_code_inventory",
        "episodic_memory",
        "finalization_inputs",
        "preflight_completion",
        "preflight_dag_job_manifest",
        "python_runtime_fingerprint",
        "python_runtime_inventory",
        "qa_resume_manifest",
        "retrieval_memory_hashes",
        "retrieval_memory_inputs",
        "semantic_memory",
        "sensed_sources",
        "visual_memory",
    } <= outputs.keys()
    assert "preflight_teacher_cache_report" not in outputs
    assert {
        "preflight_teacher_cache_report_cache_mode_only",
        "stage_failures_if_any",
    } <= conditional_outputs.keys()
    artifact_section = (ROOT / "HANDOFF.md").read_text(encoding="utf-8").split(
        "## Expected Remote Artifacts",
        maxsplit=1,
    )[1].split("## Failure Triage", maxsplit=1)[0]
    for path in (*outputs.values(), *conditional_outputs.values()):
        suffix = path.removeprefix("$WORLDMM_OUTPUT_ROOT/")
        assert suffix in artifact_section, f"HANDOFF omits generated artifact: {suffix}"
    assert "legacy single-job compatibility" not in result.stdout


def test_launch_remote_submit_requires_explicit_env_approval(tmp_path: Path) -> None:
    out_dir = tmp_path / "remote_plan"

    result = run_cli(
        "launch-remote",
        "--submit",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    assert result.returncode != 0
    assert "ExplicitApprovalRequired" in result.stderr
    assert not out_dir.exists()
    assert "ssh " not in f"{result.stdout}\n{result.stderr}"


def test_launch_remote_approved_plan_prints_dag_command(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["WORLDMM_SMVQA_REMOTE_APPROVED"] = "1"

    result = subprocess.run(
        [
            "uv",
            "run",
            "worldmm-smvqa",
            "launch-remote",
            "--submit",
            "--config",
            "configs/remote.example.yaml",
            "--out",
            str(tmp_path / "remote_plan"),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "WORLDMM_SMVQA_REMOTE_APPROVED=1 "
        "WORLDMM_DAG_PHASE=preflight "
        "bash remote-plan/submit_worldmm_smvqa_dag.sh"
    ) in result.stdout


def test_launch_remote_removes_stale_legacy_runner(tmp_path: Path) -> None:
    out_dir = tmp_path / "remote_plan"
    out_dir.mkdir()
    stale = out_dir / "run_worldmm_smvqa.sh"
    _ = stale.write_text("unsafe legacy runner\n", encoding="utf-8")

    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    assert result.returncode == 0, result.stderr
    assert not stale.exists()


def test_launch_remote_config_requires_remote_placeholders(tmp_path: Path) -> None:
    config = tmp_path / "missing-output-root.yaml"
    _ = config.write_text(
        """runtime:
  location: remote
remote:
  bastion_host: ${BASTION_HOST}
  head_node: ${HEAD_NODE}
  data_root: ${SMVQA_DATA_ROOT}
  model_path: ${GEMMA_MODEL_PATH}
""",
        encoding="utf-8",
    )

    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        str(config),
        "--out",
        str(tmp_path / "remote_plan"),
    )

    assert result.returncode != 0
    assert "MissingRemoteConfig: WORLDMM_OUTPUT_ROOT" in result.stderr
