from __future__ import annotations

import os
import subprocess
from pathlib import Path

from worldmm_smvqa.config import load_config
from worldmm_smvqa.remote_plan import (
    DAG_STAGE_SCRIPT_NAME,
    DAG_SUBMIT_SCRIPT_NAME,
    write_remote_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def test_remote_plan_writes_shell_valid_cpu_gpu_dag(  # noqa: PLR0915
    tmp_path: Path,
) -> None:
    plan = tmp_path / "remote-plan"
    _ = write_remote_plan(
        load_config(ROOT / "configs/remote.example.yaml"),
        plan,
        {},
        submit=False,
    )

    submitter = plan / DAG_SUBMIT_SCRIPT_NAME
    runner = plan / DAG_STAGE_SCRIPT_NAME
    assert submitter.is_file()
    assert runner.is_file()
    for script in (submitter, runner):
        proof = subprocess.run(
            ["bash", "-n", str(script)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert proof.returncode == 0, proof.stderr

    submit_text = submitter.read_text(encoding="utf-8")
    assert 'WORLDMM_SMVQA_REMOTE_APPROVED:-}" != "1"' in submit_text
    assert "export WORLDMM_SMVQA_REMOTE_APPROVED" in submit_text
    assert '*/"$WORLDMM_RUN_ID")' in submit_text
    assert "WORLDMM_REMOTE_NODES:=10" in submit_text
    assert "WORLDMM_GPUS_PER_NODE:=8" in submit_text
    assert "WORLDMM_TEACHER_NODES:=$WORLDMM_REMOTE_NODES" in submit_text
    assert "WORLDMM_TRAIN_NODES:=$WORLDMM_REMOTE_NODES" in submit_text
    assert "WORLDMM_TRAIN_GPUS_PER_NODE:=$WORLDMM_GPUS_PER_NODE" in submit_text
    assert "WORLDMM_TRAIN_EPOCHS:=1" in submit_text
    assert "WORLDMM_TRAIN_BATCH_SIZE:=8" in submit_text
    assert "WORLDMM_TRAIN_HIDDEN_DIM:=32" in submit_text
    assert "WORLDMM_TRAIN_LEARNING_RATE:=0.001" in submit_text
    assert "WORLDMM_TRAIN_RESUME:=" in submit_text
    assert "WORLDMM_CPU_PARTITION:=cpu-prepro-queue" in submit_text
    assert "WORLDMM_GPU_PARTITION:=gpu-vtt-queue" in submit_text
    assert submit_text.count('"--dependency=afterok:$dependency"') == 1
    for stage in (
        "preflight_ingest",
        "teacher_extract",
        "merge_utility",
        "train_qa",
        "metrics_report",
    ):
        assert f"submit_stage {stage}" in submit_text

    runner_text = runner.read_text(encoding="utf-8")
    sensor_export = "export WORLDMM_SENSOR_FRAME_MANIFEST"
    assert sensor_export in runner_text
    assert runner_text.index(sensor_export) < runner_text.index(
        "build-memory --stage chunk",
    )
    assert "worldmm-smvqa preflight" in runner_text
    assert "worldmm_smvqa.worldmm.gcut3r_teacher validate-cache" in runner_text
    assert "-m worldmm_smvqa.teacher_materializer" in runner_text
    assert "--supervision-mode counterfactual" in runner_text
    assert "-m worldmm_smvqa.spatial_train train" in runner_text
    assert '--epochs "$WORLDMM_TRAIN_EPOCHS"' in runner_text
    assert '--batch-size "$WORLDMM_TRAIN_BATCH_SIZE"' in runner_text
    assert '--hidden-dim "$WORLDMM_TRAIN_HIDDEN_DIM"' in runner_text
    assert '--learning-rate "$WORLDMM_TRAIN_LEARNING_RATE"' in runner_text
    assert 'train_args+=(--resume "$WORLDMM_TRAIN_RESUME")' in runner_text
    assert '--ntasks="$teacher_world_size"' in runner_text
    assert '--ntasks-per-node="$WORLDMM_STAGE_GPUS_PER_NODE"' in runner_text
    assert '--cpus-per-task="$teacher_cpus_per_worker"' in runner_text
    assert "--gpus-per-task=1" in runner_text
    assert '--rank "$SLURM_PROCID"' in runner_text
    assert '--world-size "$teacher_world_size"' in runner_text
    assert '--out "$teacher_shard_root/rank-${rank_id}.jsonl"' in runner_text
    assert '--sensor-frame-manifest "$WORLDMM_SENSOR_FRAME_MANIFEST"' in runner_text
    assert "sensor-frame manifest is not a non-empty file" in runner_text
    assert "teacher shard mismatch" in runner_text
    assert "-m worldmm_smvqa.qa_transformers" in runner_text
    assert "worldmm-smvqa evaluate" in runner_text


def test_dag_submitter_chains_five_jobs_without_slurm(tmp_path: Path) -> None:
    plan = tmp_path / "remote-plan"
    _ = write_remote_plan(
        load_config(ROOT / "configs/remote.example.yaml"),
        plan,
        {},
        submit=False,
    )
    fake_sbatch = tmp_path / "fake-sbatch"
    calls = tmp_path / "sbatch.calls"
    counter = tmp_path / "sbatch.counter"
    _ = fake_sbatch.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_SBATCH_CALLS"
value=1000
if [ -f "$FAKE_SBATCH_COUNTER" ]; then
  value="$(cat "$FAKE_SBATCH_COUNTER")"
fi
value=$((value + 1))
printf '%s' "$value" > "$FAKE_SBATCH_COUNTER"
printf '%s;test-cluster\n' "$value"
""",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    output_root = tmp_path / "company-output" / "gate-1"
    env = {
        **os.environ,
        "FAKE_SBATCH_CALLS": str(calls),
        "FAKE_SBATCH_COUNTER": str(counter),
        "WORLDMM_SBATCH": str(fake_sbatch),
        "WORLDMM_REMOTE_REPO": str(tmp_path),
        "WORLDMM_RUN_ID": "gate-1",
        "WORLDMM_OUTPUT_ROOT": str(output_root),
        "WORLDMM_SMVQA_REMOTE_APPROVED": "1",
        "WORLDMM_TRAIN_NODES": "2",
        "WORLDMM_TRAIN_GPUS_PER_NODE": "4",
        "WORLDMM_TRAIN_TIME": "01:23:45",
    }
    result = subprocess.run(
        ["bash", str(plan / DAG_SUBMIT_SCRIPT_NAME)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    submitted = calls.read_text(encoding="utf-8").splitlines()
    assert len(submitted) == 5
    assert "--dependency=" not in submitted[0]
    for index, job_id in enumerate(range(1001, 1005), start=1):
        assert f"--dependency=afterok:{job_id}" in submitted[index]
    assert "--gpus-per-node" not in submitted[0]
    assert "--gpus-per-node=8" in submitted[1]
    assert "--gpus-per-node" not in submitted[2]
    assert "--nodes=2" in submitted[3]
    assert "--gpus-per-node=4" in submitted[3]
    assert "--time=01:23:45" in submitted[3]
    assert "--gpus-per-node" not in submitted[4]
    manifest = (output_root / "summary/dag_jobs.env").read_text(encoding="utf-8")
    assert "PREFLIGHT_JOB_ID=1001" in manifest
    assert "REPORT_JOB_ID=1005" in manifest

    repeated = subprocess.run(
        ["bash", str(plan / DAG_SUBMIT_SCRIPT_NAME)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert repeated.returncode != 0
    assert "DAG already submitted or submitting" in repeated.stderr
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 5


def test_dag_submitter_fails_before_submission_without_approval(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "remote-plan"
    _ = write_remote_plan(
        load_config(ROOT / "configs/remote.example.yaml"),
        plan,
        {},
        submit=False,
    )
    output_root = tmp_path / "company-output" / "no-approval"
    env = {
        **os.environ,
        "WORLDMM_REMOTE_REPO": str(tmp_path),
        "WORLDMM_RUN_ID": "no-approval",
        "WORLDMM_OUTPUT_ROOT": str(output_root),
        "WORLDMM_SBATCH": "/bin/false",
    }
    _ = env.pop("WORLDMM_SMVQA_REMOTE_APPROVED", None)

    result = subprocess.run(
        ["bash", str(plan / DAG_SUBMIT_SCRIPT_NAME)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "WORLDMM_SMVQA_REMOTE_APPROVED=1 is required" in result.stderr
    assert not output_root.exists()


def test_dag_submitter_rejects_output_root_outside_run_scope(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "remote-plan"
    _ = write_remote_plan(
        load_config(ROOT / "configs/remote.example.yaml"),
        plan,
        {},
        submit=False,
    )
    wrong_output_root = tmp_path / "company-output" / "wrong-run"
    env = {
        **os.environ,
        "WORLDMM_REMOTE_REPO": str(tmp_path),
        "WORLDMM_RUN_ID": "expected-run",
        "WORLDMM_OUTPUT_ROOT": str(wrong_output_root),
        "WORLDMM_SMVQA_REMOTE_APPROVED": "1",
        "WORLDMM_SBATCH": "/bin/false",
    }

    result = subprocess.run(
        ["bash", str(plan / DAG_SUBMIT_SCRIPT_NAME)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "WORLDMM_OUTPUT_ROOT must end /expected-run" in result.stderr
    assert not wrong_output_root.exists()
