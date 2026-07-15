from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import sysconfig
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from worldmm_smvqa.attestation import (
    AttestationError,
    b64url_encode,
    canonicalize,
    loads_strict,
    signing_bytes,
    with_payload_sha256,
)

if TYPE_CHECKING:
    from pydantic import JsonValue
from worldmm_smvqa.remote_plan import (
    TEACHER_ORACLE_DOWNSTREAM_SUBMIT_SCRIPT_NAME,
    TEACHER_ORACLE_PREFLIGHT_SUBMIT_SCRIPT_NAME,
    TEACHER_ORACLE_PROVIDER_GATE_SUBMIT_SCRIPT_NAME,
    TEACHER_ORACLE_STAGE_SCRIPT_NAME,
    ExperimentGraphV1,
    ResourceSpecV1,
)
from worldmm_smvqa.remote_script import (
    dag_stage_script_text,
    dag_submit_script_text,
    teacher_oracle_downstream_submit_script_text,
    teacher_oracle_preflight_submit_script_text,
    teacher_oracle_provider_gate_submit_script_text,
    teacher_oracle_stage_script_text,
)

ROOT = Path(__file__).resolve().parents[1]
DAG_SUBMIT_SCRIPT_NAME = "submit_worldmm_smvqa_dag.sh"
DAG_STAGE_SCRIPT_NAME = "run_worldmm_smvqa_stage.sh"


def _assert_embedded_python_compiles(script: Path) -> None:
    block: list[str] = []
    start: int | None = None
    waiting_for_command_end = False
    count = 0
    for line_number, line in enumerate(
        script.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if start is None and "<<'PY'" in line:
            start = line_number + 1
            block = []
            waiting_for_command_end = line.rstrip().endswith("\\")
        elif start is not None and line == "PY":
            _ = compile("\n".join(block) + "\n", f"{script}:{start}", "exec")
            count += 1
            start = None
        elif start is not None and waiting_for_command_end:
            waiting_for_command_end = line.rstrip().endswith("\\")
        elif start is not None:
            block.append(line)
    assert start is None
    assert count > 0


def _write_legacy_renderer_scripts(plan: Path) -> None:
    plan.mkdir(parents=True)
    submitter = plan / "submit_worldmm_smvqa_dag.sh"
    runner = plan / "run_worldmm_smvqa_stage.sh"
    _ = submitter.write_text(dag_submit_script_text(), encoding="utf-8")
    submitter.chmod(0o755)
    _ = runner.write_text(dag_stage_script_text(), encoding="utf-8")
    runner.chmod(0o755)


def test_remote_plan_writes_shell_valid_cpu_gpu_dag(  # noqa: PLR0915
    tmp_path: Path,
) -> None:
    plan = tmp_path / "remote-plan"
    _write_legacy_renderer_scripts(plan)

    submitter = plan / "submit_worldmm_smvqa_dag.sh"
    runner = plan / "run_worldmm_smvqa_stage.sh"
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
        _assert_embedded_python_compiles(script)

    submit_text = submitter.read_text(encoding="utf-8")
    assert 'WORLDMM_SMVQA_REMOTE_APPROVED:-}" != "1"' in submit_text
    assert "export WORLDMM_SMVQA_REMOTE_APPROVED" in submit_text
    assert "WORLDMM_RUN_ID:=dag-" not in submit_text
    assert "WORLDMM_RUN_ID must be pinned before preflight" in submit_text
    assert "replace the placeholder WORLDMM_RUN_ID before preflight" in submit_text
    assert ".env.worldmm is required" in submit_text
    assert ".env.worldmm changed pinned WORLDMM_RUN_ID" in submit_text
    assert '"--job-name=worldmm-${WORLDMM_RUN_ID}-${stage}"' in submit_text
    assert "sbatch export value has comma/newline" in submit_text
    assert "preflight is limited to exactly 1 CPU node" in submit_text
    assert 'validate_time_limit "$WORLDMM_TEACHER_TIME"' in submit_text
    assert 'validate_time_limit "$WORLDMM_TRAIN_TIME"' in submit_text
    assert 'validate_gpu_memory "$WORLDMM_TEACHER_MEM"' in submit_text
    assert 'validate_gpu_memory "$WORLDMM_TRAIN_MEM"' in submit_text
    assert '*/"$WORLDMM_RUN_ID")' in submit_text
    assert "WORLDMM_REMOTE_NODES:=10" in submit_text
    assert "WORLDMM_GPUS_PER_NODE:=8" in submit_text
    assert "WORLDMM_REMOTE_NODES:=1" in submit_text
    assert "WORLDMM_GPUS_PER_NODE:=1" in submit_text
    assert "WORLDMM_DAG_PHASE:=preflight" in submit_text
    assert "WORLDMM_EXECUTION_PROFILE:=probe" in submit_text
    assert "WORLDMM_APPROVAL_FILE is required for run phase" in submit_text
    assert "--export=ALL" not in submit_text
    assert '"--export=$stage_exports"' in submit_text
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
    assert "--dependency=${dependency_kind}:$dependency" in submit_text
    assert "afterok|afterany" in submit_text
    assert "--no-requeue" in submit_text
    assert "submission-unknown-before-sbatch" in submit_text
    assert '"event":"submission-reconciled"' in submit_text
    assert "--comment=worldmm:${WORLDMM_RUN_ID}:${WORLDMM_DAG_PHASE}:${stage}" in (
        submit_text
    )
    assert "--reconcile-unknown-sbatch" in submit_text
    assert "ambiguous Slurm reconciliation" in submit_text
    assert '"kind": "SubmissionReconciliationV1"' in submit_text
    assert "os.O_EXCL" in submit_text
    assert "os.fsync(fd)" in submit_text
    assert "cancellation is forbidden for gate or terminal stage" in submit_text
    assert '"scientific":"not_decidable"' in submit_text
    assert '"$SCANCEL" "${submitted_job_ids[@]}"; then' in submit_text
    for stage in (
        "preflight_ingest",
        "teacher_extract",
        "merge_materialize",
        "train",
        "build_memory",
        "student_infer_retrieve",
        "qa",
        "metrics_report",
    ):
        assert f"submit_stage {stage}" in submit_text

    runner_text = runner.read_text(encoding="utf-8")
    assert not any(line.endswith("=\\") for line in runner_text.splitlines())
    assert not any(line.endswith("=\\") for line in submit_text.splitlines())
    sensor_export = "export WORLDMM_SENSOR_FRAME_MANIFEST"
    assert sensor_export in runner_text
    assert runner_text.index(sensor_export) < runner_text.index(
        "build-memory --stage chunk",
    )
    assert "worldmm-smvqa preflight" in runner_text
    assert "worldmm_smvqa.worldmm.gcut3r_teacher validate-cache" in runner_text
    assert "-m worldmm_smvqa.teacher_materializer" in runner_text
    assert "--supervision-mode counterfactual" not in runner_text
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
    assert '--world-size "$world_size"' in runner_text
    assert '--out "$shard_root/rank-${rank_id}.jsonl"' in runner_text
    assert '--sensor-frame-manifest "$WORLDMM_SENSOR_FRAME_MANIFEST"' in runner_text
    assert "sensor-frame manifest is not a non-empty file" in runner_text
    assert "teacher shard mismatch" in runner_text
    assert "WORLDMM_SPATIAL_INFER_EXE" in runner_text
    assert "--contract-version" in runner_text
    assert "--self-test" in runner_text
    assert "worldmm-spatial-infer-v1:self-test-ok" in runner_text
    assert "materialize a self-contained model tree" in runner_text
    assert "AutoProcessor.from_pretrained" in runner_text
    assert "model index references missing/empty shards" in runner_text
    assert "root not in resolved.parents" in runner_text
    assert "worldmm-spatial-infer-v1" in runner_text
    assert '"production_ready": True' in runner_text
    assert '"result_class": "student"' in runner_text
    assert '--input "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json"' in runner_text
    assert "--evidence-lane student" in runner_text
    assert "--evidence-lineage" in runner_text
    assert '--typed-memory "$WORLDMM_OUTPUT_ROOT/memory/typed_memory.jsonl"' in (
        runner_text
    )
    assert "--inference-manifest" in runner_text
    assert "--inference-sources" in runner_text
    assert '--inference-producer "$WORLDMM_SPATIAL_INFER_EXE"' in runner_text
    assert "--model-fingerprint" in runner_text
    assert "--frame-assets-manifest" in runner_text
    assert "--lineage-config" in runner_text
    assert "--sensor-frame-manifest" in runner_text
    assert "--memory-manifest" in runner_text
    assert '"$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json"' in runner_text
    assert '"inference_manifest_sha256": sha256(inference_manifest)' in runner_text
    assert '"typed_memory_sha256": sha256(typed_memory)' in runner_text
    assert "memory_artifact_hashes(memory_manifest)" in runner_text
    assert "memory_inputs.sha256" in runner_text
    assert 'sha256sum --check --status "$memory_inputs"' in runner_text
    assert '"$WORLDMM_OUTPUT_ROOT/memory/episodic.jsonl"' in runner_text
    assert '"$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv/semantic.jsonl"' in runner_text
    assert '"$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv/visual.jsonl"' in runner_text
    assert "WORLDMM_QA_EVIDENCE_INPUT" not in runner_text
    assert "WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW" in runner_text
    assert "--byte-budget-per-window" in runner_text
    assert '"byte_budget_per_window": byte_budget_per_window' in runner_text
    assert '"window_count"' in runner_text
    assert '"max_window_bytes"' in runner_text
    assert '"window_seconds": DEFAULT_TYPED_MEMORY_WINDOW_SECONDS' in runner_text
    assert "validate_typed_memory_artifact" in runner_text
    assert '--sources "$sensed_sources"' in runner_text
    assert "--sources-sha256" in runner_text
    assert "--frame-assets-sha256" in runner_text
    assert "--producer-sha256" in runner_text
    assert '"sources_sha256": sha256(sources_path)' in runner_text
    assert '"frame_assets_sha256": sha256(frame_assets)' in runner_text
    assert '"producer_sha256": sha256(producer)' in runner_text
    assert "-u WORLDMM_STUDENT_SUPERVISION_INPUT" in runner_text
    assert '--frame-root "$SMVQA_FRAME_ROOT"' in runner_text
    assert "--require-frames" in runner_text
    assert "stage environment differs from preflight env contract" in runner_text
    assert '"python_runtime": python_runtime()' in runner_text
    assert '"effective_teacher_resources": effective_teacher_resources' in (runner_text)
    assert "approved Python runtime content changed" in runner_text
    assert "approved Python runtime file inventory changed" in runner_text
    assert "approved Python base runtime content changed" in runner_text
    assert "approved Python base runtime file inventory changed" in runner_text
    assert "python_runtime.loader.sha256" in runner_text
    assert "interpreter loader/shared-library closure changed" in runner_text
    assert 'ldd "$python_runtime_root/bin/python"' in runner_text
    assert "SLURM_GPUS_ON_NODE is required for GPU allocation proof" in runner_text
    assert "Python runtime has unsafe directory/dangling symlink" in runner_text
    assert "find . -xtype f -print0" in runner_text
    assert "python_runtime.files.sha256" in runner_text
    assert "approved code snapshot file inventory changed" in runner_text
    assert "approved code snapshot contains a symlink" in runner_text
    assert "deployed_code.files.sha256" in runner_text
    assert 'sha256sum --check --status "$preflight_inputs"' in runner_text
    assert 'sha256sum --check --status "$frame_assets"' in runner_text
    assert "stage already started for this run" in runner_text
    assert "stage.${WORLDMM_STAGE}.failure.json" in runner_text
    assert "SLURM_GPUS_ON_NODE" in runner_text
    assert "teacher cache sensor observation mismatch" in runner_text
    assert "frame_assets.files.sha256" in runner_text
    assert "gemma_model.files.sha256" in runner_text
    assert "memory_model.files.sha256" in runner_text
    assert "sources=sources" in runner_text
    assert "sensor_records=sensor_records" in runner_text
    assert "qa/completed.json" in runner_text
    assert "QA outputs changed after QA stage completion" in runner_text
    assert "QA resume manifest no longer matches current QA inputs" in runner_text
    lineage_assignment = (
        '"evidence_lineage_sha256": artifact_hashes["evidence_lineage_sha256"]'
    )
    assert lineage_assignment in runner_text
    assert "diagnostics/frame_assets.sha256" in runner_text
    assert 'find "$code_snapshot" -type f -print0' in runner_text
    assert "find . -type f -print0 | sort -z | xargs -0 sha256sum" in runner_text
    assert "WORLDMM_OUTPUT_ROOT/code_snapshot" in runner_text
    assert "-m worldmm_smvqa.qa_transformers" in runner_text
    assert "worldmm-smvqa evaluate" in runner_text
    assert 'experiment_label = "PROBE" if is_probe else "E1"' in runner_text
    assert '"execution_profile": execution_profile' in runner_text
    assert "summary/run_identity.json" in runner_text
    assert "summary/remote_manifest.json" in runner_text
    assert "summary/final_report.md" in runner_text
    summary_assignment = 'summary_path="$WORLDMM_OUTPUT_ROOT/summary/summary.txt"'
    finalization_assignment = 'finalization_inputs="$WORLDMM_OUTPUT_ROOT/summary"'
    assert runner_text.index(summary_assignment) < runner_text.index(
        finalization_assignment,
    )
    assert runner_text.index("manifest_temporary.replace(manifest_path)") > (
        runner_text.index("report_temporary.replace(report_path)")
    )
    assert '"prompt_sha256": prompt_sha256' in runner_text
    assert '"code_sha256": code_sha256' in runner_text


def test_dag_submitter_requires_preflight_then_chains_run_jobs_without_slurm(  # noqa: PLR0915
    tmp_path: Path,
) -> None:
    plan = tmp_path / "remote-plan"
    _write_legacy_renderer_scripts(plan)
    _write_safe_env(tmp_path)
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
    python_path = tmp_path / ".venv/bin/python"
    python_path.parent.mkdir(parents=True)
    python_path.symlink_to(Path(sys.executable).resolve())
    fixture = tmp_path / "fixture"
    frame_root = fixture / "frames"
    gemma_model = tmp_path / "models/gemma"
    memory_model = tmp_path / "models/qwen"
    for directory in (frame_root, gemma_model, memory_model):
        directory.mkdir(parents=True)
    infer_exe = tmp_path / "spatial-infer"
    supervision = tmp_path / "supervision.jsonl"
    teacher_cache = tmp_path / "teacher-cache.jsonl"
    for path in (infer_exe, supervision, teacher_cache):
        _ = path.write_text("contract\n", encoding="utf-8")
    infer_exe.chmod(0o700)
    output_root = tmp_path / "company-output" / "gate-1"
    output_root.parent.mkdir(parents=True)
    env = {
        **os.environ,
        "FAKE_SBATCH_CALLS": str(calls),
        "FAKE_SBATCH_COUNTER": str(counter),
        "WORLDMM_SBATCH": str(fake_sbatch),
        "WORLDMM_REMOTE_REPO": str(tmp_path),
        "WORLDMM_RUN_ID": "gate-1",
        "WORLDMM_OUTPUT_ROOT": str(output_root),
        "WORLDMM_APPROVED_DATA_PREFIX": str(tmp_path),
        "WORLDMM_APPROVED_REPO_PREFIX": str(tmp_path),
        "WORLDMM_APPROVED_OUTPUT_PREFIX": str(output_root.parent),
        "WORLDMM_SMVQA_REMOTE_APPROVED": "1",
        "WORLDMM_EXECUTION_PROFILE": "full",
        "WORLDMM_REMOTE_NODES": "2",
        "WORLDMM_GPUS_PER_NODE": "4",
        "WORLDMM_TRAIN_NODES": "2",
        "WORLDMM_TRAIN_GPUS_PER_NODE": "4",
        "WORLDMM_TRAIN_TIME": "01:23:45",
        "SMVQA_DATA_ROOT": str(fixture),
        "SMVQA_FRAME_ROOT": str(frame_root),
        "GEMMA_MODEL_PATH": str(gemma_model),
        "WORLDMM_MEMORY_MODEL_PATH": str(memory_model),
        "WORLDMM_SPATIAL_INFER_EXE": str(infer_exe),
        "WORLDMM_STUDENT_SUPERVISION_INPUT": str(supervision),
        "WORLDMM_TEACHER_CACHE_INPUT": str(teacher_cache),
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
    assert len(submitted) == 1
    assert "--dependency=" not in submitted[0]
    assert "--gpus-per-node" not in submitted[0]
    preflight_manifest = (output_root / "summary/dag_jobs.preflight.env").read_text(
        encoding="utf-8"
    )
    assert "PREFLIGHT_JOB_ID=1001" in preflight_manifest

    diagnostics = output_root / "diagnostics"
    diagnostics.mkdir()
    runtime_marker = tmp_path / ".venv/runtime-marker"
    _ = runtime_marker.write_text("approved runtime\n", encoding="utf-8")
    _write_python_runtime_manifests(tmp_path / ".venv", diagnostics)
    fingerprinted = tmp_path / "fingerprinted-input"
    _ = fingerprinted.write_text("approved input\n", encoding="utf-8")
    digest = hashlib.sha256(fingerprinted.read_bytes()).hexdigest()
    _ = (diagnostics / "preflight_inputs.sha256").write_text(
        f"{digest}  {fingerprinted}\n",
        encoding="utf-8",
    )
    _ = (diagnostics / "frame_assets.sha256").write_text(
        f"{digest}  {fingerprinted}\n",
        encoding="utf-8",
    )
    for name in ("gemma_model.sha256", "memory_model.sha256"):
        _ = (diagnostics / name).write_text(
            f"{digest}  {fingerprinted}\n",
            encoding="utf-8",
        )
    empty_inventory = f"{hashlib.sha256(b'').hexdigest()}  -\n"
    for name in ("gemma_model.files.sha256", "memory_model.files.sha256"):
        _ = (diagnostics / name).write_text(empty_inventory, encoding="utf-8")
    sensed_frames = output_root / "inference_inputs/frames"
    sensed_frames.mkdir(parents=True)
    _ = (diagnostics / "frame_assets.files.sha256").write_text(
        empty_inventory,
        encoding="utf-8",
    )
    _ = (diagnostics / "preflight.completed").write_text(
        "run_id=gate-1\n",
        encoding="utf-8",
    )
    snapshot_plan = output_root / "code_snapshot/remote-plan"
    snapshot_runner = snapshot_plan / DAG_STAGE_SCRIPT_NAME
    snapshot_submitter = snapshot_plan / DAG_SUBMIT_SCRIPT_NAME
    snapshot_runner.parent.mkdir(parents=True)
    _ = snapshot_runner.write_bytes((plan / DAG_STAGE_SCRIPT_NAME).read_bytes())
    snapshot_runner.chmod(0o700)
    _ = snapshot_submitter.write_bytes((plan / DAG_SUBMIT_SCRIPT_NAME).read_bytes())
    snapshot_submitter.chmod(0o700)
    env_contract = diagnostics / "env_contract.json"
    _ = env_contract.write_text(
        json.dumps(
            {
                "approved_prefixes": {
                    "WORLDMM_APPROVED_DATA_PREFIX": str(tmp_path.resolve()),
                    "WORLDMM_APPROVED_OUTPUT_PREFIX": str(output_root.parent.resolve()),
                    "WORLDMM_APPROVED_REPO_PREFIX": str(tmp_path.resolve()),
                },
                "byte_budget_per_window": 4096,
                "effective_teacher_resources": {
                    "cpus_per_task": 32,
                    "gpus_per_node": 0,
                    "memory": "128G",
                    "nodes": 1,
                    "partition": "cpu-prepro-queue",
                    "time": "02:00:00",
                },
                "data_root": str(fixture.resolve()),
                "frame_root": str(frame_root.resolve()),
                "gemma_model_path": str(gemma_model.resolve()),
                "memory_model_path": str(memory_model.resolve()),
                "profile": "full",
                "python_runtime": _python_runtime(python_path),
                "resources": {
                    "WORLDMM_CPU_PARTITION": "cpu-prepro-queue",
                    "WORLDMM_GPU_PARTITION": "gpu-vtt-queue",
                    "WORLDMM_MATERIALIZE_CPUS": "64",
                    "WORLDMM_MATERIALIZE_MEM": "256G",
                    "WORLDMM_MATERIALIZE_NODES": "1",
                    "WORLDMM_MATERIALIZE_TIME": "06:00:00",
                    "WORLDMM_PREFLIGHT_CPUS": "32",
                    "WORLDMM_PREFLIGHT_MEM": "128G",
                    "WORLDMM_PREFLIGHT_NODES": "1",
                    "WORLDMM_PREFLIGHT_TIME": "02:00:00",
                    "WORLDMM_REPORT_CPUS": "32",
                    "WORLDMM_REPORT_MEM": "128G",
                    "WORLDMM_REPORT_NODES": "1",
                    "WORLDMM_REPORT_TIME": "02:00:00",
                    "WORLDMM_TEACHER_CPUS": "64",
                    "WORLDMM_TEACHER_GPUS_PER_NODE": "4",
                    "WORLDMM_TEACHER_MEM": "0",
                    "WORLDMM_TEACHER_NODES": "2",
                    "WORLDMM_TEACHER_TIME": "12:00:00",
                    "WORLDMM_TRAIN_CPUS": "64",
                    "WORLDMM_TRAIN_GPUS_PER_NODE": "4",
                    "WORLDMM_TRAIN_MEM": "0",
                    "WORLDMM_TRAIN_NODES": "2",
                    "WORLDMM_TRAIN_TIME": "01:23:45",
                },
                "run_fixture": str(fixture.resolve()),
                "run_id": "gate-1",
                "schema_version": 1,
                "spatial_infer_exe": str(infer_exe.resolve()),
                "student_supervision": str(supervision.resolve()),
                "teacher_mode": "cache",
                "teacher_path": str(teacher_cache.resolve()),
                "train_batch_size": 8,
                "train_epochs": 1,
                "train_hidden_dim": 32,
                "train_learning_rate": 0.001,
                "train_resume": None,
            },
        ),
        encoding="utf-8",
    )
    preflight_inputs_digest = hashlib.sha256(
        (diagnostics / "preflight_inputs.sha256").read_bytes(),
    ).hexdigest()
    env_contract_digest = hashlib.sha256(env_contract.read_bytes()).hexdigest()
    approval = tmp_path / "approval.json"
    _ = approval.write_text(
        json.dumps(
            {
                "approved": True,
                "approver": "reviewer@example.com",
                "gpus_per_node": 4,
                "env_contract_sha256": env_contract_digest,
                "nodes": 2,
                "profile": "full",
                "preflight_inputs_sha256": preflight_inputs_digest,
                "run_id": "gate-1",
                "teacher_gpus_per_node": 0,
                "teacher_nodes": 1,
                "train_gpus_per_node": 4,
                "train_nodes": 2,
            },
        ),
        encoding="utf-8",
    )
    approval.chmod(0o600)
    env.update(
        {
            "WORLDMM_APPROVAL_FILE": str(approval),
            "WORLDMM_APPROVER": "reviewer@example.com",
            "WORLDMM_DAG_PHASE": "run",
        },
    )
    shared_run = subprocess.run(
        ["bash", str(plan / DAG_SUBMIT_SCRIPT_NAME)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert shared_run.returncode != 0
    assert "run phase must use approved snapshot submitter" in shared_run.stderr
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1
    _ = runtime_marker.write_text("changed runtime\n", encoding="utf-8")
    drifted_runtime_content = subprocess.run(
        ["bash", str(snapshot_submitter)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert drifted_runtime_content.returncode != 0
    assert "approved Python runtime content changed" in (drifted_runtime_content.stderr)
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1
    _ = runtime_marker.write_text("approved runtime\n", encoding="utf-8")
    site_packages = tmp_path / ".venv/lib/python3.14/site-packages"
    site_packages.mkdir(parents=True)
    runtime_drift = site_packages / "sitecustomize.pyc"
    _ = runtime_drift.write_bytes(b"sourceless-sitecustomize\0")
    drifted_runtime = subprocess.run(
        ["bash", str(snapshot_submitter)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert drifted_runtime.returncode != 0
    assert "approved Python runtime file inventory changed" in (drifted_runtime.stderr)
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1
    runtime_drift.unlink()
    external_package = tmp_path / "external-package"
    external_package.mkdir()
    package_payload = external_package / "__init__.py"
    _ = package_payload.write_text("approved = True\n", encoding="utf-8")
    package_link = site_packages / "linked_package"
    package_link.symlink_to(external_package, target_is_directory=True)
    _ = package_payload.write_text("approved = False\n", encoding="utf-8")
    linked_runtime = subprocess.run(
        ["bash", str(snapshot_submitter)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert linked_runtime.returncode != 0
    assert "Python runtime has unsafe directory/dangling symlink" in (
        linked_runtime.stderr
    )
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1
    package_link.unlink()
    drifted_infer = tmp_path / "spatial-infer-drifted"
    _ = drifted_infer.write_text("changed contract\n", encoding="utf-8")
    drifted_infer.chmod(0o700)
    env["WORLDMM_SPATIAL_INFER_EXE"] = str(drifted_infer)
    drifted = subprocess.run(
        ["bash", str(snapshot_submitter)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert drifted.returncode != 0
    assert "current environment does not match preflight" in drifted.stderr
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1
    env["WORLDMM_SPATIAL_INFER_EXE"] = str(infer_exe)
    env["WORLDMM_TRAIN_TIME"] = "09:59:59"
    resource_drift = subprocess.run(
        ["bash", str(snapshot_submitter)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert resource_drift.returncode != 0
    assert "current environment does not match preflight" in resource_drift.stderr
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1
    env["WORLDMM_TRAIN_TIME"] = "01:23:45"
    result = subprocess.run(
        ["bash", str(snapshot_submitter)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    submitted = calls.read_text(encoding="utf-8").splitlines()
    assert len(submitted) == 8
    assert "--dependency=" not in submitted[1]
    assert "--gpus-per-node" not in submitted[1]
    for index, job_id in enumerate(range(1002, 1008), start=2):
        assert f"--dependency=afterok:{job_id}" in submitted[index]
        assert "--kill-on-invalid-dep=yes" in submitted[index]
    assert "--gpus-per-node" not in submitted[2]
    assert "--nodes=2" in submitted[3]
    assert "--gpus-per-node=4" in submitted[3]
    assert "--time=01:23:45" in submitted[3]
    assert "--nodes=2" in submitted[4]
    assert "--gpus-per-node=4" in submitted[4]
    assert "--nodes=1" in submitted[5]
    assert "--gpus-per-node=1" in submitted[5]
    assert "--nodes=2" in submitted[6]
    assert "--gpus-per-node=4" in submitted[6]
    assert "--gpus-per-node" not in submitted[7]
    manifest = (output_root / "summary/dag_jobs.env").read_text(encoding="utf-8")
    assert "APPROVAL_SHA256=" in manifest
    assert "TRAIN_JOB_ID=1004" in manifest
    assert "BUILD_MEMORY_JOB_ID=1005" in manifest
    assert "STUDENT_INFER_RETRIEVE_JOB_ID=1006" in manifest
    assert "QA_JOB_ID=1007" in manifest
    assert "REPORT_JOB_ID=1008" in manifest

    repeated = subprocess.run(
        ["bash", str(snapshot_submitter)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert repeated.returncode != 0
    assert "DAG phase already submitted or submitting" in repeated.stderr
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 8


def test_dag_submitter_fails_before_submission_without_approval(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "remote-plan"
    _write_legacy_renderer_scripts(plan)
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


def test_dag_submitter_keeps_lock_after_unparseable_successful_sbatch(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "remote-plan"
    _write_legacy_renderer_scripts(plan)
    _write_safe_env(tmp_path)
    calls = tmp_path / "sbatch.calls"
    fake_sbatch = tmp_path / "fake-sbatch"
    _ = fake_sbatch.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_SBATCH_CALLS"
printf 'submitted-without-a-parseable-id\n'
""",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o700)
    output_root = tmp_path / "company-output" / "malformed-job-id"
    output_root.parent.mkdir()
    env = {
        **os.environ,
        "FAKE_SBATCH_CALLS": str(calls),
        "WORLDMM_APPROVED_OUTPUT_PREFIX": str(output_root.parent),
        "WORLDMM_APPROVED_REPO_PREFIX": str(tmp_path),
        "WORLDMM_EXECUTION_PROFILE": "full",
        "WORLDMM_OUTPUT_ROOT": str(output_root),
        "WORLDMM_REMOTE_REPO": str(tmp_path),
        "WORLDMM_RUN_ID": "malformed-job-id",
        "WORLDMM_SBATCH": str(fake_sbatch),
        "WORLDMM_SMVQA_REMOTE_APPROVED": "1",
    }

    first = subprocess.run(
        ["bash", str(plan / DAG_SUBMIT_SCRIPT_NAME)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert first.returncode != 0
    assert "invalid sbatch job id" in first.stderr
    assert "returned no trustworthy job ID; keeping lock" in first.stderr
    lock = output_root / "summary/dag_submit.preflight.lock"
    attempts = output_root / "summary/dag_submit.preflight.attempts"
    assert lock.is_file()
    attempt = _load_json_object(attempts.read_bytes())
    assert attempt == {
        "schema_version": 1,
        "event": "submission-unknown-before-sbatch",
        "run_id": "malformed-job-id",
        "phase": "preflight",
        "stage": "preflight_ingest",
        "identity": "worldmm:malformed-job-id:preflight:preflight_ingest",
    }
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1

    repeated = subprocess.run(
        ["bash", str(plan / DAG_SUBMIT_SCRIPT_NAME)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert repeated.returncode != 0
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1


def test_dag_submitter_rejects_output_root_outside_run_scope(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "remote-plan"
    _write_legacy_renderer_scripts(plan)
    _write_safe_env(tmp_path)
    wrong_output_root = tmp_path / "company-output" / "wrong-run"
    env = {
        **os.environ,
        "WORLDMM_REMOTE_REPO": str(tmp_path),
        "WORLDMM_RUN_ID": "expected-run",
        "WORLDMM_OUTPUT_ROOT": str(wrong_output_root),
        "WORLDMM_APPROVED_REPO_PREFIX": str(tmp_path),
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


def _load_json_object(raw: bytes | str) -> dict[str, JsonValue]:
    value = loads_strict(raw)
    if not isinstance(value, dict):
        msg = "expected a JSON object"
        raise TypeError(msg)
    return value


def _python_runtime(python: Path) -> dict[str, JsonValue]:
    script = """
import json
import sys
from importlib.metadata import distributions
from pathlib import Path

packages = sorted(
    (
        distribution.metadata.get("Name") or "",
        distribution.version,
        distribution.read_text("direct_url.json") or "",
    )
    for distribution in distributions()
)
print(json.dumps({
    "version": sys.version,
    "executable": str(Path(sys.executable).resolve(strict=True)),
    "packages": packages,
}))
"""
    return _load_json_object(
        subprocess.check_output([str(python), "-c", script], text=True),
    )


def _write_python_runtime_manifests(runtime: Path, diagnostics: Path) -> None:
    inventory_rows: list[bytes] = []
    content_rows: list[str] = []
    for path in runtime.rglob("*"):
        relative = path.relative_to(runtime)
        if not (path.is_file() or path.is_symlink()):
            continue
        relative_text = f"./{relative.as_posix()}"
        kind = "l" if path.is_symlink() else "f"
        target = path.readlink() if path.is_symlink() else ""
        inventory_rows.append(f"{kind} {relative_text} {target}\0".encode())
        if path.is_file():
            content_rows.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative_text}\n"
            )
    inventory = hashlib.sha256(b"".join(sorted(inventory_rows))).hexdigest()
    _ = (diagnostics / "python_runtime.sha256").write_text(
        "".join(sorted(content_rows)),
        encoding="utf-8",
    )
    _ = (diagnostics / "python_runtime.files.sha256").write_text(
        f"{inventory}  -\n",
        encoding="utf-8",
    )
    loader_digest = subprocess.check_output(
        [
            "bash",
            "-c",
            (
                'ldd "$1" | while IFS= read -r loader; do '
                'printf "%s\\n" "${loader%% (*}"; done | '
                "LC_ALL=C sort | sha256sum"
            ),
            "bash",
            str(runtime / "bin/python"),
        ],
        text=True,
    )
    _ = (diagnostics / "python_runtime.loader.sha256").write_text(
        loader_digest,
        encoding="utf-8",
    )
    executable = Path(sys.executable).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    _ = (diagnostics / "python_base_roots.tsv").write_text(
        "".join(
            (
                f"base_prefix\t{base_prefix}\n",
                f"executable\t{executable}\n",
                f"stdlib\t{executable}\n",
                f"platstdlib\t{executable}\n",
            ),
        ),
        encoding="utf-8",
    )
    executable_digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    _ = (diagnostics / "python_base_runtime.sha256").write_text(
        f"{executable_digest}  {executable}\n",
        encoding="utf-8",
    )
    base_inventory = hashlib.sha256(f"f {executable} \0".encode()).hexdigest()
    _ = (diagnostics / "python_base_runtime.files.sha256").write_text(
        f"{base_inventory}  -\n",
        encoding="utf-8",
    )


def _runtime_descriptor(path: Path) -> dict[str, int | str]:
    state = path.lstat()
    value: dict[str, int | str] = {
        "st_dev": state.st_dev,
        "st_ino": state.st_ino,
        "st_mode": state.st_mode,
        "st_uid": state.st_uid,
        "st_gid": state.st_gid,
        "st_size": state.st_size,
        "st_mtime_ns": state.st_mtime_ns,
    }
    if stat.S_ISLNK(state.st_mode):
        value["link_target"] = str(path.readlink())
    return value


def _runtime_tree_manifest(path: Path) -> dict[str, object]:
    entries: dict[str, dict[str, int | str]] = {}

    def visit(directory: Path, relative: Path | None = None) -> None:
        if relative is None:
            relative = Path()
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            item_relative = (relative / child.name).as_posix()
            descriptor = _runtime_descriptor(child)
            mode = int(descriptor["st_mode"])
            if stat.S_ISREG(mode):
                descriptor["sha256"] = hashlib.sha256(child.read_bytes()).hexdigest()
            entries[item_relative] = descriptor
            if stat.S_ISDIR(mode):
                visit(child, relative / child.name)

    visit(path)
    inventory = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "path": str(path),
        "descriptor": _runtime_descriptor(path),
        "entries": entries,
        "inventory_sha256": inventory,
    }


def _runtime_binary_manifest(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "descriptor": _runtime_descriptor(path),
    }


def _runtime_ldd_closure(interpreter: Path) -> tuple[Path, list[Path]]:
    output = subprocess.check_output(["ldd", str(interpreter)], text=True)
    paths = {
        Path(line.split("=>", 1)[-1].strip().split(" (", 1)[0]).resolve()
        for line in output.splitlines()
        if line.split("=>", 1)[-1].strip().startswith("/")
    }
    loaders = [
        path for path in paths if "ld-linux" in path.name or "ld-musl" in path.name
    ]
    assert len(loaders) == 1
    return loaders[0], sorted(paths - {loaders[0]})


def _write_runtime_content_manifest(runtime: Path, target: Path) -> None:
    interpreter = Path(sys.executable).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    loader, libraries = _runtime_ldd_closure(interpreter)
    entry = runtime / "bin" / "python"
    _ = target.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "RuntimeContentManifestV1",
                "root": _runtime_tree_manifest(runtime),
                "entry": {
                    "path": "bin/python",
                    "sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
                    "descriptor": _runtime_descriptor(entry),
                },
                "interpreter": _runtime_binary_manifest(interpreter),
                "base_prefix": _runtime_tree_manifest(base_prefix),
                "stdlib": _runtime_tree_manifest(stdlib),
                "loader": _runtime_binary_manifest(loader),
                "shared_libraries": [
                    _runtime_binary_manifest(path) for path in libraries
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    target.chmod(0o600)


def test_runtime_inventory_binds_mutated_unlisted_and_symlink_entries(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    entry = runtime / "bin" / "python"
    entry.parent.mkdir(parents=True)
    _ = entry.write_text("#!/bin/sh\n", encoding="utf-8")
    approved = _runtime_tree_manifest(runtime)

    _ = entry.write_text("#!/bin/sh\n# mutated\n", encoding="utf-8")
    assert _runtime_tree_manifest(runtime) != approved

    _ = entry.write_text("#!/bin/sh\n", encoding="utf-8")
    unlisted = runtime / "unlisted.py"
    _ = unlisted.write_text("unlisted\n", encoding="utf-8")
    assert _runtime_tree_manifest(runtime) != approved

    unlisted.unlink()
    external = tmp_path / "external.py"
    _ = external.write_text("external\n", encoding="utf-8")
    (runtime / "linked.py").symlink_to(external)
    observed = _runtime_tree_manifest(runtime)
    assert observed != approved
    entries = cast("dict[str, dict[str, int | str]]", observed["entries"])
    assert entries["linked.py"]["link_target"] == str(external)


def _write_safe_env(repo: Path) -> None:
    path = repo / ".env.worldmm"
    _ = path.write_text("# test-owned environment\n")
    path.chmod(0o600)


def test_teacher_oracle_graph_is_accounting_gated_and_has_no_phase_b_submission() -> (
    None
):
    preflight_text = teacher_oracle_preflight_submit_script_text()
    provider_text = teacher_oracle_provider_gate_submit_script_text()
    downstream_text = teacher_oracle_downstream_submit_script_text()
    runner_text = teacher_oracle_stage_script_text()

    assert "WORLDMM_EXECUTION_PROFILE must be teacher-oracle" in preflight_text
    for name in (
        "submit_teacher_oracle_preflight.sh",
        "submit_teacher_oracle_provider_gate.sh",
        "submit_teacher_oracle_downstream.sh",
    ):
        assert name in preflight_text
    assert preflight_text == provider_text == downstream_text
    assert "teacher_oracle_gate" in provider_text
    assert "teacher_oracle_finalizer" in provider_text
    assert "PHASE_B_SUBMISSION=conditional-second-approval" not in provider_text
    assert "WORLDMM_PHASE_B_APPROVAL_FILE" in downstream_text
    assert "for variant in E0 T0 T1" in downstream_text
    assert "spatial_train" not in provider_text
    assert "WORLDMM_SPATIAL_INFER_EXE" not in provider_text
    assert "Ed25519PublicKey" in runner_text
    assert "teacher_oracle_continue.json" in runner_text
    assert "labels.jsonl" not in runner_text


def test_teacher_oracle_renderers_are_shell_valid_and_compile_embedded_python(
    tmp_path: Path,
) -> None:
    submitters = (
        tmp_path / TEACHER_ORACLE_PREFLIGHT_SUBMIT_SCRIPT_NAME,
        tmp_path / TEACHER_ORACLE_PROVIDER_GATE_SUBMIT_SCRIPT_NAME,
        tmp_path / TEACHER_ORACLE_DOWNSTREAM_SUBMIT_SCRIPT_NAME,
    )
    runner = tmp_path / TEACHER_ORACLE_STAGE_SCRIPT_NAME
    for submitter, text in zip(
        submitters,
        (
            teacher_oracle_preflight_submit_script_text(),
            teacher_oracle_provider_gate_submit_script_text(),
            teacher_oracle_downstream_submit_script_text(),
        ),
        strict=True,
    ):
        _ = submitter.write_text(text, encoding="utf-8")
    _ = runner.write_text(
        teacher_oracle_stage_script_text(),
        encoding="utf-8",
    )
    for script in (*submitters, runner):
        proof = subprocess.run(
            ["bash", "-n", str(script)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert proof.returncode == 0, proof.stderr
        _assert_embedded_python_compiles(script)


def _canonical_signed_payload(value: Mapping[str, JsonValue]) -> bytes:
    return canonicalize(value)


def _domain_signed_payload(value: Mapping[str, JsonValue], purpose: str) -> bytes:
    return signing_bytes(value, purpose)


def test_signed_attestation_golden_vectors_are_strict_and_jcs_ordered() -> None:
    value = with_payload_sha256({"": 1, "𐀀": 2, "small": 1e-6, "tiny": 1e-7})
    canonical = canonicalize(value)
    assert (
        b'"small":0.000001,"tiny":1e-7,"\xf0\x90\x80\x80":2,"\xee\x80\x80":1'
        in canonical
    )
    framed = signing_bytes(value, "phase_a-approval")
    assert framed.startswith(b"worldmm-signed-attestation-v1\x00phase_a-approval\x00")
    with pytest.raises(AttestationError, match="duplicate JSON member"):
        _ = loads_strict(b'{"a":1,"a":2}')
    with pytest.raises(AttestationError, match="safe range"):
        _ = loads_strict(b'{"n":9007199254740992}')


def _write_teacher_oracle_harness(  # noqa: PLR0915
    tmp_path: Path,
    *,
    phase: str,
    signer_valid_until: datetime | None = None,
) -> tuple[dict[str, str], Path, Path, Path, Ed25519PrivateKey]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan = tmp_path / "remote-plan"
    plan.mkdir()
    submit_scripts = {
        TEACHER_ORACLE_PREFLIGHT_SUBMIT_SCRIPT_NAME: (
            teacher_oracle_preflight_submit_script_text()
        ),
        TEACHER_ORACLE_PROVIDER_GATE_SUBMIT_SCRIPT_NAME: (
            teacher_oracle_provider_gate_submit_script_text()
        ),
        TEACHER_ORACLE_DOWNSTREAM_SUBMIT_SCRIPT_NAME: (
            teacher_oracle_downstream_submit_script_text()
        ),
    }
    for name, text in submit_scripts.items():
        script = plan / name
        _ = script.write_text(text, encoding="utf-8")
        script.chmod(0o755)
    submitter = plan / (
        TEACHER_ORACLE_PREFLIGHT_SUBMIT_SCRIPT_NAME
        if phase == "phase-a"
        else TEACHER_ORACLE_DOWNSTREAM_SUBMIT_SCRIPT_NAME
    )
    runner = plan / TEACHER_ORACLE_STAGE_SCRIPT_NAME
    _ = runner.write_text(
        teacher_oracle_stage_script_text(),
        encoding="utf-8",
    )
    runner.chmod(0o755)

    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.parent.parent.chmod(0o700)
    python.parent.chmod(0o700)
    _ = python.write_text(
        f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} "$@"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    runtime_manifest = tmp_path / "runtime-content.json"
    _write_runtime_content_manifest(tmp_path / ".venv", runtime_manifest)

    calls = tmp_path / "sbatch.calls"
    counter = tmp_path / "sbatch.counter"
    sbatch = tmp_path / "fake-sbatch"
    _ = sbatch.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_SBATCH_CALLS"
job=1000
[ ! -f "$FAKE_SBATCH_COUNTER" ] || job="$(<"$FAKE_SBATCH_COUNTER")"
job=$((job + 1))
printf %s "$job" > "$FAKE_SBATCH_COUNTER"
printf '%s;local-cluster\n' "$job"
""",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    sacct = tmp_path / "fake-sacct"
    _ = sacct.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  --version) printf '23.02.0\n' ;;
  --helpformat)
    printf '%s\n' 'JobIDRaw Cluster State ExitCode Restarts SLUID OriginalSLUID'
    ;;
  *) printf '1001|local-cluster|COMPLETED|0:0|0|sluid|sluid\n' ;;
esac
""",
        encoding="utf-8",
    )
    sacct.chmod(0o755)
    scancel = tmp_path / "fake-scancel"
    _ = scancel.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_SCANCEL_CALLS"
""",
        encoding="utf-8",
    )
    scancel.chmod(0o755)
    scontrol = tmp_path / "fake-scontrol"
    _ = scontrol.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_SCONTROL_CALLS"
""",
        encoding="utf-8",
    )
    scontrol.chmod(0o755)

    graph = ExperimentGraphV1.model_validate_json(
        (ROOT / "configs/spatial/exp_0005_teacher_oracle.example.json").read_text(
            encoding="utf-8"
        )
    )
    local_resources = ResourceSpecV1(
        partition="cpu-prepro-queue",
        nodes=1,
        cpus=1,
        memory="1G",
        time="00:01:00",
        gpus_per_node=0,
    )
    graph = graph.model_copy(
        update={
            "stage_specs": tuple(
                stage_spec.model_copy(update={"resources": local_resources})
                for stage_spec in graph.stage_specs
            )
        }
    )
    resources = tmp_path / "experiment-graph.json"
    _ = resources.write_text(graph.model_dump_json(), encoding="utf-8")
    resources.chmod(0o600)
    provider = tmp_path / "oracle-provider"
    stage = tmp_path / "oracle-stage"
    for executable in (provider, stage):
        _ = executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)
    quality_evaluator = tmp_path / "oracle-quality-evaluator"
    _ = quality_evaluator.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
[ "$1" = --contract ]
[ "$3" = --out ]
printf '%s\n' '{"schema_version":1,"outcome":"diagnostic_contract_eligible"}' > "$4"
""",
        encoding="utf-8",
    )
    quality_evaluator.chmod(0o700)

    private = Ed25519PrivateKey.generate()
    public = (
        base64.urlsafe_b64encode(
            private.public_key().public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            ),
        )
        .decode()
        .rstrip("=")
    )
    valid_until = signer_valid_until or (datetime.now(UTC) + timedelta(days=1))
    registry = tmp_path / "signers.json"
    registry_value: dict[str, JsonValue] = {
        "schema_version": 1,
        "keys": [
            {
                "key_id": "local-ed25519",
                "public_key": public,
                "valid_from": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
                "valid_until": valid_until.isoformat(),
                "purposes": [
                    "phase_a_approval",
                    "phase_b_approval",
                    "continue_receipt",
                ],
            },
        ],
    }
    registry_value = with_payload_sha256(registry_value)
    _ = registry.write_bytes(_canonical_signed_payload(registry_value))
    registry.chmod(0o600)

    root = tmp_path / "outputs" / "local-run"
    validation_receipt = tmp_path / "teacher-oracle-validation.json"
    _ = validation_receipt.write_text("{}\n", encoding="utf-8")
    validation_receipt.chmod(0o600)
    signing_key = tmp_path / "continue-receipt.key"
    _ = signing_key.write_bytes(
        base64.urlsafe_b64encode(
            private.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption(),
            )
        ).rstrip(b"=")
        + b"\n"
    )
    signing_key.chmod(0o600)
    preflight_seal = tmp_path / "preflight.seal"
    _ = preflight_seal.write_text(
        '{"kind":"PreflightSealV1","job_id":"1001"}\n',
        encoding="utf-8",
    )
    preflight_seal.chmod(0o600)
    qa_artifacts = {
        name: tmp_path / name
        for name in (
            "qa-shard-map",
            "qa-lineage",
            "qa-finalization-receipt",
            "qa-predictions",
        )
    }
    for name, path in qa_artifacts.items():
        _ = path.write_text(f"{name}\n", encoding="utf-8")
        path.chmod(0o600)
    bindings: dict[str, str] = {
        "WORLDMM_EXPERIMENT_ID": "EXP-LOCAL",
        "WORLDMM_SENSOR_AUDIT_SHA256": "a" * 64,
        "WORLDMM_PROVIDER_SHA256": "b" * 64,
        "WORLDMM_SPLIT_SHA256": "c" * 64,
        "WORLDMM_CODE_SHA": "d" * 64,
        "WORLDMM_POLICY_SHA256": "e" * 64,
        "WORLDMM_TEACHER_ORACLE_VALIDATION_SHA256": hashlib.sha256(
            validation_receipt.read_bytes()
        ).hexdigest(),
        "WORLDMM_EXPERIMENT_CONFIG_SHA256": "0" * 64,
        "WORLDMM_FRAME_ASSETS_SHA256": "1" * 64,
        "WORLDMM_BYTE_BUDGET_SHA256": "2" * 64,
        "WORLDMM_PLAN_SHA256": "3" * 64,
        "WORLDMM_QA_SHARD_MAP_SHA256": hashlib.sha256(
            qa_artifacts["qa-shard-map"].read_bytes()
        ).hexdigest(),
        "WORLDMM_QA_LINEAGE_SHA256": hashlib.sha256(
            qa_artifacts["qa-lineage"].read_bytes()
        ).hexdigest(),
        "WORLDMM_QA_FINALIZATION_RECEIPT_SHA256": hashlib.sha256(
            qa_artifacts["qa-finalization-receipt"].read_bytes()
        ).hexdigest(),
        "WORLDMM_QA_PREDICTIONS_SHA256": hashlib.sha256(
            qa_artifacts["qa-predictions"].read_bytes()
        ).hexdigest(),
        "WORLDMM_REMOTE_SNAPSHOT_SHA256": hashlib.sha256(
            "".join(
                (
                    f"{path.relative_to(tmp_path)}\0"
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}\n"
                )
                for path in sorted(tmp_path.rglob("*"))
                if path.is_file()
            ).encode()
        ).hexdigest(),
        "WORLDMM_DAG_PREFLIGHT_SUBMIT_SCRIPT_SHA256": hashlib.sha256(
            (plan / TEACHER_ORACLE_PREFLIGHT_SUBMIT_SCRIPT_NAME).read_bytes()
        ).hexdigest(),
        "WORLDMM_DAG_PROVIDER_GATE_SUBMIT_SCRIPT_SHA256": hashlib.sha256(
            (plan / TEACHER_ORACLE_PROVIDER_GATE_SUBMIT_SCRIPT_NAME).read_bytes()
        ).hexdigest(),
        "WORLDMM_DAG_DOWNSTREAM_SUBMIT_SCRIPT_SHA256": hashlib.sha256(
            (plan / TEACHER_ORACLE_DOWNSTREAM_SUBMIT_SCRIPT_NAME).read_bytes()
        ).hexdigest(),
        "WORLDMM_DAG_STAGE_SCRIPT_SHA256": hashlib.sha256(
            runner.read_bytes()
        ).hexdigest(),
        "WORLDMM_ORACLE_PROVIDER_EXECUTABLE_SHA256": hashlib.sha256(
            provider.read_bytes()
        ).hexdigest(),
        "WORLDMM_ORACLE_STAGE_EXECUTABLE_SHA256": hashlib.sha256(
            stage.read_bytes()
        ).hexdigest(),
    }
    env = {
        **os.environ,
        **bindings,
        "FAKE_SBATCH_CALLS": str(calls),
        "FAKE_SBATCH_COUNTER": str(counter),
        "FAKE_SCANCEL_CALLS": str(tmp_path / "scancel.calls"),
        "WORLDMM_SMVQA_REMOTE_APPROVED": "1",
        "WORLDMM_EXECUTION_PROFILE": "teacher-oracle",
        "WORLDMM_REMOTE_REPO": str(tmp_path),
        "WORLDMM_RUN_ID": "local-run",
        "WORLDMM_OUTPUT_ROOT": str(root),
        "WORLDMM_DAG_PHASE": phase,
        "WORLDMM_TEACHER_ORACLE_VALIDATION_RECEIPT": str(validation_receipt),
        "WORLDMM_CONTINUE_RECEIPT_KEY_ID": "local-ed25519",
        "WORLDMM_CONTINUE_RECEIPT_SIGNING_KEY": str(signing_key),
        "WORLDMM_SIGNER_REGISTRY": str(registry),
        "WORLDMM_EXPERIMENT_GRAPH_FILE": str(resources),
        "WORLDMM_RESOURCE_CONFIG_SHA256": hashlib.sha256(
            resources.read_bytes(),
        ).hexdigest(),
        "WORLDMM_ORACLE_PROVIDER_EXECUTABLE": str(provider),
        "WORLDMM_ORACLE_PROVIDER_CONFIG": str(resources),
        "WORLDMM_ORACLE_PROVIDER_CONFIG_SHA256": hashlib.sha256(
            resources.read_bytes()
        ).hexdigest(),
        "WORLDMM_ORACLE_STAGE_EXECUTABLE": str(stage),
        "WORLDMM_SACCT": str(sacct),
        "WORLDMM_SBATCH": str(sbatch),
        "WORLDMM_SCANCEL": str(scancel),
        "WORLDMM_SCONTROL": str(scontrol),
        "FAKE_SCONTROL_CALLS": str(tmp_path / "scontrol.calls"),
        "WORLDMM_SIGNER_REGISTRY_SHA256": hashlib.sha256(
            registry.read_bytes()
        ).hexdigest(),
        "WORLDMM_SLURM_CLUSTER": "local-cluster",
        "WORLDMM_PHASE_A_PRODUCERS": "geometry,semantic,place",
        "WORLDMM_PREFLIGHT_SEAL_SHA256": hashlib.sha256(
            preflight_seal.read_bytes()
        ).hexdigest(),
        "WORLDMM_ACCOUNTING_SETTLE_SECONDS": "1",
        "WORLDMM_ACCOUNTING_SETTLE_INTERVAL_SECONDS": "1",
        "WORLDMM_ORACLE_QUALITY_EVALUATOR": str(quality_evaluator),
        "WORLDMM_ORACLE_QUALITY_EVALUATOR_SHA256": hashlib.sha256(
            quality_evaluator.read_bytes()
        ).hexdigest(),
        "WORLDMM_ORACLE_QUALITY_CONTRACT": str(resources),
        "WORLDMM_ORACLE_QUALITY_CONTRACT_SHA256": hashlib.sha256(
            resources.read_bytes()
        ).hexdigest(),
        "WORLDMM_ATTESTED_RUNTIME_ROOT": str(tmp_path / ".venv"),
        "WORLDMM_QA_SHARD_MAP": str(qa_artifacts["qa-shard-map"]),
        "WORLDMM_QA_LINEAGE": str(qa_artifacts["qa-lineage"]),
        "WORLDMM_QA_FINALIZATION_RECEIPT": str(qa_artifacts["qa-finalization-receipt"]),
        "WORLDMM_QA_PREDICTIONS": str(qa_artifacts["qa-predictions"]),
        "WORLDMM_ATTESTED_RUNTIME_MANIFEST": str(runtime_manifest),
        "WORLDMM_ATTESTED_RUNTIME_MANIFEST_SHA256": hashlib.sha256(
            runtime_manifest.read_bytes()
        ).hexdigest(),
    }
    approval = tmp_path / f"{phase}-approval.json"
    approval_value: dict[str, JsonValue] = {
        "schema_version": 1,
        "kind": "SignedAttestationEnvelopeV1",
        "experiment_id": env["WORLDMM_EXPERIMENT_ID"],
        "profile": "teacher-oracle",
        "phase": phase.replace("-", "_"),
        "sensor_audit_sha256": env["WORLDMM_SENSOR_AUDIT_SHA256"],
        "provider_sha256": env["WORLDMM_PROVIDER_SHA256"],
        "split_sha256": env["WORLDMM_SPLIT_SHA256"],
        "code_sha256": env["WORLDMM_CODE_SHA"],
        "policy_sha256": env["WORLDMM_POLICY_SHA256"],
        "validation_receipt_sha256": env["WORLDMM_TEACHER_ORACLE_VALIDATION_SHA256"],
        "experiment_config_sha256": env["WORLDMM_EXPERIMENT_CONFIG_SHA256"],
        "run_id": env["WORLDMM_RUN_ID"],
        "output_root": env["WORLDMM_OUTPUT_ROOT"],
        "frame_assets_sha256": env["WORLDMM_FRAME_ASSETS_SHA256"],
        "byte_budget_sha256": env["WORLDMM_BYTE_BUDGET_SHA256"],
        "resource_config_sha256": env["WORLDMM_RESOURCE_CONFIG_SHA256"],
        "plan_sha256": env["WORLDMM_PLAN_SHA256"],
        "remote_snapshot_sha256": env["WORLDMM_REMOTE_SNAPSHOT_SHA256"],
        "dag_preflight_submit_script_sha256": env[
            "WORLDMM_DAG_PREFLIGHT_SUBMIT_SCRIPT_SHA256"
        ],
        "dag_provider_gate_submit_script_sha256": env[
            "WORLDMM_DAG_PROVIDER_GATE_SUBMIT_SCRIPT_SHA256"
        ],
        "dag_downstream_submit_script_sha256": env[
            "WORLDMM_DAG_DOWNSTREAM_SUBMIT_SCRIPT_SHA256"
        ],
        "dag_stage_script_sha256": env["WORLDMM_DAG_STAGE_SCRIPT_SHA256"],
        "oracle_provider_executable_sha256": env[
            "WORLDMM_ORACLE_PROVIDER_EXECUTABLE_SHA256"
        ],
        "oracle_stage_executable_sha256": env["WORLDMM_ORACLE_STAGE_EXECUTABLE_SHA256"],
        "preflight_seal_sha256": env["WORLDMM_PREFLIGHT_SEAL_SHA256"],
        "quality_contract_sha256": env["WORLDMM_ORACLE_QUALITY_CONTRACT_SHA256"],
        "quality_evaluator_sha256": env["WORLDMM_ORACLE_QUALITY_EVALUATOR_SHA256"],
        "attested_runtime_root": env["WORLDMM_ATTESTED_RUNTIME_ROOT"],
        "attested_runtime_manifest_sha256": env[
            "WORLDMM_ATTESTED_RUNTIME_MANIFEST_SHA256"
        ],
        "purpose": f"teacher_oracle_{phase.replace('-', '_')}_execution",
        "slurm_cluster": env["WORLDMM_SLURM_CLUSTER"],
        "accounting_settle_seconds": env["WORLDMM_ACCOUNTING_SETTLE_SECONDS"],
        "accounting_settle_interval_seconds": env[
            "WORLDMM_ACCOUNTING_SETTLE_INTERVAL_SECONDS"
        ],
        "producer_tuple": ["geometry", "semantic", "place"],
        "producer_stage_tuple": [
            "teacher_oracle_geometry",
            "teacher_oracle_semantic",
            "teacher_oracle_place",
        ],
        "oracle_provider_config_sha256": env["WORLDMM_ORACLE_PROVIDER_CONFIG_SHA256"],
        "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        "key_id": "local-ed25519",
    }
    if phase == "phase-b":
        approval_value.update(
            {
                "qa_shard_map_sha256": env["WORLDMM_QA_SHARD_MAP_SHA256"],
                "qa_lineage_sha256": env["WORLDMM_QA_LINEAGE_SHA256"],
                "qa_finalization_receipt_sha256": env[
                    "WORLDMM_QA_FINALIZATION_RECEIPT_SHA256"
                ],
                "qa_predictions_sha256": env["WORLDMM_QA_PREDICTIONS_SHA256"],
            }
        )
    approval_value = with_payload_sha256(approval_value)
    approval_value["signature"] = b64url_encode(
        private.sign(
            _domain_signed_payload(
                approval_value,
                f"{approval_value['phase']}-approval",
            )
        )
    )
    _ = approval.write_bytes(_canonical_signed_payload(approval_value))
    approval.chmod(0o600)
    env["WORLDMM_APPROVAL_FILE"] = str(approval)
    if phase == "phase-b":
        env["WORLDMM_PHASE_B_APPROVAL_FILE"] = str(approval)
    return env, submitter, approval, calls, private


def _run_teacher_submitter(
    submitter: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(submitter)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_teacher_oracle_phase_a_submission_is_signed_and_topologically_chained(
    tmp_path: Path,
) -> None:
    env, submitter, _, calls, _ = _write_teacher_oracle_harness(
        tmp_path,
        phase="phase-a",
    )

    result = _run_teacher_submitter(submitter, env)

    assert result.returncode == 0, result.stderr
    submitted = calls.read_text(encoding="utf-8").splitlines()
    assert len(submitted) == 1
    assert "teacher_oracle_preflight" in submitted[0]
    assert "--hold" in submitted[0]
    releases = Path(env["FAKE_SCONTROL_CALLS"]).read_text(encoding="utf-8").splitlines()
    assert releases == ["release 1001"]

    repeated = _run_teacher_submitter(submitter, env)
    assert repeated.returncode != 0
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1


def test_teacher_oracle_preflight_uses_fixed_producers_and_is_approval_independent(
    tmp_path: Path,
) -> None:
    fixed_env, fixed_submitter, _, fixed_calls, _ = _write_teacher_oracle_harness(
        tmp_path / "fixed-producers",
        phase="phase-a",
    )
    fixed_env["WORLDMM_PHASE_A_PRODUCERS"] = "attacker-controlled"
    fixed = _run_teacher_submitter(fixed_submitter, fixed_env)
    assert fixed.returncode == 0, fixed.stderr
    assert len(fixed_calls.read_text(encoding="utf-8").splitlines()) == 1

    for name, until in (
        ("invalid", None),
        ("expired", datetime.now(UTC) - timedelta(days=1)),
    ):
        env, submitter, approval, calls, _ = _write_teacher_oracle_harness(
            tmp_path / name,
            phase="phase-a",
            signer_valid_until=until,
        )
        if name == "invalid":
            value = _load_json_object(approval.read_bytes())
            value["signature"] = "not-a-valid-ed25519-signature"
            _ = approval.write_bytes(_canonical_signed_payload(value))
            approval.chmod(0o600)
        result = _run_teacher_submitter(submitter, env)
        assert result.returncode == 0, result.stderr
        assert len(calls.read_text(encoding="utf-8").splitlines()) == 1


def _write_phase_b_receipt(
    env: dict[str, str],
    approval: Path,
    private: Ed25519PrivateKey,
    *,
    invalid_binding: bool = False,
    consumed: bool = False,
) -> None:
    root = Path(env["WORLDMM_OUTPUT_ROOT"])
    summary = root / "summary"
    summary.mkdir(parents=True)
    terminal = summary / "teacher_oracle_terminal.json"
    _ = terminal.write_text(
        '{"provider_gate_decision":"go"}\n',
        encoding="utf-8",
    )
    terminal.chmod(0o600)
    provider_manifest_sha256: dict[str, str] = {}
    producer_jobs: dict[str, dict[str, str]] = {}
    for index, producer in enumerate(("geometry", "semantic", "place"), start=1001):
        provider_root = root / "oracle" / "providers" / producer / f"attempt-{index}"
        provider_root.mkdir(parents=True)
        provider_root.chmod(0o700)
        payload_path = provider_root / "payload.json"
        _ = payload_path.write_text('{"sealed":true}\n', encoding="utf-8")
        payload_path.chmod(0o600)
        state = payload_path.stat()
        descriptor = {
            "sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
            "uid": state.st_uid,
            "mode": stat.S_IMODE(state.st_mode),
            "nlink": state.st_nlink,
            "device": str(state.st_dev),
            "inode": str(state.st_ino),
            "size": state.st_size,
            "mtime_ns": str(state.st_mtime_ns),
        }
        marker = summary / f"teacher_oracle_{producer}.json"
        marker_payload = {
            "schema_version": 1,
            "kind": "ProviderAttemptManifestV1",
            "producer_id": producer,
            "attempt": str(index),
            "attempt_root": str(provider_root),
            "attempt_root_device": str(provider_root.stat().st_dev),
            "attempt_root_inode": str(provider_root.stat().st_ino),
            "success_marker": "teacher-oracle-producer-v1",
            "provider_artifacts": {"payload.json": descriptor["sha256"]},
            "provider_descriptors": {"payload.json": descriptor},
            "coverage": ["payload.json"],
            "causality": "sensor-audit-bound",
        }
        _ = marker.write_text(
            json.dumps(marker_payload, sort_keys=True),
            encoding="utf-8",
        )
        marker.chmod(0o600)
        provider_manifest_sha256[producer] = hashlib.sha256(
            marker.read_bytes()
        ).hexdigest()
        producer_jobs[producer] = {
            "job_id": str(index),
            "sluid": str(index),
            "original_sluid": str(index),
        }
    receipt = cast(
        "dict[str, JsonValue]",
        {
            "schema_version": 1,
            "kind": "SignedAttestationEnvelopeV1",
            "profile": "teacher-oracle",
            "decision": "go",
            "key_id": "local-ed25519",
            "bindings": {
                "experiment_id": env["WORLDMM_EXPERIMENT_ID"],
                "sensor_audit_sha256": env["WORLDMM_SENSOR_AUDIT_SHA256"],
                "provider_sha256": env["WORLDMM_PROVIDER_SHA256"],
                "split_sha256": env["WORLDMM_SPLIT_SHA256"],
                "code_sha256": env["WORLDMM_CODE_SHA"],
                "policy_sha256": env["WORLDMM_POLICY_SHA256"],
                "validation_receipt_sha256": env[
                    "WORLDMM_TEACHER_ORACLE_VALIDATION_SHA256"
                ],
                "experiment_config_sha256": env["WORLDMM_EXPERIMENT_CONFIG_SHA256"],
                "run_id": "wrong-run" if invalid_binding else env["WORLDMM_RUN_ID"],
                "output_root": env["WORLDMM_OUTPUT_ROOT"],
                "frame_assets_sha256": env["WORLDMM_FRAME_ASSETS_SHA256"],
                "byte_budget_sha256": env["WORLDMM_BYTE_BUDGET_SHA256"],
                "resource_config_sha256": env["WORLDMM_RESOURCE_CONFIG_SHA256"],
                "plan_sha256": env["WORLDMM_PLAN_SHA256"],
                "remote_snapshot_sha256": env["WORLDMM_REMOTE_SNAPSHOT_SHA256"],
                "dag_preflight_submit_script_sha256": env[
                    "WORLDMM_DAG_PREFLIGHT_SUBMIT_SCRIPT_SHA256"
                ],
                "dag_provider_gate_submit_script_sha256": env[
                    "WORLDMM_DAG_PROVIDER_GATE_SUBMIT_SCRIPT_SHA256"
                ],
                "dag_downstream_submit_script_sha256": env[
                    "WORLDMM_DAG_DOWNSTREAM_SUBMIT_SCRIPT_SHA256"
                ],
                "dag_stage_script_sha256": env["WORLDMM_DAG_STAGE_SCRIPT_SHA256"],
                "oracle_provider_executable_sha256": env[
                    "WORLDMM_ORACLE_PROVIDER_EXECUTABLE_SHA256"
                ],
                "oracle_stage_executable_sha256": env[
                    "WORLDMM_ORACLE_STAGE_EXECUTABLE_SHA256"
                ],
                "preflight_seal_sha256": env["WORLDMM_PREFLIGHT_SEAL_SHA256"],
                "quality_contract_sha256": env[
                    "WORLDMM_ORACLE_QUALITY_CONTRACT_SHA256"
                ],
                "quality_evaluator_sha256": env[
                    "WORLDMM_ORACLE_QUALITY_EVALUATOR_SHA256"
                ],
                "attested_runtime_root": env["WORLDMM_ATTESTED_RUNTIME_ROOT"],
                "attested_runtime_manifest_sha256": env[
                    "WORLDMM_ATTESTED_RUNTIME_MANIFEST_SHA256"
                ],
                "purpose": "teacher_oracle_phase_b_execution",
                "slurm_cluster": env["WORLDMM_SLURM_CLUSTER"],
                "accounting_settle_seconds": env["WORLDMM_ACCOUNTING_SETTLE_SECONDS"],
                "accounting_settle_interval_seconds": env[
                    "WORLDMM_ACCOUNTING_SETTLE_INTERVAL_SECONDS"
                ],
                "producer_tuple": ["geometry", "semantic", "place"],
                "producer_stage_tuple": [
                    "teacher_oracle_geometry",
                    "teacher_oracle_semantic",
                    "teacher_oracle_place",
                ],
                "oracle_provider_config_sha256": env[
                    "WORLDMM_ORACLE_PROVIDER_CONFIG_SHA256"
                ],
                "registry_sha256": hashlib.sha256(
                    Path(env["WORLDMM_SIGNER_REGISTRY"]).read_bytes(),
                ).hexdigest(),
            },
            "provider_manifest_sha256": provider_manifest_sha256,
            "producer_jobs": producer_jobs,
        },
    )
    receipt = with_payload_sha256(receipt)
    receipt["signature"] = b64url_encode(
        private.sign(_domain_signed_payload(receipt, "continue-receipt"))
    )
    receipt_path = summary / "teacher_oracle_continue.json"
    _ = receipt_path.write_bytes(_canonical_signed_payload(receipt))
    receipt_path.chmod(0o600)

    approval_value = _load_json_object(approval.read_bytes())
    approval_value["continue_receipt_sha256"] = hashlib.sha256(
        receipt_path.read_bytes(),
    ).hexdigest()
    approval_value["terminal_sha256"] = hashlib.sha256(
        terminal.read_bytes(),
    ).hexdigest()
    _ = approval_value.pop("signature")
    approval_value = with_payload_sha256(
        {key: value for key, value in approval_value.items() if key != "payload_sha256"}
    )
    approval_value["signature"] = b64url_encode(
        private.sign(
            _domain_signed_payload(
                approval_value,
                f"{approval_value['phase']}-approval",
            )
        )
    )
    _ = approval.write_bytes(_canonical_signed_payload(approval_value))
    approval.chmod(0o600)
    if consumed:
        os.link(
            receipt_path,
            summary / ".teacher_oracle_continue.used.json",
        )


def test_teacher_oracle_phase_b_rejects_invalid_or_replayed_receipts_before_sbatch(
    tmp_path: Path,
) -> None:
    for name, invalid_binding, consumed in (
        ("invalid-binding", True, False),
        ("replayed", False, True),
    ):
        (
            env,
            submitter,
            approval,
            calls,
            private,
        ) = _write_teacher_oracle_harness(
            tmp_path / name,
            phase="phase-b",
        )
        _write_phase_b_receipt(
            env,
            approval,
            private,
            invalid_binding=invalid_binding,
            consumed=consumed,
        )

        result = _run_teacher_submitter(submitter, env)

        assert result.returncode != 0
        assert not calls.exists()


def test_teacher_oracle_phase_b_submission_consumes_receipt_and_chains_topology(
    tmp_path: Path,
) -> None:
    env, submitter, approval, calls, private = _write_teacher_oracle_harness(
        tmp_path,
        phase="phase-b",
    )
    _write_phase_b_receipt(env, approval, private)

    result = _run_teacher_submitter(submitter, env)

    assert result.returncode == 0, result.stderr
    submitted = calls.read_text(encoding="utf-8").splitlines()
    assert len(submitted) == 11
    stages = (
        "teacher_oracle_E0_materialize",
        "teacher_oracle_E0_retrieve",
        "teacher_oracle_E0_qa",
        "teacher_oracle_T0_materialize",
        "teacher_oracle_T0_retrieve",
        "teacher_oracle_T0_qa",
        "teacher_oracle_T1_materialize",
        "teacher_oracle_T1_retrieve",
        "teacher_oracle_T1_qa",
        "teacher_oracle_evaluator",
        "teacher_oracle_finalizer_phase_b",
    )
    assert all(stage in line for stage, line in zip(stages, submitted, strict=True))
    assert "--dependency=afterok:1001" in submitted[1]
    assert "--dependency=afterok:1002" in submitted[2]
    assert "--dependency=afterok:1004" in submitted[4]
    assert "--dependency=afterok:1005" in submitted[5]
    assert "--dependency=afterok:1007" in submitted[7]
    assert "--dependency=afterok:1008" in submitted[8]
    assert "--dependency=afterok:1003:1006:1009" in submitted[9]
    assert "--dependency=afterok:1010" in submitted[10]

    summary = Path(env["WORLDMM_OUTPUT_ROOT"]) / "summary"
    receipt = summary / "teacher_oracle_continue.json"
    used_receipt = summary / ".teacher_oracle_continue.used.json"
    assert not receipt.exists()
    assert used_receipt.is_file()
    assert used_receipt.stat().st_nlink == 1


def test_teacher_oracle_phase_b_rejects_tampered_receipts_and_terminal_non_go(
    tmp_path: Path,
) -> None:
    for name in (
        "signature",
        "digest",
        "terminal-non-go",
        "missing-phase-b-approval",
    ):
        env, submitter, approval, calls, private = _write_teacher_oracle_harness(
            tmp_path / name,
            phase="phase-b",
        )
        _write_phase_b_receipt(env, approval, private)
        summary = Path(env["WORLDMM_OUTPUT_ROOT"]) / "summary"
        receipt = summary / "teacher_oracle_continue.json"

        if name == "signature":
            receipt_value = _load_json_object(receipt.read_bytes())
            receipt_value["signature"] = (
                base64.urlsafe_b64encode(
                    bytes(64),
                )
                .decode()
                .rstrip("=")
            )
            _ = receipt.write_bytes(_canonical_signed_payload(receipt_value))
            approval_value = _load_json_object(approval.read_bytes())
            approval_value["continue_receipt_sha256"] = hashlib.sha256(
                receipt.read_bytes(),
            ).hexdigest()
            _ = approval_value.pop("signature")
            approval_value = with_payload_sha256(
                {
                    key: value
                    for key, value in approval_value.items()
                    if key != "payload_sha256"
                }
            )
            approval_value["signature"] = b64url_encode(
                private.sign(
                    _domain_signed_payload(
                        approval_value,
                        f"{approval_value['phase']}-approval",
                    )
                )
            )
            _ = approval.write_bytes(_canonical_signed_payload(approval_value))
        elif name == "digest":
            approval_value = _load_json_object(approval.read_bytes())
            approval_value["continue_receipt_sha256"] = "f" * 64
            _ = approval_value.pop("signature")
            approval_value = with_payload_sha256(
                {
                    key: value
                    for key, value in approval_value.items()
                    if key != "payload_sha256"
                }
            )
            approval_value["signature"] = b64url_encode(
                private.sign(
                    _domain_signed_payload(
                        approval_value,
                        f"{approval_value['phase']}-approval",
                    )
                )
            )
            _ = approval.write_bytes(_canonical_signed_payload(approval_value))
        elif name == "terminal-non-go":
            _ = (summary / "teacher_oracle_terminal.json").write_text(
                '{"provider_gate_decision":"no-go"}\n',
                encoding="utf-8",
            )
        else:
            del env["WORLDMM_PHASE_B_APPROVAL_FILE"]

        result = _run_teacher_submitter(submitter, env)

        assert result.returncode != 0
        assert not calls.exists()
        if name == "missing-phase-b-approval":
            assert receipt.is_file()
            assert not (summary / ".teacher_oracle_continue.used.json").exists()
        else:
            assert not receipt.exists()
            assert (summary / ".teacher_oracle_continue.used.json").is_file()


def test_teacher_oracle_phase_b_concurrent_submitters_consume_receipt_once(
    tmp_path: Path,
) -> None:
    env, submitter, approval, calls, private = _write_teacher_oracle_harness(
        tmp_path,
        phase="phase-b",
    )
    _write_phase_b_receipt(env, approval, private)

    processes = [
        subprocess.Popen(
            ["bash", str(submitter)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    results = [process.communicate() for process in processes]
    returncodes = [process.returncode for process in processes]

    assert returncodes.count(0) == 1, results
    assert calls.is_file()
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 11
    summary = Path(env["WORLDMM_OUTPUT_ROOT"]) / "summary"
    receipt = summary / "teacher_oracle_continue.json"
    used_receipt = summary / ".teacher_oracle_continue.used.json"
    assert not receipt.exists()
    assert used_receipt.is_file()
    assert used_receipt.stat().st_nlink == 1


def _prepare_teacher_oracle_gate(env: dict[str, str]) -> None:
    root = Path(env["WORLDMM_OUTPUT_ROOT"])
    summary = root / "summary"
    summary.mkdir(parents=True)
    jobs = {
        "PREFLIGHT_JOB_ID": "1001",
        "PROVIDER_GEOMETRY_JOB_ID": "1002",
        "PROVIDER_SEMANTIC_JOB_ID": "1003",
        "PROVIDER_PLACE_JOB_ID": "1004",
    }
    _ = (summary / "dag_jobs.provider.env").write_text(
        "".join(f"{name}={value}\n" for name, value in jobs.items()),
        encoding="utf-8",
    )
    for producer in ("geometry", "semantic", "place"):
        provider = root / "oracle" / "providers" / producer
        provider.mkdir(parents=True)
        artifact = provider / "output.json"
        _ = artifact.write_text(f"{producer}\n", encoding="utf-8")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        _ = (summary / f"teacher_oracle_{producer}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "producer_id": producer,
                    "success_marker": "teacher-oracle-producer-v1",
                    "attempt": jobs[f"PROVIDER_{producer.upper()}_JOB_ID"],
                    "provider_artifacts": {"output.json": digest},
                    "coverage": ["output.json"],
                }
            ),
            encoding="utf-8",
        )
    env.update(
        {
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_CPUS_PER_TASK": "1",
            "SLURM_JOB_PARTITION": "cpu-prepro-queue",
            "SLURM_MEM_PER_NODE": "1024",
            "SLURM_TIMELIMIT": "00:01:00",
            "SLURM_GPUS_ON_NODE": "0",
        }
    )


def _write_teacher_gate_sacct(tmp_path: Path) -> Path:
    sacct = tmp_path / "gate-sacct"
    _ = sacct.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  --version) printf '23.02.0\n' ;;
  --helpformat)
    printf '%s\n' 'JobIDRaw Cluster State ExitCode Restarts SLUID OriginalSLUID'
    ;;
  *) printf '%s' "$FAKE_SACCT_PAYLOAD" ;;
esac
""",
        encoding="utf-8",
    )
    sacct.chmod(0o755)
    return sacct


def _run_teacher_stage(
    runner: Path, env: dict[str, str], stage: str
) -> subprocess.CompletedProcess[str]:
    stage_env = {**env, "WORLDMM_STAGE": stage}
    return subprocess.run(
        ["bash", str(runner)],
        cwd=ROOT,
        env=stage_env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("zero-rows", ""),
        (
            "duplicate-rows",
            "1001|local-cluster|COMPLETED|0:0|0|sluid|sluid\n"
            "1001|local-cluster|COMPLETED|0:0|0|sluid|sluid\n",
        ),
        (
            "wrong-cluster",
            "1001|foreign-cluster|COMPLETED|0:0|0|sluid|sluid\n",
        ),
        ("wrong-job", "9999|local-cluster|COMPLETED|0:0|0|sluid|sluid\n"),
        (
            "nonzero-exit",
            "1001|local-cluster|COMPLETED|1:0|0|sluid|sluid\n",
        ),
        ("restarts", "1001|local-cluster|COMPLETED|0:0|1|sluid|sluid\n"),
        (
            "sluid-mismatch",
            "1001|local-cluster|COMPLETED|0:0|0|sluid|original\n",
        ),
    ],
)
def test_teacher_oracle_gate_fails_closed_for_lossless_accounting_ambiguity(
    tmp_path: Path,
    name: str,
    payload: str,
) -> None:
    env, _, _, _, _ = _write_teacher_oracle_harness(
        tmp_path / name,
        phase="phase-a",
    )
    _prepare_teacher_oracle_gate(env)
    env["WORLDMM_SACCT"] = str(_write_teacher_gate_sacct(tmp_path / name))
    env["FAKE_SACCT_PAYLOAD"] = payload

    # Keep retry behavior intact while making persistent negative accounting
    # responses deterministic and fast in this local subprocess harness.
    sitecustomize = tmp_path / name / "sitecustomize.py"
    _ = sitecustomize.write_text(
        "import time\ntime.sleep = lambda _: None\n",
        encoding="utf-8",
    )
    env["PYTHONPATH"] = (
        str(sitecustomize.parent) + os.pathsep + env.get("PYTHONPATH", "")
    )

    runner = tmp_path / name / "remote-plan" / TEACHER_ORACLE_STAGE_SCRIPT_NAME
    gate = _run_teacher_stage(runner, env, "teacher_oracle_gate")

    receipt = Path(env["WORLDMM_OUTPUT_ROOT"]) / "summary/teacher_oracle_continue.json"
    assert gate.returncode != 0
    assert not receipt.exists()

    finalizer = _run_teacher_stage(runner, env, "teacher_oracle_finalizer")
    terminal = Path(env["WORLDMM_OUTPUT_ROOT"]) / "summary/teacher_oracle_terminal.json"
    assert finalizer.returncode == 0, finalizer.stderr
    terminal_payload = _load_json_object(terminal.read_bytes())
    assert terminal_payload["schema_version"] == 1
    assert terminal_payload["kind"] == "ProviderGateTerminalV1"
    assert terminal_payload["profile"] == "teacher-oracle"
    assert terminal_payload["operational_state"] == "failed"
    assert terminal_payload["provider_gate_decision"] == "gate_controller_failure"
    assert terminal_payload["scientific_state"] == "not_decidable"
    assert terminal_payload["gate_job_id"] == ""
    inventory = terminal_payload["producer_inventory"]
    assert isinstance(inventory, list)
    assert [item["producer_id"] for item in inventory if isinstance(item, dict)] == [
        "geometry",
        "semantic",
        "place",
    ]
