from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

from worldmm_smvqa.config import load_config
from worldmm_smvqa.remote_plan import (
    DAG_STAGE_SCRIPT_NAME,
    DAG_SUBMIT_SCRIPT_NAME,
    write_remote_plan,
)

ROOT = Path(__file__).resolve().parents[1]


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
    assert submit_text.count('"--dependency=afterok:$dependency"') == 1
    assert submit_text.count("--kill-on-invalid-dep=yes") == 1
    assert "submission_attempts" in submit_text
    assert "sbatch returned no trustworthy job ID; keeping lock" in submit_text
    assert submit_text.index('printf "%s\\n" "$stage" >> "$submission_attempts"') < (
        submit_text.index('raw_job_id="$("${args[@]}" "$stage_script")"')
    )
    assert "partial DAG submission may still have live jobs; keeping lock" in (
        submit_text
    )
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
    assert '"effective_teacher_resources": effective_teacher_resources' in (
        runner_text
    )
    assert "approved Python runtime content changed" in runner_text
    assert "approved Python runtime file inventory changed" in runner_text
    assert "approved Python base runtime content changed" in runner_text
    assert "approved Python base runtime file inventory changed" in runner_text
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
    _ = write_remote_plan(
        load_config(ROOT / "configs/remote.example.yaml"),
        plan,
        {},
        submit=False,
    )
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
                        "WORLDMM_APPROVED_OUTPUT_PREFIX": str(
                            output_root.parent.resolve()
                        ),
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
    assert "approved Python runtime content changed" in (
        drifted_runtime_content.stderr
    )
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
    assert "approved Python runtime file inventory changed" in (
        drifted_runtime.stderr
    )
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


def test_dag_submitter_keeps_lock_after_unparseable_successful_sbatch(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "remote-plan"
    _ = write_remote_plan(
        load_config(ROOT / "configs/remote.example.yaml"),
        plan,
        {},
        submit=False,
    )
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
    assert attempts.read_text(encoding="utf-8") == "preflight_ingest\n"
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
    _ = write_remote_plan(
        load_config(ROOT / "configs/remote.example.yaml"),
        plan,
        {},
        submit=False,
    )
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


def _python_runtime(python: Path) -> dict[str, object]:
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
    return cast(
        "dict[str, object]",
        json.loads(
            subprocess.check_output([str(python), "-c", script], text=True),
        ),
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


def _write_safe_env(repo: Path) -> None:
    path = repo / ".env.worldmm"
    _ = path.write_text("# test-owned environment\n")
    path.chmod(0o600)
