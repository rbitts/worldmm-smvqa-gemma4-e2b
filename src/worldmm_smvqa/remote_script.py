# ruff: noqa: E501
from __future__ import annotations

import hashlib
from textwrap import dedent
from typing import Protocol


class _StudentStageLike(Protocol):
    @property
    def stage_id(self) -> str: ...

    @property
    def host_class(self) -> str: ...

    @property
    def nodes(self) -> int: ...

    @property
    def gpus_per_node(self) -> int: ...

    @property
    def cpus_per_task(self) -> int: ...

    @property
    def memory_gb(self) -> int: ...

    @property
    def time_limit_minutes(self) -> int: ...

    @property
    def command_key(self) -> str: ...


class _StudentEdgeLike(Protocol):
    @property
    def from_stage(self) -> str: ...

    @property
    def to_stage(self) -> str: ...

    @property
    def dependency_kind(self) -> str: ...


class _StudentGraphLike(Protocol):
    @property
    def stages(self) -> tuple[_StudentStageLike, ...]: ...

    @property
    def edges(self) -> tuple[_StudentEdgeLike, ...]: ...


def dag_submit_script_text() -> str:
    """Render a head-node submitter for the staged CPU/GPU pipeline."""
    return dedent(
        r"""
        #!/usr/bin/env bash
        set -euo pipefail
        umask 077
        export LC_ALL=C
        export PYTHONDONTWRITEBYTECODE=1
        unset PYTHONPATH PYTHONHOME
        export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1

        if [ "${WORLDMM_SMVQA_REMOTE_APPROVED:-}" != "1" ]; then
          printf "WORLDMM_SMVQA_REMOTE_APPROVED=1 is required\n" >&2
          exit 1
        fi
        export WORLDMM_SMVQA_REMOTE_APPROVED

        SBATCH="${WORLDMM_SBATCH:-/opt/slurm/bin/sbatch}"
        default_remote_repo=/repo/VTteam/bongh.park/
        default_remote_repo+=worldmm-smvqa-gemma4-e2b
        WORLDMM_REMOTE_REPO="${WORLDMM_REMOTE_REPO:-$default_remote_repo}"
        incoming_run_id="${WORLDMM_RUN_ID:-}"
        : "${WORLDMM_APPROVED_DATA_PREFIX:=/groups/VTteam/datasets}"
        : "${WORLDMM_APPROVED_REPO_PREFIX:=/repo/VTteam/bongh.park}"
        : "${WORLDMM_APPROVED_OUTPUT_PREFIX:=/repo/VTteam/bongh.park/outputs}"
        initial_repo_resolved="$(realpath -e "$WORLDMM_REMOTE_REPO")"
        initial_repo_prefix="$(realpath -e "$WORLDMM_APPROVED_REPO_PREFIX")"
        case "$initial_repo_resolved" in
          "$initial_repo_prefix"|"$initial_repo_prefix"/*) ;;
          *)
            printf "WORLDMM_REMOTE_REPO is outside approved prefix: %s\n" \
              "$initial_repo_resolved" >&2
            exit 1
            ;;
        esac
        env_file="$WORLDMM_REMOTE_REPO/.env.worldmm"
        if [ ! -f "$env_file" ] || [ -L "$env_file" ] || \
          [ ! -O "$env_file" ] || \
          find "$env_file" -perm /022 -print -quit | grep -q .; then
          printf \
            ".env.worldmm is required; ownership/permissions unsafe: %s\n" \
            "$env_file" >&2
          exit 1
        fi
        set -a
        source "$env_file"
        set +a
        if [ -n "$incoming_run_id" ] && \
          [ "$WORLDMM_RUN_ID" != "$incoming_run_id" ]; then
          printf ".env.worldmm changed pinned WORLDMM_RUN_ID: %s -> %s\n" \
            "$incoming_run_id" "$WORLDMM_RUN_ID" >&2
          exit 1
        fi
        unset PYTHONPATH PYTHONHOME
        export PYTHONDONTWRITEBYTECODE=1
        export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1
        : "${WORLDMM_RUN_ID:?WORLDMM_RUN_ID must be pinned before preflight}"
        if [ "$WORLDMM_RUN_ID" = "REPLACE_WITH_APPROVED_RUN_ID" ]; then
          printf "replace the placeholder WORLDMM_RUN_ID before preflight\n" >&2
          exit 1
        fi
        if ! [[ "$WORLDMM_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
          printf "WORLDMM_RUN_ID has unsafe characters: %s\n" "$WORLDMM_RUN_ID" >&2
          exit 1
        fi
        : "${WORLDMM_OUTPUT_ROOT:=/repo/VTteam/bongh.park/outputs/${WORLDMM_RUN_ID}}"
        repo_resolved="$(realpath -e "$WORLDMM_REMOTE_REPO")"
        repo_prefix_resolved="$(realpath -e "$WORLDMM_APPROVED_REPO_PREFIX")"
        case "$repo_resolved" in
          "$repo_prefix_resolved"|"$repo_prefix_resolved"/*) ;;
          *)
            printf "WORLDMM_REMOTE_REPO is outside approved prefix: %s\n" \
              "$repo_resolved" >&2
            exit 1
            ;;
        esac
        case "$WORLDMM_OUTPUT_ROOT" in
          */"$WORLDMM_RUN_ID") ;;
          *)
            printf "WORLDMM_OUTPUT_ROOT must end /%s: %s\n" \
              "$WORLDMM_RUN_ID" "$WORLDMM_OUTPUT_ROOT" >&2
            exit 1
            ;;
        esac
        : "${WORLDMM_CPU_PARTITION:=cpu-prepro-queue}"
        : "${WORLDMM_GPU_PARTITION:=gpu-vtt-queue}"
        : "${WORLDMM_DAG_PHASE:=preflight}"
        : "${WORLDMM_EXECUTION_PROFILE:=probe}"
        : "${SMVQA_DATA_ROOT:=/groups/VTteam/datasets/SuperMemory-VQA/ingested}"
        : "${SMVQA_FRAME_ROOT:=$SMVQA_DATA_ROOT/frames}"
        : "${GEMMA_MODEL_PATH:=/repo/VTteam/bongh.park/gemma-4-e2b-it}"
        default_memory_model=/repo/VTteam/bongh.park/outputs/models/qwen3-vl
        : "${WORLDMM_MEMORY_MODEL_PATH:=$default_memory_model}"
        case "$WORLDMM_DAG_PHASE" in
          preflight|run) ;;
          *)
            printf "WORLDMM_DAG_PHASE must be preflight or run: %s\n" \
              "$WORLDMM_DAG_PHASE" >&2
            exit 1
            ;;
        esac
        if [ "${1:-}" = "--reconcile-unknown-sbatch" ]; then
          # This recovery surface intentionally runs before output-state and lock
          # checks: its sole authority is the immutable pre-sbatch descriptor.
          reconciliation_root="$WORLDMM_OUTPUT_ROOT/summary"
          reconciliation_journal="$reconciliation_root/dag_submit.${WORLDMM_DAG_PHASE}.attempts"
          reconciliation_lock="$reconciliation_root/dag_submit.${WORLDMM_DAG_PHASE}.lock"
          [ -d "$reconciliation_root" ] && [ -f "$reconciliation_journal" ] || {
            printf "unknown-sbatch reconciliation requires an attempt journal\n" >&2
            exit 1
          }
          WORLDMM_RECONCILIATION_ROOT="$reconciliation_root" \
            WORLDMM_RECONCILIATION_JOURNAL="$reconciliation_journal" \
            WORLDMM_SQUEUE="${WORLDMM_SQUEUE:-/opt/slurm/bin/squeue}" \
            WORLDMM_SACCT="${WORLDMM_SACCT:-/opt/slurm/bin/sacct}" \
            "$WORLDMM_REMOTE_REPO/.venv/bin/python" - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path(os.environ["WORLDMM_RECONCILIATION_ROOT"])
journal = Path(os.environ["WORLDMM_RECONCILIATION_JOURNAL"])
unknown: dict[str, dict[str, object]] = {}
resolved: set[str] = set()
for line in journal.read_text(encoding="utf-8").splitlines():
    value = json.loads(line)
    if value.get("event") == "submission-unknown-before-sbatch":
        identity = value.get("identity")
        if isinstance(identity, str):
            unknown[identity] = value
    elif value.get("event") == "submission-reconciled":
        identity = value.get("identity")
        if isinstance(identity, str):
            resolved.add(identity)
pending = {key: value for key, value in unknown.items() if key not in resolved}
rows: list[str] = []
for command in (
    [os.environ["WORLDMM_SQUEUE"], "--noheader", "--format=%i|%j|%k"],
    [os.environ["WORLDMM_SACCT"], "--noheader", "--parsable2",
     "--format=JobIDRaw,JobName,Comment"],
):
    try:
        rows.extend(
            subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
            .splitlines()
        )
    except (OSError, subprocess.CalledProcessError):
        continue
complete = True
for identity, descriptor in pending.items():
    stage = descriptor["stage"]
    name = f"worldmm-{descriptor['run_id']}-{stage}"
    matches = {
        row.split("|", 1)[0].strip().split(".", 1)[0]
        for row in rows
        if len(row.split("|")) >= 3
        and row.split("|", 1)[0].strip().split(".", 1)[0].isdigit()
        and row.split("|", 2)[1].strip() == name
        and row.split("|", 2)[2].strip() == identity
    }
    if len(matches) > 1:
        raise SystemExit(f"ambiguous Slurm reconciliation for {identity}: {sorted(matches)}")
    job_id = next(iter(matches), None)
    payload = {
        "schema_version": 1, "kind": "SubmissionReconciliationV1",
        "descriptor": descriptor, "outcome": "unique-job" if job_id else "no-job",
        "job_id": job_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    target = root / (
        "reconciliation." + hashlib.sha256(identity.encode()).hexdigest() + ".json"
    )
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        complete = False
        continue
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "schema_version": 1, "event": "submission-reconciled",
            **descriptor, "job_id": job_id or "", "outcome": payload["outcome"],
        }, sort_keys=True, separators=(",", ":")) + "\n")
if not complete:
    raise SystemExit("reconciliation descriptor already exists; lock retained")
PY
          rm -f "$reconciliation_lock"
          exit 0
        fi
        output_root_resolved="$(realpath -m "$WORLDMM_OUTPUT_ROOT")"
        output_prefix_resolved="$(realpath -e "$WORLDMM_APPROVED_OUTPUT_PREFIX")"
        case "$output_root_resolved" in
          "$output_prefix_resolved"/*) ;;
          *)
            printf "WORLDMM_OUTPUT_ROOT is outside approved prefix: %s\n" \
              "$output_root_resolved" >&2
            exit 1
            ;;
        esac
        if [ "$WORLDMM_DAG_PHASE" = "preflight" ]; then
          if [ -e "$WORLDMM_OUTPUT_ROOT" ] || [ -L "$WORLDMM_OUTPUT_ROOT" ]; then
            printf "preflight requires a new run output root: %s\n" \
              "$WORLDMM_OUTPUT_ROOT" >&2
            exit 1
          fi
          mkdir -m 700 "$WORLDMM_OUTPUT_ROOT"
        elif [ ! -d "$WORLDMM_OUTPUT_ROOT" ] || \
          [ -L "$WORLDMM_OUTPUT_ROOT" ] || [ ! -O "$WORLDMM_OUTPUT_ROOT" ] || \
          find "$WORLDMM_OUTPUT_ROOT" -maxdepth 0 -perm /022 -print -quit \
            | grep -q .; then
          printf "run output root ownership or permissions are unsafe: %s\n" \
            "$WORLDMM_OUTPUT_ROOT" >&2
          exit 1
        fi
        if [ "$WORLDMM_DAG_PHASE" = "run" ]; then
          approved_submitter="$WORLDMM_OUTPUT_ROOT/code_snapshot/remote-plan"
          approved_submitter+="/submit_worldmm_smvqa_dag.sh"
          if [ ! -x "$approved_submitter" ] || \
            [ "$(realpath -e "${BASH_SOURCE[0]}")" != \
              "$(realpath -e "$approved_submitter")" ]; then
            printf "run phase must use approved snapshot submitter: %s\n" \
              "$approved_submitter" >&2
            exit 1
          fi
        fi
        case "$WORLDMM_EXECUTION_PROFILE" in
          probe)
            : "${WORLDMM_PROBE_FIXTURE:?WORLDMM_PROBE_FIXTURE is required for probe}"
            RUN_FIXTURE="$WORLDMM_PROBE_FIXTURE"
            probe_fixture_resolved="$(realpath -e "$RUN_FIXTURE")"
            output_root_resolved="$(realpath -m "$WORLDMM_OUTPUT_ROOT")"
            case "$probe_fixture_resolved" in
              "$output_root_resolved"|"$output_root_resolved"/*)
                printf "probe fixture must be outside WORLDMM_OUTPUT_ROOT\n" >&2
                exit 1
                ;;
            esac
            : "${WORLDMM_REMOTE_NODES:=1}"
            : "${WORLDMM_GPUS_PER_NODE:=1}"
            if ! [[ "$WORLDMM_REMOTE_NODES" =~ ^[1-9][0-9]*$ ]] || \
              ! [[ "$WORLDMM_GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
              printf "probe GPU resources must be positive integers\n" >&2
              exit 1
            fi
            if [ "$WORLDMM_REMOTE_NODES" -gt 1 ] || \
              [ "$WORLDMM_GPUS_PER_NODE" -gt 1 ]; then
              printf "probe profile is limited to 1 node x 1 GPU\n" >&2
              exit 1
            fi
            ;;
          full)
            RUN_FIXTURE="$SMVQA_DATA_ROOT"
            : "${WORLDMM_REMOTE_NODES:=10}"
            : "${WORLDMM_GPUS_PER_NODE:=8}"
            ;;
          *)
            printf "WORLDMM_EXECUTION_PROFILE must be probe or full: %s\n" \
              "$WORLDMM_EXECUTION_PROFILE" >&2
            exit 1
            ;;
        esac
        : "${WORLDMM_PREFLIGHT_NODES:=1}"
        : "${WORLDMM_PREFLIGHT_CPUS:=32}"
        : "${WORLDMM_PREFLIGHT_MEM:=128G}"
        : "${WORLDMM_PREFLIGHT_TIME:=02:00:00}"
        : "${WORLDMM_TEACHER_NODES:=$WORLDMM_REMOTE_NODES}"
        : "${WORLDMM_TEACHER_GPUS_PER_NODE:=$WORLDMM_GPUS_PER_NODE}"
        : "${WORLDMM_TEACHER_CPUS:=64}"
        : "${WORLDMM_TEACHER_MEM:=0}"
        : "${WORLDMM_TEACHER_TIME:=12:00:00}"
        : "${WORLDMM_MATERIALIZE_NODES:=1}"
        : "${WORLDMM_MATERIALIZE_CPUS:=64}"
        : "${WORLDMM_MATERIALIZE_MEM:=256G}"
        : "${WORLDMM_MATERIALIZE_TIME:=06:00:00}"
        : "${WORLDMM_TRAIN_NODES:=$WORLDMM_REMOTE_NODES}"
        : "${WORLDMM_TRAIN_GPUS_PER_NODE:=$WORLDMM_GPUS_PER_NODE}"
        : "${WORLDMM_TRAIN_CPUS:=64}"
        : "${WORLDMM_TRAIN_MEM:=0}"
        : "${WORLDMM_TRAIN_TIME:=24:00:00}"
        : "${WORLDMM_TRAIN_EPOCHS:=1}"
        : "${WORLDMM_TRAIN_BATCH_SIZE:=8}"
        : "${WORLDMM_TRAIN_HIDDEN_DIM:=32}"
        : "${WORLDMM_TRAIN_LEARNING_RATE:=0.001}"
        : "${WORLDMM_TRAIN_RESUME:=}"
        : "${WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW:=4096}"
        : "${WORLDMM_REPORT_NODES:=1}"
        : "${WORLDMM_REPORT_CPUS:=32}"
        : "${WORLDMM_REPORT_MEM:=128G}"
        : "${WORLDMM_REPORT_TIME:=02:00:00}"
        if [ "$WORLDMM_CPU_PARTITION" != "cpu-prepro-queue" ] || \
          [ "$WORLDMM_GPU_PARTITION" != "gpu-vtt-queue" ]; then
          printf "partitions must be cpu-prepro-queue and gpu-vtt-queue\n" >&2
          exit 1
        fi
        for resource in \
          "$WORLDMM_PREFLIGHT_NODES" "$WORLDMM_PREFLIGHT_CPUS" \
          "$WORLDMM_MATERIALIZE_NODES" "$WORLDMM_MATERIALIZE_CPUS" \
          "$WORLDMM_REPORT_NODES" "$WORLDMM_REPORT_CPUS" \
          "$WORLDMM_TEACHER_CPUS" "$WORLDMM_TRAIN_CPUS"; do
          if ! [[ "$resource" =~ ^[1-9][0-9]*$ ]]; then
            printf "CPU stage resources must be positive integers\n" >&2
            exit 1
          fi
        done
        for nodes in \
          "$WORLDMM_PREFLIGHT_NODES" "$WORLDMM_MATERIALIZE_NODES" \
          "$WORLDMM_REPORT_NODES"; do
          if [ "$nodes" -gt 6 ]; then
            printf "CPU stage resources exceed company limit: 6 nodes\n" >&2
            exit 1
          fi
        done
        if [ "$WORLDMM_PREFLIGHT_NODES" -ne 1 ]; then
          printf "preflight is limited to exactly 1 CPU node\n" >&2
          exit 1
        fi
        for cpus in \
          "$WORLDMM_PREFLIGHT_CPUS" "$WORLDMM_MATERIALIZE_CPUS" \
          "$WORLDMM_REPORT_CPUS" "$WORLDMM_TEACHER_CPUS" \
          "$WORLDMM_TRAIN_CPUS"; do
          if [ "$cpus" -gt 128 ]; then
            printf "CPU request exceeds company per-task limit: 128\n" >&2
            exit 1
          fi
        done
        validate_cpu_memory() {
          local value="$1" amount unit
          if ! [[ "$value" =~ ^([1-9][0-9]*)([MG])$ ]]; then
            printf "CPU memory must use a positive M/G value: %s\n" "$value" >&2
            return 1
          fi
          amount="${BASH_REMATCH[1]}"
          unit="${BASH_REMATCH[2]}"
          if { [ "$unit" = G ] && [ "$amount" -gt 512 ]; } || \
            { [ "$unit" = M ] && [ "$amount" -gt 524288 ]; }; then
            printf "CPU memory exceeds company limit: 512G\n" >&2
            return 1
          fi
        }
        validate_time_limit() {
          local value="$1" hours minutes seconds
          if ! [[ "$value" =~ ^([0-9]{2}):([0-9]{2}):([0-9]{2})$ ]]; then
            printf "time limit must use HH:MM:SS: %s\n" "$value" >&2
            return 1
          fi
          hours="$((10#${BASH_REMATCH[1]}))"
          minutes="$((10#${BASH_REMATCH[2]}))"
          seconds="$((10#${BASH_REMATCH[3]}))"
          if [ "$hours" -gt 24 ] || [ "$minutes" -gt 59 ] || \
            [ "$seconds" -gt 59 ] || \
            { [ "$hours" -eq 24 ] && \
              { [ "$minutes" -ne 0 ] || [ "$seconds" -ne 0 ]; }; }; then
            printf "time limit exceeds company limit: 24:00:00\n" >&2
            return 1
          fi
        }
        validate_gpu_memory() {
          local value="$1" amount unit
          if [ "$value" = 0 ]; then
            return 0
          fi
          if ! [[ "$value" =~ ^([1-9][0-9]*)([MG])$ ]]; then
            printf "GPU-stage memory must be 0 or a positive M/G value: %s\n" \
              "$value" >&2
            return 1
          fi
          amount="${BASH_REMATCH[1]}"
          unit="${BASH_REMATCH[2]}"
          if { [ "$unit" = G ] && [ "$amount" -gt 2048 ]; } || \
            { [ "$unit" = M ] && [ "$amount" -gt 2097152 ]; }; then
            printf "GPU-stage memory exceeds company limit: 2048G\n" >&2
            return 1
          fi
        }
        validate_cpu_memory "$WORLDMM_PREFLIGHT_MEM"
        validate_cpu_memory "$WORLDMM_MATERIALIZE_MEM"
        validate_cpu_memory "$WORLDMM_REPORT_MEM"
        validate_time_limit "$WORLDMM_PREFLIGHT_TIME"
        validate_time_limit "$WORLDMM_MATERIALIZE_TIME"
        validate_time_limit "$WORLDMM_REPORT_TIME"
        validate_time_limit "$WORLDMM_TEACHER_TIME"
        validate_time_limit "$WORLDMM_TRAIN_TIME"
        validate_gpu_memory "$WORLDMM_TEACHER_MEM"
        validate_gpu_memory "$WORLDMM_TRAIN_MEM"
        for resource in \
          "$WORLDMM_TEACHER_NODES" "$WORLDMM_TEACHER_GPUS_PER_NODE" \
          "$WORLDMM_TRAIN_NODES" "$WORLDMM_TRAIN_GPUS_PER_NODE"; do
          if ! [[ "$resource" =~ ^[1-9][0-9]*$ ]]; then
            printf "GPU stage resources must be positive integers\n" >&2
            exit 1
          fi
        done
        if [ "$WORLDMM_TEACHER_NODES" -gt 10 ] || \
          [ "$WORLDMM_TRAIN_NODES" -gt 10 ] || \
          [ "$WORLDMM_TEACHER_GPUS_PER_NODE" -gt 8 ] || \
          [ "$WORLDMM_TRAIN_GPUS_PER_NODE" -gt 8 ]; then
          printf "GPU stage resources exceed company limit: 10 nodes x 8 GPUs\n" >&2
          exit 1
        fi
        if [ "$WORLDMM_EXECUTION_PROFILE" = "probe" ] && \
          { [ "$WORLDMM_TEACHER_NODES" -gt 1 ] || \
            [ "$WORLDMM_TEACHER_GPUS_PER_NODE" -gt 1 ] || \
            [ "$WORLDMM_TRAIN_NODES" -gt 1 ] || \
            [ "$WORLDMM_TRAIN_GPUS_PER_NODE" -gt 1 ]; }; then
          printf "probe profile stage resources are limited to 1 node x 1 GPU\n" >&2
          exit 1
        fi
        export WORLDMM_TRAIN_EPOCHS WORLDMM_TRAIN_BATCH_SIZE
        export WORLDMM_TRAIN_HIDDEN_DIM WORLDMM_TRAIN_LEARNING_RATE
        export WORLDMM_CPU_PARTITION WORLDMM_GPU_PARTITION
        export WORLDMM_PREFLIGHT_NODES WORLDMM_PREFLIGHT_CPUS
        export WORLDMM_PREFLIGHT_MEM WORLDMM_PREFLIGHT_TIME
        export WORLDMM_TEACHER_NODES WORLDMM_TEACHER_GPUS_PER_NODE
        export WORLDMM_TEACHER_CPUS WORLDMM_TEACHER_MEM WORLDMM_TEACHER_TIME
        export WORLDMM_MATERIALIZE_NODES WORLDMM_MATERIALIZE_CPUS
        export WORLDMM_MATERIALIZE_MEM WORLDMM_MATERIALIZE_TIME
        export WORLDMM_TRAIN_NODES WORLDMM_TRAIN_GPUS_PER_NODE
        export WORLDMM_TRAIN_CPUS WORLDMM_TRAIN_MEM WORLDMM_TRAIN_TIME
        export WORLDMM_REPORT_NODES WORLDMM_REPORT_CPUS
        export WORLDMM_REPORT_MEM WORLDMM_REPORT_TIME
        export WORLDMM_APPROVED_DATA_PREFIX WORLDMM_APPROVED_REPO_PREFIX
        export WORLDMM_APPROVED_OUTPUT_PREFIX

        verify_approved_python_runtime() {
          local python_runtime_root python_runtime_files python_loader_inventory
          local python_runtime_inventory python_runtime_resolved
          local unsafe_link unsafe_pth runtime_link runtime_target
          local pth_file pth_dir pth_line pth_candidate pth_resolved
          local inventory_current base_roots base_files base_inventory base_prefix
          local base_executable stdlib platstdlib root
          python_runtime_root="$WORLDMM_REMOTE_REPO/.venv"
          python_runtime_files="$WORLDMM_OUTPUT_ROOT/diagnostics/python_runtime.sha256"
          python_runtime_inventory="$WORLDMM_OUTPUT_ROOT/diagnostics/python_runtime.files.sha256"
          python_loader_inventory="$WORLDMM_OUTPUT_ROOT/diagnostics/python_runtime.loader.sha256"
          base_roots="$WORLDMM_OUTPUT_ROOT/diagnostics/python_base_roots.tsv"
          base_files="$WORLDMM_OUTPUT_ROOT/diagnostics/python_base_runtime.sha256"
          base_inventory="$WORLDMM_OUTPUT_ROOT/diagnostics/"
          base_inventory+="python_base_runtime.files.sha256"
          if [ ! -d "$python_runtime_root" ] || [ -L "$python_runtime_root" ] || \
            [ ! -s "$python_runtime_files" ] || \
            [ ! -s "$python_runtime_inventory" ] || \
            [ ! -s "$python_loader_inventory" ] || \
            [ ! -s "$base_roots" ] || [ ! -s "$base_files" ] || \
            [ ! -s "$base_inventory" ]; then
            printf "approved Python runtime or its manifests are missing\n" >&2
            return 1
          fi
          python_runtime_resolved="$(realpath -e "$python_runtime_root")"
          unsafe_link="$(
            find "$python_runtime_root" -type l -print0 | \
              while IFS= read -r -d '' runtime_link; do
                if [ ! -e "$runtime_link" ] || [ -d "$runtime_link" ]; then
                  printf "%s" "$runtime_link"
                  break
                fi
              done
          )"
          if [ -n "$unsafe_link" ]; then
            printf "Python runtime has unsafe directory/dangling symlink: %s\n" \
              "$unsafe_link" >&2
            return 1
          fi
          unsafe_pth="$(
            find "$python_runtime_root" \( -type f -o -type l \) \
              -name '*.pth' -print0 | \
              while IFS= read -r -d '' pth_file; do
                pth_dir="$(dirname "$pth_file")"
                while IFS= read -r pth_line || [ -n "$pth_line" ]; do
                  pth_line="${pth_line%$'\r'}"
                  case "$pth_line" in
                    ''|'#'*) continue ;;
                    'import '*|$'import\t'*)
                      printf "%s" "$pth_file"
                      break 2
                      ;;
                    /*) pth_candidate="$pth_line" ;;
                    *) pth_candidate="$pth_dir/$pth_line" ;;
                  esac
                  pth_resolved="$(realpath -m "$pth_candidate")"
                  case "$pth_resolved" in
                    "$python_runtime_resolved"|"$python_runtime_resolved"/*) ;;
                    *)
                      printf "%s" "$pth_file"
                      break 2
                      ;;
                  esac
                done < "$pth_file"
              done
          )"
          if [ -n "$unsafe_pth" ]; then
            printf "Python runtime has unsafe executable/external .pth: %s\n" \
              "$unsafe_pth" >&2
            return 1
          fi
          if ! (cd "$python_runtime_root" && \
            sha256sum --check --status "$python_runtime_files"); then
            printf "approved Python runtime content changed\n" >&2
            return 1
          fi
          inventory_current="${python_runtime_inventory}.verify.$$"
          (
            cd "$python_runtime_root"
            find . \( -type f -o -type l \) -printf '%y %p %l\0' \
              | sort -z | sha256sum
          ) > "$inventory_current"
          if ! cmp -s "$python_runtime_inventory" "$inventory_current"; then
            rm -f "$inventory_current"
            printf "approved Python runtime file inventory changed\n" >&2
            return 1
          fi
          rm -f "$inventory_current"
          inventory_current="${python_loader_inventory}.verify.$$"
          ldd "$python_runtime_root/bin/python" | while IFS= read -r loader; do
            printf '%s\n' "${loader%% (*}"
          done | LC_ALL=C sort | sha256sum > "$inventory_current"
          if ! cmp -s "$python_loader_inventory" "$inventory_current"; then
            rm -f "$inventory_current"
            printf "approved Python interpreter loader/shared-library closure changed\n" >&2
            return 1
          fi
          rm -f "$inventory_current"
          base_prefix="$(awk -F '\t' '$1 == "base_prefix" {print $2}' "$base_roots")"
          base_executable="$(awk -F '\t' '$1 == "executable" {print $2}' "$base_roots")"
          stdlib="$(awk -F '\t' '$1 == "stdlib" {print $2}' "$base_roots")"
          platstdlib="$(awk -F '\t' '$1 == "platstdlib" {print $2}' "$base_roots")"
          if [ "$(wc -l < "$base_roots")" -ne 4 ] || \
            [ -z "$base_prefix" ] || [ -z "$base_executable" ] || \
            [ -z "$stdlib" ] || [ -z "$platstdlib" ] || \
            [ "$(realpath -e "$base_executable")" != "$base_executable" ]; then
            printf "approved Python base runtime roots are invalid\n" >&2
            return 1
          fi
          for root in "$base_executable" "$stdlib" "$platstdlib"; do
            if [ ! -e "$root" ]; then
              printf "approved Python base runtime path is missing: %s\n" \
                "$root" >&2
              return 1
            fi
            case "$root" in
              "$base_prefix"|"$base_prefix"/*) ;;
              *)
                printf "approved Python base runtime path escapes prefix: %s\n" \
                  "$root" >&2
                return 1
                ;;
            esac
          done
          if ! sha256sum --check --status "$base_files"; then
            printf "approved Python base runtime content changed\n" >&2
            return 1
          fi
          inventory_current="${base_inventory}.verify.$$"
          find "$base_executable" "$stdlib" "$platstdlib" \
            \( -type f -o -type l \) -printf '%y %p %l\0' \
            | sort -zu | sha256sum > "$inventory_current"
          if ! cmp -s "$base_inventory" "$inventory_current"; then
            rm -f "$inventory_current"
            printf "approved Python base runtime file inventory changed\n" >&2
            return 1
          fi
          rm -f "$inventory_current"
        }

        if [ "$WORLDMM_DAG_PHASE" = "run" ]; then
          if [ -n "${WORLDMM_GCUT3R_EXTRACTOR:-}" ]; then
            approved_teacher_nodes="$WORLDMM_TEACHER_NODES"
            approved_teacher_gpus="$WORLDMM_TEACHER_GPUS_PER_NODE"
            teacher_mode=extractor
            teacher_path="$WORLDMM_GCUT3R_EXTRACTOR"
          else
            approved_teacher_nodes=1
            approved_teacher_gpus=0
            teacher_mode=cache
            : "${WORLDMM_TEACHER_CACHE_INPUT:?teacher cache input required}"
            teacher_path="$WORLDMM_TEACHER_CACHE_INPUT"
          fi
          : "${WORLDMM_SPATIAL_INFER_EXE:?spatial inference executable required}"
          : "${WORLDMM_STUDENT_SUPERVISION_INPUT:?student supervision required}"
          : "${WORLDMM_APPROVAL_FILE:?WORLDMM_APPROVAL_FILE is required for run phase}"
          : "${WORLDMM_APPROVER:?WORLDMM_APPROVER is required for run phase}"
          preflight_marker="$WORLDMM_OUTPUT_ROOT/diagnostics/preflight.completed"
          preflight_inputs="$WORLDMM_OUTPUT_ROOT/diagnostics/preflight_inputs.sha256"
          env_contract="$WORLDMM_OUTPUT_ROOT/diagnostics/env_contract.json"
          frame_assets="$WORLDMM_OUTPUT_ROOT/diagnostics/frame_assets.sha256"
          if [ ! -s "$preflight_marker" ] || [ ! -s "$preflight_inputs" ] || \
            [ ! -s "$env_contract" ] || [ ! -s "$frame_assets" ]; then
            printf "completed CPU preflight is required before run phase\n" >&2
            exit 1
          fi
          verify_approved_python_runtime
          preflight_inputs_sha256="$(
            sha256sum "$preflight_inputs" | cut -d ' ' -f 1
          )"
          env_contract_sha256="$(sha256sum "$env_contract" | cut -d ' ' -f 1)"
          approval_path="$(realpath -e "$WORLDMM_APPROVAL_FILE")"
          approved_repo="$(realpath -e "$WORLDMM_REMOTE_REPO")"
          case "$approval_path" in
            "$approved_repo"|"$approved_repo"/*) ;;
            *)
              printf "approval file is outside WORLDMM_REMOTE_REPO: %s\n" \
                "$approval_path" >&2
              exit 1
              ;;
          esac
          if [ ! -O "$approval_path" ] || \
            find "$approval_path" -perm /022 -print -quit | grep -q .; then
            printf "approval file permissions are unsafe\n" >&2
            exit 1
          fi
          "$WORLDMM_REMOTE_REPO/.venv/bin/python" - \
            "$approval_path" "$WORLDMM_RUN_ID" \
            "$WORLDMM_EXECUTION_PROFILE" "$approved_teacher_nodes" \
            "$approved_teacher_gpus" "$WORLDMM_TRAIN_NODES" \
            "$WORLDMM_TRAIN_GPUS_PER_NODE" "$WORLDMM_APPROVER" \
            "$preflight_inputs_sha256" "$env_contract_sha256" \
            "$env_contract" "$RUN_FIXTURE" "$SMVQA_DATA_ROOT" \
            "$SMVQA_FRAME_ROOT" "$GEMMA_MODEL_PATH" \
            "$WORLDMM_MEMORY_MODEL_PATH" "$WORLDMM_SPATIAL_INFER_EXE" \
            "$WORLDMM_STUDENT_SUPERVISION_INPUT" "$teacher_mode" \
            "$teacher_path" "$WORLDMM_TRAIN_RESUME" \
            "$WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW" <<'PY'
import json
import os
import sys
from importlib.metadata import distributions
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"approval file is not a file: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "run_id": sys.argv[2],
    "profile": sys.argv[3],
    "nodes": int(sys.argv[6]),
    "gpus_per_node": int(sys.argv[7]),
    "teacher_nodes": int(sys.argv[4]),
    "teacher_gpus_per_node": int(sys.argv[5]),
    "train_nodes": int(sys.argv[6]),
    "train_gpus_per_node": int(sys.argv[7]),
    "approved": True,
    "approver": sys.argv[8],
    "preflight_inputs_sha256": sys.argv[9],
    "env_contract_sha256": sys.argv[10],
}
for key, value in expected.items():
    actual = payload.get(key)
    if type(actual) is not type(value) or actual != value:
        raise SystemExit(
            f"approval mismatch for {key}: expected {value!r}, "
            f"got {actual!r}"
        )
if not sys.argv[8].strip():
    raise SystemExit("WORLDMM_APPROVER must be non-empty")
contract_path = Path(sys.argv[11])
contract = json.loads(contract_path.read_text(encoding="utf-8"))

def python_runtime() -> dict[str, object]:
    packages = sorted(
        [
            distribution.metadata.get("Name") or "",
            distribution.version,
            distribution.read_text("direct_url.json") or "",
        ]
        for distribution in distributions()
    )
    return {
        "version": sys.version,
        "executable": str(Path(sys.executable).resolve(strict=True)),
        "packages": packages,
    }

teacher_mode = sys.argv[19]
teacher_uses_gpu = teacher_mode == "extractor"
effective_teacher_resources = {
    "nodes": int(sys.argv[4]),
    "gpus_per_node": int(sys.argv[5]),
    "partition": os.environ[
        "WORLDMM_GPU_PARTITION" if teacher_uses_gpu else "WORLDMM_CPU_PARTITION"
    ],
    "cpus_per_task": int(
        os.environ[
            "WORLDMM_TEACHER_CPUS"
            if teacher_uses_gpu
            else "WORLDMM_PREFLIGHT_CPUS"
        ]
    ),
    "memory": os.environ[
        "WORLDMM_TEACHER_MEM" if teacher_uses_gpu else "WORLDMM_PREFLIGHT_MEM"
    ],
    "time": os.environ[
        "WORLDMM_TEACHER_TIME" if teacher_uses_gpu else "WORLDMM_PREFLIGHT_TIME"
    ],
}

current = {
    "schema_version": 1,
    "run_id": sys.argv[2],
    "profile": sys.argv[3],
    "run_fixture": str(Path(sys.argv[12]).resolve(strict=True)),
    "data_root": str(Path(sys.argv[13]).resolve(strict=True)),
    "frame_root": str(Path(sys.argv[14]).resolve(strict=True)),
    "gemma_model_path": str(Path(sys.argv[15]).resolve(strict=True)),
    "memory_model_path": str(Path(sys.argv[16]).resolve(strict=True)),
    "spatial_infer_exe": str(Path(sys.argv[17]).resolve(strict=True)),
    "student_supervision": str(Path(sys.argv[18]).resolve(strict=True)),
    "teacher_mode": teacher_mode,
    "teacher_path": str(Path(sys.argv[20]).resolve(strict=True)),
    "effective_teacher_resources": effective_teacher_resources,
    "train_resume": (
        str(Path(sys.argv[21]).resolve(strict=True)) if sys.argv[21] else None
    ),
    "train_epochs": int(os.environ["WORLDMM_TRAIN_EPOCHS"]),
    "train_batch_size": int(os.environ["WORLDMM_TRAIN_BATCH_SIZE"]),
    "train_hidden_dim": int(os.environ["WORLDMM_TRAIN_HIDDEN_DIM"]),
    "train_learning_rate": float(os.environ["WORLDMM_TRAIN_LEARNING_RATE"]),
    "byte_budget_per_window": int(sys.argv[22]),
    "python_runtime": python_runtime(),
    "approved_prefixes": {
        name: str(Path(os.environ[name]).resolve(strict=True))
        for name in (
            "WORLDMM_APPROVED_DATA_PREFIX",
            "WORLDMM_APPROVED_REPO_PREFIX",
            "WORLDMM_APPROVED_OUTPUT_PREFIX",
        )
    },
    "resources": {
        name: os.environ[name]
        for name in (
            "WORLDMM_CPU_PARTITION",
            "WORLDMM_GPU_PARTITION",
            "WORLDMM_PREFLIGHT_NODES",
            "WORLDMM_PREFLIGHT_CPUS",
            "WORLDMM_PREFLIGHT_MEM",
            "WORLDMM_PREFLIGHT_TIME",
            "WORLDMM_TEACHER_NODES",
            "WORLDMM_TEACHER_GPUS_PER_NODE",
            "WORLDMM_TEACHER_CPUS",
            "WORLDMM_TEACHER_MEM",
            "WORLDMM_TEACHER_TIME",
            "WORLDMM_MATERIALIZE_NODES",
            "WORLDMM_MATERIALIZE_CPUS",
            "WORLDMM_MATERIALIZE_MEM",
            "WORLDMM_MATERIALIZE_TIME",
            "WORLDMM_TRAIN_NODES",
            "WORLDMM_TRAIN_GPUS_PER_NODE",
            "WORLDMM_TRAIN_CPUS",
            "WORLDMM_TRAIN_MEM",
            "WORLDMM_TRAIN_TIME",
            "WORLDMM_REPORT_NODES",
            "WORLDMM_REPORT_CPUS",
            "WORLDMM_REPORT_MEM",
            "WORLDMM_REPORT_TIME",
        )
    },
}
if contract != current:
    raise SystemExit("current environment does not match preflight env_contract.json")
PY
          approval_sha256="$(sha256sum "$approval_path" | cut -d ' ' -f 1)"
          sha256sum --check --status "$preflight_inputs"
          sha256sum --check --status "$frame_assets"
          sha256sum --check --status \
            "$WORLDMM_OUTPUT_ROOT/diagnostics/gemma_model.sha256"
          sha256sum --check --status \
            "$WORLDMM_OUTPUT_ROOT/diagnostics/memory_model.sha256"
          verify_file_inventory() {
            local root="$1" expected="$2" temporary
            temporary="${expected}.verify.$$"
            (
              cd "$root"
              find . -type f -print0 | sort -z | sha256sum
            ) > "$temporary"
            if ! cmp -s "$expected" "$temporary"; then
              rm -f "$temporary"
              printf "file inventory changed after preflight: %s\n" "$root" >&2
              return 1
            fi
            rm -f "$temporary"
          }
          verify_file_inventory "$GEMMA_MODEL_PATH" \
            "$WORLDMM_OUTPUT_ROOT/diagnostics/gemma_model.files.sha256"
          verify_file_inventory "$WORLDMM_MEMORY_MODEL_PATH" \
            "$WORLDMM_OUTPUT_ROOT/diagnostics/memory_model.files.sha256"
          verify_file_inventory \
            "$WORLDMM_OUTPUT_ROOT/inference_inputs/frames" \
            "$WORLDMM_OUTPUT_ROOT/diagnostics/frame_assets.files.sha256"
        fi

        export WORLDMM_REMOTE_REPO WORLDMM_RUN_ID WORLDMM_OUTPUT_ROOT
        export WORLDMM_DAG_PHASE WORLDMM_EXECUTION_PROFILE
        export WORLDMM_REMOTE_NODES WORLDMM_GPUS_PER_NODE
        export RUN_FIXTURE SMVQA_DATA_ROOT SMVQA_FRAME_ROOT
        export WORLDMM_TRAIN_EPOCHS WORLDMM_TRAIN_BATCH_SIZE
        export WORLDMM_TRAIN_HIDDEN_DIM WORLDMM_TRAIN_LEARNING_RATE
        export WORLDMM_TRAIN_RESUME WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW
        WORLDMM_APPROVAL_SHA256="${approval_sha256:-}"
        export GEMMA_MODEL_PATH WORLDMM_MEMORY_MODEL_PATH
        export WORLDMM_APPROVAL_FILE WORLDMM_APPROVER WORLDMM_APPROVAL_SHA256
        export WORLDMM_GCUT3R_EXTRACTOR WORLDMM_TEACHER_CACHE_INPUT
        export WORLDMM_SPATIAL_INFER_EXE WORLDMM_STUDENT_SUPERVISION_INPUT
        export WORLDMM_TEACHER_NODES WORLDMM_TEACHER_GPUS_PER_NODE
        export WORLDMM_TRAIN_NODES WORLDMM_TRAIN_GPUS_PER_NODE
        if [ "$WORLDMM_DAG_PHASE" = "run" ]; then
          stage_script="$WORLDMM_OUTPUT_ROOT/code_snapshot/remote-plan"
          stage_script+="/run_worldmm_smvqa_stage.sh"
          if [ ! -x "$stage_script" ]; then
            printf "approved code snapshot stage runner missing: %s\n" \
              "$stage_script" >&2
            exit 1
          fi
        else
          stage_script="$WORLDMM_REMOTE_REPO/remote-plan/run_worldmm_smvqa_stage.sh"
        fi
        for managed_dir in logs summary; do
          managed_path="$WORLDMM_OUTPUT_ROOT/$managed_dir"
          if [ -L "$managed_path" ] || \
            { [ -e "$managed_path" ] && [ ! -d "$managed_path" ]; }; then
            printf "managed output path is not a real directory: %s\n" \
              "$managed_path" >&2
            exit 1
          fi
          if [ ! -e "$managed_path" ]; then
            mkdir -m 700 "$managed_path"
          elif [ ! -O "$managed_path" ] || \
            find "$managed_path" -maxdepth 0 -perm /022 -print -quit \
              | grep -q .; then
            printf "managed output directory permissions are unsafe: %s\n" \
              "$managed_path" >&2
            exit 1
          fi
        done
        submit_lock="$WORLDMM_OUTPUT_ROOT/summary/dag_submit.${WORLDMM_DAG_PHASE}.lock"
        if ! (set -o noclobber; printf "%s\n" "$$" > "$submit_lock") 2>/dev/null; then
          printf "DAG phase already submitted or submitting: %s\n" \
            "$submit_lock" >&2
          exit 1
        fi
        partial_jobs="$WORLDMM_OUTPUT_ROOT/summary/"
        partial_jobs+="dag_jobs.${WORLDMM_DAG_PHASE}.partial"
        : > "$partial_jobs"
        submission_attempts="$WORLDMM_OUTPUT_ROOT/summary"
        submission_attempts+="/dag_submit.${WORLDMM_DAG_PHASE}.attempts"
        : > "$submission_attempts"
        SCANCEL="${WORLDMM_SCANCEL:-/opt/slurm/bin/scancel}"
        cleanup_partial_submission() {
          local status="$?"
          if [ "$status" -eq 0 ]; then
            return
          fi
          mapfile -t submitted_rows < "$partial_jobs"
          local row stage job intent
          local -a submitted_job_ids=()
          for row in "${submitted_rows[@]}"; do
            stage="${row%%=*}"
            job="${row#*=}"
            case "$stage" in
              preflight_ingest|teacher_extract|merge_materialize|train|build_memory|\
              student_infer_retrieve|qa|metrics_report) ;;
              *)
                printf "cancellation is forbidden for gate or terminal stage: %s\n" \
                  "$stage" >&2
                return "$status"
                ;;
            esac
            [[ "$job" =~ ^[0-9]+$ ]] || continue
            submitted_job_ids+=("$job")
            intent="$WORLDMM_OUTPUT_ROOT/summary/cancellation_intent.${WORLDMM_DAG_PHASE}.${stage}.${job}.json"
            WORLDMM_CANCELLATION_INTENT="$intent" \
              WORLDMM_CANCELLATION_STAGE="$stage" \
              WORLDMM_CANCELLATION_JOB="$job" \
              "$WORLDMM_REMOTE_REPO/.venv/bin/python" - <<'PY'
import hashlib
import json
import os

target = os.environ["WORLDMM_CANCELLATION_INTENT"]
payload = {
    "schema_version": 1,
    "kind": "CancellationIntentV1",
    "run_id": os.environ["WORLDMM_RUN_ID"],
    "phase": os.environ["WORLDMM_DAG_PHASE"],
    "stage": os.environ["WORLDMM_CANCELLATION_STAGE"],
    "job_id": os.environ["WORLDMM_CANCELLATION_JOB"],
    "attempt": os.environ["WORLDMM_CANCELLATION_JOB"],
    "scope": "producer-only",
}
payload["payload_sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    os.write(fd, encoded)
    os.fsync(fd)
finally:
    os.close(fd)
PY
          done
          if [ "${#submitted_job_ids[@]}" -eq 0 ] && \
            [ ! -s "$submission_attempts" ]; then
            rm -f "$submit_lock"
          elif [ "${#submitted_job_ids[@]}" -eq 0 ]; then
            printf \
              "sbatch returned no trustworthy job ID; keeping lock: %s\n" \
              "$submit_lock" >&2
          elif [ -x "$SCANCEL" ] && \
            "$SCANCEL" "${submitted_job_ids[@]}"; then
            for row in "${submitted_rows[@]}"; do
              stage="${row%%=*}"
              job="${row#*=}"
              [[ "$job" =~ ^[0-9]+$ ]] || continue
              printf '{"schema_version":1,"event":"cancellation-accounting","run_id":"%s","stage":"%s","job_id":"%s","attempt":"%s","operational":"cancelled","scientific":"not_decidable"}\n' \
                "$WORLDMM_RUN_ID" "$stage" "$job" "$job" >> "$submission_attempts"
            done
            printf \
              "partial DAG jobs were cancelled; keeping lock, use a new run ID: %s\n" \
              "$submit_lock" >&2
          else
            printf \
              "partial DAG submission may still have live jobs; keeping lock: %s\n" \
              "$submit_lock" >&2
          fi
          return "$status"
        }
        trap cleanup_partial_submission EXIT

        submit_stage() {
          local stage="$1" partition="$2" nodes="$3" gpus="$4"
          local cpus="$5" mem="$6" time_limit="$7" dependency="$8"
          local dependency_kind="${9:-afterok}"
          local raw_job_id job_id stage_exports export_name export_value
          for export_name in \
            WORLDMM_REMOTE_REPO WORLDMM_RUN_ID WORLDMM_OUTPUT_ROOT \
            WORLDMM_DAG_PHASE WORLDMM_EXECUTION_PROFILE \
            WORLDMM_APPROVED_DATA_PREFIX WORLDMM_APPROVED_REPO_PREFIX \
            WORLDMM_APPROVED_OUTPUT_PREFIX RUN_FIXTURE SMVQA_DATA_ROOT \
            SMVQA_FRAME_ROOT GEMMA_MODEL_PATH WORLDMM_MEMORY_MODEL_PATH \
            WORLDMM_TRAIN_EPOCHS WORLDMM_TRAIN_BATCH_SIZE \
            WORLDMM_TRAIN_HIDDEN_DIM WORLDMM_TRAIN_LEARNING_RATE \
            WORLDMM_TRAIN_RESUME WORLDMM_TRAIN_NODES \
            WORLDMM_TRAIN_GPUS_PER_NODE WORLDMM_TEACHER_NODES \
            WORLDMM_TEACHER_GPUS_PER_NODE \
            WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW WORLDMM_APPROVAL_FILE \
            WORLDMM_APPROVER WORLDMM_APPROVAL_SHA256 \
            WORLDMM_GCUT3R_EXTRACTOR WORLDMM_TEACHER_CACHE_INPUT \
            WORLDMM_SPATIAL_INFER_EXE WORLDMM_STUDENT_SUPERVISION_INPUT \
            WORLDMM_CPU_PARTITION WORLDMM_GPU_PARTITION \
            WORLDMM_PREFLIGHT_NODES WORLDMM_PREFLIGHT_CPUS \
            WORLDMM_PREFLIGHT_MEM WORLDMM_PREFLIGHT_TIME \
            WORLDMM_TEACHER_CPUS WORLDMM_TEACHER_MEM WORLDMM_TEACHER_TIME \
            WORLDMM_MATERIALIZE_NODES WORLDMM_MATERIALIZE_CPUS \
            WORLDMM_MATERIALIZE_MEM WORLDMM_MATERIALIZE_TIME \
            WORLDMM_TRAIN_CPUS WORLDMM_TRAIN_MEM WORLDMM_TRAIN_TIME \
            WORLDMM_REPORT_NODES WORLDMM_REPORT_CPUS WORLDMM_REPORT_MEM \
            WORLDMM_REPORT_TIME; do
            export_value="${!export_name:-}"
            case "$export_value" in
              *','*|*$'\n'*|*$'\r'*)
                printf "sbatch export value has comma/newline: %s\n" \
                  "$export_name" >&2
                return 1
                ;;
            esac
          done
          stage_exports="WORLDMM_STAGE=$stage"
          stage_exports+=",WORLDMM_STAGE_GPUS_PER_NODE=$gpus"
          stage_exports+=",WORLDMM_SMVQA_REMOTE_APPROVED=1"
          stage_exports+=",WORLDMM_REMOTE_REPO=$WORLDMM_REMOTE_REPO"
          stage_exports+=",WORLDMM_RUN_ID=$WORLDMM_RUN_ID"
          stage_exports+=",WORLDMM_OUTPUT_ROOT=$WORLDMM_OUTPUT_ROOT"
          stage_exports+=",WORLDMM_DAG_PHASE=$WORLDMM_DAG_PHASE"
          stage_exports+=",WORLDMM_EXECUTION_PROFILE=$WORLDMM_EXECUTION_PROFILE"
          stage_exports+=",WORLDMM_APPROVED_DATA_PREFIX=$WORLDMM_APPROVED_DATA_PREFIX"
          stage_exports+=",WORLDMM_APPROVED_REPO_PREFIX=$WORLDMM_APPROVED_REPO_PREFIX"
          stage_exports+=",WORLDMM_APPROVED_OUTPUT_PREFIX="
          stage_exports+="$WORLDMM_APPROVED_OUTPUT_PREFIX"
          stage_exports+=",RUN_FIXTURE=$RUN_FIXTURE"
          stage_exports+=",SMVQA_DATA_ROOT=$SMVQA_DATA_ROOT"
          stage_exports+=",SMVQA_FRAME_ROOT=$SMVQA_FRAME_ROOT"
          stage_exports+=",GEMMA_MODEL_PATH=$GEMMA_MODEL_PATH"
          stage_exports+=",WORLDMM_MEMORY_MODEL_PATH=$WORLDMM_MEMORY_MODEL_PATH"
          stage_exports+=",WORLDMM_TRAIN_EPOCHS=$WORLDMM_TRAIN_EPOCHS"
          stage_exports+=",WORLDMM_TRAIN_BATCH_SIZE=$WORLDMM_TRAIN_BATCH_SIZE"
          stage_exports+=",WORLDMM_TRAIN_HIDDEN_DIM=$WORLDMM_TRAIN_HIDDEN_DIM"
          stage_exports+=",WORLDMM_TRAIN_LEARNING_RATE=$WORLDMM_TRAIN_LEARNING_RATE"
          stage_exports+=",WORLDMM_TRAIN_RESUME=$WORLDMM_TRAIN_RESUME"
          stage_exports+=",WORLDMM_TRAIN_NODES=$WORLDMM_TRAIN_NODES"
          stage_exports+=",WORLDMM_TRAIN_GPUS_PER_NODE=$WORLDMM_TRAIN_GPUS_PER_NODE"
          stage_exports+=",WORLDMM_TEACHER_NODES=$WORLDMM_TEACHER_NODES"
          stage_exports+=",WORLDMM_TEACHER_GPUS_PER_NODE=$WORLDMM_TEACHER_GPUS_PER_NODE"
          stage_exports+=",WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW="
          stage_exports+="$WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW"
          stage_exports+=",WORLDMM_APPROVAL_FILE=${WORLDMM_APPROVAL_FILE:-}"
          stage_exports+=",WORLDMM_APPROVER=${WORLDMM_APPROVER:-}"
          stage_exports+=",WORLDMM_APPROVAL_SHA256=$WORLDMM_APPROVAL_SHA256"
          stage_exports+=",WORLDMM_GCUT3R_EXTRACTOR=${WORLDMM_GCUT3R_EXTRACTOR:-}"
          stage_exports+=",WORLDMM_TEACHER_CACHE_INPUT=${WORLDMM_TEACHER_CACHE_INPUT:-}"
          stage_exports+=",WORLDMM_SPATIAL_INFER_EXE=${WORLDMM_SPATIAL_INFER_EXE:-}"
          stage_exports+=",WORLDMM_STUDENT_SUPERVISION_INPUT="
          stage_exports+="${WORLDMM_STUDENT_SUPERVISION_INPUT:-}"
          for resource_name in \
            WORLDMM_CPU_PARTITION WORLDMM_GPU_PARTITION \
            WORLDMM_PREFLIGHT_NODES WORLDMM_PREFLIGHT_CPUS \
            WORLDMM_PREFLIGHT_MEM WORLDMM_PREFLIGHT_TIME \
            WORLDMM_TEACHER_CPUS WORLDMM_TEACHER_MEM WORLDMM_TEACHER_TIME \
            WORLDMM_MATERIALIZE_NODES WORLDMM_MATERIALIZE_CPUS \
            WORLDMM_MATERIALIZE_MEM WORLDMM_MATERIALIZE_TIME \
            WORLDMM_TRAIN_CPUS WORLDMM_TRAIN_MEM WORLDMM_TRAIN_TIME \
            WORLDMM_REPORT_NODES WORLDMM_REPORT_CPUS \
            WORLDMM_REPORT_MEM WORLDMM_REPORT_TIME; do
            stage_exports+=",${resource_name}=${!resource_name}"
          done
          local -a args=(
            "$SBATCH" --parsable --no-requeue
            "--job-name=worldmm-${WORLDMM_RUN_ID}-${stage}"
            "--partition=$partition"
            "--nodes=$nodes"
            --ntasks-per-node=1
            "--cpus-per-task=$cpus"
            "--mem=$mem"
            "--time=$time_limit"
            "--output=$WORLDMM_OUTPUT_ROOT/logs/${stage}-%j.out"
            "--error=$WORLDMM_OUTPUT_ROOT/logs/${stage}-%j.err"
            "--export=$stage_exports"
            "--comment=worldmm:${WORLDMM_RUN_ID}:${WORLDMM_DAG_PHASE}:${stage}"
          )
          if [ "$gpus" -gt 0 ]; then
            args+=("--gpus-per-node=$gpus")
          fi
          if [ -n "$dependency" ]; then
            case "$dependency_kind" in
              afterok|afterany) ;;
              *)
                printf "unsupported Slurm dependency kind: %s\n" \
                  "$dependency_kind" >&2
                return 1
                ;;
            esac
            args+=(
              --kill-on-invalid-dep=yes
              "--dependency=${dependency_kind}:$dependency"
            )
          fi
          # Persist the deterministic Slurm identity before sbatch. A failed
          # client response is therefore reconcilable without guessing.
          local identity
          identity="worldmm:${WORLDMM_RUN_ID}:${WORLDMM_DAG_PHASE}:${stage}"
          printf '{"schema_version":1,"event":"submission-unknown-before-sbatch","run_id":"%s","phase":"%s","stage":"%s","identity":"%s"}\n' \
            "$WORLDMM_RUN_ID" "$WORLDMM_DAG_PHASE" "$stage" "$identity" \
            >> "$submission_attempts"
          raw_job_id="$("${args[@]}" "$stage_script")"
          job_id="${raw_job_id%%;*}"
          if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
            printf "invalid sbatch job id for %s: %s\n" "$stage" "$raw_job_id" >&2
            return 1
          fi
          printf "%s=%s\n" "$stage" "$job_id" >> "$partial_jobs"
          printf '{"schema_version":1,"event":"submission-reconciled","run_id":"%s","phase":"%s","stage":"%s","job_id":"%s","attempt":"%s","identity":"%s"}\n' \
            "$WORLDMM_RUN_ID" "$WORLDMM_DAG_PHASE" "$stage" "$job_id" "$job_id" \
            "$identity" >> "$submission_attempts"
          printf "%s" "$job_id"
        }

        if [ "$WORLDMM_DAG_PHASE" = "preflight" ]; then
          preflight_job="$(
            submit_stage preflight_ingest "$WORLDMM_CPU_PARTITION" \
              "$WORLDMM_PREFLIGHT_NODES" 0 "$WORLDMM_PREFLIGHT_CPUS" \
              "$WORLDMM_PREFLIGHT_MEM" "$WORLDMM_PREFLIGHT_TIME" ""
          )"
          jobs_file="$WORLDMM_OUTPUT_ROOT/summary/dag_jobs.preflight.env"
          temporary="${jobs_file}.$$"
          printf "%s\n" \
            "WORLDMM_RUN_ID=$WORLDMM_RUN_ID" \
            "WORLDMM_OUTPUT_ROOT=$WORLDMM_OUTPUT_ROOT" \
            "PREFLIGHT_JOB_ID=$preflight_job" > "$temporary"
          mv "$temporary" "$jobs_file"
          trap - EXIT
          printf "run_id=%s phase=preflight job_id=%s; review %s before run phase\n" \
            "$WORLDMM_RUN_ID" "$preflight_job" \
            "$WORLDMM_OUTPUT_ROOT/diagnostics/preflight.json"
          exit 0
        fi

        if [ -n "${WORLDMM_GCUT3R_EXTRACTOR:-}" ]; then
          teacher_partition="$WORLDMM_GPU_PARTITION"
          teacher_nodes="$WORLDMM_TEACHER_NODES"
          teacher_gpus="$WORLDMM_TEACHER_GPUS_PER_NODE"
          teacher_cpus="$WORLDMM_TEACHER_CPUS"
          teacher_mem="$WORLDMM_TEACHER_MEM"
          teacher_time="$WORLDMM_TEACHER_TIME"
        else
          teacher_partition="$WORLDMM_CPU_PARTITION"
          teacher_nodes=1
          teacher_gpus=0
          teacher_cpus="$WORLDMM_PREFLIGHT_CPUS"
          teacher_mem="$WORLDMM_PREFLIGHT_MEM"
          teacher_time="$WORLDMM_PREFLIGHT_TIME"
        fi
        teacher_job="$(
          submit_stage teacher_extract "$teacher_partition" \
            "$teacher_nodes" "$teacher_gpus" "$teacher_cpus" \
            "$teacher_mem" "$teacher_time" ""
        )"
        materialize_job="$(
          submit_stage merge_materialize "$WORLDMM_CPU_PARTITION" \
            "$WORLDMM_MATERIALIZE_NODES" 0 "$WORLDMM_MATERIALIZE_CPUS" \
            "$WORLDMM_MATERIALIZE_MEM" "$WORLDMM_MATERIALIZE_TIME" "$teacher_job"
        )"
        train_job="$(
          submit_stage train "$WORLDMM_GPU_PARTITION" \
            "$WORLDMM_TRAIN_NODES" "$WORLDMM_TRAIN_GPUS_PER_NODE" \
            "$WORLDMM_TRAIN_CPUS" "$WORLDMM_TRAIN_MEM" \
            "$WORLDMM_TRAIN_TIME" "$materialize_job"
        )"
        build_memory_job="$(
          submit_stage build_memory "$WORLDMM_GPU_PARTITION" \
            "$WORLDMM_TRAIN_NODES" "$WORLDMM_TRAIN_GPUS_PER_NODE" \
            "$WORLDMM_TRAIN_CPUS" "$WORLDMM_TRAIN_MEM" \
            "$WORLDMM_TRAIN_TIME" "$train_job"
        )"
        student_infer_job="$(
          submit_stage student_infer_retrieve "$WORLDMM_GPU_PARTITION" \
            1 1 "$WORLDMM_TRAIN_CPUS" "$WORLDMM_TRAIN_MEM" \
            "$WORLDMM_TRAIN_TIME" "$build_memory_job"
        )"
        qa_job="$(
          submit_stage qa "$WORLDMM_GPU_PARTITION" \
            "$WORLDMM_TRAIN_NODES" "$WORLDMM_TRAIN_GPUS_PER_NODE" \
            "$WORLDMM_TRAIN_CPUS" "$WORLDMM_TRAIN_MEM" \
            "$WORLDMM_TRAIN_TIME" "$student_infer_job"
        )"
        report_job="$(
          submit_stage metrics_report "$WORLDMM_CPU_PARTITION" \
            "$WORLDMM_REPORT_NODES" 0 "$WORLDMM_REPORT_CPUS" \
            "$WORLDMM_REPORT_MEM" "$WORLDMM_REPORT_TIME" "$qa_job"
        )"
        verify_approved_python_runtime

        jobs_file="$WORLDMM_OUTPUT_ROOT/summary/dag_jobs.env"
        temporary="${jobs_file}.$$"
        printf "%s\n" \
          "WORLDMM_RUN_ID=$WORLDMM_RUN_ID" \
          "WORLDMM_OUTPUT_ROOT=$WORLDMM_OUTPUT_ROOT" \
          "TEACHER_JOB_ID=$teacher_job" \
          "MATERIALIZE_JOB_ID=$materialize_job" \
          "TRAIN_JOB_ID=$train_job" \
          "BUILD_MEMORY_JOB_ID=$build_memory_job" \
          "STUDENT_INFER_RETRIEVE_JOB_ID=$student_infer_job" \
          "QA_JOB_ID=$qa_job" \
          "APPROVAL_SHA256=$approval_sha256" \
          "REPORT_JOB_ID=$report_job" > "$temporary"
        mv "$temporary" "$jobs_file"
        trap - EXIT
        printf "run_id=%s phase=run output_root=%s final_job_id=%s\n" \
          "$WORLDMM_RUN_ID" "$WORLDMM_OUTPUT_ROOT" "$report_job"
        """,
    ).lstrip()


def dag_stage_script_text() -> str:
    """Render the stage runner shared by all DAG allocations."""
    return dedent(
        r"""
        #!/usr/bin/env bash
        set -euo pipefail
        umask 077
        export LC_ALL=C
        unset PYTHONPATH PYTHONHOME
        export PYTHONDONTWRITEBYTECODE=1
        export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1

        : "${SLURM_JOB_ID:?submit this script with /opt/slurm/bin/sbatch}"
        if ! [[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]; then
          printf "SLURM_JOB_ID must be numeric: %s\n" "$SLURM_JOB_ID" >&2
          exit 1
        fi
        : "${WORLDMM_STAGE:?WORLDMM_STAGE is required}"
        case "$WORLDMM_STAGE" in
          preflight_ingest|teacher_extract|merge_materialize|train|build_memory|\
          student_infer_retrieve|qa|metrics_report) ;;
          *)
            printf "unknown WORLDMM_STAGE: %s\n" "$WORLDMM_STAGE" >&2
            exit 2
            ;;
        esac
        : "${WORLDMM_OUTPUT_ROOT:?WORLDMM_OUTPUT_ROOT is required}"
        : "${WORLDMM_RUN_ID:?WORLDMM_RUN_ID is required}"
        if ! [[ "$WORLDMM_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
          printf "WORLDMM_RUN_ID has unsafe characters: %s\n" "$WORLDMM_RUN_ID" >&2
          exit 1
        fi
        if [ "${WORLDMM_SMVQA_REMOTE_APPROVED:-}" != "1" ]; then
          printf "WORLDMM_SMVQA_REMOTE_APPROVED=1 is required\n" >&2
          exit 1
        fi
        case "$WORLDMM_OUTPUT_ROOT" in
          */"$WORLDMM_RUN_ID") ;;
          *)
            printf "WORLDMM_OUTPUT_ROOT must end /%s: %s\n" \
              "$WORLDMM_RUN_ID" "$WORLDMM_OUTPUT_ROOT" >&2
            exit 1
            ;;
        esac
        default_remote_repo=/repo/VTteam/bongh.park/
        default_remote_repo+=worldmm-smvqa-gemma4-e2b
        WORLDMM_REMOTE_REPO="${WORLDMM_REMOTE_REPO:-$default_remote_repo}"
        : "${SMVQA_DATA_ROOT:=/groups/VTteam/datasets/SuperMemory-VQA/ingested}"
        : "${SMVQA_FRAME_ROOT:=$SMVQA_DATA_ROOT/frames}"
        : "${GEMMA_MODEL_PATH:=/repo/VTteam/bongh.park/gemma-4-e2b-it}"
        default_memory_model=/repo/VTteam/bongh.park/outputs/models/qwen3-vl
        : "${WORLDMM_MEMORY_MODEL_PATH:=$default_memory_model}"
        : "${RUN_FIXTURE:?RUN_FIXTURE is required}"
        : "${WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW:=4096}"
        : "${WORLDMM_APPROVED_DATA_PREFIX:=/groups/VTteam/datasets}"
        : "${WORLDMM_APPROVED_REPO_PREFIX:=/repo/VTteam/bongh.park}"
        : "${WORLDMM_APPROVED_OUTPUT_PREFIX:=/repo/VTteam/bongh.park/outputs}"
        : "${WORLDMM_STAGE_GPUS_PER_NODE:=0}"
        : "${WORLDMM_EXECUTION_PROFILE:?WORLDMM_EXECUTION_PROFILE is required}"
        : "${WORLDMM_TRAIN_NODES:?WORLDMM_TRAIN_NODES is required}"
        : "${WORLDMM_TRAIN_GPUS_PER_NODE:?WORLDMM_TRAIN_GPUS_PER_NODE is required}"
        : "${WORLDMM_TEACHER_NODES:?WORLDMM_TEACHER_NODES is required}"
        : "${WORLDMM_TEACHER_GPUS_PER_NODE:?WORLDMM_TEACHER_GPUS_PER_NODE is required}"
        : "${WORLDMM_TRAIN_EPOCHS:=1}"
        : "${WORLDMM_TRAIN_BATCH_SIZE:=8}"
        : "${WORLDMM_TRAIN_HIDDEN_DIM:=32}"
        : "${WORLDMM_TRAIN_LEARNING_RATE:=0.001}"
        : "${WORLDMM_TRAIN_RESUME:=}"
        export RUN_FIXTURE SMVQA_DATA_ROOT SMVQA_FRAME_ROOT GEMMA_MODEL_PATH
        export WORLDMM_MEMORY_MODEL_PATH WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW
        export WORLDMM_TRAIN_EPOCHS WORLDMM_TRAIN_BATCH_SIZE
        export WORLDMM_TRAIN_HIDDEN_DIM WORLDMM_TRAIN_LEARNING_RATE
        export WORLDMM_TRAIN_RESUME WORLDMM_SMVQA_REMOTE_APPROVED
        export WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST=1

        require_under_prefix() {
          local path="$1" prefix="$2" label="$3" resolved resolved_prefix
          resolved="$(realpath -m "$path")"
          resolved_prefix="$(realpath -m "$prefix")"
          case "$resolved" in
            "$resolved_prefix"|"$resolved_prefix"/*) ;;
            *)
              printf "%s resolves outside approved prefix %s: %s\n" \
                "$label" "$resolved_prefix" "$resolved" >&2
              return 1
              ;;
          esac
        }
        require_approved_input() {
          local path="$1" label="$2" resolved
          resolved="$(realpath -e "$path")"
          if require_under_prefix "$resolved" "$WORLDMM_APPROVED_DATA_PREFIX" \
            "$label" 2>/dev/null; then
            return 0
          fi
          require_under_prefix "$resolved" "$WORLDMM_APPROVED_REPO_PREFIX" "$label"
        }
        require_under_prefix "$SMVQA_DATA_ROOT" \
          "$WORLDMM_APPROVED_DATA_PREFIX" SMVQA_DATA_ROOT
        require_approved_input "$RUN_FIXTURE" RUN_FIXTURE
        require_under_prefix "$WORLDMM_REMOTE_REPO" \
          "$WORLDMM_APPROVED_REPO_PREFIX" WORLDMM_REMOTE_REPO
        require_under_prefix "$WORLDMM_OUTPUT_ROOT" \
          "$WORLDMM_APPROVED_OUTPUT_PREFIX" WORLDMM_OUTPUT_ROOT
        if [ ! -d "$WORLDMM_OUTPUT_ROOT" ] || \
          [ -L "$WORLDMM_OUTPUT_ROOT" ] || [ ! -O "$WORLDMM_OUTPUT_ROOT" ] || \
          find "$WORLDMM_OUTPUT_ROOT" -maxdepth 0 -perm /022 -print -quit \
            | grep -q .; then
          printf "stage output root ownership or permissions are unsafe: %s\n" \
            "$WORLDMM_OUTPUT_ROOT" >&2
          exit 1
        fi
        if find "$WORLDMM_OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type l \
          -print -quit | grep -q .; then
          printf "stage output root contains a symlinked managed directory\n" >&2
          exit 1
        fi
        for managed_dir in logs summary; do
          managed_path="$WORLDMM_OUTPUT_ROOT/$managed_dir"
          if [ ! -d "$managed_path" ] || [ -L "$managed_path" ] || \
            [ ! -O "$managed_path" ] || \
            find "$managed_path" -maxdepth 0 -perm /022 -print -quit \
              | grep -q .; then
            printf "stage managed directory is unsafe: %s\n" \
              "$managed_path" >&2
            exit 1
          fi
        done
        stage_failure="$WORLDMM_OUTPUT_ROOT/summary/stage.${WORLDMM_STAGE}.failure.json"
        record_stage_failure() {
          local status="$?" temporary
          trap - EXIT
          if [ "$status" -eq 0 ] && \
            declare -F verify_stage_python_runtime >/dev/null; then
            if ! verify_stage_python_runtime sealed; then
              status=1
            fi
          fi
          if [ "$status" -ne 0 ]; then
            temporary="${stage_failure}.$$"
            printf \
              '{"schema_version":1,"run_id":"%s","stage":"%s",'\
'"job_id":"%s","exit_code":%s}\n' \
              "$WORLDMM_RUN_ID" "$WORLDMM_STAGE" "$SLURM_JOB_ID" "$status" \
              > "$temporary"
            mv -T "$temporary" "$stage_failure"
          fi
          return "$status"
        }
        trap record_stage_failure EXIT
        attest_stage_allocation() {
          local expected_nodes expected_gpus expected_cpus expected_partition
          local actual_gpus="${SLURM_GPUS_ON_NODE:-0}" value
          case "$WORLDMM_STAGE" in
            preflight_ingest)
              expected_nodes="$WORLDMM_PREFLIGHT_NODES"
              expected_gpus=0
              expected_cpus="$WORLDMM_PREFLIGHT_CPUS"
              expected_partition="$WORLDMM_CPU_PARTITION"
              ;;
            teacher_extract)
              if [ -n "${WORLDMM_GCUT3R_EXTRACTOR:-}" ]; then
                expected_nodes="$WORLDMM_TEACHER_NODES"
                expected_gpus="$WORLDMM_TEACHER_GPUS_PER_NODE"
                expected_cpus="$WORLDMM_TEACHER_CPUS"
                expected_partition="$WORLDMM_GPU_PARTITION"
              else
                expected_nodes=1
                expected_gpus=0
                expected_cpus="$WORLDMM_PREFLIGHT_CPUS"
                expected_partition="$WORLDMM_CPU_PARTITION"
              fi
              ;;
            merge_materialize)
              expected_nodes="$WORLDMM_MATERIALIZE_NODES"
              expected_gpus=0
              expected_cpus="$WORLDMM_MATERIALIZE_CPUS"
              expected_partition="$WORLDMM_CPU_PARTITION"
              ;;
            train|build_memory|qa)
              expected_nodes="$WORLDMM_TRAIN_NODES"
              expected_gpus="$WORLDMM_TRAIN_GPUS_PER_NODE"
              expected_cpus="$WORLDMM_TRAIN_CPUS"
              expected_partition="$WORLDMM_GPU_PARTITION"
              ;;
            student_infer_retrieve)
              expected_nodes=1
              expected_gpus=1
              expected_cpus="$WORLDMM_TRAIN_CPUS"
              expected_partition="$WORLDMM_GPU_PARTITION"
              ;;
            metrics_report)
              expected_nodes="$WORLDMM_REPORT_NODES"
              expected_gpus=0
              expected_cpus="$WORLDMM_REPORT_CPUS"
              expected_partition="$WORLDMM_CPU_PARTITION"
              ;;
          esac
          for value in "$expected_nodes" "$expected_gpus" "$expected_cpus" \
            "${SLURM_NNODES:-}" "${SLURM_CPUS_PER_TASK:-}" \
            "$WORLDMM_STAGE_GPUS_PER_NODE" "$actual_gpus"; do
            if ! [[ "$value" =~ ^[0-9]+$ ]]; then
              printf "stage allocation value is missing/non-numeric: %s\n" \
                "$value" >&2
              return 1
            fi
          done
          if [ "$expected_gpus" -gt 0 ] && \
            [ -z "${SLURM_GPUS_ON_NODE:-}" ]; then
            printf "SLURM_GPUS_ON_NODE is required for GPU allocation proof\n" >&2
            return 1
          fi
          if [ "$SLURM_NNODES" -ne "$expected_nodes" ] || \
            [ "$WORLDMM_STAGE_GPUS_PER_NODE" -ne "$expected_gpus" ] || \
            [ "$actual_gpus" -ne "$expected_gpus" ] || \
            [ "$SLURM_CPUS_PER_TASK" -ne "$expected_cpus" ] || \
            [ "${SLURM_JOB_PARTITION:-}" != "$expected_partition" ]; then
            printf \
              "stage allocation mismatch: nodes=%s gpus=%s "\
"actual_gpus=%s cpus=%s partition=%s\n" \
              "$SLURM_NNODES" "$WORLDMM_STAGE_GPUS_PER_NODE" \
              "$actual_gpus" "$SLURM_CPUS_PER_TASK" \
              "${SLURM_JOB_PARTITION:-<missing>}" >&2
            return 1
          fi
        }
        attest_stage_allocation
        python_runtime_root="$WORLDMM_REMOTE_REPO/.venv"
        verify_stage_python_runtime() {
          local mode="$1" runtime_resolved unsafe_link unsafe_pth
          local runtime_link runtime_target pth_file pth_dir pth_line
          local pth_candidate pth_resolved runtime_files runtime_inventory
          local runtime_loader_inventory
          local current base_roots base_files base_inventory base_prefix
          local base_executable stdlib platstdlib root
          if [ ! -d "$python_runtime_root" ] || [ -L "$python_runtime_root" ]; then
            printf "shared Python runtime is missing or symlinked: %s\n" \
              "$python_runtime_root" >&2
            return 1
          fi
          runtime_resolved="$(realpath -e "$python_runtime_root")"
          unsafe_link="$(
            find "$python_runtime_root" -type l -print0 | \
              while IFS= read -r -d '' runtime_link; do
                if [ ! -e "$runtime_link" ] || [ -d "$runtime_link" ]; then
                  printf "%s" "$runtime_link"
                  break
                fi
              done
          )"
          if [ -n "$unsafe_link" ]; then
            printf "Python runtime has unsafe directory/dangling symlink: %s\n" \
              "$unsafe_link" >&2
            return 1
          fi
          unsafe_pth="$(
            find "$python_runtime_root" \( -type f -o -type l \) \
              -name '*.pth' -print0 | \
              while IFS= read -r -d '' pth_file; do
                pth_dir="$(dirname "$pth_file")"
                while IFS= read -r pth_line || [ -n "$pth_line" ]; do
                  pth_line="${pth_line%$'\r'}"
                  case "$pth_line" in
                    ''|'#'*) continue ;;
                    'import '*|$'import\t'*) printf "%s" "$pth_file"; break 2 ;;
                    /*) pth_candidate="$pth_line" ;;
                    *) pth_candidate="$pth_dir/$pth_line" ;;
                  esac
                  pth_resolved="$(realpath -m "$pth_candidate")"
                  case "$pth_resolved" in
                    "$runtime_resolved"|"$runtime_resolved"/*) ;;
                    *) printf "%s" "$pth_file"; break 2 ;;
                  esac
                done < "$pth_file"
              done
          )"
          if [ -n "$unsafe_pth" ]; then
            printf "Python runtime has unsafe executable/external .pth: %s\n" \
              "$unsafe_pth" >&2
            return 1
          fi
          if [ "$mode" = unsealed ]; then
            return 0
          fi
          runtime_files="$WORLDMM_OUTPUT_ROOT/diagnostics/python_runtime.sha256"
          runtime_inventory="$WORLDMM_OUTPUT_ROOT/diagnostics/python_runtime.files.sha256"
          runtime_loader_inventory="$WORLDMM_OUTPUT_ROOT/diagnostics/python_runtime.loader.sha256"
          base_roots="$WORLDMM_OUTPUT_ROOT/diagnostics/python_base_roots.tsv"
          base_files="$WORLDMM_OUTPUT_ROOT/diagnostics/python_base_runtime.sha256"
          base_inventory="$WORLDMM_OUTPUT_ROOT/diagnostics/python_base_runtime.files.sha256"
          if [ ! -s "$runtime_files" ] || [ ! -s "$runtime_inventory" ] || \
            [ ! -s "$runtime_loader_inventory" ] || \
            [ ! -s "$base_roots" ] || [ ! -s "$base_files" ] || \
            [ ! -s "$base_inventory" ]; then
            printf "approved Python runtime manifests are missing\n" >&2
            return 1
          fi
          if ! (cd "$python_runtime_root" && \
            sha256sum --check --status "$runtime_files"); then
            printf "approved Python runtime content changed\n" >&2
            return 1
          fi
          current="${runtime_inventory}.stage.$$"
          (
            cd "$python_runtime_root"
            find . \( -type f -o -type l \) -printf '%y %p %l\0' \
              | sort -z | sha256sum
          ) > "$current"
          if ! cmp -s "$runtime_inventory" "$current"; then
            rm -f "$current"
            printf "approved Python runtime file inventory changed\n" >&2
            return 1
          fi
          rm -f "$current"
          current="${runtime_loader_inventory}.stage.$$"
          ldd "$python_runtime_root/bin/python" | while IFS= read -r loader; do
            printf '%s\n' "${loader%% (*}"
          done | LC_ALL=C sort | sha256sum > "$current"
          if ! cmp -s "$runtime_loader_inventory" "$current"; then
            rm -f "$current"
            printf "approved Python interpreter loader/shared-library closure changed\n" >&2
            return 1
          fi
          rm -f "$current"
          base_prefix="$(awk -F '\t' '$1 == "base_prefix" {print $2}' "$base_roots")"
          base_executable="$(awk -F '\t' '$1 == "executable" {print $2}' "$base_roots")"
          stdlib="$(awk -F '\t' '$1 == "stdlib" {print $2}' "$base_roots")"
          platstdlib="$(awk -F '\t' '$1 == "platstdlib" {print $2}' "$base_roots")"
          if [ "$(wc -l < "$base_roots")" -ne 4 ] || \
            [ -z "$base_prefix" ] || [ -z "$base_executable" ] || \
            [ -z "$stdlib" ] || [ -z "$platstdlib" ] || \
            [ "$(realpath -e "$base_executable")" != "$base_executable" ]; then
            printf "approved Python base runtime roots are invalid\n" >&2
            return 1
          fi
          for root in "$base_executable" "$stdlib" "$platstdlib"; do
            if [ ! -e "$root" ]; then
              printf "approved Python base runtime path is missing: %s\n" \
                "$root" >&2
              return 1
            fi
            case "$root" in
              "$base_prefix"|"$base_prefix"/*) ;;
              *)
                printf "approved Python base runtime path escapes prefix: %s\n" \
                  "$root" >&2
                return 1
                ;;
            esac
          done
          if ! sha256sum --check --status "$base_files"; then
            printf "approved Python base runtime content changed\n" >&2
            return 1
          fi
          current="${base_inventory}.stage.$$"
          find "$base_executable" "$stdlib" "$platstdlib" \
            \( -type f -o -type l \) -printf '%y %p %l\0' \
            | sort -zu | sha256sum > "$current"
          if ! cmp -s "$base_inventory" "$current"; then
            rm -f "$current"
            printf "approved Python base runtime file inventory changed\n" >&2
            return 1
          fi
          rm -f "$current"
        }
        if [ "$WORLDMM_STAGE" = preflight_ingest ]; then
          verify_stage_python_runtime unsealed
        else
          verify_stage_python_runtime sealed
        fi
        source "$WORLDMM_REMOTE_REPO/.venv/bin/activate"
        WORLDMM_EXECUTION_REPO="$WORLDMM_REMOTE_REPO"
        if [ "$WORLDMM_STAGE" != "preflight_ingest" ]; then
          WORLDMM_EXECUTION_REPO="$WORLDMM_OUTPUT_ROOT/code_snapshot"
          if [ ! -d "$WORLDMM_EXECUTION_REPO/src/worldmm_smvqa" ]; then
            printf "approved code snapshot missing: %s\n" \
              "$WORLDMM_EXECUTION_REPO" >&2
            exit 1
          fi
          deployed_code="$WORLDMM_OUTPUT_ROOT/diagnostics/deployed_code.sha256"
          deployed_files="$WORLDMM_OUTPUT_ROOT/diagnostics/deployed_code.files.sha256"
          if [ ! -s "$deployed_code" ] || [ ! -s "$deployed_files" ]; then
            printf "approved code snapshot manifests are missing\n" >&2
            exit 1
          fi
          if find "$WORLDMM_EXECUTION_REPO" -type l -print -quit | grep -q .; then
            printf "approved code snapshot contains a symlink\n" >&2
            exit 1
          fi
          if ! (cd "$WORLDMM_EXECUTION_REPO" && \
            sha256sum --check --status "$deployed_code"); then
            printf "approved code snapshot content changed\n" >&2
            exit 1
          fi
          deployed_files_current="${deployed_files}.stage.$$"
          (
            cd "$WORLDMM_EXECUTION_REPO"
            find . -type f -print0 | sort -z | sha256sum
          ) > "$deployed_files_current"
          if ! cmp -s "$deployed_files" "$deployed_files_current"; then
            rm -f "$deployed_files_current"
            printf "approved code snapshot file inventory changed\n" >&2
            exit 1
          fi
          rm -f "$deployed_files_current"
        fi
        export WORLDMM_EXECUTION_REPO
        export PYTHONPATH="$WORLDMM_EXECUTION_REPO/src"
        cd "$WORLDMM_EXECUTION_REPO"
        mkdir -p \
          "$WORLDMM_OUTPUT_ROOT"/{manifests,inference_inputs,teacher,training,\
          checkpoints,memory,retrieval} \
          "$WORLDMM_OUTPUT_ROOT"/{qa,metrics,diagnostics,summary}
        WORLDMM_SENSOR_FRAME_MANIFEST="$WORLDMM_OUTPUT_ROOT/manifests"
        WORLDMM_SENSOR_FRAME_MANIFEST+="/sensor_frames.jsonl"
        export WORLDMM_SENSOR_FRAME_MANIFEST

        if [ "$WORLDMM_STAGE" != "preflight_ingest" ]; then
          : "${WORLDMM_APPROVAL_FILE:?WORLDMM_APPROVAL_FILE is required}"
          : "${WORLDMM_APPROVER:?WORLDMM_APPROVER is required}"
          : "${WORLDMM_APPROVAL_SHA256:?WORLDMM_APPROVAL_SHA256 is required}"
          : "${WORLDMM_SPATIAL_INFER_EXE:?spatial inference executable required}"
          : "${WORLDMM_STUDENT_SUPERVISION_INPUT:?student supervision required}"
          approval_path="$(realpath -e "$WORLDMM_APPROVAL_FILE")"
          require_under_prefix "$approval_path" "$WORLDMM_REMOTE_REPO" \
            WORLDMM_APPROVAL_FILE
          if [ ! -O "$approval_path" ] || \
            find "$approval_path" -perm /022 -print -quit | grep -q .; then
            printf "approval file permissions are unsafe\n" >&2
            exit 1
          fi
          if [ "$(sha256sum "$approval_path" | cut -d ' ' -f 1)" != \
            "$WORLDMM_APPROVAL_SHA256" ]; then
            printf "approval file changed after DAG submission\n" >&2
            exit 1
          fi
          require_approved_input "$SMVQA_FRAME_ROOT" SMVQA_FRAME_ROOT
          require_approved_input "$GEMMA_MODEL_PATH" GEMMA_MODEL_PATH
          require_approved_input "$WORLDMM_MEMORY_MODEL_PATH" \
            WORLDMM_MEMORY_MODEL_PATH
          require_approved_input "$WORLDMM_SPATIAL_INFER_EXE" \
            WORLDMM_SPATIAL_INFER_EXE
          require_approved_input "$WORLDMM_STUDENT_SUPERVISION_INPUT" \
            WORLDMM_STUDENT_SUPERVISION_INPUT
          if [ -n "${WORLDMM_GCUT3R_EXTRACTOR:-}" ]; then
            teacher_mode=extractor
            teacher_path="$WORLDMM_GCUT3R_EXTRACTOR"
          else
            teacher_mode=cache
            : "${WORLDMM_TEACHER_CACHE_INPUT:?teacher cache input required}"
            teacher_path="$WORLDMM_TEACHER_CACHE_INPUT"
          fi
          require_approved_input "$teacher_path" WORLDMM_TEACHER_INPUT
          if [ -n "$WORLDMM_TRAIN_RESUME" ]; then
            require_approved_input "$WORLDMM_TRAIN_RESUME" WORLDMM_TRAIN_RESUME
          fi
          preflight_inputs="$WORLDMM_OUTPUT_ROOT/diagnostics/preflight_inputs.sha256"
          env_contract="$WORLDMM_OUTPUT_ROOT/diagnostics/env_contract.json"
          frame_assets="$WORLDMM_OUTPUT_ROOT/diagnostics/frame_assets.sha256"
          if [ ! -s "$preflight_inputs" ] || [ ! -s "$env_contract" ] || \
            [ ! -s "$frame_assets" ]; then
            printf "preflight contracts are missing\n" >&2
            exit 1
          fi
          "$WORLDMM_REMOTE_REPO/.venv/bin/python" - \
            "$approval_path" "$env_contract" "$preflight_inputs" \
            "$teacher_mode" "$teacher_path" <<'PY'
import hashlib
import json
import os
import sys
from importlib.metadata import distributions
from pathlib import Path

approval_path, contract_path, inputs_path = map(Path, sys.argv[1:4])
teacher_mode, teacher_path = sys.argv[4:6]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

env = os.environ
approval = json.loads(approval_path.read_text(encoding="utf-8"))

def python_runtime() -> dict[str, object]:
    packages = sorted(
        [
            distribution.metadata.get("Name") or "",
            distribution.version,
            distribution.read_text("direct_url.json") or "",
        ]
        for distribution in distributions()
    )
    return {
        "version": sys.version,
        "executable": str(Path(sys.executable).resolve(strict=True)),
        "packages": packages,
    }

teacher_nodes = int(env["WORLDMM_TEACHER_NODES"])
teacher_gpus = int(env["WORLDMM_TEACHER_GPUS_PER_NODE"])
if teacher_mode == "cache":
    teacher_nodes, teacher_gpus = 1, 0
teacher_uses_gpu = teacher_mode == "extractor"
effective_teacher_resources = {
    "nodes": teacher_nodes,
    "gpus_per_node": teacher_gpus,
    "partition": env[
        "WORLDMM_GPU_PARTITION" if teacher_uses_gpu else "WORLDMM_CPU_PARTITION"
    ],
    "cpus_per_task": int(
        env[
            "WORLDMM_TEACHER_CPUS"
            if teacher_uses_gpu
            else "WORLDMM_PREFLIGHT_CPUS"
        ]
    ),
    "memory": env[
        "WORLDMM_TEACHER_MEM" if teacher_uses_gpu else "WORLDMM_PREFLIGHT_MEM"
    ],
    "time": env[
        "WORLDMM_TEACHER_TIME" if teacher_uses_gpu else "WORLDMM_PREFLIGHT_TIME"
    ],
}
expected_approval = {
    "run_id": env["WORLDMM_RUN_ID"],
    "profile": env["WORLDMM_EXECUTION_PROFILE"],
    "nodes": int(env["WORLDMM_TRAIN_NODES"]),
    "gpus_per_node": int(env["WORLDMM_TRAIN_GPUS_PER_NODE"]),
    "teacher_nodes": teacher_nodes,
    "teacher_gpus_per_node": teacher_gpus,
    "train_nodes": int(env["WORLDMM_TRAIN_NODES"]),
    "train_gpus_per_node": int(env["WORLDMM_TRAIN_GPUS_PER_NODE"]),
    "approved": True,
    "approver": env["WORLDMM_APPROVER"],
    "preflight_inputs_sha256": sha256(inputs_path),
    "env_contract_sha256": sha256(contract_path),
}
for key, value in expected_approval.items():
    actual = approval.get(key)
    if type(actual) is not type(value) or actual != value:
        raise SystemExit(f"stage approval mismatch for {key}")
current_contract = {
    "schema_version": 1,
    "run_id": env["WORLDMM_RUN_ID"],
    "profile": env["WORLDMM_EXECUTION_PROFILE"],
    "run_fixture": str(Path(env["RUN_FIXTURE"]).resolve(strict=True)),
    "data_root": str(Path(env["SMVQA_DATA_ROOT"]).resolve(strict=True)),
    "frame_root": str(Path(env["SMVQA_FRAME_ROOT"]).resolve(strict=True)),
    "gemma_model_path": str(Path(env["GEMMA_MODEL_PATH"]).resolve(strict=True)),
    "memory_model_path": str(
        Path(env["WORLDMM_MEMORY_MODEL_PATH"]).resolve(strict=True)
    ),
    "spatial_infer_exe": str(
        Path(env["WORLDMM_SPATIAL_INFER_EXE"]).resolve(strict=True)
    ),
    "student_supervision": str(
        Path(env["WORLDMM_STUDENT_SUPERVISION_INPUT"]).resolve(strict=True)
    ),
    "teacher_mode": teacher_mode,
    "teacher_path": str(Path(teacher_path).resolve(strict=True)),
    "effective_teacher_resources": effective_teacher_resources,
    "train_resume": (
        str(Path(env["WORLDMM_TRAIN_RESUME"]).resolve(strict=True))
        if env["WORLDMM_TRAIN_RESUME"]
        else None
    ),
    "train_epochs": int(env["WORLDMM_TRAIN_EPOCHS"]),
    "train_batch_size": int(env["WORLDMM_TRAIN_BATCH_SIZE"]),
    "train_hidden_dim": int(env["WORLDMM_TRAIN_HIDDEN_DIM"]),
    "train_learning_rate": float(env["WORLDMM_TRAIN_LEARNING_RATE"]),
    "byte_budget_per_window": int(
        env["WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW"]
    ),
    "python_runtime": python_runtime(),
    "approved_prefixes": {
        name: str(Path(env[name]).resolve(strict=True))
        for name in (
            "WORLDMM_APPROVED_DATA_PREFIX",
            "WORLDMM_APPROVED_REPO_PREFIX",
            "WORLDMM_APPROVED_OUTPUT_PREFIX",
        )
    },
    "resources": {
        name: env[name]
        for name in (
            "WORLDMM_CPU_PARTITION",
            "WORLDMM_GPU_PARTITION",
            "WORLDMM_PREFLIGHT_NODES",
            "WORLDMM_PREFLIGHT_CPUS",
            "WORLDMM_PREFLIGHT_MEM",
            "WORLDMM_PREFLIGHT_TIME",
            "WORLDMM_TEACHER_NODES",
            "WORLDMM_TEACHER_GPUS_PER_NODE",
            "WORLDMM_TEACHER_CPUS",
            "WORLDMM_TEACHER_MEM",
            "WORLDMM_TEACHER_TIME",
            "WORLDMM_MATERIALIZE_NODES",
            "WORLDMM_MATERIALIZE_CPUS",
            "WORLDMM_MATERIALIZE_MEM",
            "WORLDMM_MATERIALIZE_TIME",
            "WORLDMM_TRAIN_NODES",
            "WORLDMM_TRAIN_GPUS_PER_NODE",
            "WORLDMM_TRAIN_CPUS",
            "WORLDMM_TRAIN_MEM",
            "WORLDMM_TRAIN_TIME",
            "WORLDMM_REPORT_NODES",
            "WORLDMM_REPORT_CPUS",
            "WORLDMM_REPORT_MEM",
            "WORLDMM_REPORT_TIME",
        )
    },
}
contract = json.loads(contract_path.read_text(encoding="utf-8"))
if contract != current_contract:
    raise SystemExit("stage environment differs from preflight env contract")
stage = env["WORLDMM_STAGE"]
allocated = (int(env["SLURM_NNODES"]), int(env["WORLDMM_STAGE_GPUS_PER_NODE"]))
if stage == "teacher_extract":
    expected_allocation = (teacher_nodes, teacher_gpus)
    expected_cpus = int(
        env["WORLDMM_TEACHER_CPUS"]
        if teacher_mode == "extractor"
        else env["WORLDMM_PREFLIGHT_CPUS"]
    )
    expected_partition = (
        env["WORLDMM_GPU_PARTITION"]
        if teacher_mode == "extractor"
        else env["WORLDMM_CPU_PARTITION"]
    )
elif stage in {"train", "build_memory", "qa"}:
    expected_allocation = (
        int(env["WORLDMM_TRAIN_NODES"]),
        int(env["WORLDMM_TRAIN_GPUS_PER_NODE"]),
    )
    expected_cpus = int(env["WORLDMM_TRAIN_CPUS"])
    expected_partition = env["WORLDMM_GPU_PARTITION"]
elif stage == "student_infer_retrieve":
    expected_allocation = (1, 1)
    expected_cpus = int(env["WORLDMM_TRAIN_CPUS"])
    expected_partition = env["WORLDMM_GPU_PARTITION"]
elif stage == "merge_materialize":
    expected_allocation = (int(env["WORLDMM_MATERIALIZE_NODES"]), 0)
    expected_cpus = int(env["WORLDMM_MATERIALIZE_CPUS"])
    expected_partition = env["WORLDMM_CPU_PARTITION"]
else:
    expected_allocation = (int(env["WORLDMM_REPORT_NODES"]), 0)
    expected_cpus = int(env["WORLDMM_REPORT_CPUS"])
    expected_partition = env["WORLDMM_CPU_PARTITION"]
if allocated != expected_allocation:
    raise SystemExit(
        f"stage allocation mismatch: expected={expected_allocation}, got={allocated}"
    )
slurm_gpus_on_node = env.get("SLURM_GPUS_ON_NODE")
if slurm_gpus_on_node is not None and slurm_gpus_on_node.isdecimal():
    if int(slurm_gpus_on_node) != expected_allocation[1]:
        raise SystemExit(
            "stage actual GPU allocation mismatch: "
            f"expected={expected_allocation[1]}, got={slurm_gpus_on_node}"
        )
if int(env["SLURM_CPUS_PER_TASK"]) != expected_cpus:
    raise SystemExit(
        f"stage CPU mismatch: expected={expected_cpus}, "
        f"got={env['SLURM_CPUS_PER_TASK']}"
    )
if env.get("SLURM_JOB_PARTITION", expected_partition) != expected_partition:
    raise SystemExit(f"stage partition mismatch: expected={expected_partition}")
PY
          sha256sum --check --status "$preflight_inputs"
          verify_stage_inventory() {
            local root="$1" expected="$2" temporary
            temporary="${expected}.stage.$$"
            (
              cd "$root"
              find . -type f -print0 | sort -z | sha256sum
            ) > "$temporary"
            if ! cmp -s "$expected" "$temporary"; then
              rm -f "$temporary"
              printf "file inventory changed after preflight: %s\n" "$root" >&2
              return 1
            fi
            rm -f "$temporary"
          }
          if [ "$WORLDMM_STAGE" = "qa" ]; then
            sha256sum --check --status \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/gemma_model.sha256"
            verify_stage_inventory "$GEMMA_MODEL_PATH" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/gemma_model.files.sha256"
          elif [ "$WORLDMM_STAGE" = "build_memory" ]; then
            sha256sum --check --status \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/memory_model.sha256"
            verify_stage_inventory "$WORLDMM_MEMORY_MODEL_PATH" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/memory_model.files.sha256"
          fi
          case "$WORLDMM_STAGE" in
            teacher_extract|build_memory|student_infer_retrieve|qa)
              sha256sum --check --status "$frame_assets"
              verify_stage_inventory \
                "$WORLDMM_OUTPUT_ROOT/inference_inputs/frames" \
                "$WORLDMM_OUTPUT_ROOT/diagnostics/frame_assets.files.sha256"
              ;;
          esac
        fi
        if [ "$WORLDMM_STAGE" != "preflight_ingest" ]; then
          SMVQA_FRAME_ROOT="$WORLDMM_OUTPUT_ROOT/inference_inputs/frames"
          if [ ! -d "$SMVQA_FRAME_ROOT" ]; then
            printf "sensed frame root missing: %s\n" "$SMVQA_FRAME_ROOT" >&2
            exit 1
          fi
          export SMVQA_FRAME_ROOT
        fi
        export SLURM_EXPORT_ENV=ALL
        stage_marker="$WORLDMM_OUTPUT_ROOT/summary/stage.${WORLDMM_STAGE}.started"
        if ! (
          set -o noclobber
          printf "%s\n" "$SLURM_JOB_ID" > "$stage_marker"
        ) 2>/dev/null; then
          printf "stage already started for this run: %s\n" "$stage_marker" >&2
          exit 1
        fi
        distributed_train() {
          mapfile -t hosts < <(
            /opt/slurm/bin/scontrol show hostnames "$SLURM_JOB_NODELIST"
          )
          export MASTER_ADDR="${hosts[0]}"
          export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
          WORLDMM_TEACHER_CACHE="$WORLDMM_OUTPUT_ROOT/training"
          WORLDMM_TEACHER_CACHE+="/student_teacher_cache.jsonl"
          export WORLDMM_TEACHER_CACHE
          WORLDMM_CHECKPOINT="$WORLDMM_OUTPUT_ROOT/checkpoints"
          WORLDMM_CHECKPOINT+="/spatial_student.pt"
          export WORLDMM_CHECKPOINT
          if [ -n "$WORLDMM_TRAIN_RESUME" ] && [ ! -s "$WORLDMM_TRAIN_RESUME" ]; then
            printf "WORLDMM_TRAIN_RESUME is not a non-empty file: %s\n" \
              "$WORLDMM_TRAIN_RESUME" >&2
            return 1
          fi
          export WORLDMM_TEACHER_CACHE WORLDMM_CHECKPOINT
          /opt/slurm/bin/srun \
            --nodes="$SLURM_NNODES" \
            --ntasks="$SLURM_NNODES" \
            --ntasks-per-node=1 \
            --gpus-per-task="$WORLDMM_STAGE_GPUS_PER_NODE" \
            --kill-on-bad-exit=1 \
            bash -c '
              train_args=(
                --config configs/remote.example.yaml
                --teacher-cache "$WORLDMM_TEACHER_CACHE"
                --checkpoint "$WORLDMM_CHECKPOINT"
                --epochs "$WORLDMM_TRAIN_EPOCHS"
                --batch-size "$WORLDMM_TRAIN_BATCH_SIZE"
                --hidden-dim "$WORLDMM_TRAIN_HIDDEN_DIM"
                --learning-rate "$WORLDMM_TRAIN_LEARNING_RATE"
              )
              if [ -n "$WORLDMM_TRAIN_RESUME" ]; then
                train_args+=(--resume "$WORLDMM_TRAIN_RESUME")
              fi
              python -m torch.distributed.run \
                --nnodes "$SLURM_NNODES" \
                --nproc-per-node "$WORLDMM_STAGE_GPUS_PER_NODE" \
                --node-rank "$SLURM_NODEID" \
                --master-addr "$MASTER_ADDR" \
                --master-port "$MASTER_PORT" \
                -m worldmm_smvqa.spatial_train train \
                "${train_args[@]}"'
        }

        distributed_memory() {
          local stores="$1" output_path="$2" input_path="${3:-}"
          mapfile -t hosts < <(
            /opt/slurm/bin/scontrol show hostnames "$SLURM_JOB_NODELIST"
          )
          export MASTER_ADDR="${hosts[0]}"
          export MASTER_PORT="$((25000 + SLURM_JOB_ID % 15000))"
          export WORLDMM_MEMORY_STORES="$stores"
          export WORLDMM_MEMORY_OUTPUT="$output_path"
          export WORLDMM_MEMORY_INPUT="$input_path"
          /opt/slurm/bin/srun \
            --nodes="$SLURM_NNODES" \
            --ntasks="$SLURM_NNODES" \
            --ntasks-per-node=1 \
            --gpus-per-task="$WORLDMM_STAGE_GPUS_PER_NODE" \
            --kill-on-bad-exit=1 \
            bash -c '
              memory_args=(
                build-memory
                --stores "$WORLDMM_MEMORY_STORES"
                --config configs/remote.example.yaml
                --fixture "$RUN_FIXTURE"
                --out "$WORLDMM_MEMORY_OUTPUT"
                --backend qwen
              )
              if [ -n "$WORLDMM_MEMORY_INPUT" ]; then
                memory_args+=(--input "$WORLDMM_MEMORY_INPUT")
              fi
              python -m torch.distributed.run \
                --nnodes "$SLURM_NNODES" \
                --nproc-per-node "$WORLDMM_STAGE_GPUS_PER_NODE" \
                --node-rank "$SLURM_NODEID" \
                --master-addr "$MASTER_ADDR" \
                --master-port "$MASTER_PORT" \
                -m worldmm_smvqa.cli "${memory_args[@]}"'
        }

        distributed_qa() {
          mapfile -t hosts < <(
            /opt/slurm/bin/scontrol show hostnames "$SLURM_JOB_NODELIST"
          )
          export MASTER_ADDR="${hosts[0]}"
          export MASTER_PORT="$((30000 + SLURM_JOB_ID % 20000))"
          WORLDMM_QA_EVIDENCE="$WORLDMM_OUTPUT_ROOT/retrieval"
          WORLDMM_QA_EVIDENCE+="/evidence_packs.jsonl"
          export WORLDMM_QA_EVIDENCE
          export WORLDMM_QA_OUTPUT="$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl"
          /opt/slurm/bin/srun \
            --nodes="$SLURM_NNODES" \
            --ntasks="$SLURM_NNODES" \
            --ntasks-per-node=1 \
            --gpus-per-task="$WORLDMM_STAGE_GPUS_PER_NODE" \
            --kill-on-bad-exit=1 \
            bash -c '
              python -m torch.distributed.run \
                --nnodes "$SLURM_NNODES" \
                --nproc-per-node "$WORLDMM_STAGE_GPUS_PER_NODE" \
                --node-rank "$SLURM_NODEID" \
                --master-addr "$MASTER_ADDR" \
                --master-port "$MASTER_PORT" \
                -m worldmm_smvqa.qa_transformers \
                --model "$GEMMA_MODEL_PATH" \
                --fixture "$RUN_FIXTURE" \
                --evidence-lane student \
                --evidence-lineage "$WORLDMM_QA_EVIDENCE.lineage.json" \
                --checkpoint "$WORLDMM_OUTPUT_ROOT/checkpoints/spatial_student.pt" \
                --typed-memory "$WORLDMM_OUTPUT_ROOT/memory/typed_memory.jsonl" \
                --inference-manifest \
                  "$WORLDMM_OUTPUT_ROOT/memory/typed_memory.inference.json" \
                --inference-sources \
                  "$WORLDMM_OUTPUT_ROOT/inference_inputs/sources.jsonl" \
                --inference-producer "$WORLDMM_SPATIAL_INFER_EXE" \
                --model-fingerprint \
                  "$WORLDMM_OUTPUT_ROOT/diagnostics/gemma_model.sha256" \
                --frame-assets-manifest \
                  "$WORLDMM_OUTPUT_ROOT/diagnostics/frame_assets.sha256" \
                --lineage-config \
                  "$WORLDMM_EXECUTION_REPO/configs/remote.example.yaml" \
                --sensor-frame-manifest "$WORLDMM_SENSOR_FRAME_MANIFEST" \
                --memory-manifest \
                  "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json" \
                --require-frames \
                --evidence "$WORLDMM_QA_EVIDENCE" \
                --out "$WORLDMM_QA_OUTPUT"'
        }

        case "$WORLDMM_STAGE" in
          preflight_ingest)
            code_snapshot="$WORLDMM_OUTPUT_ROOT/code_snapshot"
            if [ -e "$code_snapshot" ]; then
              printf "code snapshot already exists: %s\n" "$code_snapshot" >&2
              exit 1
            fi
            mkdir -p "$code_snapshot"/{src,configs,remote-plan}
            rsync -a --delete --exclude __pycache__ --exclude '*.py[co]' \
              "$WORLDMM_REMOTE_REPO/src/" "$code_snapshot/src/"
            rsync -a --delete --exclude __pycache__ --exclude '*.py[co]' \
              "$WORLDMM_REMOTE_REPO/configs/" "$code_snapshot/configs/"
            install -m 600 "$WORLDMM_REMOTE_REPO/pyproject.toml" \
              "$WORLDMM_REMOTE_REPO/uv.lock" "$code_snapshot/"
            cp -a "$WORLDMM_REMOTE_REPO/remote-plan/"*.sh \
              "$code_snapshot/remote-plan/"
            require_nonempty_file() {
              if [ ! -s "$1" ]; then
                printf "%s is not a non-empty file: %s\n" "$2" "$1" >&2
                return 1
              fi
              require_approved_input "$1" "$2"
            }
            require_model() {
              local model_path="$1" label="$2"
              require_approved_input "$model_path" "$label"
              if find "$model_path" -type l -print -quit | grep -q .; then
                printf \
                  "%s contains symlinks; materialize a self-contained model tree\n" \
                  "$label" >&2
                return 1
              fi
              require_nonempty_file "$model_path/config.json" \
                "$label config.json"
              if ! find "$model_path" -type f \
                \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) \
                -size +0c -print -quit | grep -q .; then
                printf "%s has no non-empty model weight file: %s\n" \
                  "$label" "$model_path" >&2
                return 1
              fi
            }

            require_model "$GEMMA_MODEL_PATH" GEMMA_MODEL_PATH
            require_model "$WORLDMM_MEMORY_MODEL_PATH" \
              WORLDMM_MEMORY_MODEL_PATH
            if find "$code_snapshot" -type l -print -quit | grep -q .; then
              printf "code snapshot contains unsupported symlinks\n" >&2
              exit 1
            fi
            require_approved_input "$SMVQA_FRAME_ROOT" SMVQA_FRAME_ROOT
            if [ -z "${WORLDMM_SPATIAL_INFER_EXE:-}" ]; then
              printf "production WORLDMM_SPATIAL_INFER_EXE is required\n" >&2
              exit 1
            fi
            if [ ! -x "$WORLDMM_SPATIAL_INFER_EXE" ]; then
              printf "WORLDMM_SPATIAL_INFER_EXE is not executable: %s\n" \
                "$WORLDMM_SPATIAL_INFER_EXE" >&2
              exit 1
            fi
            require_approved_input "$WORLDMM_SPATIAL_INFER_EXE" \
              WORLDMM_SPATIAL_INFER_EXE
            spatial_infer_contract="$(
              env \
                -u RUN_FIXTURE -u SMVQA_DATA_ROOT \
                -u WORLDMM_STUDENT_SUPERVISION_INPUT \
                -u WORLDMM_TEACHER_CACHE_INPUT \
                -u WORLDMM_APPROVAL_FILE -u WORLDMM_APPROVER \
                -u WORLDMM_APPROVAL_SHA256 -u GEMMA_MODEL_PATH \
                -u WORLDMM_MEMORY_MODEL_PATH -u WORLDMM_REMOTE_REPO \
                -u WORLDMM_OUTPUT_ROOT \
                "$WORLDMM_SPATIAL_INFER_EXE" --contract-version
            )"
            if [ "$spatial_infer_contract" != "worldmm-spatial-infer-v1" ]; then
              printf "unsupported spatial inference contract: %s\n" \
                "$spatial_infer_contract" >&2
              exit 1
            fi
            spatial_infer_self_test="$(
              env \
                -u RUN_FIXTURE -u SMVQA_DATA_ROOT \
                -u WORLDMM_STUDENT_SUPERVISION_INPUT \
                -u WORLDMM_TEACHER_CACHE_INPUT \
                -u WORLDMM_APPROVAL_FILE -u WORLDMM_APPROVER \
                -u WORLDMM_APPROVAL_SHA256 -u GEMMA_MODEL_PATH \
                -u WORLDMM_MEMORY_MODEL_PATH -u WORLDMM_REMOTE_REPO \
                -u WORLDMM_OUTPUT_ROOT \
                "$WORLDMM_SPATIAL_INFER_EXE" --self-test
            )"
            if [ "$spatial_infer_self_test" != \
              "worldmm-spatial-infer-v1:self-test-ok" ]; then
              printf "spatial inference self-test failed: %s\n" \
                "$spatial_infer_self_test" >&2
              exit 1
            fi
            printf "%s\n" "$spatial_infer_contract" \
              > "$WORLDMM_OUTPUT_ROOT/diagnostics/spatial_infer_contract.txt"
            : "${WORLDMM_STUDENT_SUPERVISION_INPUT:?student supervision required}"
            require_nonempty_file "$WORLDMM_STUDENT_SUPERVISION_INPUT" \
              WORLDMM_STUDENT_SUPERVISION_INPUT
            if [ -n "${WORLDMM_GCUT3R_EXTRACTOR:-}" ]; then
              teacher_mode=extractor
              teacher_input="$WORLDMM_GCUT3R_EXTRACTOR"
              if [ ! -x "$WORLDMM_GCUT3R_EXTRACTOR" ]; then
                printf "WORLDMM_GCUT3R_EXTRACTOR is not executable: %s\n" \
                  "$WORLDMM_GCUT3R_EXTRACTOR" >&2
                exit 1
              fi
              require_approved_input "$WORLDMM_GCUT3R_EXTRACTOR" \
                WORLDMM_GCUT3R_EXTRACTOR
            else
              teacher_mode=cache
              : "${WORLDMM_TEACHER_CACHE_INPUT:?teacher cache input required}"
              teacher_input="$WORLDMM_TEACHER_CACHE_INPUT"
              require_nonempty_file "$WORLDMM_TEACHER_CACHE_INPUT" \
                WORLDMM_TEACHER_CACHE_INPUT
              python -m worldmm_smvqa.worldmm.gcut3r_teacher validate-cache \
                --cache "$WORLDMM_TEACHER_CACHE_INPUT" \
                > "$WORLDMM_OUTPUT_ROOT/diagnostics/preflight_teacher_cache.json"
            fi
            if [ -n "$WORLDMM_TRAIN_RESUME" ]; then
              require_nonempty_file "$WORLDMM_TRAIN_RESUME" \
                WORLDMM_TRAIN_RESUME
            fi
            if ! [[ "$WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW" =~ ^[1-9][0-9]*$ ]]; then
              printf "WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW must be positive\n" >&2
              exit 1
            fi

            python - "$GEMMA_MODEL_PATH" "$WORLDMM_MEMORY_MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path
from transformers import AutoConfig, AutoProcessor
for model_path in sys.argv[1:]:
    AutoConfig.from_pretrained(model_path, local_files_only=True)
    AutoProcessor.from_pretrained(model_path, local_files_only=True)
    root = Path(model_path).resolve(strict=True)
    for index_path in root.glob("*index.json"):
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            continue
        def invalid_shard(shard: object) -> bool:
            if not isinstance(shard, str):
                return True
            candidate = root / shard
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                return True
            return (
                candidate.is_symlink()
                or root not in resolved.parents
                or not resolved.is_file()
                or resolved.stat().st_size == 0
            )

        missing = sorted(
            repr(shard) for shard in weight_map.values() if invalid_shard(shard)
        )
        if missing:
            raise SystemExit(
                f"model index references missing/empty shards: {index_path}: {missing}"
            )
PY
            python - "$WORLDMM_STUDENT_SUPERVISION_INPUT" "$teacher_mode" \
              "$teacher_input" "$WORLDMM_EXECUTION_PROFILE" \
              "$RUN_FIXTURE" <<'PY'
import sys
from pathlib import Path
from worldmm_smvqa.chunking import read_source_streams
from worldmm_smvqa.teacher_materializer import (
    _read_supervision,
    materialize_teacher_rows,
)
from worldmm_smvqa.worldmm.gcut3r_teacher import read_teacher_cache

supervision, teacher_mode, teacher_input, profile, fixture = sys.argv[1:]
if profile == "probe":
    supervision_bytes = Path(supervision).stat().st_size
    if supervision_bytes > 64 * 1024 * 1024:
        raise SystemExit(
            f"probe supervision exceeds 64 MiB: {supervision_bytes} bytes"
        )
    if teacher_mode == "cache":
        cache_bytes = Path(teacher_input).stat().st_size
        if cache_bytes > 256 * 1024 * 1024:
            raise SystemExit(
                f"probe teacher cache exceeds 256 MiB: {cache_bytes} bytes"
            )
rows = _read_supervision(Path(supervision))
if profile == "probe" and len(rows) > 10_000:
    raise SystemExit(f"probe supervision exceeds 10000 rows: {len(rows)}")
if teacher_mode == "cache":
    cache_path = Path(teacher_input)
    cache = read_teacher_cache(cache_path)
    materialized = materialize_teacher_rows(cache_path, Path(supervision))
    allowed_videos = {
        source.video_id
        for source in read_source_streams(Path(fixture), use_sensor_manifest=False)
    }
    cache_videos = {row.request.video_id for row in cache}
    if cache_videos != allowed_videos:
        missing = sorted(allowed_videos - cache_videos)
        extra = sorted(cache_videos - allowed_videos)
        raise SystemExit(
            f"teacher cache video coverage mismatch: missing={missing} extra={extra}"
        )
    if profile == "probe" and len(materialized) > 10_000:
        raise SystemExit(
            f"probe materialized supervision exceeds 10000 rows: {len(materialized)}"
        )
    print(f"teacher_cache_rows={len(cache)} materialized_rows={len(materialized)}")
print(f"student_supervision_rows={len(rows)}")
PY
            env_contract="$WORLDMM_OUTPUT_ROOT/diagnostics/env_contract.json"
            python - "$env_contract" "$WORLDMM_RUN_ID" \
              "$WORLDMM_EXECUTION_PROFILE" "$RUN_FIXTURE" \
              "$SMVQA_DATA_ROOT" "$SMVQA_FRAME_ROOT" "$GEMMA_MODEL_PATH" \
              "$WORLDMM_MEMORY_MODEL_PATH" "$WORLDMM_SPATIAL_INFER_EXE" \
              "$WORLDMM_STUDENT_SUPERVISION_INPUT" "$teacher_mode" \
              "$teacher_input" "$WORLDMM_TRAIN_RESUME" \
              "$WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW" <<'PY'
import json
import os
import sys
from importlib.metadata import distributions
from pathlib import Path

output = Path(sys.argv[1])
path_values = [str(Path(value).resolve(strict=True)) for value in sys.argv[4:11]]

def python_runtime() -> dict[str, object]:
    packages = sorted(
        [
            distribution.metadata.get("Name") or "",
            distribution.version,
            distribution.read_text("direct_url.json") or "",
        ]
        for distribution in distributions()
    )
    return {
        "version": sys.version,
        "executable": str(Path(sys.executable).resolve(strict=True)),
        "packages": packages,
    }

teacher_mode = sys.argv[11]
teacher_uses_gpu = teacher_mode == "extractor"
effective_teacher_resources = {
    "nodes": (
        int(os.environ["WORLDMM_TEACHER_NODES"]) if teacher_uses_gpu else 1
    ),
    "gpus_per_node": (
        int(os.environ["WORLDMM_TEACHER_GPUS_PER_NODE"])
        if teacher_uses_gpu
        else 0
    ),
    "partition": os.environ[
        "WORLDMM_GPU_PARTITION" if teacher_uses_gpu else "WORLDMM_CPU_PARTITION"
    ],
    "cpus_per_task": int(
        os.environ[
            "WORLDMM_TEACHER_CPUS"
            if teacher_uses_gpu
            else "WORLDMM_PREFLIGHT_CPUS"
        ]
    ),
    "memory": os.environ[
        "WORLDMM_TEACHER_MEM" if teacher_uses_gpu else "WORLDMM_PREFLIGHT_MEM"
    ],
    "time": os.environ[
        "WORLDMM_TEACHER_TIME" if teacher_uses_gpu else "WORLDMM_PREFLIGHT_TIME"
    ],
}

payload = {
    "schema_version": 1,
    "run_id": sys.argv[2],
    "profile": sys.argv[3],
    "run_fixture": path_values[0],
    "data_root": path_values[1],
    "frame_root": path_values[2],
    "gemma_model_path": path_values[3],
    "memory_model_path": path_values[4],
    "spatial_infer_exe": path_values[5],
    "student_supervision": path_values[6],
    "teacher_mode": teacher_mode,
    "teacher_path": str(Path(sys.argv[12]).resolve(strict=True)),
    "effective_teacher_resources": effective_teacher_resources,
    "train_resume": (
        str(Path(sys.argv[13]).resolve(strict=True)) if sys.argv[13] else None
    ),
    "train_epochs": int(os.environ["WORLDMM_TRAIN_EPOCHS"]),
    "train_batch_size": int(os.environ["WORLDMM_TRAIN_BATCH_SIZE"]),
    "train_hidden_dim": int(os.environ["WORLDMM_TRAIN_HIDDEN_DIM"]),
    "train_learning_rate": float(os.environ["WORLDMM_TRAIN_LEARNING_RATE"]),
    "byte_budget_per_window": int(sys.argv[14]),
    "python_runtime": python_runtime(),
    "approved_prefixes": {
        name: str(Path(os.environ[name]).resolve(strict=True))
        for name in (
            "WORLDMM_APPROVED_DATA_PREFIX",
            "WORLDMM_APPROVED_REPO_PREFIX",
            "WORLDMM_APPROVED_OUTPUT_PREFIX",
        )
    },
    "resources": {
        name: os.environ[name]
        for name in (
            "WORLDMM_CPU_PARTITION",
            "WORLDMM_GPU_PARTITION",
            "WORLDMM_PREFLIGHT_NODES",
            "WORLDMM_PREFLIGHT_CPUS",
            "WORLDMM_PREFLIGHT_MEM",
            "WORLDMM_PREFLIGHT_TIME",
            "WORLDMM_TEACHER_NODES",
            "WORLDMM_TEACHER_GPUS_PER_NODE",
            "WORLDMM_TEACHER_CPUS",
            "WORLDMM_TEACHER_MEM",
            "WORLDMM_TEACHER_TIME",
            "WORLDMM_MATERIALIZE_NODES",
            "WORLDMM_MATERIALIZE_CPUS",
            "WORLDMM_MATERIALIZE_MEM",
            "WORLDMM_MATERIALIZE_TIME",
            "WORLDMM_TRAIN_NODES",
            "WORLDMM_TRAIN_GPUS_PER_NODE",
            "WORLDMM_TRAIN_CPUS",
            "WORLDMM_TRAIN_MEM",
            "WORLDMM_TRAIN_TIME",
            "WORLDMM_REPORT_NODES",
            "WORLDMM_REPORT_CPUS",
            "WORLDMM_REPORT_MEM",
            "WORLDMM_REPORT_TIME",
        )
    },
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
            if [ "$WORLDMM_EXECUTION_PROFILE" = "probe" ]; then
              python - "$RUN_FIXTURE" "$SMVQA_DATA_ROOT" <<'PY'
import json
import sys
from pathlib import Path

probe, full = map(Path, sys.argv[1:])
for root in (probe, full):
    for name in ("sources.jsonl", "questions.jsonl", "labels.jsonl"):
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing non-empty fixture file: {path}")
        if root == probe and path.stat().st_size > 64 * 1024 * 1024:
            raise SystemExit(f"probe fixture file exceeds 64 MiB: {path}")
probe_count = sum(
    bool(line.strip())
    for line in (probe / "questions.jsonl").read_text(encoding="utf-8").splitlines()
)
full_count = sum(
    bool(line.strip())
    for line in (full / "questions.jsonl").read_text(encoding="utf-8").splitlines()
)
if not 0 < probe_count < full_count:
    raise SystemExit(
        f"probe fixture must be reduced: probe={probe_count}, full={full_count}"
    )
probe_sources = [
    json.loads(line)
    for line in (probe / "sources.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
probe_frames = sum(len(source.get("frame_metadata", ())) for source in probe_sources)
if len(probe_sources) > 4 or probe_frames > 600:
    raise SystemExit(
        "probe fixture exceeds source/frame cap: "
        f"sources={len(probe_sources)} frames={probe_frames}"
    )
PY
            fi
            worldmm-smvqa preflight \
              --fixture "$RUN_FIXTURE" \
              --out "$WORLDMM_OUTPUT_ROOT/diagnostics/preflight.json"
            worldmm-smvqa build-memory --stage sensor-frames \
              --config configs/remote.example.yaml \
              --fixture "$RUN_FIXTURE" \
              --out "$WORLDMM_OUTPUT_ROOT/manifests/sensor_frames.jsonl"
            if [ "$teacher_mode" = "cache" ]; then
              python - "$WORLDMM_TEACHER_CACHE_INPUT" \
                "$WORLDMM_SENSOR_FRAME_MANIFEST" <<'PY'
import sys
from collections import Counter
from pathlib import Path
from worldmm_smvqa.sensor_frames import read_sensor_frame_manifest
from worldmm_smvqa.worldmm.gcut3r_teacher import read_teacher_cache

cache = read_teacher_cache(Path(sys.argv[1]))
sensor = read_sensor_frame_manifest(Path(sys.argv[2]))
ground_truth_pose = tuple(
    row.request.observation_id
    for row in cache
    if row.request.pose_guidance is not None
    and row.request.pose_guidance.source == "ground_truth"
)
if ground_truth_pose:
    raise SystemExit(
        "production teacher cache may not use ground_truth pose guidance: "
        f"{ground_truth_pose[:10]}"
    )
expected = Counter(
    (record.video_id, frame.frame_ref, frame.timestamp)
    for record in sensor
    for frame in record.selected_frames
)
actual = Counter(
    (row.request.video_id, row.request.frame_ref, row.request.timestamp)
    for row in cache
)
if actual != expected:
    missing = sorted((expected - actual).elements())[:10]
    extra = sorted((actual - expected).elements())[:10]
    raise SystemExit(
        "teacher cache sensor observation mismatch: "
        f"missing={missing} extra={extra}"
    )
PY
            fi
            python - "$WORLDMM_SENSOR_FRAME_MANIFEST" "$SMVQA_FRAME_ROOT" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/frame_assets.sha256" \
              "$WORLDMM_OUTPUT_ROOT/inference_inputs/frames" <<'PY'
import hashlib
import os
import shutil
import sys
from pathlib import Path
from worldmm_smvqa.sensor_frames import read_sensor_frame_manifest
from worldmm_smvqa.video_frames import FRAME_EXTENSIONS

sensor_manifest, raw_frame_root, output, sensed_frame_root = map(Path, sys.argv[1:])
frame_root = raw_frame_root.resolve(strict=True)
sensed_frame_root.mkdir(parents=True, exist_ok=False)
rows = []
seen = set()
destinations = set()
records = read_sensor_frame_manifest(sensor_manifest)
for record in sorted(records, key=lambda row: row.video_id):
    for frame in record.selected_frames:
        if (
            Path(record.video_id).name != record.video_id
            or Path(frame.frame_ref).name != frame.frame_ref
            or record.video_id in {".", ".."}
            or frame.frame_ref in {".", ".."}
            or "\\" in record.video_id
            or "\\" in frame.frame_ref
            or "\n" in record.video_id
            or "\n" in frame.frame_ref
        ):
            raise SystemExit(
                f"unsafe video/frame path: {record.video_id}/{frame.frame_ref}"
            )
        base = frame_root / record.video_id / frame.frame_ref
        path = next(
            (
                base.with_suffix(suffix)
                for suffix in FRAME_EXTENSIONS
                if base.with_suffix(suffix).is_file()
            ),
            None,
        )
        if path is None:
            raise SystemExit(
                "selected frame asset missing: "
                f"{record.video_id}/{frame.frame_ref}"
            )
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(frame_root)
        except ValueError as exc:
            raise SystemExit(
                f"selected frame escapes frame root: {resolved}"
            ) from exc
        if resolved in seen:
            continue
        seen.add(resolved)
        destination = sensed_frame_root / record.video_id / Path(
            frame.frame_ref
        ).with_suffix(resolved.suffix.lower())
        if destination in destinations:
            raise SystemExit(f"selected frame destination collision: {destination}")
        destinations.add(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copy2(resolved, destination)
        digest = hashlib.sha256()
        with destination.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        path_text = str(destination.resolve(strict=True))
        if "\n" in path_text or "\\" in path_text:
            raise SystemExit(f"unsupported frame path characters: {resolved}")
        rows.append(f"{digest.hexdigest()}  {path_text}\n")
if not rows:
    raise SystemExit("sensor manifest resolved no frame assets")
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text("".join(rows), encoding="utf-8")
temporary.replace(output)
PY
            (
              cd "$WORLDMM_OUTPUT_ROOT/inference_inputs/frames"
              find . -type f -print0 | sort -z | sha256sum
            ) > "$WORLDMM_OUTPUT_ROOT/diagnostics/frame_assets.files.sha256"
            worldmm-smvqa build-memory --stage chunk \
              --config configs/remote.example.yaml \
              --fixture "$RUN_FIXTURE" \
              --out "$WORLDMM_OUTPUT_ROOT/manifests/source_chunks.jsonl"
            worldmm-smvqa build-memory --stage source-memories \
              --config configs/remote.example.yaml \
              --fixture "$RUN_FIXTURE" \
              --out "$WORLDMM_OUTPUT_ROOT/manifests/source_memories.jsonl"
            python - "$RUN_FIXTURE" \
              "$WORLDMM_OUTPUT_ROOT/inference_inputs/sources.jsonl" <<'PY'
import os
import sys
from pathlib import Path
from worldmm_smvqa.chunking import read_source_streams

fixture, output = map(Path, sys.argv[1:])
sources = read_source_streams(fixture)
payload = "".join(
    source.model_copy(
        update={
            "transcript": None,
            "transcript_spans": (),
            "captions": (),
            "ocr": (),
            "ocr_entries": (),
            "objects": (),
            "object_detections": (),
            "frame_metadata": tuple(
                frame.model_copy(update={"description": ""})
                for frame in source.frame_metadata
            ),
        }
    ).model_dump_json()
    + "\n"
    for source in sources
)
if not payload:
    raise SystemExit("sensed inference source inventory is empty")
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(payload, encoding="utf-8")
temporary.replace(output)
PY

            python_runtime_files="$WORLDMM_OUTPUT_ROOT/diagnostics"
            python_runtime_files+="/python_runtime.sha256"
            python_runtime_inventory="$WORLDMM_OUTPUT_ROOT/diagnostics"
            python_runtime_inventory+="/python_runtime.files.sha256"
            (
              cd "$python_runtime_root"
              find . -xtype f -print0 \
                | sort -z | xargs -0 sha256sum --
            ) > "$python_runtime_files"
            (
              cd "$python_runtime_root"
              find . \( -type f -o -type l \) -printf '%y %p %l\0' \
                | sort -z | sha256sum
            ) > "$python_runtime_inventory"
            python_loader_inventory="$WORLDMM_OUTPUT_ROOT/diagnostics/python_runtime.loader.sha256"
            ldd "$python_runtime_root/bin/python" | while IFS= read -r loader; do
              printf '%s\n' "${loader%% (*}"
            done | LC_ALL=C sort | sha256sum > "$python_loader_inventory"
            python_base_roots="$WORLDMM_OUTPUT_ROOT/diagnostics"
            python_base_roots+="/python_base_roots.tsv"
            python - "$python_base_roots" <<'PY'
import os
import sys
import sysconfig
from pathlib import Path

output = Path(sys.argv[1])
base_prefix = Path(sys.base_prefix).resolve(strict=True)
executable = Path(sys.executable).resolve(strict=True)
if executable != base_prefix and base_prefix not in executable.parents:
    raise SystemExit("Python executable resolves outside sys.base_prefix")
rows = [("base_prefix", base_prefix), ("executable", executable)]
for name in ("stdlib", "platstdlib"):
    root = Path(sysconfig.get_path(name)).resolve(strict=True)
    if root != base_prefix and base_prefix not in root.parents:
        raise SystemExit(f"{name} resolves outside sys.base_prefix: {root}")
    rows.append((name, root))
if any("\n" in str(path) or "\t" in str(path) for _, path in rows):
    raise SystemExit("Python base runtime path contains newline/tab")
payload = "".join(f"{name}\t{path}\n" for name, path in rows)
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(payload, encoding="utf-8")
temporary.replace(output)
PY
            python_base_prefix="$(
              awk -F '\t' '$1 == "base_prefix" {print $2}' \
                "$python_base_roots"
            )"
            python_base_executable="$(
              awk -F '\t' '$1 == "executable" {print $2}' \
                "$python_base_roots"
            )"
            if [ -z "$python_base_prefix" ] || \
              [ -z "$python_base_executable" ]; then
              printf "Python base runtime roots are incomplete\n" >&2
              exit 1
            fi
            python_base_files="$WORLDMM_OUTPUT_ROOT/diagnostics"
            python_base_files+="/python_base_runtime.sha256"
            python_base_inventory="$WORLDMM_OUTPUT_ROOT/diagnostics"
            python_base_inventory+="/python_base_runtime.files.sha256"
            mapfile -t python_base_stdlib_roots < <(
              awk -F '\t' '$1 == "stdlib" || $1 == "platstdlib" {print $2}' \
                "$python_base_roots"
            )
            find "$python_base_executable" "${python_base_stdlib_roots[@]}" \
              -xtype f -print0 \
              | sort -zu | xargs -0 sha256sum -- > "$python_base_files"
            find "$python_base_executable" "${python_base_stdlib_roots[@]}" \
              \( -type f -o -type l \) \
              -printf '%y %p %l\0' | sort -zu | sha256sum \
              > "$python_base_inventory"

            fingerprint="$WORLDMM_OUTPUT_ROOT/diagnostics/preflight_inputs.sha256"
            temporary="${fingerprint}.$$"
            inputs=(
              "$RUN_FIXTURE/sources.jsonl"
              "$RUN_FIXTURE/questions.jsonl"
              "$RUN_FIXTURE/labels.jsonl"
              "$GEMMA_MODEL_PATH/config.json"
              "$WORLDMM_MEMORY_MODEL_PATH/config.json"
              "$WORLDMM_SPATIAL_INFER_EXE"
              "$WORLDMM_STUDENT_SUPERVISION_INPUT"
              "$WORLDMM_OUTPUT_ROOT/inference_inputs/sources.jsonl"
              "$WORLDMM_SENSOR_FRAME_MANIFEST"
              "$WORLDMM_OUTPUT_ROOT/diagnostics/frame_assets.sha256"
              "$WORLDMM_OUTPUT_ROOT/diagnostics/frame_assets.files.sha256"
              "$WORLDMM_OUTPUT_ROOT/diagnostics/python_runtime.sha256"
              "$WORLDMM_OUTPUT_ROOT/diagnostics/python_runtime.files.sha256"
              "$WORLDMM_OUTPUT_ROOT/diagnostics/python_runtime.loader.sha256"
              "$WORLDMM_OUTPUT_ROOT/diagnostics/python_base_roots.tsv"
              "$WORLDMM_OUTPUT_ROOT/diagnostics/python_base_runtime.sha256"
              "$WORLDMM_OUTPUT_ROOT/diagnostics/python_base_runtime.files.sha256"
              "$WORLDMM_OUTPUT_ROOT/diagnostics/preflight.json"
              "$WORLDMM_OUTPUT_ROOT/manifests/source_chunks.jsonl"
              "$WORLDMM_OUTPUT_ROOT/manifests/source_memories.jsonl"
            )
            if [ "$(realpath -e "$RUN_FIXTURE")" != \
              "$(realpath -e "$SMVQA_DATA_ROOT")" ]; then
              inputs+=(
                "$SMVQA_DATA_ROOT/sources.jsonl"
                "$SMVQA_DATA_ROOT/questions.jsonl"
                "$SMVQA_DATA_ROOT/labels.jsonl"
              )
            fi
            if [ -n "${WORLDMM_GCUT3R_EXTRACTOR:-}" ]; then
              inputs+=("$WORLDMM_GCUT3R_EXTRACTOR")
            else
              inputs+=("$WORLDMM_TEACHER_CACHE_INPUT")
            fi
            if [ -n "$WORLDMM_TRAIN_RESUME" ]; then
              inputs+=("$WORLDMM_TRAIN_RESUME")
            fi
            mapfile -d '' gemma_files < <(
              find "$GEMMA_MODEL_PATH" -type f -print0 | sort -z
            )
            mapfile -d '' memory_model_files < <(
              find "$WORLDMM_MEMORY_MODEL_PATH" -type f -print0 | sort -z
            )
            mapfile -d '' code_files < <(
              find "$code_snapshot" -type f -print0 | sort -z
            )
            sha256sum -- "${gemma_files[@]}" \
              > "$WORLDMM_OUTPUT_ROOT/diagnostics/gemma_model.sha256"
            sha256sum -- "${memory_model_files[@]}" \
              > "$WORLDMM_OUTPUT_ROOT/diagnostics/memory_model.sha256"
            (
              cd "$GEMMA_MODEL_PATH"
              find . -type f -print0 | sort -z | sha256sum
            ) > "$WORLDMM_OUTPUT_ROOT/diagnostics/gemma_model.files.sha256"
            (
              cd "$WORLDMM_MEMORY_MODEL_PATH"
              find . -type f -print0 | sort -z | sha256sum
            ) > "$WORLDMM_OUTPUT_ROOT/diagnostics/memory_model.files.sha256"
            (
              cd "$code_snapshot"
              find . -type f -print0 | sort -z | xargs -0 sha256sum --
            ) > "$WORLDMM_OUTPUT_ROOT/diagnostics/deployed_code.sha256"
            (
              cd "$code_snapshot"
              find . -type f -print0 | sort -z | sha256sum
            ) > "$WORLDMM_OUTPUT_ROOT/diagnostics/deployed_code.files.sha256"
            sha256sum -- "${inputs[@]}" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/gemma_model.sha256" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/memory_model.sha256" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/gemma_model.files.sha256" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/memory_model.files.sha256" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/deployed_code.sha256" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/deployed_code.files.sha256" \
              "${code_files[@]}" > "$temporary"
            mv "$temporary" "$fingerprint"
            printf "run_id=%s\n" "$WORLDMM_RUN_ID" \
              > "$WORLDMM_OUTPUT_ROOT/diagnostics/preflight.completed"
            ;;
          teacher_extract)
            teacher_shard_root="$WORLDMM_OUTPUT_ROOT/teacher/shards"
            mkdir -p "$teacher_shard_root"
            if find "$teacher_shard_root" -mindepth 1 -maxdepth 1 \
              -name '*.jsonl' -print -quit | grep -q .; then
              printf "teacher shard directory is not empty: %s\n" \
                "$teacher_shard_root" >&2
              exit 1
            fi
            if [ -n "${WORLDMM_GCUT3R_EXTRACTOR:-}" ]; then
              if [ ! -s "$WORLDMM_SENSOR_FRAME_MANIFEST" ]; then
                printf "sensor-frame manifest is not a non-empty file: %s\n" \
                  "$WORLDMM_SENSOR_FRAME_MANIFEST" >&2
                exit 1
              fi
              if [ ! -x "$WORLDMM_GCUT3R_EXTRACTOR" ]; then
                printf "WORLDMM_GCUT3R_EXTRACTOR is not executable: %s\n" \
                  "$WORLDMM_GCUT3R_EXTRACTOR" >&2
                exit 1
              fi
              if [ "$WORLDMM_STAGE_GPUS_PER_NODE" -lt 1 ]; then
                printf "teacher extraction requires at least one GPU per node\n" >&2
                exit 1
              fi
              teacher_world_size=$((
                SLURM_NNODES * WORLDMM_STAGE_GPUS_PER_NODE
              ))
              teacher_allocated_cpus="${SLURM_CPUS_PER_TASK:-1}"
              if [ "$teacher_allocated_cpus" -lt "$WORLDMM_STAGE_GPUS_PER_NODE" ]; then
                printf "teacher CPUs per node must cover GPU workers: %s < %s\n" \
                  "$teacher_allocated_cpus" \
                  "$WORLDMM_STAGE_GPUS_PER_NODE" >&2
                exit 1
              fi
              teacher_cpus_per_worker=$((
                teacher_allocated_cpus / WORLDMM_STAGE_GPUS_PER_NODE
              ))
              /opt/slurm/bin/srun \
                --nodes="$SLURM_NNODES" \
                --ntasks="$teacher_world_size" \
                --ntasks-per-node="$WORLDMM_STAGE_GPUS_PER_NODE" \
                --cpus-per-task="$teacher_cpus_per_worker" \
                --gpus-per-task=1 \
                --gpu-bind=single:1 \
                --kill-on-bad-exit=1 \
                env \
                  -u RUN_FIXTURE -u SMVQA_DATA_ROOT \
                  -u WORLDMM_STUDENT_SUPERVISION_INPUT \
                  -u WORLDMM_TEACHER_CACHE_INPUT \
                  -u WORLDMM_APPROVAL_FILE -u WORLDMM_APPROVER \
                  -u WORLDMM_APPROVAL_SHA256 -u GEMMA_MODEL_PATH \
                  -u WORLDMM_MEMORY_MODEL_PATH -u WORLDMM_REMOTE_REPO \
                  -u WORLDMM_OUTPUT_ROOT \
                  -u WORLDMM_GCUT3R_EXTRACTOR \
                bash -c '
                  extractor="$1"
                  sources="$2"
                  frame_root="$3"
                  sensor_manifest="$4"
                  shard_root="$5"
                  world_size="$6"
                  printf -v rank_id "%05d" "$SLURM_PROCID"
                  exec "$extractor" \
                    --sources "$sources" \
                    --frame-root "$frame_root" \
                    --sensor-frame-manifest "$sensor_manifest" \
                    --rank "$SLURM_PROCID" \
                    --world-size "$world_size" \
                    --out "$shard_root/rank-${rank_id}.jsonl"' \
                  _ "$WORLDMM_GCUT3R_EXTRACTOR" \
                  "$WORLDMM_OUTPUT_ROOT/inference_inputs/sources.jsonl" \
                  "$SMVQA_FRAME_ROOT" "$WORLDMM_SENSOR_FRAME_MANIFEST" \
                  "$teacher_shard_root" "$teacher_world_size"
              teacher_shard_count="$(
                find "$teacher_shard_root" -maxdepth 1 -type f \
                  -name 'rank-*.jsonl' -size +0c -print | wc -l
              )"
              teacher_jsonl_count="$(
                find "$teacher_shard_root" -maxdepth 1 -type f \
                  -name '*.jsonl' -print | wc -l
              )"
              if [ "$teacher_shard_count" -ne "$teacher_world_size" ] || \
                [ "$teacher_jsonl_count" -ne "$teacher_world_size" ]; then
                printf \
                  "teacher shard mismatch: expected %s; valid ranks %s; total %s\n" \
                  "$teacher_world_size" "$teacher_shard_count" \
                  "$teacher_jsonl_count" >&2
                exit 1
              fi
            else
              : "${WORLDMM_TEACHER_CACHE_INPUT:?teacher cache input required}"
              if [ ! -s "$WORLDMM_TEACHER_CACHE_INPUT" ]; then
                printf "teacher cache input is not a non-empty file: %s\n" \
                  "$WORLDMM_TEACHER_CACHE_INPUT" >&2
                exit 1
              fi
              ln -sfn "$WORLDMM_TEACHER_CACHE_INPUT" \
                "$teacher_shard_root/external.jsonl"
            fi
            ;;
          merge_materialize)
            mapfile -d '' teacher_shards < <(
              find "$WORLDMM_OUTPUT_ROOT/teacher/shards" \
                \( -type f -o -type l \) -name '*.jsonl' -print0 | sort -z
            )
            if [ "${#teacher_shards[@]}" -eq 0 ]; then
              printf "no teacher JSONL shards found\n" >&2
              exit 1
            fi
            temporary="$WORLDMM_OUTPUT_ROOT/teacher/.cache.jsonl.$$"
            cat "${teacher_shards[@]}" > "$temporary"
            mv "$temporary" "$WORLDMM_OUTPUT_ROOT/teacher/cache.jsonl"
            python -m worldmm_smvqa.worldmm.gcut3r_teacher validate-cache \
              --cache "$WORLDMM_OUTPUT_ROOT/teacher/cache.jsonl" \
              > "$WORLDMM_OUTPUT_ROOT/diagnostics/teacher_cache.json"
            python - "$WORLDMM_OUTPUT_ROOT/teacher/cache.jsonl" \
              "$WORLDMM_SENSOR_FRAME_MANIFEST" \
              "$WORLDMM_EXECUTION_PROFILE" <<'PY'
import sys
from collections import Counter
from pathlib import Path
from worldmm_smvqa.sensor_frames import read_sensor_frame_manifest
from worldmm_smvqa.worldmm.gcut3r_teacher import read_teacher_cache

cache = read_teacher_cache(Path(sys.argv[1]))
sensor = read_sensor_frame_manifest(Path(sys.argv[2]))
profile = sys.argv[3]
ground_truth_pose = tuple(
    row.request.observation_id
    for row in cache
    if row.request.pose_guidance is not None
    and row.request.pose_guidance.source == "ground_truth"
)
if ground_truth_pose:
    raise SystemExit(
        "production teacher cache may not use ground_truth pose guidance: "
        f"{ground_truth_pose[:10]}"
    )
if profile == "probe" and Path(sys.argv[1]).stat().st_size > 256 * 1024 * 1024:
    raise SystemExit("probe teacher cache exceeds 256 MiB")
expected = Counter(
    (record.video_id, frame.frame_ref, frame.timestamp)
    for record in sensor
    for frame in record.selected_frames
)
actual = Counter(
    (row.request.video_id, row.request.frame_ref, row.request.timestamp)
    for row in cache
)
if actual != expected:
    missing = sorted((expected - actual).elements())[:10]
    extra = sorted((actual - expected).elements())[:10]
    raise SystemExit(
        "teacher cache sensor observation mismatch: "
        f"missing={missing} extra={extra}"
    )
PY
            : "${WORLDMM_STUDENT_SUPERVISION_INPUT:?student supervision required}"
            python -m worldmm_smvqa.teacher_materializer \
              --teacher-cache "$WORLDMM_OUTPUT_ROOT/teacher/cache.jsonl" \
              --supervision "$WORLDMM_STUDENT_SUPERVISION_INPUT" \
              --out "$WORLDMM_OUTPUT_ROOT/training/student_teacher_cache.jsonl"
            ;;
          train)
            distributed_train
            ;;
          build_memory)
            distributed_memory episodic \
              "$WORLDMM_OUTPUT_ROOT/memory/episodic.jsonl"
            distributed_memory semantic,visual \
              "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv" \
              "$WORLDMM_OUTPUT_ROOT/memory/episodic.jsonl"
            ;;
          student_infer_retrieve)
            typed_records="$WORLDMM_OUTPUT_ROOT/memory/typed_memory.jsonl"
            typed_manifest="$WORLDMM_OUTPUT_ROOT/memory"
            typed_manifest+="/typed_memory.inference.json"
            sensed_sources="$WORLDMM_OUTPUT_ROOT/inference_inputs/sources.jsonl"
            sensed_sources_sha256="$(sha256sum "$sensed_sources" | cut -d ' ' -f 1)"
            frame_assets_sha256="$(
              sha256sum "$WORLDMM_OUTPUT_ROOT/diagnostics/frame_assets.sha256" \
                | cut -d ' ' -f 1
            )"
            spatial_infer_sha256="$(
              sha256sum "$WORLDMM_SPATIAL_INFER_EXE" | cut -d ' ' -f 1
            )"
            /opt/slurm/bin/srun \
              --nodes=1 --ntasks=1 --gpus-per-task=1 \
              --kill-on-bad-exit=1 \
              env \
                -u RUN_FIXTURE -u SMVQA_DATA_ROOT \
                -u WORLDMM_STUDENT_SUPERVISION_INPUT \
                -u WORLDMM_TEACHER_CACHE_INPUT \
                -u WORLDMM_APPROVAL_FILE -u WORLDMM_APPROVER \
                -u WORLDMM_APPROVAL_SHA256 -u GEMMA_MODEL_PATH \
                -u WORLDMM_MEMORY_MODEL_PATH -u WORLDMM_REMOTE_REPO \
                -u WORLDMM_OUTPUT_ROOT \
                -u WORLDMM_SPATIAL_INFER_EXE \
              "$WORLDMM_SPATIAL_INFER_EXE" \
              --checkpoint "$WORLDMM_OUTPUT_ROOT/checkpoints/spatial_student.pt" \
              --sources "$sensed_sources" \
              --sources-sha256 "$sensed_sources_sha256" \
              --frame-root "$SMVQA_FRAME_ROOT" \
              --frame-assets-sha256 "$frame_assets_sha256" \
              --producer-sha256 "$spatial_infer_sha256" \
              --sensor-frame-manifest "$WORLDMM_SENSOR_FRAME_MANIFEST" \
              --out-records "$typed_records" \
              --out-manifest "$typed_manifest" \
              --byte-budget-per-window \
                "$WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW"
            python - \
              "$WORLDMM_OUTPUT_ROOT/checkpoints/spatial_student.pt" \
              "$WORLDMM_SENSOR_FRAME_MANIFEST" \
              "$WORLDMM_OUTPUT_ROOT/inference_inputs/sources.jsonl" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/frame_assets.sha256" \
              "$WORLDMM_SPATIAL_INFER_EXE" \
              "$typed_records" "$typed_manifest" \
              "$WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
from worldmm_smvqa.schema import SourceStreamExample
from worldmm_smvqa.sensor_frames import read_sensor_frame_manifest
from worldmm_smvqa.worldmm.typed_memory import (
    DEFAULT_TYPED_MEMORY_WINDOW_SECONDS,
    validate_typed_memory_artifact,
)

checkpoint, sensor, sources_path, frame_assets, producer, records, manifest = map(
    Path,
    sys.argv[1:8],
)
byte_budget_per_window = int(sys.argv[8])
sources = tuple(
    SourceStreamExample.model_validate_json(line)
    for line in sources_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
)
sensor_records = read_sensor_frame_manifest(sensor)
artifact = validate_typed_memory_artifact(
    records,
    byte_budget_per_window=byte_budget_per_window,
    window_seconds=DEFAULT_TYPED_MEMORY_WINDOW_SECONDS,
    sources=sources,
    sensor_records=sensor_records,
)
if os.environ["WORLDMM_EXECUTION_PROFILE"] == "probe" and (
    artifact.record_count > 10_000 or artifact.actual_bytes > 16 * 1024 * 1024
):
    raise SystemExit(
        "probe typed memory exceeds 10000 records or 16 MiB: "
        f"records={artifact.record_count} bytes={artifact.actual_bytes}"
    )

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = json.loads(manifest.read_text(encoding="utf-8"))
expected = {
    "schema_version": 1,
    "production_ready": True,
    "result_class": "student",
    "producer": "spatial-student",
    "checkpoint_sha256": sha256(checkpoint),
    "sensor_sha256": sha256(sensor),
    "sources_sha256": sha256(sources_path),
    "frame_assets_sha256": sha256(frame_assets),
    "producer_sha256": sha256(producer),
    "records_sha256": sha256(records),
    "record_count": artifact.record_count,
    "byte_budget_per_window": byte_budget_per_window,
    "actual_bytes": artifact.actual_bytes,
    "window_count": artifact.window_count,
    "max_window_bytes": artifact.max_window_bytes,
    "window_seconds": DEFAULT_TYPED_MEMORY_WINDOW_SECONDS,
}
for key, value in expected.items():
    actual = payload.get(key)
    if type(actual) is not type(value) or actual != value:
        raise SystemExit(
            f"spatial inference manifest mismatch for {key}: "
            f"expected {value!r}, got {actual!r}"
        )
if expected["record_count"] <= 0:
    raise SystemExit("spatial inference produced no typed records")
PY
            python - <<'PY' \
              > "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json"
import json
import os
from pathlib import Path

root = Path(os.environ["WORLDMM_OUTPUT_ROOT"])
payload = {
    "schema_version": 1,
    "episodic_memory": str(root / "memory/episodic.jsonl"),
    "semantic_memory": str(root / "memory/worldmm_sv/semantic.jsonl"),
    "visual_memory": str(root / "memory/worldmm_sv/visual.jsonl"),
    "spatial_memory": {"path": str(root / "memory/typed_memory.jsonl")},
}
print(json.dumps(payload, sort_keys=True))
PY
            memory_hashes="$WORLDMM_OUTPUT_ROOT/retrieval/memory_inputs.json"
            python - "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json" \
              "$memory_hashes" <<'PY'
import json
import os
import sys
from pathlib import Path
from worldmm_smvqa.qa_transformers import memory_artifact_hashes

manifest, output = map(Path, sys.argv[1:])
payload = memory_artifact_hashes(manifest)
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
            memory_inputs="$WORLDMM_OUTPUT_ROOT/retrieval/memory_inputs.sha256"
            sha256sum -- \
              "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json" \
              "$WORLDMM_OUTPUT_ROOT/memory/episodic.jsonl" \
              "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv/semantic.jsonl" \
              "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv/visual.jsonl" \
              "$WORLDMM_OUTPUT_ROOT/memory/typed_memory.jsonl" \
              "$memory_hashes" > "$memory_inputs"
            memory_inputs_sha256="$(
              sha256sum "$memory_inputs" | cut -d ' ' -f 1
            )"
            worldmm-smvqa retrieve-batch \
              --config configs/remote.example.yaml \
              --fixture "$RUN_FIXTURE" \
              --stores episodic,semantic,visual,spatial \
              --retrieval-protocol worldmm-smvqa \
              --max-frame-refs 32 \
              --input "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json" \
              --out "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl"
            sha256sum --check --status "$memory_inputs"
            python - \
              "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl" \
              "$WORLDMM_OUTPUT_ROOT/checkpoints/spatial_student.pt" \
              "$WORLDMM_EXECUTION_REPO/configs/remote.example.yaml" \
              "$WORLDMM_SENSOR_FRAME_MANIFEST" "$RUN_FIXTURE" \
              "$WORLDMM_OUTPUT_ROOT/memory/typed_memory.jsonl" \
              "$WORLDMM_OUTPUT_ROOT/memory/typed_memory.inference.json" \
              "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json" \
              "$memory_hashes" "$memory_inputs" \
              "$memory_inputs_sha256" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
from worldmm_smvqa.qa_transformers import memory_artifact_hashes

(
    evidence,
    checkpoint,
    config,
    sensor,
    data_root,
    typed_memory,
    inference_manifest,
    memory_manifest,
    memory_hashes,
    memory_inputs,
) = map(Path, sys.argv[1:-1])
expected_memory_inputs_sha256 = sys.argv[-1]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if sha256(memory_inputs) != expected_memory_inputs_sha256:
    raise SystemExit("retrieval memory-input ledger changed after retrieval")
sealed_memory_hashes = json.loads(memory_hashes.read_text(encoding="utf-8"))
current_memory_hashes = memory_artifact_hashes(memory_manifest)
if sealed_memory_hashes != current_memory_hashes:
    raise SystemExit("memory stores changed while retrieval evidence was produced")

data_digest = hashlib.sha256()
for name in ("sources.jsonl", "questions.jsonl", "labels.jsonl"):
    data_digest.update(name.encode("utf-8") + b"\0")
    with (data_root / name).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            data_digest.update(chunk)
payload = {
    "lane": "student",
    "producer": "spatial-student",
    "evidence_sha256": sha256(evidence),
    "checkpoint_sha256": sha256(checkpoint),
    "config_sha256": sha256(config),
    "sensor_sha256": sha256(sensor),
    "data_sha256": data_digest.hexdigest(),
    "typed_memory_sha256": sha256(typed_memory),
    "inference_manifest_sha256": sha256(inference_manifest),
    **sealed_memory_hashes,
}
output = Path(f"{evidence}.lineage.json")
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
            sha256sum --check --status "$memory_inputs"
            if [ "$(sha256sum "$memory_inputs" | cut -d ' ' -f 1)" != \
              "$memory_inputs_sha256" ]; then
              printf "retrieval memory-input ledger changed during lineage write\n" >&2
              exit 1
            fi
            ;;
          qa)
            distributed_qa
            python - \
              "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl" \
              "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl.manifest.json" \
              "$WORLDMM_OUTPUT_ROOT/qa/completed.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

predictions, resume_manifest, output = map(Path, sys.argv[1:])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "schema_version": 1,
    "predictions_sha256": sha256(predictions),
    "qa_resume_manifest_sha256": sha256(resume_manifest),
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
            ;;
          metrics_report)
            if [ "$WORLDMM_EXECUTION_PROFILE" = "probe" ]; then
              result_class=contract_probe
              experiment=PROBE
            else
              result_class=student
              experiment=E1
            fi
            summary_path="$WORLDMM_OUTPUT_ROOT/summary/summary.txt"
            summary_temporary="${summary_path}.$$"
            printf "%s\n" \
              "run_id=$WORLDMM_RUN_ID" \
              "output_root=$WORLDMM_OUTPUT_ROOT" \
              "result_class=$result_class" \
              "experiment=$experiment" \
              "metrics=$WORLDMM_OUTPUT_ROOT/metrics/metrics.json" \
              > "$summary_temporary"
            mv "$summary_temporary" "$summary_path"
            finalization_inputs="$WORLDMM_OUTPUT_ROOT/summary"
            finalization_inputs+="/finalization_inputs.sha256"
            sha256sum -- \
              "$WORLDMM_APPROVAL_FILE" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/deployed_code.sha256" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/deployed_code.files.sha256" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/env_contract.json" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/frame_assets.sha256" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/preflight_inputs.sha256" \
              "$WORLDMM_OUTPUT_ROOT/checkpoints/spatial_student.pt" \
              "$WORLDMM_OUTPUT_ROOT/memory/typed_memory.inference.json" \
              "$WORLDMM_OUTPUT_ROOT/memory/typed_memory.jsonl" \
              "$WORLDMM_OUTPUT_ROOT/inference_inputs/sources.jsonl" \
              "$WORLDMM_SPATIAL_INFER_EXE" \
              "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json" \
              "$WORLDMM_OUTPUT_ROOT/memory/episodic.jsonl" \
              "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv/semantic.jsonl" \
              "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv/visual.jsonl" \
              "$WORLDMM_OUTPUT_ROOT/retrieval/memory_inputs.sha256" \
              "$WORLDMM_OUTPUT_ROOT/retrieval/memory_inputs.json" \
              "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl" \
              "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl.lineage.json" \
              "$WORLDMM_OUTPUT_ROOT/diagnostics/gemma_model.sha256" \
              "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl" \
              "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl.manifest.json" \
              "$WORLDMM_OUTPUT_ROOT/qa/completed.json" \
              "$summary_path" \
              "$WORLDMM_EXECUTION_REPO/configs/remote.example.yaml" \
              "$WORLDMM_SENSOR_FRAME_MANIFEST" \
              "$RUN_FIXTURE/sources.jsonl" \
              "$RUN_FIXTURE/questions.jsonl" \
              "$RUN_FIXTURE/labels.jsonl" > "$finalization_inputs"
            finalization_inputs_sha256="$(
              sha256sum "$finalization_inputs" | cut -d ' ' -f 1
            )"
            verify_final_code_snapshot() {
              local deployed_files_current
              (
                cd "$WORLDMM_EXECUTION_REPO"
                sha256sum --check --status \
                  "$WORLDMM_OUTPUT_ROOT/diagnostics/deployed_code.sha256"
              )
              deployed_files_current="${finalization_inputs}.code.$$"
              (
                cd "$WORLDMM_EXECUTION_REPO"
                find . -type f -print0 | sort -z | sha256sum
              ) > "$deployed_files_current"
              cmp -s \
                "$WORLDMM_OUTPUT_ROOT/diagnostics/deployed_code.files.sha256" \
                "$deployed_files_current"
              rm -f "$deployed_files_current"
            }
            verify_final_code_snapshot
            worldmm-smvqa evaluate \
              --config configs/remote.example.yaml \
              --pred "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl" \
              --labels "$RUN_FIXTURE/labels.jsonl" \
              --out "$WORLDMM_OUTPUT_ROOT/metrics/metrics.json"
            sha256sum --check --status "$finalization_inputs"
            verify_final_code_snapshot
            python - "$finalization_inputs_sha256" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from worldmm_smvqa.chunking import read_source_streams
from worldmm_smvqa.fixtures import read_fixture_questions
from worldmm_smvqa.qa_transformers import (
    TransformersCliArgs,
    qa_resume_manifest,
    validate_evidence_lineage,
)
from worldmm_smvqa.report import RemoteRunManifest, render_report
from worldmm_smvqa.schema import PredictionRecord
from worldmm_smvqa.sensor_frames import read_sensor_frame_manifest

root = Path(os.environ["WORLDMM_OUTPUT_ROOT"])
expected_finalization_inputs_sha256 = sys.argv[1]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()

def verify_code_snapshot(paths: dict[str, Path]) -> None:
    execution_repo = Path(os.environ["WORLDMM_EXECUTION_REPO"])
    checked = subprocess.run(
        ["sha256sum", "--check", "--status", str(paths["deployed_code"])],
        cwd=execution_repo,
        check=False,
    )
    if checked.returncode != 0:
        raise SystemExit("approved code snapshot content changed")
    entries = tuple(execution_repo.rglob("*"))
    if any(entry.is_symlink() for entry in entries):
        raise SystemExit("approved code snapshot contains a symlink")
    digest = hashlib.sha256()
    relative_files = sorted(
        (entry.relative_to(execution_repo) for entry in entries if entry.is_file()),
        key=lambda path: os.fsencode(path.as_posix()),
    )
    for relative in relative_files:
        digest.update(os.fsencode(f"./{relative.as_posix()}"))
        digest.update(b"\0")
    expected = paths["deployed_code_files"].read_text(encoding="utf-8").split()[0]
    if digest.hexdigest() != expected:
        raise SystemExit("approved code snapshot file inventory changed")

paths = {
    "approval": Path(os.environ["WORLDMM_APPROVAL_FILE"]),
    "deployed_code": root / "diagnostics/deployed_code.sha256",
    "deployed_code_files": root / "diagnostics/deployed_code.files.sha256",
    "env_contract": root / "diagnostics/env_contract.json",
    "frame_assets": root / "diagnostics/frame_assets.sha256",
    "preflight_inputs": root / "diagnostics/preflight_inputs.sha256",
    "checkpoint": root / "checkpoints/spatial_student.pt",
    "inference_manifest": root / "memory/typed_memory.inference.json",
    "inference_sources": root / "inference_inputs/sources.jsonl",
    "inference_producer": Path(os.environ["WORLDMM_SPATIAL_INFER_EXE"]),
    "typed_memory": root / "memory/typed_memory.jsonl",
    "memory_manifest": root / "memory/memory_manifest.json",
    "episodic_memory": root / "memory/episodic.jsonl",
    "semantic_memory": root / "memory/worldmm_sv/semantic.jsonl",
    "visual_memory": root / "memory/worldmm_sv/visual.jsonl",
    "memory_inputs": root / "retrieval/memory_inputs.sha256",
    "memory_hashes": root / "retrieval/memory_inputs.json",
    "evidence_lineage": root / "retrieval/evidence_packs.jsonl.lineage.json",
    "evidence": root / "retrieval/evidence_packs.jsonl",
    "gemma_model_fingerprint": root / "diagnostics/gemma_model.sha256",
    "predictions": root / "qa/predictions.jsonl",
    "qa_resume_manifest": root / "qa/predictions.jsonl.manifest.json",
    "qa_completion": root / "qa/completed.json",
    "summary": root / "summary/summary.txt",
    "finalization_inputs": root / "summary/finalization_inputs.sha256",
    "config": Path(os.environ["WORLDMM_EXECUTION_REPO"])
    / "configs/remote.example.yaml",
    "sensor": Path(os.environ["WORLDMM_SENSOR_FRAME_MANIFEST"]),
    "sources": Path(os.environ["RUN_FIXTURE"]) / "sources.jsonl",
    "questions": Path(os.environ["RUN_FIXTURE"]) / "questions.jsonl",
    "labels": Path(os.environ["RUN_FIXTURE"]) / "labels.jsonl",
    "metrics": root / "metrics/metrics.json",
}
if sha256(paths["finalization_inputs"]) != expected_finalization_inputs_sha256:
    raise SystemExit("finalization input seal changed after evaluation")
verify_code_snapshot(paths)
artifact_hashes = {f"{name}_sha256": sha256(path) for name, path in paths.items()}
seal_check = subprocess.run(
    ["sha256sum", "--check", "--status", str(paths["finalization_inputs"])],
    check=False,
)
if seal_check.returncode != 0:
    raise SystemExit("finalization input seal no longer matches current artifacts")
qa_completion = json.loads(paths["qa_completion"].read_text(encoding="utf-8"))
expected_qa_completion = {
    "schema_version": 1,
    "predictions_sha256": sha256(paths["predictions"]),
    "qa_resume_manifest_sha256": sha256(paths["qa_resume_manifest"]),
}
if qa_completion != expected_qa_completion:
    raise SystemExit("QA outputs changed after QA stage completion")
fixture = Path(os.environ["RUN_FIXTURE"])
qa_args = TransformersCliArgs(
    model=os.environ["GEMMA_MODEL_PATH"],
    fixture=fixture,
    evidence=paths["evidence"],
    evidence_lane="student",
    evidence_lineage=paths["evidence_lineage"],
    checkpoint=paths["checkpoint"],
    typed_memory=paths["typed_memory"],
    inference_manifest=paths["inference_manifest"],
    inference_sources=paths["inference_sources"],
    inference_producer=paths["inference_producer"],
    require_frames=True,
    out=paths["predictions"],
    backend="gemma4",
    model_fingerprint=paths["gemma_model_fingerprint"],
    frame_assets_manifest=paths["frame_assets"],
    lineage_config=paths["config"],
    sensor_frame_manifest=paths["sensor"],
    memory_manifest=paths["memory_manifest"],
)
expected_resume = qa_resume_manifest(qa_args)
actual_resume = json.loads(
    paths["qa_resume_manifest"].read_text(encoding="utf-8")
)
if actual_resume != expected_resume:
    raise SystemExit("QA resume manifest no longer matches current QA inputs")
raw_sources = read_source_streams(fixture, use_sensor_manifest=False)
sensor_records = read_sensor_frame_manifest(qa_args.sensor_frame_manifest)
validate_evidence_lineage(
    paths["evidence"],
    "student",
    paths["evidence_lineage"],
    paths["checkpoint"],
    paths["typed_memory"],
    paths["inference_manifest"],
    config_path=qa_args.lineage_config,
    sensor_path=qa_args.sensor_frame_manifest,
    data_root=fixture,
    memory_manifest_path=paths["memory_manifest"],
    inference_sources_path=paths["inference_sources"],
    frame_assets_path=paths["frame_assets"],
    inference_producer_path=paths["inference_producer"],
    sources=raw_sources,
    sensor_records=sensor_records,
)
prediction_records = tuple(
    PredictionRecord.model_validate_json(line)
    for line in paths["predictions"].read_text(encoding="utf-8").splitlines()
    if line.strip()
)
actual_question_ids = tuple(record.question_id for record in prediction_records)
expected_question_ids = tuple(
    question.question_id for question in read_fixture_questions(fixture)
)
if len(actual_question_ids) != len(set(actual_question_ids)):
    raise SystemExit("predictions contain duplicate question IDs")
if set(actual_question_ids) != set(expected_question_ids):
    raise SystemExit("predictions do not cover the exact approved question set")
prompt_rows = []
for line in paths["predictions"].read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    prediction = json.loads(line)
    prompt_sha256 = prediction.get("prompt_sha256")
    frame_refs = prediction.get("input_frame_refs")
    if not isinstance(prompt_sha256, str) or len(prompt_sha256) != 64:
        raise SystemExit("prediction missing prompt_sha256")
    if (
        not isinstance(frame_refs, list)
        or not frame_refs
        or not all(isinstance(ref, str) and "/" in ref for ref in frame_refs)
    ):
        raise SystemExit("prediction missing input_frame_refs")
    prompt_rows.append(
        {
            "question_id": prediction.get("question_id"),
            "prompt_sha256": prompt_sha256,
            "input_frame_refs": frame_refs,
        }
    )
if not prompt_rows:
    raise SystemExit("predictions contain no prompt audit rows")
prompt_rows.sort(key=lambda row: str(row["question_id"]))
prompt_sha256 = sha256_json(prompt_rows)
split_parts = {
    name: sha256(Path(os.environ["RUN_FIXTURE"]) / name)
    for name in ("sources.jsonl", "questions.jsonl", "labels.jsonl")
}
split_id = sha256_json(split_parts)
metrics_payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
metric_names = ("Ans-F1", "QA-Acc", "QA-MRR")
metrics = []
for name in metric_names:
    value = metrics_payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"missing numeric E1 metric: {name}")
    if not 0.0 <= float(value) <= 100.0:
        raise SystemExit(f"E1 metric outside [0,100]: {name}={value}")
    metrics.append({"experiment": "E1", "name": name, "value": float(value)})
code_sha256 = artifact_hashes["deployed_code_sha256"]
model_sha256 = artifact_hashes["gemma_model_fingerprint_sha256"]
execution_profile = os.environ["WORLDMM_EXECUTION_PROFILE"]
is_probe = execution_profile == "probe"
result_class = "contract_probe" if is_probe else "student"
experiment_label = "PROBE" if is_probe else "E1"
experiment_id = f'{os.environ["WORLDMM_RUN_ID"]}-{experiment_label}'
for metric in metrics:
    metric["experiment"] = experiment_label
changed_artifacts = tuple(
    name
    for name, path in paths.items()
    if sha256(path) != artifact_hashes[f"{name}_sha256"]
)
if changed_artifacts:
    raise SystemExit(
        "finalization inputs changed while being validated: "
        + ",".join(changed_artifacts)
    )
if sha256(paths["finalization_inputs"]) != expected_finalization_inputs_sha256:
    raise SystemExit("finalization input seal changed before publication")
verify_code_snapshot(paths)
identity = {
    "schema_version": 1,
    "experiment_id": experiment_id,
    "result_class": result_class,
    "lane": "student",
    "profile": execution_profile,
    "run_id": os.environ["WORLDMM_RUN_ID"],
    "run_fixture": os.environ["RUN_FIXTURE"],
    "split_id": split_id,
    "split_parts": split_parts,
    "code_sha256": code_sha256,
    "model_sha256": model_sha256,
    "prompt_sha256": prompt_sha256,
    "prompt_count": len(prompt_rows),
    **artifact_hashes,
}
identity_bytes = (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode()
run_identity_sha256 = hashlib.sha256(identity_bytes).hexdigest()
manifest_payload = {
    "baseline_name": "WorldMM-SMVQA",
    "remote_status": "complete",
    "result_class": result_class,
    "experiment_id": experiment_id,
    "execution_profile": execution_profile,
    "lane": "student",
    "split_id": split_id,
    "code_sha256": code_sha256,
    "checkpoint_sha256": artifact_hashes["checkpoint_sha256"],
    "typed_memory_sha256": artifact_hashes["typed_memory_sha256"],
    "inference_manifest_sha256": artifact_hashes["inference_manifest_sha256"],
    "evidence_sha256": artifact_hashes["evidence_sha256"],
    "evidence_lineage_sha256": artifact_hashes["evidence_lineage_sha256"],
    "model_sha256": model_sha256,
    "prompt_sha256": prompt_sha256,
    "predictions_sha256": artifact_hashes["predictions_sha256"],
    "metrics_sha256": artifact_hashes["metrics_sha256"],
    "qa_resume_manifest_sha256": artifact_hashes["qa_resume_manifest_sha256"],
    "run_identity_sha256": run_identity_sha256,
    "finalization_inputs_sha256": expected_finalization_inputs_sha256,
    "local_changes": [f"deployed_code_sha256={code_sha256}"],
    "remote_command": (
        "WORLDMM_DAG_PHASE=run bash "
        f"{root}/code_snapshot/remote-plan/submit_worldmm_smvqa_dag.sh"
    ),
    "remote_job_reference": os.environ["SLURM_JOB_ID"],
    "remote_artifact_path": str(root),
    "metrics": metrics,
    "failure_reason": None,
    "not_copied_locally": [
        "full datasets and frame corpora",
        "model weights",
        "checkpoints",
        "embeddings, teacher caches, and full evidence packs",
        "sensitive artifacts and unredacted logs",
    ],
}
manifest = RemoteRunManifest.model_validate(manifest_payload)
output = root / "summary/run_identity.json"
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_bytes(identity_bytes)
manifest_path = root / "summary/remote_manifest.json"
manifest_temporary = manifest_path.with_name(
    f".{manifest_path.name}.{os.getpid()}.tmp"
)
manifest_temporary.write_text(manifest.model_dump_json(indent=2) + "\n")
report_path = root / "summary/final_report.md"
report_temporary = report_path.with_name(
    f".{report_path.name}.{os.getpid()}.tmp"
)
report_temporary.write_text(render_report(manifest), encoding="utf-8")
temporary.replace(output)
report_temporary.replace(report_path)
manifest_temporary.replace(manifest_path)
PY
            ;;
          *)
            printf "unknown WORLDMM_STAGE: %s\n" "$WORLDMM_STAGE" >&2
            exit 2
            ;;
        esac
        """,
    ).lstrip()


def _teacher_oracle_submit_script_text(graph: object | None = None) -> str:
    """Render the isolated, signed teacher-oracle DAG."""
    graph_json = getattr(graph, "model_dump_json", lambda: "")()
    graph_sha256 = hashlib.sha256(graph_json.encode("utf-8")).hexdigest()
    return (
        f"#!/usr/bin/env bash\n# graph_sha256={graph_sha256}\n"
        r"""set -euo pipefail
umask 077
: "${WORLDMM_EXECUTION_PROFILE:?teacher-oracle profile required}"
[ "$WORLDMM_EXECUTION_PROFILE" = teacher-oracle ] || {
  echo "WORLDMM_EXECUTION_PROFILE must be teacher-oracle" >&2
  exit 2
}
: "${WORLDMM_REMOTE_REPO:?}"
: "${WORLDMM_ATTESTED_RUNTIME_ROOT:?}"
[ -d "$WORLDMM_ATTESTED_RUNTIME_ROOT" ] && \
  [ -x "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" ] || {
  echo "attested runtime root is not an executable isolated runtime" >&2
  exit 2
}
: "${WORLDMM_RUN_ID:?}"
: "${WORLDMM_OUTPUT_ROOT:?}"
: "${WORLDMM_EXPERIMENT_ID:=EXP-0005}"
: "${WORLDMM_SENSOR_AUDIT_SHA256:?}"
: "${WORLDMM_PROVIDER_SHA256:?}"
: "${WORLDMM_SPLIT_SHA256:?}"
: "${WORLDMM_CODE_SHA:?}"
: "${WORLDMM_POLICY_SHA256:?}"
: "${WORLDMM_TEACHER_ORACLE_VALIDATION_RECEIPT:?}"
: "${WORLDMM_TEACHER_ORACLE_VALIDATION_SHA256:?}"
: "${WORLDMM_EXPERIMENT_CONFIG_SHA256:?}"
: "${WORLDMM_SLURM_CLUSTER:?}"
: "${WORLDMM_FRAME_ASSETS_SHA256:?}"
: "${WORLDMM_BYTE_BUDGET_SHA256:?}"
: "${WORLDMM_RESOURCE_CONFIG_SHA256:?}"
: "${WORLDMM_PLAN_SHA256:?}"
: "${WORLDMM_REMOTE_SNAPSHOT_SHA256:?}"
: "${WORLDMM_DAG_PREFLIGHT_SUBMIT_SCRIPT_SHA256:?}"
: "${WORLDMM_DAG_PROVIDER_GATE_SUBMIT_SCRIPT_SHA256:?}"
: "${WORLDMM_DAG_DOWNSTREAM_SUBMIT_SCRIPT_SHA256:?}"
: "${WORLDMM_DAG_STAGE_SCRIPT_SHA256:?}"
: "${WORLDMM_ORACLE_PROVIDER_EXECUTABLE_SHA256:?}"
: "${WORLDMM_ORACLE_STAGE_EXECUTABLE_SHA256:?}"
: "${WORLDMM_ATTESTED_RUNTIME_MANIFEST:?runtime content manifest required}"
: "${WORLDMM_ATTESTED_RUNTIME_MANIFEST_SHA256:?runtime content manifest digest required}"
case "$(basename "$0")" in
  submit_teacher_oracle_preflight.sh)
    WORLDMM_DAG_PHASE=phase-a
    WORLDMM_ORACLE_SUBMISSION_SURFACE=preflight
    expected_submit_digest="$WORLDMM_DAG_PREFLIGHT_SUBMIT_SCRIPT_SHA256"
    ;;
  submit_teacher_oracle_provider_gate.sh)
    WORLDMM_DAG_PHASE=phase-a
    WORLDMM_ORACLE_SUBMISSION_SURFACE=provider
    expected_submit_digest="$WORLDMM_DAG_PROVIDER_GATE_SUBMIT_SCRIPT_SHA256"
    ;;
  submit_teacher_oracle_downstream.sh)
    WORLDMM_DAG_PHASE=phase-b
    WORLDMM_ORACLE_SUBMISSION_SURFACE=downstream
    expected_submit_digest="$WORLDMM_DAG_DOWNSTREAM_SUBMIT_SCRIPT_SHA256"
    ;;
  *) echo "unapproved teacher-oracle submitter name" >&2; exit 2 ;;
esac
if [ "$WORLDMM_ORACLE_SUBMISSION_SURFACE" != preflight ]; then
  : "${WORLDMM_SMVQA_REMOTE_APPROVED:?explicit approval required}"
  [ "$WORLDMM_SMVQA_REMOTE_APPROVED" = 1 ] || exit 2
  : "${WORLDMM_APPROVAL_FILE:?}"
  : "${WORLDMM_SIGNER_REGISTRY:?}"
  : "${WORLDMM_SIGNER_REGISTRY_SHA256:?}"
fi
if [ "$WORLDMM_ORACLE_SUBMISSION_SURFACE" = provider ]; then
  : "${WORLDMM_PREFLIGHT_SEAL_SHA256:?approval-bound preflight seal required}"
  : "${WORLDMM_ACCOUNTING_SETTLE_SECONDS:?approval-bound accounting timeout required}"
  : "${WORLDMM_ACCOUNTING_SETTLE_INTERVAL_SECONDS:?approval-bound accounting interval required}"
  : "${WORLDMM_ORACLE_QUALITY_EVALUATOR:?approval-bound quality evaluator required}"
  : "${WORLDMM_ORACLE_QUALITY_CONTRACT:?approval-bound quality contract required}"
  : "${WORLDMM_ORACLE_QUALITY_CONTRACT_SHA256:?approval-bound quality contract digest required}"
  : "${WORLDMM_ORACLE_QUALITY_EVALUATOR_SHA256:?approval-bound quality evaluator digest required}"
fi
if [ "$WORLDMM_ORACLE_SUBMISSION_SURFACE" = downstream ]; then
  : "${WORLDMM_QA_SHARD_MAP_SHA256:?approval-bound QA shard map required}"
  : "${WORLDMM_QA_LINEAGE_SHA256:?approval-bound QA lineage required}"
  : "${WORLDMM_QA_FINALIZATION_RECEIPT_SHA256:?approval-bound QA finalization receipt required}"
  : "${WORLDMM_QA_PREDICTIONS_SHA256:?approval-bound QA prediction digest required}"
  : "${WORLDMM_QA_SHARD_MAP:?approval-bound QA shard map path required}"
  : "${WORLDMM_QA_LINEAGE:?approval-bound QA lineage path required}"
  : "${WORLDMM_QA_FINALIZATION_RECEIPT:?approval-bound QA finalization receipt path required}"
  : "${WORLDMM_QA_PREDICTIONS:?approval-bound QA predictions path required}"
fi
root="${WORLDMM_OUTPUT_ROOT%/}"
case "$root" in */"$WORLDMM_RUN_ID") ;; *) exit 2;; esac
case "$WORLDMM_DAG_PHASE" in phase-a|phase-b) ;; *) exit 2;; esac
phase_lock="$root/summary/.teacher_oracle_${WORLDMM_DAG_PHASE}_${WORLDMM_ORACLE_SUBMISSION_SURFACE}.lock"
attempt_journal="$root/summary/teacher_oracle_${WORLDMM_DAG_PHASE}_${WORLDMM_ORACLE_SUBMISSION_SURFACE}.attempts.jsonl"
submitted_job_ids=()
unknown_submission=0
transaction_committed=0
cleanup_submission() {
  [ "$transaction_committed" = 1 ] && return
  [ "$unknown_submission" = 0 ] || {
    echo "untrustworthy sbatch result; retaining phase lock for reconciliation" >&2
  }
}
trap cleanup_submission EXIT
if [ "${1:-}" = --reconcile-unknown-sbatch ]; then
  [ -f "$attempt_journal" ] || {
    echo "unknown-sbatch reconciliation requires an immutable attempt journal" >&2
    exit 1
  }
  WORLDMM_RECONCILIATION_JOURNAL="$attempt_journal" \
    WORLDMM_SQUEUE="${WORLDMM_SQUEUE:-squeue}" \
    WORLDMM_SACCT="${WORLDMM_SACCT:-sacct}" \
    "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - <<'PY'
import hashlib, json, os, stat, subprocess
from pathlib import Path

journal = Path(os.environ["WORLDMM_RECONCILIATION_JOURNAL"])
artifact = journal.with_name(f"{journal.stem}.reconciliation.json")
unknown, reconciled, base_lines = {}, set(), []
for raw_line in journal.read_bytes().splitlines(keepends=True):
    value = json.loads(raw_line)
    identity = value.get("identity")
    if not isinstance(identity, str):
        base_lines.append(raw_line)
        continue
    if value.get("event") == "submission-unknown-before-sbatch":
        unknown[identity] = value
        base_lines.append(raw_line)
    elif value.get("event") == "submission-reconciled":
        reconciled.add(identity)
    else:
        base_lines.append(raw_line)
base_journal = b"".join(base_lines)
base_journal_sha256 = hashlib.sha256(base_journal).hexdigest()

def stable_artifact():
    fd = os.open(artifact, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        state = os.fstat(fd)
        if (
            not stat.S_ISREG(state.st_mode)
            or state.st_uid != os.getuid()
            or stat.S_IMODE(state.st_mode) != 0o600
            or state.st_nlink != 1
        ):
            raise SystemExit("unsafe reconciliation artifact")
        raw = os.read(fd, state.st_size + 1)
        end = os.fstat(fd)
        if (
            len(raw) != state.st_size
            or (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns)
            != (end.st_dev, end.st_ino, end.st_size, end.st_mtime_ns)
        ):
            raise SystemExit("reconciliation artifact changed while read")
        return json.loads(raw)
    finally:
        os.close(fd)

if artifact.exists() or artifact.is_symlink():
    recorded = stable_artifact()
    queries = recorded.get("queries")
    results = recorded.get("results")
    if (
        recorded.get("schema_version") != 1
        or recorded.get("kind") != "UnknownSubmissionReconciliationV1"
        or recorded.get("journal_sha256") != base_journal_sha256
        or not isinstance(queries, list)
        or not isinstance(results, list)
        or any(not isinstance(result, dict) for result in results)
        or recorded.get("query_evidence_sha256")
        != hashlib.sha256(
            json.dumps(queries, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        or {
            result.get("identity")
            for result in results
            if isinstance(result, dict)
        }
        != set(unknown)
    ):
        raise SystemExit("reconciliation artifact does not bind retained state")
else:
    rows, queries = [], []
    for command in (
        [os.environ["WORLDMM_SQUEUE"], "--noheader", "--format=%i|%j|%k"],
        [
            os.environ["WORLDMM_SACCT"],
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,JobName,Comment",
        ],
    ):
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            raise SystemExit(f"Slurm reconciliation query failed: {exc}") from exc
        query = {
            "argv": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        queries.append(query)
        if completed.returncode != 0:
            raise SystemExit(
                f"Slurm reconciliation query failed: {command!r}: "
                f"{completed.stderr}"
            )
        rows.extend(completed.stdout.splitlines())
    query_evidence_sha256 = hashlib.sha256(
        json.dumps(queries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    results = []
    for identity, descriptor in unknown.items():
        name = f"worldmm-{descriptor['run_id']}-{descriptor['stage']}"
        matches = {
            row.split("|", 1)[0].strip().split(".", 1)[0]
            for row in rows
            if len(row.split("|")) >= 3
            and row.split("|", 1)[0].strip().split(".", 1)[0].isdigit()
            and row.split("|", 2)[1].strip() == name
            and row.split("|", 2)[2].strip() == identity
        }
        if len(matches) > 1:
            raise SystemExit(
                f"ambiguous Slurm reconciliation for {identity}: "
                f"{sorted(matches)}"
            )
        results.append({
            "schema_version": 1,
            "event": "submission-reconciled",
            "identity": identity,
            "outcome": "unique-job" if matches else "proven-no-job",
            "job_id": next(iter(matches), ""),
            "descriptor": descriptor,
            "query_evidence_sha256": query_evidence_sha256,
        })
    payload = json.dumps(
        {
            "schema_version": 1,
            "kind": "UnknownSubmissionReconciliationV1",
            "journal_sha256": base_journal_sha256,
            "query_evidence_sha256": query_evidence_sha256,
            "queries": queries,
            "results": results,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    fd = os.open(artifact, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    parent_fd = os.open(artifact.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)

journal_fd = os.open(journal, os.O_WRONLY | os.O_APPEND)
try:
    for result in results:
        identity = result.get("identity")
        if identity in reconciled:
            continue
        os.write(
            journal_fd,
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
            + b"\n",
        )
    os.fsync(journal_fd)
finally:
    os.close(journal_fd)
PY
  [ ! -d "$phase_lock" ] || rmdir "$phase_lock"
  transaction_committed=1
  exit 0
fi
if [ "$WORLDMM_DAG_PHASE" = phase-a ]; then
  case "$WORLDMM_ORACLE_SUBMISSION_SURFACE" in
    preflight)
      [ ! -e "$root" ] || {
        echo "stale Phase A output namespace" >&2; exit 1; }
      mkdir -p "$root/summary" "$root/logs" "$root/diagnostics"
      ;;
    provider)
      [ -d "$root/summary" ] && \
        [ -f "$root/summary/dag_jobs.preflight.env" ] && \
        [ ! -e "$root/summary/dag_jobs.provider.env" ] || {
        echo "invalid provider-gate output state" >&2; exit 1; }
      ;;
    *) echo "invalid teacher-oracle submission surface" >&2; exit 2 ;;
  esac
else
  [ -d "$root/summary" ] && [ ! -e "$root/summary/dag_jobs.env" ] || {
    echo "stale downstream output state" >&2; exit 1; }
fi
mkdir "$phase_lock" 2>/dev/null || {
  echo "phase submission is already locked" >&2; exit 1; }
printf '{"phase":"%s","event":"started"}\n' "$WORLDMM_DAG_PHASE" > "$attempt_journal"
stage_script="$(dirname "$0")/run_teacher_oracle_stage.sh"
WORLDMM_EXPERIMENT_GRAPH_FILE="${WORLDMM_EXPERIMENT_GRAPH_FILE:-$(dirname "$0")/experiment_graph.json}"
export WORLDMM_EXPERIMENT_GRAPH_FILE
SBATCH="${WORLDMM_SBATCH:-sbatch}"
SCONTROL="${WORLDMM_SCONTROL:-scontrol}"
verify() {
  WORLDMM_APPROVAL_PATH="$1" WORLDMM_APPROVAL_PHASE="$2" \
    "$stage_script" --verify-approval
}
prevalidate_submission() {
  local sacct="${WORLDMM_SACCT:-sacct}"
  EXPECTED_SUBMIT_DIGEST="$expected_submit_digest" \
  "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - \
    "$WORLDMM_REMOTE_REPO" "$0" "$stage_script" \
    "$WORLDMM_EXPERIMENT_GRAPH_FILE" "$WORLDMM_ORACLE_PROVIDER_EXECUTABLE" \
    "$WORLDMM_ORACLE_STAGE_EXECUTABLE" "$sacct" <<'PY'
import hashlib, json, os, re, stat, subprocess, sys
from pathlib import Path
from worldmm_smvqa.slurm_accounting import preflight_capability
root, submitter, stage_script, resources, provider, stage_executable, sacct = map(Path, sys.argv[1:8])
def secure_digest(path, expected):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        state = os.fstat(descriptor)
        if not stat.S_ISREG(state.st_mode) or state.st_mode & 0o022:
            raise SystemExit(f"unsafe approved input: {path}")
        digest = hashlib.file_digest(os.fdopen(os.dup(descriptor), "rb"), "sha256").hexdigest()
    finally:
        os.close(descriptor)
    if digest != expected:
        raise SystemExit(f"approved digest mismatch: {path}")
def tree_digest(directory):
    ignored = {
        Path(os.environ["WORLDMM_OUTPUT_ROOT"]).resolve(),
        Path(os.environ["WORLDMM_APPROVAL_FILE"]).resolve(),
        Path(os.environ.get("WORLDMM_PHASE_B_APPROVAL_FILE", "/dev/null")).resolve(),
    }
    rows = []
    for path in sorted(directory.rglob("*")):
        resolved = path.resolve()
        if any(resolved == ignored_path or ignored_path in resolved.parents for ignored_path in ignored):
            continue
        if path.is_symlink():
            raise SystemExit("snapshot contains a symlink")
        if path.is_file():
            rows.append(f"{path.relative_to(directory)}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n")
    return hashlib.sha256("".join(rows).encode()).hexdigest()
secure_digest(submitter, os.environ["EXPECTED_SUBMIT_DIGEST"])
secure_digest(stage_script, os.environ["WORLDMM_DAG_STAGE_SCRIPT_SHA256"])
secure_digest(resources, os.environ["WORLDMM_RESOURCE_CONFIG_SHA256"])
secure_digest(provider, os.environ["WORLDMM_ORACLE_PROVIDER_EXECUTABLE_SHA256"])
secure_digest(stage_executable, os.environ["WORLDMM_ORACLE_STAGE_EXECUTABLE_SHA256"])
if tree_digest(root) != os.environ["WORLDMM_REMOTE_SNAPSHOT_SHA256"]:
    raise SystemExit("remote snapshot digest mismatch")
PY
  "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - \
    "$WORLDMM_EXPERIMENT_GRAPH_FILE" "$sacct" <<'PY'
import json, re, subprocess, sys
from pathlib import Path
from worldmm_smvqa.slurm_accounting import preflight_capability

graph, sacct = sys.argv[1:]
value = json.loads(Path(graph).read_text(encoding="utf-8"))
stages = value.get("stage_specs")
if value.get("experiment_id") != "EXP-0005" or not isinstance(stages, list):
    raise SystemExit("invalid frozen experiment graph")
expected = (
    "preflight", "geometry", "semantic", "place", "gate", "terminal",
    "e0_materialize", "e0_retrieve", "e0_qa", "t0_materialize",
    "t0_retrieve", "t0_qa", "t1_materialize", "t1_retrieve", "t1_qa",
    "evaluator", "finalizer",
)
if tuple(stage.get("name") for stage in stages if isinstance(stage, dict)) != expected:
    raise SystemExit("experiment graph stage topology is not canonical")
for stage in stages:
    entry = stage.get("resources") if isinstance(stage, dict) else None
    if not isinstance(entry, dict) or set(entry) != {
        "kind", "partition", "nodes", "cpus", "memory", "time", "gpus_per_node"
    }:
        raise SystemExit(f"invalid graph resource schema for {stage}")
    memory, time = entry["memory"], entry["time"]
    match = isinstance(memory, str) and re.fullmatch(r"([1-9][0-9]*)([MG])", memory)
    duration_seconds = sum(value * multiplier for value, multiplier in zip(map(int, time.split(":")), (3600, 60, 1), strict=True)) if isinstance(time, str) and re.fullmatch(r"[0-9]{2}:[0-5][0-9]:[0-5][0-9]", time) else -1
    if (not match or int(match.group(1)) * (1024 if match.group(2) == "G" else 1) > 2 * 1024 * 1024
        or entry["partition"] not in {"cpu-prepro-queue", "gpu-vtt-queue"}
        or not isinstance(entry["nodes"], int) or not 1 <= entry["nodes"] <= 10
        or not isinstance(entry["cpus"], int) or not 1 <= entry["cpus"] <= 256
        or not isinstance(entry["gpus_per_node"], int) or not 0 <= entry["gpus_per_node"] <= 8
        or entry["gpus_per_node"] * entry["nodes"] > 80
        or not 0 <= duration_seconds <= 48 * 3600
        or (entry["gpus_per_node"] == 0) != (entry["partition"] == "cpu-prepro-queue")):
        raise SystemExit("experiment graph resource exceeds company policy")
try:
    preflight_capability(
        version_output=subprocess.check_output([sacct, "--version"], text=True),
        helpformat_output=subprocess.check_output([sacct, "--helpformat"], text=True),
    )
except Exception as exc:
    raise SystemExit(f"sacct capability preflight failed: {exc}") from exc
PY
}
submit() {
  local stage=$1 dependency=${2:-} dependency_kind=${3:-afterok} job
  local stage_exports resource_file resource_line partition nodes cpus
  local memory walltime gpus
  resource_file="${WORLDMM_EXPERIMENT_GRAPH_FILE:?frozen experiment graph required}"
  [ -s "$resource_file" ] && [ ! -L "$resource_file" ] || {
    echo "frozen experiment graph is required" >&2; return 1; }
  [ "$(sha256sum "$resource_file" | cut -d ' ' -f 1)" = \
    "$WORLDMM_RESOURCE_CONFIG_SHA256" ] || {
    echo "experiment graph digest mismatch" >&2; return 1; }
  resource_line="$(
    "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - "$resource_file" "$stage" <<'PY'
import json, re, sys
from pathlib import Path

graph, runtime_stage = sys.argv[1:]
name = {
    "teacher_oracle_preflight": "preflight",
    "teacher_oracle_geometry": "geometry",
    "teacher_oracle_semantic": "semantic",
    "teacher_oracle_place": "place",
    "teacher_oracle_gate": "gate",
    "teacher_oracle_finalizer": "terminal",
    "teacher_oracle_evaluator": "evaluator",
    "teacher_oracle_finalizer_phase_b": "finalizer",
}.get(runtime_stage, runtime_stage.removeprefix("teacher_oracle_").replace("E0_", "e0_").replace("T0_", "t0_").replace("T1_", "t1_"))
value = json.loads(Path(graph).read_text(encoding="utf-8"))
entry = next((item.get("resources") for item in value.get("stage_specs", [])
              if isinstance(item, dict) and item.get("name") == name), None)
fields = ("partition", "nodes", "cpus", "memory", "time", "gpus_per_node")
if not isinstance(entry, dict) or set(entry) != {"kind", *fields}:
    raise SystemExit(f"missing graph resources for {runtime_stage}")
if not isinstance(entry["memory"], str) or not re.fullmatch(r"[1-9][0-9]*[MG]", entry["memory"]):
    raise SystemExit("invalid graph memory")
if not isinstance(entry["time"], str) or not re.fullmatch(r"[0-9]{2}:[0-5][0-9]:[0-5][0-9]", entry["time"]):
    raise SystemExit("invalid graph time")
print("\t".join(str(entry[name]) for name in fields))
PY
  )" || return 1
  IFS=$'\t' read -r partition nodes cpus memory walltime gpus <<<"$resource_line"
  stage_exports="WORLDMM_STAGE=$stage,WORLDMM_EXECUTION_PROFILE=teacher-oracle"
  for name in WORLDMM_REMOTE_REPO WORLDMM_OUTPUT_ROOT WORLDMM_RUN_ID \
    WORLDMM_EXPERIMENT_ID WORLDMM_APPROVAL_FILE WORLDMM_PHASE_B_APPROVAL_FILE \
    WORLDMM_CONTINUE_RECEIPT WORLDMM_SIGNER_REGISTRY \
    WORLDMM_SIGNER_REGISTRY_SHA256 WORLDMM_SENSOR_AUDIT_SHA256 \
    WORLDMM_PROVIDER_SHA256 WORLDMM_SPLIT_SHA256 WORLDMM_CODE_SHA \
    WORLDMM_POLICY_SHA256 \
    WORLDMM_FRAME_ASSETS_SHA256 WORLDMM_BYTE_BUDGET_SHA256 \
    WORLDMM_RESOURCE_CONFIG_SHA256 WORLDMM_PLAN_SHA256 \
    WORLDMM_REMOTE_SNAPSHOT_SHA256 WORLDMM_DAG_PREFLIGHT_SUBMIT_SCRIPT_SHA256 \
    WORLDMM_DAG_PROVIDER_GATE_SUBMIT_SCRIPT_SHA256 \
    WORLDMM_DAG_DOWNSTREAM_SUBMIT_SCRIPT_SHA256 \
    WORLDMM_DAG_STAGE_SCRIPT_SHA256 WORLDMM_ORACLE_PROVIDER_EXECUTABLE_SHA256 \
    WORLDMM_ORACLE_STAGE_EXECUTABLE_SHA256 WORLDMM_ATTESTED_RUNTIME_ROOT \
    WORLDMM_ATTESTED_RUNTIME_MANIFEST WORLDMM_ATTESTED_RUNTIME_MANIFEST_SHA256 \
    WORLDMM_TEACHER_ORACLE_VALIDATION_RECEIPT \
    WORLDMM_TEACHER_ORACLE_VALIDATION_SHA256 WORLDMM_EXPERIMENT_CONFIG_SHA256 \
    WORLDMM_ORACLE_PROVIDER_EXECUTABLE WORLDMM_ORACLE_PROVIDER_CONFIG \
    WORLDMM_ORACLE_PROVIDER_CONFIG_SHA256 WORLDMM_ORACLE_STAGE_EXECUTABLE \
    WORLDMM_EXPERIMENT_GRAPH_FILE WORLDMM_SACCT \
    WORLDMM_SLURM_CLUSTER WORLDMM_ACCOUNTING_SETTLE_SECONDS \
    WORLDMM_ACCOUNTING_SETTLE_INTERVAL_SECONDS \
    WORLDMM_ORACLE_QUALITY_EVALUATOR WORLDMM_ORACLE_QUALITY_EVALUATOR_SHA256 \
    WORLDMM_ORACLE_QUALITY_CONTRACT WORLDMM_ORACLE_QUALITY_CONTRACT_SHA256 \
    WORLDMM_QA_SHARD_MAP WORLDMM_QA_LINEAGE WORLDMM_QA_FINALIZATION_RECEIPT \
    WORLDMM_QA_PREDICTIONS WORLDMM_QA_SHARD_MAP_SHA256 WORLDMM_QA_LINEAGE_SHA256 \
    WORLDMM_QA_FINALIZATION_RECEIPT_SHA256 WORLDMM_QA_PREDICTIONS_SHA256; do
    [ -n "${!name:-}" ] || continue
    case "${!name}" in
      *[,[:cntrl:]]*)
        echo "unsafe sbatch export: $name" >&2
        return 1
        ;;
    esac
    stage_exports+=",$name=${!name}"
  done
  if [ "$stage" = teacher_oracle_gate ]; then
    for name in WORLDMM_CONTINUE_RECEIPT_KEY_ID \
      WORLDMM_CONTINUE_RECEIPT_SIGNING_KEY; do
      [ -n "${!name:-}" ] || {
        echo "missing gate receipt signing input: $name" >&2
        return 1
      }
      case "${!name}" in
        *[,[:cntrl:]]*)
          echo "unsafe gate receipt signing input: $name" >&2
          return 1
          ;;
      esac
      stage_exports+=",$name=${!name}"
    done
  fi
  local args=(
    "$SBATCH" --parsable --no-requeue --hold
    "--job-name=worldmm-${WORLDMM_RUN_ID}-${stage}"
    "--comment=${WORLDMM_RUN_ID}:${WORLDMM_DAG_PHASE}:${stage}"
    "--output=$root/logs/${stage}-%j.out"
    "--error=$root/logs/${stage}-%j.err"
    "--partition=$partition" "--nodes=$nodes"
    "--cpus-per-task=$cpus" "--mem=$memory" "--time=$walltime"
    "--export=$stage_exports"
  )
  [ "$gpus" = 0 ] || args+=("--gpus-per-node=$gpus")
  [ -z "$dependency" ] || args+=("--dependency=${dependency_kind}:$dependency")
  # Persist uncertainty before invoking sbatch.  A shell interruption or a
  # nonzero return can otherwise leave a real held job with no reconcilable ID.
  unknown_submission=1
  printf '{"run_id":"%s","stage":"%s","identity":"%s:%s:%s","event":"submission-unknown-before-sbatch"}\n' \
    "$WORLDMM_RUN_ID" "$stage" "$WORLDMM_RUN_ID" "$WORLDMM_DAG_PHASE" "$stage" >> "$attempt_journal"
  job="$("${args[@]}" "$stage_script")"
  job="${job%%;*}"
  if [[ ! "$job" =~ ^[1-9][0-9]*$ ]]; then
    printf '{"stage":"%s","event":"unparseable-job-id"}\n' \
      "$stage" >> "$attempt_journal"
    echo "untrustworthy sbatch job id" >&2
    return 1
  fi
  submitted_job_ids+=("$job")
  printf '{"stage":"%s","identity":"%s:%s:%s","event":"submission-reconciled","job_id":"%s"}\n' \
    "$stage" "$WORLDMM_RUN_ID" "$WORLDMM_DAG_PHASE" "$stage" "$job" >> "$attempt_journal"
  unknown_submission=0
  submitted_job_id="$job"
  if [[ "$stage" =~ ^teacher_oracle_(geometry|semantic|place)$ ]]; then
    WORLDMM_EXPECTATION_STAGE="$stage" WORLDMM_EXPECTATION_JOB_ID="$job" \
      "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - "$root/summary" <<'PY'
import hashlib, json, os
from pathlib import Path

summary = Path(os.sys.argv[1])
payload = {
    "schema_version": 1,
    "kind": "ProviderAttemptExpectationV1",
    "stage": os.environ["WORLDMM_EXPECTATION_STAGE"],
    "job_id": os.environ["WORLDMM_EXPECTATION_JOB_ID"],
    "attempt": os.environ["WORLDMM_EXPECTATION_JOB_ID"],
    "input_sha256": os.environ["WORLDMM_PLAN_SHA256"],
    "resource_sha256": os.environ["WORLDMM_RESOURCE_CONFIG_SHA256"],
    "code_sha256": os.environ["WORLDMM_CODE_SHA"],
    "approval_sha256": hashlib.sha256(Path(os.environ["WORLDMM_APPROVAL_FILE"]).read_bytes()).hexdigest(),
}
target = summary / f"{payload['stage']}.expectation.json"
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    os.write(fd, encoded)
    os.fsync(fd)
finally:
    os.close(fd)
PY
  fi
}
if [ "$WORLDMM_DAG_PHASE" = phase-a ]; then
  if [ "$WORLDMM_ORACLE_SUBMISSION_SURFACE" = provider ]; then
    verify "$WORLDMM_APPROVAL_FILE" phase_a
  fi
  prevalidate_submission
  if [ "$WORLDMM_ORACLE_SUBMISSION_SURFACE" = preflight ]; then
    submit teacher_oracle_preflight || exit 1
    preflight="$submitted_job_id"
    preflight_jobs="$root/summary/dag_jobs.preflight.env"
    preflight_temporary="$root/summary/.dag_jobs.preflight.env.${BASHPID}.tmp"
    printf '%s\n' "PREFLIGHT_JOB_ID=$preflight" > "$preflight_temporary"
    mv -f "$preflight_temporary" "$preflight_jobs"
    "$SCONTROL" release "$preflight" || {
      unknown_submission=1
      printf '{"phase":"phase-a","event":"release-failed","job_id":"%s"}\n' "$preflight" >> "$attempt_journal"
      exit 1
    }
    transaction_committed=1
    printf '{"phase":"phase-a","surface":"preflight","event":"submitted-and-released"}\n' >> "$attempt_journal"
    exit 0
  fi
  [ "$WORLDMM_ORACLE_SUBMISSION_SURFACE" = provider ] || exit 2
  preflight="$(
    "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - \
      "$root/summary/dag_jobs.preflight.env" \
      "$root/diagnostics/teacher_oracle_preflight.seal" \
      "$WORLDMM_PREFLIGHT_SEAL_SHA256" "$WORLDMM_SLURM_CLUSTER" \
      "${WORLDMM_SACCT:-sacct}" "$WORLDMM_ACCOUNTING_SETTLE_SECONDS" \
      "$WORLDMM_ACCOUNTING_SETTLE_INTERVAL_SECONDS" <<'PY'
import hashlib, json, subprocess, sys, time
from pathlib import Path
from worldmm_smvqa.slurm_accounting import (
    decode_accounting, is_nonterminal, require_success, sacct_command,
)

manifest, seal = map(Path, sys.argv[1:3])
expected, cluster, sacct, seconds, interval = sys.argv[3:]
value = manifest.read_text(encoding="utf-8").strip()
import re
match = re.fullmatch(r"PREFLIGHT_JOB_ID=([1-9][0-9]*)", value)
if match is None:
    raise SystemExit("invalid preflight job manifest")
raw = seal.read_bytes()
if hashlib.sha256(raw).hexdigest() != expected:
    raise SystemExit("preflight seal digest is not approval-bound")
payload = json.loads(raw)
job_id = match.group(1)
if payload.get("kind") != "PreflightSealV1" or payload.get("job_id") != job_id:
    raise SystemExit("preflight seal allocation identity mismatch")
if set(payload) != {
    "schema_version", "kind", "job_id", "measured_audit",
    "measured_validation", "selected_sensor_inventory_digest",
    "resource_config_sha256", "remote_snapshot_sha256",
} or payload.get("schema_version") != 1:
    raise SystemExit("preflight seal schema is not closed")
for name in ("measured_audit", "measured_validation"):
    descriptor = payload.get(name)
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "sha256", "uid", "mode", "nlink", "device", "inode", "size", "mtime_ns"
    } or not re.fullmatch(r"[0-9a-f]{64}", str(descriptor.get("sha256"))):
        raise SystemExit("preflight seal measured descriptor is invalid")
if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("selected_sensor_inventory_digest"))):
    raise SystemExit("preflight seal inventory binding is invalid")
try:
    deadline = time.monotonic() + int(seconds)
    poll = float(interval)
except ValueError as exc:
    raise SystemExit("invalid approval-bound preflight settle policy") from exc
if poll <= 0 or deadline <= time.monotonic():
    raise SystemExit("invalid approval-bound preflight settle policy")
while True:
    try:
        record = decode_accounting(
            subprocess.check_output(
                sacct_command(sacct=sacct, cluster=cluster, job_id=job_id), text=True
            ),
            cluster=cluster,
            job_id=job_id,
        )
        if is_nonterminal(record):
            raise RuntimeError("preflight allocation is still nonterminal")
        require_success(record)
        break
    except RuntimeError:
        pass
    except ValueError as exc:
        # A missing row can legitimately lag Slurm completion; malformed or terminal
        # records are permanent failures and must not admit producers.
        if "expected exactly one allocation row, got 0" not in str(exc):
            raise SystemExit(f"preflight accounting rejected: {exc}") from exc
    except (OSError, subprocess.CalledProcessError):
        pass
    if time.monotonic() >= deadline:
        raise SystemExit("preflight accounting did not settle before approved deadline")
    time.sleep(min(poll, max(0.0, deadline - time.monotonic())))
accounting = {
    "schema_version": 1, "kind": "PreflightAccountingV1", "job_id": job_id,
    "cluster": record.cluster, "sluid": record.sluid,
    "original_sluid": record.original_sluid, "state": record.state,
    "exit_code": record.exit_code, "restarts": record.restarts,
    "preflight_seal_sha256": expected,
}
target = seal.with_name("teacher_oracle_preflight.accounting.json")
temporary = target.with_name(f".{target.name}.{__import__('os').getpid()}.tmp")
temporary.write_bytes(json.dumps(accounting, sort_keys=True, separators=(",", ":")).encode())
temporary.chmod(0o600)
temporary.replace(target)
print(job_id)
PY
  )"
  phase_a_jobs="$root/summary/dag_jobs.provider.env"
  phase_a_manifest=()
  producer_ids=(geometry semantic place)
  producer_jobs=()
  for producer in "${producer_ids[@]}"; do
    submit "teacher_oracle_${producer}" "$preflight" || exit 1
    job="$submitted_job_id"
    producer_jobs+=("$job")
    phase_a_manifest+=("PREFLIGHT_JOB_ID=$preflight" "PROVIDER_${producer^^}_JOB_ID=$job")
    phase_a_manifest+=("PROVIDER_${producer^^}_EXPECTATION_SHA256=$(sha256sum "$root/summary/teacher_oracle_${producer}.expectation.json" | cut -d ' ' -f 1)")
  done
  dependency="$(IFS=:; printf '%s' "${producer_jobs[*]}")"
  submit teacher_oracle_gate "$dependency" afterany || exit 1
  gate="$submitted_job_id"
  submit teacher_oracle_finalizer "$gate" afterany || exit 1
  finalizer="$submitted_job_id"
  phase_a_manifest+=(
    "PROVIDER_GATE_JOB_ID=$gate"
    "PROVIDER_GATE_TERMINAL_JOB_ID=$finalizer"
  )
  phase_a_temporary="$root/summary/.dag_jobs.provider.env.${BASHPID}.tmp"
  printf '%s\n' "${phase_a_manifest[@]}" > "$phase_a_temporary"
  mv -f "$phase_a_temporary" "$phase_a_jobs"
  for job in "${submitted_job_ids[@]}"; do
    "$SCONTROL" release "$job" || {
      unknown_submission=1
      printf '{"phase":"phase-a","event":"release-failed","job_id":"%s"}\n' "$job" >> "$attempt_journal"
      exit 1
    }
  done
  transaction_committed=1
  printf '{"phase":"phase-a","event":"submitted-and-released"}\n' >> "$attempt_journal"
  exit 0
fi
[ "$WORLDMM_DAG_PHASE" = phase-b ] || exit 2
: "${WORLDMM_PHASE_B_APPROVAL_FILE:?second approval required}"
receipt="$root/summary/teacher_oracle_continue.json"
terminal="$root/summary/teacher_oracle_terminal.json"
[ -s "$receipt" ] && [ -s "$terminal" ] || {
  echo "sealed continue receipt and terminal are required" >&2; exit 1; }
# Claim first. The hard link is the consumed immutable inode; do not unlink a
# pathname after a same-owner replacement unless it still names that inode.
used_receipt="$root/summary/.teacher_oracle_continue.used.json"
used_terminal="$root/summary/.teacher_oracle_terminal.used.json"
"$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - \
  "$receipt" "$terminal" "$used_receipt" "$used_terminal" <<'PY'
import os
import stat
import sys
from pathlib import Path

receipt, terminal, used_receipt, used_terminal = map(Path, sys.argv[1:])
def claim(source, destination, *, consume):
    if destination.exists() or destination.is_symlink():
        raise SystemExit("continuation was already consumed")
    source_state = os.lstat(source)
    if not stat.S_ISREG(source_state.st_mode):
        raise SystemExit("continuation authority is not a regular file")
    os.link(source, destination, follow_symlinks=False)
    claimed = os.lstat(destination)
    if (claimed.st_dev, claimed.st_ino) != (source_state.st_dev, source_state.st_ino):
        raise SystemExit("continuation claim inode mismatch")
    if consume:
        current = os.lstat(source)
        if (current.st_dev, current.st_ino) == (claimed.st_dev, claimed.st_ino):
            os.unlink(source)
claim(receipt, used_receipt, consume=True)
try:
    claim(terminal, used_terminal, consume=False)
except Exception:
    used_receipt.unlink(missing_ok=True)
    raise
PY
"$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - "$root/summary" <<'PY'
import hashlib, os, stat, sys
from pathlib import Path

for source_name, digest_name, target_name in (
    ("WORLDMM_QA_SHARD_MAP", "WORLDMM_QA_SHARD_MAP_SHA256", "qa_shard_map"),
    ("WORLDMM_QA_LINEAGE", "WORLDMM_QA_LINEAGE_SHA256", "qa_lineage"),
    ("WORLDMM_QA_FINALIZATION_RECEIPT", "WORLDMM_QA_FINALIZATION_RECEIPT_SHA256", "qa_finalization_receipt"),
    ("WORLDMM_QA_PREDICTIONS", "WORLDMM_QA_PREDICTIONS_SHA256", "qa_predictions"),
):
    source, target = Path(os.environ[source_name]), Path(sys.argv[1]) / f".{target_name}.used"
    if target.exists() or target.is_symlink():
        raise SystemExit(f"QA authority already consumed: {target_name}")
    fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        state = os.fstat(fd)
        if not stat.S_ISREG(state.st_mode) or state.st_uid != os.getuid() or state.st_nlink != 1:
            raise SystemExit(f"unsafe QA authority: {target_name}")
        raw = os.read(fd, state.st_size + 1)
        if len(raw) != state.st_size or os.fstat(fd).st_size != state.st_size:
            raise SystemExit(f"QA authority changed while claimed: {target_name}")
        if hashlib.sha256(raw).hexdigest() != os.environ[digest_name]:
            raise SystemExit(f"QA authority digest mismatch: {target_name}")
        os.link(source, target, follow_symlinks=False)
        claimed = os.lstat(target)
        if (claimed.st_dev, claimed.st_ino) != (state.st_dev, state.st_ino):
            raise SystemExit(f"QA authority inode mismatch: {target_name}")
    finally:
        os.close(fd)
PY
export WORLDMM_QA_SHARD_MAP="$root/summary/.qa_shard_map.used"
export WORLDMM_QA_LINEAGE="$root/summary/.qa_lineage.used"
export WORLDMM_QA_FINALIZATION_RECEIPT="$root/summary/.qa_finalization_receipt.used"
export WORLDMM_QA_PREDICTIONS="$root/summary/.qa_predictions.used"
WORLDMM_CONTINUE_RECEIPT="$used_receipt" \
  WORLDMM_CONTINUE_TERMINAL="$used_terminal" \
  verify "$WORLDMM_PHASE_B_APPROVAL_FILE" phase_b
prevalidate_submission
# verify_approval consumed and verified the claimed terminal descriptor.
export WORLDMM_CONTINUE_RECEIPT="$used_receipt"
phase_b_jobs="$root/summary/dag_jobs.env"
phase_b_manifest=()
qa_jobs=()
for variant in E0 T0 T1; do
  submit "teacher_oracle_${variant}_materialize" || exit 1
  materialize="$submitted_job_id"
  submit "teacher_oracle_${variant}_retrieve" "$materialize" || exit 1
  retrieve="$submitted_job_id"
  submit "teacher_oracle_${variant}_qa" "$retrieve" || exit 1
  qa="$submitted_job_id"
  qa_jobs+=("$qa")
  phase_b_manifest+=(
    "MATERIALIZE_${variant}_JOB_ID=$materialize"
    "RETRIEVE_${variant}_JOB_ID=$retrieve"
    "QA_${variant}_JOB_ID=$qa"
  )
done
dependency="$(IFS=:; printf '%s' "${qa_jobs[*]}")"
submit teacher_oracle_evaluator "$dependency" || exit 1
evaluator="$submitted_job_id"
submit teacher_oracle_finalizer_phase_b "$evaluator" || exit 1
finalizer="$submitted_job_id"
phase_b_manifest+=(
  "EVALUATE_JOB_ID=$evaluator"
  "FINALIZE_JOB_ID=$finalizer"
)
phase_b_temporary="$root/summary/.dag_jobs.env.${BASHPID}.tmp"
printf '%s\n' "${phase_b_manifest[@]}" > "$phase_b_temporary"
mv -f "$phase_b_temporary" "$phase_b_jobs"
for job in "${submitted_job_ids[@]}"; do
  "$SCONTROL" release "$job" || {
    unknown_submission=1
    printf '{"phase":"phase-b","event":"release-failed","job_id":"%s"}\n' "$job" >> "$attempt_journal"
    exit 1
  }
done
transaction_committed=1
printf '{"phase":"phase-b","event":"submitted-and-released"}\n' >> "$attempt_journal"
"""
    )


def teacher_oracle_preflight_submit_script_text(graph: object | None = None) -> str:
    """Render the phase-fixed preflight submitter from the validated graph."""
    return _teacher_oracle_submit_script_text(graph)


def teacher_oracle_provider_gate_submit_script_text(graph: object | None = None) -> str:
    """Render the phase-fixed provider/gate submitter from the validated graph."""
    return _teacher_oracle_submit_script_text(graph)


def teacher_oracle_downstream_submit_script_text(graph: object | None = None) -> str:
    """Render the continuation-gated downstream submitter from the validated graph."""
    return _teacher_oracle_submit_script_text(graph)


def teacher_oracle_stage_script_text(graph: object | None = None) -> str:
    """Render the label- and student-blind teacher-oracle stage runner from graph."""
    return teacher_oracle_dag_stage_script_text(graph)


def teacher_oracle_dag_stage_script_text(graph: object | None = None) -> str:
    """Render the label- and student-blind teacher-oracle stage runner."""
    graph_json = getattr(graph, "model_dump_json", lambda: "")()
    graph_sha256 = hashlib.sha256(graph_json.encode("utf-8")).hexdigest()
    return (
        f"#!/usr/bin/env bash\n# graph_sha256={graph_sha256}\n"
        r"""set -euo pipefail
umask 077
verify_approval() {
  "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - \
    "$WORLDMM_APPROVAL_PATH" "$WORLDMM_SIGNER_REGISTRY" \
    "${WORLDMM_APPROVAL_PHASE:?}" "${WORLDMM_CONTINUE_RECEIPT:-}" <<'PY'
import hashlib
import json
import os
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from worldmm_smvqa.attestation import (
    AttestationError, b64url_decode, loads_strict, require_payload_sha256,
    signing_bytes,
)
approval_path = Path(sys.argv[1])
registry_path = Path(sys.argv[2])
phase = sys.argv[3]
receipt_path = Path(sys.argv[4])

def read_secure(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        state = os.fstat(descriptor)
        if (
            not stat.S_ISREG(state.st_mode) or state.st_uid != os.getuid()
            or state.st_mode & 0o077
        ):
            raise SystemExit("unsafe signed input")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            raw = stream.read()
        end = os.fstat(descriptor)
        if (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns) != (
            end.st_dev, end.st_ino, end.st_size, end.st_mtime_ns
        ):
            raise SystemExit("signed input changed while read")
        return raw
    finally:
        os.close(descriptor)
approval_raw, registry_raw = (
    read_secure(approval_path),
    read_secure(registry_path),
)
registry_sha256 = hashlib.sha256(registry_raw).hexdigest()
if registry_sha256 != os.environ["WORLDMM_SIGNER_REGISTRY_SHA256"]:
    raise SystemExit("signer registry trust-root digest mismatch")
approval, registry = loads_strict(approval_raw), loads_strict(registry_raw)
keys = registry.get("keys")
if (
    registry.get("schema_version") != 1
    or not isinstance(keys, list)
    or any(not isinstance(entry, dict) or not isinstance(entry.get("key_id"), str)
           for entry in keys)
    or len({entry["key_id"] for entry in keys}) != len(keys)
):
    raise SystemExit("invalid or ambiguous signer registry")
require_payload_sha256(approval)
require_payload_sha256(registry)
if approval.get("kind") != "SignedAttestationEnvelopeV1":
    raise SystemExit("invalid signed attestation envelope")
signature = approval.pop("signature", None)
key_id = approval.get("key_id")
key = next((entry for entry in keys if entry["key_id"] == key_id), None)
valid_key = (
    isinstance(key, dict)
    and not key.get("revoked")
    and f"{phase}_approval" in key.get("purposes", [])
)
if not valid_key:
    raise SystemExit("unapproved signing key")
try:
    start = datetime.fromisoformat(key["valid_from"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(key["valid_until"].replace("Z", "+00:00"))
    public = b64url_decode(key["public_key"])
    signed = b64url_decode(signature)
    Ed25519PublicKey.from_public_bytes(public).verify(
        signed,
        signing_bytes(approval, f"{phase}-approval"),
    )
except Exception as exc:
    raise SystemExit("approval signature verification failed") from exc
if not start <= datetime.now(UTC) <= end:
    raise SystemExit("signer validity failure")
expected = {
    "schema_version": 1,
    "experiment_id": os.environ["WORLDMM_EXPERIMENT_ID"],
    "profile": "teacher-oracle",
    "phase": phase,
    "preflight_seal_sha256": os.environ.get("WORLDMM_PREFLIGHT_SEAL_SHA256", ""),
    "quality_contract_sha256": os.environ.get(
        "WORLDMM_ORACLE_QUALITY_CONTRACT_SHA256", ""
    ),
    "quality_evaluator_sha256": os.environ.get(
        "WORLDMM_ORACLE_QUALITY_EVALUATOR_SHA256", ""
    ),
    "sensor_audit_sha256": os.environ["WORLDMM_SENSOR_AUDIT_SHA256"],
    "provider_sha256": os.environ["WORLDMM_PROVIDER_SHA256"],
    "split_sha256": os.environ["WORLDMM_SPLIT_SHA256"],
    "code_sha256": os.environ["WORLDMM_CODE_SHA"],
    "policy_sha256": os.environ["WORLDMM_POLICY_SHA256"],
    "validation_receipt_sha256": os.environ[
        "WORLDMM_TEACHER_ORACLE_VALIDATION_SHA256"
    ],
    "experiment_config_sha256": os.environ[
        "WORLDMM_EXPERIMENT_CONFIG_SHA256"
    ],
    "run_id": os.environ["WORLDMM_RUN_ID"],
    "output_root": os.environ["WORLDMM_OUTPUT_ROOT"],
    "frame_assets_sha256": os.environ["WORLDMM_FRAME_ASSETS_SHA256"],
    "byte_budget_sha256": os.environ["WORLDMM_BYTE_BUDGET_SHA256"],
    "resource_config_sha256": os.environ["WORLDMM_RESOURCE_CONFIG_SHA256"],
    "plan_sha256": os.environ["WORLDMM_PLAN_SHA256"],
    "remote_snapshot_sha256": os.environ["WORLDMM_REMOTE_SNAPSHOT_SHA256"],
    "dag_preflight_submit_script_sha256": os.environ[
        "WORLDMM_DAG_PREFLIGHT_SUBMIT_SCRIPT_SHA256"
    ],
    "dag_provider_gate_submit_script_sha256": os.environ[
        "WORLDMM_DAG_PROVIDER_GATE_SUBMIT_SCRIPT_SHA256"
    ],
    "dag_downstream_submit_script_sha256": os.environ[
        "WORLDMM_DAG_DOWNSTREAM_SUBMIT_SCRIPT_SHA256"
    ],
    "dag_stage_script_sha256": os.environ["WORLDMM_DAG_STAGE_SCRIPT_SHA256"],
    "oracle_provider_executable_sha256": os.environ[
        "WORLDMM_ORACLE_PROVIDER_EXECUTABLE_SHA256"
    ],
    "oracle_stage_executable_sha256": os.environ[
        "WORLDMM_ORACLE_STAGE_EXECUTABLE_SHA256"
    ],
    "registry_sha256": registry_sha256,
    "attested_runtime_root": os.environ["WORLDMM_ATTESTED_RUNTIME_ROOT"],
    "attested_runtime_manifest_sha256": os.environ["WORLDMM_ATTESTED_RUNTIME_MANIFEST_SHA256"],
    "purpose": f"teacher_oracle_{phase}_execution",
    "slurm_cluster": os.environ["WORLDMM_SLURM_CLUSTER"],
    "accounting_settle_seconds": os.environ.get("WORLDMM_ACCOUNTING_SETTLE_SECONDS", ""),
    "accounting_settle_interval_seconds": os.environ.get(
        "WORLDMM_ACCOUNTING_SETTLE_INTERVAL_SECONDS", ""
    ),
    "producer_tuple": ["geometry", "semantic", "place"],
    "producer_stage_tuple": [
        "teacher_oracle_geometry", "teacher_oracle_semantic", "teacher_oracle_place",
    ],
    "oracle_provider_config_sha256": os.environ.get(
        "WORLDMM_ORACLE_PROVIDER_CONFIG_SHA256", ""
    ),
}
if phase == "phase_b":
    expected.update({
        "qa_shard_map_sha256": os.environ["WORLDMM_QA_SHARD_MAP_SHA256"],
        "qa_lineage_sha256": os.environ["WORLDMM_QA_LINEAGE_SHA256"],
        "qa_finalization_receipt_sha256": os.environ[
            "WORLDMM_QA_FINALIZATION_RECEIPT_SHA256"
        ],
        "qa_predictions_sha256": os.environ["WORLDMM_QA_PREDICTIONS_SHA256"],
        "continue_receipt_sha256": approval.get("continue_receipt_sha256"),
        "terminal_sha256": approval.get("terminal_sha256"),
    })
if any(approval.get(name) != value for name, value in expected.items()):
    raise SystemExit("approval binding mismatch")
if set(approval) != set(expected) | {"kind", "key_id", "payload_sha256"}:
    raise SystemExit("approval schema is not closed")
if phase == "phase_b":
    terminal_path = Path(
        os.environ.get(
            "WORLDMM_CONTINUE_TERMINAL",
            str(Path(os.environ["WORLDMM_OUTPUT_ROOT"]) / "summary" / "teacher_oracle_terminal.json"),
        )
    )
    terminal_raw = read_secure(terminal_path)
    terminal = loads_strict(terminal_raw)
    if (
        terminal.get("provider_gate_decision") != "go"
        or approval.get("terminal_sha256")
        != hashlib.sha256(terminal_raw).hexdigest()
    ):
        raise SystemExit("Phase B approval is not bound to terminal go truth")
    raw = read_secure(receipt_path)
    receipt = loads_strict(raw)
    receipt_signature = receipt.pop("signature", None)
    require_payload_sha256(receipt)
    if receipt.get("kind") != "SignedAttestationEnvelopeV1":
        raise SystemExit("invalid continue receipt envelope")
    receipt_key = next(
        (entry for entry in keys if entry["key_id"] == receipt.get("key_id")),
        None,
    )
    required_receipt_bindings = {
        name: value
        for name, value in expected.items()
        if name
        not in {
            "schema_version",
            "profile",
            "phase",
            "qa_shard_map_sha256",
            "qa_lineage_sha256",
            "qa_finalization_receipt_sha256",
            "qa_predictions_sha256",
            "continue_receipt_sha256",
            "terminal_sha256",
        }
    }
    try:
        if (
            not isinstance(receipt_key, dict)
            or receipt_key.get("revoked")
            or "continue_receipt" not in receipt_key.get("purposes", [])
        ):
            raise ValueError("unapproved receipt key")
        receipt_start = datetime.fromisoformat(
            receipt_key["valid_from"].replace("Z", "+00:00")
        )
        receipt_end = datetime.fromisoformat(
            receipt_key["valid_until"].replace("Z", "+00:00")
        )
        if not receipt_start <= datetime.now(UTC) <= receipt_end:
            raise ValueError("receipt signer validity failure")
        Ed25519PublicKey.from_public_bytes(
            b64url_decode(receipt_key["public_key"])
        ).verify(
            b64url_decode(receipt_signature),
            signing_bytes(receipt, "continue-receipt"),
        )
    except Exception as exc:
        raise SystemExit("continue receipt signature verification failed") from exc
    def revalidate_provider_manifest(producer):
        marker_path = Path(os.environ["WORLDMM_OUTPUT_ROOT"]) / "summary" / (
            f"teacher_oracle_{producer}.json"
        )
        marker_raw = read_secure(marker_path)
        marker = loads_strict(marker_raw)
        provider = (
            Path(os.environ["WORLDMM_OUTPUT_ROOT"]) / "oracle" / "providers" / producer
            / f"attempt-{receipt.get('producer_jobs', {}).get(producer, {}).get('job_id', '')}"
        )
        descriptors = marker.get("provider_descriptors")
        if (
            marker.get("kind") != "ProviderAttemptManifestV1"
            or not isinstance(descriptors, dict)
            or marker.get("attempt") != receipt.get("producer_jobs", {}).get(
                producer, {}
            ).get("job_id")
        ):
            raise SystemExit("continuation provider lineage mismatch")
        for relative, expected_descriptor in descriptors.items():
            if not isinstance(relative, str) or not isinstance(expected_descriptor, dict):
                raise SystemExit("continuation provider descriptor is invalid")
            payload_path = provider / relative
            if provider not in payload_path.parents:
                raise SystemExit("continuation provider payload escapes root")
            descriptor = os.open(
                payload_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                state = os.fstat(descriptor)
                observed = {
                    "uid": state.st_uid,
                    "mode": stat.S_IMODE(state.st_mode),
                    "nlink": state.st_nlink,
                    "device": str(state.st_dev),
                    "inode": str(state.st_ino),
                    "size": state.st_size,
                    "mtime_ns": str(state.st_mtime_ns),
                }
                if (
                    not stat.S_ISREG(state.st_mode) or state.st_uid != os.getuid()
                    or state.st_mode & 0o022 or state.st_nlink != 1
                    or any(expected_descriptor.get(name) != value
                           for name, value in observed.items())
                ):
                    raise SystemExit("continuation provider descriptor changed")
                with os.fdopen(os.dup(descriptor), "rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if (
                    expected_descriptor.get("sha256") != digest
                    or (
                        os.fstat(descriptor).st_size,
                        os.fstat(descriptor).st_mtime_ns,
                    ) != (state.st_size, state.st_mtime_ns)
                ):
                    raise SystemExit("continuation provider payload changed")
            finally:
                os.close(descriptor)
        return hashlib.sha256(marker_raw).hexdigest()
    actual_provider_manifests = {
        producer: revalidate_provider_manifest(producer)
        for producer in ("geometry", "semantic", "place")
    }
    valid_receipt = (
        receipt.get("decision") == "go"
        and receipt.get("profile") == "teacher-oracle"
        and receipt.get("bindings") == required_receipt_bindings
        and receipt.get("provider_manifest_sha256") == actual_provider_manifests
        and approval.get("continue_receipt_sha256") == hashlib.sha256(raw).hexdigest()
    )
    if not valid_receipt:
        raise SystemExit("Phase B approval is not bound to passing receipt")
print(
    json.dumps(
        {
            "approval_sha256": hashlib.sha256(approval_raw).hexdigest(),
            "registry_sha256": hashlib.sha256(registry_raw).hexdigest(),
            "key_id": key_id,
        }
    )
)
PY
}
[ "${1:-}" != --verify-approval ] || { verify_approval; exit 0; }
: "${WORLDMM_STAGE:?}"
: "${WORLDMM_REMOTE_REPO:?}"
: "${WORLDMM_OUTPUT_ROOT:?}"
[ "${WORLDMM_EXECUTION_PROFILE:?}" = teacher-oracle ] || exit 2
root="${WORLDMM_OUTPUT_ROOT%/}"
mkdir -p "$root/summary" "$root/diagnostics"
WORLDMM_EXPERIMENT_GRAPH_FILE="${WORLDMM_EXPERIMENT_GRAPH_FILE:-$(dirname "$0")/experiment_graph.json}"
export WORLDMM_EXPERIMENT_GRAPH_FILE
verify_stage_allocation() {
  local expected expected_partition expected_nodes expected_cpus
  local expected_memory expected_time expected_gpus
  expected="$(
    "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - \
      "$WORLDMM_EXPERIMENT_GRAPH_FILE" "$WORLDMM_STAGE" <<'PY'
import json, sys
from pathlib import Path

graph, runtime_stage = sys.argv[1:]
name = {
    "teacher_oracle_preflight": "preflight", "teacher_oracle_geometry": "geometry",
    "teacher_oracle_semantic": "semantic", "teacher_oracle_place": "place",
    "teacher_oracle_gate": "gate", "teacher_oracle_finalizer": "terminal",
    "teacher_oracle_evaluator": "evaluator",
    "teacher_oracle_finalizer_phase_b": "finalizer",
}.get(runtime_stage, runtime_stage.removeprefix("teacher_oracle_").replace("E0_", "e0_").replace("T0_", "t0_").replace("T1_", "t1_"))
config = json.loads(Path(graph).read_text(encoding="utf-8"))
stage = next((item.get("resources") for item in config.get("stage_specs", [])
              if isinstance(item, dict) and item.get("name") == name), None)
if not isinstance(stage, dict) or set(stage) != {
    "kind", "partition", "nodes", "cpus", "memory", "time", "gpus_per_node"
}:
    raise SystemExit("invalid graph stage resource attestation")
memory = stage["memory"]
memory_mb = int(memory[:-1]) * (1024 if memory.endswith("G") else 1)
print("\t".join(str(value) for value in (
    stage["partition"], stage["nodes"], stage["cpus"], memory_mb,
    stage["time"], stage["gpus_per_node"],
)))
PY
  )"
  IFS=$'\t' read -r expected_partition expected_nodes expected_cpus \
    expected_memory expected_time expected_gpus <<<"$expected"
  [ "${SLURM_JOB_NUM_NODES:-}" = "$expected_nodes" ] || {
    echo "Slurm node allocation does not match approved resources" >&2; return 1; }
  [ "${SLURM_CPUS_PER_TASK:-}" = "$expected_cpus" ] || {
    echo "Slurm CPU allocation does not match approved resources" >&2; return 1; }
  [ "${SLURM_JOB_PARTITION:-}" = "$expected_partition" ] || {
    echo "Slurm partition does not match approved resources" >&2; return 1; }
  [ "${SLURM_MEM_PER_NODE:-}" = "$expected_memory" ] || {
    echo "Slurm memory allocation does not match approved resources" >&2; return 1; }
  [ "${SLURM_TIMELIMIT:-}" = "$expected_time" ] || {
    echo "Slurm time allocation does not match approved resources" >&2; return 1; }
  if [ "$expected_gpus" -gt 0 ]; then
    [ "${SLURM_GPUS_ON_NODE:-}" = "$expected_gpus" ] || {
      echo "Slurm GPU allocation does not match approved resources" >&2; return 1; }
  elif [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [ "${SLURM_GPUS_ON_NODE}" != 0 ]; then
    echo "CPU oracle stage received unexpected GPUs" >&2; return 1
  fi
}
run_phase_b_stage() {
  local variant=$1 action=$2 previous output contract receipt
  : "${WORLDMM_ORACLE_STAGE_EXECUTABLE:?approved oracle stage executable required}"
  [ -x "$WORLDMM_ORACLE_STAGE_EXECUTABLE" ] && \
    [ ! -L "$WORLDMM_ORACLE_STAGE_EXECUTABLE" ] || {
    echo "oracle stage executable is unsafe" >&2; return 1; }
  previous="$root/oracle/$variant"
  case "$action" in
    materialize)
      output="$previous/typed_memory.jsonl"
      contract="$previous/materialize.contract.json"
      ;;
    retrieve)
      output="$previous/evidence_packs.jsonl"
      contract="$previous/retrieve.contract.json"
      ;;
    qa)
      output="$previous/predictions.jsonl"
      contract="$previous/qa.contract.json"
      ;;
    evaluate)
      output="$previous/metrics.json"
      contract="$previous/evaluate.contract.json"
      ;;
    report) output="$previous/report.md"; contract="$previous/report.contract.json" ;;
    *) return 2 ;;
  esac
  mkdir -p "$previous"
  WORLDMM_APPROVAL_PATH="$WORLDMM_PHASE_B_APPROVAL_FILE" \
    WORLDMM_APPROVAL_PHASE=phase_b verify_approval
  verify_stage_allocation
  predecessor_action=""
  case "$action" in
    retrieve) predecessor_action=materialize ;;
    qa) predecessor_action=retrieve ;;
    evaluate) predecessor_action=qa ;;
    report) predecessor_action=evaluate ;;
  esac
  predecessor_receipt=""
  if [ -n "$predecessor_action" ]; then
    predecessor_receipt="$previous/$predecessor_action.receipt.json"
    "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - "$predecessor_receipt" <<'PY'
import hashlib, json, sys
from pathlib import Path
receipt = Path(sys.argv[1])
payload = json.loads(receipt.read_text(encoding="utf-8"))
output = Path(payload["output"])
if not output.is_file() or payload.get("output_sha256") != hashlib.sha256(output.read_bytes()).hexdigest():
    raise SystemExit("predecessor receipt or output digest mismatch")
PY
  fi
  # The contract writer is deliberately separate from the executable: no labels,
  # labels path, student checkpoint, or student-inference input is ever exported.
  "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - \
    "$contract" "$variant" "$action" "$output" "$predecessor_receipt" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
contract = Path(sys.argv[1])
variant, action, output, predecessor_receipt = sys.argv[2:]
bindings = {name: os.environ[name] for name in (
    "WORLDMM_EXPERIMENT_ID", "WORLDMM_SENSOR_AUDIT_SHA256",
    "WORLDMM_PROVIDER_SHA256", "WORLDMM_SPLIT_SHA256", "WORLDMM_CODE_SHA",
    "WORLDMM_POLICY_SHA256", "WORLDMM_TEACHER_ORACLE_VALIDATION_SHA256",
    "WORLDMM_EXPERIMENT_CONFIG_SHA256", "WORLDMM_RUN_ID", "WORLDMM_OUTPUT_ROOT",
    "WORLDMM_FRAME_ASSETS_SHA256", "WORLDMM_BYTE_BUDGET_SHA256",
    "WORLDMM_RESOURCE_CONFIG_SHA256", "WORLDMM_PLAN_SHA256",
    "WORLDMM_SIGNER_REGISTRY_SHA256", "WORLDMM_REMOTE_SNAPSHOT_SHA256",
    "WORLDMM_DAG_PREFLIGHT_SUBMIT_SCRIPT_SHA256",
    "WORLDMM_DAG_PROVIDER_GATE_SUBMIT_SCRIPT_SHA256",
    "WORLDMM_DAG_DOWNSTREAM_SUBMIT_SCRIPT_SHA256",
    "WORLDMM_ORACLE_PROVIDER_EXECUTABLE_SHA256",
    "WORLDMM_ORACLE_STAGE_EXECUTABLE_SHA256", "WORLDMM_ATTESTED_RUNTIME_ROOT",
    "WORLDMM_QA_SHARD_MAP_SHA256", "WORLDMM_QA_LINEAGE_SHA256",
    "WORLDMM_QA_FINALIZATION_RECEIPT_SHA256", "WORLDMM_QA_PREDICTIONS_SHA256")}
payload = {
    "schema_version": 1, "profile": "teacher-oracle", "result_class": "teacher_oracle",
    "variant": variant, "stage": action, "output": output, "bindings": bindings,
    "continue_receipt_sha256": hashlib.sha256(
        Path(os.environ["WORLDMM_CONTINUE_RECEIPT"]).read_bytes()
    ).hexdigest(),
}
if predecessor_receipt:
    receipt = Path(predecessor_receipt)
    payload["predecessor"] = {
        "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "output_sha256": json.loads(receipt.read_text(encoding="utf-8"))["output_sha256"],
    }
temporary = contract.with_name(f".{contract.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
temporary.chmod(0o600)
temporary.replace(contract)
PY
  stage_args=(
    --stage "$action"
    --variant "$variant"
    --contract "$contract"
    --out "$output"
  )
  if [ "$variant" = T1 ] && [ "$action" = report ]; then
    stage_args+=(--all-variants "$root/oracle" \
      --final-report "$root/summary/final_report.md" \
      --manifest "$root/summary/remote_manifest.json")
  fi
  env -i PATH="$PATH" HOME="$HOME" \
    WORLDMM_EXECUTION_PROFILE=teacher-oracle \
    WORLDMM_EXPERIMENT_ID="$WORLDMM_EXPERIMENT_ID" \
    WORLDMM_SENSOR_AUDIT_SHA256="$WORLDMM_SENSOR_AUDIT_SHA256" \
    WORLDMM_PROVIDER_SHA256="$WORLDMM_PROVIDER_SHA256" \
    WORLDMM_SPLIT_SHA256="$WORLDMM_SPLIT_SHA256" \
    WORLDMM_CODE_SHA="$WORLDMM_CODE_SHA" \
    WORLDMM_POLICY_SHA256="$WORLDMM_POLICY_SHA256" \
    WORLDMM_TEACHER_ORACLE_VALIDATION_SHA256="$WORLDMM_TEACHER_ORACLE_VALIDATION_SHA256" \
    WORLDMM_EXPERIMENT_CONFIG_SHA256="$WORLDMM_EXPERIMENT_CONFIG_SHA256" \
    WORLDMM_RUN_ID="$WORLDMM_RUN_ID" \
    WORLDMM_OUTPUT_ROOT="$WORLDMM_OUTPUT_ROOT" \
    WORLDMM_FRAME_ASSETS_SHA256="$WORLDMM_FRAME_ASSETS_SHA256" \
    WORLDMM_BYTE_BUDGET_SHA256="$WORLDMM_BYTE_BUDGET_SHA256" \
    WORLDMM_RESOURCE_CONFIG_SHA256="$WORLDMM_RESOURCE_CONFIG_SHA256" \
    WORLDMM_PLAN_SHA256="$WORLDMM_PLAN_SHA256" \
    WORLDMM_SIGNER_REGISTRY_SHA256="$WORLDMM_SIGNER_REGISTRY_SHA256" \
    WORLDMM_REMOTE_SNAPSHOT_SHA256="$WORLDMM_REMOTE_SNAPSHOT_SHA256" \
    WORLDMM_DAG_PREFLIGHT_SUBMIT_SCRIPT_SHA256="$WORLDMM_DAG_PREFLIGHT_SUBMIT_SCRIPT_SHA256" \
    WORLDMM_DAG_PROVIDER_GATE_SUBMIT_SCRIPT_SHA256="$WORLDMM_DAG_PROVIDER_GATE_SUBMIT_SCRIPT_SHA256" \
    WORLDMM_DAG_DOWNSTREAM_SUBMIT_SCRIPT_SHA256="$WORLDMM_DAG_DOWNSTREAM_SUBMIT_SCRIPT_SHA256" \
    WORLDMM_DAG_STAGE_SCRIPT_SHA256="$WORLDMM_DAG_STAGE_SCRIPT_SHA256" \
    WORLDMM_ORACLE_PROVIDER_EXECUTABLE_SHA256="$WORLDMM_ORACLE_PROVIDER_EXECUTABLE_SHA256" \
    WORLDMM_ORACLE_STAGE_EXECUTABLE_SHA256="$WORLDMM_ORACLE_STAGE_EXECUTABLE_SHA256" \
    WORLDMM_ATTESTED_RUNTIME_ROOT="$WORLDMM_ATTESTED_RUNTIME_ROOT" \
    WORLDMM_QA_SHARD_MAP_SHA256="$WORLDMM_QA_SHARD_MAP_SHA256" \
    WORLDMM_QA_LINEAGE_SHA256="$WORLDMM_QA_LINEAGE_SHA256" \
    WORLDMM_QA_FINALIZATION_RECEIPT_SHA256="$WORLDMM_QA_FINALIZATION_RECEIPT_SHA256" \
    WORLDMM_QA_PREDICTIONS_SHA256="$WORLDMM_QA_PREDICTIONS_SHA256" \
    "$WORLDMM_ORACLE_STAGE_EXECUTABLE" "${stage_args[@]}"
  [ -s "$output" ] || {
    echo "oracle $variant $action emitted no output" >&2
    return 1
  }
  receipt="$previous/$action.receipt.json"
  "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - "$contract" "$output" "$receipt" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
contract, output, receipt = map(Path, sys.argv[1:])
payload = json.loads(contract.read_text(encoding="utf-8"))
payload["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
payload["contract_sha256"] = hashlib.sha256(contract.read_bytes()).hexdigest()
temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
temporary.chmod(0o600)
temporary.replace(receipt)
PY
  if [ "$variant" = T1 ] && [ "$action" = report ]; then
    "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - "$root" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from worldmm_smvqa.report import OracleRunManifest
root = Path(sys.argv[1])
receipts = []
for variant in ("E0", "T0", "T1"):
    for stage in ("materialize", "retrieve", "qa", "evaluate", "report"):
        path = root / "oracle" / variant / f"{stage}.receipt.json"
        if not path.is_file():
            raise SystemExit(f"missing oracle receipt: {path}")
        receipts.append(json.loads(path.read_text(encoding="utf-8")))
identities = [value["bindings"] for value in receipts]
if any(value != identities[0] for value in identities[1:]):
    raise SystemExit(
        "oracle variants have mismatched frame/byte/prompt/model/split identities"
    )
for variant in ("E0", "T0", "T1"):
    metrics = json.loads(
        (root / "oracle" / variant / "metrics.json").read_text(encoding="utf-8")
    )
    if metrics.get("risk_gate") != "pass" or not {
        "Ans-F1",
        "QA-Acc",
        "QA-MRR",
    } <= metrics.keys():
        raise SystemExit(f"oracle risk or metric gate failed for {variant}")
manifest = root / "summary/remote_manifest.json"
if not manifest.is_file():
    raise SystemExit("final oracle executable did not emit OracleRunManifest")
OracleRunManifest.model_validate_json(manifest.read_text(encoding="utf-8"))
report = root / "summary/final_report.md"
if not report.is_file() or not report.read_text(encoding="utf-8").strip():
    raise SystemExit("final oracle executable did not emit final report")
PY
  fi
}
verify_stage_allocation
"$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - <<'PY'
import hashlib, json, os, stat, subprocess, sys, sysconfig
from pathlib import Path

def descriptor(path):
    state = os.lstat(path)
    value = {
        "st_dev": state.st_dev, "st_ino": state.st_ino, "st_mode": state.st_mode,
        "st_uid": state.st_uid, "st_gid": state.st_gid, "st_size": state.st_size,
        "st_mtime_ns": state.st_mtime_ns,
    }
    if stat.S_ISLNK(state.st_mode):
        value["link_target"] = os.readlink(path)
    return value

def attest(path, expected=None, executable=False, require_safe_permissions=True):
    before = descriptor(path)
    if not stat.S_ISREG(before["st_mode"]) or (
        require_safe_permissions and before["st_mode"] & 0o022
    ):
        raise SystemExit(f"unsafe allocation input: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(os.dup(fd), "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected is not None and digest != expected:
            raise SystemExit(f"allocation input digest mismatch: {path}")
        if executable and not before["st_mode"] & 0o111:
            raise SystemExit(f"allocation input is not executable: {path}")
        if descriptor(path) != before or descriptor(path) != {
            "st_dev": os.fstat(fd).st_dev, "st_ino": os.fstat(fd).st_ino,
            "st_mode": os.fstat(fd).st_mode, "st_uid": os.fstat(fd).st_uid,
            "st_gid": os.fstat(fd).st_gid, "st_size": os.fstat(fd).st_size,
            "st_mtime_ns": os.fstat(fd).st_mtime_ns,
        }:
            raise SystemExit(f"allocation input changed: {path}")
        return digest, before
    finally:
        os.close(fd)

def seal_tree(root, allow_symlinks=True):
    root = Path(root)
    root_value = descriptor(root)
    if not stat.S_ISDIR(root_value["st_mode"]) or root_value["st_mode"] & 0o022:
        raise SystemExit(f"unsafe runtime closure root: {root}")
    entries = {}
    def visit(directory, relative=Path(".")):
        with os.scandir(directory) as children:
            for child in sorted(children, key=lambda item: item.name):
                path = Path(child.path)
                item_relative = (relative / child.name).as_posix()
                value = descriptor(path)
                mode = value["st_mode"]
                if not (
                    stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)
                ):
                    raise SystemExit(f"unsafe runtime closure entry: {path}")
                if stat.S_ISLNK(mode) and not allow_symlinks:
                    raise SystemExit(f"runtime root closure contains a forbidden symlink: {path}")
                if stat.S_ISREG(mode):
                    value["sha256"] = attest(path, require_safe_permissions=False)[0]
                entries[item_relative] = value
                if stat.S_ISDIR(mode):
                    visit(path, relative / child.name)
    visit(root)
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return root_value, entries, hashlib.sha256(encoded).hexdigest()

def bind_tree(name, value):
    if not isinstance(value, dict) or set(value) != {
        "path", "descriptor", "entries", "inventory_sha256"
    } or not isinstance(value["path"], str) or not Path(value["path"]).is_absolute():
        raise SystemExit(f"runtime {name} closure schema is invalid")
    observed_root, observed_entries, observed_digest = seal_tree(
        value["path"], allow_symlinks=name != "root"
    )
    if (
        value["descriptor"] != observed_root or value["entries"] != observed_entries
        or value["inventory_sha256"] != observed_digest
    ):
        raise SystemExit(f"runtime {name} closure changed or is incomplete")

def bind_binary(name, value, executable=False):
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "descriptor"}:
        raise SystemExit(f"runtime {name} schema is invalid")
    path = Path(value["path"])
    if not path.is_absolute() or not isinstance(value["sha256"], str):
        raise SystemExit(f"runtime {name} path is invalid")
    digest, observed_descriptor = attest(path, value["sha256"], executable)
    if value["descriptor"] != observed_descriptor:
        raise SystemExit(f"runtime {name} descriptor changed")
    return path, digest

def ldd_closure(interpreter):
    result = subprocess.run(["ldd", str(interpreter)], check=False, text=True,
                            capture_output=True)
    if result.returncode:
        raise SystemExit("could not resolve interpreter shared-library closure")
    paths = set()
    for line in result.stdout.splitlines():
        candidate = line.split("=>", 1)[-1].strip().split(" (", 1)[0]
        if candidate.startswith("/"):
            paths.add(Path(candidate).resolve(strict=True))
    loaders = [path for path in paths if "ld-linux" in path.name or "ld-musl" in path.name]
    if len(loaders) != 1:
        raise SystemExit("interpreter dynamic loader closure is ambiguous")
    loader = loaders[0]
    return loader, sorted(paths - {loader})

manifest_path = Path(os.environ["WORLDMM_ATTESTED_RUNTIME_MANIFEST"])
manifest_fd = os.open(manifest_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    manifest_state = os.fstat(manifest_fd)
    if not stat.S_ISREG(manifest_state.st_mode) or manifest_state.st_uid != os.getuid() or manifest_state.st_mode & 0o077:
        raise SystemExit("unsafe runtime content manifest")
    manifest_raw = os.read(manifest_fd, manifest_state.st_size + 1)
    if len(manifest_raw) != manifest_state.st_size or os.fstat(manifest_fd).st_size != manifest_state.st_size:
        raise SystemExit("runtime content manifest changed while read")
finally:
    os.close(manifest_fd)
if hashlib.sha256(manifest_raw).hexdigest() != os.environ["WORLDMM_ATTESTED_RUNTIME_MANIFEST_SHA256"]:
    raise SystemExit("runtime content manifest digest mismatch")
manifest = json.loads(manifest_raw)
if set(manifest) != {
    "schema_version", "kind", "root", "entry", "interpreter", "base_prefix",
    "stdlib", "loader", "shared_libraries"
} or manifest["schema_version"] != 1 or manifest["kind"] != "RuntimeContentManifestV1":
    raise SystemExit("runtime content manifest schema is invalid")

runtime_root = Path(os.environ["WORLDMM_ATTESTED_RUNTIME_ROOT"])
if (
    not runtime_root.is_absolute() or stat.S_ISLNK(os.lstat(runtime_root).st_mode)
    or not isinstance(manifest["root"], dict)
    or manifest["root"].get("path") != str(runtime_root)
):
    raise SystemExit("runtime root is not manifest-bound")
bind_tree("root", manifest["root"])
entry = runtime_root / "bin" / "python"
if not isinstance(manifest["entry"], dict) or manifest["entry"].get("path") != "bin/python":
    raise SystemExit("runtime entry is invalid")
entry_digest, entry_descriptor = attest(entry, manifest["entry"].get("sha256"), True)
if manifest["entry"].get("descriptor") != entry_descriptor:
    raise SystemExit("runtime entry descriptor changed")

interpreter_path, _ = bind_binary("interpreter", manifest["interpreter"], True)
actual_interpreter = Path(sys.executable).resolve(strict=True)
if actual_interpreter != interpreter_path:
    raise SystemExit("runtime entry exec target is not manifest-bound")
bind_tree("base_prefix", manifest["base_prefix"])
bind_tree("stdlib", manifest["stdlib"])
if Path(manifest["base_prefix"]["path"]) != Path(sys.base_prefix).resolve(strict=True):
    raise SystemExit("actual sys.base_prefix is not manifest-bound")
if Path(manifest["stdlib"]["path"]) != Path(sysconfig.get_path("stdlib")).resolve(strict=True):
    raise SystemExit("actual standard library is not manifest-bound")

loader, libraries = ldd_closure(interpreter_path)
loader_path, _ = bind_binary("loader", manifest["loader"])
if loader_path != loader:
    raise SystemExit("dynamic loader is not manifest-bound")
if not isinstance(manifest["shared_libraries"], list):
    raise SystemExit("shared-library closure schema is invalid")
observed_libraries = []
for value in manifest["shared_libraries"]:
    path, _ = bind_binary("shared library", value)
    observed_libraries.append(path)
if observed_libraries != libraries:
    raise SystemExit("interpreter shared-library closure changed or is incomplete")

stage = os.environ.get("WORLDMM_ORACLE_STAGE_EXECUTABLE")
if stage:
    attest(Path(stage), os.environ["WORLDMM_ORACLE_STAGE_EXECUTABLE_SHA256"], True)
provider = os.environ.get("WORLDMM_ORACLE_PROVIDER_EXECUTABLE")
if provider:
    attest(Path(provider), os.environ["WORLDMM_ORACLE_PROVIDER_EXECUTABLE_SHA256"], True)
config = os.environ.get("WORLDMM_ORACLE_PROVIDER_CONFIG")
config_digest = os.environ.get("WORLDMM_ORACLE_PROVIDER_CONFIG_SHA256")
if config:
    if not config_digest:
        raise SystemExit("provider config is not approval-bound")
    attest(Path(config), config_digest)
attest(Path(os.environ["WORLDMM_EXPERIMENT_GRAPH_FILE"]),
       os.environ["WORLDMM_RESOURCE_CONFIG_SHA256"])
PY
case "$WORLDMM_STAGE" in
teacher_oracle_preflight)
  : "${WORLDMM_SENSOR_MANIFEST:?approved sensor manifest required}"
  : "${WORLDMM_SENSOR_OBSERVATIONS:?approved sensor observations required}"
  : "${WORLDMM_SENSOR_FRAME_ROOT:?approved sensor frame root required}"
  measured_audit="$root/diagnostics/teacher_oracle_sensor_audit.measured.json"
  measured_validation="$root/diagnostics/teacher_oracle_validation.measured.json"
  "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/worldmm-smvqa" audit-sensors \
    --sensor-manifest "$WORLDMM_SENSOR_MANIFEST" \
    --observations "$WORLDMM_SENSOR_OBSERVATIONS" \
    --frame-root "$WORLDMM_SENSOR_FRAME_ROOT" --out "$measured_audit"
  "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/worldmm-smvqa" validate-teacher-oracle-inputs \
    --sensor-audit "$measured_audit" \
    --experiment-config "$WORLDMM_EXPERIMENT_GRAPH_FILE" --out "$measured_validation"
  "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - \
    "$measured_audit" "$measured_validation" <<'PY'
import hashlib
import json
import os
import re
import sys
from pathlib import Path

audit_raw, validation_raw = (Path(path).read_bytes() for path in sys.argv[1:])
audit, value = (json.loads(raw) for raw in (audit_raw, validation_raw))
if not isinstance(audit, dict) or not isinstance(value, dict):
    raise SystemExit("measured preflight outputs must be objects")
if value.get("status") != "pass" or value.get("blockers") != []:
    raise SystemExit("measured validation is not a strict pass")
if value.get("profile") != "teacher-oracle" or value.get(
    "experiment_id"
) != os.environ["WORLDMM_EXPERIMENT_ID"]:
    raise SystemExit("measured validation identity mismatch")
inventory = value.get("selected_sensor_inventory_digest")
if not isinstance(inventory, str) or re.fullmatch(r"[0-9a-f]{64}", inventory) is None:
    raise SystemExit("measured validation has invalid sensor inventory digest")
if value.get("sensor_audit_digest") not in {
    os.environ["WORLDMM_SENSOR_AUDIT_SHA256"], hashlib.sha256(audit_raw).hexdigest()
}:
    raise SystemExit("measured validation is not bound to measured audit")
PY
  "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - \
    "$root" "$measured_audit" "$measured_validation" <<'PY'
import hashlib, json, os, stat, sys
from pathlib import Path

root, audit, validation = map(Path, sys.argv[1:])
def sealed_descriptor(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        state = os.fstat(fd)
        if not stat.S_ISREG(state.st_mode) or state.st_uid != os.getuid() or state.st_mode & 0o022:
            raise SystemExit(f"unsafe measured preflight output: {path}")
        raw = os.read(fd, state.st_size + 1)
        if len(raw) != state.st_size:
            raise SystemExit(f"measured preflight output changed: {path}")
        return {
            "sha256": hashlib.sha256(raw).hexdigest(), "uid": state.st_uid,
            "mode": stat.S_IMODE(state.st_mode), "nlink": state.st_nlink,
            "device": str(state.st_dev), "inode": str(state.st_ino),
            "size": state.st_size, "mtime_ns": str(state.st_mtime_ns),
        }, raw
    finally:
        os.close(fd)
audit_descriptor, audit_raw = sealed_descriptor(audit)
validation_descriptor, validation_raw = sealed_descriptor(validation)
validation_value = json.loads(validation_raw)
payload = {
    "schema_version": 1, "kind": "PreflightSealV1",
    "job_id": os.environ.get("SLURM_JOB_ID", ""),
    "measured_audit": audit_descriptor,
    "measured_validation": validation_descriptor,
    "selected_sensor_inventory_digest": validation_value["selected_sensor_inventory_digest"],
    "resource_config_sha256": os.environ["WORLDMM_RESOURCE_CONFIG_SHA256"],
    "remote_snapshot_sha256": os.environ["WORLDMM_REMOTE_SNAPSHOT_SHA256"],
}
if not payload["job_id"].isdigit():
    raise SystemExit("preflight allocation identity is unavailable")
target = root / "diagnostics/teacher_oracle_preflight.seal"
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    os.write(fd, encoded)
    os.fsync(fd)
finally:
    os.close(fd)
PY
  ;;
teacher_oracle_geometry|teacher_oracle_semantic|teacher_oracle_place)
  WORLDMM_APPROVAL_PATH="$WORLDMM_APPROVAL_FILE" \
    WORLDMM_APPROVAL_PHASE=phase_a verify_approval
  [ -f "$root/diagnostics/teacher_oracle_preflight.seal" ] || exit 1
  producer="${WORLDMM_STAGE#teacher_oracle_}"
  : "${WORLDMM_ORACLE_PROVIDER_EXECUTABLE:?}"
  : "${WORLDMM_ORACLE_PROVIDER_CONFIG:?}"
  attempt="${SLURM_JOB_ID:?provider attempt identity is unavailable}"
  case "$attempt" in *[!0-9]*|'') exit 2 ;; esac
  provider_out="$root/oracle/providers/$producer/attempt-$attempt"
  mkdir -p "$root/oracle/providers/$producer"
  mkdir "$provider_out" || {
    echo "provider attempt root already exists" >&2; exit 1; }
  env -i \
    PATH="$PATH" \
    WORLDMM_OUTPUT_ROOT="$root" \
    WORLDMM_SENSOR_AUDIT_SHA256="$WORLDMM_SENSOR_AUDIT_SHA256" \
    WORLDMM_ORACLE_PROVIDER_CONFIG="$WORLDMM_ORACLE_PROVIDER_CONFIG" \
    WORLDMM_ORACLE_PRODUCER_ID="$producer" \
    "$WORLDMM_ORACLE_PROVIDER_EXECUTABLE" \
      --producer "$producer" \
      --config "$WORLDMM_ORACLE_PROVIDER_CONFIG" \
      --out "$provider_out"
  "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - \
    "$provider_out" "$root" "$producer" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

provider, root = map(Path, sys.argv[1:3])
producer = sys.argv[3]
def sealed_file(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        state = os.fstat(descriptor)
        if (
            not stat.S_ISREG(state.st_mode) or state.st_uid != os.getuid()
            or state.st_mode & 0o022 or state.st_nlink != 1
        ):
            raise SystemExit(f"unsafe provider payload: {path}")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        end = os.fstat(descriptor)
        if (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns) != (
            end.st_dev, end.st_ino, end.st_size, end.st_mtime_ns
        ):
            raise SystemExit(f"provider payload changed while sealed: {path}")
        return {
            "sha256": digest,
            "uid": state.st_uid,
            "mode": stat.S_IMODE(state.st_mode),
            "nlink": state.st_nlink,
            "device": str(state.st_dev),
            "inode": str(state.st_ino),
            "size": state.st_size,
            "mtime_ns": str(state.st_mtime_ns),
        }
    finally:
        os.close(descriptor)

root_state = os.lstat(provider)
if (
    not stat.S_ISDIR(root_state.st_mode) or root_state.st_uid != os.getuid()
    or root_state.st_mode & 0o022 or root_state.st_nlink < 2
):
    raise SystemExit("unsafe provider attempt root")
files = sorted(path for path in provider.rglob("*") if path.is_file() and not path.is_symlink())
if not files:
    raise SystemExit("provider emitted no artifacts")
descriptors = {str(path.relative_to(provider)): sealed_file(path) for path in files}
expectation_path = root / "summary" / f"teacher_oracle_{producer}.expectation.json"
expectation_fd = os.open(expectation_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    expectation_state = os.fstat(expectation_fd)
    expectation_raw = os.read(expectation_fd, expectation_state.st_size + 1)
    if (
        not stat.S_ISREG(expectation_state.st_mode)
        or expectation_state.st_uid != os.getuid()
        or stat.S_IMODE(expectation_state.st_mode) != 0o600
        or expectation_state.st_nlink != 1
        or len(expectation_raw) != expectation_state.st_size
    ):
        raise SystemExit("unsafe provider attempt expectation")
finally:
    os.close(expectation_fd)
expectation = json.loads(expectation_raw)
if expectation != {
    "schema_version": 1, "kind": "ProviderAttemptExpectationV1",
    "stage": f"teacher_oracle_{producer}", "job_id": os.environ["SLURM_JOB_ID"],
    "attempt": os.environ["SLURM_JOB_ID"], "input_sha256": os.environ["WORLDMM_PLAN_SHA256"],
    "resource_sha256": os.environ["WORLDMM_RESOURCE_CONFIG_SHA256"],
    "code_sha256": os.environ["WORLDMM_CODE_SHA"],
    "approval_sha256": hashlib.sha256(
        Path(os.environ["WORLDMM_APPROVAL_FILE"]).read_bytes()
    ).hexdigest(),
}:
    raise SystemExit("provider attempt expectation mismatch")
expectation_sha256 = hashlib.sha256(expectation_raw).hexdigest()
payload = {
    "schema_version": 1,
    "kind": "ProviderAttemptManifestV1",
    "producer_id": producer,
    "stage": f"teacher_oracle_{producer}",
    "attempt": os.environ.get("SLURM_JOB_ID", ""),
    "attempt_root": str(provider),
    "attempt_root_device": str(root_state.st_dev),
    "attempt_root_inode": str(root_state.st_ino),
    "success_marker": "teacher-oracle-producer-v1",
    "provider_executable_sha256": os.environ["WORLDMM_ORACLE_PROVIDER_EXECUTABLE_SHA256"],
    "provider_config_sha256": os.environ["WORLDMM_ORACLE_PROVIDER_CONFIG_SHA256"],
    "resource_config_sha256": os.environ["WORLDMM_RESOURCE_CONFIG_SHA256"],
    "code_sha256": os.environ["WORLDMM_CODE_SHA"],
    "input_bindings": {
        "sensor_audit_sha256": os.environ["WORLDMM_SENSOR_AUDIT_SHA256"],
        "provider_sha256": os.environ["WORLDMM_PROVIDER_SHA256"],
        "split_sha256": os.environ["WORLDMM_SPLIT_SHA256"],
    },
    "provider_artifacts": {name: value["sha256"] for name, value in descriptors.items()},
    "provider_descriptors": descriptors,
    "coverage": sorted(descriptors),
    "causality": "sensor-audit-bound",
    "expectation_sha256": expectation_sha256,
}
if not payload["attempt"].isdigit():
    raise SystemExit("provider attempt identity is unavailable")
temporary = root / "summary" / f".teacher_oracle_{producer}.json.tmp"
temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
temporary.chmod(0o600)
temporary.replace(root / "summary" / f"teacher_oracle_{producer}.json")
PY
  ;;
teacher_oracle_gate)
  WORLDMM_APPROVAL_PATH="$WORLDMM_APPROVAL_FILE" \
    WORLDMM_APPROVAL_PHASE=phase_a verify_approval
  "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - "$root" <<'PY'
import hashlib, json, os, re, stat, subprocess, sys, time
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from worldmm_smvqa.attestation import (
    b64url_decode, b64url_encode, canonicalize, loads_strict,
    require_payload_sha256, signing_bytes, with_payload_sha256,
)
from worldmm_smvqa.slurm_accounting import (
    decode_accounting, is_nonterminal, require_success, sacct_command,
)

root = Path(sys.argv[1])
producers = ("geometry", "semantic", "place")
summary = root / "summary"
result_target = summary / "provider_gate_result.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
def write_once(path, value):
    encoded = canonicalize(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def inventory(jobs):
    values = []
    for producer in producers:
        marker = summary / f"teacher_oracle_{producer}.json"
        values.append({
            "producer_id": producer,
            "job_id": jobs.get(f"PROVIDER_{producer.upper()}_JOB_ID", ""),
            "marker_sha256": (
                hashlib.sha256(marker.read_bytes()).hexdigest()
                if marker.is_file() else None
            ),
        })
    return values

def failure(kind, detail, jobs):
    payload = {
        "schema_version": 1,
        "kind": "ProviderGateFailureV1",
        "profile": "teacher-oracle",
        "failure_kind": kind,
        "detail": detail,
        "producer_inventory": inventory(jobs),
    }
    write_once(result_target, payload)

def digest(value):
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None

def valid_number(value):
    return isinstance(value, int | float) and not isinstance(value, bool)

def validate_quality(value, manifests):
    if not isinstance(value, dict):
        raise ValueError("provider quality result is not an object")
    outcome = value.get("outcome")
    empirical = {
        "empirical_pass", "empirical_no_go", "empirical_not_measurable",
    }
    diagnostic = {
        "diagnostic_contract_eligible", "diagnostic_contract_ineligible",
    }
    common = {"schema_version", "kind", "profile", "outcome"}
    if outcome in empirical:
        required = common | {
            "contract_sha256", "evaluator_sha256", "provider_manifest_sha256",
            "denominator", "metrics", "thresholds", "confidence_interval",
        }
        if set(value) != required:
            raise ValueError("ProviderGateResultV1 empirical discriminator has unexpected fields")
        if (
            value["schema_version"] != 1
            or value["kind"] != "ProviderGateResultV1"
            or value["profile"] != "teacher-oracle"
            or not all(digest(value[name]) for name in ("contract_sha256", "evaluator_sha256"))
            or value["contract_sha256"] != os.environ["WORLDMM_ORACLE_QUALITY_CONTRACT_SHA256"]
            or value["evaluator_sha256"] != os.environ["WORLDMM_ORACLE_QUALITY_EVALUATOR_SHA256"]
            or value["provider_manifest_sha256"] != manifests
            or not isinstance(value["denominator"], int) or isinstance(value["denominator"], bool)
            or value["denominator"] <= 0
            or not isinstance(value["metrics"], dict) or not value["metrics"]
            or not isinstance(value["thresholds"], dict) or not value["thresholds"]
            or not isinstance(value["confidence_interval"], dict)
            or any(not isinstance(name, str) or not valid_number(metric)
                   for name, metric in value["metrics"].items())
            or any(not isinstance(name, str) or not valid_number(threshold)
                   for name, threshold in value["thresholds"].items())
        ):
            raise ValueError("ProviderGateResultV1 empirical evidence is invalid")
    elif outcome in diagnostic:
        required = common | {
            "contract_sha256", "evaluator_sha256", "provider_manifest_sha256",
            "eligibility_evidence",
        }
        evidence = value.get("eligibility_evidence")
        if (
            set(value) != required
            or value["schema_version"] != 1
            or value["kind"] != "ProviderGateResultV1"
            or value["profile"] != "teacher-oracle"
            or not all(digest(value[name]) for name in ("contract_sha256", "evaluator_sha256"))
            or value["contract_sha256"] != os.environ["WORLDMM_ORACLE_QUALITY_CONTRACT_SHA256"]
            or value["evaluator_sha256"] != os.environ["WORLDMM_ORACLE_QUALITY_EVALUATOR_SHA256"]
            or value["provider_manifest_sha256"] != manifests
            or not isinstance(evidence, dict)
            or set(evidence) != {"eligible", "basis"}
            or evidence["eligible"] != (outcome == "diagnostic_contract_eligible")
            or not isinstance(evidence["basis"], str) or not evidence["basis"]
        ):
            raise ValueError("ProviderGateResultV1 diagnostic eligibility evidence is invalid")
    else:
        raise ValueError("ProviderGateResultV1 has unknown outcome")

def verify_descriptor(path, expected):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        state = os.fstat(descriptor)
        required = {
            "uid": state.st_uid,
            "mode": stat.S_IMODE(state.st_mode),
            "nlink": state.st_nlink,
            "device": str(state.st_dev),
            "inode": str(state.st_ino),
            "size": state.st_size,
            "mtime_ns": str(state.st_mtime_ns),
        }
        if (
            not stat.S_ISREG(state.st_mode) or state.st_uid != os.getuid()
            or state.st_mode & 0o022 or state.st_nlink != 1
            or any(expected.get(name) != value for name, value in required.items())
        ):
            raise ValueError(f"provider descriptor mismatch: {path}")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if (
            expected.get("sha256") != actual
            or (
                os.fstat(descriptor).st_size,
                os.fstat(descriptor).st_mtime_ns,
            ) != (state.st_size, state.st_mtime_ns)
        ):
            raise ValueError(f"provider payload changed while consumed: {path}")
    finally:
        os.close(descriptor)

jobs = {}
failure_kind = "producer_admission"
try:
    jobs = dict(
        line.split("=", 1) for line in
        (summary / "dag_jobs.provider.env").read_text().splitlines() if "=" in line
    )
    ids = [jobs.get("PREFLIGHT_JOB_ID", "")] + [
        jobs.get(f"PROVIDER_{producer.upper()}_JOB_ID") for producer in producers
    ]
    if not all(value.isdigit() for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("invalid or duplicate producer lineage")
    manifests = {}
    for producer in producers:
        marker_path = summary / f"teacher_oracle_{producer}.json"
        marker = loads_strict(marker_path.read_bytes())
        provider = root / "oracle" / "providers" / producer / f"attempt-{jobs[f'PROVIDER_{producer.upper()}_JOB_ID']}"
        root_state = os.lstat(provider)
        descriptors = marker.get("provider_descriptors")
        if (
            marker.get("kind") != "ProviderAttemptManifestV1"
            or marker.get("attempt_root") != str(provider)
            or marker.get("attempt_root_device") != str(root_state.st_dev)
            or marker.get("attempt_root_inode") != str(root_state.st_ino)
            or not isinstance(descriptors, dict)
        ):
            raise ValueError(f"invalid provider attempt root: {producer}")
        expectation_raw = (summary / f"teacher_oracle_{producer}.expectation.json").read_bytes()
        expectation = loads_strict(expectation_raw)
        expected_expectation = {
            "schema_version": 1, "kind": "ProviderAttemptExpectationV1",
            "stage": f"teacher_oracle_{producer}",
            "job_id": jobs[f"PROVIDER_{producer.upper()}_JOB_ID"],
            "attempt": jobs[f"PROVIDER_{producer.upper()}_JOB_ID"],
            "input_sha256": os.environ["WORLDMM_PLAN_SHA256"],
            "resource_sha256": os.environ["WORLDMM_RESOURCE_CONFIG_SHA256"],
            "code_sha256": os.environ["WORLDMM_CODE_SHA"],
            "approval_sha256": hashlib.sha256(
                Path(os.environ["WORLDMM_APPROVAL_FILE"]).read_bytes()
            ).hexdigest(),
        }
        if (
            expectation != expected_expectation
            or jobs.get(f"PROVIDER_{producer.upper()}_EXPECTATION_SHA256")
            != hashlib.sha256(expectation_raw).hexdigest()
            or marker.get("expectation_sha256")
            != hashlib.sha256(expectation_raw).hexdigest()
        ):
            raise ValueError(f"provider expectation mismatch: {producer}")
        actual = {}
        for relative, descriptor in descriptors.items():
            if not isinstance(relative, str) or not isinstance(descriptor, dict):
                raise ValueError(f"invalid provider descriptor: {producer}")
            payload_path = provider / relative
            if payload_path.parent != provider and provider not in payload_path.parents:
                raise ValueError(f"provider payload escapes attempt root: {producer}")
            verify_descriptor(payload_path, descriptor)
            actual[relative] = descriptor["sha256"]
        if (
            marker_path.stat().st_size == 0
            or marker.get("schema_version") != 1
            or marker.get("producer_id") != producer
            or marker.get("success_marker") != "teacher-oracle-producer-v1"
            or marker.get("attempt") != jobs[f"PROVIDER_{producer.upper()}_JOB_ID"]
            or marker.get("provider_artifacts") != actual
            or marker.get("coverage") != sorted(actual)
            or not actual
            or marker.get("stage") != f"teacher_oracle_{producer}"
            or marker.get("provider_executable_sha256") != os.environ["WORLDMM_ORACLE_PROVIDER_EXECUTABLE_SHA256"]
            or marker.get("provider_config_sha256") != os.environ["WORLDMM_ORACLE_PROVIDER_CONFIG_SHA256"]
            or marker.get("resource_config_sha256") != os.environ["WORLDMM_RESOURCE_CONFIG_SHA256"]
            or marker.get("code_sha256") != os.environ["WORLDMM_CODE_SHA"]
            or marker.get("input_bindings") != {
                "sensor_audit_sha256": os.environ["WORLDMM_SENSOR_AUDIT_SHA256"],
                "provider_sha256": os.environ["WORLDMM_PROVIDER_SHA256"],
                "split_sha256": os.environ["WORLDMM_SPLIT_SHA256"],
            }
        ):
            raise ValueError(f"invalid producer admission: {producer}")
        manifests[producer] = hashlib.sha256(marker_path.read_bytes()).hexdigest()

    failure_kind = "accounting"
    cluster = os.environ["WORLDMM_SLURM_CLUSTER"]
    deadline = time.monotonic() + int(os.environ["WORLDMM_ACCOUNTING_SETTLE_SECONDS"])
    interval = float(os.environ["WORLDMM_ACCOUNTING_SETTLE_INTERVAL_SECONDS"])
    if deadline <= time.monotonic() or interval <= 0:
        raise ValueError("invalid approval-bound accounting settle policy")
    records = {}
    while True:
        records.clear()
        waiting = False
        for job in ids:
            try:
                record = decode_accounting(
                    subprocess.check_output(
                        sacct_command(
                            sacct=os.environ.get("WORLDMM_SACCT", "sacct"),
                            cluster=cluster, job_id=job,
                        ), text=True,
                    ),
                    cluster=cluster, job_id=job,
                )
                if is_nonterminal(record):
                    waiting = True
                    break
                records[job] = require_success(record)
            except ValueError as exc:
                if "expected exactly one allocation row, got 0" in str(exc):
                    waiting = True
                    break
                raise ValueError(f"producer accounting rejected {job}: {exc}") from exc
            except (OSError, subprocess.CalledProcessError):
                waiting = True
                break
        if records and not waiting:
            break
        if time.monotonic() >= deadline:
            raise ValueError("producer accounting did not settle before approved deadline")
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))

    failure_kind = "evaluator"
    evaluator = Path(os.environ["WORLDMM_ORACLE_QUALITY_EVALUATOR"])
    contract = Path(os.environ["WORLDMM_ORACLE_QUALITY_CONTRACT"])
    if (
        not evaluator.is_file() or not contract.is_file()
        or hashlib.sha256(evaluator.read_bytes()).hexdigest()
           != os.environ["WORLDMM_ORACLE_QUALITY_EVALUATOR_SHA256"]
        or hashlib.sha256(contract.read_bytes()).hexdigest()
           != os.environ["WORLDMM_ORACLE_QUALITY_CONTRACT_SHA256"]
    ):
        raise ValueError("approval-bound provider quality evaluator or contract digest mismatch")
    temporary = result_target.with_name(f".{result_target.name}.{os.getpid()}.tmp")
    subprocess.run(
        [str(evaluator), "--contract", str(contract), "--out", str(temporary)],
        check=True, env={"PATH": os.environ.get("PATH", ""), "WORLDMM_OUTPUT_ROOT": str(root)},
    )
    failure_kind = "schema"
    quality = loads_strict(temporary.read_bytes())
    validate_quality(quality, manifests)
    temporary.unlink(missing_ok=True)
    if quality["outcome"] in {
        "empirical_pass", "diagnostic_contract_eligible",
    }:
        bindings = {
            "preflight_seal_sha256": os.environ["WORLDMM_PREFLIGHT_SEAL_SHA256"],
            "quality_contract_sha256": os.environ["WORLDMM_ORACLE_QUALITY_CONTRACT_SHA256"],
            "quality_evaluator_sha256": os.environ["WORLDMM_ORACLE_QUALITY_EVALUATOR_SHA256"],
            "experiment_id": os.environ["WORLDMM_EXPERIMENT_ID"],
            "sensor_audit_sha256": os.environ["WORLDMM_SENSOR_AUDIT_SHA256"],
            "provider_sha256": os.environ["WORLDMM_PROVIDER_SHA256"],
            "split_sha256": os.environ["WORLDMM_SPLIT_SHA256"],
            "code_sha256": os.environ["WORLDMM_CODE_SHA"],
            "policy_sha256": os.environ["WORLDMM_POLICY_SHA256"],
            "validation_receipt_sha256": os.environ["WORLDMM_TEACHER_ORACLE_VALIDATION_SHA256"],
            "experiment_config_sha256": os.environ["WORLDMM_EXPERIMENT_CONFIG_SHA256"],
            "run_id": os.environ["WORLDMM_RUN_ID"],
            "output_root": os.environ["WORLDMM_OUTPUT_ROOT"],
            "frame_assets_sha256": os.environ["WORLDMM_FRAME_ASSETS_SHA256"],
            "byte_budget_sha256": os.environ["WORLDMM_BYTE_BUDGET_SHA256"],
            "resource_config_sha256": os.environ["WORLDMM_RESOURCE_CONFIG_SHA256"],
            "plan_sha256": os.environ["WORLDMM_PLAN_SHA256"],
            "remote_snapshot_sha256": os.environ["WORLDMM_REMOTE_SNAPSHOT_SHA256"],
            "dag_preflight_submit_script_sha256": os.environ["WORLDMM_DAG_PREFLIGHT_SUBMIT_SCRIPT_SHA256"],
            "dag_provider_gate_submit_script_sha256": os.environ["WORLDMM_DAG_PROVIDER_GATE_SUBMIT_SCRIPT_SHA256"],
            "dag_downstream_submit_script_sha256": os.environ["WORLDMM_DAG_DOWNSTREAM_SUBMIT_SCRIPT_SHA256"],
            "dag_stage_script_sha256": os.environ["WORLDMM_DAG_STAGE_SCRIPT_SHA256"],
            "oracle_provider_executable_sha256": os.environ["WORLDMM_ORACLE_PROVIDER_EXECUTABLE_SHA256"],
            "oracle_stage_executable_sha256": os.environ["WORLDMM_ORACLE_STAGE_EXECUTABLE_SHA256"],
            "registry_sha256": hashlib.sha256(Path(os.environ["WORLDMM_SIGNER_REGISTRY"]).read_bytes()).hexdigest(),
            "attested_runtime_root": os.environ["WORLDMM_ATTESTED_RUNTIME_ROOT"],
            "attested_runtime_manifest_sha256": os.environ["WORLDMM_ATTESTED_RUNTIME_MANIFEST_SHA256"],
            "purpose": "teacher_oracle_phase_b_execution",
            "slurm_cluster": os.environ["WORLDMM_SLURM_CLUSTER"],
            "accounting_settle_seconds": os.environ["WORLDMM_ACCOUNTING_SETTLE_SECONDS"],
            "accounting_settle_interval_seconds": os.environ["WORLDMM_ACCOUNTING_SETTLE_INTERVAL_SECONDS"],
            "producer_tuple": list(producers),
            "producer_stage_tuple": [f"teacher_oracle_{producer}" for producer in producers],
            "oracle_provider_config_sha256": os.environ["WORLDMM_ORACLE_PROVIDER_CONFIG_SHA256"],
        }
        receipt = with_payload_sha256({
            "schema_version": 1, "kind": "SignedAttestationEnvelopeV1",
            "profile": "teacher-oracle", "decision": "go",
            "producer_jobs": {
                producer: {
                    "job_id": jobs[f"PROVIDER_{producer.upper()}_JOB_ID"],
                    "sluid": records[jobs[f"PROVIDER_{producer.upper()}_JOB_ID"]].sluid,
                    "original_sluid": records[jobs[f"PROVIDER_{producer.upper()}_JOB_ID"]].original_sluid,
                } for producer in producers
            },
            "provider_manifest_sha256": manifests, "bindings": bindings,
            "key_id": os.environ["WORLDMM_CONTINUE_RECEIPT_KEY_ID"],
        })
        key_path = Path(os.environ["WORLDMM_CONTINUE_RECEIPT_SIGNING_KEY"])
        key_fd = os.open(key_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            key_state = os.fstat(key_fd)
            if (
                not stat.S_ISREG(key_state.st_mode) or key_state.st_uid != os.getuid()
                or stat.S_IMODE(key_state.st_mode) != 0o600 or key_state.st_nlink != 1
                or key_state.st_size != 44
            ):
                raise ValueError("unsafe continue-receipt private-key file")
            encoded_raw = os.read(key_fd, 45)
            end_state = os.fstat(key_fd)
            if (
                len(encoded_raw) != 44 or not encoded_raw.endswith(b"\n")
                or (key_state.st_dev, key_state.st_ino, key_state.st_size, key_state.st_mtime_ns)
                != (end_state.st_dev, end_state.st_ino, end_state.st_size, end_state.st_mtime_ns)
            ):
                raise ValueError("continue-receipt private-key changed while read")
            encoded = encoded_raw[:-1].decode("ascii")
            private = b64url_decode(encoded)
        finally:
            os.close(key_fd)
        if len(private) != 32:
            raise ValueError("invalid continue-receipt private-key file format")
        receipt["signature"] = b64url_encode(
            Ed25519PrivateKey.from_private_bytes(private).sign(
                signing_bytes(receipt, "continue-receipt")
            )
        )
        write_once(summary / "teacher_oracle_continue.json", receipt)
    write_once(result_target, quality)
except Exception as exc:
    try:
        failure(failure_kind, str(exc), jobs)
    except FileExistsError:
        pass
    raise SystemExit(f"provider gate emitted {failure_kind} failure: {exc}") from exc
PY
  ;;
teacher_oracle_finalizer)
  WORLDMM_APPROVAL_PATH="$WORLDMM_APPROVAL_FILE" \
    WORLDMM_APPROVAL_PHASE=phase_a verify_approval
  "$WORLDMM_ATTESTED_RUNTIME_ROOT/bin/python" - "$root" <<'PY'
import hashlib, json, os, subprocess, sys
from pathlib import Path

from worldmm_smvqa.slurm_accounting import (
    decode_accounting, is_cancelled, require_success, sacct_command,
)

root = Path(sys.argv[1])
summary = root / "summary"
producers = ("geometry", "semantic", "place")
jobs = dict(
    line.split("=", 1) for line in
    (summary / "dag_jobs.provider.env").read_text().splitlines() if "=" in line
)
gate = jobs.get("PROVIDER_GATE_JOB_ID", "")
continuation = summary / "teacher_oracle_continue.json"

def producer_inventory():
    return [
        {
            "producer_id": producer,
            "job_id": jobs.get(f"PROVIDER_{producer.upper()}_JOB_ID", ""),
            "marker_sha256": (
                hashlib.sha256((summary / f"teacher_oracle_{producer}.json").read_bytes()).hexdigest()
                if (summary / f"teacher_oracle_{producer}.json").is_file() else None
            ),
        }
        for producer in producers
    ]

def write_terminal_once(path, payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"conflicting terminal artifact: {path.name}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


operational, scientific, decision = "failed", "not_decidable", "gate_controller_failure"
try:
    record = decode_accounting(
        subprocess.check_output(
            sacct_command(
                sacct=os.environ.get("WORLDMM_SACCT", "sacct"),
                cluster=os.environ["WORLDMM_SLURM_CLUSTER"], job_id=gate,
            ), text=True,
        ),
        cluster=os.environ["WORLDMM_SLURM_CLUSTER"], job_id=gate,
    )
    if is_cancelled(record):
        decision = "gate_controller_failure"
    else:
        require_success(record)
        quality = json.loads((summary / "provider_gate_result.json").read_bytes())
        kind, outcome = quality.get("kind"), quality.get("outcome")
        if kind == "ProviderGateFailureV1":
            if (
                set(quality) != {
                    "schema_version", "kind", "profile", "failure_kind", "detail",
                    "producer_inventory",
                }
                or quality.get("schema_version") != 1
                or quality.get("profile") != "teacher-oracle"
                or quality.get("failure_kind") not in {
                    "producer_admission", "accounting", "evaluator", "schema",
                }
                or not isinstance(quality.get("detail"), str)
                or quality.get("producer_inventory") != producer_inventory()
            ):
                raise ValueError("invalid ProviderGateFailureV1")
            decision = "gate_failure"
        elif kind == "ProviderGateResultV1" and quality.get("schema_version") == 1 and quality.get("profile") == "teacher-oracle":
            eligible = outcome in {
                "empirical_pass", "diagnostic_contract_eligible",
            }
            complete = {
                "empirical_no_go": ("complete", "no_go"),
                "empirical_not_measurable": ("complete", "not_measurable"),
                "diagnostic_contract_ineligible": ("complete", "not_decidable"),
            }
            if eligible:
                receipt = json.loads(continuation.read_bytes())
                if (
                    receipt.get("decision") != "go"
                    or receipt.get("profile") != "teacher-oracle"
                    or not isinstance(receipt.get("signature"), str)
                    or not receipt["signature"]
                ):
                    raise ValueError("invalid continuation receipt")
                operational, scientific, decision = (
                    "authorized", "pending", "continuation_authorized",
                )
            elif outcome in complete:
                operational, scientific = complete[outcome]
                decision = outcome
            else:
                raise ValueError("invalid ProviderGateResultV1 outcome")
        else:
            raise ValueError("missing typed gate result")
except Exception:
    operational, scientific, decision = "failed", "not_decidable", "gate_controller_failure"

if operational != "authorized":
    continuation.unlink(missing_ok=True)
terminal = {
    "schema_version": 1,
    "kind": "ProviderGateTerminalV1",
    "profile": "teacher-oracle",
    "provider_gate_decision": "go" if decision == "continuation_authorized" else decision,
    "operational_state": operational,
    "scientific_state": scientific,
    "gate_job_id": gate,
    "producer_inventory": producer_inventory(),
}
write_terminal_once(summary / "teacher_oracle_terminal.json", terminal)
write_terminal_once(summary / "provider_gate_terminal_v1.json", terminal)
PY
  ;;
teacher_oracle_E0_materialize|teacher_oracle_E0_retrieve|\
teacher_oracle_E0_qa|teacher_oracle_E0_evaluate|teacher_oracle_E0_report|\
teacher_oracle_T0_materialize|teacher_oracle_T0_retrieve|\
teacher_oracle_T0_qa|teacher_oracle_T0_evaluate|teacher_oracle_T0_report|\
teacher_oracle_T1_materialize|teacher_oracle_T1_retrieve|\
teacher_oracle_T1_qa|teacher_oracle_T1_evaluate|teacher_oracle_T1_report)
  phase_b="${WORLDMM_STAGE#teacher_oracle_}"
  variant="${phase_b%%_*}"
  action="${phase_b#*_}"
  run_phase_b_stage "$variant" "$action"
  ;;
teacher_oracle_evaluator)
  for variant in E0 T0 T1; do
    run_phase_b_stage "$variant" evaluate
  done
  ;;
teacher_oracle_finalizer_phase_b)
  for variant in E0 T0 T1; do
    run_phase_b_stage "$variant" report
  done
  ;;
*) exit 2 ;;
esac
"""
    )


def student_submit_script_text(graph: _StudentGraphLike) -> str:
    """Render the reviewed held student graph without submitting it locally."""
    stages = tuple(graph.stages)
    edges = tuple(graph.edges)
    stage_ids = {str(stage.stage_id) for stage in stages}
    dependency_map: dict[str, tuple[str, list[str]]] = {}
    for edge in edges:
        source = str(edge.from_stage)
        target = str(edge.to_stage)
        kind = str(edge.dependency_kind)
        if source not in stage_ids or target not in stage_ids:
            msg = "student graph edge references an unknown stage"
            raise ValueError(msg)
        prior = dependency_map.get(target)
        if prior is not None and prior[0] != kind:
            msg = "student graph mixes dependency kinds for one stage"
            raise ValueError(msg)
        dependency_map.setdefault(target, (kind, []))[1].append(source)
    dependency_rows = [
        f'EDGE["{target}"]="{kind}:{";".join(parents)}"'
        for target, (kind, parents) in dependency_map.items()
    ]
    stage_rows: list[str] = []
    for stage in stages:
        stage_id = str(stage.stage_id)
        host_class = str(stage.host_class)
        nodes = int(stage.nodes)
        gpus = int(stage.gpus_per_node)
        cpus = int(stage.cpus_per_task)
        memory = int(stage.memory_gb)
        minutes = int(stage.time_limit_minutes)
        stage_rows.append(
            f'STAGE["{stage_id}"]="{host_class}|{nodes}|{gpus}|{cpus}|{memory}|{minutes}"'
        )
    unheld = (
        "model_load_gate",
        "model_load_terminal",
        "control_primary",
        "control_backup",
        "control_actuator",
        "student_watchdog",
    )
    return dedent(
        f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            : "${{WORLDMM_SMVQA_REMOTE_APPROVED:?explicit approval is required}}"
            [[ "$WORLDMM_SMVQA_REMOTE_APPROVED" == "1" ]] || exit 2
            : "${{WORLDMM_RUN_ID:?WORLDMM_RUN_ID is required}}"
            : "${{WORLDMM_OUTPUT_ROOT:?WORLDMM_OUTPUT_ROOT is required}}"
            readonly SBATCH=/opt/slurm/bin/sbatch
            readonly SCONTROL=/opt/slurm/bin/scontrol
            readonly SCANCEL=/opt/slurm/bin/scancel
            readonly STAGE_RUNNER="${{WORLDMM_REMOTE_REPO}}/remote-plan/run_student_stage.sh"
            declare -A STAGE EDGE JOB
            {chr(10).join(stage_rows)}
            {chr(10).join(dependency_rows)}
            readonly UNHELD_STAGES="{" ".join(unheld)}"

            is_unheld() {{
              [[ " $UNHELD_STAGES " == *" $1 "* ]]
            }}

            submit_stage() {{
              local stage="$1" spec="${{STAGE[$1]}}" host nodes gpus cpus memory minutes
              IFS='|' read -r host nodes gpus cpus memory minutes <<<"$spec"
              local args=(
                "--parsable" "--no-requeue"
                "--job-name=worldmm-${{WORLDMM_RUN_ID}}-${{stage}}"
                "--nodes=$nodes" "--cpus-per-task=$cpus"
                "--mem=${{memory}}G" "--time=$minutes"
                "--output=${{WORLDMM_OUTPUT_ROOT}}/logs/${{stage}}-%j.log"
                "--export=NONE"
              )
              if [[ "$host" == gpu ]]; then
                args+=("--partition=gpu-vtt-queue" "--gpus-per-node=$gpus")
              else
                args+=("--partition=cpu-prepro-queue")
              fi
              if ! is_unheld "$stage"; then args+=("--hold"); fi
              if [[ -n "${{EDGE[$stage]:-}}" ]]; then
                local edge="${{EDGE[$stage]}}" kind="${{edge%%:*}}" parents="${{edge#*:}}"
                local dependency_ids="" parent
                IFS=';' read -ra parent_stages <<<"$parents"
                for parent in "${{parent_stages[@]}}"; do
                  dependency_ids+="${{dependency_ids:+:}}${{JOB[$parent]}}"
                done
                args+=("--dependency=${{kind}}:${{dependency_ids}}")
              fi
              JOB["$stage"]="$("$SBATCH" "${{args[@]}}" "$STAGE_RUNNER" "$stage")"
            }}

            # Submission order preserves job-id causality. No workload is released here.
            submit_stage preflight_ingest
            submit_stage model_load_workers
            submit_stage model_load_gate
            submit_stage model_load_terminal
            submit_stage control_primary
            submit_stage control_backup
            submit_stage control_actuator
            submit_stage student_watchdog
            submit_stage teacher_extract
            submit_stage merge_materialize
            submit_stage train
            submit_stage spatial_infer
            submit_stage qwen_episodic
            submit_stage qwen_semantic_visual
            submit_stage retrieval_join
            submit_stage qa
            submit_stage metrics_report

            # A controller/actuator validates the immutable submission manifest and approval.
            # Controllers publish proposals only; only control_actuator may call scontrol/scancel.
            printf '%s\\n' "held student graph generated; no workload released"
            """
    ).lstrip()


def student_stage_script_text(graph: _StudentGraphLike) -> str:
    """Render command-key dispatch for the strict student graph."""
    rows: list[str] = []
    for stage in tuple(graph.stages):
        stage_id = str(stage.stage_id)
        command_key = str(stage.command_key)
        rows.append(f'  {stage_id}) command_key="{command_key}" ;;')
    return dedent(
        f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            readonly stage="${{1:?student stage id is required}}"
            case "$stage" in
            {chr(10).join(rows)}
              *) exit 2 ;;
            esac
            export PYTHONNOUSERSITE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
            case "$command_key" in
              student.model_load_workers)
                exec python -m worldmm_smvqa.model_load worker
                ;;
              student.model_load_gate)
                exec python -m worldmm_smvqa.model_load gate
                ;;
              student.model_load_terminal)
                exec python -m worldmm_smvqa.model_load terminal
                ;;
              student.control_primary|student.control_backup)
                exec python -m worldmm_smvqa.model_load controller "$command_key"
                ;;
              student.control_actuator)
                exec python -m worldmm_smvqa.model_load actuator
                ;;
              student.student_watchdog)
                exec python -m worldmm_smvqa.model_load watchdog
                ;;
              *)
                exec worldmm-smvqa remote-stage --command-key "$command_key"
                ;;
            esac
            """
    ).lstrip()
