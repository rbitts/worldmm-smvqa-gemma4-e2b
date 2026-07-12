from __future__ import annotations

from textwrap import dedent


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
        WORLDMM_REMOTE_REPO="${WORLDMM_REMOTE_REPO:-/repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b}"
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
          local python_runtime_root python_runtime_files
          local python_runtime_inventory python_runtime_resolved
          local unsafe_link unsafe_pth runtime_link runtime_target
          local pth_file pth_dir pth_line pth_candidate pth_resolved
          local inventory_current base_roots base_files base_inventory
          local base_prefix base_executable stdlib platstdlib root
          python_runtime_root="$WORLDMM_REMOTE_REPO/.venv"
          python_runtime_files="$WORLDMM_OUTPUT_ROOT/diagnostics/python_runtime.sha256"
          python_runtime_inventory="$WORLDMM_OUTPUT_ROOT/diagnostics/python_runtime.files.sha256"
          base_roots="$WORLDMM_OUTPUT_ROOT/diagnostics/python_base_roots.tsv"
          base_files="$WORLDMM_OUTPUT_ROOT/diagnostics/python_base_runtime.sha256"
          base_inventory="$WORLDMM_OUTPUT_ROOT/diagnostics/python_base_runtime.files.sha256"
          if [ ! -d "$python_runtime_root" ] || [ -L "$python_runtime_root" ] || \
            [ ! -s "$python_runtime_files" ] || \
            [ ! -s "$python_runtime_inventory" ] || \
            [ ! -s "$base_roots" ] || [ ! -s "$base_files" ] || \
            [ ! -s "$base_inventory" ]; then
            printf "approved Python runtime or its manifests are missing\n" >&2
            return 1
          fi
          python_runtime_resolved="$(realpath -e "$python_runtime_root")"
          unsafe_link="$(
            find "$python_runtime_root" -type l -print0 | \
              while IFS= read -r -d '' runtime_link; do
                if [ ! -e "$runtime_link" ]; then
                  printf "%s" "$runtime_link"
                  break
                fi
                if [ -d "$runtime_link" ]; then
                  runtime_target="$(realpath -e "$runtime_link")"
                  case "$runtime_target" in
                    "$python_runtime_resolved"|"$python_runtime_resolved"/*) ;;
                    *)
                      printf "%s" "$runtime_link"
                      break
                      ;;
                  esac
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
        partial_jobs="$WORLDMM_OUTPUT_ROOT/summary/dag_jobs.${WORLDMM_DAG_PHASE}.partial"
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
          mapfile -t submitted_job_ids < <(
            cut -d= -f2 "$partial_jobs" | grep -E '^[0-9]+$' || true
          )
          if [ "${#submitted_job_ids[@]}" -eq 0 ] && \
            [ ! -s "$submission_attempts" ]; then
            rm -f "$submit_lock"
          elif [ "${#submitted_job_ids[@]}" -eq 0 ]; then
            printf \
              "sbatch returned no trustworthy job ID; keeping lock: %s\n" \
              "$submit_lock" >&2
          elif [ -x "$SCANCEL" ] && \
            "$SCANCEL" "${submitted_job_ids[@]}"; then
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
          stage_exports+=",WORLDMM_APPROVED_OUTPUT_PREFIX=$WORLDMM_APPROVED_OUTPUT_PREFIX"
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
          stage_exports+=",WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW=$WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW"
          stage_exports+=",WORLDMM_APPROVAL_FILE=${WORLDMM_APPROVAL_FILE:-}"
          stage_exports+=",WORLDMM_APPROVER=${WORLDMM_APPROVER:-}"
          stage_exports+=",WORLDMM_APPROVAL_SHA256=$WORLDMM_APPROVAL_SHA256"
          stage_exports+=",WORLDMM_GCUT3R_EXTRACTOR=${WORLDMM_GCUT3R_EXTRACTOR:-}"
          stage_exports+=",WORLDMM_TEACHER_CACHE_INPUT=${WORLDMM_TEACHER_CACHE_INPUT:-}"
          stage_exports+=",WORLDMM_SPATIAL_INFER_EXE=${WORLDMM_SPATIAL_INFER_EXE:-}"
          stage_exports+=",WORLDMM_STUDENT_SUPERVISION_INPUT=${WORLDMM_STUDENT_SUPERVISION_INPUT:-}"
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
            "$SBATCH" --parsable
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
          )
          if [ "$gpus" -gt 0 ]; then
            args+=("--gpus-per-node=$gpus")
          fi
          if [ -n "$dependency" ]; then
            args+=(--kill-on-invalid-dep=yes "--dependency=afterok:$dependency")
          fi
          printf "%s\n" "$stage" >> "$submission_attempts"
          raw_job_id="$("${args[@]}" "$stage_script")"
          job_id="${raw_job_id%%;*}"
          if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
            printf "invalid sbatch job id for %s: %s\n" "$stage" "$raw_job_id" >&2
            return 1
          fi
          printf "%s=%s\n" "$stage" "$job_id" >> "$partial_jobs"
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
        WORLDMM_REMOTE_REPO="${WORLDMM_REMOTE_REPO:-/repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b}"
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
                if [ ! -e "$runtime_link" ]; then
                  printf "%s" "$runtime_link"
                  break
                fi
                if [ -d "$runtime_link" ]; then
                  runtime_target="$(realpath -e "$runtime_link")"
                  case "$runtime_target" in
                    "$runtime_resolved"|"$runtime_resolved"/*) ;;
                    *) printf "%s" "$runtime_link"; break ;;
                  esac
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
          base_roots="$WORLDMM_OUTPUT_ROOT/diagnostics/python_base_roots.tsv"
          base_files="$WORLDMM_OUTPUT_ROOT/diagnostics/python_base_runtime.sha256"
          base_inventory="$WORLDMM_OUTPUT_ROOT/diagnostics/python_base_runtime.files.sha256"
          if [ ! -s "$runtime_files" ] || [ ! -s "$runtime_inventory" ] || \
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
        mkdir -p "$WORLDMM_OUTPUT_ROOT"/\
          {manifests,inference_inputs,teacher,training,checkpoints,memory,retrieval,qa,metrics,diagnostics,summary}
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
