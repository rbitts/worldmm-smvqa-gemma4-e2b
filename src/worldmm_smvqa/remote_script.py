from __future__ import annotations

from textwrap import dedent


def script_text() -> str:
    stage_lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=worldmm-smvqa",
        "#SBATCH --partition=gpu-vtt-queue",
        "#SBATCH --nodes=10",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --gpus-per-node=8",
        "#SBATCH --output=remote-plan/logs/worldmm-smvqa-%j.out",
        "#SBATCH --error=remote-plan/logs/worldmm-smvqa-%j.err",
        "set -euo pipefail",
        "",
        ': "${SLURM_JOB_ID:?submit this script with /opt/slurm/bin/sbatch}"',
        ': "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"',
        (
            'WORLDMM_REMOTE_REPO="${WORLDMM_REMOTE_REPO:-'
            '/repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b}"'
        ),
        'if [ -f "$WORLDMM_REMOTE_REPO/.env.worldmm" ]; then',
        "  set -a",
        '  source "$WORLDMM_REMOTE_REPO/.env.worldmm"',
        "  set +a",
        "fi",
        (': "${SMVQA_DATA_ROOT:=/groups/VTteam/datasets/SuperMemory-VQA/ingested}"'),
        ': "${SMVQA_FRAME_ROOT:=$SMVQA_DATA_ROOT/frames}"',
        (': "${GEMMA_MODEL_PATH:=/repo/VTteam/bongh.park/gemma-4-e2b-it}"'),
        ': "${WORLDMM_RUN_ID:=slurm-${SLURM_JOB_ID}}"',
        (
            ': "${WORLDMM_OUTPUT_ROOT:='
            '/repo/VTteam/bongh.park/outputs/${WORLDMM_RUN_ID}}"'
        ),
        'if [ "${WORLDMM_SMVQA_REMOTE_APPROVED:-}" != "1" ]; then',
        '  printf "WORLDMM_SMVQA_REMOTE_APPROVED=1 is required\\n" >&2',
        "  exit 1",
        "fi",
        'case "$WORLDMM_OUTPUT_ROOT" in',
        '  */"$WORLDMM_RUN_ID") ;;',
        "  *)",
        (
            '    printf "WORLDMM_OUTPUT_ROOT must end /%s: %s\\n" '
            '"$WORLDMM_RUN_ID" "$WORLDMM_OUTPUT_ROOT" >&2'
        ),
        "    exit 1",
        "    ;;",
        "esac",
        (
            'WORLDMM_SENSOR_FRAME_MANIFEST="$WORLDMM_OUTPUT_ROOT/'
            'manifests/sensor_frames.jsonl"'
        ),
        ': "${WORLDMM_MODEL_ID:=google/gemma-4-E2B-it}"',
        ': "${WORLDMM_MEMORY_MODEL_ID:=Qwen/Qwen3-VL-8B-Instruct}"',
        (
            ': "${WORLDMM_MEMORY_MODEL_PATH:='
            '/repo/VTteam/bongh.park/outputs/models/qwen3-vl}"'
        ),
        ': "${WORLDMM_MEMORY_SHARD_TIMEOUT_SECONDS:=86400}"',
        ': "${WORLDMM_SPATIAL_TOKEN_BUDGET:=16}"',
        ': "${WORLDMM_SPATIAL_BYTE_BUDGET:=4096}"',
        ': "${WORLDMM_SPATIAL_QUANTIZATION_M:=0.25}"',
        ': "${WORLDMM_SPATIAL_SELECTOR_PATH:=}"',
        ': "${WORLDMM_SPATIAL_EXPERIMENT_CONFIG:=}"',
        (
            ': "${WORLDMM_TRITON_CACHE_ROOT:=${SLURM_TMPDIR:-/tmp}/'
            'worldmm-triton-${SLURM_JOB_ID}}"'
        ),
        "WORLDMM_REMOTE_NODES=10",
        "WORLDMM_GPUS_PER_NODE=8",
        'REMOTE_JOB_ID_OR_PROCESS_REF="slurm-${SLURM_JOB_ID}"',
        "mapfile -t worldmm_hosts < <(",
        '  /opt/slurm/bin/scontrol show hostnames "$SLURM_JOB_NODELIST"',
        ")",
        'MASTER_ADDR="${worldmm_hosts[0]}"',
        'MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"',
        ('WORLDMM_SLURM_STDOUT="$WORLDMM_OUTPUT_ROOT/logs/slurm-${SLURM_JOB_ID}.out"'),
        ('WORLDMM_SLURM_STDERR="$WORLDMM_OUTPUT_ROOT/logs/slurm-${SLURM_JOB_ID}.err"'),
        (
            "export MASTER_ADDR MASTER_PORT REMOTE_JOB_ID_OR_PROCESS_REF "
            "WORLDMM_REMOTE_NODES WORLDMM_GPUS_PER_NODE"
        ),
        (
            "export WORLDMM_REMOTE_REPO SMVQA_DATA_ROOT SMVQA_FRAME_ROOT "
            "GEMMA_MODEL_PATH WORLDMM_RUN_ID WORLDMM_OUTPUT_ROOT "
            "WORLDMM_MODEL_ID WORLDMM_MEMORY_MODEL_ID"
        ),
        (
            "export WORLDMM_MEMORY_MODEL_PATH WORLDMM_SLURM_STDOUT "
            "WORLDMM_SLURM_STDERR WORLDMM_MEMORY_SHARD_TIMEOUT_SECONDS"
        ),
        "export WORLDMM_SENSOR_FRAME_MANIFEST",
        (
            "export WORLDMM_SPATIAL_TOKEN_BUDGET WORLDMM_SPATIAL_BYTE_BUDGET "
            "WORLDMM_SPATIAL_QUANTIZATION_M WORLDMM_SPATIAL_SELECTOR_PATH "
            "WORLDMM_SPATIAL_EXPERIMENT_CONFIG WORLDMM_TRITON_CACHE_ROOT"
        ),
        (
            "export WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST=1 "
            "WORLDMM_SMVQA_REMOTE_APPROVED"
        ),
        "",
        (
            'mkdir -p "$WORLDMM_OUTPUT_ROOT"/'
            "{manifests,chunks,source_refs,memory,retrieval,qa,metrics,logs,models,"
            "summary,diagnostics,ablation}"
        ),
        'exec > >(tee -a "$WORLDMM_SLURM_STDOUT")',
        'exec 2> >(tee -a "$WORLDMM_SLURM_STDERR" >&2)',
        'cd "$WORLDMM_REMOTE_REPO"',
        'source "$WORLDMM_REMOTE_REPO/.venv/bin/activate"',
        "",
        "# stage 0: fetch Gemma 4 E2B model onto company storage",
        'if [ ! -e "$GEMMA_MODEL_PATH/config.json" ]; then',
        '  hf download "$WORLDMM_MODEL_ID" --local-dir "$GEMMA_MODEL_PATH"',
        "fi",
        'if [ ! -e "$WORLDMM_MEMORY_MODEL_PATH/config.json" ]; then',
        (
            '  hf download "$WORLDMM_MEMORY_MODEL_ID" --local-dir '
            '"$WORLDMM_MEMORY_MODEL_PATH"'
        ),
        "fi",
        "",
        "python - <<'PY' > \"$WORLDMM_OUTPUT_ROOT/summary/job.json\"",
        "import hashlib, importlib.metadata, json, os, subprocess",
        "from pathlib import Path",
        "def sha256(path):",
        "    digest = hashlib.sha256()",
        "    with path.open('rb') as stream:",
        "        for chunk in iter(lambda: stream.read(1024 * 1024), b''):",
        "            digest.update(chunk)",
        "    return digest.hexdigest()",
        'repo = Path(os.environ["WORLDMM_REMOTE_REPO"])',
        'data_root = Path(os.environ["SMVQA_DATA_ROOT"])',
        'model_root = Path(os.environ["GEMMA_MODEL_PATH"])',
        'spatial_config_value = os.environ["WORLDMM_SPATIAL_EXPERIMENT_CONFIG"]',
        "spatial_config = Path(spatial_config_value) if spatial_config_value else None",
        "packages = {}",
        "for name in ('torch', 'transformers', 'pydantic'):",
        "    try:",
        "        packages[name] = importlib.metadata.version(name)",
        "    except importlib.metadata.PackageNotFoundError:",
        "        packages[name] = None",
        "payload = {",
        '    "data_sha256": {',
        "        name: sha256(data_root / name)",
        "        for name in ('sources.jsonl', 'questions.jsonl', 'labels.jsonl')",
        "    },",
        '    "gpus_per_node": int(os.environ["WORLDMM_GPUS_PER_NODE"]),',
        '    "git_commit": subprocess.check_output(',
        "        ('git', 'rev-parse', 'HEAD'), cwd=repo, text=True",
        "    ).strip(),",
        '    "git_dirty": bool(subprocess.check_output(',
        "        ('git', 'status', '--porcelain'), cwd=repo, text=True",
        "    ).strip()),",
        '    "job_id": os.environ["REMOTE_JOB_ID_OR_PROCESS_REF"],',
        '    "master_addr": os.environ["MASTER_ADDR"],',
        '    "master_port": int(os.environ["MASTER_PORT"]),',
        '    "memory_model_path": os.environ["WORLDMM_MEMORY_MODEL_PATH"],',
        '    "spatial_quantization_m": float(',
        '        os.environ["WORLDMM_SPATIAL_QUANTIZATION_M"]',
        "    ),",
        '    "spatial_experiment_config": spatial_config_value,',
        '    "spatial_experiment_sha256": (',
        "        sha256(spatial_config) if spatial_config is not None else None",
        "    ),",
        '    "spatial_selector_path": os.environ["WORLDMM_SPATIAL_SELECTOR_PATH"],',
        '    "sensor_frame_manifest": os.environ["WORLDMM_SENSOR_FRAME_MANIFEST"],',
        '    "sensor_rate_hz": 1.0,',
        '    "spatial_token_budget": int(',
        '        os.environ["WORLDMM_SPATIAL_TOKEN_BUDGET"]',
        "    ),",
        '    "spatial_byte_budget": int(',
        '        os.environ["WORLDMM_SPATIAL_BYTE_BUDGET"]',
        "    ),",
        '    "triton_cache_root": os.environ["WORLDMM_TRITON_CACHE_ROOT"],',
        '    "model_config_sha256": sha256(model_root / "config.json"),',
        '    "model_path": str(model_root),',
        '    "nodes": int(os.environ["WORLDMM_REMOTE_NODES"]),',
        '    "output_root": os.environ["WORLDMM_OUTPUT_ROOT"],',
        '    "packages": packages,',
        '    "stderr": os.environ["WORLDMM_SLURM_STDERR"],',
        '    "stdout": os.environ["WORLDMM_SLURM_STDOUT"],',
        "}",
        "print(json.dumps(payload, indent=2, sort_keys=True))",
        "PY",
        (
            'printf "%s\\n" "$SLURM_JOB_ID" > '
            '"$WORLDMM_OUTPUT_ROOT/summary/slurm_job_id.txt"'
        ),
        "",
        "run_distributed_memory() {",
        '  local stores="$1"',
        '  local output_path="$2"',
        '  local input_path="${3:-}"',
        '  export WORLDMM_MEMORY_STORES="$stores"',
        '  export WORLDMM_MEMORY_OUTPUT="$output_path"',
        '  export WORLDMM_MEMORY_INPUT="$input_path"',
        "  /opt/slurm/bin/srun \\",
        '    --nodes="$WORLDMM_REMOTE_NODES" \\',
        '    --ntasks="$WORLDMM_REMOTE_NODES" \\',
        "    --ntasks-per-node=1 \\",
        '    --gpus-per-task="$WORLDMM_GPUS_PER_NODE" \\',
        "    --kill-on-bad-exit=1 \\",
        "    bash -c '",
        "      memory_args=(build-memory \\",
        '        --stores "$WORLDMM_MEMORY_STORES" \\',
        "        --config configs/remote.example.yaml \\",
        '        --fixture "$SMVQA_DATA_ROOT" \\',
        '        --out "$WORLDMM_MEMORY_OUTPUT" \\',
        "        --backend qwen)",
        '      if [ -n "$WORLDMM_MEMORY_INPUT" ]; then',
        '        memory_args+=(--input "$WORLDMM_MEMORY_INPUT")',
        "      fi",
        "      python -m torch.distributed.run \\",
        '        --nnodes "$WORLDMM_REMOTE_NODES" \\',
        '        --nproc-per-node "$WORLDMM_GPUS_PER_NODE" \\',
        '        --node-rank "$SLURM_NODEID" \\',
        '        --master-addr "$MASTER_ADDR" \\',
        '        --master-port "$MASTER_PORT" \\',
        '        -m worldmm_smvqa.cli "${memory_args[@]}"\'',
        "}",
        "",
        "run_distributed_qa() {",
        '  local evidence_path="$1"',
        '  local output_path="$2"',
        '  export WORLDMM_QA_EVIDENCE="$evidence_path"',
        '  export WORLDMM_QA_OUTPUT="$output_path"',
        "  # qa_transformers sets rank-local TRITON_CACHE_DIR under this root.",
        "  /opt/slurm/bin/srun \\",
        '    --nodes="$WORLDMM_REMOTE_NODES" \\',
        '    --ntasks="$WORLDMM_REMOTE_NODES" \\',
        "    --ntasks-per-node=1 \\",
        '    --gpus-per-task="$WORLDMM_GPUS_PER_NODE" \\',
        "    --kill-on-bad-exit=1 \\",
        "    bash -c 'python -m torch.distributed.run \\",
        '      --nnodes "$WORLDMM_REMOTE_NODES" \\',
        '      --nproc-per-node "$WORLDMM_GPUS_PER_NODE" \\',
        '      --node-rank "$SLURM_NODEID" \\',
        '      --master-addr "$MASTER_ADDR" \\',
        '      --master-port "$MASTER_PORT" \\',
        "      -m worldmm_smvqa.qa_transformers \\",
        '      --model "$GEMMA_MODEL_PATH" \\',
        '      --fixture "$SMVQA_DATA_ROOT" \\',
        '      --evidence "$WORLDMM_QA_EVIDENCE" \\',
        '      --out "$WORLDMM_QA_OUTPUT"\'',
        "}",
        "",
        "run_ablation_lane() {",
        '  local lane="$1"',
        '  local stores="$2"',
        '  local protocol="$3"',
        '  local lane_root="$WORLDMM_OUTPUT_ROOT/ablation/$lane"',
        '  mkdir -p "$lane_root"/{retrieval,qa,metrics}',
        "  worldmm-smvqa retrieve-batch \\",
        "    --config configs/remote.example.yaml \\",
        '    --fixture "$SMVQA_DATA_ROOT" \\',
        '    --stores "$stores" \\',
        '    --retrieval-protocol "$protocol" \\',
        "    --max-frame-refs 32 \\",
        '    --input "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json" \\',
        '    --out "$lane_root/retrieval/evidence_packs.jsonl"',
        '  run_distributed_qa "$lane_root/retrieval/evidence_packs.jsonl" \\',
        '    "$lane_root/qa/predictions.jsonl"',
        "  worldmm-smvqa evaluate \\",
        "    --config configs/remote.example.yaml \\",
        '    --pred "$lane_root/qa/predictions.jsonl" \\',
        '    --labels "$SMVQA_DATA_ROOT/labels.jsonl" \\',
        '    --out "$lane_root/metrics/official_metrics.json"',
        "}",
        "",
        "# stage 1: prepare source manifests and 1 Hz sensor-frame manifest",
        (
            'worldmm-smvqa validate-schema --input "$SMVQA_DATA_ROOT" '
            "--config configs/remote.example.yaml"
        ),
        (
            "worldmm-smvqa build-memory --stage sensor-frames "
            "--config configs/remote.example.yaml --fixture "
            '"$SMVQA_DATA_ROOT" --out "$WORLDMM_SENSOR_FRAME_MANIFEST"'
        ),
        (
            'printf "source_root=%s\\nsensor_frame_manifest=%s\\n" '
            '"$SMVQA_DATA_ROOT" "$WORLDMM_SENSOR_FRAME_MANIFEST" > '
            '"$WORLDMM_OUTPUT_ROOT/manifests/source_roots.txt"'
        ),
        (
            "python - <<'PY' > "
            '"$WORLDMM_OUTPUT_ROOT/manifests/spatial_experiment.json"'
        ),
        "import os",
        (
            "from worldmm_smvqa.worldmm.spatial_compression import "
            "resolve_spatial_experiment_config"
        ),
        "config = resolve_spatial_experiment_config(os.environ)",
        "print(config.model_dump_json(indent=2))",
        "PY",
        ("python - <<'PY' > \"$WORLDMM_OUTPUT_ROOT/manifests/question_ids.txt\""),
        "import json, os",
        "from pathlib import Path",
        'labels = Path(os.environ["SMVQA_DATA_ROOT"]) / "labels.jsonl"',
        "with labels.open(encoding='utf-8') as source:",
        "    for line in source:",
        "        print(json.loads(line)['question_id'])",
        "PY",
        "",
        "# stage 2: build 30s/30m chunks",
        (
            "worldmm-smvqa build-memory --stage chunk "
            "--config configs/remote.example.yaml --fixture "
            '"$SMVQA_DATA_ROOT" --out '
            '"$WORLDMM_OUTPUT_ROOT/chunks/source_chunks.jsonl"'
        ),
        "",
        "# stage 3: generate/load captions OCR object frame refs",
        (
            "worldmm-smvqa build-memory --stage source-memories "
            "--config configs/remote.example.yaml --fixture "
            '"$SMVQA_DATA_ROOT" --out '
            '"$WORLDMM_OUTPUT_ROOT/source_refs/source_memories.jsonl"'
        ),
        "",
        "# stage 4: build WorldMM stores with the Qwen memory constructor",
        'run_distributed_memory "episodic" \\',
        '  "$WORLDMM_OUTPUT_ROOT/memory/episodic.jsonl"',
        'run_distributed_memory "semantic,visual" \\',
        '  "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv" \\',
        '  "$WORLDMM_OUTPUT_ROOT/memory/episodic.jsonl"',
        'run_distributed_memory "spatial" \\',
        '  "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv"',
        "python - <<'PY' > \"$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json\"",
        "import json, os",
        "from pathlib import Path",
        'root = Path(os.environ["WORLDMM_OUTPUT_ROOT"])',
        "def line_count(path):",
        "    if not path.exists():",
        "        return 0",
        "    return sum(1 for _line in path.open(encoding='utf-8'))",
        'spatial = root / "memory/worldmm_sv/spatial.jsonl"',
        "payload = {",
        '    "source_memories": str(root / "source_refs/source_memories.jsonl"),',
        '    "sensor_frame_manifest": os.environ["WORLDMM_SENSOR_FRAME_MANIFEST"],',
        '    "sensor_rate_hz": 1.0,',
        '    "episodic_memory": str(root / "memory/episodic.jsonl"),',
        '    "semantic_memory": str(root / "memory/worldmm_sv/semantic.jsonl"),',
        '    "visual_memory": str(root / "memory/worldmm_sv/visual.jsonl"),',
        '    "spatial_memory": {"path": str(spatial), "count": line_count(spatial)},',
        '    "spatial_compression": str(',
        '        root / "memory/worldmm_sv/spatial.stats.jsonl"',
        "    ),",
        '    "spatial_experiment": str(',
        '        root / "manifests/spatial_experiment.json"',
        "    ),",
        "}",
        "print(json.dumps(payload, sort_keys=True))",
        "PY",
        "",
        "# stage 5: retrieve all QA under causal cutoff",
        "worldmm-smvqa retrieve-batch \\",
        "  --config configs/remote.example.yaml \\",
        '  --fixture "$SMVQA_DATA_ROOT" \\',
        "  --stores episodic,semantic,visual,spatial \\",
        "  --retrieval-protocol worldmm-smvqa \\",
        "  --max-frame-refs 32 \\",
        '  --input "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json" \\',
        '  --out "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl"',
        "",
        "# stage 6: run Gemma 4 E2B QA",
        'run_distributed_qa "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl" \\',
        '  "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl"',
        "",
        "# stage 7: evaluate official metrics",
        (
            "worldmm-smvqa evaluate --config configs/remote.example.yaml "
            '--pred "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl" '
            '--labels "$SMVQA_DATA_ROOT/labels.jsonl" --out '
            '"$WORLDMM_OUTPUT_ROOT/metrics/official_metrics.json"'
        ),
        "worldmm-smvqa diagnose-spatial \\",
        "  --config configs/remote.example.yaml \\",
        '  --input "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl" \\',
        '  --labels "$SMVQA_DATA_ROOT/labels.jsonl" \\',
        '  --out "$WORLDMM_OUTPUT_ROOT/diagnostics/spatial_diagnostics.json"',
        "",
        "# stage 8: run ablation lanes",
        'run_ablation_lane "without_spatial" "episodic,semantic,visual" \\',
        '  "worldmm-smvqa"',
        (
            'run_ablation_lane "protocol_legacy_round_robin" '
            '"episodic,semantic,visual,spatial" \\'
        ),
        '  "legacy-round-robin"',
        "",
        "# stage 9: write summary",
        'cat > "$WORLDMM_OUTPUT_ROOT/summary/summary.txt" <<EOF',
        "remote_job_reference=$REMOTE_JOB_ID_OR_PROCESS_REF",
        "output_root=$WORLDMM_OUTPUT_ROOT",
        "stdout=$WORLDMM_SLURM_STDOUT",
        "stderr=$WORLDMM_SLURM_STDERR",
        "metrics=$WORLDMM_OUTPUT_ROOT/metrics/official_metrics.json",
        "sensor_frame_manifest=$WORLDMM_SENSOR_FRAME_MANIFEST",
        "sensor_rate_hz=1.0",
        "spatial_compression=$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv/spatial.stats.jsonl",
        (
            "ablation_without_spatial=$WORLDMM_OUTPUT_ROOT/ablation/"
            "without_spatial/metrics/official_metrics.json"
        ),
        (
            "ablation_protocol_legacy=$WORLDMM_OUTPUT_ROOT/ablation/"
            "protocol_legacy_round_robin/metrics/official_metrics.json"
        ),
        "EOF",
        "python - <<'PY' > \"$WORLDMM_OUTPUT_ROOT/summary/remote_manifest.json\"",
        "import json, os",
        "from pathlib import Path",
        'root = Path(os.environ["WORLDMM_OUTPUT_ROOT"])',
        "metrics = json.loads(",
        "    (root / 'metrics/official_metrics.json').read_text(encoding='utf-8')",
        ")",
        "spatial_stats = [",
        "    json.loads(line)",
        "    for line in (root / 'memory/worldmm_sv/spatial.stats.jsonl')",
        "    .read_text(encoding='utf-8')",
        "    .splitlines()",
        "    if line.strip()",
        "]",
        "spatial_raw_bytes = sum(row['raw_bytes'] for row in spatial_stats)",
        "spatial_compressed_bytes = sum(",
        "    row['compressed_bytes'] for row in spatial_stats",
        ")",
        "spatial_token_count = sum(row['token_count'] for row in spatial_stats)",
        "spatial_compression_ratio = (",
        "    spatial_raw_bytes / spatial_compressed_bytes",
        "    if spatial_compressed_bytes",
        "    else 0.0",
        ")",
        "payload = {",
        '    "baseline_name": "WorldMM-SMVQA",',
        '    "remote_status": "complete",',
        '    "local_changes": [',
        '        "distributed memory generation",',
        '        "shared 1 Hz sensor-frame manifest",',
        '        "causal compressed spatial memory",',
        '        "batch causal retrieval",',
        '        "resumable distributed Gemma QA",',
        '        "spatial retrieval diagnostics and ablations",',
        "    ],",
        '    "remote_command": (',
        '        "/opt/slurm/bin/sbatch remote-plan/run_worldmm_smvqa.sh"',
        "    ),",
        '    "remote_job_reference": os.environ["REMOTE_JOB_ID_OR_PROCESS_REF"],',
        '    "remote_artifact_path": str(root),',
        '    "metrics": [',
        "        *(",
        "            {'name': name, 'value': metrics[name]}",
        "            for name in ('Ans-F1', 'QA-Acc', 'QA-MRR')",
        "        ),",
        "        {'name': 'Spatial-Raw-Bytes', 'value': spatial_raw_bytes},",
        (
            "        {'name': 'Spatial-Compressed-Bytes', "
            "'value': spatial_compressed_bytes},"
        ),
        (
            "        {'name': 'Spatial-Compression-Ratio', "
            "'value': spatial_compression_ratio},"
        ),
        (
            "        {'name': 'Spatial-Selected-Record-Count', "
            "'value': spatial_token_count},"
        ),
        "    ],",
        '    "not_copied_locally": [',
        '        "full datasets",',
        '        "model weights",',
        '        "checkpoints",',
        '        "raw model outputs",',
        "    ],",
        "}",
        "print(json.dumps(payload, indent=2, sort_keys=True))",
        "PY",
        "worldmm-smvqa report \\",
        "  --config configs/remote.example.yaml \\",
        '  --run-manifest "$WORLDMM_OUTPUT_ROOT/summary/remote_manifest.json" \\',
        '  --out "$WORLDMM_OUTPUT_ROOT/summary/final_report.md"',
        "",
    ]
    return "\n".join(stage_lines)


def dag_submit_script_text() -> str:
    """Render a head-node submitter for the staged CPU/GPU pipeline."""
    return dedent(
        r"""
        #!/usr/bin/env bash
        set -euo pipefail

        if [ "${WORLDMM_SMVQA_REMOTE_APPROVED:-}" != "1" ]; then
          printf "WORLDMM_SMVQA_REMOTE_APPROVED=1 is required\n" >&2
          exit 1
        fi
        export WORLDMM_SMVQA_REMOTE_APPROVED

        SBATCH="${WORLDMM_SBATCH:-/opt/slurm/bin/sbatch}"
        WORLDMM_REMOTE_REPO="${WORLDMM_REMOTE_REPO:-/repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b}"
        : "${WORLDMM_RUN_ID:=dag-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
        if [ -f "$WORLDMM_REMOTE_REPO/.env.worldmm" ]; then
          set -a
          source "$WORLDMM_REMOTE_REPO/.env.worldmm"
          set +a
        fi
        : "${WORLDMM_OUTPUT_ROOT:=/repo/VTteam/bongh.park/outputs/${WORLDMM_RUN_ID}}"
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

        # GPU stages default to the full company allocation. Override *_NODES
        # for an explicitly approved bounded probe.
        : "${WORLDMM_REMOTE_NODES:=10}"
        : "${WORLDMM_GPUS_PER_NODE:=8}"
        : "${WORLDMM_PREFLIGHT_NODES:=1}"
        : "${WORLDMM_PREFLIGHT_CPUS:=32}"
        : "${WORLDMM_PREFLIGHT_MEM:=128G}"
        : "${WORLDMM_PREFLIGHT_TIME:=02:00:00}"
        : "${WORLDMM_TEACHER_NODES:=$WORLDMM_REMOTE_NODES}"
        : "${WORLDMM_TEACHER_GPUS_PER_NODE:=$WORLDMM_GPUS_PER_NODE}"
        : "${WORLDMM_TEACHER_CPUS:=64}"
        : "${WORLDMM_TEACHER_MEM:=0}"
        : "${WORLDMM_TEACHER_TIME:=12:00:00}"
        : "${WORLDMM_UTILITY_NODES:=1}"
        : "${WORLDMM_UTILITY_CPUS:=64}"
        : "${WORLDMM_UTILITY_MEM:=256G}"
        : "${WORLDMM_UTILITY_TIME:=06:00:00}"
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
        : "${WORLDMM_REPORT_NODES:=1}"
        : "${WORLDMM_REPORT_CPUS:=32}"
        : "${WORLDMM_REPORT_MEM:=128G}"
        : "${WORLDMM_REPORT_TIME:=02:00:00}"

        export WORLDMM_REMOTE_REPO WORLDMM_RUN_ID WORLDMM_OUTPUT_ROOT
        export WORLDMM_TRAIN_EPOCHS WORLDMM_TRAIN_BATCH_SIZE
        export WORLDMM_TRAIN_HIDDEN_DIM WORLDMM_TRAIN_LEARNING_RATE
        export WORLDMM_TRAIN_RESUME
        stage_script="$WORLDMM_REMOTE_REPO/remote-plan/run_worldmm_smvqa_stage.sh"
        mkdir -p "$WORLDMM_OUTPUT_ROOT"/{logs,summary}
        submit_lock="$WORLDMM_OUTPUT_ROOT/summary/dag_submit.lock"
        if ! (set -o noclobber; printf "%s\n" "$$" > "$submit_lock") 2>/dev/null; then
          printf "DAG already submitted or submitting: %s\n" "$submit_lock" >&2
          exit 1
        fi
        partial_jobs="$WORLDMM_OUTPUT_ROOT/summary/dag_jobs.partial"
        : > "$partial_jobs"

        submit_stage() {
          local stage="$1" partition="$2" nodes="$3" gpus="$4"
          local cpus="$5" mem="$6" time_limit="$7" dependency="$8"
          local raw_job_id job_id
          local -a args=(
            "$SBATCH" --parsable
            "--job-name=worldmm-${stage}"
            "--partition=$partition"
            "--nodes=$nodes"
            --ntasks-per-node=1
            "--cpus-per-task=$cpus"
            "--mem=$mem"
            "--time=$time_limit"
            "--output=$WORLDMM_OUTPUT_ROOT/logs/${stage}-%j.out"
            "--error=$WORLDMM_OUTPUT_ROOT/logs/${stage}-%j.err"
            "--export=ALL,WORLDMM_STAGE=$stage,WORLDMM_STAGE_GPUS_PER_NODE=$gpus"
          )
          if [ "$gpus" -gt 0 ]; then
            args+=("--gpus-per-node=$gpus")
          fi
          if [ -n "$dependency" ]; then
            args+=("--dependency=afterok:$dependency")
          fi
          raw_job_id="$("${args[@]}" "$stage_script")"
          job_id="${raw_job_id%%;*}"
          if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
            printf "invalid sbatch job id for %s: %s\n" "$stage" "$raw_job_id" >&2
            return 1
          fi
          printf "%s=%s\n" "$stage" "$job_id" >> "$partial_jobs"
          printf "%s" "$job_id"
        }

        preflight_job="$(
          submit_stage preflight_ingest "$WORLDMM_CPU_PARTITION" \
            "$WORLDMM_PREFLIGHT_NODES" 0 "$WORLDMM_PREFLIGHT_CPUS" \
            "$WORLDMM_PREFLIGHT_MEM" "$WORLDMM_PREFLIGHT_TIME" ""
        )"
        teacher_job="$(
          submit_stage teacher_extract "$WORLDMM_GPU_PARTITION" \
            "$WORLDMM_TEACHER_NODES" "$WORLDMM_TEACHER_GPUS_PER_NODE" \
            "$WORLDMM_TEACHER_CPUS" "$WORLDMM_TEACHER_MEM" \
            "$WORLDMM_TEACHER_TIME" "$preflight_job"
        )"
        utility_job="$(
          submit_stage merge_utility "$WORLDMM_CPU_PARTITION" \
            "$WORLDMM_UTILITY_NODES" 0 "$WORLDMM_UTILITY_CPUS" \
            "$WORLDMM_UTILITY_MEM" "$WORLDMM_UTILITY_TIME" "$teacher_job"
        )"
        train_job="$(
          submit_stage train_qa "$WORLDMM_GPU_PARTITION" \
            "$WORLDMM_TRAIN_NODES" "$WORLDMM_TRAIN_GPUS_PER_NODE" \
            "$WORLDMM_TRAIN_CPUS" "$WORLDMM_TRAIN_MEM" \
            "$WORLDMM_TRAIN_TIME" "$utility_job"
        )"
        report_job="$(
          submit_stage metrics_report "$WORLDMM_CPU_PARTITION" \
            "$WORLDMM_REPORT_NODES" 0 "$WORLDMM_REPORT_CPUS" \
            "$WORLDMM_REPORT_MEM" "$WORLDMM_REPORT_TIME" "$train_job"
        )"

        jobs_file="$WORLDMM_OUTPUT_ROOT/summary/dag_jobs.env"
        temporary="${jobs_file}.$$"
        printf "%s\n" \
          "WORLDMM_RUN_ID=$WORLDMM_RUN_ID" \
          "WORLDMM_OUTPUT_ROOT=$WORLDMM_OUTPUT_ROOT" \
          "PREFLIGHT_JOB_ID=$preflight_job" \
          "TEACHER_JOB_ID=$teacher_job" \
          "UTILITY_JOB_ID=$utility_job" \
          "TRAIN_QA_JOB_ID=$train_job" \
          "REPORT_JOB_ID=$report_job" > "$temporary"
        mv "$temporary" "$jobs_file"
        printf "run_id=%s output_root=%s final_job_id=%s\n" \
          "$WORLDMM_RUN_ID" "$WORLDMM_OUTPUT_ROOT" "$report_job"
        """,
    ).lstrip()


def dag_stage_script_text() -> str:
    """Render the stage runner shared by all DAG allocations."""
    return dedent(
        r"""
        #!/usr/bin/env bash
        set -euo pipefail

        : "${SLURM_JOB_ID:?submit this script with /opt/slurm/bin/sbatch}"
        : "${WORLDMM_STAGE:?WORLDMM_STAGE is required}"
        : "${WORLDMM_OUTPUT_ROOT:?WORLDMM_OUTPUT_ROOT is required}"
        : "${WORLDMM_RUN_ID:?WORLDMM_RUN_ID is required}"
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
        : "${WORLDMM_STAGE_GPUS_PER_NODE:=0}"
        : "${WORLDMM_TRAIN_EPOCHS:=1}"
        : "${WORLDMM_TRAIN_BATCH_SIZE:=8}"
        : "${WORLDMM_TRAIN_HIDDEN_DIM:=32}"
        : "${WORLDMM_TRAIN_LEARNING_RATE:=0.001}"
        : "${WORLDMM_TRAIN_RESUME:=}"
        export SMVQA_DATA_ROOT SMVQA_FRAME_ROOT GEMMA_MODEL_PATH
        export WORLDMM_TRAIN_EPOCHS WORLDMM_TRAIN_BATCH_SIZE
        export WORLDMM_TRAIN_HIDDEN_DIM WORLDMM_TRAIN_LEARNING_RATE
        export WORLDMM_TRAIN_RESUME WORLDMM_SMVQA_REMOTE_APPROVED
        export WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST=1
        cd "$WORLDMM_REMOTE_REPO"
        source "$WORLDMM_REMOTE_REPO/.venv/bin/activate"
        mkdir -p "$WORLDMM_OUTPUT_ROOT"/\
          {manifests,teacher,training,checkpoints,retrieval,qa,metrics,diagnostics,summary}
        WORLDMM_SENSOR_FRAME_MANIFEST=\
          "$WORLDMM_OUTPUT_ROOT/manifests/sensor_frames.jsonl"
        export WORLDMM_SENSOR_FRAME_MANIFEST

        distributed_train() {
          mapfile -t hosts < <(
            /opt/slurm/bin/scontrol show hostnames "$SLURM_JOB_NODELIST"
          )
          export MASTER_ADDR="${hosts[0]}"
          export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
          export WORLDMM_TEACHER_CACHE=\
            "$WORLDMM_OUTPUT_ROOT/training/student_teacher_cache.jsonl"
          export WORLDMM_CHECKPOINT=\
            "$WORLDMM_OUTPUT_ROOT/checkpoints/spatial_student.pt"
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

        distributed_qa() {
          mapfile -t hosts < <(
            /opt/slurm/bin/scontrol show hostnames "$SLURM_JOB_NODELIST"
          )
          export MASTER_ADDR="${hosts[0]}"
          export MASTER_PORT="$((30000 + SLURM_JOB_ID % 20000))"
          export WORLDMM_QA_EVIDENCE=\
            "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl"
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
                --fixture "$SMVQA_DATA_ROOT" \
                --evidence "$WORLDMM_QA_EVIDENCE" \
                --out "$WORLDMM_QA_OUTPUT"'
        }

        case "$WORLDMM_STAGE" in
          preflight_ingest)
            worldmm-smvqa preflight \
              --fixture "$SMVQA_DATA_ROOT" \
              --out "$WORLDMM_OUTPUT_ROOT/diagnostics/preflight.json"
            worldmm-smvqa build-memory --stage sensor-frames \
              --config configs/remote.example.yaml \
              --fixture "$SMVQA_DATA_ROOT" \
              --out "$WORLDMM_OUTPUT_ROOT/manifests/sensor_frames.jsonl"
            worldmm-smvqa build-memory --stage chunk \
              --config configs/remote.example.yaml \
              --fixture "$SMVQA_DATA_ROOT" \
              --out "$WORLDMM_OUTPUT_ROOT/manifests/source_chunks.jsonl"
            worldmm-smvqa build-memory --stage source-memories \
              --config configs/remote.example.yaml \
              --fixture "$SMVQA_DATA_ROOT" \
              --out "$WORLDMM_OUTPUT_ROOT/manifests/source_memories.jsonl"
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
              export WORLDMM_GCUT3R_EXTRACTOR teacher_shard_root
              export teacher_world_size SMVQA_DATA_ROOT
              /opt/slurm/bin/srun \
                --nodes="$SLURM_NNODES" \
                --ntasks="$teacher_world_size" \
                --ntasks-per-node="$WORLDMM_STAGE_GPUS_PER_NODE" \
                --cpus-per-task="$teacher_cpus_per_worker" \
                --gpus-per-task=1 \
                --gpu-bind=single:1 \
                --kill-on-bad-exit=1 \
                bash -c '
                  printf -v rank_id "%05d" "$SLURM_PROCID"
                  exec "$WORLDMM_GCUT3R_EXTRACTOR" \
                    --fixture "$SMVQA_DATA_ROOT" \
                    --sensor-frame-manifest "$WORLDMM_SENSOR_FRAME_MANIFEST" \
                    --rank "$SLURM_PROCID" \
                    --world-size "$teacher_world_size" \
                    --out "$teacher_shard_root/rank-${rank_id}.jsonl"'
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
          merge_utility)
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
            : "${WORLDMM_STUDENT_SUPERVISION_INPUT:?student supervision required}"
            python -m worldmm_smvqa.teacher_materializer \
              --teacher-cache "$WORLDMM_OUTPUT_ROOT/teacher/cache.jsonl" \
              --supervision "$WORLDMM_STUDENT_SUPERVISION_INPUT" \
              --out "$WORLDMM_OUTPUT_ROOT/training/student_teacher_cache.jsonl"
            : "${WORLDMM_UTILITY_CACHE_INPUT:?counterfactual utility cache is required}"
            : "${WORLDMM_SPLIT_MANIFEST_INPUT:?split manifest is required}"
            python -m worldmm_smvqa.spatial_selector_train prepare \
              --fixture "$SMVQA_DATA_ROOT" \
              --out "$WORLDMM_OUTPUT_ROOT/training/selector_rows.jsonl" \
              --utility-cache "$WORLDMM_UTILITY_CACHE_INPUT" \
              --split-manifest "$WORLDMM_SPLIT_MANIFEST_INPUT" \
              --supervision-mode counterfactual
            python -m worldmm_smvqa.spatial_selector_train train \
              --config configs/remote.example.yaml \
              --input "$WORLDMM_OUTPUT_ROOT/training/selector_rows.jsonl" \
              --out "$WORLDMM_OUTPUT_ROOT/training/selector.json"
            ;;
          train_qa)
            distributed_train
            : "${WORLDMM_QA_EVIDENCE_INPUT:?student-backed evidence packs are required}"
            ln -sfn "$WORLDMM_QA_EVIDENCE_INPUT" \
              "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl"
            distributed_qa
            ;;
          metrics_report)
            worldmm-smvqa evaluate \
              --config configs/remote.example.yaml \
              --pred "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl" \
              --labels "$SMVQA_DATA_ROOT/labels.jsonl" \
              --out "$WORLDMM_OUTPUT_ROOT/metrics/official_metrics.json"
            printf "%s\n" \
              "run_id=$WORLDMM_RUN_ID" \
              "output_root=$WORLDMM_OUTPUT_ROOT" \
              "metrics=$WORLDMM_OUTPUT_ROOT/metrics/official_metrics.json" \
              > "$WORLDMM_OUTPUT_ROOT/summary/summary.txt"
            ;;
          *)
            printf "unknown WORLDMM_STAGE: %s\n" "$WORLDMM_STAGE" >&2
            exit 2
            ;;
        esac
        """,
    ).lstrip()
