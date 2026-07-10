from __future__ import annotations


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
            "export WORLDMM_MEMORY_MODEL_PATH WORLDMM_SLURM_STDOUT "
            "WORLDMM_SLURM_STDERR WORLDMM_MEMORY_SHARD_TIMEOUT_SECONDS"
        ),
        "export WORLDMM_SENSOR_FRAME_MANIFEST",
        (
            "export WORLDMM_SPATIAL_TOKEN_BUDGET "
            "WORLDMM_SPATIAL_QUANTIZATION_M WORLDMM_SPATIAL_SELECTOR_PATH "
            "WORLDMM_SPATIAL_EXPERIMENT_CONFIG WORLDMM_TRITON_CACHE_ROOT"
        ),
        "export WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST=1",
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
