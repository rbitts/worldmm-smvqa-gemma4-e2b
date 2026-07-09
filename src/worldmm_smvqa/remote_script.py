from __future__ import annotations


def script_text() -> str:
    without_spatial_flags = _retrieve_flags(
        "episodic,semantic,visual",
        "worldmm-smvqa",
    )
    legacy_protocol_flags = _retrieve_flags(
        "episodic,semantic,visual,spatial",
        "legacy-round-robin",
    )
    without_spatial_flags_line = f'  "retrieve_flags": "{without_spatial_flags}",'
    legacy_protocol_flags_line = f'  "retrieve_flags": "{legacy_protocol_flags}",'
    stage_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        ': "${WORLDMM_REMOTE_REPO:?WORLDMM_REMOTE_REPO is required}"',
        ': "${SMVQA_DATA_ROOT:?SMVQA_DATA_ROOT is required}"',
        ': "${SMVQA_FRAME_ROOT:=$SMVQA_DATA_ROOT/frames}"',
        ': "${GEMMA_MODEL_PATH:?GEMMA_MODEL_PATH is required}"',
        ': "${WORLDMM_OUTPUT_ROOT:?WORLDMM_OUTPUT_ROOT is required}"',
        ': "${WORLDMM_REMOTE_NODES:=1}"',
        ': "${WORLDMM_GPUS_PER_NODE:=8}"',
        ': "${WORLDMM_DDP_LAUNCHER:=python -m torch.distributed.run}"',
        ': "${WORLDMM_MODEL_ID:=google/gemma-4-E2B-it}"',
        "export WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST=1",
        'cd "$WORLDMM_REMOTE_REPO"',
        'read -r -a worldmm_ddp_launcher_argv <<< "$WORLDMM_DDP_LAUNCHER"',
        "",
        (
            'mkdir -p "$WORLDMM_OUTPUT_ROOT"/'
            "{manifests,chunks,source_refs,memory,retrieval,qa,metrics,logs,"
            "summary,diagnostics,ablation}"
        ),
        "",
        "# stage 0: fetch Gemma 4 E2B model onto company storage",
        'if [ ! -e "$GEMMA_MODEL_PATH/config.json" ]; then',
        '  hf download "$WORLDMM_MODEL_ID" --local-dir "$GEMMA_MODEL_PATH"',
        "fi",
        "",
        "# stage 1: prepare source manifests",
        (
            'worldmm-smvqa validate-schema --input "$SMVQA_DATA_ROOT" '
            "--config configs/remote.example.yaml"
        ),
        (
            'printf "%s\\n" "$SMVQA_DATA_ROOT" > '
            '"$WORLDMM_OUTPUT_ROOT/manifests/source_roots.txt"'
        ),
        (
            "python - <<'PY' > "
            '"$WORLDMM_OUTPUT_ROOT/manifests/question_ids.txt"'
        ),
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
        "# stage 4: build WorldMM episodic semantic visual spatial stores",
        (
            "worldmm-smvqa build-memory --store episodic "
            "--config configs/remote.example.yaml --fixture "
            '"$SMVQA_DATA_ROOT" --out '
            '"$WORLDMM_OUTPUT_ROOT/memory/episodic.jsonl"'
        ),
        (
            "worldmm-smvqa build-memory --stores semantic,visual,spatial "
            "--config configs/remote.example.yaml --fixture "
            '"$SMVQA_DATA_ROOT" --out "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv"'
        ),
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
        '    "episodic_memory": str(root / "memory/episodic.jsonl"),',
        '    "semantic_memory": str(root / "memory/worldmm_sv/semantic.jsonl"),',
        '    "visual_memory": str(root / "memory/worldmm_sv/visual.jsonl"),',
        '    "spatial_memory": {"path": str(spatial), "count": line_count(spatial)},',
        "}",
        "print(json.dumps(payload, sort_keys=True))",
        "PY",
        "",
        "# stage 5: retrieve per QA under causal cutoff",
        ': > "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl"',
        "while IFS= read -r question_id; do",
        "safe_question_file=$(python - \"$question_id\" <<'PY'",
        "import hashlib, sys",
        "print('q_' + hashlib.sha256(sys.argv[1].encode('utf-8')).hexdigest()[:16])",
        "PY",
        ")",
        'tmp="$WORLDMM_OUTPUT_ROOT/retrieval/${safe_question_file}.json"',
        (
            "worldmm-smvqa retrieve --config configs/remote.example.yaml "
            '--fixture "$SMVQA_DATA_ROOT" --stores episodic,semantic,visual,spatial '
            "--retrieval-protocol worldmm-smvqa --max-frame-refs 32 "
            '--question "$question_id" --out "$tmp"'
        ),
        (
            'python - "$tmp" '
            "\"$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl\" <<'PY'"
        ),
        "import json, sys",
        "from pathlib import Path",
        "payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))",
        "with Path(sys.argv[2]).open('a', encoding='utf-8') as target:",
        "    target.write(json.dumps(payload, separators=(',', ':')) + '\\n')",
        "PY",
        'done < "$WORLDMM_OUTPUT_ROOT/manifests/question_ids.txt"',
        "",
        "# stage 6: run Gemma 4 E2B QA",
        (
            'rm -f "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl" '
            '"$WORLDMM_OUTPUT_ROOT/qa/predictions".rank*-of*.jsonl'
        ),
        (
            '"${worldmm_ddp_launcher_argv[@]}" --nnodes "$WORLDMM_REMOTE_NODES" '
            '--nproc-per-node "$WORLDMM_GPUS_PER_NODE" '
            "-m worldmm_smvqa.qa_transformers --model "
            '"$GEMMA_MODEL_PATH" --fixture "$SMVQA_DATA_ROOT" --evidence '
            '"$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl" --out '
            '"$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl"'
        ),
        "",
        "# stage 7: evaluate official metrics",
        (
            "worldmm-smvqa evaluate --config configs/remote.example.yaml "
            '--pred "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl" '
            '--labels "$SMVQA_DATA_ROOT/labels.jsonl" --out '
            '"$WORLDMM_OUTPUT_ROOT/metrics/official_metrics.json"'
        ),
        "",
        "# stage 8: write summary",
        (
            'printf "remote_job_reference='
            '${REMOTE_JOB_ID_OR_PROCESS_REF:-manual-process}\\n'
            'metrics=Ans-F1,QA-Acc,QA-MRR\\n" > '
            '"$WORLDMM_OUTPUT_ROOT/summary/summary.txt"'
        ),
        "",
        "# stage 9: document equivalent ablation reruns under output root",
        'cat > "$WORLDMM_OUTPUT_ROOT/ablation/without_spatial.json" <<EOF',
        "{",
        '  "rerun": "retrieve/qa/evaluate with --stores episodic,semantic,visual",',
        without_spatial_flags_line,
        '  "output_root": "$WORLDMM_OUTPUT_ROOT/ablation/without_spatial"',
        "}",
        "EOF",
        'cat > "$WORLDMM_OUTPUT_ROOT/ablation/protocol_legacy_round_robin.json" <<EOF',
        "{",
        '  "rerun": "retrieve/qa/evaluate with legacy round-robin protocol",',
        legacy_protocol_flags_line,
        '  "output_root": "$WORLDMM_OUTPUT_ROOT/ablation/protocol_legacy_round_robin"',
        "}",
        "EOF",
        "",
    ]
    return "\n".join(stage_lines)


def _retrieve_flags(stores: str, protocol: str) -> str:
    return f"--stores {stores} --retrieval-protocol {protocol} --max-frame-refs 32"
