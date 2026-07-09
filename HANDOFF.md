# WorldMM-SMVQA Handoff

## Status

WorldMM-SMVQA implementation is complete locally.

Implemented pipeline:

```text
SuperMemory-VQA stream
-> 30s clips / 30m shards
-> caption / OCR / object / frame memories
-> WorldMM episodic / semantic / visual / spatial memories
-> protocol-aware causal retrieval before question_time
-> Gemma 4 E2B QA interface with retrieved memory text + sampled video frames
-> Ans-F1 / QA-Acc / QA-MRR report
```

All plan items are checked in `.omo/plans/worldmm-smvqa-gemma4-e2b.md`.
Orchestration state is completed in `.omo/boulder.json`.

## What Exists

- CLI package: `src/worldmm_smvqa/`
- Local/remote configs: `configs/local.example.yaml`, `configs/remote.example.yaml`
- Remote wrapper: `scripts/remote/run_worldmm_smvqa.sh`
- Tiny fixture: `tests/fixtures/tiny_smvqa/`
- Main docs: `README.md`
- Evidence root: `.omo/evidence/worldmm-smvqa/`

## Spatial Memory (v2)

The v2 local path adds an explicit fourth WorldMM memory store: `spatial`.
Spatial records are built from source-stream inputs only:

- optional `pose_samples`: `timestamp`, `x`, `y`, `z`, optional `yaw`
- optional `gaze_samples`: `timestamp`, `x`, `y`, `z`
- object detections and frame metadata already present in the fixture schema

`$SMVQA_DATA_ROOT` schema now includes optional `pose_samples` and
`gaze_samples` on `sources.jsonl`. Empty fields are valid; when provided, they
must be timestamp-sorted and use the same coordinate convention as local tests:
`x`/`y` horizontal plane, `z` vertical, meters.

The store writes deterministic zone, anchor, near-relation, object-state, and
trajectory snippets. It is suitable for the current lexical retrieval path and
does not require 3D reconstruction, model downloads, or local GPU work.

## Retrieval Contract

This is a WorldMM-augmented Video-RAG/EgoButler-compatible retrieval path, not
an exact WorldMM agentic/PPR implementation.

Contract:

- Video-RAG eligibility: retrieve only from same-video `shard_30m` chunks with
  `end_time <= question_time`.
- EgoButler hierarchy: select coarse-to-fine through
  `shard_30m -> clip_30s -> memory records`.
- WorldMM policy route: order enabled stores deterministically from question
  text, with location questions routing `spatial` first.
- Evidence trace: every evidence pack carries `retrieval_trace` with eligible
  shard ids, selected clip ids, policy route, store order, candidate counts,
  causal-filter count, and frame-ref count.
- Frame cap: remote and local reruns for this lane must use
  `--max-frame-refs 32`.
- QA parity: from the selected pre-question shard, sample up to 32 frames
  uniformly and pass those frame files to Gemma together with retrieved memory
  snippets.

Remote rerun must use:

```bash
--retrieval-protocol worldmm-smvqa --max-frame-refs 32
```

Smoke now also writes `spatial_diagnostics.json` and `ablation.json` when
ablation flags are provided. Use `--ablation-stores episodic,semantic,visual`
for the without-spatial pass and `--ablation-protocol legacy-round-robin` for
the protocol-only pass.

Primary commands:

```bash
uv run worldmm-smvqa smoke --fixture tests/fixtures/tiny_smvqa --out .omo/evidence/worldmm-smvqa/final-smoke
uv run worldmm-smvqa launch-remote --dry-run --config configs/remote.example.yaml --out .omo/evidence/worldmm-smvqa/remote-plan
uv run worldmm-smvqa report --run-manifest tests/fixtures/tiny_smvqa/remote_manifest.example.json --out .omo/evidence/worldmm-smvqa/report.md
```

## Verified Locally

Final gates passed:

- `pytest`: 118 passed
- `ruff`: passed
- `basedpyright`: 0 errors, 0 warnings
- local smoke: writes `metrics.json`, `predictions.jsonl`, `evidence_packs.jsonl`, `memory_manifest.json`
- spatial smoke: writes `spatial_memory.jsonl` and `spatial_diagnostics.json`
- review-work final rerun: goal, QA, code quality, security, context all PASS
- runtime audit: causal frame leakage, DDP QA shard collision, and shell/path injection hypotheses ruled out

Key evidence:

- `.omo/evidence/worldmm-smvqa/final/F1-plan-compliance.json`
- `.omo/evidence/worldmm-smvqa/final/F2-code-quality.json`
- `.omo/evidence/worldmm-smvqa/final/F3-smoke-qa.json`
- `.omo/evidence/worldmm-smvqa/final/F4-scope-fidelity.json`
- `.omo/evidence/worldmm-smvqa/final/review-work-final-goal.json`
- `.omo/evidence/worldmm-smvqa/final/review-work-final-qa.json`
- `.omo/evidence/worldmm-smvqa/final/review-work-final-code-quality.json`
- `.omo/evidence/worldmm-smvqa/final/review-work-final-security.json`
- `.omo/evidence/worldmm-smvqa/final/review-work-final-context.json`
- `.omo/evidence/worldmm-smvqa/final/runtime-audit/audit.json`

## Important Fixes Already Made

- Future frame refs no longer leak into earlier chunks/evidence.
- Retrieval remains causal: candidates end at or before `question_time`.
- `qa_transformers` has a CLI path and supports DDP rank shard/merge behavior.
- Remote retrieval evidence packs are valid JSONL.
- Remote script writes `memory_manifest.json`.
- `remote.py` stale duplicate path was removed.
- `scripts/remote/run_worldmm_smvqa.sh` delegates to the generated launch plan.
- `build-memory` invalid stages/stores fail nonzero.
- Remote script uses hashed per-question temp paths to avoid traversal.
- Displayed remote shell commands are quoted safely.
- Report manifest text is Markdown-escaped to prevent section spoofing.

## Not Done Locally

No real benchmark was run on this host.

Not done locally by rule:

- no SuperMemory-VQA full dataset download
- no Gemma 4 E2B model download
- no real Gemma inference
- no full benchmark evaluation
- no remote job submission
- no checkpoints or model artifacts copied back

Therefore there are no real SuperMemory-VQA leaderboard numbers yet.
Only tiny-fixture/mock smoke metrics exist locally.

## Remote Run Prerequisites

Run only on approved company compute through bastion/head node.

Required environment variables:

```bash
export WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST=1
export WORLDMM_SMVQA_REMOTE_APPROVED=1
export BASTION_HOST=...
export HEAD_NODE=...
export REMOTE_JOB_LAUNCHER=...
export SMVQA_DATA_ROOT=...
export SMVQA_FRAME_ROOT=...  # optional, defaults to $SMVQA_DATA_ROOT/frames
export GEMMA_MODEL_PATH=...
export WORLDMM_OUTPUT_ROOT=...
export WORLDMM_REMOTE_NODES=...
export WORLDMM_GPUS_PER_NODE=...
export WORLDMM_DDP_LAUNCHER='python -m torch.distributed.run'
export REMOTE_JOB_ID_OR_PROCESS_REF=...
export WORLDMM_RUN_ID=...
export WORLDMM_REMOTE_REPO=...
export WORLDMM_MODEL_ID=google/gemma-4-E2B-it  # optional, this is the default
export HF_TOKEN=...  # remote only, gated Gemma download
```

`configs/remote.example.yaml` must resolve to `runtime.location=remote`.

Remote host one-time setup (through bastion/head node):

```bash
uv sync --extra remote
```

The dry-run plan prints, in order: repo rsync to `$WORLDMM_REMOTE_REPO`, plan
rsync to `$WORLDMM_REMOTE_REPO/remote-plan/`, and the ssh launch command. The
remote script cds into `$WORLDMM_REMOTE_REPO` and downloads the model with
`hf download` (stage 0) before building memories and running DDP QA.

Frame assets for real QA must be available under
`$SMVQA_FRAME_ROOT/<video_id>/<frame_ref>.jpg` by default; `.jpeg`, `.png`, and
`.webp` are also accepted.

Still open: no SuperMemory-VQA dataset download/ingest step exists.
`$SMVQA_DATA_ROOT` must already contain `sources.jsonl` / `questions.jsonl` /
`labels.jsonl` in the fixture schema, with captions/OCR/objects/frame refs and
optional `pose_samples` / `gaze_samples` precomputed. Defining that ingest (raw
dataset -> schema) is the next task.

## Next Step: Produce Real Numbers

Generate the remote plan locally:

```bash
uv run worldmm-smvqa launch-remote \
  --dry-run \
  --config configs/remote.example.yaml \
  --out .omo/evidence/worldmm-smvqa/remote-plan
```

After explicit approval for remote execution, submit from the approved remote workflow:

```bash
WORLDMM_SMVQA_REMOTE_APPROVED=1 \
uv run worldmm-smvqa launch-remote \
  --submit \
  --config configs/remote.example.yaml \
  --out .omo/evidence/worldmm-smvqa/remote-plan
```

Expected remote outputs live under `$WORLDMM_OUTPUT_ROOT` and include:

- memory manifests
- retrieval evidence packs
- retrieval trace-bearing evidence packs
- Gemma 4 E2B predictions
- official metrics: `Ans-F1`, `QA-Acc`, `QA-MRR`
- spatial diagnostics, ablation outputs, diagnostics, and logs
- report manifest

Copy back only lightweight results:

- metrics
- logs
- plots
- summaries
- small sample outputs

Do not copy full datasets, model weights, checkpoints, or sensitive artifacts back to this host.

## Final Report After Remote Run

Create or update a remote run manifest containing:

- local code/config changed
- remote command used
- remote job ID or process reference
- remote artifact path on company storage
- key metrics or failure reason
- what was not copied locally

Then generate:

```bash
uv run worldmm-smvqa report \
  --run-manifest <remote_manifest.json> \
  --out .omo/evidence/worldmm-smvqa/final-remote-report.md
```

## Caveat

Use the baseline name `WorldMM-SMVQA`.

Do not call this an exact Video-RAG or EgoButler reproduction unless a separate reproduction lane is run and the implementation deltas are reported.
