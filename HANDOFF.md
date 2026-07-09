# WorldMM-SMVQA Handoff

## Status

WorldMM-SMVQA implementation is complete locally.

Implemented pipeline:

```text
SuperMemory-VQA stream
-> 30s clips / 30m shards
-> caption / OCR / object / frame memories
-> WorldMM episodic / semantic / visual memories
-> causal retrieval before question_time
-> Gemma 4 E2B QA interface
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

Primary commands:

```bash
uv run worldmm-smvqa smoke --fixture tests/fixtures/tiny_smvqa --out .omo/evidence/worldmm-smvqa/final-smoke
uv run worldmm-smvqa launch-remote --dry-run --config configs/remote.example.yaml --out .omo/evidence/worldmm-smvqa/remote-plan
uv run worldmm-smvqa report --run-manifest tests/fixtures/tiny_smvqa/remote_manifest.example.json --out .omo/evidence/worldmm-smvqa/report.md
```

## Verified Locally

Final gates passed:

- `pytest`: 70 passed
- `ruff`: passed
- `basedpyright`: 0 errors, 0 warnings
- local smoke: writes `metrics.json`, `predictions.jsonl`, `evidence_packs.jsonl`, `memory_manifest.json`
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
export GEMMA_MODEL_PATH=...
export WORLDMM_OUTPUT_ROOT=...
export WORLDMM_REMOTE_NODES=...
export WORLDMM_GPUS_PER_NODE=...
export WORLDMM_DDP_LAUNCHER='python -m torch.distributed.run'
export REMOTE_JOB_ID_OR_PROCESS_REF=...
export WORLDMM_RUN_ID=...
export WORLDMM_REMOTE_REPO=...
```

`configs/remote.example.yaml` must resolve to `runtime.location=remote`.

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
- Gemma 4 E2B predictions
- official metrics: `Ans-F1`, `QA-Acc`, `QA-MRR`
- diagnostics and logs
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
