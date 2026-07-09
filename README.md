# WorldMM-SMVQA

Minimal scaffold for a WorldMM-style SuperMemory-VQA benchmark package.

Local commands are limited to code, config, tiny fixtures, and dry-runs. Real
model inference, dataset download, full evaluation, checkpoints, and large
artifacts belong on approved remote compute only.

## Local Smoke

Local smoke uses only `tests/fixtures/tiny_smvqa` and the mock QA backend.

```bash
uv run worldmm-smvqa --help
uv run worldmm-smvqa smoke \
  --fixture tests/fixtures/tiny_smvqa \
  --out .omo/evidence/worldmm-smvqa/smoke
```

## Remote Launch

Generate a launch plan locally. Do not submit or start remote work without
explicit approval.

```bash
uv run worldmm-smvqa launch-remote \
  --dry-run \
  --config configs/remote.example.yaml \
  --out .omo/evidence/worldmm-smvqa/remote-plan
```

Exact environment variables used by remote config and scripts:

- `WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST=1`
- `WORLDMM_SMVQA_REMOTE_APPROVED=1`
- `BASTION_HOST`
- `HEAD_NODE`
- `REMOTE_JOB_LAUNCHER`
- `SMVQA_DATA_ROOT`
- `GEMMA_MODEL_PATH`
- `WORLDMM_OUTPUT_ROOT`
- `WORLDMM_REMOTE_NODES`
- `WORLDMM_GPUS_PER_NODE`
- `WORLDMM_DDP_LAUNCHER`
- `REMOTE_JOB_ID_OR_PROCESS_REF`
- `WORLDMM_RUN_ID`
- `WORLDMM_REMOTE_REPO`

Remote-only commands also require `runtime.location=remote` in the selected
config.

## No Local Downloads

Do not download production models or full SuperMemory-VQA data locally. Do not
run real training, real evaluation, Gemma inference, large preprocessing, or
checkpoint creation on this host. Local outputs should be tiny fixtures, dry-run
plans, reports, logs, summaries, and small sample artifacts only.

## Metrics

Official report metrics are:

- `Ans-F1`: answerability F1 across answerable/unanswerable decisions.
- `QA-Acc`: top-1 multiple-choice accuracy on answerable questions.
- `QA-MRR`: reciprocal-rank quality for the correct answer choice.

Diagnostics can include memory recall at K, causal violation count, prompt token
summary, and memory size summary. If the remote run has not executed, reports
must say pending or failed and must not invent metric values.

## Baseline Name

Use `WorldMM-SMVQA` for this implementation. Do not call it an exact
Video-RAG or EgoButler reproduction unless a separate reproduction lane ran and
the implementation deltas are reported.

## Final Report

The handoff report is generated from a remote run manifest:

```bash
uv run worldmm-smvqa report \
  --run-manifest tests/fixtures/tiny_smvqa/remote_manifest.example.json \
  --out .omo/evidence/worldmm-smvqa/report.md
```

Required report sections:

- local code/config changed
- remote command used
- remote job ID or process reference
- remote artifact path on company storage
- key metrics or failure reason
- what was not copied locally
