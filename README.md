# WorldMM-SMVQA

Minimal scaffold for a WorldMM-style SuperMemory-VQA benchmark package.

## Design Documents

- [Spatial token compression architecture](docs/spatial-token-compression.md)
- [Spatial token research roadmap](docs/spatial-token-research-roadmap.md)

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

## Spatial Memory

WorldMM-SMVQA now builds a fourth WorldMM store, `spatial`, alongside
`episodic`, `semantic`, and `visual`.

The source schema accepts optional timed spatial signals:

- `pose_samples`: wearer pose samples with `timestamp`, `x`, `y`, `z`, and
  optional `yaw`.
- `gaze_samples`: gaze target samples with `timestamp`, `x`, `y`, and `z`.

`x` and `y` are the horizontal plane used for zones and distances; `z` is
vertical, in meters. Empty fields are valid, so existing source fixtures remain
compatible. When present, pose/gaze samples are sliced into the same clip and
shard windows as captions, OCR, objects, and frame metadata.

The `spatial` store contains compact tokens for zones, object anchors, and
relations plus wearer trajectory summaries. Gaze targets are preferred for
anchors when available; otherwise the anchor approximates the wearer pose at
detection time.

## Compressed Spatial Memory Model

The on-device path is one composable model:

```text
SpatialGeometryEncoder
  -> SpatialProjectionHead
  -> SpatialTokenDecoder
  -> SpatialTokenSelector
  -> SpatialMemoryCodec
  -> spatial store
```

The default experiment is
`configs/spatial/source_compact_v1.json`:

- `structured-v1` encoder: source object geometry, gaze, pose, and SLAM-style
  primitives.
- `identity-v1` projection: preserves the baseline scalar feature schema.
- `delta-topk-v1` decoder: suppresses repeated static observations, canonicalizes
  inverse relations, and emits spatial-token candidates.
- `linear-v1` selector: applies a keep-score gate and causal 16-record cap per
  30s window; future candidates cannot evict admitted past tokens.
- `compact-json-v1` codec: quantized object, relation, and zone tokens.

This replaces repeated per-window object geometry snapshots in the main spatial
artifact. The original detailed builders remain available for diagnostics and
compression comparison.

The design follows object-centric 3D scene graphs such as ConceptGraphs, learned
fixed-budget token selection such as TokenLearner, and duplicate-first reduction
such as DART. Raw visual-token pruning is not the primary spatial path because
localization-heavy tasks can regress when importance comes only from generic VLM
attention.

Run the complete local memory -> retrieval -> mock-QA path with one experiment:

```bash
WORLDMM_SPATIAL_EXPERIMENT_CONFIG=configs/spatial/source_compact_v1.json \
uv run worldmm-smvqa smoke \
  --fixture tests/fixtures/tiny_smvqa \
  --out .omo/evidence/worldmm-smvqa/compressed-spatial-smoke
```

The spatial model does not answer QA separately:

```text
episodic + semantic + visual + spatial
  -> WorldMM retrieval
  -> one evidence pack
  -> Gemma QA decoder
```

To add CUT3R or another geometry backbone, implement and register
`SpatialGeometryEncoder`, `SpatialProjectionHead`, and optionally
`SpatialTokenDecoder`, `SpatialTokenSelector`, or `SpatialMemoryCodec`. The
encoder can pass an in-process opaque latent state through projection to the
decoder; only decoder output tokens are persisted. List plugin modules and
component names in a new experiment JSON. Projected feature names are dynamic,
so the selector trainer accepts new dimensions without core changes.

Prepare QA-supervised keep/drop rows locally on the tiny fixture:

```bash
uv run python -m worldmm_smvqa.spatial_selector_train prepare \
  --fixture tests/fixtures/tiny_smvqa \
  --experiment configs/spatial/source_compact_v1.json \
  --out /tmp/spatial-selector-rows.jsonl
```

Real selector training remains remote-only:

```bash
WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST=1 \
python -m worldmm_smvqa.spatial_selector_train train \
  --config configs/remote.example.yaml \
  --input "$WORLDMM_OUTPUT_ROOT/manifests/spatial-selector-rows.jsonl" \
  --out "$WORLDMM_OUTPUT_ROOT/models/spatial-selector.json"
```

Set the trained weight path in a new experiment JSON, then run the normal remote
memory, retrieval, Gemma QA, metrics, and ablation workflow. The effective
experiment is written to `manifests/spatial_experiment.json`.
Distributed spatial build writes rank-level compression measurements to
`memory/worldmm_sv/spatial.stats.jsonl`.

## 1 Hz Sensor Input Contract

Every remote run first writes one run-scoped `sensor_frames.jsonl` from the raw
`sources.jsonl.frame_metadata` inventory:

```bash
worldmm-smvqa build-memory \
  --stage sensor-frames \
  --fixture "$SMVQA_DATA_ROOT" \
  --out "$WORLDMM_OUTPUT_ROOT/manifests/sensor_frames.jsonl"
```

Selection is timestamp-only: keep the first available frame in each fixed
one-second window relative to the source start. Missing windows stay missing;
frames are never synthesized. Set `WORLDMM_SENSOR_FRAME_MANIFEST` to that file.
The shared source reader then applies the same at-most-1-Hz RGB inventory to
memory construction, retrieval, and QA. It also removes unselected frame OCR and
object detections and excludes legacy captions without frame/timestamp grounding.

Pose, gaze, audio, and transcript streams retain their device-provided rates.
QA's `32` frame cap is applied only after this manifest, over the selected
pre-question shard.

## Retrieval Contract

The default retrieval protocol is `worldmm-smvqa`: a WorldMM-augmented
Video-RAG/EgoButler-compatible retrieval path, not an exact implementation of
WorldMM's published agentic loop, PPR graph retrieval, embedding search, LLM
reranking, or STOP-controlled iterative retrieval.

Video-RAG shard eligibility is enforced first. Only `shard_30m` chunks from the
question video with `end_time <= question_time` are eligible. QA then samples up
to `32` frames uniformly from the selected pre-question shard and sends those
frames together with the retrieved memory text.

Within eligible shards, retrieval follows the EgoButler hierarchy:

```text
shard_30m -> clip_30s -> memory records
```

The WorldMM policy route then orders enabled stores from the raw question text.
Location questions route `spatial` first; event/time questions route `episodic`
first; category/relation questions route `semantic` first; appearance, OCR, and
frame questions route `visual` first. Disabled stores are never selected.

Every evidence pack includes `retrieval_trace` with eligible shard ids, selected
clip ids, policy route, store order, candidate counts, causal-filter count, and
frame-ref count. The QA prompt includes both `sampled_video_frames_json` and
`retrieved_evidence_json`; real Gemma runs use the sampled frame files as
multimodal image inputs.

## Spatial Diagnostics And Ablations

Smoke writes `spatial_diagnostics.json` with spatial relation accuracy,
per-store Memory Recall@K, and protocol Recall@K.

Run the with/without-spatial ablation locally on the tiny fixture:

```bash
uv run worldmm-smvqa smoke \
  --fixture tests/fixtures/tiny_smvqa \
  --out .omo/evidence/worldmm-smvqa/ablation-without-spatial \
  --ablation-stores episodic,semantic,visual
```

Run the protocol-only ablation:

```bash
uv run worldmm-smvqa smoke \
  --fixture tests/fixtures/tiny_smvqa \
  --out .omo/evidence/worldmm-smvqa/ablation-legacy-protocol \
  --ablation-protocol legacy-round-robin
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
- `SMVQA_FRAME_ROOT` (optional, default `$SMVQA_DATA_ROOT/frames`)
- `GEMMA_MODEL_PATH`
- `WORLDMM_OUTPUT_ROOT`
- `WORLDMM_REMOTE_NODES`
- `WORLDMM_GPUS_PER_NODE`
- `WORLDMM_DDP_LAUNCHER`
- `WORLDMM_TRITON_CACHE_ROOT` (optional node-local root; QA appends global rank)
- `WORLDMM_SENSOR_FRAME_MANIFEST` (generated run-scoped 1 Hz RGB inventory)
- `REMOTE_JOB_ID_OR_PROCESS_REF`
- `WORLDMM_RUN_ID`
- `WORLDMM_REMOTE_REPO`
- `WORLDMM_MODEL_ID` (optional, default `google/gemma-4-E2B-it`)
- `HF_TOKEN` (remote only, required for the gated Gemma download)

Remote-only commands also require `runtime.location=remote` in the selected
config.

## Remote Setup

On the remote host (via bastion/head node), install the inference stack once:

```bash
uv sync --extra remote
```

The generated plan prints three copy/paste commands, in order:

1. `rsync` the repo to `$WORLDMM_REMOTE_REPO` on company storage.
2. `rsync` the generated plan directory to `$WORLDMM_REMOTE_REPO/remote-plan/`.
3. `ssh` through `$BASTION_HOST` to run the plan script with the job launcher.

The remote script `cd`s into `$WORLDMM_REMOTE_REPO`, downloads
`$WORLDMM_MODEL_ID` to `$GEMMA_MODEL_PATH` via `hf download` if missing
(stage 0), then runs memory build, retrieval, DDP QA, and evaluation. All
remote-side environment variables must be set in the remote execution
environment.

Frame assets for real QA must be readable as
`$SMVQA_FRAME_ROOT/<video_id>/<frame_ref>.jpg` by default. Existing `.jpeg`,
`.png`, or `.webp` files with the same stem are also accepted.

Remote reruns that should match the spatial retrieval contract must pass:

```bash
--retrieval-protocol worldmm-smvqa --max-frame-refs 32
```

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
