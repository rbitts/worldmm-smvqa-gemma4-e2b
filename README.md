# WorldMM-SMVQA

Local-preparation implementation for explicit, compressed spatial memory on
SuperMemory-VQA. This repository prepares and validates code, contracts, tiny
fixtures, and launch plans; production artifacts belong on company compute.

## Design Documents

- [Documentation index](docs/README.md)
- [Spatial Memory project home](docs/spatial-memory/README.md)
- [Current implementation status](docs/spatial-memory/status.md)
- [Evidence-to-result traceability](docs/spatial-memory/traceability.md)

The previous spatial architecture, roadmap, and implementation-review files are
retained as migration sources. New canonical research updates belong under
`docs/spatial-memory/`.

Local commands are limited to code, config, tiny fixtures, and dry-runs. Real
model inference, dataset download, full evaluation, checkpoints, and large
artifacts belong on approved remote compute only.

No real model or dataset download, training, benchmark evaluation, or remote job
has been run as part of this implementation work.

## Prepared-Data Preflight

Run preflight before any expensive stage:

```bash
uv run worldmm-smvqa preflight \
  --fixture tests/fixtures/tiny_smvqa \
  --out /tmp/worldmm-preflight.json
```

It validates all three JSONL schemas, question/label ID parity, video scope,
time ranges, answer/evidence contracts, optional frame-file resolution, task and
choice distributions, spatial/pose coverage, and the derived at-most-1-Hz frame
inventory. Errors return a non-zero exit code; warnings remain visible in the
JSON report.

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
- `linear-v1` selector: applies a keep-score gate plus causal 16-record and
  4096-byte caps per 30s window. Byte cost is the actual serialized
  `SpatialTokenRecord` JSONL size. Candidates at the same observation time are
  greedily ordered by score per byte; future candidates cannot evict admitted
  past tokens.
- `compact-json-v1` codec: quantized object, relation, and zone tokens.

The final cap includes static tokens and wearer trajectory summaries. A
trajectory summary is admitted only when both record and serialized-byte budget
remain in its 30-second window.

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

## Typed Teacher And Student Preparation

`worldmm.typed_memory` defines explicit `object`, `plane`, `portal`,
`free_space`, `landmark`, `event`, and `no_write` records. Geometry,
uncertainty, validity, instance identity, provenance, and evidence references
stay explicit; writable records are charged by canonical serialized JSONL bytes.

G-CUT3R is an external teacher contract, not a bundled dependency. An external
provider receives one causal observation plus only the previous opaque state and
must emit typed records. The cache records backend/provider identity, request and
response digests, prefix hash continuity, state references, and observation
cutoffs. The package never downloads or imports G-CUT3R automatically. Remote
provider paths use:

- `WORLDMM_GCUT3R_CODE_PATH`
- `WORLDMM_GCUT3R_CHECKPOINT_PATH`
- `WORLDMM_GCUT3R_EXTRACTOR` for the staged DAG, or
  `WORLDMM_TEACHER_CACHE_INPUT` for a precomputed cache

Validate and materialize an external cache:

```bash
python -m worldmm_smvqa.worldmm.gcut3r_teacher validate-cache \
  --cache "$WORLDMM_TEACHER_CACHE_INPUT"

python -m worldmm_smvqa.teacher_materializer \
  --teacher-cache "$WORLDMM_TEACHER_CACHE_INPUT" \
  --supervision "$WORLDMM_STUDENT_SUPERVISION_INPUT" \
  --out /tmp/student_teacher_cache.jsonl
```

The materializer performs a complete one-to-one join between digest-validated
teacher records and supervision, rejects split leakage and target mismatch, and
uses the typed record's actual serialized byte cost.

Prepare selector rows only from an explicit counterfactual utility cache and
split manifest:

```bash
python -m worldmm_smvqa.spatial_selector_train prepare \
  --fixture "$SMVQA_DATA_ROOT" \
  --experiment configs/spatial/source_compact_v1.json \
  --utility-cache "$WORLDMM_UTILITY_CACHE_INPUT" \
  --split-manifest "$WORLDMM_SPLIT_MANIFEST_INPUT" \
  --supervision-mode counterfactual \
  --out /tmp/spatial-selector-rows.jsonl
```

Utility combines deletion-induced QA loss/score change, geometry coverage,
uncertainty reduction, pose information, surprise, redundancy, and actual bytes.
Rows carry hashes of both inputs. Split validation keeps question, participant,
session, and video groups out of conflicting train/validation/test partitions.

The typed candidate head is a PyTorch multi-head model for record type, typed
geometry, association, uncertainty, byte rate, and teacher distillation. It is
not yet a raw RGB/IMU encoder and has no checkpoint-to-record inference decoder.
CPU
`dry-run` validates one forward/loss pass; `train` is CUDA and remote-only, uses
`torch.distributed`, `DistributedSampler`, DDP, reduced validation metrics, and
atomic resumable checkpoints:

```bash
python -m worldmm_smvqa.spatial_train dry-run \
  --teacher-cache /tmp/student_teacher_cache.jsonl

# Company GPU environment only, normally launched by the generated Slurm DAG.
python -m torch.distributed.run ... \
  -m worldmm_smvqa.spatial_train train \
  --config configs/remote.example.yaml \
  --teacher-cache "$WORLDMM_OUTPUT_ROOT/training/student_teacher_cache.jsonl" \
  --checkpoint "$WORLDMM_OUTPUT_ROOT/checkpoints/spatial_student.pt"
```

## Deterministic Geometry Proofs

QA prompt construction removes spatial snippets and geometry dictionaries, then
passes only executor proofs. Retrieval bundles the required spatial relation and
endpoint objects; the deterministic executor supports
distance, relative direction, near, last-seen, and count. Each answerable proof
contains the operation, entity IDs, coordinate frame, value, propagated
uncertainty, provenance, and evidence refs. Ambiguous entities, frame mismatch,
unsupported provenance, missing yaw, or excessive uncertainty yield an explicit
unanswerable proof. Count requires an explicit complete-index certificate. Model
output may persist only known answerable proof IDs. Current memory IDs can still
encode entity labels; an opaque-ID layer remains future work.

## Learned-Path Boundary

Local preparation does not yet close the learned end-to-end path. Missing pieces
are checkpoint inference, type-specific geometry decoding, open-world
association, and generation of student-backed spatial evidence. The staged DAG
therefore fails closed on externally supplied teacher, supervision, utility,
split, and student-evidence artifacts. The source-compact path is a tested
heuristic baseline, not evidence that the learned G-CUT3R student is reproduced.

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

Core environment variables used by remote config and scripts:

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
- `WORLDMM_SPATIAL_BYTE_BUDGET` (optional, default `4096` per 30s complete-token window)
- `WORLDMM_TRAIN_EPOCHS`, `WORLDMM_TRAIN_BATCH_SIZE`,
  `WORLDMM_TRAIN_HIDDEN_DIM`, `WORLDMM_TRAIN_LEARNING_RATE`
- `WORLDMM_TRAIN_RESUME` (optional existing checkpoint path; missing/empty files fail)
- `WORLDMM_MODEL_ID` (optional, default `google/gemma-4-E2B-it`)
- `HF_TOKEN` (remote only, required for the gated Gemma download)

The staged DAG intentionally fails closed unless externally produced inputs are
provided:

- `WORLDMM_GCUT3R_EXTRACTOR` **or** `WORLDMM_TEACHER_CACHE_INPUT`
- `WORLDMM_STUDENT_SUPERVISION_INPUT`
- `WORLDMM_UTILITY_CACHE_INPUT`
- `WORLDMM_SPLIT_MANIFEST_INPUT`
- `WORLDMM_QA_EVIDENCE_INPUT` (evidence generated from the trained student path)

CPU/GPU resource controls include `WORLDMM_CPU_PARTITION`,
`WORLDMM_GPU_PARTITION`, and the `WORLDMM_{PREFLIGHT,TEACHER,UTILITY,TRAIN,REPORT}_*`
node/CPU/GPU/memory/time variables. GPU stages inherit the project contract of
10 nodes and 8 GPUs per node; set
`WORLDMM_TEACHER_NODES`, `WORLDMM_TRAIN_NODES`, and their GPU counts explicitly
lower for an approved bounded probe.

Remote-only commands also require `runtime.location=remote` in the selected
config.

## Remote Setup

On the remote host (via bastion/head node), install the inference stack once:

```bash
uv sync --extra remote
```

The generated plan writes a legacy single-job script plus the preferred staged
DAG:

```text
preflight_ingest (CPU)
  -> teacher_extract (GPU)
  -> merge_utility (CPU)
  -> train_qa (GPU, torch.distributed/DDP)
  -> metrics_report (CPU)
```

Each dependency uses Slurm `afterok`; any failed or missing prerequisite blocks
all downstream work. Submission is guarded by a run-scoped lock, validates
numeric `sbatch --parsable` IDs, writes every job ID to
`summary/dag_jobs.env`, and keeps outputs under a unique
`$WORLDMM_OUTPUT_ROOT`. The submitter independently requires and exports
`WORLDMM_SMVQA_REMOTE_APPROVED=1`; `WORLDMM_OUTPUT_ROOT` must end exactly in
`/$WORLDMM_RUN_ID`.

External teacher extraction runs one worker per allocated GPU. Each invocation
receives `--rank`, `--world-size`, the generated 1 Hz
`--sensor-frame-manifest`, and a unique `rank-NNNNN.jsonl` output path. Empty,
missing, stale, or incomplete shard sets stop the DAG.

The generated plan prints three copy/paste commands, in order:

1. `rsync` the repo to `$WORLDMM_REMOTE_REPO` on company storage.
2. `rsync` the generated plan directory to `$WORLDMM_REMOTE_REPO/remote-plan/`.
3. `ssh` through `$BASTION_HOST` and `$HEAD_NODE` to run
   `remote-plan/submit_worldmm_smvqa_dag.sh`.

Both `rsync` commands use the same default remote repository as the SSH command
and exclude `.env*`. `expected_outputs.json` lists only outputs produced by the
preferred DAG; its job reference points to `summary/dag_jobs.env`.

Plan generation never opens SSH, runs a remote shell, downloads an artifact, or
submits a job. The DAG also does not fabricate teacher caches, supervision,
counterfactual utility, split assignments, or student-backed evidence; missing
inputs stop their stage with a clear shell error.

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
- `QA-Acc`: top-1 four-choice accuracy across all questions, including the
  unanswerable/N/A choice.
- `QA-MRR`: reciprocal rank of the gold four-choice answer across all questions.

These three official metrics are serialized on the benchmark's `0–100` scale.
Diagnostic Memory-Recall@K remains a `0–1` fraction.

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
