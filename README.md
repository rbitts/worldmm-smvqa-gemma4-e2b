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

Current Goal: audit real sensor availability, then measure a causal offline
teacher-oracle object/location ceiling under the same serialized-byte budget.
G-CUT3R is offline teacher only. Raw student implementation starts only after an
Oracle Go result; see the canonical [roadmap](docs/spatial-memory/roadmap.md).

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

The student contract path is separate from the heuristic smoke and EXP-0005:
`worldmm-smvqa mock-dag --fixture configs/spatial/model_boundary_contract_v1.json`
runs a CPU/offline, outputless production-consumer wiring check. Its success is only
`mock` evidence. A remote `contract_probe` proves loadability only; neither result can
claim real forward compatibility, model quality, or official/student completion.
Student evidence and reports must bind the model-contract, student-architecture, and
model-load-consensus digests.

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

## Compressed Spatial Memory Baseline

The current local heuristic baseline is one composable pipeline. It is not a
profiled on-device learned model:

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

## Offline Teacher And Legacy Student Scaffold

`worldmm.typed_memory` defines explicit `object`, `plane`, `portal`,
`free_space`, `landmark`, `event`, and `no_write` records. Geometry,
uncertainty, validity, instance identity, provenance, and evidence references
stay explicit; writable records are charged by canonical serialized JSONL bytes.

G-CUT3R is an offline external teacher/oracle contract, not an on-device model or
bundled dependency. An external
provider receives one causal observation plus only the previous opaque state and
must emit typed records. The cache records backend/provider identity, request and
response digests, prefix hash continuity, state references, and observation
cutoffs. The package never downloads or imports G-CUT3R automatically. The
staged DAG accepts only:

- `WORLDMM_GCUT3R_EXTRACTOR` for the staged DAG, or
  `WORLDMM_TEACHER_CACHE_INPUT` for a precomputed cache

The approved extractor is a trusted wrapper and owns its G-CUT3R code,
checkpoint loading, and provenance. Each distributed rank must write exactly
one non-empty rank shard. After merge, the request multiset must equal selected
sensor `(video_id, frame_ref, timestamp)` observations exactly, which also
rejects duplicate requests across ranks.

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
For a precomputed cache, remote preflight and merge additionally require its
request `(video_id, frame_ref, timestamp)` tuples to equal the selected sensor
manifest observations exactly, with no missing or extra request.
Production cache/extractor output rejects
`pose_guidance.source=ground_truth`; only `imu`, `vio`, or `slam` pose guidance
is accepted.

`WORLDMM_STUDENT_SUPERVISION_INPUT` is UTF-8 JSONL. Each row has this schema:

```json
{"observation_id":"obs-0001","memory_id":"mem-0001","group_id":"participant-001/session-001","split":"train","features":[0.1,0.2],"teacher_embedding":[0.3,0.4],"geometry_target":[1.0,2.0,3.0],"association_target":0}
```

There must be exactly one row for every teacher
`(observation_id, memory_id)` key. Vectors must be non-empty finite arrays with
consistent dimensions per field across the file. Both `train` and `validation`
must exist; a `group_id` cannot cross splits; training association IDs must be
contiguous from zero; validation may use only association IDs seen in training.
The repository validates and joins these supplied targets but does not derive
their geometry or embedding values.

The following selector preparation is a separate P2 research scaffold, not an
input to the staged typed DAG. Use it only with an explicit counterfactual
utility cache and split manifest:

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

The feature-level typed candidate head is a PyTorch multi-head model for record type, typed
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
distance, relative direction, near, last-seen, last-location, and count. Each
answerable proof contains the operation, entity IDs, coordinate frame, value, propagated
uncertainty, provenance, and evidence refs. Ambiguous entities, frame mismatch,
unsupported provenance, missing inferred frame evidence, low confidence, missing
yaw, or excessive uncertainty yield an explicit unanswerable proof. Count,
last-seen, and last-location require an explicit complete-index certificate. Model
output may persist only known answerable proof IDs. Current memory IDs can still
encode entity labels; an opaque-ID layer remains future work.

Relative-direction proofs accept only raw IMU or explicitly online-causal VIO
pose samples. Each sample must carry `observed_through_time` between its pose
timestamp and question time; offline SLAM, ground-truth/model pose, and missing
or future certificates abstain.

Student QA independently converts canonical `typed_memory.jsonl` back through
the retrieval projection and requires every spatial evidence item to exact-match
its memory ID, video, snippet, frame refs, time range, and geometry. The
byte-budgeted typed artifact is not a complete entity index. Count, last-seen,
and last-location abstain. Pair proofs also require explicit entity IDs in the question unless an
end-to-end completeness certificate establishes label uniqueness; the generated
production DAG does not issue one. Pair proofs use the records' actual local
frame and reject cross-video pairs.

## Learned-Path Boundary

Local preparation now contains a strict causal sensor schema and a selected
teacher-point to object/place target compiler. It does not contain the semantic
mask/place provider, production RGB encoder, native-sensor integration, or
open-world association. The staged legacy DAG therefore requires an approved
external teacher extractor or causal cache, explicit student supervision, and
`WORLDMM_SPATIAL_INFER_EXE`. It builds
student-backed evidence itself from returned typed records; prebuilt student
evidence, counterfactual utility, and selector split files are not staged-DAG
inputs or substitutes. The source-compact path remains a tested heuristic
baseline, not evidence that a learned or on-device G-CUT3R student is reproduced.

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
`retrieved_evidence_json`; every sampled-frame row carries `video_id`,
`frame_ref`, and `timestamp`. Prediction audit refs use the unambiguous
`<video_id>/<frame_ref>` form. Real Gemma runs use the sampled frame files as
multimodal image inputs. Prompt and resume contracts are
`qa-prompt-prediction-schema-v4` and `qa-resume-manifest-v5`.

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

`HANDOFF.md` is the operational source of truth for company execution. Read
[WorldMM-SMVQA Experiment Handoff](HANDOFF.md) before generating or approving a
plan. This section only summarizes the production typed DAG.

Generate a plan locally. Plan generation performs no SSH, download, model run,
or Slurm submission:

```bash
uv run worldmm-smvqa launch-remote \
  --dry-run \
  --config configs/remote.example.yaml \
  --out .omo/evidence/worldmm-smvqa/remote-plan
```

Default `WORLDMM_DAG_PHASE=preflight` submits one CPU preflight job. An operator
then reviews `diagnostics/preflight_inputs.sha256`,
`diagnostics/env_contract.json`, the frame/model fingerprints, and warnings;
creates the run-scoped approval JSON; and invokes `WORLDMM_DAG_PHASE=run`. The
approved run submits exactly seven `afterok` stages:

```text
preflight_ingest
  -> operator approval
  -> teacher_extract
  -> merge_materialize
  -> train
  -> build_memory
  -> student_infer_retrieve
  -> qa
  -> metrics_report
```

`probe` is the default profile and requires a reduced
`WORLDMM_PROBE_FIXTURE`; it rejects the full prepared-data root and allocations
above 1 node x 1 GPU. Full scale needs a new run ID, preflight inventory,
environment contract, and explicit approval.

Probe additionally caps each fixture JSONL and supervision at 64 MiB,
supervision/materialized rows at 10,000, teacher cache at 256 MiB, source rows at
4, raw frame metadata at 600, and typed output at 10,000 records/16 MiB. These
are fail-closed probe guards; they do not alter the separately approved full
profile.

Learned inference requires executable `WORLDMM_SPATIAL_INFER_EXE`. Preflight
must receive exactly `worldmm-spatial-infer-v1` from `--contract-version` and
`worldmm-spatial-infer-v1:self-test-ok` from `--self-test`. The latter checks
lightweight CLI/schema/canonical-writer conformance, not model accuracy.
Inference receives the checkpoint, frame root, sensor-frame manifest, and a
sanitized copy at `inference_inputs/sources.jsonl`; questions, labels, training
supervision, model paths, and approval data are not passed to it.
It also receives SHA-256 values for sanitized sources, the selected-frame
content manifest, and the producer executable. Its manifest must echo
`sources_sha256`, `frame_assets_sha256`, and `producer_sha256`; the repository
recomputes them independently.
Student QA repeats those exact comparisons using
`--inference-sources`, `--frame-assets-manifest`, and
`--inference-producer`; its resume manifest binds the sanitized source and
producer bytes.

Preflight first applies the run's at-most-1-Hz sensor manifest, then clears
transcript, transcript spans, captions, OCR, object labels, and object
detections. It retains source identity/time, pose/gaze, selected `frame_refs`,
and selected frame metadata with descriptions erased. Only selected frame files
are copied under `inference_inputs/frames/`; all post-preflight adapters and QA
receive that run-scoped copied root, never the full original frame root.
External adapters are approved trusted executables with a known-variable
denylist, not sandboxed `env -i` processes. Ambient `PATH`, `HOME`, `PYTHONPATH`,
and Slurm state remain; use a separate hardened runtime for untrusted code.

The executable must write canonical typed JSONL plus a production manifest.
Records are validated with `window_seconds=30.0`, grouped by
`(source_video_id, floor(first_seen_time / 30.0))`, and limited by
`WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW=4096` by default. The repository
recounts canonical UTF-8 JSONL bytes and revalidates checkpoint, sensor, record,
record-count, window-count, maximum-window-byte, and total-byte fields before
retrieval.
Validation streams JSONL and rejects any canonical record row above 1 MiB.

QA then requires real frames and
`retrieval/evidence_packs.jsonl.lineage.json`, bound to evidence, checkpoint,
typed memory, inference manifest, spatial config,
`manifests/sensor_frames.jsonl`, and run data. Preflight also writes
`diagnostics/gemma_model.sha256` and `diagnostics/frame_assets.sha256` alongside
the deployed-code/input inventory. `metrics_report` writes
`metrics/metrics.json`, `summary/run_identity.json`,
`summary/remote_manifest.json`, and
`summary/final_report.md` from the recomputed digests and metrics.

Preflight also emits filename-inventory digests for selected frames and both
model trees (`frame_assets.files.sha256`, `gemma_model.files.sha256`, and
`memory_model.files.sha256`) plus the memory-model content fingerprint.
Both model trees must be self-contained and symlink-free; local-only
`AutoConfig`/`AutoProcessor` loading and every shard referenced by a weight index
are checked before execution.
Successful QA writes `qa/completed.json`, binding prediction and QA-resume
manifest digests before `metrics_report` may consume them.
Before evaluation, finalization writes `summary/finalization_inputs.sha256` over
QA outputs, evidence lineage, config, sensor manifest, split files, and other
run-critical inputs; it rechecks the seal after evaluation and includes it in
the run identity.

Profile determines the claim: `probe` emits `result_class=contract_probe` with
experiment label `PROBE`; `full` emits `result_class=student` with label `E1`.
Artifact filenames are profile-neutral; embedded profile/result class remains
authoritative. Neither output is official E1/E2/E3.
Official reporting remains blocked until three immutable, matched E1/E2/E3
ablation identities are generated and validated. Missing production inputs,
frames, lineage, or approval stops the DAG; no heuristic or prebuilt QA evidence
substitutes for learned inference.

## No Local Downloads

Do not download production models or full SuperMemory-VQA data locally. Do not
run real training, real evaluation, Gemma inference, large preprocessing, or
checkpoint creation on this host. Local outputs should be tiny fixtures, dry-run
plans, reports, logs, summaries, and small sample artifacts only. Remote
copyback is limited to metrics, reviewed non-sensitive diagnostics, redacted
lightweight logs/plots, summaries, and explicitly approved small samples.

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
