# WorldMM-SMVQA Experiment Handoff

> Canonical research design and current readiness status live under
> [docs/spatial-memory](docs/spatial-memory/README.md). This file contains only
> operational company-compute handoff instructions.

## Purpose

This document is the next-run handoff for official SuperMemory-VQA experiments.
It is not a local completion report.

Goal: run WorldMM-SMVQA on company compute and measure whether explicit spatial
memory plus the WorldMM retrieval policy improves official QA metrics under the
benchmark-compatible input contract:

```text
retrieved memory text + 32 uniformly sampled pre-question video frames
+ question + answer choices
```

## Current State

Local preparation code and a staged Slurm plan are ready for a company-side
artifact-contract probe, not a final reproduction run.
GPU stages default to the project allocation of 10 nodes x 8 GPUs; an operator
must explicitly lower those settings for a bounded probe. Real data/model
execution remains unverified.

Implemented locally:

- 30s clip / 30m shard chunking
- source memories for captions, transcript, OCR, objects, frame metadata
- WorldMM stores: `episodic`, `semantic`, `visual`, `spatial`
- pluggable spatial model:
  `encoder -> projection head -> token decoder -> selector -> codec`
- compact causal spatial tokens with quantization, duplicate suppression,
  canonical relations, and full per-window record/serialized-byte budgets,
  including trajectory summaries
- explicit typed teacher records for object, plane, portal, free-space,
  landmark, event, and no-write candidates
- causal external G-CUT3R provider/cache contract with request/response digests,
  prefix hash chain, state continuity, pose/depth guidance, and no automatic
  download/import
- teacher-cache/supervision materializer with actual serialized-byte targets
  and train/validation leakage checks; geometry and association targets remain
  externally supplied
- deletion-based counterfactual QA utility, actual-byte value, and explicit
  participant/session/question split manifest with input hashes
- remote-only PyTorch typed student training with genuine
  `torch.distributed` DDP, distributed sampling/validation, and atomic resume
  checkpoints; CPU dry-run performs only a forward/loss check
- effective spatial experiment persisted in local/remote manifests
- Video-RAG causal shard eligibility
- EgoButler-style `shard_30m -> clip_30s -> memory records` retrieval
- WorldMM store routing with spatial-first route for location questions
- retrieval traces in every evidence pack
- QA prompt containing sampled frame manifest and retrieved memory JSON
- deterministic geometry executor for distance, direction, near, last-seen, and
  count, with entity IDs, coordinate frame, uncertainty, provenance, evidence,
  and proof-ID validation in predictions
- real Gemma path using multimodal frame inputs when frame files exist
- distributed Qwen memory generation partitioned by video across ranks
- batch retrieval that loads memory artifacts once
- resumable atomic Gemma QA rank checkpoints and merge
- E1/E2/E3 retrieval, QA, evaluation, diagnostics, and report paths in the
  legacy single-job lane; the preferred typed DAG currently runs E1 only
- `srun` + `torch.distributed.run` rendezvous from Slurm node metadata
- fail-closed CPU/GPU Slurm DAG using `afterok` dependencies, run-scoped
  submission locking, validated `sbatch --parsable` job IDs, and job manifest
- remote dry-run plan with a direct ProxyJump head-node command
- SuperMemory-VQA multi-video question scope via `QuestionRequest.video_ids`
- retrieval/protocol filtering over `video_ids` with single-video fallback
- frame sampling guard that rejects trace shards outside the question video scope
- metadata contract tests proving raw `all_qa.json` is QA label metadata, not
  memory source-stream input

Verified locally only on tiny mock fixture:

- `ruff check .`
- `basedpyright`
- `pytest -q`
- `worldmm-smvqa smoke --fixture tests/fixtures/tiny_smvqa ...`
- `python -m worldmm_smvqa.qa_transformers --backend mock ...`
- `worldmm-smvqa launch-remote --dry-run ...`
- lightweight SuperMemory metadata contract checks from
  `tests/test_supermemory_metadata_contract.py`

No real model/data download, teacher extraction, training, benchmark evaluation,
checkpoint creation, remote shell, or remote job occurred during local
preparation. Only code checks, tiny mock-fixture work, and dry-run plan generation
are in scope here.

Run `pytest`, `ruff`, `basedpyright`, and generated-script `bash -n` immediately
before transfer; record the live results instead of preserving a stale test
count here. Main residual risks are the external G-CUT3R adapter, actual
teacher/supervision/utility/evidence artifacts, real model compatibility, and
prepared dataset/frame completeness.

Blocking gaps before claiming final-method reproduction:

- no installed G-CUT3R implementation or raw RGB/IMU/VIO student encoder
- no checkpoint inference and type-specific geometry decoder
- no open-world association/pointer memory in the learned student
- QA-aware selector and typed DDP student remain separate learned paths
- no internal counterfactual utility producer
- no internal student checkpoint -> typed memory -> evidence path; the typed
  DAG still requires `WORLDMM_QA_EVIDENCE_INPUT`
- preferred typed DAG does not yet run E2/E3, byte Pareto, or final report

## Official Parity Checklist

Before trusting remote numbers, verify all items below in the remote artifacts.

- Dataset split: use the intended SuperMemory-VQA official split and setting.
- Shared/video setting: question-answering receives video frames, not only text.
- Causal cutoff: retrieval uses only memory ending at or before `question_time`.
- Video-RAG shard unit: eligible retrieval scope is same-video `shard_30m`.
- SuperMemory-VQA scope: official QA retrieval must use the full question
  `video_ids` pool, not only the primary `video_id`.
- EgoButler hierarchy: retrieval traverses `shard_30m -> clip_30s -> records`.
- Frame input: QA receives at most 32 frames sampled uniformly from the selected
  pre-question shard after applying the shared 1 Hz sensor-frame manifest.
- Sensor input: memory construction, retrieval, and QA use the same run-scoped
  `sensor_frames.jsonl`; no unselected RGB frame is available downstream.
- Frame scope: sampled frames must come from the selected shard video, and that
  shard video must be inside `question.video_ids` or the single-video fallback.
- Memory input: retrieved memory snippets are passed to Gemma together with the
  frames.
- Answer schema: prediction JSON keeps answerability, ranked choices, answer,
  confidence, and supporting memory IDs.
- Metrics: report official `Ans-F1`, `QA-Acc`, and `QA-MRR`.
- Copy policy: only metrics, logs, summaries, diagnostics, and small samples are
  copied back locally.

## Required Data Contract

`$SMVQA_DATA_ROOT` must already contain prepared files in the fixture schema:

- `sources.jsonl`
- `questions.jsonl`
- `labels.jsonl`

The current company-prepared root is:

```text
/groups/VTteam/datasets/SuperMemory-VQA/ingested/
```

Each `sources.jsonl` row should include available source signals:

- transcript spans
- captions
- OCR entries
- object detections
- frame metadata with `frame_ref` and `timestamp`
- optional `pose_samples`
- optional `gaze_samples`

Frame files for real Gemma QA must be readable under:

```text
$SMVQA_FRAME_ROOT/<video_id>/<frame_ref>.jpg
```

Accepted alternatives with the same stem: `.jpeg`, `.png`, `.webp`.

The prepared company root already contains the three required JSONL files. Raw
official dataset -> fixture-schema ingest remains outside this repo; rerun that
external ingest only when rebuilding or changing the prepared dataset.

`frame_metadata` is the candidate RGB inventory. Each run derives an at-most-1-Hz
timestamp-only inventory at
`$WORLDMM_OUTPUT_ROOT/manifests/sensor_frames.jsonl`. If prepared metadata is
already sparser than 1 Hz, the command does not synthesize missing frames.

Important SuperMemory-VQA metadata contract:

- Official `data/json/all_qa.json` and `qa_person_*.json` are QA metadata only.
  They must become `labels.jsonl` / `questions.jsonl`, never `sources.jsonl`.
- `sources.jsonl` must be built from source signals: video/MPS/transcript/caption
  preprocessing. QA fields such as answer, choices, answerability, and evidence
  must not enter memory builders.
- Preserve top-level SuperMemory `video_ids` on both `QALabelExample` and
  `QuestionRequest`. The primary `video_id` remains canonical for output
  compatibility, but retrieval uses `video_ids` as the allowed pool.
- Convert SuperMemory choices to stable IDs `A`, `B`, `C`, `D`. Store
  `QALabelExample.answer` as the correct choice ID, not `correct_answer` text.
- Use relative seconds consistently for source memory and `question_time`. Do
  not mix absolute session start timestamps with per-video relative spans unless
  the ingest explicitly normalizes all streams to the same timeline.
- Known metadata edge cases from the checked metadata sample: 4,853 QA rows,
  all with multi-video `video_ids`; most answer evidence videos differ from the
  primary question video. A single-video-only ingest will drop valid evidence.

## Remote Environment

Run only on approved company compute through bastion/head node.

Company infrastructure:

- bastion / ProxyJump: `sr-gpu-bastion`
- Slurm head node: `sr-gpu-head`
- GPU nodes: `gpu-vtt-queue-st-p5-node-[1-10]`
- GPU partition: `gpu-vtt-queue`
- GPU capacity: 8 x H100 80GB per node
- Slurm binaries: `/opt/slurm/bin/`
- CPU preprocessing/merge nodes: `cpu-prepro-queue-02-dy-m6i-node-[1-6]`

Use the operator's existing certificate-based SSH configuration. Do not place
certificates, private keys, tokens, or authentication options in this repo.

### Environment Injection

Set a unique run ID in each shell or Slurm job. Keep all outputs below a
run-scoped directory so reruns do not overwrite another experiment.

```bash
export WORLDMM_RUN_ID="${WORLDMM_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

export WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST=1
export WORLDMM_SMVQA_REMOTE_APPROVED=1

export BASTION_HOST=sr-gpu-bastion
export HEAD_NODE=sr-gpu-head
export WORLDMM_REMOTE_REPO=/repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b
export SMVQA_DATA_ROOT=/groups/VTteam/datasets/SuperMemory-VQA/ingested
export SMVQA_FRAME_ROOT="$SMVQA_DATA_ROOT/frames"
export GEMMA_MODEL_PATH=/repo/VTteam/bongh.park/gemma-4-e2b-it
export WORLDMM_OUTPUT_ROOT="/repo/VTteam/bongh.park/outputs/$WORLDMM_RUN_ID"

export WORLDMM_REMOTE_NODES=10
export WORLDMM_GPUS_PER_NODE=8
export WORLDMM_TRAIN_EPOCHS=1
export WORLDMM_TRAIN_BATCH_SIZE=8
export WORLDMM_TRAIN_HIDDEN_DIM=32
export WORLDMM_TRAIN_LEARNING_RATE=0.001
# Optional; must name an existing non-empty checkpoint.
# export WORLDMM_TRAIN_RESUME=/approved/path/to/spatial_student.pt
export WORLDMM_DDP_LAUNCHER='python -m torch.distributed.run'
export WORLDMM_TRITON_CACHE_ROOT=\
"${SLURM_TMPDIR:-/tmp}/worldmm-triton-${SLURM_JOB_ID}"

# Required only when downloading gated/missing models. Inject from the company
# secret mechanism or interactive shell; never save this value in the repo.
# export HF_TOKEN=...

# Memory-model paths must be on company storage. These models may be downloaded
# remotely if they are not already present.
export WORLDMM_MEMORY_MODEL_PATH=/repo/VTteam/bongh.park/outputs/models/qwen3-vl

# One config controls spatial encoder, projection head, token decoder, codec,
# selector weights, token budget, and quantization for the full QA run.
export WORLDMM_SPATIAL_EXPERIMENT_CONFIG=\
"$WORLDMM_REMOTE_REPO/configs/spatial/source_compact_v1.json"

# Required external artifacts for the typed spatial DAG. Either supply an
# executable extractor or an already-materialized causal teacher cache.
# Extractor contract: one process/GPU, accepting --rank, --world-size,
# --sensor-frame-manifest, --fixture, and rank-specific JSONL --out.
# export WORLDMM_GCUT3R_EXTRACTOR=/approved/path/to/extractor
export WORLDMM_TEACHER_CACHE_INPUT=/approved/path/to/teacher-cache.jsonl
export WORLDMM_STUDENT_SUPERVISION_INPUT=/approved/path/to/student-supervision.jsonl
export WORLDMM_UTILITY_CACHE_INPUT=/approved/path/to/selector-utility.jsonl
export WORLDMM_SPLIT_MANIFEST_INPUT=/approved/path/to/selector-splits.json
export WORLDMM_QA_EVIDENCE_INPUT=/approved/path/to/student-evidence-packs.jsonl
```

When running an in-process external provider rather than an extractor wrapper,
also set `WORLDMM_GCUT3R_CODE_PATH` and
`WORLDMM_GCUT3R_CHECKPOINT_PATH`. This repository does not download or import a
G-CUT3R implementation. `WORLDMM_QA_EVIDENCE_INPUT` must be produced by the
student-backed memory/retrieval path; the DAG will not substitute heuristic or
mock evidence.

Activate the checked-in remote virtual environment:

```bash
source /repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b/.venv/bin/activate
cd /repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b
python --version  # expected: Python 3.13
```

For repeated jobs, place the non-secret exports in an untracked file on company
storage, for example `$WORLDMM_REMOTE_REPO/.env.worldmm`, then source it from the
Slurm script. Keep `HF_TOKEN` outside that file unless the company secret
mechanism mounts it securely.

Local plan generation:

```bash
uv run worldmm-smvqa launch-remote \
  --dry-run \
  --config configs/remote.example.yaml \
  --out .omo/evidence/worldmm-smvqa/remote-plan
```

`--submit` still generates an approved plan only; local safety rules prevent
automatic SSH or expensive submission. It prints a ProxyJump command that runs
the generated head-node DAG submitter:

```bash
WORLDMM_SMVQA_REMOTE_APPROVED=1 \
uv run worldmm-smvqa launch-remote \
  --submit \
  --config configs/remote.example.yaml \
  --out .omo/evidence/worldmm-smvqa/remote-plan
```

Run the printed sync and SSH lines only after approval. The submitter prints the
run ID, output root, and final report-stage job ID. All five IDs are written to
`$WORLDMM_OUTPUT_ROOT/summary/dag_jobs.env`. Both generated `rsync` commands use
the SSH command's same default remote repository and exclude `.env*`; create
`.env.worldmm` only on company storage. `expected_outputs.json` describes the
preferred DAG's produced files, not legacy single-job artifacts. Capture the
returned summary in the experiment log.

## Remote Execution Handoff

### 1. Preflight

Run on `sr-gpu-head` or an allocated CPU preprocessing node:

```bash
source "$WORLDMM_REMOTE_REPO/.venv/bin/activate"
cd "$WORLDMM_REMOTE_REPO"
mkdir -p "$WORLDMM_OUTPUT_ROOT"/{manifests,chunks,source_refs,memory,retrieval,qa,metrics,diagnostics,logs,summary,ablation}

worldmm-smvqa preflight \
  --fixture "$SMVQA_DATA_ROOT" \
  --out "$WORLDMM_OUTPUT_ROOT/diagnostics/preflight.json"

export WORLDMM_SENSOR_FRAME_MANIFEST="$WORLDMM_OUTPUT_ROOT/manifests/sensor_frames.jsonl"
worldmm-smvqa build-memory \
  --stage sensor-frames \
  --config configs/remote.example.yaml \
  --fixture "$SMVQA_DATA_ROOT" \
  --out "$WORLDMM_SENSOR_FRAME_MANIFEST"

wc -l \
  "$SMVQA_DATA_ROOT/sources.jsonl" \
  "$SMVQA_DATA_ROOT/questions.jsonl" \
  "$SMVQA_DATA_ROOT/labels.jsonl" \
  | tee "$WORLDMM_OUTPUT_ROOT/logs/preflight-counts.txt"
```

Before expensive model work, additionally verify:

- question IDs are unique and match between `questions.jsonl` and `labels.jsonl`
- every `video_id`/`video_ids` entry exists in `sources.jsonl`
- every selected `frame_ref` resolves below `$SMVQA_FRAME_ROOT`
- manifest `sensor_rate_hz` is `1.0`, and selected count is no greater than raw
  source frame count
- `question_time` and source timestamps use the same relative-second convention
- the number of questions with no pre-question records in any eligible shard is
  known

`preflight` already validates the three JSONL schemas, unique/matching
question-label IDs, source-video scope, time ranges, answer/evidence contracts,
task/choice/store distributions, spatial/pose coverage, optional frame-file
resolution through `SMVQA_FRAME_ROOT`, and the derived at-most-1-Hz count. Any
reported error returns non-zero and must stop the DAG; warnings require review
but remain machine-readable in the report.

### 2. Materialize Teacher Data And Train The Typed Student

The primary staged DAG uses these artifacts:

```text
causal external G-CUT3R cache + explicit supervision
  -> typed student rows
counterfactual deletion utility + explicit split manifest
  -> byte-aware selector rows
  -> DDP typed student checkpoint
```

Teacher records are explicit `object`, `plane`, `portal`, `free_space`,
`landmark`, `event`, or `no_write` values with entity/instance IDs, local frame,
geometry, uncertainty, validity, provenance, and evidence refs. A cache row is
bound to its causal prefix by request/response/prefix SHA-256 values and previous
state reference. Validate before joining supervision:

```bash
python -m worldmm_smvqa.worldmm.gcut3r_teacher validate-cache \
  --cache "$WORLDMM_TEACHER_CACHE_INPUT"

python -m worldmm_smvqa.teacher_materializer \
  --teacher-cache "$WORLDMM_TEACHER_CACHE_INPUT" \
  --supervision "$WORLDMM_STUDENT_SUPERVISION_INPUT" \
  --out "$WORLDMM_OUTPUT_ROOT/training/student_teacher_cache.jsonl"
```

The materializer requires a complete one-to-one join, consistent feature/teacher/
geometry dimensions, disjoint train/validation groups, contiguous training
association targets, no unseen validation association target, and actual
canonical serialized bytes for every writable record.

Build the selector rows on a CPU node. Counterfactual mode is the production
contract; it must not silently fall back to lexical evidence overlap:

```bash
python -m worldmm_smvqa.spatial_selector_train prepare \
  --fixture "$SMVQA_DATA_ROOT" \
  --experiment "$WORLDMM_SPATIAL_EXPERIMENT_CONFIG" \
  --utility-cache "$WORLDMM_UTILITY_CACHE_INPUT" \
  --split-manifest "$WORLDMM_SPLIT_MANIFEST_INPUT" \
  --supervision-mode counterfactual \
  --out "$WORLDMM_OUTPUT_ROOT/training/selector_rows.jsonl"

WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST=1 \
python -m worldmm_smvqa.spatial_selector_train train \
  --config configs/remote.example.yaml \
  --input "$WORLDMM_OUTPUT_ROOT/training/selector_rows.jsonl" \
  --out "$WORLDMM_OUTPUT_ROOT/training/selector.json"
```

Each selector row carries hashes for the utility cache and split manifest.
Deletion-induced QA loss/score delta is combined with geometry coverage,
uncertainty reduction, pose information gain, surprise, redundancy, and actual
serialized bytes. Split validation rejects conflicting question, participant,
session, or video grouping.

Locally, only validate the materialized cache with a CPU forward/loss pass:

```bash
python -m worldmm_smvqa.spatial_train dry-run \
  --teacher-cache "$WORLDMM_OUTPUT_ROOT/training/student_teacher_cache.jsonl"
```

Real training is CUDA-only and remote-only. The generated `train_qa` Slurm stage
uses `srun` plus one `torch.distributed.run` agent per node, DDP,
`DistributedSampler`, all-reduced byte loss and validation metrics, and a
rank-zero atomic checkpoint:

```bash
python -m torch.distributed.run ... \
  -m worldmm_smvqa.spatial_train train \
  --config configs/remote.example.yaml \
  --teacher-cache "$WORLDMM_OUTPUT_ROOT/training/student_teacher_cache.jsonl" \
  --checkpoint "$WORLDMM_OUTPUT_ROOT/checkpoints/spatial_student.pt"
```

The source-compact commands below remain useful as the heuristic baseline and
artifact-contract probe. They are not a substitute for the external teacher,
student training, or student-backed evidence required by the typed DAG.

```bash
worldmm-smvqa build-memory \
  --stage chunk \
  --config configs/remote.example.yaml \
  --fixture "$SMVQA_DATA_ROOT" \
  --out "$WORLDMM_OUTPUT_ROOT/chunks/source_chunks.jsonl"

worldmm-smvqa build-memory \
  --stage source-memories \
  --config configs/remote.example.yaml \
  --fixture "$SMVQA_DATA_ROOT" \
  --out "$WORLDMM_OUTPUT_ROOT/source_refs/source_memories.jsonl"

worldmm-smvqa build-memory \
  --store episodic \
  --backend qwen \
  --config configs/remote.example.yaml \
  --fixture "$SMVQA_DATA_ROOT" \
  --out "$WORLDMM_OUTPUT_ROOT/memory/episodic.jsonl"

worldmm-smvqa build-memory \
  --stores semantic,visual \
  --backend qwen \
  --config configs/remote.example.yaml \
  --fixture "$SMVQA_DATA_ROOT" \
  --input "$WORLDMM_OUTPUT_ROOT/memory/episodic.jsonl" \
  --out "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv"

worldmm-smvqa build-memory \
  --stores spatial \
  --backend mock \
  --config configs/remote.example.yaml \
  --fixture "$SMVQA_DATA_ROOT" \
  --out "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv"
```

Create the retrieval manifest only after all four store files are present:

```bash
python - <<'PY' > "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json"
import json
import os
from pathlib import Path

root = Path(os.environ["WORLDMM_OUTPUT_ROOT"])
spatial = root / "memory/worldmm_sv/spatial.jsonl"
payload = {
    "source_memories": str(root / "source_refs/source_memories.jsonl"),
    "episodic_memory": str(root / "memory/episodic.jsonl"),
    "semantic_memory": str(root / "memory/worldmm_sv/semantic.jsonl"),
    "visual_memory": str(root / "memory/worldmm_sv/visual.jsonl"),
    "spatial_memory": {
        "path": str(spatial),
        "count": sum(1 for _ in spatial.open(encoding="utf-8")),
    },
    "spatial_experiment": str(
        root / "manifests/spatial_experiment.json"
    ),
    "spatial_compression": str(
        root / "memory/worldmm_sv/spatial.stats.jsonl"
    ),
}
print(json.dumps(payload, sort_keys=True))
PY
```

Memory-stage acceptance checks:

```bash
test -s "$WORLDMM_OUTPUT_ROOT/memory/episodic.jsonl"
test -s "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv/semantic.jsonl"
test -s "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv/visual.jsonl"
test -s "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv/spatial.jsonl"
test -s "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv/spatial.stats.jsonl"
test -s "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json"
test -s "$WORLDMM_OUTPUT_ROOT/manifests/spatial_experiment.json"
```

### 3. Build Retrieval Evidence

Full-dataset retrieval uses `retrieve-batch`; memory artifacts load once and all
question packs are written atomically.

For a single-question probe:

```bash
question_id="$(head -n 1 "$SMVQA_DATA_ROOT/questions.jsonl" | python -c 'import json,sys; print(json.load(sys.stdin)["question_id"])')"

worldmm-smvqa retrieve \
  --config configs/remote.example.yaml \
  --fixture "$SMVQA_DATA_ROOT" \
  --stores episodic,semantic,visual,spatial \
  --retrieval-protocol worldmm-smvqa \
  --max-frame-refs 32 \
  --input "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json" \
  --question "$question_id" \
  --out "$WORLDMM_OUTPUT_ROOT/retrieval/probe.json"
```

Do not start QA until the final evidence file has exactly one valid JSON object
per expected question and every pack has a `retrieval_trace`.

### 4. Verify Geometry Proofs Before Model QA

For supported geometry questions, retrieval must include the relevant spatial
relation plus both endpoint object records. The deterministic executor plans and
computes `distance`, `relative_direction`, `near`, `last_seen`, or `count`; it
does not ask Gemma to infer metric geometry from token prose.

An answerable proof contains:

- stable proof and entity IDs
- operation and coordinate frame
- computed value and propagated uncertainty
- provenance and evidence refs

Ambiguous identity, incompatible coordinate frames, missing wearer yaw,
unsupported provenance, absent geometry, or excessive uncertainty produces an
explicit unanswerable proof with a reason. The QA prompt receives only these
proof objects, and a prediction may retain only known answerable proof IDs.
Audit `geometry_proof_ids` and `geometry_proofs` in sample predictions before
accepting metric results.

### 5. One-Node Real QA Probe

Run one small real-model probe before multi-node execution:

```bash
export WORLDMM_REMOTE_NODES=1
export WORLDMM_GPUS_PER_NODE=8

python -m torch.distributed.run \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 8 \
  -m worldmm_smvqa.qa_transformers \
  --model "$GEMMA_MODEL_PATH" \
  --fixture "$SMVQA_DATA_ROOT" \
  --evidence "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.probe.jsonl" \
  --out "$WORLDMM_OUTPUT_ROOT/qa/predictions.probe.jsonl"
```

Acceptance checks:

- every rank shard completes
- merged prediction count equals probe evidence count
- sampled frame files exist
- malformed model JSON is retried twice; failure preserves rank checkpoint and
  raw output for restart/debugging
- GPU memory and runtime are recorded in `$WORLDMM_OUTPUT_ROOT/logs`

### 6. Multi-Node Slurm Shape

Use generated `remote-plan/submit_worldmm_smvqa_dag.sh`. It submits five
fail-closed stages:

```text
preflight_ingest  CPU
  -> teacher_extract  GPU
  -> merge_utility    CPU
  -> train_qa         GPU
  -> metrics_report   CPU
```

Dependencies are `afterok`, so a failed preflight, teacher cache, materializer,
utility/split validation, student train, or QA blocks every later stage. The
submitter uses a run-scoped no-clobber lock, rejects non-numeric
`sbatch --parsable` output, and atomically writes all stage IDs to
`$WORLDMM_OUTPUT_ROOT/summary/dag_jobs.env`. External inputs required by a stage
are checked with shell `${VAR:?message}` guards. The submitter independently
requires and exports `WORLDMM_SMVQA_REMOTE_APPROVED=1`, and rejects any
`WORLDMM_OUTPUT_ROOT` that does not end exactly in `/$WORLDMM_RUN_ID`.

GPU stages default to `WORLDMM_REMOTE_NODES=10` and
`WORLDMM_GPUS_PER_NODE=8`; teacher/train inherit that full allocation. For an
approved bounded probe, explicitly lower `WORLDMM_TEACHER_NODES`,
`WORLDMM_TRAIN_NODES`, and their per-node GPU variables. Training starts one
`torchrun` agent per node through `/opt/slurm/bin/srun`. Teacher extraction
instead starts one process per GPU, divides allocated CPUs across those workers,
passes explicit global rank/world size plus the generated 1 Hz sensor-frame
manifest, and requires one non-empty rank-specific JSONL shard per worker.

Target Slurm shape:

```bash
#SBATCH --partition=gpu-vtt-queue
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --output=/repo/VTteam/bongh.park/outputs/%x-%j.out
#SBATCH --error=/repo/VTteam/bongh.park/outputs/%x-%j.err

MASTER_ADDR="$(/opt/slurm/bin/scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR MASTER_PORT
export REMOTE_JOB_ID_OR_PROCESS_REF="$SLURM_JOB_ID"

/opt/slurm/bin/srun \
  --nodes="$SLURM_NNODES" \
  --ntasks="$SLURM_NNODES" \
  --ntasks-per-node=1 \
  bash -lc '
    source /repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b/.venv/bin/activate
    cd /repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b
    python -m torch.distributed.run \
      --nnodes "$SLURM_NNODES" \
      --nproc-per-node 8 \
      --node-rank "$SLURM_NODEID" \
      --master-addr "$MASTER_ADDR" \
      --master-port "$MASTER_PORT" \
      -m worldmm_smvqa.qa_transformers \
      --model "$GEMMA_MODEL_PATH" \
      --fixture "$SMVQA_DATA_ROOT" \
      --evidence "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl" \
      --out "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl"
  '
```

This is the underlying GPU launcher shape. Before accepting a full 10-node run,
use explicit one-node and two-node overrides for bounded probes; keep the DAG
dependency and run-scoped output contracts unchanged.

### 7. Evaluate And Produce Diagnostics

After prediction count and uniqueness checks pass:

```bash
worldmm-smvqa evaluate \
  --config configs/remote.example.yaml \
  --pred "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl" \
  --labels "$SMVQA_DATA_ROOT/labels.jsonl" \
  --out "$WORLDMM_OUTPUT_ROOT/metrics/official_metrics.json"

worldmm-smvqa diagnose-spatial \
  --config configs/remote.example.yaml \
  --input "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl" \
  --labels "$SMVQA_DATA_ROOT/labels.jsonl" \
  --out "$WORLDMM_OUTPUT_ROOT/diagnostics/spatial_diagnostics.json"
```

The typed DAG evaluates its supplied student-backed E1 evidence only. Run E2/E3
separately with the same split/model/frame contract; the legacy single-job plan
contains those baseline lanes but does not replace the typed student path.

### 8. Build The Run Manifest And Report

The typed DAG writes metrics plus `summary/summary.txt`. After E1/E2/E3 and
diagnostics finish, create the complete run manifest from actual artifacts (the
legacy lane can generate this shape automatically):

```json
{
  "baseline_name": "WorldMM-SMVQA",
  "remote_status": "complete",
  "local_changes": ["<git commit SHA and summary>"],
  "remote_command": "<sbatch command>",
  "remote_job_reference": "<Slurm job ID>",
  "remote_artifact_path": "<WORLDMM_OUTPUT_ROOT>",
  "metrics": [
    {"name": "Ans-F1", "value": 0.0},
    {"name": "QA-Acc", "value": 0.0},
    {"name": "QA-MRR", "value": 0.0}
  ],
  "failure_reason": null,
  "not_copied_locally": [
    "full datasets",
    "model weights",
    "embeddings",
    "full evidence packs"
  ]
}
```

Replace placeholder metric values with the actual output. Generate the report:

```bash
uv run worldmm-smvqa report \
  --run-manifest "$WORLDMM_OUTPUT_ROOT/summary/remote_manifest.json" \
  --out "$WORLDMM_OUTPUT_ROOT/summary/final-report.md"
```

Copy back only the final report, metrics, diagnostics, logs, plots, and small
sample predictions/evidence packs.

## Experiment Matrix

Run these experiments under the same dataset split, model, frame root, and output
root family.

| ID | Purpose | Stores | Retrieval Protocol | QA Input | Expected Output |
| --- | --- | --- | --- | --- | --- |
| E0 | Remote smoke on tiny/prepared sample | all | `worldmm-smvqa` | mock or small real probe | pipeline sanity |
| E1 | Main WorldMM-SMVQA run | episodic,semantic,visual,spatial | `worldmm-smvqa` | Gemma + 32 frames + memory | official metrics |
| E2 | Spatial ablation | episodic,semantic,visual | `worldmm-smvqa` | same as E1 | delta vs E1 |
| E3 | Retrieval protocol ablation | episodic,semantic,visual,spatial | `legacy-round-robin` | same as E1 | delta vs E1 |
| E4 | Retrieval-only audit | all | `worldmm-smvqa` | no Gemma required | trace/causal/frame audit |
| E5 | Sample-level QA audit | all | `worldmm-smvqa` | Gemma + saved samples | inspect errors |

Minimum required run for a report: E1, E2, E3.

## Main Run Contract

E1 must use:

```bash
--stores episodic,semantic,visual,spatial \
--retrieval-protocol worldmm-smvqa \
--max-frame-refs 32
```

The typed DAG does not yet build student-backed evidence internally. The
operator must generate E1 evidence with the contract above and point
`WORLDMM_QA_EVIDENCE_INPUT` at it. The `train_qa` stage links that validated
artifact and runs:

```bash
python -m worldmm_smvqa.qa_transformers \
  --model "$GEMMA_MODEL_PATH" \
  --fixture "$SMVQA_DATA_ROOT" \
  --evidence "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl" \
  --out "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl"
```

`qa_transformers` resolves frames from `$SMVQA_FRAME_ROOT`.

## Metrics To Report

Official metrics:

- `Ans-F1`
- `QA-Acc`
- `QA-MRR`

Serialize these three metrics on the paper's `0–100` scale. Keep diagnostic
Memory-Recall@K values as `0–1` fractions.

Required diagnostics:

- total questions
- answerable / unanswerable counts
- evidence pack count
- retrieval causal-filter count
- causal violation count, expected `0`
- frame count distribution, expected max `32`
- selected store distribution
- protocol trace presence rate, expected `100%`
- spatial Memory Recall@K
- per-store Memory Recall@K
- spatial relation accuracy, if relation labels exist

Recommended slices:

- location/spatial questions
- OCR/text questions
- object/color/appearance questions
- long-horizon questions crossing 30m shard boundary
- answerable vs unanswerable

## Decision Criteria

Treat E1 as valid only if:

- no causal leakage appears in retrieval or frames
- all prediction rows parse as `PredictionRecord`
- every evidence pack has `retrieval_trace`
- no frame count exceeds 32
- QA used frame files plus retrieved memory text
- metrics were computed against official labels

Spatial memory is useful if:

- E1 improves location/spatial slices over E2
- E1 does not materially regress aggregate `QA-Acc` or `QA-MRR`
- retrieved spatial evidence appears in supporting IDs or top evidence for
  spatial questions

Retrieval policy is useful if:

- E1 improves over E3 on aggregate or target slices
- causal-filter and selected-clip traces remain valid

If E1 underperforms E2, inspect:

- spatial snippets too lexical/noisy
- store routing over-prioritizing spatial
- frame shard sampling not aligned with retrieved evidence
- Gemma prompt over-weighting retrieved memory text

If E1 underperforms E3, inspect:

- coarse-to-fine clip selection too narrow
- lexical scoring missing relevant records
- store order suppressing visual/OCR evidence
- selected 30m shard not matching answer evidence

## Expected Remote Artifacts

The typed staged DAG directly creates or links these paths under
`$WORLDMM_OUTPUT_ROOT`:

- `diagnostics/preflight.json`
- `diagnostics/teacher_cache.json`
- `manifests/sensor_frames.jsonl`
- `manifests/source_chunks.jsonl`
- `manifests/source_memories.jsonl`
- `teacher/cache.jsonl`
- `training/student_teacher_cache.jsonl`
- `training/selector_rows.jsonl`
- `training/selector.json`
- `checkpoints/spatial_student.pt`
- `retrieval/evidence_packs.jsonl` (link to validated student-backed input)
- `qa/predictions.jsonl`
- `metrics/official_metrics.json`
- `summary/dag_jobs.env`
- `summary/summary.txt`

The source-compact legacy/E1-E3 lanes additionally produce these artifacts when
they are run. Do not claim them from the typed DAG alone:

- `manifests/source_roots.txt`
- `manifests/question_ids.txt`
- `manifests/spatial_experiment.json`
- `chunks/source_chunks.jsonl`
- `source_refs/source_memories.jsonl`
- `memory/episodic.jsonl`
- `memory/worldmm_sv/semantic.jsonl`
- `memory/worldmm_sv/visual.jsonl`
- `memory/worldmm_sv/spatial.jsonl`
- `memory/worldmm_sv/spatial.stats.jsonl`
- `memory/memory_manifest.json`
- `diagnostics/spatial_diagnostics.json`
- `summary/job.json`
- `summary/slurm_job_id.txt`
- `summary/remote_manifest.json`
- `summary/final_report.md`
- `ablation/without_spatial/qa/predictions.jsonl`
- `ablation/without_spatial/metrics/official_metrics.json`
- `ablation/protocol_legacy_round_robin/qa/predictions.jsonl`
- `ablation/protocol_legacy_round_robin/metrics/official_metrics.json`

Copy back only:

- metrics
- logs
- diagnostics
- summaries
- small sample predictions/evidence packs

Do not copy back:

- full datasets
- model weights
- checkpoints
- full video/frame corpora
- sensitive company artifacts

## Failure Triage

If model download fails:

- verify `HF_TOKEN`
- verify `$WORLDMM_MODEL_ID`
- verify `$GEMMA_MODEL_PATH` is on approved storage

If QA fails on missing frames:

- inspect `$SMVQA_FRAME_ROOT/<video_id>/<frame_ref>.*`
- verify `sources.jsonl.frame_metadata.frame_ref`
- verify frame extraction finished before QA

If metrics are empty:

- verify `labels.jsonl`
- verify prediction question IDs match label question IDs
- verify no rank shard merge failure

If retrieval gives empty evidence:

- verify source chunks contain pre-question records
- verify `question_time` unit is seconds
- verify `QuestionRequest.video_ids` contains every allowed SuperMemory evidence
  video
- verify source `video_id` strings match the SuperMemory `video_ids` strings
- verify frame trace shard videos are inside `question.video_ids`

## Final Report Template

After remote run, create or update a remote manifest with:

- code commit hash
- remote command used
- remote job ID or process reference
- dataset split and setting
- model ID/path
- output root
- E1/E2/E3 metric table
- key diagnostic table
- failure reason, if any
- files copied back locally
- files intentionally not copied back

Then generate:

```bash
uv run worldmm-smvqa report \
  --run-manifest <remote_manifest.json> \
  --out .omo/evidence/worldmm-smvqa/final-remote-report.md
```

## Naming

Use `WorldMM-SMVQA` for this implementation.

Do not call it an exact Video-RAG, EgoButler, or WorldMM reproduction unless a
separate reproduction lane is run and implementation deltas are reported.
