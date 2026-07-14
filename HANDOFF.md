# WorldMM-SMVQA Experiment Handoff

| Field | Value |
| --- | --- |
| Page ID | SM-OPERATIONS-HANDOFF |
| Confluence parent | SM-OPERATIONS |
| Page role | Company execution, approval, and artifact handoff runbook |
| Status | Active preparation; no remote run executed |
| Last reviewed | 2026-07-12 |

> Canonical research design and current readiness status live under
> [docs/spatial-memory](docs/spatial-memory/README.md). This file contains only
> operational company-compute handoff instructions.

## Purpose

This document is the next-run handoff toward official SuperMemory-VQA
experiments. The current Goal is a sensor audit followed by a bounded offline
teacher-oracle object/location experiment, not a student contract probe, official
benchmark claim, or local completion report.

Goal: first measure whether causal, evidence-bound offline teacher object/place
records improve the object/location slice under the same actual-byte budget. Only
an Oracle Go result may authorize a minimal hybrid on-device student. Final QA still
uses the benchmark-compatible input contract:

```text
retrieved memory text + 32 uniformly sampled pre-question video frames
+ question + answer choices
```

## Current State

Local causal sensor, teacher-target, typed-memory, and proof contracts are ready.
The existing staged Slurm plan remains a legacy student artifact-contract scaffold;
it does not execute EXP-0005 and must not substitute for the current Goal. EXP-0005
still needs a company-side provider, semantic mask/place adapter, sensor coverage
report, and a reviewed launch config. Real data/model execution remains unverified.

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
  prefix hash chain, state continuity, independent camera intrinsics,
  pose/depth guidance, and no automatic
  download/import
- strict on-device observation contract for calibrated camera, optional depth/gaze,
  and trusted raw-IMU or online-causal-VIO pose
- selected teacher-point to evidence-bound object target compiler with explicit
  extent and uncertainty floors; semantic mask/place extraction remains external
- teacher-cache/supervision materializer with actual serialized-byte targets
  and train/validation leakage checks; geometry and association targets remain
  externally supplied
- deletion-based counterfactual QA utility, actual-byte value, and explicit
  participant/session/question split manifest with input hashes; this remains a
  separate P2 scaffold and is not an input to the generated DAG
- remote-only PyTorch feature-level candidate-head training with genuine
  `torch.distributed` DDP, distributed sampling/validation, and atomic resume
  checkpoints; this is not a raw RGB/IMU or device student, and CPU dry-run performs
  only a forward/loss check
- production external inference contract with typed-record byte-budget and
  checkpoint/sensor/record digest revalidation
- effective spatial experiment persisted in local/remote manifests
- Video-RAG causal shard eligibility
- EgoButler-style `shard_30m -> clip_30s -> memory records` retrieval
- WorldMM store routing with spatial-first route for location questions
- retrieval traces in every evidence pack
- QA prompt containing sampled frame manifest and retrieved memory JSON
- deterministic geometry executor for distance, direction, near, last-seen,
  last-location, and count, with inferred evidence/confidence gates, entity IDs,
  coordinate frame, uncertainty, provenance, evidence, and proof-ID validation
- real Gemma path using multimodal frame inputs when frame files exist
- distributed Qwen memory generation partitioned by video across ranks
- batch retrieval that loads memory artifacts once
- resumable atomic Gemma QA rank checkpoints and merge
- phased typed E1 retrieval, QA, evaluation, identity, and report path; E2/E3
  remain unimplemented
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
- `python -m worldmm_smvqa.qa_transformers --backend mock --evidence-lane heuristic ...`
- `worldmm-smvqa launch-remote --dry-run ...`
- lightweight SuperMemory metadata contract checks from
  `tests/test_supermemory_metadata_contract.py`

No real model/data download, teacher extraction, training, benchmark evaluation,
checkpoint creation, remote shell, or remote job occurred during local
preparation. Only code checks, tiny mock-fixture work, and dry-run plan generation
are in scope here.

### Execution Modes And Stop Conditions

Do not treat every command in this document as one executable lane. Declare one
mode in the run manifest before allocating compute:

| Mode | Current readiness | Allowed claim |
| --- | --- | --- |
| `teacher-oracle` | **BLOCKED** pending adapter and approval | EXP-0005 diagnostic only; never emits `student` / `E1` |
| `probe` | Deferred legacy learned contract run | Allowed only after EXP-0005 Go and architecture/config review; emits `contract_probe` / `PROBE` |
| `full` | Conditional learned E1 run | Requires a new full-profile approval and the same production contracts; emits `student` / `E1` |
| `official-e1-e2-e3` | **BLOCKED** | No official claim until all three experiment identities are complete |

Learned mode is valid only when `WORLDMM_SPATIAL_INFER_EXE` produces a
`production_ready=true` manifest whose digests are revalidated by the remote
stage. Stop before GPU submission if a required artifact or executable is
missing; never substitute mock, heuristic, or unrelated external evidence.

Run `pytest`, `ruff`, `basedpyright`, and generated-script `bash -n` immediately
before transfer; record the live results instead of preserving a stale test
count here. Main residual risks are the external G-CUT3R adapter, actual
teacher/supervision/inference/evidence artifacts, real model compatibility, and
prepared dataset/frame completeness.

Blocking gaps before claiming final-method reproduction:

- no verified company sensor coverage report, repository-owned G-CUT3R provider,
  or semantic mask/place adapter for EXP-0005
- no evidence that teacher object/place records improve object/location QA under
  the same actual-byte budget
- no raw RGB semantic encoder, native-sensor device integration, causal open-world
  association, or target-device profile; production student inference would still
  require `WORLDMM_SPATIAL_INFER_EXE`
- preferred typed DAG generates a probe/full student report, but does not yet
  run E2/E3, byte Pareto, or a matched official E1/E2/E3 report

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
- Frame integrity: selected frame bytes must match `frame_assets.sha256`, and
  student predictions must record non-empty, video-namespaced
  `input_frame_refs` as `<video_id>/<frame_ref>`.
- Model integrity: fingerprint all runtime files under the resolved Gemma model
  and memory-model paths. Both trees must be self-contained with no symlink;
  local-only `AutoConfig` and `AutoProcessor` loading must pass, and every shard
  named by any `*index.json` weight map must exist and be non-empty.
- Memory input: retrieved memory snippets are passed to Gemma together with the
  frames.
- Memory artifact integrity: student evidence lineage contains the exact
  `memory_manifest.json`, episodic, semantic, and visual SHA-256 values in
  addition to the typed-spatial digest. QA recomputes all four values from the
  manifest and referenced files before model invocation.
- Answer schema: prediction JSON keeps answerability, ranked choices, answer,
  confidence, supporting memory IDs, `input_frame_refs`, and `prompt_sha256`.
- Prompt audit: sampled-frame JSON carries `video_id`, `frame_ref`, and
  `timestamp`; prompt/resume versions are `qa-prompt-prediction-schema-v4` and
  `qa-resume-manifest-v5`.
- Metrics: report official `Ans-F1`, `QA-Acc`, and `QA-MRR`.
- Copy policy: only metrics, summaries, reviewed non-sensitive diagnostics,
  redacted lightweight logs or plots, and approved small samples are copied back
  locally.

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

EXP-0005 additionally requires a reviewed adapter from prepared frame assets and
platform metadata to `CausalSensorObservation`. Each selected RGB observation must
carry camera intrinsics independently of optional depth. Any pose used for online
proof must satisfy the trusted causal policy below; optional gaze must be an explicit
origin-plus-direction ray. The current three prepared JSONL files do not by themselves
prove that these signals or readable frame assets exist. Audit actual coverage before
requesting GPU time; do not synthesize missing calibration or mark offline SLAM causal.

Canonical `pose_samples` JSON is unit-explicit:

- `x`, `y`, `z`: meters; x/y are horizontal and z is vertical
- `roll_degrees`, `pitch_degrees`, `yaw_degrees`: degrees
- `pose_covariance_xyz_m_rpy_deg`: row-major 6x6 covariance for
  `[x_m, y_m, z_m, roll_deg, pitch_deg, yaw_deg]`; index 35 is yaw variance in
  degrees squared
- `coordinate_frame`: must equal the typed spatial record frame used by the
  direction proof
- `source` + `processing_mode`: runtime direction proofs trust only
  `(imu, raw)` or `(vio, online_causal)`
- `observed_through_time`: causal certificate satisfying
  `timestamp <= observed_through_time <= question_time`

Yaw 0 degrees faces +Y; positive yaw rotates toward +X, and +X is wearer-right
at yaw 0. The loader accepts legacy `roll`/`pitch`/`yaw`/`pose_covariance`
input aliases only for migration. All newly prepared or adapter-produced JSON
must serialize the canonical names above. Radian-named fields are rejected;
never pass radians under a degree field. Offline SLAM, ground-truth/model pose,
and a missing or future causal certificate cannot ground a direction answer.

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

Pin one unique run ID as a literal in the environment file and reuse it for
both preflight and run. Do not use `date`, command substitution, or a fallback:
separate submitter invocations would otherwise resolve different run IDs. Keep
all outputs below a run-scoped directory so reruns do not overwrite another
experiment. The block
below is the required non-secret content template for the company-side
`$WORLDMM_REMOTE_REPO/.env.worldmm`; exporting it only in a transient local
shell is insufficient.

```bash
# Replace once before preflight. The submitter rejects this placeholder.
export WORLDMM_RUN_ID=REPLACE_WITH_APPROVED_RUN_ID

export WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST=1
# Set only after the preflight review and explicit operator approval below.
# export WORLDMM_SMVQA_REMOTE_APPROVED=1

export BASTION_HOST=sr-gpu-bastion
export HEAD_NODE=sr-gpu-head
export WORLDMM_REMOTE_REPO=/repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b
export SMVQA_DATA_ROOT=/groups/VTteam/datasets/SuperMemory-VQA/ingested
export SMVQA_FRAME_ROOT="$SMVQA_DATA_ROOT/frames"
export GEMMA_MODEL_PATH=/repo/VTteam/bongh.park/gemma-4-e2b-it
export WORLDMM_OUTPUT_ROOT="/repo/VTteam/bongh.park/outputs/$WORLDMM_RUN_ID"

export WORLDMM_APPROVED_DATA_PREFIX=/groups/VTteam/datasets
export WORLDMM_APPROVED_REPO_PREFIX=/repo/VTteam/bongh.park
export WORLDMM_APPROVED_OUTPUT_PREFIX=/repo/VTteam/bongh.park/outputs

export WORLDMM_EXECUTION_PROFILE=probe
export WORLDMM_REMOTE_NODES=1
export WORLDMM_GPUS_PER_NODE=1
export WORLDMM_PROBE_FIXTURE="/repo/VTteam/bongh.park/probe-fixtures/$WORLDMM_RUN_ID"
export WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW=4096
export WORLDMM_TRAIN_EPOCHS=1
export WORLDMM_TRAIN_BATCH_SIZE=8
export WORLDMM_TRAIN_HIDDEN_DIM=32
export WORLDMM_TRAIN_LEARNING_RATE=0.001
# Optional; must name an existing non-empty checkpoint.
# export WORLDMM_TRAIN_RESUME=/approved/path/to/spatial_student.pt

# Both model directories must be self-contained, symlink-free, and already
# contain every config/processor/index-referenced runtime file before preflight.
export WORLDMM_MEMORY_MODEL_PATH=/repo/VTteam/bongh.park/outputs/models/qwen3-vl

# Required external artifacts for the typed spatial DAG. Either supply an
# executable extractor or an already-materialized causal teacher cache.
# Extractor contract: one process/GPU, accepting --sources, --frame-root,
# --sensor-frame-manifest, --rank, --world-size, and rank-specific JSONL --out.
# The trusted wrapper owns and records its G-CUT3R code/checkpoint provenance.
# export WORLDMM_GCUT3R_EXTRACTOR=/approved/path/to/extractor
export WORLDMM_TEACHER_CACHE_INPUT=/approved/path/to/teacher-cache.jsonl
export WORLDMM_STUDENT_SUPERVISION_INPUT=/approved/path/to/student-supervision.jsonl
# Required for production learned inference; exact CLI contract is below.
export WORLDMM_SPATIAL_INFER_EXE=/approved/path/to/production-spatial-infer
```

`WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST=1` only unlocks code paths that are
forbidden on the development host. `WORLDMM_SMVQA_REMOTE_APPROVED=1` only permits
plan submission/stage execution. Neither is run authorization. The `run` phase
also requires the post-preflight, owned, permission-restricted approval JSON
whose profile, resources, environment, and input digests match exactly.

The three approved-prefix values may be overridden only to reviewed company
roots. They are exported to every stage, resolved into `env_contract.json`, and
therefore bound by approval; changing one after preflight fails the run.

The typed DAG has no external-evidence bypass lane or alternate evidence result
class. Any future adapter must satisfy the same checkpoint/typed-memory/
inference/evidence lineage gates, enter the existing `student` path, and emit
the required `result_class=student`; it must not introduce a weaker class.

Every `/approved/path/...` value above is a placeholder, not a runnable default.
Replace it with an approved company-storage path; preflight records and hashes
each production input. Fail closed if path, digest, schema, producer run, or row
count is unknown. External paths must resolve below the configured approved data
or repository prefix; symlink escape is rejected.

Probe mode requires teacher cache and student supervision produced only from the
reduced probe sources. Full-dataset cache/supervision is invalid for a probe.
The fixture itself must be outside `$WORLDMM_OUTPUT_ROOT`; otherwise preflight
rejects it so an external executable cannot discover question/label files by
walking its output directory. In cache mode, preflight rejects cache video IDs
outside the probe fixture, requires cache requests to cover exactly the sensor
manifest's selected `(video_id, frame_ref, timestamp)` observations with no
missing or extra request, and requires a complete one-to-one join between every
typed teacher record and supervision key. It hashes and schema-checks these
artifacts but cannot prove how an externally supplied cache/supervision file was
created; this semantic-origin check remains a current limitation. Prefer
probe-time extraction plus separately materialized probe supervision.

This repository does not download or import G-CUT3R or model artifacts. External
QA evidence is outside the learned DAG; the DAG builds its own evidence from the
validated spatial inference executable.

### Immutable Run Identity And Input Inventory

The local reviewed tree is the transfer authority. Pin and verify it before
generating the plan:

```bash
: "${WORLDMM_CODE_SHA:?set the approved 40-character git SHA}"

local_repo="$(git rev-parse --show-toplevel)"
cd "$local_repo"
test "$(git rev-parse HEAD)" = "$WORLDMM_CODE_SHA"
test -z "$(git status --porcelain)"
```

Run the generated repository rsync with `--delete`. It mirrors this tree while
excluding `.git`, `.venv`, `.omo`, and `.env*`; remote Git HEAD is not execution
authority and no remote checkout follows the sync. The preflight stage creates
the immutable `$WORLDMM_OUTPUT_ROOT/code_snapshot/`, copying `src/`, `configs/`,
generated `remote-plan/*.sh`, `pyproject.toml`, and `uv.lock`. It fingerprints
that snapshot, datasets, manifests, all model runtime files recursively,
executables, supervision, and resume input into
`diagnostics/preflight_inputs.sha256`.
The approved `run` phase must invoke the snapshot's DAG submitter; it then uses
the snapshot's stage runner and Python source via `PYTHONPATH`, while reusing
only the preserved shared
`$WORLDMM_REMOTE_REPO/.venv`; live checkout source/config mutations cannot
change code used inside submitted run stages. The environment contract binds
the shared interpreter path/version and installed distribution name/version/
direct-URL inventory. Preflight also seals every virtual-environment file and
filename plus the resolved interpreter and stdlib/platstdlib roots; submitter,
stages, and finalization recheck those seals. Executable `.pth` lines and `.pth`
paths outside the virtual environment are rejected. `PYTHONHOME`/ambient
`PYTHONPATH` and user-site imports are disabled; run stages use only the exact
snapshot `src` path. Keep the managed base Python and shared virtual environment
unchanged for the run. Their bytes are verified, not copied into the code
snapshot. Preflight also writes the exact path/profile/budget/resource contract to
`diagnostics/env_contract.json`, snapshot checksums to
`diagnostics/deployed_code.sha256`, every regular file recursively below the
Gemma model path to `diagnostics/gemma_model.sha256`, and every selected frame's
content checksum to
`diagnostics/frame_assets.sha256`. It writes the corresponding filename-set
digests to `gemma_model.files.sha256`, `memory_model.files.sha256`, and
`frame_assets.files.sha256`, plus the recursive memory-model content manifest at
`memory_model.sha256`.

These preflight artifacts are the authoritative deployed inventory. Approval
binds their SHA-256 values. `env_contract.json` also binds CPU/GPU partitions
and every preflight, teacher, materialize, train, and report node/GPU/CPU/memory/
time variable. Every subsequent stage rechecks approval, the complete
environment contract, input hashes, and selected frame bytes; submission uses
the bound resource values and each stage independently checks its node/GPU,
CPU, and partition allocation before work starts.

Teacher allocation is mode-dependent. `WORLDMM_TEACHER_NODES` and
`WORLDMM_TEACHER_GPUS_PER_NODE` describe extractor capacity. With
`WORLDMM_GCUT3R_EXTRACTOR`, those values are the effective teacher allocation.
With a precomputed `WORLDMM_TEACHER_CACHE_INPUT`, the effective allocation is
always 1 node, 0 GPUs, and the CPU partition; the raw extractor-capacity values
do not reserve GPUs. `env_contract.json.effective_teacher_resources`, the
approval fields `teacher_nodes` / `teacher_gpus_per_node`, the generated
`sbatch` arguments, and the stage allocation check must all agree on that
effective value. For the cache-mode probe shown here, approve `1` / `0` even
when the generic GPU profile is `1` / `1`.

The untracked company-side `$WORLDMM_REMOTE_REPO/.env.worldmm` is mandatory for
both phases, including a one-off probe. Both model directories are cache-only:
do not set or pass `HF_TOKEN`, and do not download a missing runtime file during
preflight. The submitter accepts only an owner-owned regular `.env.worldmm` with
no group/world write bits, below the initially approved repository prefix. It
is trusted shell configuration, not passive data.

Before executing any generated ProxyJump/preflight SSH line, create or replace
the file on company storage, inspect every line, syntax-check it, restrict it to
the operator, and record its checksum in the approval ticket:

```bash
cd "$WORLDMM_REMOTE_REPO"
umask 077
env_tmp="$(mktemp "$WORLDMM_REMOTE_REPO/.env.worldmm.tmp.XXXXXX")"
chmod 600 "$env_tmp"
"${EDITOR:?set EDITOR}" "$env_tmp"  # paste and resolve the template above
bash -n "$env_tmp"
sed -n '1,240p' "$env_tmp"          # mandatory exact-content review
mv -T "$env_tmp" "$WORLDMM_REMOTE_REPO/.env.worldmm"

env_file="$WORLDMM_REMOTE_REPO/.env.worldmm"
test -f "$env_file" && test ! -L "$env_file"
test "$(stat -c %u "$env_file")" = "$(id -u)"
test "$(stat -c %a "$env_file")" = 600
sha256sum "$env_file"                # record this digest in the approval ticket
```

Confirm the pinned value is literal and stable before either phase:

```bash
source "$WORLDMM_REMOTE_REPO/.env.worldmm"
test "$WORLDMM_RUN_ID" != REPLACE_WITH_APPROVED_RUN_ID
case "$WORLDMM_RUN_ID" in *'$('*|'`'*) exit 1 ;; esac
printf 'pinned run: %s\noutput: %s\nprobe: %s\n' \
  "$WORLDMM_RUN_ID" "$WORLDMM_OUTPUT_ROOT" "$WORLDMM_PROBE_FIXTURE"
```

Do not put credentials, tokens, certificates, or shell commands unrelated to
this run in the file. Repeat review and checksum capture after any edit. The
generated repository and plan syncs exclude `.env*`, so this company-side file
is neither overwritten nor supplied by the local tree.

Local plan generation:

```bash
export BASTION_HOST=sr-gpu-bastion
export HEAD_NODE=sr-gpu-head
: "${BASTION_HOST:?set local ProxyJump host}"
: "${HEAD_NODE:?set local Slurm head host}"

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

Run the printed sync and shared-repository preflight SSH line only after the
mandatory `.env.worldmm` checks above and after explicitly enabling the
execution flag; this still does not authorize the run phase. Default
`WORLDMM_DAG_PHASE=preflight` submits one CPU job and writes
`summary/dag_jobs.preflight.env`; reviewed `WORLDMM_DAG_PHASE=run` writes the
seven run-stage IDs to `summary/dag_jobs.env`, but that second invocation must
use the preflight-created snapshot submitter. Repository sync uses `--delete`
and excludes `.git`, `.venv`, `.omo`, and `.env*`; plan sync excludes `.env*`.
Plan generation removes stale `run_worldmm_smvqa.sh` and emits only the phased
DAG submitter/stage runner. Capture both phase summaries.

After sync, activate the preserved remote virtual environment:

```bash
source "$WORLDMM_REMOTE_REPO/.venv/bin/activate"
cd "$WORLDMM_REMOTE_REPO"
python --version  # expected: Python 3.13
```

## Remote Execution Handoff

### 0. Prepare The Reduced Probe Fixture

`probe` refuses the full fixture. Pin a reviewed question with existing frame
files and an expected answer/proof, then create the reduced fixture before
preflight:

```bash
: "${WORLDMM_PROBE_QUESTION_ID:?set an approved probe question ID}"
export WORLDMM_PROBE_QUESTION_ID
export WORLDMM_PROBE_FIXTURE="/repo/VTteam/bongh.park/probe-fixtures/$WORLDMM_RUN_ID"

python - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["SMVQA_DATA_ROOT"])
out = Path(os.environ["WORLDMM_PROBE_FIXTURE"])
question_id = os.environ["WORLDMM_PROBE_QUESTION_ID"]
out.mkdir(parents=True, exist_ok=True)

def rows(name):
    return [json.loads(line) for line in (source / name).read_text().splitlines()]

questions = [row for row in rows("questions.jsonl") if row["question_id"] == question_id]
labels = [row for row in rows("labels.jsonl") if row["question_id"] == question_id]
assert len(questions) == len(labels) == 1, "probe question/label must match once"
video_ids = set(questions[0].get("video_ids") or [questions[0]["video_id"]])
sources = [row for row in rows("sources.jsonl") if row["video_id"] in video_ids]
assert sources, "probe source rows missing"

for name, selected in (
    ("questions.jsonl", questions),
    ("labels.jsonl", labels),
    ("sources.jsonl", sources),
):
    (out / name).write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected),
        encoding="utf-8",
    )
PY
```

Choose a question whose selected source set stays within the probe limits. The
generated preflight verifies that this fixture is outside the run output,
non-empty, smaller than `$SMVQA_DATA_ROOT`, at most 4 source rows, and at most
600 total `frame_metadata` entries. Each of its three fixture JSONL files is at
most 64 MiB. Supervision is at most 64 MiB and 10,000 rows; a precomputed teacher
cache is at most 256 MiB; exact cache/supervision materialization is at most
10,000 rows. Probe typed output is at most 10,000 records and 16 MiB in addition
to the per-window byte budget. All limits fail closed. Full mode ignores these
probe-only caps and uses the full root.

The reviewed expected answer/proof is currently an operator acceptance record,
not a field bound by the generated approval JSON. Store it in the approved
ticket or a separately signed sidecar and compare it manually after QA. Passing
the DAG alone does not prove the expected semantic answer/proof matched.

### 1. Preflight-Only, Approval, Then Scale

Run on `sr-gpu-head` or an allocated CPU preprocessing node:

These commands are an operator precheck in a disposable scratch directory. They
must not create `$WORLDMM_OUTPUT_ROOT`: the generated `preflight` phase in
Section 6 requires that run root to be absent so it can create it atomically.

```bash
source "$WORLDMM_REMOTE_REPO/.venv/bin/activate"
cd "$WORLDMM_REMOTE_REPO"
export RUN_FIXTURE="$WORLDMM_PROBE_FIXTURE"
WORLDMM_PRECHECK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/worldmm-precheck.XXXXXX")"
trap 'rm -rf "$WORLDMM_PRECHECK_ROOT"' EXIT

worldmm-smvqa preflight \
  --fixture "$RUN_FIXTURE" \
  --out "$WORLDMM_PRECHECK_ROOT/preflight.json"

export WORLDMM_SENSOR_FRAME_MANIFEST="$WORLDMM_PRECHECK_ROOT/sensor_frames.jsonl"
worldmm-smvqa build-memory \
  --stage sensor-frames \
  --config configs/remote.example.yaml \
  --fixture "$RUN_FIXTURE" \
  --out "$WORLDMM_SENSOR_FRAME_MANIFEST"

wc -l \
  "$RUN_FIXTURE/sources.jsonl" \
  "$RUN_FIXTURE/questions.jsonl" \
  "$RUN_FIXTURE/labels.jsonl" \
  | tee "$WORLDMM_PRECHECK_ROOT/preflight-counts.txt"
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

The generated preflight phase queues only its CPU job. Review its report, sensor
manifest, `preflight_inputs.sha256`, `env_contract.json`, and
`deployed_code.sha256`, `gemma_model.sha256`, `frame_assets.sha256`, and the
spatial inference contract. Confirm that `env_contract.json` contains the
reviewed partitions and every stage's node/GPU/CPU/memory/time values. Then
create an untracked, operator-owned approval file below
`$WORLDMM_REMOTE_REPO/.omo/approvals/` and `chmod 600` it. Minimum probe JSON for
an existing teacher cache:

```bash
sha256sum \
  "$WORLDMM_OUTPUT_ROOT/diagnostics/preflight_inputs.sha256" \
  "$WORLDMM_OUTPUT_ROOT/diagnostics/env_contract.json"
```

```json
{
  "run_id": "<WORLDMM_RUN_ID>",
  "profile": "probe",
  "nodes": 1,
  "gpus_per_node": 1,
  "teacher_nodes": 1,
  "teacher_gpus_per_node": 0,
  "train_nodes": 1,
  "train_gpus_per_node": 1,
  "approved": true,
  "approver": "<operator identity>",
  "preflight_inputs_sha256": "<sha256 of preflight_inputs.sha256>",
  "env_contract_sha256": "<sha256 of env_contract.json>"
}
```

When an extractor is used, `teacher_gpus_per_node` must match its allocation.
The run submitter validates ownership, permissions, run/profile/allocation, and
`WORLDMM_APPROVER`, then stores the approval SHA-256 in `dag_jobs.env`. Every run
stage repeats these checks and verifies the preflight input list, exact resource
environment contract, selected frame-content manifest, and stage node/GPU
allocation before executing.

Use the phased `probe` commands in Section 6. That profile rejects more than one
node or one GPU; an existing teacher cache is linked on CPU. Scale only after a
successful probe and new approval.

### 2. Materialize Teacher Data And Train The Typed Student

The primary staged DAG uses these artifacts:

```text
causal external G-CUT3R cache + explicit supervision
  -> typed student rows -> DDP typed student checkpoint
  -> production spatial inference executable -> typed memory -> evidence
```

Teacher records are explicit `object`, `plane`, `portal`, `free_space`,
`landmark`, `event`, or `no_write` values with entity/instance IDs, local frame,
geometry, uncertainty, validity, provenance, and evidence refs. A cache row is
bound to its causal prefix by request/response/prefix SHA-256 values and previous
state reference. Validate before joining supervision:

Production teacher input may use pose guidance sourced from `imu`, `vio`, or
`slam`. A cache or extractor shard containing
`pose_guidance.source=ground_truth` is rejected during preflight/merge.

```bash
python -m worldmm_smvqa.worldmm.gcut3r_teacher validate-cache \
  --cache "$WORLDMM_TEACHER_CACHE_INPUT"

python -m worldmm_smvqa.teacher_materializer \
  --teacher-cache "$WORLDMM_TEACHER_CACHE_INPUT" \
  --supervision "$WORLDMM_STUDENT_SUPERVISION_INPUT" \
  --out "$WORLDMM_OUTPUT_ROOT/training/student_teacher_cache.jsonl"
```

The materializer enforces a complete one-to-one join, compatible dimensions,
disjoint train/validation groups, valid association targets, and canonical bytes.
For a precomputed cache, preflight and merge also require exact equality between
cache request `(video_id, frame_ref, timestamp)` tuples and selected sensor
observations.

Extractor mode launches one process per allocated GPU. Every rank must create
exactly one non-empty `rank-*.jsonl` shard and no additional JSONL file. The
merged request multiset must equal the selected sensor observations exactly;
missing work, extra work, and a duplicated request across ranks all fail. The
approved extractor is a trusted wrapper that owns G-CUT3R code/checkpoint
loading and records their provenance; the repository does not inject or inspect
an internal G-CUT3R implementation.

`WORLDMM_STUDENT_SUPERVISION_INPUT` is UTF-8 JSONL with one row per teacher
`(observation_id, memory_id)` key:

```json
{"observation_id":"obs-0001","memory_id":"mem-0001","group_id":"participant-001/session-001","split":"train","features":[0.1,0.2],"teacher_embedding":[0.3,0.4],"geometry_target":[1.0,2.0,3.0],"association_target":0}
```

`features`, `teacher_embedding`, and `geometry_target` are non-empty finite
arrays whose dimensions stay consistent per field across the file. Both
`train` and `validation` rows are required. A `group_id` cannot cross splits;
training association IDs must be contiguous from zero; validation may reference
only training-known association IDs. These vectors and associations are
external supervision: the repository validates and joins them but does not
derive their semantic correctness.

Locally, only validate the materialized cache with a CPU forward/loss pass:

```bash
python -m worldmm_smvqa.spatial_train dry-run \
  --teacher-cache "$WORLDMM_OUTPUT_ROOT/training/student_teacher_cache.jsonl"
```

Real training is CUDA-only and remote-only. The generated `train` stage
uses `srun` plus one `torch.distributed.run` agent per node, DDP,
`DistributedSampler`, all-reduced byte loss and validation metrics, and a
rank-zero atomic checkpoint:

```bash
export WORLDMM_EXECUTION_REPO="$WORLDMM_OUTPUT_ROOT/code_snapshot"
test -s "$WORLDMM_EXECUTION_REPO/configs/remote.example.yaml"
```

```bash
python -m torch.distributed.run ... \
  -m worldmm_smvqa.spatial_train train \
  --config "$WORLDMM_EXECUTION_REPO/configs/remote.example.yaml" \
  --teacher-cache "$WORLDMM_OUTPUT_ROOT/training/student_teacher_cache.jsonl" \
  --checkpoint "$WORLDMM_OUTPUT_ROOT/checkpoints/spatial_student.pt"
```

#### Production Spatial Inference Contract

Learned mode requires the operator-supplied executable. The repo must recompute
checkpoint, sensor, and records digests after it returns. Preflight requires
`--contract-version` to print exactly `worldmm-spatial-infer-v1` and
`--self-test` to print exactly `worldmm-spatial-infer-v1:self-test-ok`. The
self-test is lightweight CLI, schema, and canonical-writer conformance before
teacher/training submission; it is not a model-accuracy test:

```bash
: "${WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW:?set approved per-window budget}"
test -x "$WORLDMM_SPATIAL_INFER_EXE"
spatial_infer_clean() {
  env \
    -u RUN_FIXTURE -u SMVQA_DATA_ROOT \
    -u WORLDMM_STUDENT_SUPERVISION_INPUT \
    -u WORLDMM_TEACHER_CACHE_INPUT \
    -u WORLDMM_APPROVAL_FILE -u WORLDMM_APPROVER \
    -u WORLDMM_APPROVAL_SHA256 -u GEMMA_MODEL_PATH \
    -u WORLDMM_MEMORY_MODEL_PATH -u WORLDMM_REMOTE_REPO \
    -u WORLDMM_OUTPUT_ROOT -u WORLDMM_SPATIAL_INFER_EXE \
    "$@"
}
test "$(spatial_infer_clean \
  "$WORLDMM_SPATIAL_INFER_EXE" --contract-version)" = \
  worldmm-spatial-infer-v1
test "$(spatial_infer_clean \
  "$WORLDMM_SPATIAL_INFER_EXE" --self-test)" = \
  worldmm-spatial-infer-v1:self-test-ok
sensed_sources="$WORLDMM_OUTPUT_ROOT/inference_inputs/sources.jsonl"
sensed_sources_sha256="$(sha256sum "$sensed_sources" | cut -d ' ' -f 1)"
frame_assets_sha256="$(sha256sum \
  "$WORLDMM_OUTPUT_ROOT/diagnostics/frame_assets.sha256" | cut -d ' ' -f 1)"
producer_sha256="$(sha256sum "$WORLDMM_SPATIAL_INFER_EXE" | cut -d ' ' -f 1)"
spatial_infer_clean "$WORLDMM_SPATIAL_INFER_EXE" \
  --checkpoint "$WORLDMM_OUTPUT_ROOT/checkpoints/spatial_student.pt" \
  --sources "$sensed_sources" \
  --sources-sha256 "$sensed_sources_sha256" \
  --frame-root "$WORLDMM_OUTPUT_ROOT/inference_inputs/frames" \
  --frame-assets-sha256 "$frame_assets_sha256" \
  --producer-sha256 "$producer_sha256" \
  --sensor-frame-manifest "$WORLDMM_SENSOR_FRAME_MANIFEST" \
  --out-records "$WORLDMM_OUTPUT_ROOT/memory/typed_memory.jsonl" \
  --out-manifest "$WORLDMM_OUTPUT_ROOT/memory/typed_memory.inference.json" \
  --byte-budget-per-window "$WORLDMM_SPATIAL_BYTE_BUDGET_PER_WINDOW"
```

Preflight applies `sensor_frames.jsonl`, then writes a sanitized mode-`0600`
`sources.jsonl`: transcript, transcript spans, captions, OCR, object labels, and
object detections are empty; source identity/time, pose/gaze, selected
`frame_refs`, and selected frame metadata remain, with frame descriptions
erased. It copies only selected frame files to
`inference_inputs/frames/<video_id>/`; questions and labels never enter that
directory. Every post-preflight stage receives this copied sensed frame root,
not the full original frame directory. Teacher extraction and spatial inference
run through approved, digest-bound trusted executables. A denylist unsets known
sensitive variables including fixture/data-root, run-output, supervision,
teacher-cache, approval, Gemma/memory-model, repository, and executable-path
variables; required source/frame/sensor/checkpoint/rank/output/budget values are
explicit arguments. This is not a sandbox and does not use `env -i`: ambient
`PATH`, `HOME`, `PYTHONPATH`, Slurm variables, and other non-denylisted state
remain available. Do not run an untrusted or adversarial executable under this
contract; it requires a separately hardened container/sandbox outside this DAG.

The manifest must contain `schema_version=1`, `production_ready=true`,
`result_class=student`, `producer=spatial-student`, `checkpoint_sha256`,
`sensor_sha256`, `sources_sha256`, `frame_assets_sha256`, `producer_sha256`,
`records_sha256`, `record_count`, `byte_budget_per_window`, `window_count`,
`max_window_bytes`, `window_seconds=30.0`, and total `actual_bytes`. The stage
independently recomputes all three adapter-input/producer digests and requires
the manifest echoes to match. Student QA receives the exact sanitized sources,
frame-assets manifest, and inference-producer paths, rehashes them again, and
rejects any origin mismatch before model invocation.

That `result_class=student` describes the typed inference artifact. The final run
manifest is still `contract_probe` for profile `probe`; it becomes `student`
only for profile `full`.

An independent repository validator parses every persisted row against the full
typed-record schema and requires canonical UTF-8 compact JSON with sorted keys,
one trailing newline per row, no blank rows, no persisted `no_write`, and unique
`memory_id`. Validation is streaming and each canonical row is capped at 1 MiB.
Budget groups are
`(source_video_id, floor(first_seen_time / 30.0))`; every group must fit the
approved per-window budget. Manifest counters and digests must exactly match the
validator's recount before retrieval.

Contextual grounding is also mandatory: `source_video_id` must name a sanitized
source; validity and first/last-seen times must stay inside that source's time
bounds. `observed`, `multi_view_fused`, and `human_confirmed` records require
non-empty `evidence_refs`. Typed-record evidence values are bare `frame_ref`
strings and every supplied ref must be a selected sensor frame for the same
`source_video_id`, with its timestamp inside the record's first/last-seen
interval. For those grounded provenance classes, minimum/maximum evidence time
must equal first/last seen and `observation_count` must equal the number of
unique evidence refs. This prevents backdating; window accounting also uses
`first_seen_time`, not validity start. Typed evidence differs intentionally from
QA prediction audit refs, which use `<video_id>/<frame_ref>`.

The source-compact commands below remain useful as the heuristic baseline and
artifact-contract probe. They are not a substitute for the external teacher,
student training, or student-backed evidence required by the typed DAG.

```bash
worldmm-smvqa build-memory \
  --stage chunk \
  --config "$WORLDMM_EXECUTION_REPO/configs/remote.example.yaml" \
  --fixture "$SMVQA_DATA_ROOT" \
  --out "$WORLDMM_OUTPUT_ROOT/chunks/source_chunks.jsonl"

worldmm-smvqa build-memory \
  --stage source-memories \
  --config "$WORLDMM_EXECUTION_REPO/configs/remote.example.yaml" \
  --fixture "$SMVQA_DATA_ROOT" \
  --out "$WORLDMM_OUTPUT_ROOT/source_refs/source_memories.jsonl"

worldmm-smvqa build-memory \
  --store episodic \
  --backend qwen \
  --config "$WORLDMM_EXECUTION_REPO/configs/remote.example.yaml" \
  --fixture "$SMVQA_DATA_ROOT" \
  --out "$WORLDMM_OUTPUT_ROOT/memory/episodic.jsonl"

worldmm-smvqa build-memory \
  --stores semantic,visual \
  --backend qwen \
  --config "$WORLDMM_EXECUTION_REPO/configs/remote.example.yaml" \
  --fixture "$SMVQA_DATA_ROOT" \
  --input "$WORLDMM_OUTPUT_ROOT/memory/episodic.jsonl" \
  --out "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv"

worldmm-smvqa build-memory \
  --stores spatial \
  --backend mock \
  --config "$WORLDMM_EXECUTION_REPO/configs/remote.example.yaml" \
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

The `student_infer_retrieve` stage validates production typed memory, builds the
memory manifest, and runs `retrieve-batch` against `$RUN_FIXTURE`. It writes
`retrieval/evidence_packs.jsonl` plus a lineage sidecar bound to checkpoint,
typed memory, inference manifest, config, sensor manifest, fixture, the memory
manifest itself, and the episodic, semantic, and visual store digests. The
lineage records content digests, not just manifest paths: changing any
referenced store after retrieval invalidates student QA even when evidence-pack
bytes are unchanged.

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

Ambiguous identity, incompatible coordinate frames, missing causally certified wearer yaw
or its degree-squared covariance,
unsupported provenance, absent geometry, or excessive uncertainty produces an
explicit unanswerable proof with a reason. The QA prompt receives only these
proof objects, and a prediction may retain only known answerable proof IDs.
Audit `geometry_proof_ids` and `geometry_proofs` in sample predictions before
accepting metric results.

For student evidence, QA reprojects `typed_memory.jsonl` through the canonical
typed-to-retrieval converter. Every spatial `EvidenceItem` must exact-match its
typed projection on memory ID, video, snippet, frame refs, start/end time, and
geometry before any proof runs. The persisted typed artifact remains
byte-budgeted and may omit objects, newer states, or events, so it is not a
complete entity index. Production count and last-seen abstain unless a separate
end-to-end completeness certificate is introduced and validated. Without that
certificate, pair operations require explicit entity IDs in the question; a
unique label inside retrieved top-k evidence is not enough. Pair operations
derive the actual local frame and return an unanswerable proof for cross-video
entities.

### 5. One-GPU Reduced-Fixture Real QA Probe

The generated `probe` DAG runs every GPU stage at 1 node x 1 GPU, forces
`$RUN_FIXTURE=$WORLDMM_PROBE_FIXTURE`, and passes `--require-frames` to real
Gemma QA. The question ID, expected answer/proof, and preflight fingerprints must
be operator-reviewed before approval; as noted in Section 0, the generated
approval schema does not bind the expected answer/proof itself.

Acceptance checks:

- exactly one prediction is produced for `WORLDMM_PROBE_QUESTION_ID`
- `input_frame_refs` is non-empty, resolves through the approved frame-assets
  manifest, uses `<video_id>/<frame_ref>`, and `prompt_sha256` is present;
  text-only fallback fails the probe
- the answer/proof behavior matches the reviewed expectation, including an
  explicit abstention when that is the expected result
- malformed model JSON gets two attempts total (one retry); failure preserves rank checkpoint and
  raw output for restart/debugging
- model, checkpoint, typed memory, memory manifest, episodic/semantic/visual
  stores, evidence, and prediction digests are present in the generated
  lineage and identity artifacts
- runtime comes from Slurm accounting; GPU-memory/VRAM telemetry is accepted
  only when a separate approved profiler was run and its artifact exists

Only after this complete learned probe passes may the operator approve full
scale with a new environment contract and approval file.

### 6. Phased Slurm DAG

The generated submitter defaults to a safe 1-node x 1-GPU profile and requires
two invocations:

```bash
source "$WORLDMM_REMOTE_REPO/.env.worldmm"
test "$WORLDMM_RUN_ID" != REPLACE_WITH_APPROVED_RUN_ID
export WORLDMM_DAG_PHASE=preflight
export WORLDMM_EXECUTION_PROFILE=probe
export WORLDMM_SMVQA_REMOTE_APPROVED=1
bash "$WORLDMM_REMOTE_REPO/remote-plan/submit_worldmm_smvqa_dag.sh"

# After preflight completes and its hashes/warnings are approved:
source "$WORLDMM_REMOTE_REPO/.env.worldmm"
grep -Fx "WORLDMM_RUN_ID=$WORLDMM_RUN_ID" \
  "$WORLDMM_OUTPUT_ROOT/summary/dag_jobs.preflight.env"
export WORLDMM_DAG_PHASE=run
export WORLDMM_SMVQA_REMOTE_APPROVED=1
export WORLDMM_APPROVAL_FILE="$WORLDMM_REMOTE_REPO/.omo/approvals/$WORLDMM_RUN_ID.json"
export WORLDMM_APPROVER='<same identity as approval JSON>'
bash "$WORLDMM_OUTPUT_ROOT/code_snapshot/remote-plan/submit_worldmm_smvqa_dag.sh"
```

The preflight phase submits only `preflight_ingest` and writes
`summary/dag_jobs.preflight.env`. The run phase verifies the saved input hashes,
then submits this `afterok` chain. The shared-repository submitter is valid only
for preflight. In `run`, it compares its own resolved path with the approved
snapshot path and fails closed when invoked from the shared repository:

```text
teacher_extract -> merge_materialize -> train -> build_memory
  -> student_infer_retrieve (1 node x 1 GPU) -> qa -> metrics_report
```

With an existing teacher cache, `teacher_extract` uses CPU; an extractor uses the
approved GPU allocation. Training, non-spatial memory construction, spatial
inference/retrieval, and QA are independently gated stages. Spatial inference is
always 1 node x 1 GPU. E2/E3 and independent stage retry remain **BLOCKED/TODO**.

Full scale requires a separate approval and explicit literal values. Create a
new `.env.worldmm`; do not override the probe file from a transient shell:

```bash
export WORLDMM_RUN_ID=20260711T000000Z-full  # replace with the approved ID
export WORLDMM_OUTPUT_ROOT="/repo/VTteam/bongh.park/outputs/$WORLDMM_RUN_ID"
export WORLDMM_DAG_PHASE=preflight
export WORLDMM_EXECUTION_PROFILE=full
export WORLDMM_REMOTE_NODES=10
export WORLDMM_GPUS_PER_NODE=8
```

Save those values in the mandatory file, repeat its owner/`0600`/content/checksum
review, use the same unchanged file for both phases, rerun the full-profile
preflight, then create a new approval JSON with its exact hashes and
teacher/train allocation. Never reuse the probe environment contract. Keep
`probe` until the one-GPU run passes.

### 6A. Monitoring, Cancellation, And Retry

```bash
test -x \
  "$WORLDMM_OUTPUT_ROOT/code_snapshot/remote-plan/submit_worldmm_smvqa_dag.sh"

source "$WORLDMM_OUTPUT_ROOT/summary/dag_jobs.preflight.env"
/opt/slurm/bin/squeue -j "$PREFLIGHT_JOB_ID"

source "$WORLDMM_OUTPUT_ROOT/summary/dag_jobs.env"
job_ids="$TEACHER_JOB_ID,$MATERIALIZE_JOB_ID,$TRAIN_JOB_ID,$BUILD_MEMORY_JOB_ID"
job_ids+=",$STUDENT_INFER_RETRIEVE_JOB_ID,$QA_JOB_ID,$REPORT_JOB_ID"
/opt/slurm/bin/squeue -j "$job_ids" --format='%.18i %.24j %.9T %.10M %.6D %R'
/opt/slurm/bin/sacct -j "$job_ids" \
  --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,MaxRSS
tail -n 100 "$WORLDMM_OUTPUT_ROOT"/logs/*.{out,err}
```

After a stage creates its started marker, any non-zero exit automatically writes
`summary/stage.<stage>.failure.json` with `schema_version`, `run_id`, `stage`,
`job_id`, and `exit_code`. Failures before the marker cannot create this file;
use Slurm stdout/stderr and accounting for those. Record the decisive log, input
digests, and next action in the operator ticket without modifying the generated
failure manifest.

During submission, each validated job ID is first written to
`summary/dag_jobs.<phase>.partial`. If a later `sbatch` call fails, the exit trap
invokes `/opt/slurm/bin/scancel` for all earlier numeric IDs. When any job was
submitted, the phase lock is retained whether cancellation succeeds or fails;
cancellation failure may leave live jobs. The lock is removed only when
submission failed before any job ID existed. A fully successful phase also
retains its lock. Confirm queue state, record cancellation outcome, and use a
new run ID instead of deleting the lock.

Retry rules:

- Never manually delete a successful phase's submit lock, partial teacher
  shards, checkpoints, or evidence to force an in-place retry.
- For changed code/config/input, use a new run ID and output root.
- Reuse/resume only when every manifest digest matches; QA then uses its atomic
  rank progress. The QA resume manifest directly binds `memory_manifest.json`
  and the evidence-lineage file. That lineage contains the individual
  episodic/semantic/visual digests, which QA revalidates against current store
  bytes before continuing.
- `WORLDMM_TRAIN_RESUME` may name an existing non-empty student checkpoint, but
  record its digest and parent run.
- Stage-only retry remains **BLOCKED/TODO**; otherwise start a new run.

Use `/opt/slurm/bin/scancel <approved-job-ids>` only with operator approval and
record the reason.

### 7. Evaluate And Produce Diagnostics

After prediction count and uniqueness checks pass:

```bash
worldmm-smvqa evaluate \
  --config "$WORLDMM_EXECUTION_REPO/configs/remote.example.yaml" \
  --pred "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl" \
  --labels "$RUN_FIXTURE/labels.jsonl" \
  --out "$WORLDMM_OUTPUT_ROOT/metrics/metrics.json"

worldmm-smvqa diagnose-spatial \
  --config "$WORLDMM_EXECUTION_REPO/configs/remote.example.yaml" \
  --input "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl" \
  --labels "$RUN_FIXTURE/labels.jsonl" \
  --out "$WORLDMM_OUTPUT_ROOT/diagnostics/spatial_diagnostics.json"
```

The typed DAG evaluates only its internally built student-backed evidence. Probe
labels those metrics `PROBE`; full labels them `E1`. E2/E3 runners are not
generated; implement them with the same split/model/frame contract before
comparison.

`evaluate` is part of `metrics_report`; `diagnose-spatial` above is a manual
post-run command. The generated DAG does not currently create
`spatial_diagnostics.json` or VRAM telemetry automatically. Treat either as
available only when its command or an approved profiler was explicitly run and
the resulting artifact exists.

### 8. Build The Run Manifest And Report

`metrics_report` validates all three QA metrics on the `0–100` scale and requires
every prediction to contain `input_frame_refs` and `prompt_sha256`. It derives the
fixture split ID and aggregate prompt digest, hashes approval, deployed code,
environment, frames, model, checkpoint, all memory artifacts, evidence, QA
resume state, predictions, labels, and metrics. Before evaluation it writes
`summary/finalization_inputs.sha256` over QA outputs, evidence/lineage, the
memory manifest, episodic/semantic/visual stores, typed memory, snapshot config,
sensor manifest, split files, and other run-critical inputs. It rechecks the
seal after evaluation and rehashes every finalization path again immediately
before publishing identity/report artifacts. Any mutation aborts finalization;
`summary/remote_manifest.json` is written last as the completion marker.

- `metrics/metrics.json`
- `summary/run_identity.json`
- `summary/remote_manifest.json`
- `summary/final_report.md`
- `summary/summary.txt`

Conditional failure artifact:

- `summary/stage.<stage>.failure.json` for non-zero exits after that stage's
  started marker; pre-marker failures exist only in Slurm logs/accounting

The generated manifest is profile-sensitive: `probe` yields
`result_class=contract_probe`, experiment `PROBE`; `full` yields
`result_class=student`, experiment `E1`. Filenames are profile-neutral, so trust
the embedded profile/result class. Do not manually replace digests.
`result_class=official` with `remote_status=complete` is
deliberately rejected until immutable E1/E2/E3 manifests are implemented. Remote
Git HEAD is never used as runtime identity. Its `remote_command` must identify
the snapshot run submitter,
`$WORLDMM_OUTPUT_ROOT/code_snapshot/remote-plan/submit_worldmm_smvqa_dag.sh`;
a shared-repository run command is invalid.

Copy back only the final report, metrics, reviewed non-sensitive diagnostics,
redacted lightweight logs or plots, and approved small prediction/evidence
samples.

## Experiment Matrix

Run these experiments under the same dataset split, model, frame root, and output
root family.

| ID | Purpose | Stores | Retrieval Protocol | QA Input | Expected Output |
| --- | --- | --- | --- | --- | --- |
| E0 | Remote smoke on tiny/prepared sample | all | `worldmm-smvqa` | mock or small real probe | pipeline sanity |
| E1 | Main WorldMM-SMVQA run | episodic,semantic,visual,spatial | `worldmm-smvqa` | Gemma + 32 frames + memory | student E1 metrics |
| E2 | Spatial ablation | episodic,semantic,visual | `worldmm-smvqa` | same as E1 | delta vs E1 |
| E3 | Retrieval protocol ablation | episodic,semantic,visual,spatial | `legacy-round-robin` | same as E1 | delta vs E1 |
| E4 | Retrieval-only audit | all | `worldmm-smvqa` | no Gemma required | trace/causal/frame audit |
| E5 | Sample-level QA audit | all | `worldmm-smvqa` | Gemma + saved samples | inspect errors |

Minimum student handoff: E1. An official report requires immutable E1/E2/E3
identities; that completion path is currently blocked.

Each experiment is a separate immutable identity, not only a different output
filename:

```text
<WORLDMM_RUN_ID>-E1-worldmm-all-stores
<WORLDMM_RUN_ID>-E2-worldmm-without-spatial
<WORLDMM_RUN_ID>-E3-legacy-round-robin-all-stores
```

Write `experiments/<experiment_id>/manifest.json` before QA. It must bind the
run/code/dataset/split/model/prompt/sensor-frame digests, student checkpoint,
exact stores, retrieval protocol, maximum frame count, typed-memory,
inference-manifest, and evidence digests, output paths, and result class
(`mock`, `heuristic`, or `student`). E1/E2/E3 comparisons
are invalid when any shared digest differs except the intentional store/protocol
ablation. The current typed DAG does not create these three identities; this is
a **BLOCKED/TODO** for `official-e1-e2-e3` mode.

## Main Run Contract

E1 must use:

```bash
--stores episodic,semantic,visual,spatial \
--retrieval-protocol worldmm-smvqa \
--max-frame-refs 32
```

The `student_infer_retrieve` stage builds E1 evidence and its lineage sidecar;
the dependent `qa` stage runs:

```bash
export WORLDMM_SENSOR_FRAME_MANIFEST="$WORLDMM_OUTPUT_ROOT/manifests/sensor_frames.jsonl"
export SMVQA_FRAME_ROOT="$WORLDMM_OUTPUT_ROOT/inference_inputs/frames"
test -s "$WORLDMM_SENSOR_FRAME_MANIFEST"
test -d "$SMVQA_FRAME_ROOT"

python -m worldmm_smvqa.qa_transformers \
  --model "$GEMMA_MODEL_PATH" \
  --fixture "$RUN_FIXTURE" \
  --evidence "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl" \
  --evidence-lane student \
  --evidence-lineage "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl.lineage.json" \
  --checkpoint "$WORLDMM_OUTPUT_ROOT/checkpoints/spatial_student.pt" \
  --typed-memory "$WORLDMM_OUTPUT_ROOT/memory/typed_memory.jsonl" \
  --inference-manifest "$WORLDMM_OUTPUT_ROOT/memory/typed_memory.inference.json" \
  --inference-sources "$WORLDMM_OUTPUT_ROOT/inference_inputs/sources.jsonl" \
  --inference-producer "$WORLDMM_SPATIAL_INFER_EXE" \
  --memory-manifest "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json" \
  --model-fingerprint "$WORLDMM_OUTPUT_ROOT/diagnostics/gemma_model.sha256" \
  --frame-assets-manifest "$WORLDMM_OUTPUT_ROOT/diagnostics/frame_assets.sha256" \
  --lineage-config "$WORLDMM_EXECUTION_REPO/configs/remote.example.yaml" \
  --sensor-frame-manifest "$WORLDMM_SENSOR_FRAME_MANIFEST" \
  --require-frames \
  --out "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl"
```

The generated stage exports
`SMVQA_FRAME_ROOT=$WORLDMM_OUTPUT_ROOT/inference_inputs/frames`; manual runs must
do the same. `qa_transformers` resolves frames only from that copied sensed root.
Student evidence lineage must declare `lane=student`,
`producer=spatial-student`, and evidence/checkpoint/typed-memory/inference-manifest/
config/sensor/data SHA-256 values. It must also contain
`memory_manifest_sha256`, `episodic_memory_sha256`,
`semantic_memory_sha256`, and `visual_memory_sha256`. QA reads the supplied
memory manifest, verifies that all four store paths are present, recomputes the
manifest and all four store bytes, and requires exact lineage matches. Typed
memory must match both the manifest path/hash and the production inference
manifest. The manifest must be schema version 1, contain only the four
fixed run-layout store paths, and use no symlink. Retrieval seals the manifest
and all four stores in `retrieval/memory_inputs.sha256` before reading them and
rechecks that seal immediately afterward.

Every student prediction records the exact `input_frame_refs` and
`prompt_sha256`. `input_frame_refs` uses `<video_id>/<frame_ref>`, while each
prompt frame-manifest row includes `video_id`, `frame_ref`, and `timestamp`.
Prompt and resume contracts are `qa-prompt-prediction-schema-v4` and
`qa-resume-manifest-v5`. The QA resume manifest binds model/runtime-file,
selected-frame, config, sensor, evidence, checkpoint, typed-memory, and
inference-manifest fingerprints, plus the sanitized inference sources and
inference-producer bytes. It also directly binds the memory-manifest
digest and evidence-lineage digest; the latter transitively binds individual
episodic/semantic/visual bytes and is revalidated before a student QA start or
resume. Missing frames fail instead of falling back to text-only QA. On success,
`qa/completed.json` binds the final predictions digest to the resume-manifest
digest and is rechecked before reporting.

E2/E3 must share E1's fixture, Gemma snapshot, prompt schema, sensor manifest,
and checkpoint-derived base memory; only the declared ablation may change.

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

- `code_snapshot/` (`src`, `configs`, `remote-plan`, `pyproject.toml`, `uv.lock`)
- `diagnostics/preflight.json`
- `diagnostics/preflight_inputs.sha256`
- `diagnostics/env_contract.json`
- `diagnostics/deployed_code.sha256`
- `diagnostics/deployed_code.files.sha256`
- `diagnostics/python_runtime.sha256`
- `diagnostics/python_runtime.files.sha256`
- `diagnostics/python_base_roots.tsv`
- `diagnostics/python_base_runtime.sha256`
- `diagnostics/python_base_runtime.files.sha256`
- `diagnostics/gemma_model.sha256`
- `diagnostics/gemma_model.files.sha256`
- `diagnostics/memory_model.sha256`
- `diagnostics/memory_model.files.sha256`
- `diagnostics/frame_assets.sha256`
- `diagnostics/frame_assets.files.sha256`
- `diagnostics/spatial_infer_contract.txt`
- `diagnostics/preflight.completed`
- `diagnostics/teacher_cache.json`
- `inference_inputs/sources.jsonl`
- `inference_inputs/frames/` (only selected copied frame assets)
- `manifests/sensor_frames.jsonl`
- `manifests/source_chunks.jsonl`
- `manifests/source_memories.jsonl`
- `teacher/cache.jsonl`
- `training/student_teacher_cache.jsonl`
- `checkpoints/spatial_student.pt`
- `memory/episodic.jsonl`
- `memory/worldmm_sv/semantic.jsonl`
- `memory/worldmm_sv/visual.jsonl`
- `memory/typed_memory.jsonl`
- `memory/typed_memory.inference.json`
- `memory/memory_manifest.json`
- `retrieval/evidence_packs.jsonl`
- `retrieval/evidence_packs.jsonl.lineage.json`
- `retrieval/memory_inputs.sha256`
- `retrieval/memory_inputs.json`
- `qa/predictions.jsonl`
- `qa/predictions.jsonl.manifest.json`
- `qa/completed.json`
- `metrics/metrics.json`
- `summary/dag_jobs.preflight.env`
- `summary/dag_jobs.env`
- `summary/finalization_inputs.sha256`
- `summary/run_identity.json`
- `summary/remote_manifest.json`
- `summary/final_report.md`
- `summary/summary.txt`
- `logs/*-*.out`
- `logs/*-*.err`

Conditional artifacts:

- `diagnostics/preflight_teacher_cache.json` only in cache mode: no
  `WORLDMM_GCUT3R_EXTRACTOR` and a valid `WORLDMM_TEACHER_CACHE_INPUT`
- `summary/stage.*.failure.json` only when a post-marker stage fails
- `summary/dag_submit.*.lock`, `summary/dag_jobs.*.partial`, and
  `summary/dag_submit.*.attempts` are submission/recovery ledgers retained after
  any attempted or completed phase

`diagnostics/spatial_diagnostics.json` and VRAM-profiler output are optional
manual artifacts, not guaranteed DAG outputs.

The only operator-created identity file is the untracked, permission-restricted
`$WORLDMM_REMOTE_REPO/.omo/approvals/$WORLDMM_RUN_ID.json`.

Copy back only:

- metrics
- redacted lightweight logs and plots
- reviewed non-sensitive diagnostics
- summaries
- approved small sample predictions/evidence packs

Do not copy back:

- full datasets
- model weights
- checkpoints
- full video/frame corpora
- sensitive company artifacts

## Failure Triage

Start with `summary/stage.<stage>.failure.json` when present and correlate its
`job_id`/`exit_code` with Slurm logs. If absent, determine whether the failure
occurred before the stage marker, during DAG submission, or outside the stage
runner; absence alone does not mean success.

If cache-only model preflight fails:

- do not download or repair the model inside the run
- verify `$GEMMA_MODEL_PATH` and `$WORLDMM_MEMORY_MODEL_PATH` already contain
  complete approved runtime files on company storage and contain no symlinks
- verify local-only `AutoConfig`/`AutoProcessor` loading, `config.json`,
  non-empty weights, every shard referenced by `*index.json`, and the recursive
  model fingerprints
- start a new preflight/run identity after an approved model directory is fixed

If QA fails on missing frames:

- inspect `$WORLDMM_OUTPUT_ROOT/inference_inputs/frames/<video_id>/<frame_ref>.*`
- verify `sources.jsonl.frame_metadata.frame_ref`
- verify selected-frame copy and `diagnostics/frame_assets.sha256` before QA

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

## Final Report Handoff

Use the automatically generated `summary/remote_manifest.json` and
`summary/final_report.md`; do not reconstruct either from shell history.
`remote_manifest.json` is authoritative. `final_report.md` is a derived view;
after copyback, regenerate it from the manifest and require byte equality:

```bash
uv run worldmm-smvqa report \
  --run-manifest summary/remote_manifest.json \
  --out /tmp/worldmm-final-report.md
cmp -s summary/final_report.md /tmp/worldmm-final-report.md
```

- Compare the shared run ID, profile-appropriate `PROBE` or `E1` experiment,
  result class, code/config/model/data/checkpoint/memory/evidence/prediction
  digests, and prompt/split identity with `summary/run_identity.json`.
- Require `summary/remote_manifest.json` to bind the exact
  `run_identity.json` SHA-256 and verify that digest from the file bytes.
- Verify the three metrics, remote job/process reference, output root, artifact
  locations, and copy/no-copy policy against `summary/remote_manifest.json`,
  `summary/final_report.md`, `metrics/metrics.json`, and the DAG job summaries.
  These operational fields are not validated by comparing them to
  `run_identity.json`.

## Naming

Use `WorldMM-SMVQA` for this implementation.

Do not call it an exact Video-RAG, EgoButler, or WorldMM reproduction unless a
separate reproduction lane is run and implementation deltas are reported.
