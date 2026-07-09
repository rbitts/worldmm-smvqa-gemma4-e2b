# WorldMM-SMVQA Experiment Handoff

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

Code is ready for remote experiment execution.

Implemented locally:

- 30s clip / 30m shard chunking
- source memories for captions, transcript, OCR, objects, frame metadata
- WorldMM stores: `episodic`, `semantic`, `visual`, `spatial`
- Video-RAG causal shard eligibility
- EgoButler-style `shard_30m -> clip_30s -> memory records` retrieval
- WorldMM store routing with spatial-first route for location questions
- retrieval traces in every evidence pack
- QA prompt containing sampled frame manifest and retrieved memory JSON
- real Gemma path using multimodal frame inputs when frame files exist
- DDP-aware `qa_transformers` shard/merge path
- remote dry-run launch plan

Verified locally only on tiny mock fixture:

- `ruff check .`
- `basedpyright`
- `pytest -q`
- `worldmm-smvqa smoke --fixture tests/fixtures/tiny_smvqa ...`
- `python -m worldmm_smvqa.qa_transformers --backend mock ...`
- `worldmm-smvqa launch-remote --dry-run ...`

No real model, dataset, benchmark evaluation, checkpoint, or remote job was run
locally.

## Official Parity Checklist

Before trusting remote numbers, verify all items below in the remote artifacts.

- Dataset split: use the intended SuperMemory-VQA official split and setting.
- Shared/video setting: question-answering receives video frames, not only text.
- Causal cutoff: retrieval uses only memory ending at or before `question_time`.
- Video-RAG shard unit: eligible retrieval scope is same-video `shard_30m`.
- EgoButler hierarchy: retrieval traverses `shard_30m -> clip_30s -> records`.
- Frame input: QA receives at most 32 frames sampled uniformly from the selected
  pre-question shard.
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

Open gap: raw official dataset -> this schema ingest is not implemented in this
repo. If `$SMVQA_DATA_ROOT` is not already prepared, implement or run that ingest
first on company compute.

## Remote Environment

Run only on approved company compute through bastion/head node.

Required variables:

```bash
export WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST=1
export WORLDMM_SMVQA_REMOTE_APPROVED=1
export BASTION_HOST=...
export HEAD_NODE=...
export REMOTE_JOB_LAUNCHER=...
export WORLDMM_REMOTE_REPO=...
export SMVQA_DATA_ROOT=...
export SMVQA_FRAME_ROOT=...  # optional, defaults to $SMVQA_DATA_ROOT/frames
export GEMMA_MODEL_PATH=...
export WORLDMM_OUTPUT_ROOT=...
export WORLDMM_REMOTE_NODES=...
export WORLDMM_GPUS_PER_NODE=...
export WORLDMM_DDP_LAUNCHER='python -m torch.distributed.run'
export WORLDMM_MODEL_ID=google/gemma-4-E2B-it
export REMOTE_JOB_ID_OR_PROCESS_REF=...
export HF_TOKEN=...
```

Remote setup:

```bash
uv sync --extra remote
```

Local dry-run only:

```bash
uv run worldmm-smvqa launch-remote \
  --dry-run \
  --config configs/remote.example.yaml \
  --out .omo/evidence/worldmm-smvqa/remote-plan
```

Submit only after explicit approval:

```bash
WORLDMM_SMVQA_REMOTE_APPROVED=1 \
uv run worldmm-smvqa launch-remote \
  --submit \
  --config configs/remote.example.yaml \
  --out .omo/evidence/worldmm-smvqa/remote-plan
```

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

The generated remote script already uses this for retrieval and runs:

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

Under `$WORLDMM_OUTPUT_ROOT`:

- `manifests/source_roots.txt`
- `manifests/question_ids.txt`
- `chunks/source_chunks.jsonl`
- `source_refs/source_memories.jsonl`
- `memory/episodic.jsonl`
- `memory/worldmm_sv/semantic.jsonl`
- `memory/worldmm_sv/visual.jsonl`
- `memory/worldmm_sv/spatial.jsonl`
- `memory/memory_manifest.json`
- `retrieval/evidence_packs.jsonl`
- `qa/predictions.jsonl`
- `metrics/official_metrics.json`
- `diagnostics/spatial_diagnostics.json`
- `summary/summary.txt`
- `ablation/without_spatial.json`
- `ablation/protocol_legacy_round_robin.json`

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
- verify same `video_id` across sources/questions/labels

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
