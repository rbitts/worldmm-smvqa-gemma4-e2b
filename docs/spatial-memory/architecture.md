# Architecture

| Field | Value |
|---|---|
| Page ID | SM-ARCHITECTURE |
| Status | Living specification |
| Last updated | 2026-07-12 |
| Current implementation | Source-compact baseline plus production typed handoff bridge |
| Target implementation | Guided transient geometry to learned typed persistent memory |

## Architecture Principle

Dense geometry is temporary computation. Long-term memory is an explicit,
versioned spatial database containing only records needed for geometry QA,
localization, and meaningful change history.

G-CUT3R or another geometry foundation model is a teacher or front-end. Its
recurrent state and dense point maps are not persistent memory.

## Memory Layers

```text
Fast pose memory
    recent alignment, motion, IMU/VIO state
    frequent updates, short lifetime

Transient geometry working state
    point maps, depth, confidence, candidate slots
    used while processing observations, then discarded

Persistent explicit map
    place, structure, object, free-space, and event records
    infrequent verified writes

Relocalization memory
    sparse position, viewing ray, descriptor, uncertainty

Evidence reservoir
    bounded crops or events for text, appearance, surprise, and uncertainty
```

The separation prevents tracking updates from overwriting auditable long-term
facts.

## Current Local Baseline

```text
prepared source streams
    -> causal at-most-1-Hz RGB inventory
    -> structured object, pose, and gaze geometry
    -> zone, object, relation, and trajectory candidates
    -> per-window token and actual-byte writer
    -> causal WorldMM retrieval
    -> deterministic geometry executor
    -> four-choice mock QA
```

The baseline is intentionally heuristic. It validates contracts and failure
boundaries before adding a large provider or learned writer.

Relevant implementation:

- `src/worldmm_smvqa/sensor_frames.py`
- `src/worldmm_smvqa/worldmm/spatial.py`
- `src/worldmm_smvqa/worldmm/spatial_compression.py`
- `src/worldmm_smvqa/retrieval.py`
- `src/worldmm_smvqa/worldmm/geometry_executor.py`

## Target Learned Path

```text
1 Hz RGB --------------------------+
native-rate IMU/VIO ---------------+-> guided geometry teacher
optional calibrated depth ---------+        |
                                              v
                                  causal transient geometry state
                                              |
                                              v
                                     typed candidate decoder
                         object / plane / portal / free-space
                              landmark / event / no-write
                                              |
                            QA utility, geometry novelty,
                         uncertainty reduction, pose information,
                           event surprise, redundancy, bytes
                                              |
                                              v
                                  hard actual-byte writer
                                              |
                                              v
                                  local-frame persistent DB
```

The trained checkpoint does not expose enough information for this repository's
generic student head to invent production typed geometry. The current handoff
therefore delegates type-specific decode and open-world association to an
approved executable with the exact `worldmm-spatial-infer-v1` contract:

```text
spatial_student.pt + sanitized sources + frame/sensor inventory
    -> WORLDMM_SPATIAL_INFER_EXE
    -> canonical typed_memory.jsonl
    -> typed_memory.inference.json
    -> repository schema, byte, digest, retrieval, and QA validation
```

The executable receives no questions or labels. Its sources input is the
run-scoped sanitized copy under `inference_inputs/`, not the full fixture.
Preflight first applies the at-most-1-Hz sensor manifest, clears transcript,
caption, OCR, and object fields, and erases selected-frame descriptions; it
retains source identity/time, pose/gaze, selected frame refs, and selected frame
timestamps. Only those selected frame files are copied to
`inference_inputs/frames/`, which becomes the frame root for every
post-preflight adapter and QA stage. This is a production bridge, not evidence
injection: repository retrieval builds the evidence packs from returned typed
records and binds their lineage to the checkpoint, inference manifest, memory
manifest, and exact episodic, semantic, visual, and typed-spatial bytes.

Relevant preparation code:

- `src/worldmm_smvqa/worldmm/gcut3r_teacher.py`
- `src/worldmm_smvqa/teacher_materializer.py`
- `src/worldmm_smvqa/spatial_train.py`
- `src/worldmm_smvqa/worldmm/typed_memory.py`

## Persistent Record Types

| Type | Minimum geometry | Primary purpose |
|---|---|---|
| Place/submap | Local SE(3) anchor, covariance, topology | Stable coordinate ownership and loop correction |
| Plane | Normal, offset, extent, uncertainty | Wall, floor, ceiling, support geometry |
| Portal | Position, orientation, extent, connected frames | Room transition and topology |
| Free space | Coarse polygon or tile, height, validity | Reachability and visibility |
| Object | Entity and instance IDs, centroid, extent, orientation | Metric and semantic object QA |
| Landmark | Position, viewing ray, descriptor, quality | Relocalization and association |
| Event | Kind, entities, before/after state, validity | Moved, appeared, disappeared, opened, closed |
| No-write | Decision trace only | Explicit rejection; never persisted |

Every persistent record also requires source video, local frame, temporal
validity, confidence or uncertainty, provenance, and evidence references.

## Identity and Time

- Explicit source instance IDs take priority.
- Heuristic association is one-to-one within an observation time.
- Revisited detections may reuse a prior ID only when temporal and geometry
  compatibility permits it.
- A move closes the previous validity interval and creates a new state or event.
- Retrieval selects the latest causally valid, non-conflicting state.
- Missing frames, conflicting latest states, or incomplete indexes cause
  abstention where required.

## Coordinate Frames

Records are owned by local place or submap frames. Cross-frame operations require
an explicit transform. Missing frames never silently match a requested frame.

Supported frame meanings must remain distinct:

- wearer-relative egocentric;
- place/submap allocentric;
- object-centric;
- globally optimized building frame.

Loop closure should update a submap transform rather than rewrite every object.

## Writer Objective

Candidate `i` is useful when it reduces future loss or uncertainty enough to
justify its actual storage cost.

```text
utility(i)
    = future QA loss reduction
    + geometry coverage
    + uncertainty reduction
    + relocalization information
    + event surprise
    - redundancy
    - serialized byte cost
```

Inference uses a hard writer with a real byte budget. Token count or a latent
regularizer is not accepted as a substitute for serialized size.

Production typed artifacts are canonical UTF-8 JSONL. Budget accounting groups
records by
`(source_video_id, floor(first_seen_time / window_seconds))`; the current
contract fixes `window_seconds=30.0` and defaults to 4,096 bytes per window.
The repository rejects noncanonical rows, duplicate IDs, persisted `no_write`
records, empty output, manifest mismatches, canonical rows over 1 MiB, and any
over-budget window. Validation streams the artifact. Learned candidate utility
inside the external decoder remains an unverified model property until the
company-side probe runs.

Production validation also joins records to sanitized sources and selected
sensor frames. Source video and all record times must be in source bounds.
`observed`, `multi_view_fused`, and `human_confirmed` require evidence; each
typed `evidence_ref` is a bare same-video selected `frame_ref` whose timestamp
falls within first/last seen. Its min/max timestamps must equal first/last seen,
and `observation_count` must equal unique evidence count. This also prevents
backdating because 30-second accounting keys on first seen. QA prediction audits
use the separate `<video_id>/<frame_ref>` namespace.

## Query Path

```text
question
    -> language-to-operator planner
    -> causal entity and geometry retrieval
    -> deterministic operation
    -> proof object
    -> answer-choice verification
    -> language explanation or abstention
```

A proof contains:

- subject and object entity roles;
- operation and value;
- coordinate frame;
- uncertainty and unit;
- provenance;
- evidence references;
- a stable hash of every behavior-affecting query option.

Count and last-seen operations require a complete-index certificate. The model
prompt does not receive raw geometry dictionaries that could bypass the proof.
The byte-budgeted student artifact cannot provide that certificate because
selection may omit an object, newer state, or change event. Production student
count and last-seen therefore abstain. Pair proofs may use all persisted causal
objects in question-video scope only when the question names explicit entity
IDs. Label-only selectors also require a completeness certificate proving
uniqueness. Retrieved spatial evidence must first exact-match the canonical
typed retrieval projection on ID, video, snippet, frame refs, times, and
geometry. Pair proofs use record-local frames and reject cross-video pairs.

## Causal and Trust Boundaries

- Evidence end time cannot exceed question time.
- External teacher/inference adapters are digest-bound trusted executables. The
  DAG denylist-scrubs known sensitive variables but is not an OS sandbox or
  `env -i`; ambient `PATH`, `HOME`, `PYTHONPATH`, and Slurm state remain.
- Evidence video must belong to the question scope.
- Evidence IDs must be known, unique, and present exactly once per question.
- Student QA recomputes evidence, checkpoint, typed-memory, inference-manifest,
  config, sensor, and data hashes before invoking the model. It also parses the
  supplied memory manifest and requires its SHA-256 plus the referenced
  episodic, semantic, and visual file SHA-256 values to match student evidence
  lineage. Typed memory keeps its separate inference-bound digest.
- Resume artifacts are also bound to question, source, evidence, backend,
  model, prompt, schema, evidence lane, required-frame policy, the memory
  manifest, and the evidence-lineage file. The evidence-lineage digest is the
  transitive resume binding for individual non-spatial store digests, which QA
  still recomputes before a student QA start or resume.
- QA prompt/resume contracts are versioned as
  `qa-prompt-prediction-schema-v4` and `qa-resume-manifest-v5`. Prompt frame
  rows carry `video_id`, `frame_ref`, and `timestamp`; persisted prediction refs
  use `<video_id>/<frame_ref>` so equal frame names in different videos cannot
  collide.
- `qa/completed.json` seals predictions against the resume manifest. The report
  stage then writes and rechecks `summary/finalization_inputs.sha256` so QA,
  evidence lineage, memory manifest, episodic/semantic/visual stores, typed
  memory, snapshot config, sensor, and split inputs cannot change between
  evaluation and final identity/report generation. Finalization also rehashes
  every named path immediately before publication and writes the remote
  manifest last as the completion marker.
- Production QA fails when an evidence pack resolves no real input frame.
- `observed`, `multi_view_fused`, `model_inferred`, and `relation_inferred`
  provenance remain distinguishable.

## Decisions

- [ADR-0001: Explicit typed memory](decisions/adr-0001-explicit-typed-memory.md)
- [ADR-0002: G-CUT3R as teacher](decisions/adr-0002-gcut3r-as-teacher.md)
- [ADR-0003: Value per actual byte](decisions/adr-0003-value-per-byte-writer.md)
- [ADR-0004: Deterministic geometry proof](decisions/adr-0004-deterministic-geometry-proof.md)

## Current Boundary

The repository-owned model still ends at `spatial_student.pt`; the staged
production lane continues through the external typed inference contract and
then returns to repository-owned validation, retrieval, QA, metrics,
profile-bound run identity, and report generation. That lane is implemented but
has not run on company data. A `probe` run is explicitly `contract_probe` /
`PROBE`; only the separately approved `full` profile is `student` / `E1`.
Official E1/E2/E3 remains blocked because matched E2/E3 identities are not
generated. See
[Current Status](status.md),
[EXP-0002](experiments/exp-0002-typed-memory-bridge.md), and the operational
source of truth is repository `HANDOFF.md`, imported under the
[Operations](operations/README.md) parent in Confluence and not duplicated here.

[Back to project home](README.md)
