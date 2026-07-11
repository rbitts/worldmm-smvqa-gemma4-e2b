# Architecture

| Field | Value |
|---|---|
| Page ID | SM-ARCHITECTURE |
| Status | Living specification |
| Last updated | 2026-07-11 |
| Current implementation | Source-compact baseline plus typed-training preparation |
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

The target student must decode its own checkpoint outputs into typed records.
External precomputed evidence may be used for baselines but cannot be the final
learned path.

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

Current typed records already support deterministic actual-byte admission. The
learned value predictor and checkpoint-to-record inference path remain open.

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

## Causal and Trust Boundaries

- Evidence end time cannot exceed question time.
- Evidence video must belong to the question scope.
- Evidence IDs must be known, unique, and present exactly once per question.
- Resume artifacts are bound to question, source, evidence, backend, model,
  prompt, and schema hashes.
- `observed`, `multi_view_fused`, `model_inferred`, and `relation_inferred`
  provenance remain distinguishable.

## Decisions

- [ADR-0001: Explicit typed memory](decisions/adr-0001-explicit-typed-memory.md)
- [ADR-0002: G-CUT3R as teacher](decisions/adr-0002-gcut3r-as-teacher.md)
- [ADR-0003: Value per actual byte](decisions/adr-0003-value-per-byte-writer.md)
- [ADR-0004: Deterministic geometry proof](decisions/adr-0004-deterministic-geometry-proof.md)

## Current Boundary

The learned lane currently ends at `spatial_student.pt`. It does not yet produce
typed persistent evidence consumed by QA. See [Current Status](status.md) and
[EXP-0004](experiments/exp-0004-gcut3r-provider.md).

[Back to project home](README.md)
