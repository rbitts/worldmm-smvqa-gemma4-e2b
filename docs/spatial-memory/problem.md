# Problem and Research Questions

| Field | Value |
|---|---|
| Page ID | SM-PROBLEM |
| Status | Active |
| Last updated | 2026-07-11 |
| Scope | SuperMemory-VQA and eventual AI-glass deployment |

## Problem

An AI-glass system must answer later questions grounded in space and time while
observing only sparse RGB frames and retaining limited long-term memory.

The system must support questions such as:

- Where was an object last observed?
- How far apart were two objects?
- Was an object moved, added, or removed?
- What was left, right, in front of, or behind the wearer?
- Which objects were supported by or contained in another object?
- Can the wearer move between two places through known portals and free space?

Text-only answers are insufficient. Geometry answers require an auditable proof
containing entity identities, operation, coordinate frame, uncertainty,
provenance, temporal validity, and evidence references.

## Constraints

### RQ-001: Sparse sensing

RGB observations are approximately 1 Hz. Consecutive views can have low overlap,
wide baselines, blur, occlusion, or room transitions. Native-rate IMU/VIO may be
available and must remain separate from the RGB sampling budget.

### RQ-002: Lifelong storage

Memory growth must follow new places, objects, and meaningful changes rather
than frame count. Repeated visits to an unchanged room must converge toward a
stable current-state representation.

### RQ-003: Explicit geometry-grounded QA

Metric answers cannot be guessed by a language model. A deterministic executor
must compute distance, direction, visibility, topology, temporal state, and
other supported operators from explicit records.

### RQ-004: Unknown future questions

The writer does not know future questions. Optimizing only for a fixed QA
template distribution can delete information needed by out-of-distribution
questions.

### RQ-005: Device model cost

The final system needs a compact student path. Large geometry foundation models
may serve as offline teachers, but their reported server-GPU throughput does not
establish on-device feasibility.

### RQ-006: Causality and provenance

No question may use observations after its question time. Model-inferred
geometry must remain distinguishable from direct or multi-view observation.

## Core Hypothesis

Future geometry QA can be preserved more efficiently by directly generating
typed sufficient statistics than by generating and then compressing generic
dense features.

```text
image and sensor history
    -> transient dense reasoning
    -> sparse typed memory records
    -> deterministic geometry operations
    -> answer and proof
```

## Success Criteria

### Correctness

- Every answerable geometry response carries a matching deterministic proof.
- Entity identity, frame, validity, uncertainty, and provenance remain intact.
- Causal violation count is zero.
- Unknown or unsupported geometry causes abstention rather than fabrication.

### Compression

- Storage is measured using actual serialized bytes.
- Results report bytes per hour, new area, object, and change event.
- Repeated-visit growth is measured explicitly.
- QA and geometry quality are reported as a Pareto curve against bytes.

### Model

- Sparse RGB and pose guidance share a causal observation contract.
- Student output decodes into valid typed records.
- Learned association supports unseen instances or explicitly emits `NEW`.
- The deployed write gate is the gate supervised by QA and geometry utility.

### Evaluation

- Official split, data digest, model digest, checkpoint digest, configuration,
  seed, and frame manifest are recorded.
- Main, without-spatial, and retrieval-protocol ablations share those contracts.
- Paper-reported results and project-reproduced results are never mixed.

## Non-Goals

- Lifelong storage of dense point maps, Gaussian scenes, or recurrent-state
  snapshots.
- Photorealistic reconstruction as the primary metric.
- A spatial-only language model.
- Claiming on-device feasibility from server-GPU experiments.
- Adding a learned codec before explicit actual-byte baselines are measured.

[Back to project home](README.md)
