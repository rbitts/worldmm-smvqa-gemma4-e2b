# Ray-Aware Pointer Memory with Adaptive Updates for Streaming 3D Reconstruction

| Field | Value |
|---|---|
| Page ID | SM-PAPER-RAY-AWARE-POINTER |
| Status | reviewed; code unavailable |
| Publication | arXiv:2605.05749 v3, 2026 preprint |
| Primary source | [arXiv](https://arxiv.org/abs/2605.05749) |
| Official code | Not linked by the paper or arXiv record as of 2026-07-11 |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-007 |

## 30-second summary

This paper adds viewing direction and source time to spatial pointers, then uses spatial and angular disagreement to distinguish redundancy, novelty, and loop revisits. Its retain-or-replace policy avoids feature averaging and bounds redundant growth. It directly motivates ray-aware landmarks for 1 Hz association, but its random replacement is only a baseline for a value-aware writer.

## Problem addressed

Position-only or appearance-driven pointer fusion can merge different surfaces, duplicate the same view, and destabilize geometry when viewpoint changes. Long streams also require recognizing revisits without retaining every observation.

## Relevant method

- Store 3D position, unit observation ray, feature embedding, and source frame index per pointer.
- Jointly compare Euclidean position and ray angle.
- Classify close/similar-ray observations as redundancy, close/different-ray observations as revisit candidates, and distant observations as novel geometry.
- Retain either the old or new pointer instead of averaging redundant features.
- Trigger pose refinement for loop candidates selected with spatial, angular, temporal, and information criteria.

## Paper-reported evidence

**Reported claim.** Against Point3R, the paper reports 7-Scenes Acc/Comp changing from `0.085/0.087` to `0.035/0.025`, while NC changes from `0.739` to `0.685`. On NRGBD it reports Acc/Comp changing from `0.077/0.069` to `0.061/0.022`, while NC changes from `0.835` to `0.771`. Thus distance and completeness improve in those tables, but normal consistency decreases. The reported GPU-memory figures are reserved runtime memory, not serialized pointer bytes.

**Project inference.** Position, viewing ray, and timestamp should be tested together for wide-baseline association and revisit detection.

**Project result.** None. This repository has not reproduced the method, and no official code was linked when checked.

## What this supports here

- Adding ray or view-cone information to relocalization landmarks.
- Distinguishing same-place redundancy from a new surface revealed at a different angle.
- Using retain-or-replace as a simple baseline against learned value-per-byte replacement.
- Triggering loop validation rather than treating every nearby pointer as the same observation.

## What it does not prove

- That random retain-or-replace preserves QA-important evidence.
- Better normal consistency; the reported NC values above move in the unfavorable direction.
- Entity-level typed records, deterministic geometry proof, or provenance-complete QA.
- Serialized fixed-byte memory, 1 Hz AI-glass performance, multi-day retention, or SuperMemory-VQA improvement.
- Reproducibility from public code; an official implementation was not linked.

## Project reproduction status

Not reproduced and currently blocked on an official implementation or a separately validated clean-room baseline. Do not infer a repository URL. The safe first project experiment is the pointer tuple and deterministic association rule, not the paper's full pose-refinement system.

## References

- Feifei Li, Qi Song, Chi Zhang, and Rui Huang. [Ray-Aware Pointer Memory with Adaptive Updates for Streaming 3D Reconstruction](https://arxiv.org/abs/2605.05749). arXiv v3, 2026.

[Back to paper index](README.md)
