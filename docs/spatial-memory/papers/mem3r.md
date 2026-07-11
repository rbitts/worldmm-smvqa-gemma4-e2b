# Mem3R: Streaming 3D Reconstruction with Hybrid Memory via Test-Time Training

| Field | Value |
|---|---|
| Page ID | SM-PAPER-MEM3R |
| Status | reviewed; code and checkpoint pending |
| Publication | arXiv:2604.07279 v1, 2026 preprint |
| Primary source | [arXiv](https://arxiv.org/abs/2604.07279) · [Project page](https://lck666666.github.io/Mem3R/) |
| Official code | [lck666666/Mem3R](https://github.com/lck666666/Mem3R) — repository says code and checkpoint are coming soon |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-006 |

## 30-second summary

Mem3R separates streaming camera tracking from geometric mapping. A lightweight implicit fast-weight MLP handles pose-related state while a fixed-size explicit token state carries geometry. This supports separating transient pose memory from persistent map records in this project, but Mem3R's “explicit” tokens are still model state rather than a queryable QA database.

## Problem addressed

A single compressed recurrent state must retain global geometry and track the current camera. Those competing roles cause temporal forgetting and drift on long sequences. Mem3R decouples them while keeping recurrent inference memory bounded.

## Relevant method

- Implicit fast-weight MLP memory performs camera tracking and is updated through test-time training.
- A separate fixed-size token state maintains geometric context.
- A channel-wise module fuses the candidate geometry state with the previous state.
- Existing CUT3R update policies such as TTT3R and TTSA3R can be added without replacing the hybrid split.

## Paper-reported evidence

**Reported claim.** The paper reports reducing model size from 793M to 644M parameters. Its official project page lists 26 FPS for both CUT3R and Mem3R, with GPU memory decreasing from 7,930 MiB to 7,340 MiB. The paper also reports that adding TTT3R reduces Absolute Trajectory Error by up to 39% over the corresponding base implementation on 500–1,000-frame sequences.

**Project inference.** Pose tracking can remain implicit and short-lived while geometry needed for QA should be compiled into explicit persistent records.

**Project result.** None. This repository has not reproduced Mem3R.

## What this supports here

- Maintaining separate fast pose, working geometry, and persistent QA memory lifecycles.
- Avoiding a single global recurrent state that must serve tracking and long-term retrieval.
- Testing hybrid implicit-explicit memory after the provider and typed decoder are independently validated.

## What it does not prove

- That its token state exposes object IDs, coordinate frames, uncertainty, provenance, or deterministic proof operations.
- Actual-byte compression of a lifelong spatial database.
- Future-QA-aware selection or object-centric consolidation.
- Performance on 1 Hz AI-glass streams, SuperMemory-VQA, or on-device hardware.
- Reproducibility from the current repository; implementation and checkpoint were not released when checked.

## Project reproduction status

Not reproduced, and the official repository is currently a placeholder. Hybrid pose/map separation remains a design reference. It should not be integrated before the geometry provider and typed record decoder have independent project baselines.

## References

- Changkun Liu, Jiezhi Yang, Zeman Li, Yuan Deng, Jiancong Guo, and Luca Ballan. [Mem3R: Streaming 3D Reconstruction with Hybrid Memory via Test-Time Training](https://arxiv.org/abs/2604.07279). 2026.
- [Official project page](https://lck666666.github.io/Mem3R/).
- [Official repository](https://github.com/lck666666/Mem3R).

[Back to paper index](README.md)
