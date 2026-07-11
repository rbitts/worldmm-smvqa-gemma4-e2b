# TTSA3R: Training-Free Temporal-Spatial Adaptive Persistent State for Streaming 3D Reconstruction

| Field | Value |
|---|---|
| Page ID | SM-PAPER-TTSA3R |
| Status | reviewed; code available |
| Publication | arXiv:2601.22615 v3, 2026 preprint |
| Primary source | [arXiv](https://arxiv.org/abs/2601.22615) |
| Official code | [anonus2357/ttsa3r](https://github.com/anonus2357/ttsa3r) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-006 |

## 30-second summary

TTSA3R is a training-free CUT3R state-update policy combining temporal state evolution with spatial observation quality. It supports a more selective transient state update than one global confidence gate. It remains an implicit recurrent-state method, not a durable explicit-memory compressor.

## Problem addressed

Persistent recurrent state forgets history when all tokens are updated similarly. A temporal-only or spatial-only signal can miss where an observation is stale, dynamic, or poorly aligned. TTSA3R combines both signals to update fewer inappropriate regions.

## Relevant method

- A Temporal Adaptive Update Module measures state evolution and regulates token update magnitude.
- A Spatial Contextual Update Module uses observation–state alignment and scene dynamics to locate regions needing updates.
- The two masks are fused so temporal and spatial evidence jointly controls the persistent-state transition.
- The intervention is training-free and uses a pretrained recurrent reconstruction model.

## Paper-reported evidence

**Reported claim.** On NRGBD when sequence length increases from 50 to 250 frames, the v3 paper reports a `1.33×` reconstruction-error increase for TTSA3R versus more than `4×` for CUT3R. Its ablation reports Bonn Abs Rel and TUM Dynamics ATE of `0.078/0.046` for CUT3R and `0.064/0.026` with both temporal and spatial modules. The paper reports 18.0 FPS and 6 GB on one NVIDIA A6000 for TTSA3R.

**Project inference.** Temporal and spatial masks are useful baselines for deciding where a transient geometry state may update after sparse or dynamic observations.

**Project result.** None. This repository has not reproduced TTSA3R.

## What this supports here

- Separating temporal staleness from spatial observation quality in a working-state gate.
- Testing whether pose, ray, novelty, and dynamics signals reduce destructive writes.
- Comparing training-free state stabilization with a learned QA-aware persistent writer.

## What it does not prove

- Stable association under severe occlusion or very low visual overlap.
- Entity IDs, typed geometry, temporal events, metric proof, or provenance.
- Actual-byte reduction; fixed latent-state memory is not serialized lifelong storage.
- SuperMemory-VQA improvement, 1 Hz wide-baseline operation, multi-day retention, or AI-glass feasibility.

## Project reproduction status

Not reproduced. Use the released code as a transient-state update baseline. Do not treat the reported reconstruction stability as evidence for QA memory quality.

## References

- Zhijie Zheng, Xinhao Xiang, and Jiawei Zhang. [TTSA3R: Training-Free Temporal-Spatial Adaptive Persistent State for Streaming 3D Reconstruction](https://arxiv.org/abs/2601.22615). arXiv v3, 2026.
- [Official repository](https://github.com/anonus2357/ttsa3r).

[Back to paper index](README.md)
