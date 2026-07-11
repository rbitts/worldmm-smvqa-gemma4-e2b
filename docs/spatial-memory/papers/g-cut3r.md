# G-CUT3R: Guided 3D Reconstruction with Camera and Depth Prior Integration

| Field | Value |
|---|---|
| Page ID | SM-PAPER-G-CUT3R |
| Status | Reviewed from primary sources; external teacher implementation blocked |
| Publication | ICLR 2026 |
| Primary source | [Official OpenReview record](https://openreview.net/forum?id=J7DiMqmIFl) |
| Official code | OpenReview supplementary material is described by the paper as including source code; no standalone official repository was verified |
| Last checked | 2026-07-11 |
| Project links | [Paper index](README.md), [Architecture](../architecture.md), [ADR-0002](../decisions/adr-0002-gcut3r-as-teacher.md), [Status](../status.md) |
| Project claims | [Traceability](../traceability.md): C-001 |

## 30-second summary

G-CUT3R extends CUT3R so optional camera intrinsics, camera poses, and depth maps
can guide recurrent reconstruction. Each modality has a dedicated encoder. Its
features enter CUT3R decoder blocks through zero-initialized convolutions, allowing
one model to accept any available combination of priors while retaining the
sequential state mechanism.

## Problem addressed

Feed-forward reconstruction models commonly ignore geometric information already
available from calibration, VIO, SLAM, RGB-D, or LiDAR. Pairwise guided methods can
also require expensive global alignment. G-CUT3R targets lightweight prior fusion
inside an online recurrent reconstruction model.

## Relevant method

Camera intrinsics and poses are encoded as ray images; depth has its own spatial
encoding. Four-block modality-specific transformer encoders produce features with
the same 768-dimensional interface as CUT3R. Zero-initialized convolution layers
inject these features into decoder stages. Training randomly varies the available
modalities so one checkpoint supports arbitrary combinations of intrinsics, pose,
and depth guidance.

The paper initializes from CUT3R, trains on four-image sequences using twelve
datasets, and reports training for ten days on four A100 GPUs. Evaluation includes
low-overlap 3D reconstruction, video depth, pose estimation, and fusion ablations.

## Paper-reported evidence

These are paper results, not results from this repository.

| Dataset or condition | Metric | Reported result | Source location |
|---|---|---:|---|
| 7-Scenes, 3–5 low-overlap views, resolution 512 | Mean accuracy, unguided / pose / all priors | 0.098 / 0.061 / 0.048 | Table 1, paper p. 7 |
| NRGBD, 3–5 low-overlap views, resolution 224 | Mean normal consistency, unguided / depth / all priors | 0.708 / 0.746 / 0.767 | Table 1, paper p. 7 |
| Bonn, ten-frame video depth, resolution 224 | Abs Rel, unguided / pose / intrinsics plus pose | 0.126 / 0.105 / 0.104 | Table 2, paper p. 8 |
| Waymo, four-view reconstruction | Fourth-view L2, no ZeroConv versus ZeroConv, all priors | 1.959 versus 1.155 | Table 3, paper p. 9 |
| ScanNet++, four-view reconstruction | Fourth-view L2, no ZeroConv versus ZeroConv, all priors | 0.078 versus 0.064 | Table 3, paper p. 9 |

The paper notes that its unguided fine-tuned variant and original CUT3R are not a
fair matched pair because they saw different training data. Guidance effects should
therefore be read against the same-data G-CUT3R unguided variant.

## What this supports here

**Paper claim:** optional pose, intrinsic, and depth priors can improve a
CUT3R-derived reconstruction model, and zero-initialized feature injection provides
a workable fusion mechanism.

**Project inference:** high-rate IMU/VIO pose and optional calibrated depth are
valuable teacher inputs when RGB is sampled near 1 Hz and view overlap is low.
G-CUT3R is therefore the preferred external teacher candidate, while its dense
state remains transient.

The modality-drop training scheme also motivates explicit provider metadata and
ablations for pose-only, depth-only, combined, and RGB-only inputs.

## What it does not prove

- It does not evaluate AI-glasses power, thermal limits, or on-device latency.
- It does not evaluate lifelong state retention, typed persistent memory, actual
  serialized bytes, or geometry-grounded QA.
- Low-overlap 3–5-view experiments are not equivalent to months of 1 Hz video.
- Better reconstruction does not prove better SuperMemory-VQA accuracy.
- Inferred geometry still requires uncertainty and provenance; it cannot be treated
  as directly observed fact.
- No standalone official code repository was independently verified on the check
  date; reproducibility depends on the official OpenReview supplementary material.

## Project reproduction status

The repository implements an external provider protocol, causal cache hashes,
teacher materialization, and a DDP candidate-head training scaffold. It does not
contain the G-CUT3R extractor or checkpoint and has not run G-CUT3R inference.
Checkpoint-to-typed-memory inference remains a P0 blocker.

## References

- [Official OpenReview record](https://openreview.net/forum?id=J7DiMqmIFl)
- [Official arXiv record, version 2](https://arxiv.org/abs/2508.11379)

[Back to paper index](README.md)
