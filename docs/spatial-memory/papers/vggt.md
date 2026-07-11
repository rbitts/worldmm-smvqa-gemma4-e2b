# VGGT: Visual Geometry Grounded Transformer

| Field | Value |
|---|---|
| Page ID | SM-PAPER-VGGT |
| Status | Reviewed from primary sources; not reproduced in this project |
| Publication | CVPR 2025 Best Paper, pp. 5294–5306 |
| Primary source | [CVPR Open Access paper](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html) |
| Official code | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) |
| Last checked | 2026-07-11 |
| Project links | [Paper index](README.md), [Architecture](../architecture.md), [Roadmap](../roadmap.md) |
| Project claims | [Traceability](../traceability.md): C-001 |

## 30-second summary

VGGT is a roughly 1.2-billion-parameter feed-forward transformer that jointly
predicts camera parameters, depth maps, point maps, and point tracks from one to
hundreds of images. It alternates frame-wise and global attention and produces
directly usable multi-view geometry without mandatory test-time optimization.

## Problem addressed

Traditional reconstruction requires multi-stage visual geometry and iterative
optimization. Pairwise learned models still need global alignment for many images.
VGGT asks whether one large multi-task network can infer the scene's core 3D
attributes jointly in a single forward pass.

## Relevant method

DINOv2 patch tokens, camera tokens, and register tokens enter 24 alternating
frame-wise and global attention layers. Separate camera and DPT heads predict
intrinsics, extrinsics, depth, point maps, and uncertainty. A tracking head uses
dense features for cross-view point correspondence. The first camera defines the
world frame.

The model is trained jointly on camera, depth, point-map, and tracking losses over
2–24 sampled views. The paper reports training for 160,000 iterations on 64 A100
GPUs over nine days.

## Paper-reported evidence

These are paper results, not results from this repository.

| Dataset or condition | Metric | Reported result | Source location |
|---|---|---:|---|
| RealEstate10K, unseen, ten views, one H100 | AUC at 30 / runtime | 85.3 / about 0.2 s | Table 1, paper p. 7 |
| CO3Dv2, ten views, one H100 | AUC at 30 / runtime | 88.2 / about 0.2 s | Table 1, paper p. 7 |
| ETH3D point-map estimation, depth plus camera heads | Accuracy / completeness / overall / runtime | 0.873 / 0.482 / 0.677 / about 0.2 s | Table 3, paper p. 7 |
| Ten 336 by 518 frames, H100, Flash Attention 3 | Backbone runtime / peak memory | 0.14 s / 3.63 GB | Table 9, paper p. 10 |
| One hundred 336 by 518 frames, same setting | Backbone runtime / peak memory | 3.12 s / 21.15 GB | Table 9, paper p. 10 |

Table 9 also reports 200 frames at 8.75 seconds and 40.63 GB. This is direct
evidence that the non-recurrent all-view design becomes expensive as the view set
grows, despite strong short-sequence throughput.

## What this supports here

**Paper claim:** joint feed-forward prediction can provide high-quality cameras,
depth, point maps, and tracks from sparse multi-view inputs without mandatory
global alignment.

**Project inference:** VGGT is a useful non-recurrent teacher comparator for
CUT3R-derived geometry. It can test whether typed-record quality depends on a
recurrent provider or mainly on the provider's multi-view geometry accuracy.

Its output heads also define useful teacher targets: pose, metric-consistent depth,
point maps, tracks, and uncertainty.

## What it does not prove

- It is not a streaming fixed-state memory and processes the selected view set
  jointly.
- It does not establish bounded memory over lifelong input; Table 9 shows growing
  runtime and GPU memory with more frames.
- It does not provide object identities, temporal validity, typed spatial records,
  or QA proofs.
- It does not evaluate 1 Hz AI-glasses sensing, SuperMemory-VQA, or on-device use.
- Strong reconstruction metrics do not by themselves establish future-QA utility
  or value per stored byte.

## Project reproduction status

VGGT is not installed, downloaded, or executed in this repository. No model
weights, predictions, or benchmark artifacts exist locally. It remains a planned
teacher/provider comparison, not an implemented lane.

## References

- [CVPR 2025 Open Access record](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html)
- [Official project page](https://vggt.robots.ox.ac.uk/)
- [Official code](https://github.com/facebookresearch/vggt)
- [Official arXiv record](https://arxiv.org/abs/2503.11651)

[Back to paper index](README.md)
