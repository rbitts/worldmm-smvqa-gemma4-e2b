# CUT3R: Continuous 3D Perception Model with Persistent State

| Field | Value |
|---|---|
| Page ID | SM-PAPER-CUT3R |
| Status | Reviewed from primary sources; not reproduced in this project |
| Publication | CVPR 2025 Oral, pp. 10510–10522 |
| Primary source | [CVPR Open Access paper](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html) |
| Official code | [CUT3R/CUT3R](https://github.com/CUT3R/CUT3R) |
| Last checked | 2026-07-11 |
| Project links | [Paper index](README.md), [Architecture](../architecture.md), [ADR-0002](../decisions/adr-0002-gcut3r-as-teacher.md) |
| Project claims | [Traceability](../traceability.md): C-001, C-006 |

## 30-second summary

CUT3R is an online recurrent 3D perception model. Each incoming RGB image reads
from and updates a persistent latent state, then predicts camera pose and dense
point maps in both camera and shared world coordinates. A virtual ray-map query
can read the state to predict an unobserved view. The model handles ordered video,
unordered photographs, and static or dynamic scenes without supplied camera
intrinsics or poses.

## Problem addressed

Pairwise reconstruction models require global alignment to combine many views,
while classical SfM and SLAM can fail under sparse overlap, dynamic content, or
degenerate motion. CUT3R targets continuous online reconstruction with a learned
scene prior and a fixed-shape recurrent state.

## Relevant method

An image encoder produces tokens for the current frame. Two interconnected
transformer decoders implement state update and state readout through cross
attention. Output heads predict camera-frame point maps, world-frame point maps,
confidence, and pose. A separate ray-map encoder queries the state without
updating it for virtual-view prediction.

The published implementation uses a ViT-L image encoder, ViT-B decoders, 16 by 16
patches, and 768 state tokens of dimension 768. Training progresses from four-view
sequences to sequences as long as 64 views across 32 datasets.

## Paper-reported evidence

These are paper results, not results from this repository.

| Dataset or condition | Metric | Reported result | Source location |
|---|---|---:|---|
| KITTI video depth, 512 by 144, A100 | Online throughput | 16.58 FPS | Table 2, paper p. 5 |
| KITTI video depth, per-sequence scale | Abs Rel / delta below 1.25 | 0.118 / 88.1 | Table 2, paper p. 5 |
| 7-Scenes, sparse 3–5 frames | Mean accuracy / completeness / normal consistency | 0.126 / 0.154 / 0.727 | Table 4, paper p. 7 |
| NRGBD, sparse 2–4 frames | Mean accuracy / completeness / normal consistency | 0.099 / 0.076 / 0.837 | Table 4, paper p. 7 |
| 7-Scenes, online then frozen-state revisit | Mean accuracy, before versus after revisit | 0.126 versus 0.113 | Table 5, paper p. 8 |

The revisit experiment also reports 7-Scenes mean completeness improving from
0.154 to 0.107. This supports the paper's narrower claim that the recurrent state
can refine predictions after additional observations.

## What this supports here

**Paper claim:** recurrent image-state interaction can provide online camera and
dense geometry estimates in a common frame, including sparse-view conditions.

**Project inference:** CUT3R is a suitable transient geometry teacher or front-end
for converting sparse observations into typed object, structure, landmark, and
event candidates. Its state should be consumed during construction, not serialized
at every timestamp as long-term memory.

The common-frame point maps, pose estimates, and confidence outputs are useful
teacher signals for the planned typed decoder and association model.

## What it does not prove

- The latent state is not an explicit database of entities, relationships, or
  temporal validity intervals.
- Virtual-view output is model inference, not direct observation, and must carry
  different provenance.
- Fixed state shape does not prove stable lifelong retention or bounded semantic
  forgetting.
- Published GPU throughput does not establish AI-glasses deployment feasibility.
- The paper does not evaluate SuperMemory-VQA, 1 Hz year-long memory, actual-byte
  storage, or geometry-grounded QA.

## Project reproduction status

The project defines a provider/cache boundary for a CUT3R-derived external teacher
but does not vendor, download, or execute CUT3R locally. There is no project
checkpoint, official-dataset reconstruction result, or CUT3R-to-typed-record
reproduction yet.

## References

- [CVPR 2025 Open Access record](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html)
- [Official project page](https://cut3r.github.io/)
- [Official code](https://github.com/CUT3R/CUT3R)
- [Official arXiv record](https://arxiv.org/abs/2501.12387)

[Back to paper index](README.md)
