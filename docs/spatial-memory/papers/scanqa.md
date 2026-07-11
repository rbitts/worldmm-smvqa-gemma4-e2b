# ScanQA: 3D Question Answering for Spatial Scene Understanding

| Field | Value |
|---|---|
| Page ID | SM-PAPER-SCANQA |
| Status | Reviewed; code and dataset available |
| Publication | CVPR 2022, pp. 19129–19139; arXiv:2112.10482 v3 |
| Primary source | [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2022/html/Azuma_ScanQA_3D_Question_Answering_for_Spatial_Scene_Understanding_CVPR_2022_paper.html) |
| Official code | [ATR-DBI/ScanQA](https://github.com/ATR-DBI/ScanQA) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-003 |

## 30-second summary

ScanQA couples free-form 3D question answering with localization of referenced objects. Its ablations show that object localization and semantic classification both improve answers. This supports explicit entity geometry and grounding losses, but the input is an already complete RGB-D scan rather than a compressed causal stream.

## Problem addressed

Image QA does not directly model 3D alignment, direction, or object grounding. ScanQA asks a model to answer questions about an entire indoor 3D scan and identify the 3D bounding boxes of the objects referred to by the question.

## Relevant method

- Extract object proposals from colored point clouds with VoteNet and PointNet++.
- Encode the question with a bidirectional LSTM.
- Fuse object proposals and question features with transformer layers.
- Jointly train answer classification, object localization, object classification, and detector losses.
- Associate free-form answers with one or more referenced object IDs.

## Paper-reported evidence

| Dataset | Condition | Metric | Reported result | Location |
|---|---|---|---|---|
| ScanQA | Dataset | Scale | 41,363 questions; 58,191 answers; 800 scenes | Section 3 and Table 2 |
| Test with objects | ScanQA | EM@1 / EM@10 | 23.45 / 56.51 | Table 3 |
| Test without objects | ScanQA | EM@1 / EM@10 | 20.90 / 54.11 | Table 3 |
| Test with objects | Answer loss only | EM@1 / EM@10 | 12.16 / 42.77 | Table 4 |
| Test with objects | Answer + localization | EM@1 / EM@10 | 20.46 / 51.67 | Table 4 |
| Test with objects | Answer + classification + localization | EM@1 / EM@10 | 23.45 / 56.51 | Table 4 |

## What this supports here

- Answers should be grounded to persistent entity IDs and geometry.
- Entity localization and semantic classification deserve explicit training losses.
- Multiple referenced objects must be supported for relation questions.
- The project infers that small entity-level geometry records can preserve this utility more directly than generic per-point features.

## What it does not prove

- That dense point clouds should be retained as long-term memory.
- Temporal identity, last-seen queries, movement events, or causal validity.
- 1 Hz sparse RGB sensing, pose uncertainty, actual-byte budgets, or lifelong growth.
- On-device or SuperMemory-VQA performance.

## Project reproduction status

Not reproduced. ScanQA is a grounding-loss and external-evaluation reference. No ScanQA data or model artifacts were downloaded locally.

## References

- Daichi Azuma et al. [ScanQA: 3D Question Answering for Spatial Scene Understanding](https://openaccess.thecvf.com/content/CVPR2022/html/Azuma_ScanQA_3D_Question_Answering_for_Spatial_Scene_Understanding_CVPR_2022_paper.html). CVPR 2022.
- [Official arXiv record](https://arxiv.org/abs/2112.10482).
- [Official project, repository, and dataset instructions](https://github.com/ATR-DBI/ScanQA).
- [Back to paper index](README.md).
