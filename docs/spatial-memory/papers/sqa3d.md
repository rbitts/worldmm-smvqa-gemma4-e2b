# SQA3D: Situated Question Answering in 3D Scenes

| Field | Value |
|---|---|
| Page ID | SM-PAPER-SQA3D |
| Status | Reviewed; code and dataset available |
| Publication | ICLR 2023; arXiv:2210.07474 v5 |
| Primary source | [OpenReview](https://openreview.net/forum?id=IDJx97BC38) · [arXiv](https://arxiv.org/abs/2210.07474) |
| Official code | [SilongYong/SQA3D](https://github.com/SilongYong/SQA3D) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-003 |

## 30-second summary

SQA3D asks questions from a described agent position and orientation inside a complete 3D scene. Situation input and auxiliary pose supervision improve QA, supporting explicit wearer coordinate frames and pose-aware learning. It evaluates static, fully scanned indoor scenes, not causal sparse-stream memory.

## Problem addressed

Scene-level QA can ignore where an embodied agent stands and faces. SQA3D requires a model to understand a textual situation, ground the implied position and orientation in a 3D scan, and answer spatial, navigation, commonsense, and multi-hop questions from that situated perspective.

## Relevant method

- Collect descriptions and questions around agent situations in 650 ScanNet scenes.
- Encode situation and question separately.
- Use VoteNet object tokens, then cross-attend them to the situation and question.
- Optionally supervise position and rotation heads alongside answer classification.
- Evaluate exact-match accuracy over 706 answer candidates.

## Paper-reported evidence

| Dataset | Condition | Metric | Reported result | Location |
|---|---|---|---|---|
| SQA3D | Dataset | Scale | 650 scenes; 6.8k situations; 20,369 descriptions; 33,403 questions | Section 2.2 and Table 2 |
| SQA3D | Blind situation + question | Accuracy | 43.65% | Table 3 |
| SQA3D | ScanQA without situation | Accuracy | 45.27% | Table 3 |
| SQA3D | ScanQA with situation | Accuracy | 46.58% | Table 3 |
| SQA3D | ScanQA with situation and pose auxiliary losses | Accuracy | 47.20% | Table 3 |
| SQA3D | Amateur human | Accuracy | 90.06% | Table 3 |

## What this supports here

- Coordinate-frame identity must be explicit for egocentric direction questions.
- Pose supervision is a useful auxiliary objective for geometry-grounded QA.
- Situation grounding should be evaluated separately from answer classification.
- The project infers that explicit frame metadata and wearer pose should survive compression.

## What it does not prove

- Sparse 1 Hz reconstruction from RGB or pose drift handling.
- Long-term memory, temporal identity, change events, or causal write policies.
- Typed-record compression, actual-byte budgets, or on-device feasibility.
- Results on dynamic scenes or SuperMemory-VQA.

## Project reproduction status

Not reproduced. SQA3D is retained as an external situated-QA reference and a possible auxiliary evaluation, not as evidence that the current memory writer is complete.

## References

- Xiaojian Ma et al. [SQA3D: Situated Question Answering in 3D Scenes](https://openreview.net/forum?id=IDJx97BC38). ICLR 2023.
- [Official project page](https://sqa3d.github.io/).
- [Official repository](https://github.com/SilongYong/SQA3D).
- [Official dataset release](https://zenodo.org/records/7792397).
- [Back to paper index](README.md).
