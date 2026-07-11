# LONG3R: Long Sequence Streaming 3D Reconstruction

| Field | Value |
|---|---|
| Page ID | SM-PAPER-LONG3R |
| Status | reviewed; code stub only |
| Publication | ICCV 2025; arXiv:2507.18255 v1 |
| Primary source | [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_LONG3R_Long_Sequence_Streaming_3D_Reconstruction_ICCV_2025_paper.html) · [arXiv](https://arxiv.org/abs/2507.18255) · [Project page](https://zgchen33.github.io/LONG3R/) |
| Official code | [zgchen33/LONG3R](https://github.com/zgchen33/LONG3R/) — repository says code is coming soon |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-002 |

## 30-second summary

LONG3R is a recurrent streaming-reconstruction model for long image sequences. It gates memory before decoding and maintains a 3D spatio-temporal memory that prunes redundant spatial information while changing resolution across the scene. It is evidence for testing adaptive geometry memory in the transient reconstruction front end, not evidence that latent memory is a sufficient long-term QA database.

## Problem addressed

Existing multi-view reconstruction systems either require offline global optimization or lose quality when a recurrent state is used beyond short sequences. LONG3R targets online reconstruction over longer streams without making inference cost grow with all previous frames.

## Relevant method

- A recurrent model updates memory for every new observation, using recent frames as short-term temporal memory.
- Attention-based memory gating selects relevant entries before the refined decoder uses them.
- A dual-source refined decoder combines selected memory with adjacent-frame features.
- Long-term pointmap patches are grouped into adaptive voxels; one token with the highest cumulative attention is retained per voxel.
- Two-stage curriculum training separates shorter-context learning from long-sequence capability.

## Paper-reported evidence

**Reported claim.** The default experiment uses 10-frame short-term memory and 3,000 long-term tokens. On 7-Scenes the paper reports that memory gating reduces tokens by 27% and increases throughput from 18.0 to 21.4 FPS. On 200-frame Replica sequences it reports Acc/Comp of `11.93/2.73`, compared with `16.29/4.02` for Spann3R and `28.30/6.61` for CUT3R. The method predicts frame `t` with features from `t+1`, so this is one-frame-latency streaming rather than zero-lookahead processing.

**Project inference.** Memory gating and adaptive spatial resolution are reasonable comparison points for fixed-voxel and fixed-slot transient geometry memories.

**Project result.** None. This repository has not reproduced LONG3R.

## What this supports here

- Comparing fixed voxels with adaptive-resolution spatial slots.
- Using geometry relevance and redundancy in a working-memory write gate.
- Evaluating whether repeated visits stop adding equivalent spatial state.

These are project hypotheses derived from the method, not LONG3R results on SuperMemory-VQA.

## What it does not prove

- That latent reconstruction attention identifies future-QA-important facts.
- That the memory is an explicit entity, relation, event, or provenance database.
- Actual serialized-byte savings for lifelong memory.
- Accuracy under 1 Hz monocular AI-glass sensing, multi-day revisits, or SuperMemory-VQA.
- On-device feasibility. The official code repository contained no released implementation when checked.
- Zero-lookahead causality; its refined prediction uses the next frame's features.

## Project reproduction status

Not reproduced. Use as a design reference and future baseline only. Do not report LONG3R-derived performance as a project result until code, checkpoint, dataset digest, and evaluation output are recorded in an experiment page.

## References

- Zhuoguang Chen, Minghui Qin, Tianyuan Yuan, Zhe Liu, and Hang Zhao. [LONG3R: Long Sequence Streaming 3D Reconstruction](https://arxiv.org/abs/2507.18255). ICCV 2025.
- [ICCV 2025 open-access publication](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_LONG3R_Long_Sequence_Streaming_3D_Reconstruction_ICCV_2025_paper.html).
- [Official project page](https://zgchen33.github.io/LONG3R/).
- [Official repository](https://github.com/zgchen33/LONG3R/).

[Back to paper index](README.md)
