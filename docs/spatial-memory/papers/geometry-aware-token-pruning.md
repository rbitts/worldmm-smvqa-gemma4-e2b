# Seeing Once is Enough? Online Geometry-Aware Token Pruning for 3D Question Answering

| Field | Value |
|---|---|
| Page ID | SM-PAPER-GEOMETRY-AWARE-TOKEN-PRUNING |
| Status | Reviewed; recent workshop preprint |
| Authors | Ruei-Chi Lai, Bolivar Solarte, Chin-Hsuan Wu, Yi-Hsuan Tsai, Min Sun |
| Publication | ICLR 2026 Workshop on Efficient Spatial Reasoning |
| Version checked | arXiv:2607.04079v1, 2026-07-05 |
| Primary source | [arXiv](https://arxiv.org/abs/2607.04079) · [OpenReview](https://openreview.net/forum?id=jnDbE6cV2D) |
| Official code | Not published or not linked by the checked primary sources |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-011 |

## 30-Second Summary

This work projects posed RGB-D observations into a shared voxel space and
removes image tokens corresponding to regions already covered by previous
views. It is direct evidence that geometry overlap can reduce online 3D-QA input
tokens without requiring an offline pass over the full scene. It prunes VLM
input, not persistent spatial memory records.

## Problem Addressed

Multi-view 3D question answering sends many overlapping image tokens to a
multimodal language model. Existing frame-selection and token-merging methods
often preprocess the complete sequence offline, which does not fit an online
stream.

## Relevant Method

- Use depth and camera pose to project each frame into a shared voxel space.
- Detect spatial overlap between the current view and previously observed
  regions.
- Prune redundant image tokens before they enter the language model.
- Operate online and training-free with an existing multimodal model.

## Paper-Reported Evidence

The authors report up to 50% lower token usage. Applied to Qwen2.5-VL-7B and
Qwen3-VL-8B, the method reports improved results on ScanQA, SQA3D, and
OpenEQA-HM3D. The arXiv abstract does not establish results for monocular input
without depth and pose.

These are paper results, not results reproduced by this repository.

## What This Supports Here

- Compute geometry novelty before retaining visual evidence or generating
  redundant downstream candidates.
- Maintain a causal coverage map that can reject already-observed regions.
- Test geometry-overlap selection as a query-agnostic complement to QA utility.
- Evaluate both token cost and geometry-sensitive QA under online processing.

## What It Does Not Prove

- That overlap-pruned image tokens are sufficient for future unknown questions.
- That the method works with approximately 1 Hz monocular RGB without reliable
  depth and pose.
- That VLM token reduction yields a compact explicit persistent map.
- That static overlap alone preserves object movement and interaction events.
- That the workshop result transfers to SuperMemory-VQA.

## Project Reproduction Status

Not reproduced. The source identity has been verified, but no RGB-D voxel
pruner or reported benchmark run exists in this repository. The current writer
uses explicit records and actual-byte budgets, so a comparison requires a
separate transient-input ablation rather than replacing the persistent writer.

## References

- [Paper index](README.md)
- [arXiv:2607.04079](https://arxiv.org/abs/2607.04079)
- [OpenReview](https://openreview.net/forum?id=jnDbE6cV2D)
