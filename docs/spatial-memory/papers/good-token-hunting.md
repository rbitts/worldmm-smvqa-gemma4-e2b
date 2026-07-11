# Good Token Hunting: A Hitchhiker's Guide to Token Selection for Visual Geometry Transformers

| Field | Value |
|---|---|
| Page ID | SM-PAPER-GOOD-TOKEN-HUNTING |
| Status | Reviewed; recent preprint |
| Authors | Shuhong Zheng, Michael Oechsle, Erik Sandström, Marie-Julie Rakotosaona, Federico Tombari, Igor Gilitschenski |
| Publication | arXiv preprint |
| Version checked | arXiv:2605.23892v1, 2026-05-22 |
| Primary source | [arXiv](https://arxiv.org/abs/2605.23892) · [Project page](https://zsh2000.github.io/good-token-hunting.github.io/) |
| Official code | [zsh2000/gotohunt](https://github.com/zsh2000/gotohunt) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-011 |

## 30-Second Summary

Good Token Hunting reduces global-attention cost in visual geometry
transformers by selecting key/value tokens in two stages: diverse frames first,
then redundant tokens within retained frames. It is direct evidence that scene
coverage and layer-aware selection matter for multi-view geometry compute. It
is not evidence that the selected tokens form a durable or QA-sufficient memory.

## Problem Addressed

Global attention over long multi-view sequences grows quadratically with input
length. Uniform pruning can discard distinct viewpoints or geometry needed by
later layers.

## Relevant Method

- Inter-frame selection preserves a diverse set of views to cover the scene.
- Intra-frame selection removes further redundancy from retained views.
- The intra-frame policy is layer-aware and uses global-attention entropy.
- Selection restricts which key/value tokens each query attends to; it does not
  serialize a persistent world model.

## Paper-Reported Evidence

The authors report more than 85% acceleration on scenes with 500 images while
maintaining or sometimes improving the base model's geometry performance. The
official project page also reports camera-pose experiments on 7-Scenes, Neural
RGB-D, and TUM-Dynamics and reconstruction examples with each query attending
to tokens from 25 of up to 500 frames.

These are paper results, not results reproduced by this repository.

## What This Supports Here

- Separate view-level diversity from within-view token reduction.
- Include spatial coverage and novelty in a candidate's utility rather than
  relying only on a local confidence or attention score.
- Evaluate selection against geometry and pose metrics, not only throughput.
- Treat layer or representation stage as relevant when pruning transient
  geometry features.

## What It Does Not Prove

- That retained attention tokens preserve future geometry-grounded QA.
- That the method works with approximately 1 Hz monocular AI-glass input.
- That attention-compute reduction reduces persistent serialized bytes.
- That the reported server-side speedup transfers to an on-device student.

## Project Reproduction Status

Not reproduced. The repository has an independent causal, actual-byte writer
and linear selector baseline, but no GoToHunt implementation or benchmark run.
Any transfer from transient attention selection to persistent memory remains a
project hypothesis.

## References

- [Paper index](README.md)
- [arXiv:2605.23892](https://arxiv.org/abs/2605.23892)
- [Official project page](https://zsh2000.github.io/good-token-hunting.github.io/)
- [Official code](https://github.com/zsh2000/gotohunt)
