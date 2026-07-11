# TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?

| Field | Value |
|---|---|
| Page ID | SM-PAPER-TOKENLEARNER |
| Status | Reviewed |
| Authors | Michael S. Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, Anelia Angelova |
| Publication | NeurIPS 2021; extended arXiv version |
| Version checked | arXiv:2106.11297v4, 2022-04-03 |
| Primary source | [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/6a30e32e56fce5cf381895dfe6ca7b6f-Abstract.html) · [Extended arXiv paper](https://arxiv.org/abs/2106.11297) |
| Official code | [Scenic TokenLearner](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-010 |

## 30-Second Summary

TokenLearner converts a dense image-like feature tensor into a small,
input-adaptive set of tokens using learned spatial attention and pooling. It
supports using a fixed token count as an explicit compute budget. Its tokens
are recognition features, not metric entities, coordinate-bearing records, or
long-term memory entries.

## Problem Addressed

Vision transformers repeatedly process hundreds or thousands of patch tokens,
with quadratic self-attention cost. Fixed grids spend equal compute on useful
and redundant spatial regions.

## Relevant Method

- Learned spatial weight maps identify input-dependent regions.
- Each weight map reweights the feature tensor; global spatial pooling produces
  one learned token.
- A small fixed output count, commonly 8 or 16, replaces a much larger patch
  sequence for subsequent transformer layers.
- An optional TokenFuser maps processed tokens back to a spatial feature map
  when dense output is required.

## Paper-Reported Evidence

The paper reports that TokenLearner can cut transformer computation by about
half or more without damaging classification performance. In one reported
ImageNet comparison, ViT-L/16 uses 363.1 GFLOPs for 87.35 top-1 accuracy, while
the 16-token variant inserted at layer 12 uses 178.1 GFLOPs for 87.68. The
paper also evaluates Kinetics-400, Kinetics-600, Charades, and AViD.

These are paper results, not results reproduced by this repository.

## What This Supports Here

- Use a fixed number of learned slots to make candidate-decoder cost explicit.
- Compare slot budgets such as 4, 8, 16, and 32 under the same downstream QA
  and geometry tests.
- Make slots input-adaptive instead of uniformly pooling dense provider output.

## What It Does Not Prove

- That eight generic visual tokens retain metric coordinates or instance IDs.
- That a fixed latent bottleneck is sufficient for unknown future questions.
- That recognition accuracy predicts spatial relation or last-seen accuracy.
- That transient tokens should be stored as the persistent representation.

## Project Reproduction Status

Not reproduced. The current decoder emits heuristic explicit object, relation,
and zone candidates; it is not a TokenLearner module. A learned fixed-slot
decoder remains planned and must emit typed geometry plus uncertainty rather
than generic visual features.

## References

- [Paper index](README.md)
- [arXiv:2106.11297](https://arxiv.org/abs/2106.11297)
- [Official implementation in Scenic](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner)
