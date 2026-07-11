# Perceiver: General Perception with Iterative Attention

| Field | Value |
|---|---|
| Page ID | SM-PAPER-PERCEIVER |
| Status | Reviewed |
| Authors | Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, Joao Carreira |
| Publication | ICML 2021, PMLR 139 |
| Version checked | arXiv:2103.03206v2, 2021-06-23 |
| Primary source | [PMLR](https://proceedings.mlr.press/v139/jaegle21a.html) · [arXiv](https://arxiv.org/abs/2103.03206) |
| Official code | [DeepMind Perceiver](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-010 |

## 30-Second Summary

Perceiver uses an asymmetric cross-attention operation to read a very large
input array into a much smaller learned latent array, then performs most
processing in latent space. It supports the architectural idea of fixed learned
queries reading dense geometry-provider features. It does not define explicit
spatial records or a lifelong retention policy.

## Problem Addressed

Standard transformers apply self-attention directly to every input element,
making high-dimensional pixels, point clouds, audio, and video expensive and
encouraging modality-specific preprocessing.

## Relevant Method

- A fixed-size learned latent array cross-attends to the input array.
- Latent self-attention performs deeper processing at the smaller latent size.
- Cross-attention can be repeated, iteratively distilling information from a
  large input without full input-to-input attention.
- Positional and modality information enters through the input representation;
  the latent slots are not inherently metric or semantic entities.

## Paper-Reported Evidence

The paper reports scaling to hundreds of thousands of inputs. It obtains
ImageNet performance comparable to ResNet-50 and ViT while attending directly
to 50,000 pixels without 2D convolutions, and reports competitive results for
point clouds, audio, video, and video-plus-audio.

These are paper results, not results reproduced by this repository.

## What This Supports Here

- Decode a large transient feature set through a bounded learned latent array.
- Keep candidate count independent of raw point or patch count.
- Attach separate object, structure, landmark, and event heads to latent slots
  instead of treating the slots themselves as the database schema.

## What It Does Not Prove

- That latent slots are stable across time or correspond to persistent objects.
- That the bottleneck preserves precise distance, pose, or topology.
- That fixed latent count bounds serialized memory across repeated visits.
- That a Perceiver trained for classification transfers to SuperMemory-VQA.

## Project Reproduction Status

Not reproduced. No Perceiver decoder is implemented. The repository currently
uses explicit heuristic candidates plus typed-record training scaffolding. A
Perceiver-style decoder should be added only after the provider-to-record
inference path exists and the heuristic baseline exposes a candidate bottleneck.

## References

- [Paper index](README.md)
- [ICML paper](https://proceedings.mlr.press/v139/jaegle21a.html)
- [arXiv:2103.03206](https://arxiv.org/abs/2103.03206)
- [Official code](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver)
