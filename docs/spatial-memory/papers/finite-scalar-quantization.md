# Finite Scalar Quantization: VQ-VAE Made Simple

| Field | Value |
|---|---|
| Page ID | SM-PAPER-FSQ |
| Status | Reviewed; not reproduced in this project |
| Publication | ICLR 2024 |
| Primary source | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e2dd53601de57c773343a7cdf09fae1c-Abstract-Conference.html) |
| Official code | [Google Research FSQ](https://github.com/google-research/google-research/tree/master/fsq) |
| Last checked | 2026-07-11 |
| Project links | [Paper index](README.md) · [Project home](../README.md) · [Problem](../problem.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-012 |

## 30-second summary

Finite Scalar Quantization (FSQ) replaces learned nearest-neighbor vector
quantization with a low-dimensional vector whose individual dimensions are
rounded to fixed finite levels. The Cartesian product of those levels is an
implicit codebook. The paper reports performance competitive with VQ on image
generation and dense prediction while avoiding learned-codebook collapse and
VQ-specific auxiliary machinery.

For this project, FSQ is a useful compact-latent baseline. It is not the main
answer to lifelong spatial memory because it makes already-generated features
smaller; it does not decide which geometry-grounded facts should exist.

## Problem addressed

VQ-VAE discretization normally learns a codebook and maps every encoder vector
to its nearest entry. Large learned codebooks can be underused and often need
commitment losses, codebook updates, reseeding, splitting, or entropy
regularization. FSQ targets this optimization and codebook-utilization problem.

## Relevant method

**Paper claim.** The encoder first projects each latent vector to a small number
of dimensions, typically fewer than ten. Dimension `i` is bounded and rounded
to one of `L_i` fixed levels with a straight-through gradient. The implicit
codebook contains `product(L_i)` combinations, but no embedding table is
learned. A mixed-radix mapping converts each combination to a discrete index.

The same downstream models used with VQ tokens can consume FSQ indices. The
paper evaluates this replacement in MaskGIT and UViM rather than proposing a
new spatial-memory representation.

## Paper-reported evidence

**Paper-reported result.** On 256-by-256 ImageNet generation with MaskGIT, the
paper reports comparable FID, precision, recall, and qualitative samples for
FSQ and VQ. On UViM, FSQ is reported as competitive with VQ for NYU depth,
colorization, and panoptic segmentation. In the NYU depth comparison without
codebook splitting, the paper reports 99% FSQ code usage; the corresponding VQ
variant uses 0.78% and has worse RMSE. See Sections 5.2 and 5.3 and Table 2 of
the [conference paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/e2dd53601de57c773343a7cdf09fae1c-Paper-Conference.pdf).

These are results reported by the authors. This repository has not reproduced
them.

## What this supports here

**Project inference.** FSQ supports keeping a simple scalar-discrete baseline
for compact association descriptors or other latent fields when a typed record
still needs a learned code. It also supports testing whether a learned VQ
codebook is necessary before accepting its training and maintenance cost.

Any use here must still measure actual serialized bytes and downstream
geometry-QA quality. Codebook size alone is not a storage result.

## What it does not prove

- It does not select objects, planes, portals, landmarks, or change events.
- It does not preserve entity identity, coordinate frames, uncertainty,
  provenance, temporal validity, or causal evidence.
- It does not show that scalar-quantized generic features are sufficient for
  SuperMemory-VQA.
- It does not establish performance at 1 Hz, across repeated visits, or on an
  AI-glass device.
- It does not replace the project's actual-byte writer or deterministic
  geometry executor.

## Project reproduction status

**Project result.** Not reproduced. No FSQ module, checkpoint, or benchmark
artifact is currently part of the project. FSQ remains an optional compression
baseline to add only if compact learned descriptors become necessary after the
explicit typed-record baseline is measured.

## References

- Mentzer, Minnen, Agustsson, and Tschannen. [Finite Scalar Quantization:
  VQ-VAE Made Simple](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e2dd53601de57c773343a7cdf09fae1c-Abstract-Conference.html).
  ICLR 2024.
- Authors' [official FSQ JAX implementation](https://github.com/google-research/google-research/tree/master/fsq).
- [Paper appendix and reference implementation](https://proceedings.iclr.cc/paper_files/paper/2024/file/e2dd53601de57c773343a7cdf09fae1c-Supplementary-Conference.pdf).
