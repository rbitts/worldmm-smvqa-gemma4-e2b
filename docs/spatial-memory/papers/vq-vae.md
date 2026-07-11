# Neural Discrete Representation Learning (VQ-VAE)

| Field | Value |
|---|---|
| Page ID | SM-PAPER-VQ-VAE |
| Status | Reviewed; not reproduced in this project |
| Publication | NeurIPS 2017 |
| Primary source | [NeurIPS proceedings](https://papers.neurips.cc/paper_files/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html) |
| Official code | [Google DeepMind Sonnet VQ-VAE example](https://github.com/google-deepmind/sonnet/blob/v1/sonnet/examples/vqvae_example.ipynb) |
| Last checked | 2026-07-11 |
| Project links | [Paper index](README.md) · [Project home](../README.md) · [Problem](../problem.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-012 |

## 30-second summary

VQ-VAE learns discrete latent indices by replacing each encoder output with its
nearest vector from a learned codebook. A straight-through estimator trains the
encoder through the non-differentiable lookup, while codebook and commitment
terms train the embeddings and keep encoder outputs near them. A separate
autoregressive prior can then model the discrete sequence.

For this project, VQ-VAE defines the conventional "generate a latent, then
quantize it" baseline. The proposed spatial-memory direction instead tries to
generate only explicit QA-relevant records before any optional field-level
quantization.

## Problem addressed

Continuous VAE latents can be ignored when paired with a strong autoregressive
decoder, producing posterior collapse. The paper seeks useful discrete latent
representations that can be learned without supervision and modeled by a
separate prior.

## Relevant method

**Paper claim.** An encoder produces `z_e(x)`. Each vector is replaced by the
nearest learned embedding `e_k`, yielding discrete `z_q(x)`. The decoder
reconstructs the input from `z_q`. Training combines reconstruction, codebook,
and commitment terms. Stop-gradient operators direct the codebook and encoder
updates, while the straight-through estimator copies decoder gradients through
the discrete lookup. After representation learning, an autoregressive model
learns a prior over code indices.

The quantizer therefore reduces a dense latent vector to an index, but the
meaning of that index remains distributed and decoder-dependent.

## Paper-reported evidence

**Paper-reported result.** The authors report useful discrete representations
across images, video, and speech. The paper demonstrates reconstructions and
samples, speaker conversion, and unsupervised discovery of phoneme-like speech
units. It also reports that the discrete bottleneck avoids the posterior
collapse observed with continuous latent variables under powerful decoders.
See the [NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf),
especially Sections 2 through 4.

These are author-reported results. They are not project results.

## What this supports here

**Project inference.** VQ-VAE supports a generic discrete-latent comparison for
any learned spatial descriptor that cannot be represented directly as typed
geometry. It provides the conventional baseline against which FSQ or direct
record generation can be evaluated at equal serialized-byte budgets.

It also clarifies an architectural boundary: discrete latent compression and
semantic record selection are different operations and need separate
ablations.

## What it does not prove

- A code index is not an explicit object, plane, portal, relation, or event.
- The paper does not optimize future QA utility, geometry information gain, or
  serialized bytes for heterogeneous records.
- It does not guarantee stable code meanings across checkpoints or long-lived
  updates.
- It does not provide metric coordinate frames, uncertainty, provenance, or
  temporal validity.
- It does not evaluate sparse 1 Hz observations, SuperMemory-VQA, repeated
  visits, or on-device geometry processing.

## Project reproduction status

**Project result.** Not reproduced. The repository contains no VQ-VAE training
run, learned codebook, or benchmark result. If a learned-descriptor baseline is
later required, the minimum comparison is VQ-VAE versus FSQ versus the direct
typed record under the same real byte budget and QA protocol.

## References

- van den Oord, Vinyals, and Kavukcuoglu. [Neural Discrete Representation
  Learning](https://papers.neurips.cc/paper_files/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html).
  NeurIPS 2017.
- Google DeepMind. [Sonnet VQ-VAE example](https://github.com/google-deepmind/sonnet/blob/v1/sonnet/examples/vqvae_example.ipynb).
- Google DeepMind. [Maintained Sonnet VQ-VAE layer](https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py).
