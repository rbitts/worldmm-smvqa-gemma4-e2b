# End-to-end Optimized Image Compression

| Field | Value |
|---|---|
| Page ID | SM-PAPER-E2E-IMAGE-COMPRESSION |
| Status | Reviewed; project uses the principle, not a reproduction |
| Publication | ICLR 2017 |
| Primary source | [ICLR paper](https://openreview.net/pdf?id=rJxdQ3jeg) |
| Official code | [Authors' code and model release](https://www.cns.nyu.edu/~lcv/iclr2017/) |
| Last checked | 2026-07-11 |
| Project links | [Paper index](README.md) · [Project home](../README.md) · [Problem](../problem.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-005, C-012 |

## 30-second summary

This work trains an analysis transform, quantizer, entropy model, and synthesis
transform jointly under a rate-distortion trade-off. Additive uniform noise
acts as a differentiable training proxy for quantization, while the learned
probability model estimates expected code length. The result established the
now-standard principle that compression must optimize task distortion and rate
together, rather than minimizing reconstruction error first and compressing
afterward.

For this project, the relevant transfer is the objective, not the image codec:
charge every candidate its real serialized-byte cost and retain information
according to downstream geometry-QA distortion.

## Problem addressed

Hand-designed image codecs separately choose transforms, quantization, and
entropy coding. Direct optimization is difficult because quantization is
discrete and the true bitstream rate is not differentiable. The paper seeks an
end-to-end trainable transform codec that operates at chosen points on the
rate-distortion curve.

## Relevant method

**Paper claim.** A nonlinear analysis transform maps an image into latent
coefficients. Uniform scalar quantization produces discrete codes, and a
nonlinear synthesis transform reconstructs the image. Training replaces
quantization with additive uniform noise. A learned density model supplies a
differentiable estimate of rate. The Lagrangian combines expected rate and
distortion, with a trade-off parameter selecting the operating point.

The original transforms use convolutional stages and generalized divisive
normalization. Those image-specific components are less relevant here than the
joint rate-distortion formulation.

## Paper-reported evidence

**Paper-reported result.** On held-out natural images, the authors report that
the optimized codec generally improves rate-distortion performance over JPEG
and JPEG 2000. They report improved visual quality across tested rates and
support that observation with MS-SSIM measurements. The released project page
also states that displayed rates and quality values were computed from learned
transforms, uniform scalar quantization, and entropy-coded bitstreams. See the
[official project page](https://www.cns.nyu.edu/~lcv/iclr2017/) and Section 4 of
the [ICLR paper](https://openreview.net/pdf?id=rJxdQ3jeg).

These are paper results for image reconstruction, not results for this spatial
memory.

## What this supports here

**Project inference.** The project should optimize a rate-task-distortion
objective in which rate is actual serialized bytes and distortion includes
future QA, grounding, geometry, association, uncertainty, and causal failures.
It supports comparing methods on a quality-versus-bytes Pareto curve and
rejecting token count or nominal bit width as sufficient rate measurements.

The repository's hard actual-byte writer follows this principle at the record
selection boundary. That implementation is an application of the idea, not a
reproduction of the paper's codec.

## What it does not prove

- Pixel reconstruction distortion is not geometry-grounded QA distortion.
- A continuous image entropy model does not define costs for heterogeneous
  object, plane, portal, landmark, or event records.
- The paper does not solve unknown future questions, entity association,
  coordinate frames, provenance, or temporal validity.
- It does not show that learned entropy estimates match this project's final
  serialization format.
- It does not evaluate SuperMemory-VQA, sparse 1 Hz sensing, repeated visits, or
  AI-glass deployment.

## Project reproduction status

**Project result.** The paper's image codec has not been reproduced. The local
pipeline does enforce actual serialized-byte caps for explicit records, but no
learned entropy model or end-to-end rate-distortion experiment has run. A valid
future result must report the exact serialization version, bytes, QA metrics,
geometry metrics, and Pareto operating points together.

## References

- Ballé, Laparra, and Simoncelli. [End-to-end Optimized Image
  Compression](https://openreview.net/pdf?id=rJxdQ3jeg). ICLR 2017.
- Authors' [official project, source, and model page](https://www.cns.nyu.edu/~lcv/iclr2017/).
- TensorFlow. [TensorFlow Compression](https://github.com/tensorflow/compression),
  linked by the authors as the maintained Python compression implementation.
