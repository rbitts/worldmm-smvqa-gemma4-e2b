# QVGGT: Post-Training Quantized Visual Geometry Grounded Transformer

| Field | Value |
|---|---|
| Page ID | SM-PAPER-QVGGT |
| Status | Reviewed; official implementation not available |
| Publication | CVPR 2026, pages 7536-7545 |
| Primary source | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_QVGGT_Post-Training_Quantized_Visual_Geometry_Grounded_Transformer_CVPR_2026_paper.html) |
| Official code | Not published on the [official project page](https://ddsacu.github.io/QVGGT/) as of 2026-07-11 |
| Last checked | 2026-07-11 |
| Project links | [Paper index](README.md) · [Project home](../README.md) · [Problem](../problem.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-012 |

## 30-second summary

QVGGT applies post-training quantization to the 1.2-billion-parameter VGGT
geometry model. It combines block-sensitivity-based mixed precision, filtering
of high-variance camera and register tokens during activation calibration, a
PCA-derived camera-information compensation token, and task-aware quantization
scale search across camera, depth, and point-map heads.

For this project, QVGGT is evidence that geometry models need geometry-aware
quantization calibration. It is not evidence that quantizing a geometry model
compresses the persistent spatial memory produced by that model.

## Problem addressed

Generic post-training quantization treats transformer blocks and calibration
tokens too uniformly. VGGT has heterogeneous block sensitivity and special
camera and register tokens with high-variance activations. Their outliers can
distort calibration and propagate error across camera pose, depth, and point-map
predictions.

## Relevant method

**Paper claim.** QVGGT uses three components:

1. Per-block sensitivity analysis assigns higher precision to fragile
   frame-wise or global transformer blocks.
2. Camera and register tokens are omitted while collecting activation
   statistics. A global compensation token derived by top-K PCA is injected
   into the camera head to restore camera information.
3. Quantization scales are selected with an objective combining layer
   reconstruction, multi-head supervision, and cross-head geometric
   consistency among pose, depth, and point maps.

This is post-training model quantization: it changes weights and activation
precision without defining a new long-term scene-memory schema.

## Paper-reported evidence

**Paper-reported result.** The authors report near-lossless W4A16 results across
camera-pose and reconstruction benchmarks. Relative to FP32, the abstract
reports 3 to 4.9 times memory reduction and up to 2.8 times real-hardware
speedup while preserving the accuracy of all three geometry heads. Evaluations
cover camera pose on CO3Dv2 and RealEstate10K and reconstruction on 7-Scenes and
Neural RGB-D. See the [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_QVGGT_Post-Training_Quantized_Visual_Geometry_Grounded_Transformer_CVPR_2026_paper.pdf)
and its supplementary material.

These numbers are author-reported. No QVGGT result in this repository exists.

## What this supports here

**Project inference.** If a large visual-geometry teacher or compact student is
quantized for deployment, calibration should preserve geometry-head outputs and
cross-head consistency, not only transformer-layer reconstruction. Mixed
precision should follow measured block sensitivity, and special pose-related
tokens should receive explicit treatment.

QVGGT is therefore relevant to a later model-deployment phase. It does not
replace the current priority: produce explicit typed records, validate their
causal geometry proofs, and measure their serialized bytes.

## What it does not prove

- Model-weight and activation compression is not persistent-memory compression.
- The paper does not select or serialize objects, planes, portals, landmarks,
  free space, relations, or events.
- It does not optimize future QA utility or repeated-visit memory growth.
- It does not establish G-CUT3R, CUT3R, or project-student compatibility.
- It does not evaluate 1 Hz lifelong streams, SuperMemory-VQA, or the target
  AI-glass hardware.
- The official project page does not currently provide code, so independent
  reproduction details remain incomplete.

## Project reproduction status

**Project result.** Not reproduced. No QVGGT code, calibration data, quantized
checkpoint, hardware profile, or benchmark artifact is present locally. The
paper can currently justify a future geometry-aware PTQ ablation only; it
cannot justify a claimed deployment result.

## References

- Pan, Wang, and Wang. [QVGGT: Post-Training Quantized Visual Geometry Grounded
  Transformer](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_QVGGT_Post-Training_Quantized_Visual_Geometry_Grounded_Transformer_CVPR_2026_paper.html).
  CVPR 2026.
- Authors' [official QVGGT project page](https://ddsacu.github.io/QVGGT/).
- Authors' [arXiv record, version 1](https://arxiv.org/abs/2605.31124).
