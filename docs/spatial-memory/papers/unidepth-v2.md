# UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler

| Field | Value |
|---|---|
| Page ID | SM-PAPER-UNIDEPTH-V2 |
| Status | Reviewed from primary sources; not reproduced in this project |
| Publication | IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(3), 2026, pp. 2354–2367 |
| Primary source | [Official DOI](https://doi.org/10.1109/TPAMI.2025.3628473) |
| Official code | [lpiccinelli-eth/UniDepth](https://github.com/lpiccinelli-eth/UniDepth) |
| Last checked | 2026-07-11 |
| Project links | [Paper index](README.md), [Architecture](../architecture.md), [Roadmap](../roadmap.md) |
| Project claims | [Traceability](../traceability.md): C-001 |

## 30-second summary

UniDepthV2 predicts metric 3D points and depth from a single RGB image without
requiring camera intrinsics. A self-prompted dense camera representation conditions
the depth module, while a pseudo-spherical output separates camera rays from radial
depth. The model also predicts per-pixel uncertainty and is released in small,
base, and large variants.

## Problem addressed

Monocular metric depth models often generalize poorly when scene scale, camera
intrinsics, or domain changes. Requiring ground-truth intrinsics also restricts
in-the-wild use. UniDepthV2 targets universal metric depth and 3D prediction from
RGB alone while improving edge sharpness, efficiency, input-shape robustness, and
confidence output.

## Relevant method

A camera module predicts dense ray angles and prompts a depth module. The output
space uses azimuth, elevation, and radial depth so camera and depth errors are
separated. Geometric invariance training uses transformed views of the same image.
An edge-guided scale-shift-invariant loss sharpens discontinuities, and an
uncertainty head learns absolute log-depth error ranking.

The paper reports training on 16 million images from 23 datasets, then evaluates
zero-shot transfer on ten unseen indoor, outdoor, and challenging datasets.

## Paper-reported evidence

These are paper results, not results from this repository.

| Dataset or condition | Metric | Reported result | Source location |
|---|---|---:|---|
| SUN-RGBD zero-shot, UniDepthV2-Large | delta1 / absolute relative error | 96.4 / 6.8 | Table I, paper p. 7 |
| IBims-1 zero-shot, UniDepthV2-Large | delta1 / absolute relative error | 94.5 / 7.8 | Table I, paper p. 7 |
| NuScenes zero-shot, UniDepthV2-Large | delta1 / absolute relative error | 87.0 / 15.0 | Table II, paper p. 7 |
| A6000, mixed precision, 0.5-megapixel input, Small | Latency / parameters / memory | 23.0 ms / 34.18M / 0.66 GiB | Table VIII, paper p. 10 |
| Same setting, Large | Latency / parameters / memory | 65.4 ms / 353.8M / 3.47 GiB | Table VIII, paper p. 10 |
| Aggregated zero-shot uncertainty, Large | nAUSE / Spearman correlation | 0.645 / 0.299 | Table VII, paper p. 9 |

The uncertainty table is an important qualification. In-domain Large reports
nAUSE 0.199 and rank correlation 0.744, but zero-shot values degrade to 0.645 and
0.299. The paper therefore supports informative uncertainty ranking under domain
shift, not perfectly calibrated absolute confidence.

## What this supports here

**Paper claim:** monocular RGB can provide metric depth, inferred camera geometry,
3D points, and a useful uncertainty signal without supplied intrinsics across a
broad zero-shot test suite.

**Project inference:** UniDepthV2 is a stronger metric single-frame baseline than
relative-depth-only providers. Its small variant is a candidate for per-frame
geometry proposals, and its uncertainty can inform a write gate after project-side
calibration.

Metric depth may be combined with externally trusted VIO pose to create temporary
world-frame candidates before typed-record selection.

## What it does not prove

- Single-image metric depth does not establish temporal consistency, loop closure,
  instance identity, or a persistent common coordinate frame.
- The uncertainty output is not fully calibrated under domain shift, as shown by
  the zero-shot uncertainty metrics.
- Paper GPU latency does not establish AI-glasses power, thermal, or memory fit.
- It does not evaluate 1 Hz lifelong streams, typed spatial compression,
  SuperMemory-VQA, or deterministic geometry QA proofs.
- Metric depth accuracy does not determine which records maximize future QA value
  per byte.

## Project reproduction status

UniDepthV2 is not installed, downloaded, or executed locally. No official model
weights or predictions are stored in the repository. It remains a planned
lightweight metric-depth baseline and potential student component.

## References

- [Official DOI](https://doi.org/10.1109/TPAMI.2025.3628473)
- [Official arXiv record, version 2](https://arxiv.org/abs/2502.20110)
- [Official code and model documentation](https://github.com/lpiccinelli-eth/UniDepth)

[Back to paper index](README.md)
