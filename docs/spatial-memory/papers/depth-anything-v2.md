# Depth Anything V2

| Field | Value |
|---|---|
| Page ID | SM-PAPER-DEPTH-ANYTHING-V2 |
| Status | Reviewed from primary sources; not reproduced in this project |
| Publication | NeurIPS 2024, Main Conference Track |
| Primary source | [Official NeurIPS paper record](https://proceedings.neurips.cc/paper_files/paper/2024/hash/26cfdcd8fe6fd75cc53e92963a656c58-Abstract-Conference.html) |
| Official code | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) |
| Last checked | 2026-07-11 |
| Project links | [Paper index](README.md), [Architecture](../architecture.md), [Roadmap](../roadmap.md) |
| Project claims | [Traceability](../traceability.md): C-001 |

## 30-second summary

Depth Anything V2 is a family of monocular depth foundation models from 25 million
to 1.3 billion parameters. A DINOv2-G teacher learns from precise synthetic depth,
labels 62 million real images, and distills robust student models. The base models
predict affine-invariant relative inverse depth; separate indoor and outdoor models
are fine-tuned for metric depth.

## Problem addressed

Real depth labels often miss thin structures and fail on transparent or reflective
surfaces. Synthetic labels are precise but have domain and coverage gaps. Depth
Anything V2 combines synthetic supervision with large-scale pseudo-labeled real
images to improve details, robustness, efficiency, and transferability.

## Relevant method

The training pipeline has three stages: train a large DINOv2-G teacher on 595,000
synthetic images, generate pseudo depth for more than 62 million unlabeled real
images, then train DINOv2-based students on pseudo-labeled real images. DPT is the
depth decoder. The released relative-depth students span ViT-S, ViT-B, and ViT-L;
metric variants are fine-tuned using indoor or outdoor metric supervision.

## Paper-reported evidence

These are paper results, not results from this repository.

| Dataset or condition | Metric | Reported result | Source location |
|---|---|---:|---|
| DA-2K, ViT-S relative-depth model | Accuracy / parameters / V100 latency | 95.3% / about 25M / 60 ms | Figure 1 and Table 3, paper pp. 1 and 8 |
| DA-2K, ViT-L relative-depth model | Accuracy / parameters / V100 latency | 97.1% / about 335M / 213 ms | Figure 1 and Table 3, paper pp. 1 and 8 |
| DA-2K, Marigold comparison | Accuracy / parameters / V100 latency | 86.8% / 948M / 5.2 s | Figure 1 and Table 3, paper pp. 1 and 8 |
| ViT-S, synthetic-only versus pseudo-real-only training | DA-2K accuracy | 89.8% versus 95.3% | Table 5, paper p. 9 |
| Metric fine-tuning, ViT-L | NYU-D delta1 / Abs Rel; KITTI delta1 / Abs Rel | 0.984 / 0.056; 0.983 / 0.045 | Table 4, paper p. 9 |

The conventional zero-shot table reports results comparable to Depth Anything V1
on several datasets. The paper explicitly argues that those noisy benchmarks do
not fully measure its claimed gains on fine structures, transparent surfaces, and
complex layouts.

## What this supports here

**Paper claim:** a relatively small monocular model can produce strong per-frame
depth, and teacher-to-student pseudo-labeling transfers geometry capability across
model sizes.

**Project inference:** Depth Anything V2-Small is a practical lightweight geometry
baseline or student feature source when a full multi-view teacher is unnecessary.
Its metric variants can provide per-frame depth candidates for fusion with trusted
VIO poses.

The teacher/pseudo-label/student pipeline also supports using a large offline
geometry provider while deploying a smaller on-device candidate encoder.

## What it does not prove

- The main relative-depth models do not supply absolute metric scale.
- Per-frame monocular accuracy does not establish temporal consistency, common
  world coordinates, camera pose, or instance association.
- The metric results are from domain-specific fine-tuning, not a universal 1 Hz
  streaming reconstruction test.
- It does not evaluate persistent memory, actual storage bytes, SuperMemory-VQA,
  or geometry-grounded QA.
- Reported V100 latency does not establish AI-glasses power or thermal feasibility.

## Project reproduction status

Depth Anything V2 is not installed, downloaded, or executed locally. It is a
planned lightweight depth-provider baseline. No project result currently measures
its temporal consistency, typed-record utility, or SuperMemory-VQA impact.

## References

- [Official NeurIPS 2024 record](https://proceedings.neurips.cc/paper_files/paper/2024/hash/26cfdcd8fe6fd75cc53e92963a656c58-Abstract-Conference.html)
- [Official project page](https://depth-anything-v2.github.io/)
- [Official code](https://github.com/DepthAnything/Depth-Anything-V2)
- [Official arXiv record](https://arxiv.org/abs/2406.09414)

[Back to paper index](README.md)
