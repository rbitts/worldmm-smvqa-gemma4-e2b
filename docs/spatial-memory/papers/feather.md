# Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration

| Field | Value |
|---|---|
| Page ID | SM-PAPER-FEATHER |
| Status | Reviewed |
| Method name | FEATHER: Fast and Effective Acceleration wiTH Ensemble cRiteria |
| Authors | Mark Endo, Xiaohan Wang, Serena Yeung-Levy |
| Publication | ICCV 2025 |
| Version checked | arXiv:2412.13180v2, 2025-07-31 |
| Primary source | [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Endo_Feather_the_Throttle_Revisiting_Visual_Token_Pruning_for_Vision-Language_Model_ICCV_2025_paper.html) · [arXiv](https://arxiv.org/abs/2412.13180) |
| Official code | [markendo/FEATHER](https://github.com/markendo/FEATHER) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-011 |

## 30-Second Summary

FEATHER shows that aggressive early visual-token pruning can look successful on
broad VLM benchmarks while failing localization-sensitive tasks and deleting
most tokens from parts of the image. It uses multi-stage pruning with early
uniform sampling to preserve broad coverage. The paper is a warning that average
QA performance alone is an unsafe selector objective for spatial memory.

## Problem Addressed

Early token pruning accelerates vision-language models, but common criteria can
produce spatially biased retention. Benchmarks that do not demand fine-grained
localization may hide this failure.

## Relevant Method

- Diagnose the spatial distribution of retained visual tokens.
- Use early uniform sampling to ensure broad image coverage.
- Apply later pruning stages with multiple criteria to preserve relevant detail.
- Evaluate with vision-centric localization tasks, not only aggregate VLM tasks.

## Paper-Reported Evidence

The authors report that a prior acceleration approach prunes most tokens near
the top of images despite retaining strong scores on many benchmarks. At
comparable compute savings, FEATHER reports more than five-times performance
improvement over that approach on vision-centric localization benchmarks.

These are paper results, not results reproduced by this repository.

## What This Supports Here

- Treat region, object, and coordinate coverage as writer guardrails.
- Report localization, relation recall, and coordinate error alongside QA score.
- Include spatial retention heatmaps or coverage diagnostics in pruning
  experiments before accepting a selector.
- Keep a query-agnostic geometry core even when learned utility is low.

## What It Does Not Prove

- That uniform image coverage is optimal for 3D world coverage.
- That its VLM pruning policy preserves temporal changes or instance identity.
- That localization gains transfer to SuperMemory-VQA.
- That transient token coverage controls persistent serialized memory growth.

## Project Reproduction Status

Not reproduced. Current tests enforce explicit identity, geometry, causality,
and byte limits, but no FEATHER token-pruning implementation or localization
benchmark has run.

## References

- [Paper index](README.md)
- [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/html/Endo_Feather_the_Throttle_Revisiting_Visual_Token_Pruning_for_Vision-Language_Model_ICCV_2025_paper.html)
- [arXiv:2412.13180](https://arxiv.org/abs/2412.13180)
- [Official project page](https://web.stanford.edu/~markendo/projects/feather)
- [Official code](https://github.com/markendo/FEATHER)
