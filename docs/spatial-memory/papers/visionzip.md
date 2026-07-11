# VisionZip: Longer is Better but Not Necessary in Vision Language Models

| Field | Value |
|---|---|
| Page ID | SM-PAPER-VISIONZIP |
| Status | Reviewed |
| Authors | Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, Jiaya Jia |
| Publication | CVPR 2025 |
| Version checked | arXiv:2412.04467v2, 2026-03-15 |
| Primary source | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_VisionZip_Longer_is_Better_but_Not_Necessary_in_Vision_Language_CVPR_2025_paper.html) · [arXiv](https://arxiv.org/abs/2412.04467) |
| Official code | [JIA-Lab-research/VisionZip](https://github.com/JIA-Lab-research/VisionZip) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-011 |

## 30-Second Summary

VisionZip reduces redundant visual tokens before language-model processing. It
retains dominant tokens and merges contextual information so the language model
receives a shorter visual sequence. This supports early redundancy reduction,
but the output remains a VLM input representation rather than an explicit,
auditable spatial database.

## Problem Addressed

Modern vision-language models improve quality partly by increasing visual-token
length, which raises prefilling and generation cost. CLIP- and SigLIP-derived
visual sequences contain substantial redundancy.

## Relevant Method

- Identify dominant tokens that carry globally important visual information.
- Preserve contextual information by merging other token content instead of
  simply deleting every non-dominant token.
- Apply the compacted representation to image, video, and multi-turn VLM use.

## Paper-Reported Evidence

The paper reports at least 5% gains over the previous state of the art across
nearly all tested compression settings and an eight-times improvement in
prefilling time. It also reports that compressed LLaVA-NeXT-13B runs faster
than LLaVA-NeXT-7B while achieving better evaluated results.

These are paper results, not results reproduced by this repository.

## What This Supports Here

- Reduce redundant provider features before the typed candidate decoder.
- Compare deletion against merge-and-summarize policies for contextual evidence.
- Include VLM prefilling as a separate cost from persistent memory bytes.

## What It Does Not Prove

- That dominant visual tokens preserve object identity or metric coordinates.
- That token merging preserves uncertainty and provenance composition.
- That VLM benchmark quality predicts geometry-grounded QA quality.
- That transient input compression bounds lifelong memory growth.

## Project Reproduction Status

Not reproduced. The repository does not implement VisionZip. Current compaction
operates on explicit record candidates after geometry extraction and therefore
tests a different representation and objective.

## References

- [Paper index](README.md)
- [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_VisionZip_Longer_is_Better_but_Not_Necessary_in_Vision_Language_CVPR_2025_paper.html)
- [arXiv:2412.04467](https://arxiv.org/abs/2412.04467)
- [Official code](https://github.com/JIA-Lab-research/VisionZip)
