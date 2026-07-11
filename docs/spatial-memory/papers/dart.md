# Stop Looking for Important Tokens in Multimodal Language Models: Duplication Matters More

| Field | Value |
|---|---|
| Page ID | SM-PAPER-DART |
| Status | Reviewed |
| Method name | DART: Duplication-Aware Reduction of Tokens |
| Authors | Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, Linfeng Zhang |
| Publication | EMNLP 2025 main, pages 9961-9980 |
| Version checked | arXiv:2502.11494v2, 2025-06-08 |
| Primary source | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.505/) · [arXiv](https://arxiv.org/abs/2502.11494) |
| Official code | [ZichenWen1/DART](https://github.com/ZichenWen1/DART) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-011 |

## 30-Second Summary

DART argues that redundancy is a better pruning signal than estimated token
importance for multimodal language models. It keeps tokens with low similarity
to a small pivot set and requires no additional training. This supports removing
duplicate candidates before an expensive learned memory writer, but it does not
identify which spatial facts must survive long-term storage.

## Problem Addressed

Long visual-token sequences dominate multimodal model inference cost. Existing
importance-based pruning can underperform random pruning and can interact poorly
with efficient attention kernels.

## Relevant Method

- Select a small subset of pivot tokens.
- Measure other tokens' duplication relative to the pivots.
- Retain tokens with low duplication, preserving distinct information.
- Apply the reduction at inference without retraining the source model.

## Paper-Reported Evidence

The authors report pruning 88.9% of visual tokens while maintaining comparable
performance. They report 1.99 times total-time speedup and 2.99 times prefilling
speedup, together with compatibility with efficient attention operators.

These are paper results, not results reproduced by this repository.

## What This Supports Here

- Deduplicate identical or near-identical coordinate, instance, relation, and
  observation candidates before value-per-byte selection.
- Preserve novel candidates rather than treating high attention as the only
  measure of value.
- Measure whether duplicate removal improves the downstream writer's Pareto
  curve under the same serialized-byte budget.

## What It Does Not Prove

- That low feature duplication implies high future QA value.
- That generic VLM similarity is safe for metric geometry records.
- That inference-token savings translate to persistent-memory savings.
- That DART preserves temporal state changes or object identity.

## Project Reproduction Status

Not reproduced. The current writer performs deterministic same-key retention
and actual-byte admission, but it is not DART and has no pivot-token stage.
Duplicate-first reduction remains a candidate ablation, not a reported project
result.

## References

- [Paper index](README.md)
- [EMNLP paper](https://aclanthology.org/2025.emnlp-main.505/)
- [arXiv:2502.11494](https://arxiv.org/abs/2502.11494)
- [Official code](https://github.com/ZichenWen1/DART)
