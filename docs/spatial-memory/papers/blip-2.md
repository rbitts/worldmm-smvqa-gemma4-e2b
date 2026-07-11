# BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

| Field | Value |
|---|---|
| Page ID | SM-PAPER-BLIP-2 |
| Status | Reviewed |
| Authors | Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi |
| Publication | ICML 2023, PMLR 202 |
| Version checked | arXiv:2301.12597v3, 2023-06-15 |
| Primary source | [PMLR](https://proceedings.mlr.press/v202/li23q.html) · [arXiv](https://arxiv.org/abs/2301.12597) |
| Official code | [LAVIS BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-010 |

## 30-Second Summary

BLIP-2 connects a frozen image encoder to a frozen language model through a
lightweight Querying Transformer, or Q-Former. Learned queries extract a small
visual representation through cross-attention. This supports a compact adapter
between a large geometry teacher and a task-specific decoder, but BLIP-2 itself
does not preserve metric geometry or persistent memory.

## Problem Addressed

End-to-end vision-language pre-training becomes expensive as both the visual
encoder and language model grow. Frozen pretrained components also have a
modality gap that requires a trainable bridge.

## Relevant Method

- A lightweight Q-Former contains learned query embeddings.
- Queries cross-attend to frozen image-encoder features.
- Stage one bootstraps image-text representation learning from the frozen image
  encoder; stage two bootstraps generation through the frozen language model.
- Only the bridge and associated projections need task-specific training in the
  central setup.

## Paper-Reported Evidence

BLIP-2 reports state-of-the-art results across several vision-language tasks
with far fewer trainable parameters than prior systems. The paper reports an
8.7% improvement over Flamingo-80B on zero-shot VQAv2 while using 54 times fewer
trainable parameters.

These are paper results, not results reproduced by this repository.

## What This Supports Here

- Keep a large pretrained geometry provider frozen while training a small query
  adapter and typed output heads.
- Use bounded learned queries to extract task-relevant information from dense
  provider features.
- Stage training so representation alignment precedes downstream QA utility.

## What It Does Not Prove

- That Q-Former outputs encode coordinate frames, uncertainty, or provenance.
- That vision-language alignment preserves low-level metric accuracy.
- That query tokens remain stable object identities across days.
- That the architecture satisfies an actual serialized-byte budget.

## Project Reproduction Status

Not reproduced. No Q-Former exists in the repository. Current code defines a
provider cache contract and typed candidate head, but not a learned query bridge
between G-CUT3R features and persistent records.

## References

- [Paper index](README.md)
- [ICML paper](https://proceedings.mlr.press/v202/li23q.html)
- [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)
- [Official code](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
