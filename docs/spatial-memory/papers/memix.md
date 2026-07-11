# MeMix: Writing Less, Remembering More for Streaming 3D Reconstruction

| Field | Value |
|---|---|
| Page ID | SM-PAPER-MEMIX |
| Status | reviewed; code available |
| Publication | arXiv:2603.15330 v1, 2026 preprint |
| Primary source | [arXiv](https://arxiv.org/abs/2603.15330) · [Project page](https://dongjiacheng06.github.io/MeMix/) |
| Official code | [dongjiacheng06/MeMix](https://github.com/dongjiacheng06/MeMix) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-002, C-011 |

## 30-second summary

MeMix is a training-free update policy for recurrent streaming 3D reconstruction. It divides one recurrent state into memory patches, updates the least-aligned patches, and preserves the rest exactly. It provides a strong baseline for selective working-state updates, but it neither creates typed explicit records nor optimizes future QA value per stored byte.

## Problem addressed

Uniformly rewriting a fixed recurrent state can accumulate drift and erase useful history over long streams. MeMix aims to reduce destructive writes without fine-tuning, extra learned parameters, or sequence-length-dependent inference memory.

## Relevant method

- Partition the recurrent state into independent memory patches.
- Score alignment between candidate state patches and current image tokens.
- Update only the bottom-k, least-aligned patches with a binary mask.
- Preserve unselected patches exactly.
- Apply the module to CUT3R, TTT3R, or TTSA3R without retraining the backbone.

## Paper-reported evidence

**Reported claim.** Across the paper's standard reconstruction benchmarks and matched backbone settings, MeMix reports an average 15.3% reduction in completeness error, with up to 40.0%, on 300–500-frame 7-Scenes streams. The official project page also reports unchanged peak GPU memory for its tested backbones; for CUT3R it lists 14.39 versus 14.13 FPS and 5.31 GB in both cases without and with MeMix.

The default setting updates 708 of 768 state tokens, or about 92.2%. It therefore reduces destructive rewriting but is not a strongly sparse write policy under that setting.

**Project inference.** Bottom-k patch update is a useful training-free baseline against a task-specific event or value-per-byte gate.

**Project result.** None. This repository has not reproduced MeMix.

## What this supports here

- Measuring writes per minute rather than only final reconstruction quality.
- Testing skip, update, insert, and expire decisions only when observations disagree with retained state.
- Preserving stable working-memory regions instead of rewriting the whole recurrent state.

The learned QA-aware gate proposed by this project is an extension, not a MeMix contribution.

## What it does not prove

- That least alignment equals geometry novelty, semantic importance, or future QA utility.
- Persistent-memory byte reduction; constant GPU state is not a serialized storage measurement.
- Entity identity, metric proof, uncertainty, temporal validity, or provenance support.
- Performance under 1 Hz sensing, multi-day memory, AI-glass hardware, or SuperMemory-VQA.
- Thousand-frame and kilometer-scale behavior; the paper does not evaluate either regime.

## Project reproduction status

Not reproduced. Treat the released implementation as a future selective-update baseline. A valid project result requires the same teacher, frame manifest, byte budget, and QA evaluation as competing writers.

## References

- Jiacheng Dong, Huan Li, Sicheng Zhou, Wenhao Hu, Weili Xu, and Yan Wang. [MeMix: Writing Less, Remembering More for Streaming 3D Reconstruction](https://arxiv.org/abs/2603.15330). 2026.
- [Official project page](https://dongjiacheng06.github.io/MeMix/).
- [Official repository](https://github.com/dongjiacheng06/MeMix).

[Back to paper index](README.md)
