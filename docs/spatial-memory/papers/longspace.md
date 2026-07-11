# LongSpace: Exploring Long-Horizon Spatial Memory from Perception to Recall in Video

| Field | Value |
|---|---|
| Page ID | SM-PAPER-LONGSPACE |
| Status | Reviewed; recent preprint; announced code unavailable when checked |
| Publication | arXiv:2606.05677 v1, 2026-06-04 |
| Primary source | [arXiv](https://arxiv.org/abs/2606.05677) |
| Official code | [Announced repository](https://github.com/ShiqiangLang/LongSpace), unavailable when checked |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-008 |

## 30-second summary

LongSpace introduces a long room-tour benchmark and a model that injects 3D structure into early decoder layers while maintaining hierarchical KV memory across video chunks. Its long-memory and layer-aware ablations show that organized cross-chunk evidence matters more than recent-frame sampling. Its memory remains latent KV state rather than an auditable typed geometry database.

## Problem addressed

Long-video models must retain layouts, routes, viewpoint changes, and object states across observations separated by many minutes. Uniform frame sampling and recent windows lose distant spatial evidence or dilute it with redundant tokens.

## Relevant method

- Build LongSpace-Bench from continuous real-world room-tour videos.
- Process video at 1 fps in 32-frame chunks with four-frame overlap.
- Align 3D geometry features and inject a structural residual into the first eight decoder layers.
- Divide hierarchical KV memory into sensory, working, and long-memory roles.
- Select and compress memory using salience, state change, recency, temporal coverage, and role-specific budgets.
- Retrieve segment-level and then token-level evidence for each question.

## Paper-reported evidence

| Dataset | Condition | Metric | Reported result | Location |
|---|---|---|---|---|
| LongSpace-Bench | Dataset | Scale | 445 videos; about 159 hours; 4,073 QA pairs; 21.4-minute average | Table 1 and Section 3.2 |
| LongSpace-Bench | LongSpace | Overall | 49.2 | Table 2 |
| LongSpace-Bench | Uniform 32 frames → recent windows → long memory | Overall | 36.1 → 37.7 → 49.2 | Figure 5 and Section 5 |
| LongSpace-Bench | Layer-agnostic → layer-aware memory | Overall | 41.8 → 49.2 | Table 5 |
| LongSpace-Bench | Long memory over uniform, short / medium / long videos | Improvement | +4.8 / +12.8 / +15.1 points | Figure 5 and Section 5 |

## What this supports here

- At 1 fps, cross-chunk evidence is more useful than retaining only recent frames.
- Memory organization and retrieval deserve separate ablation from raw capacity.
- Geometry can be injected selectively into early model layers.
- State change and temporal coverage are useful candidate-writer signals.

## What it does not prove

- Explicit object identities, coordinate frames, uncertainty, provenance, or geometry proof.
- Actual serialized-byte optimization; KV capacity is not persistent database size.
- Active exploration, wearable IMU/VIO, multi-day revisits, or on-device execution.
- SuperMemory-VQA performance.

## Project reproduction status

Not reproduced. The paper-linked repository returned 404 when checked, so implementation claims remain paper-only until an official release is available.

## References

- Shiqiang Lang et al. [LongSpace: Exploring Long-Horizon Spatial Memory from Perception to Recall in Video](https://arxiv.org/abs/2606.05677). arXiv:2606.05677 v1.
- [Paper-announced repository](https://github.com/ShiqiangLang/LongSpace), unavailable on 2026-07-11.
- [Back to paper index](README.md).
