# Spann3R: 3D Reconstruction with Spatial Memory

| Field | Value |
|---|---|
| Page ID | SM-PAPER-SPANN3R |
| Status | reviewed; code available |
| Publication | IEEE 3DV 2025, pp. 78–89; arXiv:2408.16061 v1 |
| Primary source | [DOI](https://doi.org/10.1109/3DV66043.2025.00013) · [arXiv](https://arxiv.org/abs/2408.16061) · [Project page](https://hengyiwang.github.io/projects/spanner) |
| Official code | [HengyiWang/spann3r](https://github.com/HengyiWang/spann3r) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-002 |

## 30-second summary

Spann3R incrementally predicts globally aligned pointmaps while managing recent dense working memory and sparse long-term feature memory. It supplies an early, concrete baseline for “recent detail plus selected history.” Its attention-ranked generic tokens support reconstruction, not explicit geometry-grounded QA records.

## Problem addressed

Pairwise DUSt3R pointmaps require global alignment before a complete scene is available. Spann3R instead predicts each frame in one global coordinate system and retrieves prior 3D information online, without camera inputs or test-time global alignment.

## Relevant method

- Keep dense working memory for the latest five frames.
- Move older information into sparse long-term memory.
- Insert a working-memory token only when similarity is below a threshold.
- Rank long-term tokens by accumulated retrieval attention and retain the top-k after a threshold is reached.
- Query both memories to predict the next globally aligned pointmap.

## Paper-reported evidence

**Reported claim.** The memory ablation reports Acc/Comp/NC of `0.2554/0.1470/0.5964` with working memory only and `0.0342/0.0241/0.6635` with full memory. The paper reports 4,000 long-term tokens as sufficient for most tested scenes and approximately 65 FPS with about 11 GB VRAM on one RTX 4090.

**Project inference.** A recent dense buffer plus a selected historical reservoir is a useful baseline for transient geometry, provided retrieval attention is not mislabeled as QA utility.

**Project result.** None. This repository has not reproduced Spann3R.

## What this supports here

- Separating detailed short-term geometry from sparse long-term context.
- Testing attention-based retention against geometry novelty and future-QA utility.
- Measuring memory and reconstruction failure across room transitions and longer sequences.

## What it does not prove

- Explicit object, plane, portal, event, coordinate-frame, or provenance storage.
- That reconstruction attention predicts future QA value or serialized value per byte.
- Robust loop closure or multi-room lifelong mapping; the paper reports drift-related limitations.
- SuperMemory-VQA improvement, 1 Hz wearable performance, or AI-glass feasibility.
- The repository's later v1.01 checkpoint results as 3DV paper results; those are post-paper release notes.

## Project reproduction status

Not reproduced. The released code makes it a practical transient-memory baseline. Any reproduction must distinguish the published model from the repository's later v1.01 checkpoint and training changes.

## References

- Hengyi Wang and Lourdes Agapito. [3D Reconstruction with Spatial Memory](https://doi.org/10.1109/3DV66043.2025.00013). IEEE 3DV 2025, pp. 78–89.
- [Official arXiv record](https://arxiv.org/abs/2408.16061).
- [Official project page](https://hengyiwang.github.io/projects/spanner).
- [Official repository](https://github.com/HengyiWang/spann3r).

[Back to paper index](README.md)
