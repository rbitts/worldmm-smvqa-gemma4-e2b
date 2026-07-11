# Advancing 3D Scene Understanding with MV-ScanQA Multi-View Reasoning Evaluation and TripAlign Pre-training Dataset

| Field | Value |
|---|---|
| Page ID | SM-PAPER-MV-SCANQA |
| Status | Reviewed; code and dataset available |
| Publication | ACM Multimedia 2025, pp. 12973–12980; DOI 10.1145/3746027.3758244 |
| Primary source | [ACM DOI](https://doi.org/10.1145/3746027.3758244) · [arXiv](https://arxiv.org/abs/2508.11058) |
| Official code | [matthewdm0816/MVScanQA](https://github.com/matthewdm0816/MVScanQA) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-008 |

## 30-second summary

MV-ScanQA composes questions so their relevant objects cannot usually be covered by one view. Its analysis and LEGO baseline show measurable gains from retaining complementary viewpoints and aligning groups of visible objects. It is a static ScanNet multi-view benchmark, not a temporal or lifelong memory benchmark.

## Problem addressed

Most 3D vision-language benchmarks can be solved from a single favorable view, so they do not test integration across viewpoints. MV-ScanQA creates compositional questions whose relevant objects require multiple views and supplies TripAlign for view-dependent multi-object alignment.

## Relevant method

- Define an object as witnessed when its projected image overlap satisfies IoSA greater than 0.5.
- Pair source questions that share an object anchor while each contributes non-subset object information.
- Use an LLM to compose a verifiable question from the pair.
- Build TripAlign with one million `<2D view, visible 3D-object set, text>` triplets.
- Build LEGO on a 2D vision-language model plus a 3D detector and discard object proposals not visible in each selected view.

## Paper-reported evidence

| Dataset | Condition | Metric | Reported result | Location |
|---|---|---|---|---|
| ScanQA / ScanRefer / Nr3D | Questions requiring more than one view | Share | 6% / 4% / 7% | Section 3.1 |
| MV-ScanQA | Questions requiring at least two / at least three views | Share | 68% / 13% | Section 3 |
| MV-ScanQA | LEGO single-view → four-view input | Exact match, all | 30.0 → 34.1 | Table 2 |
| MV-ScanQA | LEGO single-view → four-view input, N≥4 | Exact match | 23.3 → 30.2 | Table 2 |
| ScanQA | From scratch → egocentric extension → TripAlign | Exact match | 25.13 → 27.22 → 28.43 | Table 6 |

The accepted paper reports 34.1 exact match for multi-view LEGO. The current official repository separately reports 33.7 from a cleaned evaluation script. These are distinct provenance records and must not be merged.

## What this supports here

- View coverage and ray diversity belong in evidence selection.
- Complementary views should not be discarded as spatial duplicates.
- Entity-group alignment is a relevant training target for multi-object questions.
- The project infers that ray-aware landmarks and provenance should retain which views support each entity.

## What it does not prove

- Temporal ordering, object persistence, or causal stream processing.
- 1 Hz monocular geometry, IMU/VIO guidance, or lifelong revisit behavior.
- Actual-byte compression or fixed-capacity persistent memory.
- SuperMemory-VQA transfer.

## Project reproduction status

Not reproduced. A future comparison must pin both the paper metric and the official repository evaluation script because their reported exact-match values differ.

## References

- Wentao Mo et al. [Advancing 3D Scene Understanding with MV-ScanQA Multi-View Reasoning Evaluation and TripAlign Pre-training Dataset](https://doi.org/10.1145/3746027.3758244). ACM Multimedia 2025.
- [Official arXiv record](https://arxiv.org/abs/2508.11058).
- [Official project and datasets](https://matthewdm0816.github.io/tripalign-mvscanqa/).
- [Official repository](https://github.com/matthewdm0816/MVScanQA).
- [Back to paper index](README.md).
