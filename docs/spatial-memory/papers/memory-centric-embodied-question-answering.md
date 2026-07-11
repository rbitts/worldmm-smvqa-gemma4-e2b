# Memory-Centric Embodied Question Answering

| Field | Value |
|---|---|
| Page ID | SM-PAPER-MEMORYEQA |
| Status | Reviewed; preprint; code available |
| Publication | arXiv:2505.13948 v2, 2025-12-13; peer-reviewed venue not verified |
| Primary source | [Version-pinned arXiv v2](https://arxiv.org/abs/2505.13948v2) |
| Official code | [memory-eqa/MemoryEQA](https://github.com/memory-eqa/MemoryEQA) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-002, C-004, C-008 |

## 30-second summary

MemoryEQA makes memory available to planning, stopping, and answering instead of consulting it only at answer time. Its ablations show that selective retrieval and adaptive retrieval size contribute more than update gating alone. This supports memory-aware control and retrieval, but its structured text and dense-vector library are not explicit metric geometry or actual-byte compression.

## Problem addressed

Planner-centric embodied-QA systems can explore redundantly or stop too early because their memory is used only by the answer module. Multi-target questions spanning rooms require the agent to retain observations, retrieve the right subset, and expose that subset to every decision module.

## Relevant method

- Convert observations into structured text and multimodal features stored in a vector library.
- Persist scene memory so later tasks in the same environment can reuse it.
- Gate updates using position, orientation, structural and semantic similarity, and field-of-view checks.
- Retrieve different memory for planner, stopping, and answering queries.
- Adapt retrieval thresholds and top-k size using query-feature entropy.
- Build MT-HM3D with comparison, relationship, counting, and attribute questions over multiple objects and regions.

## Paper-reported evidence

All values below are pinned to arXiv v2. Other official project surfaces contain stale, conflicting values and are not used as paper results.

| Dataset | Condition | Metric | Reported result | Location |
|---|---|---|---|---|
| MT-HM3D | Dataset | Scale | 1,587 QA samples over 500 scenes | Section 4 |
| MT-HM3D | GPT-4o ExploreEQA → MemoryEQA | Success | 33.21 → 43.11, +9.90 percentage points | Table 2 |
| MT-HM3D | No strategy → update → update + retrieval → all + adaptive k | Success | 33.18 → 33.41 → 39.69 → 41.95 | Table 3 |
| MT-HM3D | No module injection → stop → stop + answer → stop + answer + planner | Success | 30.22 → 35.10 → 40.99 → 41.95 | Table 4 |
| HM-EQA | GPT-4o ExploreEQA → MemoryEQA | Success | 47.40 → 61.40 | Table 2 |

## What this supports here

- Memory should inform retrieval, stopping, and planning, not only final answering.
- Redundant observations require an explicit update decision.
- Selective, module-specific retrieval can matter more than adding more memory.
- Adaptive retrieval size is a relevant baseline for a value-per-byte writer.

## What it does not prove

- Explicit typed metric records or deterministic geometry proof.
- Actual serialized-byte optimization or bounded lifelong growth.
- 1 Hz sparse monocular sensing, IMU/VIO guidance, or AI-glasses execution.
- That entropy is the best selector for future geometry-grounded questions.
- SuperMemory-VQA performance.

## Project reproduction status

Not reproduced. Any future comparison must pin arXiv v2 and its tables because the official project and repository have reported conflicting headline numbers.

## References

- Mingliang Zhai, Zhi Gao, Yuwei Wu, and Yunde Jia. [Memory-Centric Embodied Question Answering](https://arxiv.org/abs/2505.13948v2). arXiv:2505.13948 v2.
- [Official project page](https://memory-eqa.github.io/).
- [Official repository](https://github.com/memory-eqa/MemoryEQA).
- [Official MT-HM3D dataset](https://huggingface.co/datasets/zmling/MT-HM3D).
- [Back to paper index](README.md).
