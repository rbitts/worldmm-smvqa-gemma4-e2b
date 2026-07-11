# Worth Remembering: Surprise-Gated Robot Episodic Memory

| Field | Value |
|---|---|
| Page ID | SM-PAPER-SURPRISE-EPISODIC-MEMORY |
| Status | Reviewed; recent preprint; official code not published |
| Publication | arXiv:2606.03787 v3, 2026-06-06 |
| Primary source | [Version-pinned arXiv v3](https://arxiv.org/abs/2606.03787v3) |
| Official code | Not published or verified |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-004 |

## 30-second summary

Worth Remembering uses prediction surprise in V-JEPA-2 latent space to select sparse visual episodes without knowing future questions. At an equal episode budget it beats uniform and random writes on long-horizon robot QA. This supports a small surprise-gated evidence reservoir, not replacing the stable geometry core.

## Problem addressed

A robot cannot retain every frame, but future tasks are unknown. Fixed-rate or task-specific writes can miss brief important events and retain long stretches of redundant observations. The paper seeks a causal, unsupervised signal for deciding what event deserves episodic storage.

## Relevant method

- Embed a causal sliding window of 64 recent frames with V-JEPA-2 and pool to a 1,024-dimensional latent.
- Fit a diagonal Gaussian over the preceding latent window and compute a robust normalized prediction-error score.
- Trigger on local maxima above `median + 1.4826 × MAD`, then apply non-maximum suppression.
- Store timestamp, robot pose, surprise score, an eight-frame episode, and a retrieval embedding.
- Add selected episodes as a visual layer above DAAAM's 4D scene graph.

## Paper-reported evidence

| Dataset | Condition | Metric | Reported result | Location |
|---|---|---|---|---|
| OC-NaVQA | DAAAM → surprise episodic memory | QA accuracy | 0.711 → 0.796 | Tables 1–2 |
| OC-NaVQA | DAAAM → surprise episodic memory | Position error | 41.75 m → 36.57 m | Tables 1–2 |
| OC-NaVQA | DAAAM → surprise episodic memory | Temporal error | 1.792 min → 1.510 min | Tables 1–2 |
| OC-NaVQA | Surprise gate | Retention rate | 1.28 episodes/min; about 1.7% of frames | Section 4.1 |
| OC-NaVQA | Surprise gate | Reasoning-token cost | About 13% increase | Section 4.1 |
| Kinetics-GEBD | Online unsupervised surprise gate | Mean F1 | 0.833 | Table 3 |

Uniform and random episodic-memory baselines receive the same episode budget in the QA ablation. The paper reports relative improvements of 12.0% in accuracy, 12.4% in position error, and 15.7% in temporal error over DAAAM.

## What this supports here

- Surprise is a plausible query-agnostic write signal for rare events.
- A small visual evidence reservoir can complement structured spatial memory.
- Pose and timestamp should accompany stored episodes.
- Equal-budget comparison against uniform and random selection is required.

## What it does not prove

- Surprise can replace static geometry coverage; important walls, doors, and free space may be unsurprising.
- Visual episodes are byte-efficient compared with typed records.
- Robustness to repeated surprising subjects; the paper identifies missing habituation as a limitation.
- Exact 1 Hz sensing, on-device execution, or SuperMemory-VQA performance.

## Project reproduction status

Not reproduced. The current repository has a surprise feature in selector training, but no V-JEPA-2 gate or visual episodic reservoir. No production model was downloaded locally.

## References

- Nicolas Gorlo, Derek K. Wise, Alberto Speranzon, and Luca Carlone. [Worth Remembering: Surprise-Gated Robot Episodic Memory](https://arxiv.org/abs/2606.03787v3). arXiv:2606.03787 v3.
- [Back to paper index](README.md).
