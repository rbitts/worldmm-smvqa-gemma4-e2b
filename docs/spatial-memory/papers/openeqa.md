# OpenEQA: Embodied Question Answering in the Era of Foundation Models

| Field | Value |
|---|---|
| Page ID | SM-PAPER-OPENEQA |
| Status | Reviewed; code and benchmark available |
| Publication | CVPR 2024, pp. 16488–16498 |
| Primary source | [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2024/html/Majumdar_OpenEQA_Embodied_Question_Answering_in_the_Era_of_Foundation_Models_CVPR_2024_paper.html) |
| Official code | [facebookresearch/open-eqa](https://github.com/facebookresearch/open-eqa) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-003, C-008, C-009 |

## 30-second summary

OpenEQA defines open-vocabulary embodied QA for both episodic-memory histories and active exploration. Its smart-glasses setting and poor spatial performance of strong foundation-model baselines make it direct evidence that retaining visual history is not enough: the memory must preserve queryable spatial structure. It is a benchmark paper, not a spatial-memory compression method.

## Problem addressed

An embodied agent must understand a real environment well enough to answer natural-language questions. In EM-EQA it receives a historical observation sequence, analogous to a smart-glasses memory. In A-EQA it must gather evidence through active exploration. Open-vocabulary answers require an evaluation method that tolerates equivalent wording.

## Relevant method

- Collect human-written questions across seven embodied-QA categories from real-world video tours and scans.
- Evaluate blind language models, caption-based agents, scene-graph agents, and multi-frame vision-language models.
- Score open answers with LLM-Match: an evaluator assigns 1–5 and the paper normalizes the aggregate to 0–100.
- Validate LLM-Match against independent human judgements.

## Paper-reported evidence

| Dataset | Condition | Metric | Reported result | Location |
|---|---|---|---|---|
| OpenEQA | Dataset | Questions and environments | 1,636 questions; more than 180 environments; seven categories | Figure 2 and Section 2.3 |
| EM-EQA | Multi-frame GPT-4V, 500-question subset | LLM-Match | 49.6 ± 2.0 | Table 2 |
| EM-EQA | Human | LLM-Match | 86.8 ± 0.6 | Table 2 |
| EM-EQA | Blind GPT-4 | LLM-Match | 33.5 ± 1.0 | Table 2 |
| LLM-Match validation | 300 sampled questions | Spearman correlation with human scoring | 0.909; bootstrap CI 0.883–0.928 | Section 5 |

The paper also reports that object localization and spatial understanding were among the hardest categories. Scene-graph agents did not outperform frame-caption agents on spatial questions.

## What this supports here

- A historical observation sequence is a valid smart-glasses QA setting.
- Long visual context can contain irrelevant evidence and still fail spatial reasoning.
- Spatial evaluation must be separated by question type rather than hidden in one aggregate score.
- A geometry executor and grounded proof are project responses to the spatial gap; OpenEQA does not prescribe them.

## What it does not prove

- Typed object, plane, portal, landmark, or event records.
- Metric geometry proofs or deterministic spatial execution.
- A byte-budgeted writer, 1 Hz sparse sensing, lifelong storage, or on-device execution.
- Performance on SuperMemory-VQA.

## Project reproduction status

Not reproduced. This repository targets SuperMemory-VQA and uses OpenEQA as an external task-design and evaluation reference. No OpenEQA data or model artifacts were downloaded locally.

## References

- Arjun Majumdar et al. [OpenEQA: Embodied Question Answering in the Era of Foundation Models](https://openaccess.thecvf.com/content/CVPR2024/html/Majumdar_OpenEQA_Embodied_Question_Answering_in_the_Era_of_Foundation_Models_CVPR_2024_paper.html). CVPR 2024.
- [Official project page](https://open-eqa.github.io/).
- [Official benchmark, code, and evaluator](https://github.com/facebookresearch/open-eqa).
- [Back to paper index](README.md).
