# SuperMemory-VQA: An Egocentric Visual Question-Answering Benchmark for Long-Horizon Memory

| Field | Value |
|---|---|
| Page ID | SM-PAPER-SUPERMEMORY-VQA |
| Status | Reviewed from primary sources; project benchmark not run locally |
| Publication | arXiv:2606.00825v1, 2026 |
| Primary source | [Official arXiv record](https://arxiv.org/abs/2606.00825) |
| Official code | [AIoT-MLSys-Lab/supermemory-vqa](https://github.com/AIoT-MLSys-Lab/supermemory-vqa) |
| Last checked | 2026-07-11 |
| Project links | [Paper index](README.md), [Problem](../problem.md), [Architecture](../architecture.md), [Status](../status.md) |
| Project claims | [Traceability](../traceability.md): C-008, C-009 |

## 30-second summary

SuperMemory-VQA evaluates whether an AI-glasses memory assistant can retrieve and
reason over long-horizon, multimodal, egocentric recordings while abstaining when
evidence is insufficient. It contains 52.9 hours from ten participants and 4,853
human-verified four-choice questions across six practical memory tasks. Every
question includes an explicit unanswerable option. The benchmark exposes gaps in
retrieval, temporal integration, exact state tracking, and answerability.

## Problem addressed

Existing egocentric benchmarks mainly test short-clip perception, action
recognition, or generic QA. They do not directly test practical memory questions
spanning sessions, days, modalities, and disjoint evidence moments. SuperMemory-VQA
defines a benchmark for object and location memory, conversational memory, visual
scene recall, in-context retrieval, timeline reconstruction, and intent recall.

## Relevant method

The dataset uses Meta Aria recordings with RGB, audio, eye gaze, IMU, and SLAM
trajectories. Its annotation pipeline creates grounded descriptions, proposes QA
pairs, verifies causality and evidence, and ends with human review. Questions use
ordered choices representing correct, vague, wrong, and unanswerable answers.

The paper evaluates Video-RAG and EgoButler with open and closed VLMs. Both systems
receive only evidence preceding the question end time. The official metrics are:

- Ans-F1: binary answerable-versus-unanswerable F1.
- QA-Acc: exact four-way multiple-choice accuracy.
- QA-MRR: reciprocal rank of the correct option under ordered answer scores.

## Paper-reported evidence

These are paper results, not results from this repository.

| Dataset or condition | Metric | Reported result | Source location |
|---|---|---:|---|
| Full dataset | Duration / participants / QA pairs | 52.9 hours / 10 / 4,853 | Section 3.1 and Table 1, paper pp. 4–5 |
| Full dataset | Questions requiring multiple evidence items | 34% | Table 1, paper p. 5 |
| Gemini-3-Flash with Video-RAG | Ans-F1 / QA-Acc / QA-MRR | 83.9 / 61.0 / 76.0 | Table 2, paper p. 8 |
| Mean over ten evaluated VLMs, Video-RAG versus EgoButler | QA-Acc | 46.6 versus 41.4 | Section 5.1, paper p. 9 |
| Qwen3-8B, text-only, Person 1, 1,017 questions | QA-Acc | 23.8%, versus 25% chance | Table 3, paper p. 11 |

The strongest reported Video-RAG configuration reaches only 61.0 QA-Acc despite
83.9 Ans-F1. The paper interprets this as evidence that answerability detection is
necessary but insufficient: precise evidence retrieval and grounded reasoning
remain difficult.

## What this supports here

**Paper claim:** long-horizon egocentric memory requires causal retrieval,
multi-evidence reasoning, and explicit abstention evaluation.

**Project inference:** persistent memory should retain exact object state, temporal
validity, counts, spatial evidence, and provenance rather than generic summaries
alone. The repository therefore treats four-way QA-Acc and QA-MRR as primary
benchmark outputs and rejects evidence after the question cutoff.

The dataset's RGB, IMU, trajectory, and SLAM streams also justify evaluating an
explicit spatial-memory branch rather than restricting the system to captions.

## What it does not prove

- It does not show that explicit typed geometry memory improves benchmark scores.
- It does not validate G-CUT3R, CUT3R, or any particular geometry encoder.
- Its reported baselines do not exhaust gaze, trajectory, IMU, or SLAM inputs.
- It does not establish an on-device storage, latency, or power target.
- Benchmark accuracy alone does not prove metric geometry correctness; separate
  proof, uncertainty, and provenance checks remain necessary.

## Project reproduction status

The repository implements the prepared-data contract, causal preflight checks,
four-choice metrics, evidence validation, and tiny synthetic smoke tests. No
official SuperMemory-VQA dataset was copied locally, and no official benchmark run
was performed on this host. Current local mock scores are sanity checks and must
not be reported as paper reproduction results.

## References

- [Official arXiv record and paper](https://arxiv.org/abs/2606.00825)
- [Official code](https://github.com/AIoT-MLSys-Lab/supermemory-vqa)
- [Official dataset](https://huggingface.co/datasets/OSU-AIoT-MLSys-Lab/SuperMemory-VQA)

[Back to paper index](README.md)
