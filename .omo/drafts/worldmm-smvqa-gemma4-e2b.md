---
slug: worldmm-smvqa-gemma4-e2b
status: drafting
intent: clear
pending-action: write .omo/plans/worldmm-smvqa-gemma4-e2b.md
approach: Create a decision-complete implementation plan for a WorldMM-SMVQA baseline: ingest SuperMemory-VQA source streams, chunk them at 30 seconds and 30 minutes, build caption/OCR/object/frame memories, structure them as WorldMM episodic/semantic/visual memories, retrieve only pre-question evidence, run Gemma 4 E2B QA, and report Ans-F1 / QA-Acc / QA-MRR. Local host work is limited to code, configs, tiny fixtures, and dry-runs; real dataset/model download and evaluation run remotely through the bastion/head node per AGENTS.md.
---

# Draft: worldmm-smvqa-gemma4-e2b

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
| C1 | Local scaffold, config, CLI, tiny fixtures, and tests | active | `AGENTS.md` local host rules |
| C2 | SuperMemory-VQA source-stream ingestion and label-separated schema | active | SuperMemory-VQA paper/dataset card |
| C3 | 30s clip and 30min shard chunking with causal question-time cutoff | active | SuperMemory-VQA implementation details |
| C4 | Caption/OCR/object/frame memory builders from allowed source data only | active | Video-RAG/EgoButler baseline descriptions |
| C5 | WorldMM episodic/semantic/visual memory stores and adaptive retrieval | active | WorldMM paper/project page |
| C6 | Gemma 4 E2B QA runner over retrieved evidence packs | active | Gemma 4 model card |
| C7 | Official metrics, anti-leakage checks, reports, and remote launch workflow | active | SuperMemory-VQA metrics + `AGENTS.md` remote rules |

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
<!-- assumption | adopted default | rationale | reversible? -->
| project shape | New Python project in this empty repo using `uv`, `pytest`, `ruff`, `mypy`, `pydantic`, `typer`, `datasets`, `transformers`, optional `faiss-cpu` locally and GPU FAISS remotely | Empty repo; Python is the official repo/runtime path; minimal standard tooling | yes |
| local vs remote | Local executes only tiny fixture tests and dry-runs; remote executes dataset/model download, memory build, real QA, and metrics | Required by `AGENTS.md`; avoids accidental large artifacts locally | no |
| model | `google/gemma-4-E2B-it` via Transformers/AutoProcessor/AutoModelForMultimodalLM unless remote runtime requires equivalent pinned checkpoint mirror | User asked Gemma 4 2B QA; official Gemma docs list E2B | yes |
| baseline name | Report as `WorldMM-SMVQA` / `WorldMM-augmented Video-RAG/EgoButler`, not exact paper baseline reproduction | Same protocol, different memory architecture | yes |
| label handling | QA labels and `answer.evidence_list` are evaluator-only; memory builders cannot import or access them | Prevents label leakage | no |
| test strategy | TDD for schema/metrics/leakage/cutoff/retrieval; tests-after for remote launch scripts because they are wrapper commands | High-risk logic gets failing-first proof | yes |

## Findings (cited - path:lines)
- Project `AGENTS.md` says this host is for code/config/tiny tests only; production model/dataset download, real evaluation, training, checkpointing, and large artifacts must run only on company resources through bastion/head node. Local final deliverables must report remote command, job/process reference, remote artifact path, metrics/failure, and what was not copied locally.
- SuperMemory-VQA contains 52.9 hours, 4,853 grounded QA pairs, RGB video, audio transcription, eye gaze, IMU, and SLAM trajectories; each question is multiple choice with explicit unanswerable option. Source: https://arxiv.org/html/2606.00825v1 lines 66-70.
- SuperMemory-VQA baseline setup evaluates Video-RAG and EgoButler; Video-RAG precomputes ASR/OCR/object-detection databases, queries FAISS, and sends retrieved text plus frames to the VLM. Source: https://arxiv.org/html/2606.00825v1 lines 194-198.
- SuperMemory-VQA implementation details: open-source models on 4xA100, greedy decoding for answer generation, Video-RAG 30-minute shards, EgoButler 30-second captions plus hour/day summaries, and causal cutoff at question end time. Source: https://arxiv.org/html/2606.00825v1 lines 200-206.
- SuperMemory-VQA metrics are Ans-F1, QA-Acc, and QA-MRR. Source: https://arxiv.org/html/2606.00825v1 lines 207-209.
- WorldMM constructs three complementary memories: episodic multi-scale textual event graphs, semantic knowledge graph, and visual feature/frame memory; adaptive retrieval chooses memory source and query iteratively. Source: https://worldmm.github.io/ lines 20-33.
- Gemma 4 E2B is an effective 2.3B model, supports text/image/audio, 128K context, and is intended for multimodal reasoning workflows. Source: https://ai.google.dev/gemma/docs/core/model_card_4 lines 293-320.

## Decisions (with rationale)
- The requested baseline will be planned as `WorldMM-SMVQA`: a protocol-compliant SuperMemory-VQA baseline with WorldMM-style memory construction, not an exact reproduction of the paper's reported Video-RAG/EgoButler numbers.
- The implementation must maintain a strict two-surface dataset API: `SourceStreamExample` for memory builders and `QALabelExample` only for evaluators.
- The first runnable milestone is tiny local fixtures proving chunking, anti-leakage, retrieval, prompt packing, and metrics. Full benchmark numbers require a remote job.
- The plan will include exact local commands and remote launch templates, but will not execute remote jobs because `AGENTS.md` requires asking before expensive/multi-node runs.
- The plan will include ablations: episodic only, semantic only, visual only, E+S, E+V, S+V, E+S+V.

## Scope IN
- New Python package and CLI for `prepare-data`, `build-memory`, `retrieve`, `qa`, `evaluate`, `report`, `smoke`, and `launch-remote`.
- Source-stream schema separating allowed memory inputs from eval-only labels.
- 30-second clip chunker and 30-minute shard chunker.
- Caption/OCR/object/frame memory construction interfaces with fixture implementations and remote hooks for expensive generators.
- WorldMM episodic event graph, semantic relation/habit graph, and visual memory store.
- Adaptive retrieval constrained by question-time causal cutoff.
- Gemma 4 E2B QA prompt/decoder producing answerability, ranked choices, final answer, confidence, supporting memory IDs.
- Official metrics plus memory diagnostics: Recall@K against eval-only evidence, causal violation count, prompt tokens, memory size.
- Local tiny fixture tests and remote-only full benchmark plan.

## Scope OUT (Must NOT have)
- No local production model download.
- No local full SuperMemory-VQA download or real evaluation.
- No use of QA labels, `answer.evidence_list`, `is_answerable`, choice type, or verification score inside memory builders/retrievers.
- No fine-tuning, LoRA, SD-QST training, or claim of trained memory tokenizer.
- No copying full dataset, model weights, checkpoints, or sensitive artifacts back to local host.
- No hardcoded company paths; remote storage and bastion values come from env/config.

## Open questions
- None blocking. Default remote command templates will use environment variables for bastion/head node and approved storage paths.

## Approval gate
status: approved-by-latest-user-scope-change
pending action: write `.omo/plans/worldmm-smvqa-gemma4-e2b.md`
approach summary: write a detailed implementation plan for WorldMM-SMVQA baseline with local tiny verification, remote full benchmark execution, anti-leakage gates, and official SuperMemory-VQA metrics.
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
