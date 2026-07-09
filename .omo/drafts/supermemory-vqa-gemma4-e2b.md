---
slug: supermemory-vqa-gemma4-e2b
status: drafting
intent: clear
pending-action: write .omo/plans/supermemory-vqa-gemma4-e2b.md
approach: Build a fresh Python benchmark harness in this empty repo that can download/load SuperMemory-VQA, build official-style causal memory/evidence indices without using answer labels as retrieval input, run a Video-RAG-style baseline with Gemma 4 E2B IT, emit per-question JSONL predictions, and compute official Ans-F1 / QA-Acc / QA-MRR metrics.
---

# Draft: supermemory-vqa-gemma4-e2b

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
| C1 | Repository scaffold and reproducible CLI for dataset/eval/model runs | active | `/home/rbits/workspace/lab` empty; `find . -maxdepth 2` showed only `.` |
| C2 | SuperMemory-VQA data loader with schema validation, local cache, subset/full run modes | active | HF dataset card: 4,853 rows, JSON format, 703 GB |
| C3 | Official benchmark semantics: ordered choices, causal cutoff, Ans-F1 / QA-Acc / QA-MRR | active | arXiv 2606.00825 §4.2-4.3 and Appendix D.2 |
| C4 | Gemma 4 E2B IT QA stage: encode retrieved memory/evidence into the QA prompt, decode deterministic ranked choices | active | Google Gemma 4 docs + HF `google/gemma-4-E2B-it` |
| C5 | Official-style Video-RAG/EgoButler memory retrieval/evidence builders before QA, sufficient to reproduce SuperMemory-VQA baseline semantics | active | arXiv §4.1-4.2 implementation details |
| C6 | WorldMM-augmented memory baseline: build Video-RAG/EgoButler-compatible memory stores that include WorldMM episodic, semantic, and visual memories, evaluated under the same SuperMemory-VQA causal/metric protocol | active | WorldMM paper/project page |
| C7 | Memory creation ladder: v0 official retrieval memory, v1 Super Ledger-compatible memory, v2 static/dynamic structured memory, v3 semantic-geometry binding | active | User gap question; shared ChatGPT SD-QST spec |
| C8 | Reporting, artifacts, QA, and final verification wave | active | ulw-plan full workflow verification strategy |

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
<!-- assumption | adopted default | rationale | reversible? -->
| runtime | Python 3.10+, `uv`, Hugging Face `datasets` + `transformers`, model `google/gemma-4-E2B-it` | Official repo is Python 3.10+; HF model card exposes Transformers load path; simplest path to measurable benchmark | yes |
| benchmark scope | Implement both `--limit N` smoke/subset and `--split test --full` full benchmark; final metric command must work on full data when local storage/GPU/model access exist | Dataset is 703 GB; worker needs smoke proof without requiring full download every test | yes |
| baseline target | First numeric target is official-style zero-shot evaluation, not SD-QST training | User asked for Gemma 4 2B QA metrics; training would delay first valid numbers | yes |
| retrieval baseline | Video-RAG-style text/evidence baseline first: build memory from allowed pre-question sources such as transcripts, generated captions, OCR/object detections, frames, and metadata; never from ground-truth answer evidence labels | Paper baseline uses 30-minute shards, FAISS, 32 sampled frames; label leakage would invalidate baseline numbers | yes |
| memory creation depth | Plan must explicitly split memory creation into v0/v1/v2/v3; v0 official-style memory construction/retrieval is required for first benchmark numbers, v1-v3 close SD-QST-specific memory-generation gap | User asked about memory-making gap; avoids pretending retrieval index equals structured SD-QST memory | yes |
| benchmark flow | Retrieve causal memory/evidence first, then run Gemma 4 E2B QA encoding/decoding over that retrieved context | This matches the official baseline pattern: memory/evidence construction precedes answer generation and ranking | yes |
| WorldMM role | Add WorldMM as a protocol-compliant, WorldMM-augmented memory-construction baseline; report it separately from exact Video-RAG/EgoButler reproduction | WorldMM builds episodic, semantic, and visual memories from source streams; this follows SuperMemory-VQA evaluation protocol if labels are excluded, but it is not identical to the paper's reported baseline implementation | yes |
| model decoding | Greedy or deterministic evaluation mode with strict JSON repair/retry bounded to one retry | Paper uses greedy decoding for answer-generation; metrics require reproducible ranks | yes |
| privacy/license | Treat dataset as research-only CC BY-NC-SA 4.0; never add code that identifies participants or bypasses redactions | HF license/privacy statements | no for compliance |

## Findings (cited - path:lines)
- Repo state: `/home/rbits/workspace/lab` is empty; no existing benchmark code or tests to preserve.
- Shared ChatGPT spec: first implementation priority is v0 alignment: dataset loader, causal cutoff, four-choice scoring, Gemma decoding, Video-RAG-style baseline reproduction.
- arXiv paper: SuperMemory-VQA has 52.9 hours and 4,853 grounded QA pairs, with RGB video, audio transcription, eye gaze, IMU, and SLAM trajectories. Source: https://arxiv.org/html/2606.00825v1 lines 66-70.
- arXiv paper: Video-RAG baseline uses per-session 30-minute shards, FAISS indices, preceding shards only, merged top texts, and 32 uniformly sampled frames from the most relevant shard. Source: https://arxiv.org/html/2606.00825v1 lines 200-206.
- arXiv paper: official metrics are Ans-F1, QA-Acc, QA-MRR; Ans-F1 is answerable/unanswerable F1, QA-Acc is four-way accuracy, QA-MRR ranks answer choices. Source: https://arxiv.org/html/2606.00825v1 lines 207-209.
- arXiv Appendix D.2: choices are Correct, Vague, Wrong, N/A; evidence always occurs before question starts. Source: https://arxiv.org/html/2606.00825v1 lines 600-620.
- HF dataset card: dataset is JSON, English, Visual Question Answering, 4,853 rows, 703 GB, CC BY-NC-SA 4.0. Source: https://huggingface.co/datasets/OSU-AIoT-MLSys-Lab/SuperMemory-VQA lines 55-71 and 2055-2063.
- HF dataset card: primary setting is zero-shot evaluation on released QA labels; fine-tuned/optimized use must be reported separately. Source: https://huggingface.co/datasets/OSU-AIoT-MLSys-Lab/SuperMemory-VQA lines 2009-2017.
- Gemma docs: Gemma 4 small sizes include E2B/E4B effective parameter models, E2B supports text/image/audio, small models have 128K context. Source: https://ai.google.dev/gemma/docs/core lines 291-305 and model card lines 313-320.
- HF model search: `google/gemma-4-E2B-it` exposes Transformers usage with `AutoProcessor` and `AutoModelForMultimodalLM`.
- Memory generation gap: a causal retrieval index is enough to run Gemma 4 E2B and report official QA metrics, but it is not the same as SD-QST memory creation. Missing pieces are raw multimodal stream alignment, dense caption/Super Ledger generation, modality-specific causal indexing, static/dynamic structured tokens, semantic-geometry binding, and memory-quality metrics such as evidence Recall@K and spatial relation accuracy.
- Benchmark flow correction: SuperMemory-VQA itself is a dataset/evaluation protocol. The evaluated systems first build/retrieve memory evidence under causal cutoff, then the VLM/Gemma stage encodes the question, answer choices, and retrieved evidence, and decodes scores/ranking for the answer choices.
- Baseline correction: official-style baseline must include memory/evidence construction before QA. Ground-truth `answer.evidence_list`, correct/vague/wrong labels, and final answer are evaluation labels only; using them to build retrieval memory is label leakage and must fail QA.
- WorldMM fact: WorldMM constructs complementary multimodal memories: episodic memory over multi-scale factual events, semantic memory as continuously updated high-level conceptual knowledge, and visual memory preserving detailed scene information. Source: https://worldmm.github.io/ and https://arxiv.org/abs/2512.02425.
- Planning correction: WorldMM can be a strong protocol-compliant baseline for SuperMemory-VQA if its memory construction consumes only allowed pre-question inputs and its retrieval output is fed to the same Gemma 4 E2B QA/evaluator path. It should be reported separately from exact Video-RAG/EgoButler reproduction because its memory architecture is different from the paper's reported baselines.
- Baseline input clarification: Video-RAG/EgoButler-style baselines do build memory from the dataset recordings and allowed derived modalities, but not from the QA labels. In practice this means preprocessing each session/video into transcripts, captions, OCR/object detections, frame references, shards, summaries, and indices before any question is answered.
- Hybrid baseline clarification: a valid plan can build Video-RAG-style shards and EgoButler-style hierarchical summaries while enriching those memory banks with WorldMM's episodic memory, semantic memory, and visual memory. This is best named `WorldMM-SMVQA` or `WorldMM-augmented Video-RAG/EgoButler`, not "exact official baseline reproduction."

## Decisions (with rationale)
- Plan will create a new Python project because the current repo is empty.
- Plan will not build SD-QST training first; first done state is "runnable official-style benchmark numbers with Gemma 4 E2B IT."
- Plan will include a memory ladder: v0 retrieval memory for first metrics, v1 Super Ledger-compatible memory schema, v2 static/dynamic structured token generation, v3 semantic-geometry binding. v1-v3 may be planned as later waves unless the user explicitly wants them in the first executable milestone.
- Plan will keep retrieval and QA as separate task groups: memory retrieval/indexing produces evidence packs; Gemma QA consumes only those evidence packs and emits answerability + ranked choices.
- Plan will include an explicit anti-leakage gate: retrieval memory/index builders cannot read ground-truth answer labels, `answer.evidence_list`, `is_answerable`, or correct-choice fields except inside evaluator/diagnostics modules.
- Plan will include two baseline lanes: (1) exact reproduction lane: Video-RAG/EgoButler-style memory construction under SuperMemory-VQA protocol; (2) WorldMM-augmented lane: construct Video-RAG/EgoButler-compatible memories enriched with WorldMM episodic/semantic/visual memories from the same allowed source stream and evaluate with the exact same Gemma 4 E2B QA + metrics pipeline.
- Plan will treat `dataset` as two separated surfaces: `source stream` for memory construction and `QA labels` for evaluation only. Any code path that lets memory builders access labels is out of scope and should fail tests.
- Plan will require deterministic, artifact-backed outputs: raw predictions JSONL, metrics JSON, run manifest, model/dataset revisions, and failure logs.
- Plan will make each task runnable by an agent without human intervention, including happy/failure QA commands and exact expected observables.

## Scope IN
- Project scaffold: `pyproject.toml`, package layout, CLI entrypoint, config, typed schemas, tests.
- Dataset loader for HF/local SuperMemory-VQA, schema normalization, cache paths, sample/subset/full modes.
- Causal cutoff validator: no evidence after question time / question start.
- Four-choice parser/ranker supporting Correct/Vague/Wrong/N/A labels and answerable classification.
- Gemma 4 E2B IT runner through Transformers with deterministic decoding, structured JSON output, one bounded repair retry, and resumable JSONL output.
- Video-RAG-style baseline: 30-minute shard metadata, causal retrieval over preceding shards built from allowed memory sources, optional frame selection hooks, top-k evidence packing, 32-frame cap config.
- EgoButler-style baseline parity: 30-second window captions, hour/day summaries, hierarchical memory bank, coarse-to-fine retrieval where generated summaries are available or can be generated.
- WorldMM-augmented baseline: multi-scale episodic event graph, evolving semantic relation graph, visual memory store with frame/embedding references, adaptive retrieval policy, all constrained by question-time causal cutoff and exposed through the same evidence-pack interface as Video-RAG/EgoButler.
- Memory creation v0: official-style causal memory index from allowed released sources: redacted transcripts, generated dense captions when available, OCR/object-detection text if generated, frame metadata, task/session/video/time metadata, and pre-question chunks. Ground-truth answer evidence is excluded from retrieval inputs.
- Memory creation v1: Super Ledger-compatible internal schema that can ingest generated dense captions later without changing evaluator interfaces.
- Memory creation v2: static/dynamic memory-token schema for room/layout/surface/object-anchor/relation/free-space and object-state/last-seen/trajectory/interaction/event/uncertainty.
- Memory creation v3: semantic-geometry binding interface for CLIP/SigLIP/DINO-style semantic features plus SLAM/CUT3R-style geometry primitives; planned as interface + fixture-backed validation first, not full model training.
- Memory diagnostics: retrieval evidence hit rate / Memory Recall@K, causal violation count, token budget, prompt budget, and optional spatial evidence accuracy where annotations support it.
- Metrics: Ans-F1, QA-Acc, QA-MRR with unit tests against hand-calculated examples.
- CLI commands: `prepare-data`, `build-index`, `run-baseline`, `evaluate`, `report`, `smoke`.
- Agent-executed QA: tiny fixture, malformed fixture, smoke subset, CLI help, no-data failure path.
- Documentation: exact setup/run commands for smoke, subset, full benchmark, and expected artifacts.

## Scope OUT (Must NOT have)
- No SD-QST static/dynamic tokenizer training in this plan.
- No claim that v0 retrieval memory is the full SD-QST memory generator.
- No use of ground-truth answer evidence/labels as retrieval memory input.
- No claim that WorldMM-augmented numbers are the paper's exact official baseline table unless the plan also reports implementation deltas and runs the exact reproduction lane.
- No LoRA/fine-tuning against SuperMemory-VQA labels in the first benchmark plan.
- No new model architecture claims beyond baseline Gemma 4 E2B QA.
- No privacy bypass, no raw audio reconstruction, no participant identification.
- No requirement that full 703 GB dataset be downloaded during normal test suite; full benchmark is an explicit CLI run.
- No mobile/LiteRT deployment path unless user later asks.

## Open questions
- None blocking. Adopted defaults above can be vetoed at approval.

## Approval gate
status: awaiting-approval
pending action: write `.omo/plans/supermemory-vqa-gemma4-e2b.md`
approach summary: after approval, append a decision-complete task plan with waves for scaffold, data/schema, metric math, causal retrieval baseline, Gemma runner, reporting, and final QA. Every todo will include exact commands, expected PASS/FAIL observables, evidence paths, and Conventional Commit messages.
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
