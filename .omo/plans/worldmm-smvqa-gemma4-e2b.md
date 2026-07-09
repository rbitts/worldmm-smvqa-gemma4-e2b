# worldmm-smvqa-gemma4-e2b - Work Plan

## TL;DR (For humans)
**What you'll get:** A full implementation path for a WorldMM-style SuperMemory-VQA baseline: it builds memory from the allowed video/transcript stream, retrieves only pre-question evidence, runs Gemma 4 E2B for answerability and choice ranking, then reports official benchmark numbers.

**Why this approach:** It keeps SuperMemory-VQA's official evaluation protocol intact while upgrading the memory bank with WorldMM's episodic, semantic, and visual memories. It also respects the project rule that real dataset/model/evaluation work happens only on company compute, not this local host.

**What it will NOT do:** It will not use ground-truth answer evidence as retrieval memory, run full evaluation locally, download Gemma/SuperMemory-VQA locally, or claim exact paper baseline reproduction without reporting implementation deltas.

**Effort:** Large
**Risk:** High - multimodal memory construction, remote-only full evaluation, and strict anti-leakage requirements.
**Decisions to sanity-check:** Use `WorldMM-SMVQA` as the baseline name; keep exact Video-RAG/EgoButler reproduction as a comparison lane; run full benchmark only via bastion/head node.

Your next move: start work with `$start-work .omo/plans/worldmm-smvqa-gemma4-e2b.md`, or run a high-accuracy review first. Full execution detail follows below.

---

> TL;DR (machine): Large/high-risk plan for a Python WorldMM-augmented SuperMemory-VQA baseline with local tiny tests, remote full benchmark execution, Gemma 4 E2B QA, anti-leakage gates, and official metrics.

## Scope
### Must have
- New Python benchmark package in this repo with `pyproject.toml`, `src/worldmm_smvqa/`, `tests/`, `configs/`, `scripts/`, and `README.md`.
- CLI commands:
  - `worldmm-smvqa prepare-fixture`
  - `worldmm-smvqa validate-schema`
  - `worldmm-smvqa build-memory`
  - `worldmm-smvqa retrieve`
  - `worldmm-smvqa qa`
  - `worldmm-smvqa evaluate`
  - `worldmm-smvqa report`
  - `worldmm-smvqa smoke`
  - `worldmm-smvqa launch-remote`
- Dataset API split:
  - `SourceStreamExample`: allowed memory-builder inputs only.
  - `QALabelExample`: evaluator-only labels and evidence.
- 30-second clip chunking and 30-minute shard chunking.
- Memory builders:
  - captions / ASR transcript memory
  - OCR memory interface
  - object-detection memory interface
  - frame reference / visual embedding memory interface
  - WorldMM episodic memory graph
  - WorldMM semantic memory graph
  - WorldMM visual memory store
- Retrieval constrained by question time/end time before Gemma QA.
- Gemma 4 E2B QA interface producing strict JSON:
  - `answerable`
  - `ranked_choices`
  - `answer`
  - `confidence`
  - `supporting_memory_ids`
  - `prompt_token_count`
  - `raw_model_output_path`
- Official metrics:
  - Ans-F1
  - QA-Acc
  - QA-MRR
- Memory diagnostics:
  - Memory Recall@K against eval-only evidence labels
  - causal violation count
  - retrieval source distribution
  - prompt token budget
  - memory size by store
- Local-only tiny fixture tests and dry-runs.
- Remote-only full benchmark download/build/eval launch scripts via bastion/head node.
- Final report records local code/config changes, remote command, remote job/process reference, remote artifact path, metrics/failure, and what was not copied locally.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- Must not download SuperMemory-VQA full dataset on local host.
- Must not download Gemma 4 E2B model weights on local host.
- Must not run real evaluation or model inference on local host.
- Must not use `answer.evidence_list`, `is_answerable`, correct answer, choice labels, or verification scores inside memory builders or retrievers.
- Must not hardcode bastion, company storage, dataset path, model path, or internal URLs; use env/config variables.
- Must not copy full datasets, model weights, checkpoints, or sensitive artifacts back to local.
- Must not do LoRA/fine-tuning/training.
- Must not describe WorldMM-SMVQA as the paper's exact Video-RAG/EgoButler baseline unless exact reproduction lane also runs and deltas are reported.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD for schema split, anti-leakage, chunking, causal cutoff, retrieval, prompt packing, metrics; tests-after for remote wrapper scripts and README command smoke.
- Local allowed checks:
  - `uv run pytest -q`
  - `uv run ruff check .`
  - `uv run mypy src`
  - `uv run worldmm-smvqa smoke --fixture tests/fixtures/tiny_smvqa --out .omo/evidence/smoke-worldmm-smvqa`
- Local forbidden checks:
  - any command that downloads model weights
  - any command that downloads full SuperMemory-VQA
  - any real benchmark eval
- Remote full-run checks:
  - invoked only through `worldmm-smvqa launch-remote` or generated `scripts/remote/run_worldmm_smvqa.sh`
  - requires explicit user approval before actual execution because AGENTS.md says to ask before expensive/long-running/multi-node jobs.
- Evidence directory convention:
  - local task evidence: `.omo/evidence/worldmm-smvqa/task-N/*` where `N` is the todo number
  - remote plan evidence manifest: `.omo/evidence/worldmm-smvqa/remote-manifest.json`
  - remote actual artifact path: env-configured approved storage, not local.

## Execution strategy
### Parallel execution waves
- Wave 1: scaffold, schema, fixtures, anti-leakage tests.
- Wave 2: chunking, memory store interfaces, metric math.
- Wave 3: WorldMM memory builders and retrieval.
- Wave 4: Gemma QA interface, evidence prompt packing, local dry-run mocks.
- Wave 5: remote launch scripts, report, README, final verification.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | none | 2-14 | none |
| 2 | 1 | 3-14 | 3 after fixture paths settle |
| 3 | 1 | 4-14 | 2 |
| 4 | 2,3 | 5,6,7,8 | none |
| 5 | 4 | 7,8,10 | 6 |
| 6 | 4 | 7,8,10 | 5 |
| 7 | 5,6 | 8,10,11 | none |
| 8 | 7 | 10,11 | 9 |
| 9 | 4 | 12,14 | 8 |
| 10 | 8 | 11,12 | none |
| 11 | 10 | 12,13 | none |
| 12 | 9,11 | 13,14 | none |
| 13 | 12 | 14 | none |
| 14 | 13 | final verification | none |

## Todos
> Implementation + Test = ONE todo. Never separate.

- [x] 1. Scaffold Python project and local-only guardrails
  What to do / Must NOT do: Create `pyproject.toml`, `src/worldmm_smvqa/`, `tests/`, `configs/`, `scripts/remote/`, `.gitignore`, and base CLI. Add a startup guard that refuses local commands marked `requires_remote=true` unless `WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST=1` is set and config says `runtime.location=remote`. Must not add production data/model download commands to local smoke path.
  Parallelization: Wave 1 | Blocked by: none | Blocks: all
  References (executor has NO interview context - be exhaustive): `AGENTS.md`; SuperMemory-VQA paper lines 200-206 for remote compute; Gemma model card lines 293-320 for model scale.
  Acceptance criteria (agent-executable): `uv run worldmm-smvqa --help` exits 0; `uv run worldmm-smvqa smoke --help` exits 0; `uv run worldmm-smvqa launch-remote --dry-run --config configs/remote.example.yaml` prints commands without network calls.
  QA scenarios (name the exact tool + invocation): happy: `uv run worldmm-smvqa --help > .omo/evidence/worldmm-smvqa/task-1/help.txt` and file contains `build-memory`; failure: `uv run worldmm-smvqa qa --config configs/local.example.yaml --real-model` exits non-zero and stderr contains `remote-only`. Evidence `.omo/evidence/worldmm-smvqa/task-1/`.
  Commit: Y | `chore(scaffold): create worldmm smvqa benchmark project`

- [x] 2. Define source-stream and QA-label schemas with anti-leakage boundaries
  What to do / Must NOT do: Implement Pydantic models in `src/worldmm_smvqa/schema.py`: `SourceStreamExample`, `StreamChunk`, `MemoryRecord`, `QuestionRequest`, `QALabelExample`, `PredictionRecord`, `MetricRecord`. Memory builders accept only `SourceStreamExample`/`StreamChunk`, never `QALabelExample`. Add explicit prohibited fields list: `answer`, `answer_choices.choice_ltype`, `is_answerable`, `evidence_list`, `verification_score`.
  Parallelization: Wave 1 | Blocked by: 1 | Blocks: 4-14
  References: SuperMemory-VQA Appendix D.2 lines 600-623; AGENTS.md safety rules.
  Acceptance criteria: `uv run pytest tests/test_schema_anti_leakage.py -q` passes; a fixture containing labels cannot be passed to memory builder without a `LeakageError`.
  QA scenarios: happy: `uv run pytest tests/test_schema_anti_leakage.py::test_source_stream_excludes_labels -q`; failure: `uv run pytest tests/test_schema_anti_leakage.py::test_memory_builder_rejects_qa_label_example -q` must fail before implementation with `LeakageError` missing, then pass. Evidence `.omo/evidence/worldmm-smvqa/task-2/pytest.txt`.
  Commit: Y | `feat(schema): separate source streams from evaluation labels`

- [x] 3. Add tiny SuperMemory-VQA fixture generator
  What to do / Must NOT do: Add `tests/fixtures/tiny_smvqa/` with 2 videos, 4 questions, fake transcript spans, fake OCR/object/frame metadata, and eval-only labels. Add `worldmm-smvqa prepare-fixture --out tests/fixtures/tiny_smvqa.generated` to recreate it. Must not include real participant data.
  Parallelization: Wave 1 | Blocked by: 1 | Blocks: 4-14
  References: SuperMemory-VQA task categories lines 113-134; privacy/de-identification from paper lines 169-171.
  Acceptance criteria: `uv run worldmm-smvqa validate-schema --input tests/fixtures/tiny_smvqa` exits 0 and reports `source_examples=2`, `qa_examples=4`.
  QA scenarios: happy: `uv run worldmm-smvqa prepare-fixture --out /tmp/worldmm-smvqa-fixture && uv run worldmm-smvqa validate-schema --input /tmp/worldmm-smvqa-fixture`; failure: malformed fixture missing `question_time` exits non-zero with `question_time`. Evidence `.omo/evidence/worldmm-smvqa/task-3/`.
  Commit: Y | `test(fixtures): add tiny smvqa stream fixture`

- [x] 4. Implement 30-second clip and 30-minute shard chunking
  What to do / Must NOT do: Add `src/worldmm_smvqa/chunking.py` to split source streams into 30s clips and 30min shards with stable IDs: `video_id:start:end:granularity`. Clip/shard generation must preserve modality refs and source timestamps. No labels allowed in chunk metadata.
  Parallelization: Wave 2 | Blocked by: 2,3 | Blocks: 5-8,10-14
  References: SuperMemory-VQA implementation lines 202-206; user-requested structure.
  Acceptance criteria: `uv run pytest tests/test_chunking.py -q` verifies exact boundary behavior at 0, 29.999, 30, 1799.999, 1800 seconds.
  QA scenarios: happy: `uv run worldmm-smvqa build-memory --fixture tests/fixtures/tiny_smvqa --stage chunk --out .omo/evidence/worldmm-smvqa/task-4/chunks.jsonl` creates both `clip_30s` and `shard_30m`; failure: unsorted timestamps fail with `TemporalOrderError`. Evidence `.omo/evidence/worldmm-smvqa/task-4/`.
  Commit: Y | `feat(chunking): split streams into clips and shards`

- [x] 5. Build allowed caption/OCR/object/frame memory sources
  What to do / Must NOT do: Add `src/worldmm_smvqa/memory_sources.py` with builders for `CaptionMemory`, `OCRMemory`, `ObjectMemory`, and `FrameMemoryRef`. Local fixture mode uses fake generated captions/OCR/object rows. Remote mode has hooks for generated artifacts but does not run expensive caption/OCR/object models locally.
  Parallelization: Wave 2 | Blocked by: 4 | Blocks: 7,8,10
  References: Video-RAG description lines 195-196; EgoButler captioning lines 197-204; AGENTS.md local/remote rules.
  Acceptance criteria: `uv run pytest tests/test_memory_sources.py -q` passes and proves builders read only source-stream fields.
  QA scenarios: happy: `uv run worldmm-smvqa build-memory --fixture tests/fixtures/tiny_smvqa --stage source-memories --out .omo/evidence/worldmm-smvqa/task-5/source_memories.jsonl`; failure: add eval-only `answer.evidence_list` to input and assert memory output contains no `evidence_list` values. Evidence `.omo/evidence/worldmm-smvqa/task-5/`.
  Commit: Y | `feat(memory): build allowed source memory records`

- [x] 6. Implement official metric math and diagnostics first
  What to do / Must NOT do: Add `src/worldmm_smvqa/metrics.py` for Ans-F1, QA-Acc, QA-MRR, Memory Recall@K, causal violation count, prompt token summary, memory size summary. Metrics may read `QALabelExample`; memory/retrieval modules may not.
  Parallelization: Wave 2 | Blocked by: 4 | Blocks: 9,12,14
  References: SuperMemory-VQA metrics lines 207-209; Appendix D.2 choices lines 603-620.
  Acceptance criteria: `uv run pytest tests/test_metrics.py -q` passes hand-calculated fixtures including answerable false positives, wrong top-1 with correct at rank 3, and missing correct choice.
  QA scenarios: happy: `uv run worldmm-smvqa evaluate --pred tests/fixtures/tiny_smvqa/predictions.good.jsonl --labels tests/fixtures/tiny_smvqa/labels.jsonl --out .omo/evidence/worldmm-smvqa/task-6/metrics.json` outputs all metrics; failure: malformed ranked choices with duplicates fails with `InvalidPredictionError`. Evidence `.omo/evidence/worldmm-smvqa/task-6/`.
  Commit: Y | `feat(metrics): compute smvqa official scores`

- [x] 7. Build WorldMM episodic memory
  What to do / Must NOT do: Add `src/worldmm_smvqa/worldmm/episodic.py`. Convert 30s clips and 30min shards into multi-scale event graph nodes with temporal edges, source modality refs, confidence, and text embedding IDs. No graph edge can point beyond the source chunk time span.
  Parallelization: Wave 3 | Blocked by: 5,6 | Blocks: 8,10,11
  References: WorldMM project lines 27-33; SuperMemory-VQA long-horizon/multi-evidence lines 155-164.
  Acceptance criteria: `uv run pytest tests/test_worldmm_episodic.py -q` verifies node IDs, temporal edges, and multi-scale parent-child links.
  QA scenarios: happy: `uv run worldmm-smvqa build-memory --fixture tests/fixtures/tiny_smvqa --store episodic --out .omo/evidence/worldmm-smvqa/task-7/episodic.jsonl`; failure: overlapping invalid event spans fail with `InvalidTemporalGraphError`. Evidence `.omo/evidence/worldmm-smvqa/task-7/`.
  Commit: Y | `feat(worldmm): build episodic event memory`

- [x] 8. Build WorldMM semantic and visual memories
  What to do / Must NOT do: Add `src/worldmm_smvqa/worldmm/semantic.py` and `visual.py`. Semantic memory stores relation/habit triples derived from repeated source events and captions. Visual memory stores frame refs, embedding refs, OCR/object refs, and timestamp grounding. Local fixture mode uses deterministic fake embeddings; remote mode records artifact pointers to real embeddings.
  Parallelization: Wave 3 | Blocked by: 7 | Blocks: 10,11
  References: WorldMM project lines 28-44; Gemma visual/audio capabilities lines 365-374.
  Acceptance criteria: `uv run pytest tests/test_worldmm_semantic_visual.py -q` passes with no external model calls.
  QA scenarios: happy: `uv run worldmm-smvqa build-memory --fixture tests/fixtures/tiny_smvqa --store semantic,visual --out .omo/evidence/worldmm-smvqa/task-8/worldmm_sv`; failure: visual memory without timestamp fails with `MissingGroundingError`. Evidence `.omo/evidence/worldmm-smvqa/task-8/`.
  Commit: Y | `feat(worldmm): build semantic and visual memory stores`

- [x] 9. Implement remote-only artifact/config workflow
  What to do / Must NOT do: Add `configs/local.example.yaml`, `configs/remote.example.yaml`, and `scripts/remote/run_worldmm_smvqa.sh`. Local config must point only to tiny fixture paths. Remote config uses env vars: `SMVQA_DATA_ROOT`, `GEMMA_MODEL_PATH`, `WORLDMM_OUTPUT_ROOT`, `BASTION_HOST`, `HEAD_NODE`, `REMOTE_JOB_LAUNCHER`. No hardcoded internal paths.
  Parallelization: Wave 3 | Blocked by: 4 | Blocks: 12,14
  References: AGENTS.md remote resource rules and expected deliverables.
  Acceptance criteria: `uv run worldmm-smvqa launch-remote --dry-run --config configs/remote.example.yaml` prints an `ssh "$BASTION_HOST"` command and does not connect.
  QA scenarios: happy: dry-run command writes `.omo/evidence/worldmm-smvqa/task-9/remote_dry_run.sh`; failure: missing `WORLDMM_OUTPUT_ROOT` exits with `MissingRemoteConfig`. Evidence `.omo/evidence/worldmm-smvqa/task-9/`.
  Commit: Y | `build(remote): add remote benchmark launch templates`

- [x] 10. Implement adaptive causal retrieval over WorldMM memories
  What to do / Must NOT do: Add `src/worldmm_smvqa/retrieval.py`. Retrieval accepts `QuestionRequest` and memory stores, filters all candidates to `candidate.end_time <= question_time`, then iteratively retrieves from episodic/semantic/visual stores until evidence budget is met. It outputs `EvidencePack` with memory IDs, snippets, frame refs, source store, time spans, and retrieval scores.
  Parallelization: Wave 4 | Blocked by: 7,8 | Blocks: 11,12
  References: WorldMM adaptive retrieval lines 31-33; SuperMemory-VQA causal cutoff lines 202-206.
  Acceptance criteria: `uv run pytest tests/test_retrieval_causal.py -q` proves post-question memory is excluded and ablation flags select store subsets.
  QA scenarios: happy: `uv run worldmm-smvqa retrieve --fixture tests/fixtures/tiny_smvqa --question q_object_001 --stores episodic,semantic,visual --out .omo/evidence/worldmm-smvqa/task-10/evidence_pack.json`; failure: injected post-question high-score memory is not returned and command writes `causal_filtered_count>0`. Evidence `.omo/evidence/worldmm-smvqa/task-10/`.
  Commit: Y | `feat(retrieval): retrieve causal worldmm evidence`

- [x] 11. Implement Gemma 4 E2B QA interface with mock local backend and remote real backend
  What to do / Must NOT do: Add `src/worldmm_smvqa/qa.py`. Local backend is `MockQABackend` for tiny fixtures only. Remote backend loads Gemma 4 E2B via configured path or `google/gemma-4-E2B-it` using Transformers. The QA prompt contains question, four choices, retrieved evidence pack, and strict JSON output instructions. Local real-model use must error.
  Parallelization: Wave 4 | Blocked by: 10 | Blocks: 12,13
  References: Gemma model card lines 293-320, 365-374; SuperMemory-VQA four-choice metrics lines 207-209.
  Acceptance criteria: `uv run pytest tests/test_qa_prompt.py -q` proves prompt includes no labels and parser rejects malformed JSON after bounded retry.
  QA scenarios: happy: `uv run worldmm-smvqa qa --fixture tests/fixtures/tiny_smvqa --backend mock --out .omo/evidence/worldmm-smvqa/task-11/predictions.jsonl`; failure: `uv run worldmm-smvqa qa --fixture tests/fixtures/tiny_smvqa --backend gemma4 --local` exits with `remote-only`. Evidence `.omo/evidence/worldmm-smvqa/task-11/`.
  Commit: Y | `feat(qa): add gemma e2b qa interface`

- [x] 12. Wire end-to-end local smoke pipeline
  What to do / Must NOT do: Add `worldmm-smvqa smoke` that runs validate-schema -> chunk -> build source memories -> build WorldMM stores -> retrieve -> mock QA -> evaluate -> report on tiny fixture. Must finish under 30 seconds locally and use no network.
  Parallelization: Wave 5 | Blocked by: 9,11 | Blocks: 13,14
  References: AGENTS.md local host rules.
  Acceptance criteria: `uv run worldmm-smvqa smoke --fixture tests/fixtures/tiny_smvqa --out .omo/evidence/worldmm-smvqa/task-12/smoke` exits 0 and writes `metrics.json`, `predictions.jsonl`, `evidence_packs.jsonl`, `memory_manifest.json`.
  QA scenarios: happy: smoke command above; failure: `WORLDMM_SMVQA_DISABLE_MOCK=1 uv run worldmm-smvqa smoke ...` exits with `NoLocalModelBackend`. Evidence `.omo/evidence/worldmm-smvqa/task-12/`.
  Commit: Y | `feat(cli): wire local worldmm smvqa smoke`

- [x] 13. Add remote full benchmark plan command and manifest contract
  What to do / Must NOT do: Implement `launch-remote` to generate a remote script that runs: prepare source manifests, build 30s/30m chunks, generate/load captions/OCR/object/frame refs, build WorldMM E/S/V stores, retrieve per QA under causal cutoff, run Gemma 4 E2B QA, evaluate, and write summary. It may generate commands but must not submit unless user passes `--submit` and required env confirms approval.
  Parallelization: Wave 5 | Blocked by: 12 | Blocks: 14
  References: AGENTS.md remote workflow; SuperMemory-VQA full data size and compute notes; WorldMM memory construction.
  Acceptance criteria: `uv run worldmm-smvqa launch-remote --dry-run --config configs/remote.example.yaml --out .omo/evidence/worldmm-smvqa/task-13/remote_plan` writes `run_worldmm_smvqa.sh`, `expected_outputs.json`, and `copyback_policy.txt`.
  QA scenarios: happy: dry-run artifact contains commands for all pipeline stages and output paths under `$WORLDMM_OUTPUT_ROOT`; failure: `--submit` without `WORLDMM_SMVQA_REMOTE_APPROVED=1` exits with `ExplicitApprovalRequired`. Evidence `.omo/evidence/worldmm-smvqa/task-13/`.
  Commit: Y | `build(remote): generate full benchmark run plan`

- [x] 14. Add report, README, and final result handoff template
  What to do / Must NOT do: Add `worldmm-smvqa report` and `README.md` documenting local smoke, remote launch, exact environment variables, no-local-download policy, metric interpretation, and baseline naming. Report must include local code/config changed, remote command used, remote job ID/process ref, remote artifact path, key metrics/failure, and what was not copied locally.
  Parallelization: Wave 5 | Blocked by: 13 | Blocks: final verification
  References: AGENTS.md expected deliverables; SuperMemory-VQA metrics; WorldMM baseline naming.
  Acceptance criteria: `uv run worldmm-smvqa report --run-manifest tests/fixtures/tiny_smvqa/remote_manifest.example.json --out .omo/evidence/worldmm-smvqa/task-14/report.md` writes all required AGENTS.md sections.
  QA scenarios: happy: report command above; failure: manifest missing remote artifact path exits with `IncompleteRemoteManifest`. Evidence `.omo/evidence/worldmm-smvqa/task-14/`.
  Commit: Y | `docs(report): document worldmm smvqa benchmark workflow`

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [x] F1. Plan compliance audit: verify every todo maps to the user flow `SuperMemory-VQA stream -> 30s/30m chunk -> caption/OCR/object/frame memory -> WorldMM E/S/V memory -> pre-question retrieval -> Gemma 4 E2B QA -> metrics`. Command: `rg -n "30-second|30-minute|caption|OCR|object|frame|episodic|semantic|visual|Gemma 4 E2B|Ans-F1|QA-Acc|QA-MRR" .omo/plans/worldmm-smvqa-gemma4-e2b.md`.
- [x] F2. Code quality review: after implementation, run `uv run ruff check . && uv run mypy src && uv run pytest -q`.
- [x] F3. Real manual QA: local smoke command `uv run worldmm-smvqa smoke --fixture tests/fixtures/tiny_smvqa --out .omo/evidence/worldmm-smvqa/final-smoke` exits 0 and writes metrics/predictions/evidence/memory manifests.
- [x] F4. Scope fidelity: verify no local full dataset/model artifacts exist with `find . -maxdepth 5 \( -name "*.safetensors" -o -name "*.pt" -o -name "*.parquet" -size +100M \) -print` returning empty, and verify remote run plan uses env vars not hardcoded paths.

## Commit strategy
- Do not auto-commit unless user asks.
- Use atomic Conventional Commits:
  - `chore(scaffold): create worldmm smvqa benchmark project`
  - `feat(schema): separate source streams from evaluation labels`
  - `test(fixtures): add tiny smvqa stream fixture`
  - `feat(chunking): split streams into clips and shards`
  - `feat(memory): build allowed source memory records`
  - `feat(metrics): compute smvqa official scores`
  - `feat(worldmm): build episodic event memory`
  - `feat(worldmm): build semantic and visual memory stores`
  - `build(remote): add remote benchmark launch templates`
  - `feat(retrieval): retrieve causal worldmm evidence`
  - `feat(qa): add gemma e2b qa interface`
  - `feat(cli): wire local worldmm smvqa smoke`
  - `build(remote): generate full benchmark run plan`
  - `docs(report): document worldmm smvqa benchmark workflow`
- Final commit footer when committing: `Plan: .omo/plans/worldmm-smvqa-gemma4-e2b.md`

## Success criteria
- Local code/config/tests can be built and verified without real dataset/model downloads.
- Memory builders cannot access QA labels or answer evidence; anti-leakage tests fail before implementation and pass after.
- Tiny fixture smoke produces memory manifests, evidence packs, mock predictions, and metric JSON.
- Remote launch dry-run produces a complete full benchmark script and copyback policy.
- Full benchmark path, when run remotely after explicit approval, can produce Gemma 4 E2B predictions and official SuperMemory-VQA Ans-F1 / QA-Acc / QA-MRR.
- Final report includes local changes, remote command, remote job/process reference, remote artifact path, metrics/failure reason, and what was not copied locally.
