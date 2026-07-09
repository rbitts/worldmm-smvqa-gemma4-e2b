# worldmm-smvqa-spatial-memory - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** An explicit spatial memory for the WorldMM-SMVQA baseline plus a retrieval layer that stays compatible with SuperMemory-VQA's Video-RAG shard protocol and EgoButler's hierarchical-memory style while using WorldMM memories as the selectable evidence banks. The system learns WHERE things are from movement/gaze, retrieves them through a protocol-aware controller, and reports with/without-spatial deltas.

**Why this approach:** SuperMemory-VQA validity comes from causal video-shard retrieval and no label leakage; EgoButler compatibility comes from coarse-to-fine hierarchical summaries; WorldMM compatibility comes from adaptive selection over episodic/semantic/visual/spatial memories. The plan makes these explicit as one deterministic retrieval contract before any heavier agentic/PPR/embedding retriever is attempted.

**What it will NOT do:** No 3D reconstruction models, no new dependencies, no changes to official metrics. No LLM retriever, PPR, FAISS, or embedding search in this phase; those are documented as WorldMM/Video-RAG fidelity upgrades behind a deterministic retrieval interface. No real dataset/model downloads or remote job submission from this machine.

**Effort:** Medium
**Risk:** High - retrieval behavior becomes part of the claim, not just store wiring; the plan must preserve Video-RAG shard causality, EgoButler hierarchy, and WorldMM memory selection with traceable tests.
**Decisions to sanity-check:** Object positions approximate the viewer's location (gaze-corrected when available) rather than true object geometry; retrieval now has a protocol trace and policy controller; the with/without-spatial ablation is part of this phase.

Your next move: start work with `$start-work .omo/plans/worldmm-smvqa-spatial-memory.md`, or run a high-accuracy review first. Full execution detail follows below.

---

> TL;DR (machine): High-risk plan - add 4th "spatial" store plus protocol-aware retrieval: SuperMemory-VQA Video-RAG 30m shard eligibility, EgoButler coarse-to-fine summaries, WorldMM adaptive store/granularity policy, relation/Recall@K diagnostics, ablation artifact, v3 binder interface stub.

## Scope
### Must have
- Optional spatial signals on the source schema: `PoseSample` (timestamp, x, y, z, optional yaw) and `GazeSample` (timestamp, x, y, z target point) tuples on `SourceStreamExample`, sliced per chunk window like the other timed fields and covered by the temporal-order validator.
- Tiny fixture upgraded with synthetic pose/gaze tracks and two new spatial questions (one answerable from pre-question anchors, one whose only spatial evidence lies after `question_time` and must resolve unanswerable).
- A fourth memory store `"spatial"` in `src/worldmm_smvqa/worldmm/`, mirroring the episodic/semantic/visual pattern:
  - static layer: `ZoneRecord` (grid clustering of pose samples), `SpatialAnchorRecord` (object anchored at wearer pose at detection time, gaze target preferred when a gaze sample falls inside the detection interval), `SpatialRelationRecord` (near-relations derived from anchor geometry within a zone and overlapping time).
  - dynamic layer: per-chunk `ObjectStateSnapshotRecord` ("as of chunk end, object X last seen in zone Z at t=...") plus a wearer trajectory summary per chunk; snapshots carry `base_score` recency so newer state wins retrieval ties.
  - every record renders to a deterministic text snippet so the EXISTING lexical retrieval, causal cutoff, evidence pack, and Gemma QA path work unchanged.
- Full wiring: `RetrievalStore` literal, `STORE_ORDER`, retrieval record adapter, `SUPPORTED_BUILD_STORES`, `build-memory --stores ...,spatial`, retrieve default stores, smoke pipeline, remote launch plan stage 4 + `memory_manifest.json` key.
- Spatial diagnostics: relation accuracy against hand-labeled fixture expectations + per-store Memory Recall@K, written as a JSON artifact during smoke.
- v3 stub: a typed `SemanticGeometryBinder` protocol (semantic embedding x geometry primitive binding) with a deterministic no-op implementation and fixture-backed tests - interface only, no model weights.
- Local ablation: smoke comparison with/without the spatial store producing a metrics-delta artifact, and the remote plan documenting the same ablation flags for the company-compute rerun.
- Retrieval contract is a first-class deliverable, not an afterthought:
  - SuperMemory-VQA / Video-RAG compatibility: retrieval starts from 30-minute `shard_30m` eligibility; only shards whose `end_time <= question_time` are eligible, text candidates are drawn from eligible shards, and frame evidence is capped by config at 32 refs per question. This is traced in every `EvidencePack`.
  - EgoButler compatibility: within each eligible shard, retrieval builds a hierarchy `shard_30m -> clip_30s -> memory records`; optional hour/day summaries can be added later, but the first implementation MUST support coarse-to-fine selection from shard summary to clip records to per-store evidence.
  - WorldMM compatibility: a deterministic `WorldMMRetrievalPolicy` selects memory stores and hierarchy depth from the raw question text (location/where/near/zone -> spatial first; what-happened/when -> episodic first; category/object relation -> semantic first; appearance/OCR/frame -> visual first; default balanced). The policy then retrieves from WorldMM E/S/V/Spatial stores inside the Video-RAG/EgoButler eligible pools. It replaces blind round-robin evidence allocation.
  - Traceability: `EvidencePack` gains `retrieval_trace` with selected protocol path, eligible shard ids, selected clip ids, policy route, store order, candidate counts before/after causal filtering, and frame count. QA prompt still consumes snippets/frames; metrics unchanged.
- An explicit "Retrieval implementation delta vs WorldMM" section in README/HANDOFF: WorldMM's published retrieval uses an agentic iterative loop and graph/embedding retrieval components; this phase implements a deterministic controller that embeds SuperMemory-VQA Video-RAG and EgoButler constraints into WorldMM memory selection. `ponytail:` comments name the ceiling and the upgrade path: PPR graph retrieval, LLM reranking, embedding search, and STOP-controlled iterative retrieval.
- Anti-leakage preserved: the spatial builder consumes only `SourceStreamExample`/`StreamChunk` inputs; label access stays test-enforced.
### Must NOT have (guardrails, anti-slop, scope boundaries)
- NO 3D reconstruction / depth models / CUT3R integration and NO CLIP/SigLIP/DINO model downloads - v3 stays a typed interface with a no-op binder (`ponytail:` ceiling comments name the upgrade path).
- NO new dependencies: geometry is stdlib `math` only; clustering is a fixed-size grid, not sklearn.
- NO official-metric changes. Retrieval scoring MAY change only inside the new protocol-aware retrieval layer, with tests proving causal shard eligibility, coarse-to-fine hierarchy, and policy routing. No label-derived scoring.
- NO agentic/iterative LLM retrieval loop, NO PPR graph retrieval, NO LLM reranker, NO FAISS/embedding search in this phase - those are documented as fidelity upgrades behind the deterministic controller.
- NO QA/prompt-format changes and NO metric-logic changes (`metrics.py` untouched).
- NO real dataset/model downloads or remote submission on this host; remote ablation execution stays behind the existing approval gates (HANDOFF prerequisites).
- NO weakening of existing tests: fixture-count pins may be updated only where the fixture intentionally grew, with the diff called out in the commit body.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD for geometry/state logic (zones, anchors, relations, snapshots - hand-computed expected values written as failing pytest FIRST), tests-after for pure wiring (literal/CLI/remote-plan string changes). Framework: existing pytest + `uv run ruff check .` + `uv run basedpyright` after every todo.
- Gates: full `uv run pytest` green (baseline 70 passed), zero ruff/basedpyright findings, and the local smoke (`uv run worldmm-smvqa smoke --fixture tests/fixtures/tiny_smvqa --out <dir>`) must keep producing `metrics.json`, `predictions.jsonl`, `evidence_packs.jsonl`, `memory_manifest.json` PLUS the new `spatial_diagnostics.json`. Every produced `EvidencePack` must include a `retrieval_trace` proving shard eligibility, hierarchy path, WorldMM policy route, and frame cap.
- Evidence: `.omo/evidence/worldmm-smvqa-spatial/task-<N>/` (pytest transcripts, smoke output listings, jq checks).

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.
- Wave 1 (foundations): todo 1 (schema + chunk slicing), todo 2 (fixture upgrade + pinned-count reconciliation).
- Wave 2 (spatial store core, after wave 1): todo 3 (types + static layer), todo 4 (dynamic layer + snippet rendering + recency).
- Wave 3 (retrieval contract, after wave 2): todo 5 (retrieval trace schema), todo 6 (Video-RAG shard eligibility), todo 7 (EgoButler coarse-to-fine hierarchy), todo 8 (WorldMM retrieval policy), todo 9 (integrated retrieval adapter).
- Wave 4 (wiring, after wave 3): todo 10 (CLI), todo 11 (smoke + remote plan + manifest), todo 12 (spatial diagnostics), todo 13 (v3 binder stub), todo 14 (ablation), todo 15 (docs).
- Final verification wave: F1-F4.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | - | 2, 3, 4 | - |
| 2 | 1 | 5, 6, 7, 8, 10 | 3 |
| 3 | 1 | 4, 5 | 2 |
| 4 | 3 | 5 | - |
| 5 | 1, 2 | 6, 7, 8, 9 | 3, 4 |
| 6 | 5 | 9, 11, 12, 14 | 7, 8 |
| 7 | 5 | 9, 11, 12, 14 | 6, 8 |
| 8 | 5 | 9, 11, 12, 14 | 6, 7 |
| 9 | 3, 4, 6, 7, 8 | 10, 11, 12, 14 | - |
| 10 | 9 | 11 | 12, 13 |
| 11 | 10 | 14, 15 | 12, 13 |
| 12 | 9 | 15 | 10, 11, 13 |
| 13 | 3 | 15 | 10, 11, 12 |
| 14 | 9, 11 | 15 | 13 |
| 15 | 11, 12, 13, 14 | F1-F4 | - |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [x] 1. Schema: pose/gaze samples + chunk slicing + temporal validation
  What to do: in `src/worldmm_smvqa/schema.py` add `class PoseSample(FrozenModel): timestamp: float; x: float; y: float; z: float; yaw: float | None = None` and `class GazeSample(FrozenModel): timestamp: float; x: float; y: float; z: float` next to the existing timed metadata models; add `pose_samples: tuple[PoseSample, ...] = ()` and `gaze_samples: tuple[GazeSample, ...] = ()` to `SourceStreamExample` (schema.py:77-86; empty defaults keep old fixtures valid). COORDINATE CONVENTION (binding, document in the PoseSample docstring and assert in tests): x/y = horizontal plane used for zones and distances, z = vertical, meters. In `src/worldmm_smvqa/chunking.py` extend `_build_chunk` (chunking.py:126-155) to slice both by `_inside(sample.timestamp, window)` and `_require_temporal_order` (chunking.py:179-199) to assert both timestamp tuples are sorted. TDD: write the failing test FIRST in `tests/test_spatial_schema.py` - a source with pose/gaze samples chunks into 30s windows with only in-window samples, and an out-of-order pose track raises the existing temporal-order error type.
  Must NOT do: no changes to existing field semantics; no required (non-default) new fields.
  Parallelization: Wave 1 | Blocked by: - | Blocks: 2, 3, 4
  References: `src/worldmm_smvqa/schema.py:40-90` (TimedModel/metadata model style), `src/worldmm_smvqa/chunking.py:126-155,179-210`, `tests/` existing chunking tests for assertion style.
  Acceptance criteria: `uv run pytest tests/test_spatial_schema.py` green after RED captured; `uv run pytest` all green; `uv run ruff check .` and `uv run basedpyright` clean.
  QA scenarios: happy = pytest transcript (RED then GREEN) to `.omo/evidence/worldmm-smvqa-spatial/task-1/pytest.txt`; failure = `uv run worldmm-smvqa validate-schema --input tests/fixtures/tiny_smvqa` still exits 0 on the UNCHANGED fixture (backward compat receipt in same dir).
  Commit: Y | feat(schema): add pose and gaze samples with chunk slicing

- [x] 2. Fixture: synthetic pose/gaze tracks + two spatial questions
  What to do: the fixture's source of truth is `src/worldmm_smvqa/fixture_data.py` (`tiny_fixture_examples()`, consumed by `prepare-fixture` via `src/worldmm_smvqa/fixtures.py:42-44`) - update it FIRST, then regenerate/update the checked-in `tests/fixtures/tiny_smvqa/*.jsonl` and `predictions.good.jsonl` to match. The fixture currently has FOUR questions (q_fake_001..q_fake_004, `tests/fixtures/tiny_smvqa/questions.jsonl`); the new spatial questions are `q_fake_005` and `q_fake_006`. IMPORTANT protocol fixture change: make `fake_video_001.end_time` at least `1900.0` so chunking creates a first `shard_30m` ending at `1800.0`; add sparse pose/frame samples through 1900s. Put the answerable spatial evidence before 1800s and ask `q_fake_005` AFTER that shard closes: `question_time=1850.0`, "Where was the fake mug last seen?", gold = place-type choice "beside the notebook", `is_answerable: true`, evidence `fake_video_001:5:12:spatial`. This is mandatory because Video-RAG eligibility is `shard_30m.end_time <= question_time`; a 180s source/question at 120s has no eligible 30m shard. `fake_video_001` gets `pose_samples` every ~10s near early evidence plus sparse samples near 1800/1850 to keep temporal validation simple; `fake_video_002` gets analogous pose/gaze but can remain short for the unanswerable/future case unless tests need long-video parity. `q_fake_006` (video_002, question_time 15.0, "Which zone was the blue magnet in?", magnet detection at 132-137s is AFTER question_time -> unanswerable with the EXACT existing convention: `answer: ""`, `is_answerable: false`, `evidence_list: ()` per `tests/test_metrics.py:54-56,137-141`). Update the pinned tests (counts 4 -> 6 and golden strings) at EXACTLY: `tests/test_fixture_generator.py:33,51`, `tests/test_smoke_pipeline.py:79,103`, `tests/test_qa_prompt.py:138,170`, `tests/test_remote_contract.py:88,169`, `tests/test_metrics.py:49`, plus `predictions.good.jsonl` rows for the two new questions.
  Must NOT do: do not alter existing q_fake_001-004 rows; do not weaken any assertion beyond the intentional count/golden change.
  Parallelization: Wave 1 | Blocked by: 1 | Blocks: 5, 6, 7, 8, 10
  References: `src/worldmm_smvqa/fixture_data.py` (source of truth), `src/worldmm_smvqa/fixtures.py:42-44`, `tests/fixtures/tiny_smvqa/*.jsonl`, `src/worldmm_smvqa/schema.py:102-131`, `src/worldmm_smvqa/metrics.py:220-257`, `tests/test_metrics.py:50-58,137-141` (unanswerable shape).
  Acceptance criteria: `uv run worldmm-smvqa validate-schema --input tests/fixtures/tiny_smvqa` exits 0; `uv run pytest` all green; `uv run worldmm-smvqa evaluate --pred tests/fixtures/tiny_smvqa/predictions.good.jsonl ...` (existing invocation in tests) still passes.
  QA scenarios: happy = validate-schema + pytest transcripts to `.omo/evidence/worldmm-smvqa-spatial/task-2/`; failure = corrupt one pose timestamp ordering in a COPY of the fixture and confirm `uv run worldmm-smvqa build-memory --stage chunk --fixture <copy> --out /tmp/wm-bad-chunks.jsonl` exits nonzero (temporal validation is in chunking, not validate-schema), receipt in same dir.
  Commit: Y | test(fixture): add pose gaze tracks and spatial questions to tiny fixture

- [x] 3. Spatial static layer: zones, anchors, relations (+types)
  What to do: create `src/worldmm_smvqa/worldmm/spatial_types.py` with `ZoneRecord` (zone_id: str, video_id, centroid x/y/z, cell: tuple[int,int], visit intervals: tuple[tuple[float,float],...]), `SpatialAnchorRecord` (memory_id, store: Literal["spatial"], video_id, object label, x/y/z, zone_id, start_time/end_time = detection interval, frame_refs, confidence, provenance: Literal["pose","gaze"]), `SpatialRelationRecord` (memory_id, store: Literal["spatial"], video_id, subject/relation("near")/object labels, zone_id, start_time/end_time), `SpatialBuildSummary`, plus a `InvalidSpatialInputError` dataclass exception - all mirroring `src/worldmm_smvqa/worldmm/episodic_types.py` and `visual.py:36-50` style. Create `src/worldmm_smvqa/worldmm/spatial.py` with: `build_zones(pose_samples, *, cell_size: float = 2.0)` - grid cell = `(floor(x/cell_size), floor(y/cell_size))` on the x/y horizontal plane (todo 1 convention), merge consecutive samples in the same cell into visit intervals, zone_id = `zone_<video_id>_<cellx>_<celly>` deterministic; `build_object_anchors(source)` - for each `object_detections` entry, position = gaze sample target if one exists with timestamp inside the detection interval (provenance "gaze"), else wearer pose interpolated linearly at the detection midpoint (provenance "pose"; `# ponytail: anchor approximates viewer position, upgrade path = v3 depth/geometry binding`); anchor `frame_refs` = the `frame_metadata` entries (schema.py:71-75) whose `timestamp` falls inside the detection interval, else empty tuple - deterministic; `derive_relations(anchors, *, near_threshold: float = 1.5)` - pairs in the same zone with overlapping intervals and euclidean x/y distance <= threshold emit one "near" relation with lexicographically ordered subject/object. Pure stdlib `math`. TDD: `tests/test_spatial_static.py` FIRST with hand-computed fixtures (3-4 poses, 2 detections, 1 gaze) asserting exact zone ids, anchor coords, provenance, frame_refs, and the single expected near-relation; plus a failure case (unsorted pose input raises `InvalidSpatialInputError`).
  Must NOT do: no numpy/sklearn; no reading of QALabelExample anywhere in the module (anti-leakage).
  Parallelization: Wave 2 | Blocked by: 1 | Blocks: 4, 5
  References: `src/worldmm_smvqa/worldmm/episodic_types.py:12-50`, `src/worldmm_smvqa/worldmm/visual.py:36-60`, `src/worldmm_smvqa/worldmm/semantic.py:61-91` (builder + write_fixture pattern), `src/worldmm_smvqa/schema.py:60-90` (ObjectMetadata has label/confidence/times only - NO frame ref; FrameMetadata carries frame_ref+timestamp).
  Acceptance criteria: RED captured then `uv run pytest tests/test_spatial_static.py` green; full pytest/ruff/basedpyright clean.
  QA scenarios: happy = RED+GREEN transcripts to `.omo/evidence/worldmm-smvqa-spatial/task-3/`; failure = unsorted-pose test asserts the exact exception message.
  Commit: Y | feat(spatial): add zone anchor and relation static memory layer

- [x] 4. Spatial dynamic layer: per-chunk state snapshots + snippets + recency
  What to do: in `src/worldmm_smvqa/worldmm/spatial.py` add `ObjectStateSnapshotRecord` (in spatial_types.py: memory_id, store "spatial", video_id, object label, zone_id, last_seen_time, x/y/z, start_time/end_time = the 30s chunk window, snippet-ready) and `build_object_state_snapshots(clip_chunks, anchors)` - for each 30s clip chunk emit, per object already seen at or before chunk end, one snapshot "as of t=<chunk_end>s, <object> last seen in <zone_id> at t=<last_seen>s near (<x>,<y>)"; also `build_trajectory_summaries(clip_chunks, zones)` - per chunk one record "wearer in <zone_id(s)> during [<start>,<end>]s". Snippet templates are module-level `Final` constants. Recency: each snapshot/summary carries `base_score = end_time / max(1.0, stream_end_time)` (in [0,1], comparable to the confidence-based base_scores at `retrieval.py:145,158,177`), so newer state wins ties via the existing `+ base_score * 0.01` term (`retrieval.py:203`). Causality note: snapshot `end_time` = chunk end, so the existing `end_time <= question_time` filter (`retrieval.py:63-65`) only admits fully-past snapshots - never encode any post-chunk information in the snippet. TDD: `tests/test_spatial_dynamic.py` FIRST - hand-built 3-chunk stream asserts snapshot count, exact snippet strings, monotonically increasing base_score, and that an object first seen in chunk 3 has NO snapshot in chunks 1-2.
  Must NOT do: no "until now"/future phrasing in snippets; no state carried across videos.
  Parallelization: Wave 2 | Blocked by: 3 | Blocks: 5
  References: `src/worldmm_smvqa/retrieval.py:63-65,140-180,195-205`, `src/worldmm_smvqa/chunking.py:52-75` (clip_30s granularity), todo 3 outputs.
  Acceptance criteria: RED then GREEN on `uv run pytest tests/test_spatial_dynamic.py`; full gates clean.
  QA scenarios: happy = RED+GREEN transcripts to `.omo/evidence/worldmm-smvqa-spatial/task-4/`; failure = future-leak test (object seen only later has no earlier snapshot) is the named regression guard.
  Commit: Y | feat(spatial): add causal object state snapshots and trajectory summaries

- [x] 5. Retrieval trace schema + protocol contract
  What to do: extend `src/worldmm_smvqa/retrieval_types.py` with `RetrievalTrace` and nested records: `eligible_shard_ids`, `selected_clip_ids`, `policy_route`, `store_order`, `candidate_counts`, `causal_filtered_count`, `frame_ref_count`, `protocols: tuple[Literal["smvqa-video-rag","egobutler","worldmm"], ...]`. Add `retrieval_trace: RetrievalTrace` to `EvidencePack` with a default factory for backward fixture compatibility if needed. TDD: `tests/test_retrieval_protocol_trace.py` FIRST, asserting `EvidencePack.model_validate_json` requires/round-trips trace in new outputs and that a missing trace in legacy fixture paths gets only the documented default.
  Must NOT do: no label fields in trace; no prompt changes.
  Parallelization: Wave 3 | Blocked by: 1, 2 | Blocks: 6, 7, 8, 9
  References: `src/worldmm_smvqa/retrieval_types.py`, `src/worldmm_smvqa/retrieval.py:47-75`, prior draft `.omo/drafts/supermemory-vqa-gemma4-e2b.md:40,72-74`.
  Acceptance criteria: RED then GREEN on `uv run pytest tests/test_retrieval_protocol_trace.py`; full gates clean.
  QA scenarios: happy = trace round-trip transcript to `.omo/evidence/worldmm-smvqa-spatial/task-5/`; failure = hand-built malformed trace with frame count > cap fails validation.
  Commit: Y | feat(retrieval): add protocol trace to evidence packs

- [x] 6. SuperMemory-VQA Video-RAG shard eligibility layer
  What to do: add `src/worldmm_smvqa/retrieval_protocols.py` with `eligible_video_rag_shards(question, chunks)` selecting only `StreamChunk.granularity == "shard_30m"` with same video and `end_time <= question.question_time`; add `filter_records_to_shards(records, eligible_shards)` so candidates must fall inside an eligible shard span; enforce `max_frame_refs=32` at evidence packing. Tests FIRST: a question at 45s sees only shard(s) ending <=45s; a post-question injected high-score memory inside a future shard is excluded even before score sorting; selected frame refs never exceed 32.
  Must NOT do: no FAISS dependency; text score can remain lexical for now, but candidate pool must obey shard eligibility.
  Parallelization: Wave 3 | Blocked by: 5 | Blocks: 9, 11, 12, 14
  References: `.omo/drafts/supermemory-vqa-gemma4-e2b.md:40,72`, `src/worldmm_smvqa/chunking.py:52-75,126-155`, `src/worldmm_smvqa/retrieval.py:57-65`.
  Acceptance criteria: `uv run pytest tests/test_retrieval_video_rag_protocol.py` RED then GREEN; trace includes eligible shard ids and frame count.
  QA scenarios: happy = shard eligibility test transcript to `.omo/evidence/worldmm-smvqa-spatial/task-6/`; failure = future-shard high-score candidate is filtered and increments causal/shard exclusion count.
  Commit: Y | feat(retrieval): enforce video-rag shard eligibility

- [x] 7. EgoButler coarse-to-fine hierarchy layer
  What to do: add hierarchy helpers in `retrieval_protocols.py`: `build_egobutler_hierarchy(chunks, records)` creates deterministic parent links `shard_30m -> clip_30s -> memory_ids`, and `coarse_to_fine_candidates(question, hierarchy, records)` first scores eligible shard/clip snippets, then expands only selected clips to memory records. Since current code has 30s clips and 30m shards but no hour/day summaries, represent hour/day summary nodes as optional future inputs and keep the first implementation at shard->clip->record. Add `ponytail:` comment: hour/day summaries upgrade when ingest produces them. Tests FIRST: selected evidence memory ids all belong to selected clip ids; disabling coarse-to-fine via test flag changes trace but not causal cutoff; hierarchy has no cross-video edges.
  Must NOT do: no synthetic labels or answer evidence in summaries; no generated summaries locally.
  Parallelization: Wave 3 | Blocked by: 5 | Blocks: 9, 11, 12, 14
  References: `.omo/drafts/supermemory-vqa-gemma4-e2b.md:73`, `src/worldmm_smvqa/chunking.py`, `src/worldmm_smvqa/worldmm/episodic.py:41-100`.
  Acceptance criteria: `uv run pytest tests/test_retrieval_egobutler_protocol.py` RED then GREEN; trace includes selected clip ids.
  QA scenarios: happy = hierarchy transcript to `.omo/evidence/worldmm-smvqa-spatial/task-7/`; failure = cross-video memory injection excluded.
  Commit: Y | feat(retrieval): add egobutler coarse-to-fine hierarchy

- [x] 8. WorldMM retrieval policy controller
  What to do: add `WorldMMRetrievalPolicy` in `retrieval_protocols.py`. Input: raw `QuestionRequest` and available stores. Output: ordered route with `store_order`, `hierarchy_depth`, and `reason` string. Rules: location terms (`where`, `last seen`, `near`, `zone`, `left`, `right`) -> `("spatial","episodic","semantic","visual")`; event/time terms -> episodic first; category/relation terms -> semantic first; visual/OCR/frame/color terms -> visual first; default balanced `STORE_ORDER`. Use raw question text and answer choices, not `_query_terms`, because `STOP_WORDS` drops "where". Tests FIRST: q_fake_005 routes spatial first, an OCR question routes visual first, event question routes episodic first, and no route includes a disabled store.
  Must NOT do: no LLM call, no PPR, no embedding search; deterministic only.
  Parallelization: Wave 3 | Blocked by: 5 | Blocks: 9, 11, 12, 14
  References: `src/worldmm_smvqa/retrieval.py:27,195-230`, `.omo/drafts/supermemory-vqa-gemma4-e2b.md:50-53,74`.
  Acceptance criteria: `uv run pytest tests/test_worldmm_retrieval_policy.py` RED then GREEN; policy route appears in trace.
  QA scenarios: happy = routing tests transcript to `.omo/evidence/worldmm-smvqa-spatial/task-8/`; failure = disabled-store test proves policy cannot route to stores omitted by CLI/ablation.
  Commit: Y | feat(retrieval): add worldmm retrieval policy controller

- [x] 9. Integrated retrieval adapter: spatial store + protocol-aware evidence selection
  What to do: extend `type RetrievalStore` to include `"spatial"`; add `_spatial_candidate`; extend `build_retrieval_records(...)` with spatial records; replace blind `_adaptive_evidence` round-robin with protocol-aware flow: Video-RAG eligible shards -> EgoButler coarse-to-fine selected clips -> WorldMM policy store order -> score candidates within selected clips -> evidence pack capped by budget and 32 frame refs. Keep `_score_candidate` lexical unless todo 8 route reason supplies only ordering. Update all callers. Tests-after: spatial future snapshot filtered; q_fake_005 retrieves spatial evidence; q_fake_006 unanswerable because magnet evidence after question_time excluded; trace contains all three protocols.
  Must NOT do: no label reads, no metric changes, no prompt changes.
  Parallelization: Wave 3 | Blocked by: 3, 4, 6, 7, 8 | Blocks: 10, 11, 12, 14
  References: `src/worldmm_smvqa/retrieval_types.py`, `src/worldmm_smvqa/retrieval.py:27,47-75,84-180,195-245`, `tests/test_retrieval_causal.py:40-60`.
  Acceptance criteria: `uv run pytest tests/test_retrieval_causal.py tests/test_retrieval_*protocol*.py tests/test_worldmm_retrieval_policy.py tests/test_spatial_*.py` green; full gates clean.
  QA scenarios: happy = integrated retrieval pytest transcript to `.omo/evidence/worldmm-smvqa-spatial/task-9/`; failure = future spatial + future shard injection both excluded.
  Commit: Y | feat(retrieval): blend video-rag egobutler and worldmm retrieval

- [x] 10. CLI wiring: build-memory and retrieve support spatial + retrieval protocol flags
  What to do: add `"spatial"` to `SUPPORTED_BUILD_STORES`; extend semantic/visual/spatial build group; update retrieve and QA defaults to include spatial. Add optional retrieve/smoke flags `--retrieval-protocol worldmm-smvqa` (default), `--retrieval-protocol legacy-round-robin` (test/debug only), and `--max-frame-refs 32`; thread through `ParsedArgs` and retrieval calls. Existing `--stores` still controls enabled stores for ablation. Tests-after: `--stores semantic,visual,spatial` writes spatial file; `retrieve --retrieval-protocol worldmm-smvqa` emits trace; bogus protocol exits nonzero.
  Must NOT do: no new top-level commands; legacy flag is not used by remote default.
  Parallelization: Wave 4 | Blocked by: 9 | Blocks: 11
  References: `src/worldmm_smvqa/cli_commands.py:40,60-80,199-233`, `src/worldmm_smvqa/cli_args.py`, `src/worldmm_smvqa/qa.py:180-195`.
  Acceptance criteria: CLI happy/failure invocations pass; full gates clean.
  QA scenarios: happy = CLI transcripts to `.omo/evidence/worldmm-smvqa-spatial/task-10/`; failure = bogus protocol and bogus store both nonzero; cleanup temp dirs.
  Commit: Y | feat(cli): expose spatial and protocol-aware retrieval

- [x] 11. Smoke + remote launch plan + memory manifest include spatial/protocol retrieval
  What to do: extend smoke types and helpers so outputs include `spatial_memory`, `retrieval_trace` in evidence packs, and `spatial_diagnostics.json`. Remote stage 4 builds `semantic,visual,spatial`; remote stage 5 explicitly passes `--stores episodic,semantic,visual,spatial --retrieval-protocol worldmm-smvqa --max-frame-refs 32`; `remote_plan.ExpectedOutputs` lists spatial memory, retrieval trace-bearing evidence packs, and ablation outputs. Tests-after: update remote golden strings and smoke tests.
  Must NOT do: no remote submission; no stage reorder beyond retrieval args.
  Parallelization: Wave 4 | Blocked by: 10 | Blocks: 14, 15
  References: `src/worldmm_smvqa/smoke.py:47-70,97,144,180`, `src/worldmm_smvqa/remote_script.py:60-110`, `src/worldmm_smvqa/remote_plan.py:122-140`, `tests/test_remote_launch_plan.py`, `tests/test_remote_contract.py`.
  Acceptance criteria: smoke output evidence packs have `retrieval_trace.protocols` containing all three protocol names; dry-run plan contains `--retrieval-protocol worldmm-smvqa` and `--max-frame-refs 32`.
  QA scenarios: happy = smoke + dry-run transcripts to `.omo/evidence/worldmm-smvqa-spatial/task-11/`; failure = missing env var still fails fast; cleanup temp dirs.
  Commit: Y | feat(remote): run protocol-aware spatial retrieval in launch plan

- [x] 12. Spatial/retrieval diagnostics: relation accuracy + per-store/protocol Recall@K
  What to do: add `tests/fixtures/tiny_smvqa/expected_relations.jsonl` (hand-labeled: the mug-notebook "near" relation etc., matching todo 3 geometry); add `src/worldmm_smvqa/worldmm/spatial_diagnostics.py` with `relation_accuracy(predicted_relations, expected)` (exact-match precision/recall/F1 over (subject, relation, object, zone_id) tuples) and `memory_recall_at_k(evidence_packs, labels, k)` computed per store and per protocol trace. NOTE: no reusable evidence-span parser exists - `metrics.py:259-282` does exact string intersection for recall and only parses the end time for causal diagnostics - so WRITE a new parser `parse_evidence_span("video:start:end:store") -> (video_id, start, end, store)` in spatial_diagnostics.py with its own tests (malformed span -> typed error); a retrieved EvidenceItem counts as a hit when its video matches and its [start_time,end_time] overlaps the span interval. Wire smoke (via the todo-11 store-parametric helper) to write `spatial_diagnostics.json` `{relation_accuracy: {...}, recall_at_k: {episodic: .., semantic: .., visual: .., spatial: ..}, protocol_recall_at_k: {...}, k: 6}`. TDD for the pure functions with hand-computed values in `tests/test_spatial_diagnostics.py`.
  Must NOT do: diagnostics module MAY read labels (evaluator surface) but must stay importable ONLY from smoke/evaluate paths, never from builders - assert no `spatial.py` -> diagnostics import.
  Parallelization: Wave 4 | Blocked by: 9 | Blocks: 15
  References: `src/worldmm_smvqa/metrics.py:56-115,259-282`, draft `.omo/drafts/supermemory-vqa-gemma4-e2b.md:79` (previously agreed diagnostics: Recall@K + spatial relation accuracy), `src/worldmm_smvqa/smoke.py:47-70` (SmokeArtifacts to extend with the diagnostics path).
  Acceptance criteria: RED then GREEN on `uv run pytest tests/test_spatial_diagnostics.py`; smoke output contains `spatial_diagnostics.json` with `jq -e '.relation_accuracy.f1 >= 0 and (.recall_at_k | has("spatial"))'`; full gates clean.
  QA scenarios: happy = pytest + jq transcripts to `.omo/evidence/worldmm-smvqa-spatial/task-12/`; failure = empty expected_relations file yields a defined zero-division-safe result (explicit test).
  Commit: Y | feat(diagnostics): add spatial and protocol retrieval diagnostics

- [x] 13. v3 stub: SemanticGeometryBinder interface + no-op binder
  What to do: add `src/worldmm_smvqa/worldmm/geometry_binding.py` - `class GeometryPrimitive(FrozenModel)` (frame_ref, x/y/z, source: Literal["slam_pose","gaze"]), `class BoundAnchor(FrozenModel)` (anchor memory_id, embedding_ref: str | None, primitive), `class SemanticGeometryBinder(Protocol)` with `def bind(self, anchor: SpatialAnchorRecord, primitives: Sequence[GeometryPrimitive]) -> BoundAnchor: ...`, and `class NoopBinder` returning `embedding_ref=None` with the nearest-in-time primitive. `# ponytail: interface only - real CLIP/SigLIP/DINO x SLAM/CUT3R binding is the v3 upgrade path, gated on remote GPU work.` Tests-after: `tests/test_geometry_binding.py` validates NoopBinder determinism on fixture anchors and that the Protocol is runtime-checkable where needed.
  Must NOT do: no model imports, no transformers/torch dependency in this module, no wiring into retrieval yet.
  Parallelization: Wave 4 | Blocked by: 3 | Blocks: 15
  References: draft `.omo/drafts/supermemory-vqa-gemma4-e2b.md` v3 decision ("interface + fixture-backed validation first, not full model training"), `src/worldmm_smvqa/worldmm/spatial_types.py` (todo 3).
  Acceptance criteria: `uv run pytest tests/test_geometry_binding.py` green; `rg -n 'import torch|import transformers' src/worldmm_smvqa/worldmm/geometry_binding.py` finds nothing; full gates clean.
  QA scenarios: happy = pytest transcript to `.omo/evidence/worldmm-smvqa-spatial/task-13/`; failure = empty-primitives call raises a typed error (explicit test).
  Commit: Y | feat(worldmm): add semantic-geometry binding interface stub

- [x] 14. Ablation: with/without spatial and protocol comparisons
  What to do: add `--ablation-stores` plus optional `--ablation-protocol legacy-round-robin`; run SECOND retrieval+QA pass through the protocol-aware helper. Smoke writes `ablation.json` with baseline stores/protocol, ablation stores/protocol, baseline metrics, ablation metrics, deltas, and trace summaries. Required ablations: without spatial (`episodic,semantic,visual`) and protocol-only (`legacy-round-robin`) on tiny fixture. Remote launch plan documents equivalent reruns under `$WORLDMM_OUTPUT_ROOT/ablation/` but does not submit.
  Must NOT do: no duplicate memory building for the ablation pass (reuse built records, filter stores at retrieve time); no remote submission.
  Parallelization: Wave 4 | Blocked by: 9, 11 | Blocks: 15
  References: `src/worldmm_smvqa/smoke.py:97,180`, `src/worldmm_smvqa/cli_args.py:16,27,105,224`, `src/worldmm_smvqa/cli_commands.py:146-151`, `src/worldmm_smvqa/retrieval.py:49-62`, `src/worldmm_smvqa/remote_plan.py:122-140`.
  Acceptance criteria: `uv run worldmm-smvqa smoke --fixture tests/fixtures/tiny_smvqa --out /tmp/wm-ablation --ablation-stores episodic,semantic,visual` exits 0 and `jq -e '.delta | has("Ans-F1")' /tmp/wm-ablation/ablation.json`; full gates clean.
  QA scenarios: happy = smoke + jq transcripts to `.omo/evidence/worldmm-smvqa-spatial/task-14/`; failure = `--ablation-stores spatial,bogus` exits nonzero; cleanup `rm -rf /tmp/wm-ablation` with receipt.
  Commit: Y | feat(smoke): add spatial and retrieval protocol ablations

- [x] 15. Docs closeout: README + HANDOFF spatial and retrieval contract sections
  What to do: README.md - document the fourth store, pose/gaze schema fields, Video-RAG shard eligibility, EgoButler hierarchy, WorldMM policy route, `retrieval_trace`, diagnostics artifact, and ablation command. HANDOFF.md - add "Spatial Memory (v2)" and "Retrieval Contract" sections stating: this is a WorldMM-augmented Video-RAG/EgoButler-compatible retrieval path, not an exact WorldMM agentic/PPR implementation; `$SMVQA_DATA_ROOT` schema now includes optional `pose_samples`/`gaze_samples`; remote rerun must use `--retrieval-protocol worldmm-smvqa --max-frame-refs 32`. Keep naming caveat.
  Must NOT do: no company hostnames/paths; no removal of existing sections.
  Parallelization: Wave 4 | Blocked by: 11, 12, 13, 14 | Blocks: F1-F4
  References: README.md, HANDOFF.md "Still open" paragraph, AGENTS.md "Safety Rules".
  Acceptance criteria: `rg -F 'spatial' README.md HANDOFF.md` hits both; `rg -n 'hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN' README.md HANDOFF.md` finds nothing; `uv run pytest` green.
  QA scenarios: happy = rg transcripts to `.omo/evidence/worldmm-smvqa-spatial/task-15/`; failure = secret-value scan must exit 1 (no matches), receipt in same file.
  Commit: Y | docs: document explicit spatial memory and retrieval contract

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [x] F1. Plan compliance audit - re-run every todo's acceptance command from its evidence transcript; all 15 satisfied.
- [x] F2. Code quality review - full `uv run pytest`, `uv run ruff check .`, `uv run basedpyright`; no leftover TODOs without `ponytail:` justification in new modules.
- [x] F3. Real manual QA - fresh smoke on the upgraded fixture: q_fake_005 evidence pack contains a spatial snippet mentioning the mug's zone and `retrieval_trace.protocols` contains `smvqa-video-rag`, `egobutler`, and `worldmm`; q_fake_006 resolves unanswerable with the magnet snapshot causally filtered (inspect `evidence_packs.jsonl` + `spatial_diagnostics.json` + `ablation.json` by hand with jq).
- [x] F4. Scope fidelity - `rg -n 'import torch|import numpy|sklearn' src/worldmm_smvqa/worldmm/` empty; `git status --short` shows only intended files; anti-leakage: `rg -n 'QALabelExample' src/worldmm_smvqa/worldmm/spatial.py src/worldmm_smvqa/worldmm/spatial_types.py` empty.

## Commit strategy
- Conventional Commits, one per todo as listed (15 commits; each todo individually green before commit).
- `.omo/evidence/` stays untracked (gitignored); `.omo/plans/worldmm-smvqa-spatial-memory.md` + draft are committed with the docs commit.
- Final commit footer: `Plan: .omo/plans/worldmm-smvqa-spatial-memory.md`.

## Success criteria
- A fourth explicit spatial store exists (zones, object anchors, near-relations, causal per-chunk state snapshots, trajectory summaries) built ONLY from allowed source-stream inputs, retrievable through the protocol-aware Video-RAG/EgoButler/WorldMM retrieval path, and visible in smoke and remote launch-plan outputs.
- The upgraded fixture proves behavior end-to-end: the "last seen" spatial question retrieves a spatial snippet; the post-question spatial evidence case is causally filtered and unanswerable.
- `spatial_diagnostics.json` (relation accuracy + per-store Recall@K) and `ablation.json` (with/without-spatial metric delta) are produced by local smoke.
- v3 `SemanticGeometryBinder` interface + NoopBinder exist with tests and zero model dependencies.
- All gates green: full pytest, ruff, basedpyright; no new dependencies in pyproject.toml; docs updated with the ingest schema extension note.
- Remote ablation rerun is PLANNED and documented (launch plan includes spatial; ablation flags documented) but NOT executed from this host; submission stays behind the existing approval gate.
