---
slug: worldmm-smvqa-spatial-memory
status: drafting
intent: clear
pending-action: write .omo/plans/worldmm-smvqa-spatial-memory.md
approach: <fill: the approach you intend to plan>
---

# Draft: worldmm-smvqa-spatial-memory

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
<!-- assumption | adopted default | rationale | reversible? -->

## Findings (cited - path:lines)

## Decisions (with rationale)

## Scope IN

## Scope OUT (Must NOT have)

## Open questions

## Approval gate
status: drafting
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->

## Prometheus state (2026-07-09)
- intent: clear, review_required: false
- prior discussion source: .omo/drafts/supermemory-vqa-gemma4-e2b.md memory ladder v2 (static/dynamic spatial token schema: room/layout/surface/object-anchor/relation/free-space + object-state/last-seen/trajectory/interaction/event/uncertainty) and v3 (semantic-geometry binding CLIP/SigLIP/DINO + SLAM/CUT3R, interface+fixture-first)
- repo facts: stores wired via RetrievalStore literal (retrieval_types.py:7), STORE_ORDER (retrieval.py:27), SUPPORTED_BUILD_STORES (cli_commands.py:40), remote stage 4 (remote_script.py:68-91), lexical scoring + causal cutoff + per-video scoping (retrieval.py), records surface as text snippets in EvidencePack -> QA prompt unchanged
- dataset: SuperMemory-VQA has SLAM trajectories + gaze + IMU (draft findings; HF card); current SourceStreamExample has NO pose fields -> schema extension needed; ingest gap still open
- design default: 4th store "spatial"; PoseSample schema ext; static layer = zone clustering (grid) + object anchors (pose-at-detection, gaze-ray when available) + derived relations; dynamic layer = last-seen/object-state/trajectory; all render to text snippets; spatial relation accuracy + Recall@K diagnostics; v3 = typed binding interface + fixture validation only
- forks to ask: (1) geometry fidelity default pose-anchored vs CUT3R/depth lane now; (2) v2+v3-stub vs v2-only; (3) remote ablation rerun in-scope vs local-only
- status: awaiting-approval, pending action: write .omo/plans/worldmm-smvqa-spatial-memory.md

## Update (turn 2, auto-continuation)
- user silent after brief; announced defaults adopted per brief ("skip = default"): (1) pose-anchored geometry, (2) v2 full + v3 interface stub, (3) remote ablation planned-not-executed
- plan written: 11 todos / 4 waves + final wave
- metis: 16 findings (MAJOR-GAPS), spot-verified (fixture already has 4 questions incl q_fake_004; fixture source of truth = fixture_data.py; unanswerable convention answer="" + empty evidence; run_qa hardcodes 3 stores), ALL 16 folded in
- status: plan-complete, awaiting user choice: $start-work OR high-accuracy dual review first

## Update (retrieval review correction)
- user corrected retrieval requirement: retrieval must satisfy SuperMemory-VQA Video-RAG video-shard protocol and EgoButler coarse-to-fine/hierarchy, blended into WorldMM retrieval.
- plan updated from spatial-store-only to protocol-aware retrieval: RetrievalTrace, Video-RAG 30m shard eligibility, EgoButler hierarchy, WorldMM policy, integrated retrieval adapter, protocol ablations.
- momus round 1 verdict ITERATE: fixed two blockers: q_fake_005 now at question_time=1850 on video_001 extended past 1900 so first 30m shard is eligible; corrupted pose failure QA now uses build-memory --stage chunk because validate-schema does not call temporal-order validation.
- independent Codex CLI review attempted read-only with gpt-5.5 but failed due sandbox/bwrap and cancelled codegraph; not counted as approval.

## Review result
- native momus round 1: ITERATE, blockers: q_fake_005 not 30m-shard eligible; corrupt-pose QA used validate-schema which does not call chunk temporal validation.
- fixes applied: q_fake_005 now uses fake_video_001.end_time>=1900 and question_time=1850 so first shard ending 1800 is eligible; corrupt-pose failure now uses `worldmm-smvqa build-memory --stage chunk`.
- native momus round 2: OKAY. Quote: "Both prior blockers fixed... retrieval contract explicitly covers Video-RAG + EgoButler + WorldMM policy across lines 35-40 and todos 6-9."
- independent Codex CLI review attempted with read-only sandbox and gpt-5.5; failed to read files due bwrap sandbox / cancelled codegraph. Not counted as approval.
- status: plan reviewed OKAY by native reviewer; dual high-accuracy incomplete only because independent CLI environment could not read workspace.
