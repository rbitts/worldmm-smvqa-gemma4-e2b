---
slug: worldmm-smvqa-phase2
status: drafting
intent: clear
pending-action: write .omo/plans/worldmm-smvqa-phase2.md
approach: <fill: the approach you intend to plan>
---

# Draft: worldmm-smvqa-phase2

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
- intent: clear
- review_required: false
- status: awaiting-user-answer (interview, 1 question)
- context: HANDOFF.md phases done locally; user says company-side execution done; NO remote manifest/metrics on this host (only tests/fixtures/tiny_smvqa/remote_manifest.example.json); open gap per HANDOFF: no SMVQA dataset ingest step.
- pending fork (owner-fact): what did the company-side run produce?
  A) real benchmark done -> phase = retrieve lightweight results + write remote_manifest.json + `worldmm-smvqa report` final report
  B) blocked on ingest gap -> phase = define/implement raw SMVQA -> schema ingest pipeline (remote)
  C) setup/dry-run only -> phase = submit-readiness checklist + ingest gap plan
- next action after answer: fill .omo/plans/worldmm-smvqa-phase2.md todos, approval gate, then present.

## Update (turn 3)
- user answer: assume remote official SMVQA evaluation COMPLETE (branch A). Counts as approval -> write plan.
- intent: clear (kept), review_required: false
- adopted defaults (reversible internals, recorded not asked):
  - phase2 scope = retrieve lightweight results + real remote_manifest.json + `worldmm-smvqa report` final report + HANDOFF/README status update + per-ltype/answerable breakdown analysis (remote-side, summary-only pullback)
  - test strategy: tests-after, only for new code (breakdown script); ops todos verified by agent-executed jq/rg checks
  - env contract: executor requires BASTION_HOST, WORLDMM_OUTPUT_ROOT, WORLDMM_RUN_ID, REMOTE_JOB_ID_OR_PROCESS_REF set; todos fail fast via ${VAR:?}
- facts: metrics file = $WORLDMM_OUTPUT_ROOT/metrics/official_metrics.json (Ans-F1/QA-Acc/QA-MRR aliases, metrics.py:47-49); manifest schema report.py:44-56 extra=forbid, baseline_name literal "WorldMM-SMVQA"; report sections report.py:11-18
- external: SuperMemory-VQA = HF OSU-AIoT-MLSys-Lab/SuperMemory-VQA, 4,853 MCQ incl unanswerable, cc-by-nc-sa-4.0 (arxiv 2606.00825); WorldMM = arxiv 2512.02425
- status: writing plan todos now

## Update (turn 4)
- plan written: .omo/plans/worldmm-smvqa-phase2.md (7 todos, 3 waves + final wave)
- metis gap analysis: 15 findings (verdict MAJOR-GAPS), key claims independently verified (memory_manifest under memory/ per remote_script.py:80; .omo/evidence/ gitignored, plans/drafts tracked), ALL 15 folded into the plan
- status: plan-complete, awaiting user choice: $start-work OR high-accuracy dual review first
