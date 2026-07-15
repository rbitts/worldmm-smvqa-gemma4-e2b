# EXP-0005: Causal teacher-oracle object/location ceiling

| 항목 | 값 |
| --- | --- |
| Page ID | SM-EXP-0005 |
| Experiment ID | EXP-0005 |
| 상태 | Local contract implemented; company execution unverified and not run |
| Profile/lane/result | `teacher-oracle` / `teacher_oracle` / `teacher_oracle` |
| Variants | `E0`, `T0`, `T1` |
| Remote provenance | **No remote session, Slurm job ID, provider output, company artifact, or metric exists** |

## Decision

Measure whether an offline, evidence-bound teacher improves the object/location QA slice over E0. This is a bounded teacher-oracle ceiling, not a student, on-device, official benchmark, or G-CUT3R reproduction claim. Phase A and Phase B each require separate cryptographically signed approval.

`E0` is the shared semantic object-presence control: its persisted `object_presence_v1` contains no geometry, place, entity, or identity field. `T0` adds teacher selected geometry; `T1` adds geometry plus place. The checked-in EXP-0005 configuration literally uses `frame_bound_place`: only same-observation object→place is allowed, and cross-observation last-location/last-seen/count is forbidden. Identity is absent in this mode; a separately approved `stable_last_location` configuration must include the identity producer/capability/resource exactly once. All variants use the same split, causal selected-frame inventory, and QA backend/prompt. They have the same 4,096-byte serialized-byte **cap** per 30-second window; realized serialized bytes are measured per variant and must not be described as equal merely because caps are equal. The audit constant is exactly `30,000,000` microseconds.

## Implemented local contract

The repository implements independent camera intrinsics, optional depth/gaze, trusted causal IMU/raw or VIO/online-causal pose rules, selected teacher-point object targets, evidence/confidence gates, actual-byte validation, and `teacher_oracle` output manifests. Its local dry-run validates the profile/config and creates `operator_contract.json`; it neither contacts company systems nor produces remote evidence.
The reviewed example config is structurally strict only after every `REPLACE_*` binding is replaced with reviewed data. It declares the exact nine capability contracts; RGB/intrinsics policy and approved roots; the fixed frame-bound geometry/semantic/place producer set; evaluator-only QA label roots; `afterany` gate and terminal dependencies; a complete per-stage resource map for preflight, all Phase-A jobs, terminal, materialize, retrieve, QA, evaluator, and finalizer; exact accounting query/settle policy; and signer, quality, and sealed-resolver declarations. It is not evidence and is not runnable while placeholders remain.

## Required company evidence and stop gates

| Gate | Required evidence | Stop condition |
| --- | --- | --- |
| CPU preflight/sensor audit | Remote CPU preflight actually runs `sensor-audit-v1`, validates the audit with `validate-teacher-oracle-inputs`, and records audit/config digests in the strict teacher-only validation receipt | Missing/synthetic/leaking sensor fact or digest mismatch |
| Input boundary | Closed allowlisted producer schema; reject question/choice/answer/label/evidence and all aliases; approved `SMVQA_FRAME_ROOT` is retained | Any student/QA input or unapproved frame root |
| Provider readiness | Approved external executable/revision/checkpoint/config, semantic mask/place provider and ontology, signed provenance/capability evidence | Input absent, unapproved, or unverifiable |
| Accounting readiness | Exact top-level `/opt/slurm/bin/sacct -X -n -P -j <IDs> --format=JobIDRaw,State,ExitCode` rows after the approval-bound settle policy | Missing, ambiguous, or unparsable accounting row |
| Phase A approval | RFC 8785-compatible canonical-JSON Ed25519 approval verified using an allowlisted, unrevoked, valid key authorized for `phase_a_approval`; all run/digest/policy/capacity bindings match | Missing, altered, unsafe, unsigned, or mismatched approval |
| Phase-B admission | Signed/sealed `summary/teacher_oracle_continue.json` plus a separate `phase_b_approval` that cryptographically binds its receipt SHA-256 | Receipt absent, invalid, or terminal state not Go |

Preflight validates the final immutable `.env.worldmm` *before sourcing*: it must be an operator-owned, regular, non-symlink mode-`0600` file whose SHA-256 equals the independently supplied expected digest. The same unchanged file is reused for both phases; approval paths and `WORLDMM_SMVQA_REMOTE_APPROVED=1` are command-scoped. Tracked source deployment is fixed to the approved SHA and verified through a remote content manifest, while the generated plan is transferred separately. Preflight staging and plan artifacts are outside the not-yet-created run root. The sensor manifest and observations are generated from approved production inputs; only after production are their generated audit/manifest digests verified. No approval, receipt, job, audit, manifest, provider output, or result is presently available.

Phase A is the generated CPU preflight script followed, after separate approval, by the generated provider-gate script: fixed producers → `afterany` gate → `afterany` terminal. The gate admits each producer only after exact top-level `sacct` `JobIDRaw,State,ExitCode` success, marker, artifact rehash, and attempt lineage. Failure, OOM, node failure, timeout, corruption, or unintentional cancellation is canonical failure; intentional cancellation requires a sealed intent before `scancel`. No polling/manual fallback is allowed. The terminal produces either the sole continuation receipt or the early `provider_gate_terminal_v1` report; Phase-B IDs do not exist on the latter branch.

Phase B is a separately approved generated downstream submission. Its approval binds the SHA-256 of the exact continuation and terminal file bytes, atomically consumes the continuation, and only then submits per-variant materialize/retrieve/label-blind QA followed by evaluator/finalizer. The full profile is `oracle_variants_terminal_v1`. The authoritative commands, script names, job manifests, monitoring, cancellation, recovery, and early/full lightweight copyback procedure are in `HANDOFF.md`; no company command is executed by default.

## Scientific Go/No-Go

Before Phase B, freeze the object/location slice, utility/confidence-interval rule, selective-risk/error bound, seed, split/input digests, frame inventory, and byte cap. Scientific Go/No-Go is derived from the frozen per-variant byte, frame, metric, and risk evidence, not from operational status or an approver assertion. Phase B may run only `E0`, `T0`, and `T1`; teacher and memory construction remain blind to question, choice, answer, label, evidence, and aliases.

Scientific Go requires the predeclared utility and risk rules, valid provenance from audit through result manifests, no invalid/duplicate/persisted-no-write/future/off-scope evidence, and evidence-bound confidence/abstention. Scientific No-Go or not-measurable is a result, not a reason to relabel a legacy run as success.

## Boundaries after completion

Allowed output claim: a run-scoped `teacher_oracle` EXP-0005 conclusion with reviewed evidence. Forbidden claims: student quality, target-device performance, official E1/E2/E3, generic benchmark improvement, or provider quality beyond the measured contract. Company data, frames, weights, checkpoints, provider cache/output, and large artifacts stay remote. Copy back only reviewed lightweight metrics, manifests, reports, terminal receipts, redacted logs/plots, and approved small samples.

`probe` and `full` are deferred legacy lanes: they emit `contract_probe`/`PROBE` and `student`/`E1`, respectively, and cannot satisfy EXP-0005.
