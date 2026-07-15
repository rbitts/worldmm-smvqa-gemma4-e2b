# EXP-0004: G-CUT3R-compatible provider gate

| 항목 | 값 |
| --- | --- |
| Page ID | SM-EXP-0004 |
| Experiment ID | EXP-0004 |
| 상태 | Local provider/cache contract implemented; company provider validation not run |
| 실행 identity | `execution_profile=teacher-oracle` provider-gate preparation |
| Remote provenance | **No remote run, job ID, provider, artifact, or metric exists** |

## 목적과 경계

EXP-0004 selects or rejects an offline external G-CUT3R-compatible provider for EXP-0005. It is not a G-CUT3R reproduction, persistent-memory format, student runtime, or benchmark result. The repository does not bundle, download, import, or automatically invoke provider code/weights.

The local contract covers external provider/cache provenance, causal cutoff validation, rank-specific cache planning, request/response digest lineage, and label-blind teacher materialization. Local `launch-remote --dry-run` produces planning evidence only; it does not contact company systems.

## Company-only preflight and gate

Before Phase A, a remote CPU preflight actually runs `audit-sensors` and `validate-teacher-oracle-inputs` against production sensor inputs. It sources only a staged bootstrap environment; after review, audit/config digests atomically enter the immutable final `.env.worldmm`, which Phase A and B reuse unchanged. The audit is `sensor-audit-v1`, uses the fixed `30,000,000`-microsecond window, and binds both sensor-audit and experiment-config digests in the strict teacher-only validation receipt. It reports selected RGB readability, actual calibration, native causal pose/VIO, and depth availability without synthesis. Its staging paths are outside the not-yet-created run root. A missing signal, provider, digest, or capability is blocked/not measurable, never assumed valid.

The audit producer schema is closed and allowlisted. It rejects every question, choice, answer, label, and evidence field or alias; those values are not available to the audit, provider request, cache, teacher construction, or memory materialization. `SMVQA_FRAME_ROOT` remains the approved input root. Production generates the sensor manifest/observations; only then are their generated audit/manifest digests verified.

A Phase-A approval is an RFC 8785-compatible canonical-JSON Ed25519 signature, verified against an allowlisted, unrevoked, purpose-authorized key in the signed registry, not a named approver string. It binds the run, digests, policy/capacity, and Phase A. The producer uses `--no-requeue`; Phase A is producer → gate → terminal finalizer. The Phase-B approval and continue receipt are separately signed and cryptographically verified.

## Gate acceptance invariants

The gate strictly decodes every provider shard; malformed, extra, duplicate, or unparseable rows are No-Go. Before writing a continue receipt, it verifies:

| Invariant | Required result |
| --- | --- |
| Exact request coverage | The causal `(video_id, frame_ref, timestamp)` request multiset exactly matches selected sensor inventory; no missing, extra, or cross-rank duplicate request |
| Causality | Request cutoff and returned observation time never exceed the source/question cutoff |
| Signed lineage | All provider, checkpoint, config, source, and split lineage is signed, present, and exactly matches the approved bindings |
| Record inputs | Teacher construction receives no question, choice, answer, label, evidence, or alias |
| Output boundary | Transient provider output is not persistent memory; only validated typed records flow to EXP-0005 |
| Accounting | Exactly one cluster-qualified, duplicate-visible, allocation-only, parsable `sacct` row has `State%64=COMPLETED`, `ExitCode=0:0`, zero restarts, correct SLUID lineage, marker, and provider artifacts |

The gate writes the conditional mode-`0600`, atomic, signed/sealed artifact `summary/teacher_oracle_continue.json` only after all invariants pass; it carries Phase-A producer lineage and bindings and is an operational receipt, not scientific Go. Failure/indeterminacy writes `summary/teacher_oracle_terminal.json` with `provider_gate_decision=not_decidable` and no continue receipt; success writes that terminal artifact with `provider_gate_decision=go`. `summary/dag_jobs.teacher_oracle.env` records actual Phase-A IDs, while `summary/dag_jobs.phase_b.env` exists only after second-approved Phase-B admission. Fake test job IDs/receipts are not company evidence. Failed, cancelled, requeued, missing, ambiguous, or invalid attempts are No-Go and require a new run ID/root and new Phase-A approval.

## Next relation

A passing operational provider gate may admit bounded EXP-0005 `E0`/`T0`/`T1` only after receipt-bound second approval. It does not authorize probe/full legacy lanes or a student/device claim. No company provider execution or provider comparison result exists.
