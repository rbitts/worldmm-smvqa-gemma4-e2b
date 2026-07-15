# Spatial memory status

| 항목 | 값 |
| --- | --- |
| Page ID | SM-STATUS |
| 기준일 | 2026-07-15 |
| 현재 Goal | EXP-0005 teacher-oracle 승인 전 preflight 준비 |
| 로컬 상태 | `teacher-oracle` profile/lane/result contract, sensor audit, provider gate, approval/receipt verification implementation and local tests only |
| 원격 상태 | **원격 접속·Slurm 제출·job ID·sensor audit·provider·approval·receipt·company artifact·metric 모두 없음** |

## 핵심 결론

EXP-0005 is an offline teacher-oracle diagnostic. `E0`, `T0`, and `T1` share the 30,000,000-microsecond audit window, causal frame inventory, QA backend, split, and the same serialized-byte cap. The cap is not a claim that realized bytes are equal; each variant's realized bytes must be retained as evidence. Results, if any, are `result_class=teacher_oracle` only, never student/device/official E1/E2/E3 claims.

## Student contract delivery status

The model-free student boundary contract, local mock DAG, checkpoint-v2 wiring, and
digest propagation are local implementation surfaces only. No production provider lock
has been accepted, no physical all-rank model-load consensus exists, and no student
remote plan has been approved or run. Therefore there is no student checkpoint,
completion terminal, manifest, metric, or quality claim. EXP-0005 remains unchanged and
authoritative for the teacher-oracle experiment.

## Local implementation evidence and limits

| Surface | Local implementation | Remote evidence still required |
| --- | --- | --- |
| Profile | `WORLDMM_EXECUTION_PROFILE=teacher-oracle`, reviewed EXP-0005 config | Approved run ID, output root, split and immutable digests |
| Audit/preflight | `audit-sensors`, `validate-teacher-oracle-inputs`, `sensor-audit-v1`, strict teacher-only validation receipt contract | Actual CPU execution against approved production sensor inputs and audit/config digests |
| Input boundary | Closed allowlisted producer schema rejects question/choice/answer/label/evidence aliases | Review of actual production manifest and approved `SMVQA_FRAME_ROOT` |
| Provider gate | Strict shard decode, exact causal request coverage, signed provider/checkpoint/config/source/split lineage contract | Approved provider, checkpoint/config/ontology, capability and provenance evidence |
| Approval | RFC 8785-compatible canonical Ed25519 registry/approval and signed continue-receipt verification contract | Valid allowlisted keys, separate Phase-A and Phase-B signed approvals, and receipt |
| Slurm | `--no-requeue`, `State%64` lossless-accounting contract, conditional Phase-A/Phase-B DAG | Cluster capability evidence and actual accounting rows |

Local dry-run plan generation is not remote execution. Bootstrap preflight uses a separate staged `bootstrap.env`; only reviewed passing audit/validation results atomically create the immutable final `.env.worldmm`, reused unchanged by both phases. Tracked source is deployed at the approved SHA by archive plus verified remote content manifest; generated plan transfer is separate. The plan, bootstrap, audit, and validation receipt remain outside the not-yet-created run root until first-approved Phase A creates it.
The reviewed EXP-0005 example is intentionally non-runnable until all `REPLACE_*` values are replaced. After replacement it carries the strict capability, sensor-root, producer-input, evaluator-only QA-root, topology, Phase-B resource, accounting-settle, signer, quality, and resolver bindings. The separately reviewed `WORLDMM_ORACLE_RESOURCES_FILE` is the exact Slurm resource producer; its digest is bound as `WORLDMM_RESOURCE_CONFIG_SHA256`.

## Required stop gates

1. No provider submission until a remote CPU preflight actually runs `sensor-audit-v1`, validates the strict teacher-only receipt, and binds audit/config digests.
2. No Phase A until reviewed production evidence and a valid cryptographic Phase-A approval exist.
3. No Phase B until the signed/sealed continue receipt is reviewed and a separate receipt-bound Phase-B approval verifies.
4. Operational provider Go is not scientific Go. Scientific decision derives only from frozen per-variant byte/frame/metric/risk evidence.

## Known blockers

- Selected RGB readability, calibration, causal pose/VIO, and depth coverage have not been audited remotely.
- No approved provider/checkpoint/semantic provider/ontology, production manifest, split digest, signer registry, approval, cluster capability evidence, or unique remote run root has been provided.
- No E0/T0/T1 remote output, metric, confidence interval, realized-byte evidence, or selective-risk result exists.
- The conditional receipt path is mode-`0600` and single-use only: Phase B hard-links `summary/teacher_oracle_continue.json` to `summary/.teacher_oracle_continue.used.json` before removing the original. This describes a conditional code path, not an existing receipt or remote result.
- The submitter propagates `WORLDMM_CONTINUE_RECEIPT_KEY_ID` and the path in `WORLDMM_CONTINUE_RECEIPT_SIGNING_KEY` only to the gate allocation. The gate opens the operator-owned regular mode-`0600` key with `O_NOFOLLOW`; missing or unsafe signing material prevents a Go receipt. This is locally implemented behavior, not evidence that a key, receipt, or Phase-A run exists.
- Raw RGB student/device architecture and matched official E1/E2/E3 remain outside this experiment.

Company datasets, frames, weights, checkpoints, teacher caches, and provider outputs remain company-side. Only explicitly approved lightweight reports, metrics, receipts, manifests, and redacted logs/plots may be copied back.

[프로젝트 홈으로 돌아가기](README.md)
