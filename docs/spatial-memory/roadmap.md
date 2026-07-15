# Spatial memory roadmap

| 항목 | 값 |
| --- | --- |
| Page ID | SM-ROADMAP |
| 최종 갱신 | 2026-07-14 |
| 우선순위 원칙 | student 투자 전에 bounded teacher-oracle QA utility를 검증 |

## P0: EXP-0004 → EXP-0005

The canonical goal is EXP-0005 using the `teacher-oracle` profile. Only local implementation and tests exist. `launch-remote --dry-run`, `audit-sensors`, and `validate-teacher-oracle-inputs` are prepared contracts; no remote command has been executed and no company evidence exists.

```text
tracked source at approved SHA, deployed by `git archive` with verified remote content manifest; generated plan transferred separately
  -> bootstrap-only remote CPU `sensor-audit-v1` and strict teacher-only validation receipt staged outside run root
  -> review then atomically create immutable final `.env.worldmm`
  -> cryptographic Phase-A approval and provider-gate Phase A
  -> signed/sealed conditional continue receipt or terminal no-go receipt
  -> receipt-bound cryptographic Phase-B approval and bounded E0/T0/T1 Phase B
```

P0 is blocked until company-only inputs are reviewed: selected RGB/calibration/causal pose/VIO/depth coverage, production sensor manifest and its generated digest, provider/checkpoint/config and semantic provider/ontology, prepared split/digests, Slurm accounting capability, signer registry/approvals, and unique run root. No remote run has occurred.

| Gate | Required truth | Failure outcome |
| --- | --- | --- |
| CPU preflight | Remote CPU run actually produces `sensor-audit-v1` and strict teacher-only validation receipt, binding audit/config digests; the approved `SMVQA_FRAME_ROOT` is retained | No Phase A |
| Producer boundary | Closed allowlisted schema rejects question/choice/answer/label/evidence aliases | No Phase A |
| Provider contract | Strict shard decode, exact causal request coverage, and signed provider/checkpoint/config/source/split lineage | No continue receipt |
| Operational Phase A | `--no-requeue`; exactly one duplicate-visible, cluster-qualified, nontruncated `State%64` accounting allocation plus marker/artifacts | `no_go`; new attempt only |
| Phase-B admission | Cryptographically verified signed/sealed continue receipt and separate receipt-bound Phase-B approval | No E0/T0/T1 submission |
| Scientific result | Frozen per-variant byte/frame/metric/risk evidence, byte-matched caps for E0/T0/T1 | Teacher-oracle Go/No-Go only |

`summary/dag_jobs.teacher_oracle.env`, `summary/teacher_oracle_continue.json`, and `summary/teacher_oracle_terminal.json` are Phase-A conditional artifacts; `summary/dag_jobs.phase_b.env` is conditional on Phase-B admission. Fake test receipts/job IDs demonstrate local behavior only. The terminal operational record is distinct from a scientific conclusion. A provider operational Go is not QA improvement. Equal byte caps are not equal realized bytes.

## Priority after P0

| Priority | Work | Admission evidence |
| --- | --- | --- |
| P1 | Add teacher-backed spatial operators only after P0 | Reviewed teacher-oracle utility and risk evidence |
| P2 | Student lane design | Separate proposal, approval, and matched evaluation plan |
| P3 | Existing/`NEW` association and change memory | Held-out association utility and error targets |
| P4 | Matched official E1/E2/E3 and byte Pareto | Immutable matched identities and separate approval |
| P5 | Additional spatial types/operators | Demonstrated held-out byte value after object/location |

`probe` and `full` remain deferred legacy lanes. `probe` emits `contract_probe`/`PROBE`; `full` emits `student`/`E1`; neither can satisfy P0. No official claim is available. Company datasets, weights, checkpoints, frames, and teacher caches remain company-side; only approved lightweight review artifacts may be copied back.

운영 절차는 repository `HANDOFF.md`를 따른다.
