# Spatial Memory 실험

| 항목 | 값 |
|---|---|
| Page ID | SM-EXPERIMENTS |
| Confluence parent | SM-ROOT |
| 상태 | 활성 |
| 최종 검토 | 2026-07-13 |

## 핵심 결론

공식 learned-method claim을 뒷받침하는 experiment는 없다. EXP-0001은 local
plumbing만 검증한다. 필요한 evidence 순서는 provider validation, typed bridge
probe, full matched E1/E2/E3 run, byte-Pareto 분석이다.

| ID | Experiment | 근거 | 결정 |
|---|---|---|---|
| [EXP-0001](exp-0001-source-compact-baseline.md) | Source-compact baseline | Tiny synthetic sanity 통과 | Heuristic E0 유지, 품질 claim 금지 |
| [EXP-0004](exp-0004-gcut3r-provider.md) | G-CUT3R provider | Contract check만 완료 | Training evidence 신뢰 전 provider 검증 |
| [EXP-0002](exp-0002-typed-memory-bridge.md) | Typed-memory bridge | Contract 구현, 미실행 | 즉시 승인 대상 probe |
| [EXP-0003](exp-0003-byte-pareto.md) | QA utility versus byte | Design만 완료 | Learned bridge와 matched identity 이후 실행 |

## 근거 사용 기준

- `Local sanity`는 code path만 입증하며 benchmark 성능을 입증하지 않는다.
- Pinned split, digest, checkpoint, run ID를 가진 `Benchmark completed`만 model
  comparison에 사용할 수 있다.
- Leakage, digest mismatch, missing artifact가 있으면 run은 `Invalid`다.
- 완료된 benchmark page는 immutable이며 조건 변경 시 새 EXP ID를 만든다.
- 계획 상태 page에 가짜 run ID, hash, path, metric을 쓰지 않는다.

## 작성·import 원칙

실행이 가까운 experiment만 `docs/spatial-memory/experiments/TEMPLATE.md`로 만든다.
H1 하나, 첫 table metadata, relative link를 유지하고 대형 artifact 대신 path와
digest를 기록한다. 각 `EXP-*` page는 Confluence에서 이 page의 direct child다.

[프로젝트 홈으로 돌아가기](../README.md)
