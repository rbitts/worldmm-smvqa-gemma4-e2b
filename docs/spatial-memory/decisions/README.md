# Spatial Memory 아키텍처 결정

| 항목 | 값 |
|---|---|
| Page ID | SM-DECISIONS |
| 상태 | 활성 decision index |
| 최종 갱신 | 2026-07-14 |
| Confluence parent | SM-ROOT |

## 핵심 요약

Architecture decision 6개가 채택됐다. G-CUT3R는 offline teacher로 제한하고,
student 투자 전에 teacher-oracle utility를 검증한다. Device compiler는 native
geometry와 작은 semantic perception을 결합하며 evidence-bound inference만 proof에
들어간다.

| ID | 결정 | 근거 상태 | 다음 실행 |
|---|---|---|---|
| [ADR-0001](adr-0001-explicit-typed-memory.md) | Explicit typed record persist | 일부 구현 | Checkpoint-produced record 검증 |
| [ADR-0002](adr-0002-gcut3r-as-teacher.md) | G-CUT3R-compatible external teacher 사용 | 일부 구현 | Pinned provider/cache 검증 |
| [ADR-0003](adr-0003-value-per-byte-writer.md) | Actual byte당 value로 선택 | 일부 구현 | Hard budget에서 learned writer 측정 |
| [ADR-0004](adr-0004-deterministic-geometry-proof.md) | Deterministic proof 요구 | Local verified | Real typed evidence에서 재검증 |
| [ADR-0005](adr-0005-hybrid-on-device-compiler.md) | Hybrid device compiler, teacher는 offline only | 일부 구현 | EXP-0005 oracle gate |
| [ADR-0006](adr-0006-evidence-bound-inferred-geometry.md) | Evidence/confidence가 있는 inferred geometry만 proof 허용 | Local verified | Confidence calibration |

## 의사결정 관리 원칙

- ADR은 decision, rationale, execution state, trade-off를 기록하며 benchmark
  result는 기록하지 않는다.
- Paper claim, project inference, project measurement를 분리한다.
- Decision 변경은 새 ADR을 만들고 기존 ADR의 대체 범위를 명시한다.
- Experiment page만 evidence level을 local sanity에서 benchmark verified로 올릴
  수 있다.
- 각 ADR은 [추적성](../traceability.md)에서 claim·experiment와 연결한다.

새 ADR은 `docs/spatial-memory/decisions/TEMPLATE.md`를 사용한다.

[프로젝트 홈으로 돌아가기](../README.md)
