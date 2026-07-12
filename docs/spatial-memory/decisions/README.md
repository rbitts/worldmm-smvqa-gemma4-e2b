# Spatial Memory Architecture Decisions

| Field | Value |
|---|---|
| Page ID | SM-DECISIONS |
| Status | Active decision index |
| Last updated | 2026-07-11 |
| Confluence parent | SM-ROOT |

이 디렉터리는 spatial-memory 설계 결정을 기록한다. 각 ADR은 한 가지 결정을
문제, 근거, 구현, 검증 결과와 연결한다. Confluence로 옮길 때 이 페이지를
부모 페이지로, 각 ADR을 하위 페이지로 유지한다.

| ID | 결정 | 상태 |
|---|---|---|
| [ADR-0001](adr-0001-explicit-typed-memory.md) | 영구 메모리를 explicit typed record로 저장 | Accepted; partially implemented |
| [ADR-0002](adr-0002-gcut3r-as-teacher.md) | G-CUT3R를 외부 teacher로 사용 | Accepted; partially implemented |
| [ADR-0003](adr-0003-value-per-byte-writer.md) | actual-byte 기준 value-per-byte writer 사용 | Accepted; partially implemented |
| [ADR-0004](adr-0004-deterministic-geometry-proof.md) | geometry 답변을 deterministic proof로 제한 | Accepted; locally verified |

## 읽는 순서

1. [추적표](../traceability.md)에서 해결하려는 문제를 찾는다.
2. 해당 ADR에서 결정과 대안을 확인한다.
3. Implementation 링크로 현재 코드 상태를 확인한다.
4. Verification에서 검증 범위와 미검증 범위를 구분한다.

## 상태 규칙

| 상태 | 의미 |
|---|---|
| Proposed | 결정 전 검토 중 |
| Accepted design | 설계 결정은 채택했지만 구현되지 않음 |
| Accepted; partially implemented | 일부 경로만 구현 또는 검증됨 |
| Accepted; locally verified | 로컬 fixture와 단위 검증 완료; 실제 benchmark 재현은 별도 |
| Superseded | 새 ADR로 대체됨 |

상태가 바뀌면 기존 결론을 덮어쓰지 않는다. 결정 변경은 새 ADR을 만들고 기존
ADR의 Supersession에 링크한다. 실제 benchmark 결과는 ADR이 아니라 experiment
페이지에 기록한다.

## 새 ADR 작성

Repository `docs/spatial-memory/decisions/TEMPLATE.md`를 복사한다. 논문이 보고한 결과, 프로젝트의 설계
추론, 프로젝트에서 측정한 결과를 서로 다른 문장과 링크로 구분한다.
