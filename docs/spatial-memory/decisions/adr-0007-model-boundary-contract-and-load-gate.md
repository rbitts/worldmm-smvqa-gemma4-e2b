# ADR-0007: Model boundary contract and remote load gate

| 항목 | 값 |
|---|---|
| Page ID | SM-ADR-0007 |
| Confluence parent | SM-DECISIONS |
| ADR | ADR-0007 |
| 상태 | 구현 채택; provider lock은 승인 대기 |
| 결정일 | 2026-07-15 |
| 추적성 | [C-013](../traceability.md) |
| 아키텍처 | [Spatial-memory 아키텍처](../architecture.md) |

## 핵심 결론

- **결정:** 하나의 immutable model-free boundary fixture와 student architecture를 local mock, checkpoint, retrieval/QA lineage, report, remote load receipt의 공통 신뢰 원점으로 사용한다.
- **범위:** Local은 production consumer를 사용하는 CPU/mock contract 검증만 수행한다. 실제 model load, forward, training, evaluation은 별도 승인된 company operation이다.
- **다음 gate:** 승인된 provider lock과 모든 물리 rank의 유효한 load consensus가 없으면 remote student workload와 optimizer construction을 허용하지 않는다.

## 근거와 판단 이유

기존 경계는 개별적으로 존재하지만 local smoke가 G-CUT3R teacher→student lineage와 Qwen/spatial retrieval fan-in→Gemma topology를 하나의 검토된 digest로 결속하지 않았다. Config/processor preflight도 실제 weight loadability를 증명하지 않는다. Contract fixture와 실제 물리 rank별 load receipt를 분리하면 local wiring claim과 remote loadability claim을 혼동하지 않고 fail closed할 수 있다.

## 구현 방향

- `model_contract.py`가 declared-order boundary/fixture digest를 소유한다.
- `mock_dag.py`는 network, GPU, production data/weights 없이 production-owned consumer wiring을 검사한다.
- Checkpoint v2는 local mock authorization과 remote consensus authorization을 구분한다.
- Student `EvidenceLineage`, QA resume manifest, `StudentRunManifest`는 model contract, student architecture, consensus digests를 전파한다.
- Remote student graph는 accepted provider lock과 physical all-rank consensus 뒤에만 workload release를 허용한다.
- EXP-0005 teacher-oracle renderer와 생성 artifact bytes는 변경하지 않는다.

## 검증 결과와 남은 과제

Local verification은 tiny/mock pytest, static checks, teacher-oracle byte regression에 한정한다. 이것은 real forward compatibility, model quality, company GPU availability를 입증하지 않는다. Provider conformance, load probe, training은 각각 별도 승인과 unique run ID가 필요하다.

## 대안

| 대안 | 미선택 이유 |
|---|---|
| Training process에서 네 모델을 inline load | 실패 진단과 lifecycle 격리가 약하고 optimizer 이전 물리 합의가 불명확하다. |
| Rank 0 또는 role-partitioned load check | 승인된 10×8 또는 pinned 1×1 물리 matrix 전체를 증명하지 못한다. |
| 네 모델의 직렬 mock chain | 실제 teacher/student lineage와 retrieval fan-in topology를 왜곡한다. |

## Trade-off 검토

엄격한 schema, digest propagation, receipt/state-machine 검증 비용이 늘어난다. 대신 stale/mismatched artifact가 QA 또는 completion claim으로 승격되는 것을 막고 local mock/contract-probe 결과가 student/official 결과로 오인되지 않게 한다.

## 대체 이력

없음.
