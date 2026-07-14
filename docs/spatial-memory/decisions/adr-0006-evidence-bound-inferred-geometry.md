# ADR-0006: Evidence-bound inferred geometry만 proof에 허용

| 항목 | 값 |
|---|---|
| Page ID | SM-ADR-0006 |
| Confluence parent | SM-DECISIONS |
| ADR | ADR-0006 |
| 상태 | 채택; local contract 검증 완료 |
| 결정일 | 2026-07-14 |
| 추적성 | [Provenance와 geometry QA 요구사항](../traceability.md) |
| 아키텍처 | [Typed record와 proof 경계](../architecture.md) |

## 핵심 결론

- **결정:** `model_inferred` record도 selected sensor-frame evidence, causal
  validity, coordinate frame, uncertainty, confidence, artifact lineage를 모두
  만족하면 deterministic proof 입력으로 사용할 수 있다.
- **구분:** Provenance는 계속 `model_inferred`로 남는다. `observed` 또는
  `human_confirmed`로 승격하지 않는다. `relation_inferred`는 metric proof 입력으로
  허용하지 않는다.
- **다음 gate:** EXP-0005에서 confidence calibration과 abstention error를 측정한다.
  현재 default threshold 0.5는 contract default이지 benchmark로 검증된 값이 아니다.

## 근거와 판단 이유

- Offline teacher와 device perception의 geometry는 본질적으로 inferred다. 이를
  모두 거부하면 learned compiler가 유효한 record를 생성해도 downstream geometry
  QA가 항상 abstain한다.
- 반대로 memory ID 자기참조만으로 inference를 grounded evidence로 취급하면 proof가
  원관측까지 추적되지 않는다.
- Deterministic executor는 provenance 자체가 아니라 evidence, uncertainty,
  confidence, completeness 조건을 함께 검사해야 한다.

## 구현 방향

- Contextual typed artifact validation은 `model_inferred`에도 selected same-video
  `evidence_refs`와 observation interval/count 일치를 요구한다.
- Geometry executor는 memory record ID와 observation evidence를 분리한다. Record ID만
  다시 evidence로 적은 자기참조도 거부한다. Record ID는 proof audit reference에는
  포함되지만 grounding evidence를 대신하지 못한다.
- Inferred fact는 query의 `min_inferred_confidence` 이상이어야 한다. Metric operation은
  기존 coordinate-frame과 uncertainty limit도 그대로 적용한다. Proof는 사용된
  fact의 최소 confidence를 노출하고 stable hash에 포함한다.
- Object record의 optional `place_label`과 `last_location` operator는 complete entity
  index와 grounded evidence가 있을 때만 semantic place를 반환한다.
- QA answer-choice validation은 `last_location` proof value와 정확히 한 place choice가
  일치해야 한다. Time과 location을 동시에 요구하는 단일 query는 임의로 하나를
  선택하지 않고 abstain한다.

## 검증 결과와 남은 과제

- Tiny local tests는 missing frame evidence, low confidence, missing place label,
  incomplete entity index, stale transition event에서 abstain함을 검사한다.
- Evidence-bound teacher target과 place projection, proof-to-choice contradiction
  검사도 local unit test로 확인했다.
- Real model confidence calibration, semantic place ontology, false association,
  benchmark selective risk는 미검증이다.

## 배경

ADR-0004의 초기 executor whitelist는 observed/object-geometry/fused/human provenance만
grounded로 취급했다. 이는 teacher/student lane을 안전하게 차단했지만 실제 inferred
record를 proof까지 연결하지 못했다.

## 대안

| 대안 | 미선택 이유 |
|---|---|
| 모든 inferred geometry 거부 | Learned compiler가 downstream QA 효용을 만들 수 없음 |
| Provenance 문자열만 보고 inferred geometry 허용 | Evidence, confidence, causality 없는 hallucinated geometry도 통과함 |
| LLM이 inferred snippet을 직접 판단 | Deterministic proof와 choice consistency를 우회함 |
| Memory ID를 evidence로 간주 | 원관측까지 추적되지 않는 자기참조가 됨 |

## Trade-off 검토

- Teacher/student record가 proof에 기여하지만 검증 경로가 더 엄격해진다.
- Confidence threshold와 place ontology는 dataset/hardware별 calibration이 필요하다.
- 조건 하나라도 빠지면 answer를 추정하지 않고 abstain한다.

## 대체 이력

[ADR-0004](adr-0004-deterministic-geometry-proof.md)의 grounded-provenance whitelist
조항을 2026-07-14부터 이 ADR이 대체한다. Deterministic proof와 QA trust-boundary
결정은 유지한다.
