# ADR-0004: Geometry 답변에 deterministic proof 요구

| 항목 | 값 |
|---|---|
| Page ID | SM-ADR-0004 |
| ADR | ADR-0004 |
| 프로젝트 claim | C-003, C-009 |
| 상태 | 채택; 로컬 검증 완료 |
| 결정일 | 2026-07-11 |
| 추적성 | [Geometry-grounded QA와 auditability 요구사항](../traceability.md) |
| 아키텍처 | [Geometry executor-QA 경계](../architecture.md) |

## 핵심 결론

지원되는 geometry operation은 deterministic executor가 explicit record에서
계산한다. Executor는 answerable value 또는 이유가 있는 abstention을 반환한다.
Proof에는 operation, subject/object entity role, coordinate frame, uncertainty와
unit, provenance, evidence references, stable proof ID를 포함한다.

QA model에는 raw spatial geometry를 주지 않고 검증된 proof만 geometry fact로
제공한다. Geometry 질문에 answerable proof가 있으면 model output은 proof ID를
인용해야 하며, machine-decodable choice가 proof value와 모순되면 prediction을
거부한다.

## 근거와 판단 이유

- [ConceptGraphs](../papers/conceptgraphs.md)는 explicit object relation과 queryable
  scene representation의 관련 근거다.
- [OpenEQA](../papers/openeqa.md)는 embodied/episodic question answering 평가 맥락을
  제공한다. 이 프로젝트의 proof contract와 executor는 repository 설계이며 해당
  논문의 직접 결과로 주장하지 않는다.
- `src/worldmm_smvqa/worldmm/geometry_executor.py`는
  deterministic operations와 proof hash를 구현한다.
- `src/worldmm_smvqa/qa_prompt.py`는 spatial payload를 숨기고
  proof-only geometry prompt를 구성한다.
- `src/worldmm_smvqa/qa.py`는 proof ID와 selected choice의 일관성을
  검증한다.

## 구현 방향

- 지원 operation은 `distance`, `near`, `relative_direction`, `last_seen`, `count`다.
- Metric pair operation은 grounded provenance, 동일 coordinate frame, 완전한 XYZ,
  uncertainty limit를 요구한다.
- `near` threshold와 direction boundary가 uncertainty interval과 겹치면 abstain한다.
- `count`와 `last_seen`은 complete entity index certificate가 없으면 abstain한다.
- 같은 certificate가 label uniqueness를 보장하지 않으면 pair operation도
  question에 explicit entity ID가 있어야 한다. Retrieved top-k의 유일 label은
  uniqueness 근거가 아니다.
- Direction은 동일 spatial frame의 trusted causal wearer pose가 없으면 abstain한다.
  Source JSON은 `yaw_degrees`와
  `pose_covariance_xyz_m_rpy_deg=[x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg]`
  row-major 6x6를 사용하며 covariance index 35는 degree²다. Yaw 0°는 +Y,
  positive yaw는 +X 방향이고 yaw 0°에서 +X가 wearer-right다. Production
  proof는 `(source=imu, processing_mode=raw)` 또는 `(source=vio,
  processing_mode=online_causal)`인 pose와 `timestamp <= observed_through_time <=
  question_time` certificate를 요구한다. Offline SLAM, ground-truth/model pose,
  누락·미래 certificate는 거부한다.
- Proof hash는 query parameters와 result, entity role, provenance, evidence를 포함한
  canonical payload에서 생성한다.
- QA trust boundary는 unknown, duplicate, unanswerable proof ID와 choice contradiction을
  거부한다.

## 검증 결과와 남은 과제

- `tests/test_geometry_executor.py`는 지원 operation과
  proof contents를 검사한다.
- `tests/test_geometry_executor_safety.py`는
  frame mismatch, uncertainty, ambiguity, conflicting state, completeness abstention을
  검사한다.
- `tests/test_qa_prompt.py`는 raw spatial payload가 prompt에
  노출되지 않는지 검사한다.
- `tests/test_qa_trust_boundaries.py`는 proof citation,
  support ID와 answer-choice consistency를 검사한다.
- 로컬 tiny fixture에서 causal proof path를 검사했다. Official benchmark 결과나
  unrestricted natural-language geometry parser의 정확도를 주장하지 않는다.

## 배경

LLM이 spatial snippet이나 latent feature만 보고 거리, 방향, 개수, last-seen 값을
생성하면 답을 좌표와 관측 근거로 검증하기 어렵다. Coordinate frame, uncertainty,
entity role, temporal completeness가 빠지면 문법적으로 그럴듯한 답도 잘못될 수
있다. Retrieved payload 자체를 prompt에 그대로 넣으면 untrusted spatial text가
geometry fact처럼 사용될 위험도 있다.

## 대안

| 대안 | 미선택 이유 |
|---|---|
| LLM이 snippet에서 geometry를 직접 추론 | 좌표계, 수치, 근거를 재현 가능하게 검증하기 어려움 |
| Raw geometry dictionary를 prompt에 전달 | unvalidated payload가 fact 또는 instruction처럼 사용될 수 있음 |
| 모든 spatial relation을 graph edge로 미리 저장 | pairwise growth가 크고 좌표에서 계산 가능한 관계를 중복 저장함 |
| 불확실성을 무시한 point estimate | near threshold와 direction boundary에서 과도한 확신을 생성함 |
| 근거가 부족해도 best-effort answer 생성 | geometry-grounded 계약을 위반함; abstention이 더 안전함 |

## Trade-off 검토

- Geometry answer가 entity, frame, uncertainty, provenance, evidence로 감사 가능하다.
- 지원되지 않거나 모호한 질문은 답변하지 않는다. Coverage보다 correctness와
  traceability를 우선한다.
- LLM은 question interpretation과 answer rendering을 담당하고 metric computation은
  executor가 담당한다.
- Allocentric/egocentric 구분, interval query, reachability, visibility, support와
  containment는 후속 operator와 별도 검증이 필요하다.
- Complete-index certificate의 생성 주체와 scope를 persistent memory pipeline에서
  명시해야 한다.

## 대체 이력

없음.
