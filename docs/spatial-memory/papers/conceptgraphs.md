# ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning

| 항목 | 값 |
| --- | --- |
| Page ID | SM-PAPER-CONCEPTGRAPHS |
| 상태 | Primary source 검토 완료; project 미재현 |
| 출판 | IEEE ICRA 2024, pp. 5021-5028 |
| 1차 출처 | [IEEE](https://ieeexplore.ieee.org/document/10610243), [arXiv:2309.16650](https://arxiv.org/abs/2309.16650) |
| 공식 code | [concept-graphs/concept-graphs](https://github.com/concept-graphs/concept-graphs) |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [논문 목록](README.md), [문제 정의](../problem.md), [아키텍처](../architecture.md), [추적성](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-002 |

## 핵심 결론

**프로젝트 추론:** object별 explicit node와 compact entity-level descriptor는 dense per-point semantics보다 spatial memory의 검색·관계 추론 단위에 적합하다. 따라서 typed object record, persistent instance ID, compact evidence reference 설계의 근거로 사용한다.

ConceptGraphs가 relation을 mapping time에 생성한다는 사실은 우리 프로젝트의 query-time deterministic relation 계산을 직접 검증하지 않는다. 여기서는 object-centric representation의 가능성만 채택한다.

## 근거 상태

**프로젝트 결과:** 미재현. 이 저장소에서 ConceptGraphs 모델·데이터·공식 benchmark를 실행하지 않았다. 현재 프로젝트의 typed object schema는 개념적 연결일 뿐 논문 재현 결과가 아니다.

## 논문 핵심

**논문 주장:** posed RGB-D observations에서 여러 view의 object를 associate하고, object-level geometry와 open-vocabulary semantics 및 관계를 갖는 3D scene graph를 만든다. Dense point별 semantic feature 대신 object 중심 graph를 사용해 language query와 robot planning을 지원한다.

## 근거

**논문 보고 결과:**

- Replica human evaluation에서 기본 구성의 평균 node-label precision은 `0.71`, edge-relation precision은 `0.88`로 보고됐다. 출처: Table I.
- Replica zero-shot semantic segmentation에서 `mAcc 40.63`, `F-mIoU 35.95`를 보고했다. 비교한 ConceptFusion은 각각 `24.16`, `31.31`이었다. ConceptFusion+SAM의 `F-mIoU 38.70`보다는 낮았다. 출처: Table II.
- Open-set text retrieval, landmark-based relocalization, map update, navigation·manipulation planning을 robot demonstration으로 제시했다.

이 수치는 논문의 posed RGB-D 실험 조건에 한정된다.

## 판단 한계

- 1 Hz monocular RGB와 IMU/VIO 조건의 geometry 안정성
- 장기간 반복 방문, temporal event, validity interval 처리
- 실제 serialized byte hard budget 또는 value-per-byte writer
- 미래 질문을 모르는 상태의 QA-sufficient write policy
- uncertainty·provenance가 포함된 deterministic geometry proof
- SuperMemory-VQA 성능이나 AI-glass on-device 성능

논문도 small/thin object 누락, duplicate object, caption 오류, language-model 비용, temporal dynamics 부재를 한계로 다룬다.

## 문제 배경

Dense 3D open-vocabulary map은 point마다 큰 feature를 저장하고, object 간 관계를 직접 표현하지 못한다. ConceptGraphs는 별도 3D 학습이나 fine-tuning 없이 compact하고 queryable한 object-centric representation을 만드는 문제를 다룬다.

## 관련 방법

- Posed RGB-D frame에서 2D instance mask와 visual-language feature를 추출한다.
- Mask를 3D point cloud로 lift하고 geometry와 semantics를 이용해 view 간 동일 object를 associate·fuse한다.
- Fused object마다 geometry, multi-view semantic descriptor, caption을 갖는 node를 만든다.
- Vision-language model이 object caption을 만들고 language model이 object 간 relation edge를 생성한다.
- Graph node의 CLIP feature와 caption을 language retrieval, relocalization, planning에 사용한다.

## 참고문헌

- [공식 project page](https://concept-graphs.github.io/)
- [IEEE publication](https://ieeexplore.ieee.org/document/10610243)
- [arXiv paper](https://arxiv.org/abs/2309.16650)
- [공식 code](https://github.com/concept-graphs/concept-graphs)
- [상위 논문 목록](README.md)
