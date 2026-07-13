# FOUND-IT: Foundation-model-first Task-driven 3D Scene Graphs with Granularity on Demand

| 항목 | 값 |
| --- | --- |
| Page ID | SM-PAPER-FOUND-IT |
| 상태 | Primary source 검토 완료; official code 부재로 reproduction 차단 |
| 출판 | arXiv preprint 2605.25371v2, 2026 |
| 1차 출처 | [arXiv:2605.25371v2](https://arxiv.org/abs/2605.25371), [official project](https://nicolasgorlo.com/FOUND-IT_26/) |
| 공식 code | 2026-07-11 기준 미공개; paper에 publication 이후 공개 예정으로 표기 |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [논문 목록](README.md), [문제 정의](../problem.md), [아키텍처](../architecture.md), [추적성](../traceability.md) |
| 프로젝트 claim | [추적성](../traceability.md): C-004 |

## 핵심 결론

**프로젝트 추론:** evidence reservoir와 explicit cache를 분리하고, 모든 possible object를 write time에 materialize하지 않는 `granularity on demand`의 가능성을 보여준다. Dense occupancy 대신 sparse place tile을 직접 만드는 구조도 typed free-space record에 참고할 수 있다.

## 근거 상태

**프로젝트 결과:** 미재현; official code 사용 불가. FOUND-IT 구현과 benchmark를 실행하지 않았다. 공개된 VGGT-SLAM은 기반 component일 뿐 FOUND-IT 전체 코드가 아니다.

## 논문 핵심

**논문 주장:** uncalibrated monocular video의 visual memory를 먼저 유지하고 task query가 생겼을 때 필요한 object·region granularity만 explicit 3D scene graph로 materialize하면, 고정 task list 없이 task-driven mapping을 지원할 수 있다.

## 근거

**논문 보고 결과:**

- Clio open-set object extraction에서 Cubicle `osR 0.89 / IoU 0.37`, Apartment `0.66 / 0.28`, Office `0.70 / 0.30`을 보고했다. 출처: Table I.
- ASHiTA SG3D에서 sub-task accuracy `39.7%`, full-task accuracy `19.0%`를 보고했다. DAAAM 비교값 `22.2%`, `11.2%`에 대한 sub-task 상대 향상이 abstract의 약 `79%`다. 출처: Table II.
- HOV-SG region benchmark에서 precision `85.34`, recall `83.72`, semantic accuracy `78.05`를 보고했다. 출처: Table III.
- RTX 3090의 VGGT pipeline은 약 `6 FPS`; lighter configuration의 Spot+Jetson Thor full scene graph demonstration은 약 `4 Hz`로 보고했다. 출처: Table IV와 Section IV-G.

Semantic extraction 정량 비교는 모든 method에 benchmark depth와 ground-truth pose를 제공한 조건이다.

## 판단 한계

- 미래 질문이 오기 전 원본 frame을 삭제해도 되는 lifelong compression
- Write-time QA utility와 hard byte budget
- Dynamic object·change event·validity history 미평가
- 1 Hz sparse sensing 성능
- SuperMemory-VQA와 deterministic geometry proof
- 실제 memory bytes, growth curve, byte Pareto

FOUND-IT은 keyframe과 geometry evidence를 계속 보존하므로 그 자체가 lifelong storage 해법은 아니다. Geometry mapping 실패가 downstream graph에 전파되고 floor가 보이지 않으면 traversability가 약해질 수 있다.

## 문제 배경

기존 scene graph는 mapping time에 object와 region granularity를 확정한다. 미래 task가 바뀌면 너무 거친 object를 다시 분해하거나 불필요한 detail을 계속 유지해야 한다. FOUND-IT은 granularity를 query time으로 늦추고 monocular geometric foundation model을 사용한다.

## 관련 방법

- VGGT-SLAM 계열 frontend가 uncalibrated monocular keyframe에서 geometry, confidence, pose, intrinsics를 추정하고 submap을 정렬한다.
- Visual memory는 keyframe embedding을 보존하고, cache memory는 query 후 추출된 point cloud와 3D bounding box만 유지한다.
- Text-keyframe retrieval 뒤 segmentation과 back-projection을 수행해 requested object를 explicit cache에 추가한다.
- Geometric foundation model의 intermediate token에 ground head를 붙여 sparse traversable place graph를 만든다.
- Query-time region clustering과 tool-calling agent로 task 관련 graph를 점진적으로 구성한다.

## 참고문헌

- [arXiv paper, canonical v2 metadata](https://arxiv.org/abs/2605.25371)
- [공식 project page](https://nicolasgorlo.com/FOUND-IT_26/)
- [Public base project: VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM)
- [상위 논문 목록](README.md)
