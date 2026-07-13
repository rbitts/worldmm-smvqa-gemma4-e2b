# HOV-SG: Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation

| 항목 | 값 |
| --- | --- |
| Page ID | SM-PAPER-HOV-SG |
| 상태 | Primary source 검토 완료; project 미재현 |
| 출판 | Robotics: Science and Systems XX, 2024 |
| 1차 출처 | [RSS proceedings](https://www.roboticsproceedings.org/rss20/p077.html), [arXiv:2403.17846](https://arxiv.org/abs/2403.17846) |
| 공식 code | [hovsg/HOV-SG](https://github.com/hovsg/HOV-SG) |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [논문 목록](README.md), [문제 정의](../problem.md), [아키텍처](../architecture.md), [추적성](../traceability.md) |
| 프로젝트 claim | [추적성](../traceability.md): C-002 |

## 핵심 결론

**프로젝트 추론:** place/submap 아래 room, portal, object를 local hierarchy로 보존하면 dense global feature map보다 compact하게 검색 범위를 제한할 수 있다. Entity별 compact semantic prototype과 floor-room-object containment edge를 저장하는 설계 근거다.

## 근거 상태

**프로젝트 결과:** 미재현. HOV-SG 데이터와 공식 pipeline을 실행하지 않았다. 이 프로젝트의 place/object hierarchy는 선행 설계를 참고한 것이며 논문 결과 재현은 아니다.

## 논문 핵심

**논문 주장:** segment-level open-vocabulary map을 `root → floor → room → object` hierarchy로 바꾸면 dense open-vocabulary map보다 작으면서 multi-floor language navigation에 유용한 representation을 만들 수 있다.

## 근거

**논문 보고 결과:**

- Dense open-vocabulary map인 VLMaps 대비 representation size를 평균 약 `75%` 줄였다고 보고했다. 출처: Section IV-D와 Table VII.
- HM3DSem 8개 scene에서 floor count `100%`, region precision `84.10%`, recall `83.59%`를 보고했다. 출처: Table II.
- Object-room-floor retrieval success는 `28.00%`; privileged floor 정보를 받은 ConceptGraphs 비교값은 `16.31%`였다. 출처: Table V.
- Object-room retrieval은 `31.48%`; 비교값은 `29.26%`였다. 출처: Table V.
- 공식 프로젝트는 41개 real-world multi-floor trial에서 약 `55%` success를 보고한다.

## 판단 한계

- 1 Hz monocular sensing이나 uncertain VIO에서의 map quality
- Online learned typed record generation 미지원
- lifelong revisit consolidation과 dynamic object history
- QA utility 또는 actual-byte hard budget 기반 writer
- Geometry-grounded QA proof, uncertainty, provenance 미지원
- SuperMemory-VQA 성능

논문은 posed RGB-D를 요구하고, static environment를 가정하며, many hyperparameters와 open-plan room segmentation 한계를 가진다. Mapping 자체도 glass-class real-time 조건을 입증하지 않는다.

## 문제 배경

Dense open-vocabulary feature map은 큰 건물에서 저장과 검색 비용이 높고, floor·room처럼 object보다 높은 abstraction을 직접 표현하기 어렵다. HOV-SG는 large-scale, multi-story environment를 계층적으로 검색하고 navigation에 쓰는 문제를 다룬다.

## 관련 방법

- Posed RGB-D와 odometry에서 mask와 visual-language feature를 만들고 3D segment로 fuse한다.
- Height histogram으로 floor를, BEV와 Watershed로 room을 구분한다.
- Object를 room에 associate하여 `floor-room-object` graph를 구성한다.
- 각 계층에 open-vocabulary feature를 저장한다.
- Cross-floor Voronoi graph와 language instruction 분해를 사용해 floor, room, object 순으로 navigation target을 좁힌다.

## 참고문헌

- [RSS publication](https://www.roboticsproceedings.org/rss20/p077.html)
- [공식 project page](https://hovsg.github.io/)
- [arXiv paper](https://arxiv.org/abs/2403.17846)
- [공식 code](https://github.com/hovsg/HOV-SG)
- [상위 논문 목록](README.md)
