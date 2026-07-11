# Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction and Optimization

| Field | Value |
| --- | --- |
| Page ID | SM-PAPER-HYDRA |
| Status | Reviewed from primary sources; not reproduced in this project |
| Publication | Robotics: Science and Systems XVIII, 2022 |
| Primary source | [RSS proceedings](https://www.roboticsproceedings.org/rss18/p050.html), [arXiv:2201.13360](https://arxiv.org/abs/2201.13360) |
| Official code | [MIT-SPARK/Hydra](https://github.com/MIT-SPARK/Hydra) |
| Last checked | 2026-07-11 |
| Project links | [Papers index](README.md), [Problem](../problem.md), [Architecture](../architecture.md), [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-006 |

## 30-second summary

**Paper claim:** geometry, object, place, room, building을 layered 3D scene graph로 online 구축하고, hierarchical loop-closure descriptor와 graph optimization으로 전체 계층을 함께 보정할 수 있다.

## Problem addressed

3D Dynamic Scene Graph는 robot의 explicit world model로 유용하지만, sensor stream에서 여러 계층을 real-time으로 만들고 drift·loop closure를 일관되게 반영하기 어렵다. Hydra는 incremental construction과 global correction을 함께 다룬다.

## Relevant method

- Local ESDF active window에서 mesh와 generalized Voronoi diagram 기반 place graph를 증분 구축한다.
- Place graph를 community-detection 방식으로 room에 clustering한다.
- Appearance, object, place statistics를 결합한 hierarchical loop-closure descriptor를 사용한다.
- Embedded deformation graph로 mesh, place, object, room layer를 함께 보정한다.
- 빠른 frontend와 느린 global backend를 병렬 구조로 분리한다.

## Paper-reported evidence

**Paper-reported results:**

- Simulated·real data에서 online으로 동작하면서 batch offline method와 유사한 reconstruction accuracy를 보고했다. 출처: Section VI-B.
- Ground-truth trajectory 조건에서 object의 `80-100%`를 found/correct로 보고했고, place 위치 오차는 `25 cm` 미만이었다. 출처: Figure 7.
- Scene-graph loop closure는 permissive visual baseline보다 `10 cm, 1°` 이내의 closure를 약 2배 검출했다고 보고했다. 출처: Figure 10과 Section VI-B.
- Batch 방식은 중간 크기 scene의 전체 graph 생성에 40초 이상이 걸린 반면 Hydra frontend cost는 local active window 덕분에 bounded한 동작을 보였다. 출처: Figure 9.

## What this supports here

**Project inference:** pose/tracking working state와 persistent map을 분리하고, current submap만 active하게 유지하며, loop closure 시 개별 object를 모두 다시 쓰지 않고 hierarchy transform을 보정하는 구조의 근거다. Object, place, room 같은 typed explicit records도 Hydra의 layered graph와 정렬된다.

## What it does not prove

- Open-vocabulary semantics 또는 unseen object retrieval
- 1 Hz monocular RGB의 low-overlap robustness
- learned QA-sufficient writer나 serialized byte budget
- long-term dynamic event history와 validity interval
- geometry QA executor와 proof object
- SuperMemory-VQA 또는 AI-glass resource envelope

원 논문은 depth/stereo, semantic segmentation, VIO에 의존한다. Open floor-plan room segmentation, semantic room type, richer relation·affordance, QA 활용은 직접 검증하지 않았다.

## Project reproduction status

**Project result:** Not reproduced. Hydra binary, sensor stack, official benchmark를 실행하지 않았다. 현재 local submap·typed schema는 설계 참조이며 Hydra accuracy 재현 결과가 아니다.

## References

- [RSS publication](https://www.roboticsproceedings.org/rss18/p050.html)
- [arXiv paper](https://arxiv.org/abs/2201.13360)
- [Official code](https://github.com/MIT-SPARK/Hydra)
- [Hydra foundations follow-up, IJRR 2024](https://doi.org/10.1177/02783649241229725)
- [3D Dynamic Scene Graphs, RSS 2020](https://www.roboticsproceedings.org/rss16/p079.html)
- [Parent papers index](README.md)
