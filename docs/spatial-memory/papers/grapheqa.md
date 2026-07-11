# GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied Question Answering

| Field | Value |
| --- | --- |
| Page ID | SM-PAPER-GRAPHEQA |
| Status | Reviewed from primary sources; not reproduced in this project |
| Publication | CoRL 2025, PMLR 305:2714-2742 |
| Primary source | [PMLR](https://proceedings.mlr.press/v305/saxena25a.html), [arXiv:2412.14480](https://arxiv.org/abs/2412.14480) |
| Official code | [SaumyaSaxena/graph_eqa](https://github.com/SaumyaSaxena/graph_eqa) |
| Last checked | 2026-07-11 |
| Project links | [Papers index](README.md), [Problem](../problem.md), [Architecture](../architecture.md), [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-003, C-004 |

## 30-second summary

**Paper claim:** real-time metric-semantic 3D scene graph와 소수의 task-relevant image를 함께 VLM memory로 사용하면, unseen environment의 embodied QA에서 graph-only 또는 vision-only memory보다 효율적인 exploration과 높은 task success를 얻는다.

## Problem addressed

Embodied QA agent는 질문에 답하기 위해 처음 보는 환경을 탐색해야 한다. Frame history만으로는 global structure가 약하고, graph만으로는 색상·미세 외형 같은 visual detail이 빠진다. GraphEQA는 두 memory를 결합하고 hierarchical planning에 연결한다.

## Relevant method

- RGB-D, semantic segmentation, pose·intrinsics로 Hydra 기반 hierarchical 3D scene graph를 online 구축한다.
- Room, object, frontier를 graph에 연결하고 별도 2D occupancy/frontier map을 유지한다.
- 질문과 관련된 image `K=2` 및 current view를 visual memory로 선택한다.
- VLM planner가 질문, scene graph, visual memory, robot state, action history를 받아 answer, confidence, 다음 room/object/frontier action을 생성한다.

## Paper-reported evidence

**Paper-reported results:**

- HM-EQA와 OpenEQA simulation에서 주요 baseline보다 높은 success rate와 적은 planning step을 보고했다. 출처: Table 1.
- 보고된 ablation에서 `GraphEQA-SG`는 success `13.6%`, `GraphEQA-Vis`는 `45.7%`, full GraphEQA는 `63.5%`였다. 출처: Table 2.
- 같은 ablation의 planning steps는 각각 `8.8`, `1.0`, `5.1`, 이동 거리는 `33.0 m`, `3.9 m`, `12.6 m`였다. 출처: Table 2.
- 두 실제 environment에서도 system demonstration을 제시했다. 출처: Figure 4와 Section 4.2.

## What this supports here

**Project inference:** explicit metric-semantic graph와 제한된 visual evidence reservoir는 상보적이다. Geometry core는 distance·direction·topology를, selected crop은 색상·텍스트·미세 상태를 담당하게 하는 설계의 직접적인 선행 근거다.

## What it does not prove

- 질문을 미리 모르는 passive AI-glass episodic memory
- write-time future-QA utility 또는 actual-byte compression
- 1 Hz monocular low-overlap sensing
- RGB-D·known pose 없이 scene graph를 만드는 방법
- lifelong consolidation, 반복 방문, change event retention
- deterministic geometry executor, covariance, provenance
- SuperMemory-VQA 또는 glass-class hardware 성능

GraphEQA의 VLM confidence는 metric geometry uncertainty calibration과 동일하지 않다.

## Project reproduction status

**Project result:** Not reproduced. GraphEQA simulator, robot stack, benchmark를 이 저장소에서 실행하지 않았다. 현재 geometry executor와 evidence pack은 이 논문 결과의 재현물이 아니다.

## References

- [PMLR publication](https://proceedings.mlr.press/v305/saxena25a.html)
- [Official project page](https://saumyasaxena.github.io/grapheqa/)
- [arXiv paper](https://arxiv.org/abs/2412.14480)
- [Official code](https://github.com/SaumyaSaxena/graph_eqa)
- [Parent papers index](README.md)
