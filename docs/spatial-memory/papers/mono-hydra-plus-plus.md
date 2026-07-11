# Mono-Hydra++: Real-Time Monocular Scene Graph Construction with Multi-Task Learning for 3D Indoor Mapping

| Field | Value |
| --- | --- |
| Page ID | SM-PAPER-MONO-HYDRA-PLUS-PLUS |
| Status | Reviewed from primary sources; reproduction blocked by unavailable weights |
| Publication | arXiv preprint 2605.17661v1, 2026; submitted to ISPRS Journal |
| Primary source | [arXiv:2605.17661](https://arxiv.org/abs/2605.17661) |
| Official code | [BavanthaU/mono-hydra-pp](https://github.com/BavanthaU/mono-hydra-pp); checkpoints and ONNX exports are not included |
| Last checked | 2026-07-11 |
| Project links | [Papers index](README.md), [Problem](../problem.md), [Architecture](../architecture.md), [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-001, C-006 |

## 30-second summary

**Paper claim:** monocular RGB와 IMU를 multi-task depth·semantic model, visual-inertial odometry, causal pose-aware fusion, Hydra backend에 연결하면 RGB-D 없이도 real-time metric-semantic hierarchy와 3D scene graph를 만들 수 있다.

## Problem addressed

RGB-D와 LiDAR는 lightweight robot에 payload·power 부담이 있다. Monocular scene graph는 scale, drift, depth, semantics, dynamics를 동시에 처리해야 한다. Mono-Hydra++는 RGB+IMU만으로 이 pipeline을 구성한다.

## Relevant method

- Frozen DINOv3 backbone과 Mamba decoder 기반 M2H-MX가 metric depth와 semantic label을 예측한다.
- SuperPoint-assisted RVIO2-style frontend가 IMU, visual track, sparse predicted-depth factor를 결합한다.
- Dynamic·unreliable semantic region을 VIO update에서 mask한다.
- Past frame의 depth·semantics를 VIO pose로 current frame에 warp하고 causal short-window fusion한다.
- VIO odometry, pose graph, loop-closure proposal, fused evidence를 Hydra backend로 전달해 mesh와 building-room-place-object graph를 만든다.

## Paper-reported evidence

**Paper-reported results:**

- Selected Go-SLAM ScanNet subset에서 average ATE `6.91 cm`; 비교한 strongest RGB-D baseline보다 `1.6%` 낮다고 보고했다. 출처: Section 4.5.
- Calibrated 7-Scenes에서 average ATE `0.033 m`; strongest calibrated baseline `0.047 m` 대비 `29.8%` 개선을 보고했다. 출처: Section 4.6.
- ScanNet에서 global mIoU `44.96`, Radius F1@0.5m `42.59`, Box F1@0.25 `33.81`을 보고했다. 출처: Section 4.8.
- Jetson Orin NX 16GB에서 TensorRT FP16 M2H-MX-L perception component가 `25.53 FPS`로 동작했다. 출처: Section 4.10.

`25.53 FPS`는 full scene-graph pipeline 속도가 아니라 perception component 결과다. ScanNet 비교도 selected sequence와 서로 다른 sensor setting의 reference라는 저자 주의가 있다.

## What this supports here

**Project inference:** 1 Hz RGB 사이의 low overlap을 high-rate IMU/VIO pose prior로 보완하고, fast pose memory와 persistent graph memory를 분리하는 설계 근거다. Past-only pose warp는 causal short-term geometry stabilization의 직접 사례다.

## What it does not prove

- 1 Hz로 subsample된 RGB 조건의 정확도
- Open-vocabulary language grounding
- Long-term storage compression, learned writer, actual-byte budget
- QA 또는 SuperMemory-VQA 성능
- Small-object evidence와 dynamic event history의 안정적 retention
- 전체 pipeline의 glass-class latency·energy

논문은 object-level semantic preservation이 약해 missing object node, wrong object-room assignment, weak relation으로 전파될 수 있다고 보고한다. 긴 temporal window는 VIO drift와 stale dynamic evidence를 유입할 수 있다.

## Project reproduction status

**Project result:** Not reproduced; official weights unavailable. 공식 source code는 공개됐지만 model checkpoint와 ONNX export가 포함되지 않았다. 논문 pipeline과 benchmark를 실행하지 않았으며 현재 G-CUT3R teacher skeleton은 Mono-Hydra++ 재현이 아니다.

## References

- [arXiv paper](https://arxiv.org/abs/2605.17661)
- [Official code](https://github.com/BavanthaU/mono-hydra-pp)
- [Original Mono-Hydra paper](https://arxiv.org/abs/2308.05515)
- [Original Mono-Hydra code](https://github.com/UAV-Centre-ITC/Mono_Hydra)
- [Parent papers index](README.md)
