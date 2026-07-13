# Mono-Hydra++: Real-Time Monocular Scene Graph Construction with Multi-Task Learning for 3D Indoor Mapping

| 항목 | 값 |
| --- | --- |
| Page ID | SM-PAPER-MONO-HYDRA-PLUS-PLUS |
| 상태 | Primary source 검토 완료; weight 부재로 reproduction 차단 |
| 출판 | arXiv preprint 2605.17661v1, 2026; submitted to ISPRS Journal |
| 1차 출처 | [arXiv:2605.17661](https://arxiv.org/abs/2605.17661) |
| 공식 code | [BavanthaU/mono-hydra-pp](https://github.com/BavanthaU/mono-hydra-pp); checkpoint와 ONNX export 미포함 |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [논문 목록](README.md), [문제 정의](../problem.md), [아키텍처](../architecture.md), [추적성](../traceability.md) |
| 프로젝트 claim | [추적성](../traceability.md): C-001, C-006 |

## 핵심 결론

**프로젝트 추론:** 1 Hz RGB 사이의 low overlap을 high-rate IMU/VIO pose prior로 보완하고, fast pose memory와 persistent graph memory를 분리하는 설계 근거다. Past-only pose warp는 causal short-term geometry stabilization의 직접 사례다.

## 근거 상태

**프로젝트 결과:** 미재현; official weight 사용 불가. 공식 source code는 공개됐지만 model checkpoint와 ONNX export가 포함되지 않았다. 논문 pipeline과 benchmark를 실행하지 않았으며 현재 G-CUT3R teacher skeleton은 Mono-Hydra++ 재현이 아니다.

## 논문 핵심

**논문 주장:** monocular RGB와 IMU를 multi-task depth·semantic model, visual-inertial odometry, causal pose-aware fusion, Hydra backend에 연결하면 RGB-D 없이도 real-time metric-semantic hierarchy와 3D scene graph를 만들 수 있다.

## 근거

**논문 보고 결과:**

- Selected Go-SLAM ScanNet subset에서 average ATE `6.91 cm`; 비교한 strongest RGB-D baseline보다 `1.6%` 낮다고 보고했다. 출처: Section 4.5.
- Calibrated 7-Scenes에서 average ATE `0.033 m`; strongest calibrated baseline `0.047 m` 대비 `29.8%` 개선을 보고했다. 출처: Section 4.6.
- ScanNet에서 global mIoU `44.96`, Radius F1@0.5m `42.59`, Box F1@0.25 `33.81`을 보고했다. 출처: Section 4.8.
- Jetson Orin NX 16GB에서 TensorRT FP16 M2H-MX-L perception component가 `25.53 FPS`로 동작했다. 출처: Section 4.10.

`25.53 FPS`는 full scene-graph pipeline 속도가 아니라 perception component 결과다. ScanNet 비교도 selected sequence와 서로 다른 sensor setting의 reference라는 저자 주의가 있다.

## 판단 한계

- 1 Hz로 subsample된 RGB 조건의 정확도
- Open-vocabulary language grounding
- Long-term storage compression, learned writer, actual-byte budget 미평가
- QA 또는 SuperMemory-VQA 성능
- Small-object evidence와 dynamic event history의 안정적 retention
- 전체 pipeline의 glass-class latency·energy

논문은 object-level semantic preservation이 약해 missing object node, wrong object-room assignment, weak relation으로 전파될 수 있다고 보고한다. 긴 temporal window는 VIO drift와 stale dynamic evidence를 유입할 수 있다.

## 문제 배경

RGB-D와 LiDAR는 lightweight robot에 payload·power 부담이 있다. Monocular scene graph는 scale, drift, depth, semantics, dynamics를 동시에 처리해야 한다. Mono-Hydra++는 RGB+IMU만으로 이 pipeline을 구성한다.

## 관련 방법

- Frozen DINOv3 backbone과 Mamba decoder 기반 M2H-MX가 metric depth와 semantic label을 예측한다.
- SuperPoint-assisted RVIO2-style frontend가 IMU, visual track, sparse predicted-depth factor를 결합한다.
- Dynamic·unreliable semantic region을 VIO update에서 mask한다.
- Past frame의 depth·semantics를 VIO pose로 current frame에 warp하고 causal short-window fusion한다.
- VIO odometry, pose graph, loop-closure proposal, fused evidence를 Hydra backend로 전달해 mesh와 building-room-place-object graph를 만든다.

## 참고문헌

- [arXiv paper](https://arxiv.org/abs/2605.17661)
- [공식 code](https://github.com/BavanthaU/mono-hydra-pp)
- [Original Mono-Hydra paper](https://arxiv.org/abs/2308.05515)
- [Original Mono-Hydra code](https://github.com/UAV-Centre-ITC/Mono_Hydra)
- [상위 논문 목록](README.md)
