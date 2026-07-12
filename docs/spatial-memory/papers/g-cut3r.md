# G-CUT3R: Guided 3D Reconstruction with Camera and Depth Prior Integration

| Field | Value |
|---|---|
| Page ID | SM-PAPER-G-CUT3R |
| Status | Reviewed from primary sources; external teacher implementation blocked |
| Publication | ICLR 2026 |
| Primary source | [Official OpenReview record](https://openreview.net/forum?id=J7DiMqmIFl) |
| Official code | OpenReview supplementary material is described by the paper as including source code; no standalone official repository was verified |
| Last checked | 2026-07-11 |
| Project links | [Paper index](README.md), [Architecture](../architecture.md), [ADR-0002](../decisions/adr-0002-gcut3r-as-teacher.md), [Status](../status.md) |
| Project claims | [Traceability](../traceability.md): C-001 |

## 30-second summary

G-CUT3R는 CUT3R를 확장하므로 선택적 camera intrinsics, 카메라 포즈 및 깊이 맵이 반복 재구성을 안내할 수 있다. 각 양식에는 전용 encoder가 있다. 이 기능은 초기화되지 않은 컨볼루션을 통해 CUT3R decoder 블록에 들어가므로 하나의 모델이 순차 상태 메커니즘을 유지하면서 사용 가능한 모든 사전 조합을 허용할 수 있다.

## Problem addressed

Feed-forward 재구성 모델은 일반적으로 보정, VIO, SLAM, RGB-D 또는 LiDAR에서 이미 사용 가능한 기하학적 정보를 무시한다. 쌍별 안내 방법에는 비용이 많이 드는 전역 정렬이 필요할 수도 있다. G-CUT3R는 온라인 반복 재구성 모델 내에서 경량 사전 융합을 목표로 한다.

## Relevant method

Camera intrinsics 및 포즈는 광선 이미지로 인코딩된다. 깊이에는 자체 공간 인코딩이 있다. 4블록 양식별 transformer 인코더는 CUT3R와 동일한 768차원 인터페이스를 사용하여 기능을 생성한다. 0으로 초기화된 컨볼루션 레이어는 이러한 기능을 decoder 단계에 주입한다. 훈련은 사용 가능한 양식을 무작위로 변경하므로 하나의 체크포인트가 내장 기능, 자세 및 깊이 안내의 임의 조합을 지원한다.

이 논문은 CUT3R에서 초기화되고, 12개의 데이터 세트를 사용하여 4개의 이미지 시퀀스에 대해 훈련하고, 4개의 A100 GPU에서 10일 동안의 훈련을 보고한다. 평가에는 낮은 중첩 3D 재구성, 비디오 깊이, pose estimation 및 융합 절제가 포함된다.

## Paper-reported evidence

이는 이 저장소의 결과가 아닌 논문 결과이다.

| Dataset or condition | Metric | Reported result | Source location |
|---|---|---:|---|
| 7-Scenes, 3–5 low-overlap views, resolution 512 | Mean accuracy, unguided / pose / all priors | 0.098 / 0.061 / 0.048 | Table 1, paper p. 7 |
| NRGBD, 3–5 low-overlap views, resolution 224 | Mean normal consistency, unguided / depth / all priors | 0.708 / 0.746 / 0.767 | Table 1, paper p. 7 |
| Bonn, ten-frame video depth, resolution 224 | Abs Rel, unguided / pose / intrinsics plus pose | 0.126 / 0.105 / 0.104 | Table 2, paper p. 8 |
| Waymo, four-view reconstruction | Fourth-view L2, no ZeroConv versus ZeroConv, all priors | 1.959 versus 1.155 | Table 3, paper p. 9 |
| ScanNet++, four-view reconstruction | Fourth-view L2, no ZeroConv versus ZeroConv, all priors | 0.078 versus 0.064 | Table 3, paper p. 9 |

논문에서는 유도되지 않은 미세 조정 변형과 원본 CUT3R가 서로 다른 훈련 데이터를 보았기 때문에 공정하게 일치하는 쌍이 아니라고 지적한다. 따라서 유도 효과는 동일한 데이터 G-CUT3R 비유도 변형에 대해 읽어야 한다.

## What this supports here

**논문 주장:** 선택적 포즈, 고유 및 깊이 사전은 CUT3R 파생 재구성 모델을 개선할 수 있으며, 0으로 초기화된 특징 주입은 실행 가능한 융합 메커니즘을 제공한다.

**프로젝트 추론:** 고속 IMU/VIO 포즈 및 선택적 보정 깊이는 RGB가 1Hz 근처로 샘플링되고 뷰 중첩이 낮을 때 귀중한 교사 입력이다. 따라서 G-CUT3R는 선호되는 외부 교사 후보이지만 밀집 상태는 일시적으로 유지된다.

양식 드롭 훈련 방식은 또한 포즈 전용, 깊이 전용, 결합 및 RGB-only 입력에 대한 명시적 제공자 메타데이터 및 절제에 동기를 부여한다.

## What it does not prove

- AI-안경 전력, 열 제한 또는 장치 내 대기 시간은 평가하지 않는다.
- persistent memory 형식의 평생 상태 보존, 실제 직렬화된 바이트 또는 기하학적 기반 QA는 평가하지 않는다.
- 낮은 오버랩 3~5개 보기 실험은 몇 달 동안 1Hz 비디오를 시청하는 것과 동일하지 않는다.
- 더 나은 재구성이 더 나은 SuperMemory-VQA 정확도를 입증하지는 않는다.
- 추론된 형상에는 여전히 uncertainty 및 provenance가 필요하다. 그것은 직접적으로 관찰된 사실로 취급될 수 없다.
- 확인 날짜에 독립적으로 검증된 독립형 공식 코드 저장소는 없다. 재현성은 공식 OpenReview 보충 자료에 따라 다릅니다.

## Project reproduction status

저장소는 외부 공급자 프로토콜, 인과 캐시 해시, 교사 구체화 및 DDP 후보자 헤드 교육 스캐폴드를 구현한다. G-CUT3R 추출기 또는 체크포인트가 포함되어 있지 않으며 G-CUT3R 추론을 실행하지 않았다. 체크포인트에서 입력된 메모리로의 추론은 여전히 ​​P0 차단제로 남아 있다.

## References

- [Official OpenReview record](https://openreview.net/forum?id=J7DiMqmIFl)
- [Official arXiv record, version 2](https://arxiv.org/abs/2508.11379)

[Back to paper index](README.md)
