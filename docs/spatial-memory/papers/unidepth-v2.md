# UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler

| Field | Value |
|---|---|
| Page ID | SM-PAPER-UNIDEPTH-V2 |
| Status | Reviewed from primary sources; not reproduced in this project |
| Publication | IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(3), 2026, pp. 2354–2367 |
| Primary source | [Official DOI](https://doi.org/10.1109/TPAMI.2025.3628473) |
| Official code | [lpiccinelli-eth/UniDepth](https://github.com/lpiccinelli-eth/UniDepth) |
| Last checked | 2026-07-11 |
| Project links | [Paper index](README.md), [Architecture](../architecture.md), [Roadmap](../roadmap.md) |
| Project claims | [Traceability](../traceability.md): C-001 |

## 30-second summary

UniDepthV2는 camera intrinsics 없이 단일 RGB 이미지에서 메트릭 3D 포인트와 깊이를 예측한다. 자체 프롬프트된 조밀한 카메라 표현은 깊이 모듈을 조절하는 반면 의사 구형 출력은 카메라 광선을 방사형 깊이에서 분리한다. 이 모델은 또한 픽셀당 uncertainty를 예측하고 소형, 기본 및 대형 변형으로 출시된다.

## Problem addressed

단안 metric depth 모델은 장면 규모, camera intrinsics 또는 도메인 변경 시 일반화가 잘 안되는 경우가 많다. 실측 내장 함수를 요구하면 실제 사용도 제한된다. UniDepthV2는 RGB만으로 범용 metric depth 및 3D 예측을 목표로 하는 동시에 에지 선명도, 효율성, 입력 형태 견고성 및 신뢰도 출력을 향상시킵니다.

## Relevant method

카메라 모듈은 조밀한 광선 각도를 예측하고 깊이 모듈을 표시한다. 출력 공간은 방위각, 고도, 방사형 깊이를 사용하므로 카메라 오류와 깊이 오류가 분리된다. 기하학적 불변성 훈련은 동일한 이미지의 변환된 보기를 사용한다. 에지 유도 스케일 이동 불변 손실은 불연속성을 날카롭게 하고 uncertainty 헤드는 절대 로그 깊이 오류 순위를 학습한다.

이 논문은 23개 데이터 세트의 1,600만 개 이미지에 대한 교육을 보고한 다음 보이지 않는 10개의 실내, 실외 및 까다로운 데이터 세트에 대한 zero-shot 전송을 평가한다.

## Paper-reported evidence

이는 이 저장소의 결과가 아닌 논문 결과이다.

| Dataset or condition | Metric | Reported result | Source location |
|---|---|---:|---|
| SUN-RGBD zero-shot, UniDepthV2-Large | delta1 / absolute relative error | 96.4 / 6.8 | Table I, paper p. 7 |
| IBims-1 zero-shot, UniDepthV2-Large | delta1 / absolute relative error | 94.5 / 7.8 | Table I, paper p. 7 |
| NuScenes zero-shot, UniDepthV2-Large | delta1 / absolute relative error | 87.0 / 15.0 | Table II, paper p. 7 |
| A6000, mixed precision, 0.5-megapixel input, Small | Latency / parameters / memory | 23.0 ms / 34.18M / 0.66 GiB | Table VIII, paper p. 10 |
| Same setting, Large | Latency / parameters / memory | 65.4 ms / 353.8M / 3.47 GiB | Table VIII, paper p. 10 |
| Aggregated zero-shot uncertainty, Large | nAUSE / Spearman correlation | 0.645 / 0.299 | Table VII, paper p. 9 |

uncertainty 테이블은 중요한 자격이다. 도메인 내 대규모 보고서에서는 nAUSE 0.199 및 순위 상관 관계가 0.744이지만 zero-shot 값은 0.645 및 0.299로 저하된다. 따라서 이 논문은 완벽하게 보정된 절대 신뢰도가 아닌 도메인 이동 하에서 유익한 uncertainty 순위를 지원한다.

## What this supports here

**논문 주장:** 단안식 RGB는 광범위한 zero-shot 테스트 제품군에서 제공된 내장 기능 없이 metric depth, 추론된 카메라 형상, 3D 포인트 및 유용한 uncertainty 신호를 제공할 수 있다.

**프로젝트 추론:** UniDepthV2는 상대 깊이 전용 제공자보다 더 강력한 측정 기준 단일 프레임 기준이다. 작은 변형은 프레임당 지오메트리 제안의 후보이며 uncertainty는 프로젝트 측 보정 후 쓰기 게이트에 알릴 수 있다.

Metric depth는 외부적으로 신뢰할 수 있는 VIO 포즈와 결합되어 입력된 레코드 선택 전에 임시 월드 프레임 후보를 생성할 수 있다.

## What it does not prove

- 단일 이미지 metric depth는 시간적 일관성, loop closure, 인스턴스 ID 또는 지속적인 공통 coordinate frame를 설정하지 않는다.
- zero-shot uncertainty 측정항목에서 볼 수 있듯이 uncertainty 출력은 도메인 이동에 따라 완전히 보정되지 않는다.
- 용지 GPU 대기 시간은 AI-안경 전원, 열 또는 메모리 적합성을 설정하지 않는다.
- 1Hz 평생 스트림, 유형화된 공간 압축, SuperMemory-VQA 또는 결정론적 기하학 QA 증명은 평가하지 않는다.
- Metric depth 정확도는 바이트당 향후 QA 값을 최대화하는 레코드를 결정하지 않는다.

## Project reproduction status

UniDepthV2는 로컬로 설치, 다운로드 또는 실행되지 않는다. 공식적인 모델 가중치나 예측은 저장소에 저장되지 않는다. 이는 계획된 경량 메트릭 깊이 기준선과 잠재적인 학생 구성 요소로 남아 있다.

## References

- [Official DOI](https://doi.org/10.1109/TPAMI.2025.3628473)
- [Official arXiv record, version 2](https://arxiv.org/abs/2502.20110)
- [Official code and model documentation](https://github.com/lpiccinelli-eth/UniDepth)

[Back to paper index](README.md)
