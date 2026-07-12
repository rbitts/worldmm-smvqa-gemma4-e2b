# Depth Anything V2

| Field | Value |
|---|---|
| Page ID | SM-PAPER-DEPTH-ANYTHING-V2 |
| Status | Reviewed from primary sources; not reproduced in this project |
| Publication | NeurIPS 2024, Main Conference Track |
| Primary source | [Official NeurIPS paper record](https://proceedings.neurips.cc/paper_files/paper/2024/hash/26cfdcd8fe6fd75cc53e92963a656c58-Abstract-Conference.html) |
| Official code | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) |
| Last checked | 2026-07-11 |
| Project links | [Paper index](README.md), [Architecture](../architecture.md), [Roadmap](../roadmap.md) |
| Project claims | [Traceability](../traceability.md): C-001 |

## 30-second summary

Depth Anything V2는 2,500만 ~ 13억 개의 매개변수를 포함하는 monocular depth 기초 모델 제품군이다. DINOv2-G 교사는 정확한 합성 심도를 통해 학습하고 6,200만 개의 실제 이미지에 라벨을 지정하고 강력한 학생 모델을 추출한다. 기본 모델은 아핀 불변 상대 역 깊이를 예측한다. 별도의 실내 및 실외 모델은 metric depth에 맞게 미세 조정됐다.

## Problem addressed

실제 깊이 라벨은 종종 얇은 구조를 놓치거나 투명하거나 반사되는 표면에서 실패한다. 합성 라벨은 정확하지만 도메인 및 적용 범위에 차이가 있다. Depth Anything V2는 합성 감독과 대규모 의사 레이블이 붙은 실제 이미지를 결합하여 세부 사항, 견고성, 효율성 및 전송 가능성을 향상시킵니다.

## Relevant method

훈련 파이프라인은 3단계로 구성된다. 595,000개의 합성 이미지로 대규모 DINOv2-G 교사를 ​​훈련하고, 레이블이 지정되지 않은 6,200만 개가 넘는 실제 이미지에 대해 의사 깊이를 생성한 다음, 의사 레이블이 지정된 실제 이미지로 DINOv2 기반 학생을 훈련한다. DPT는 깊이 decoder이다. 공개된 상대적 심층 학생은 ViT-S, ViT-B 및 ViT-L에 걸쳐 있다. 미터법 변형은 실내 또는 실외 미터법 감독을 사용하여 미세 조정된다.

## Paper-reported evidence

이는 이 저장소의 결과가 아닌 논문 결과이다.

| Dataset or condition | Metric | Reported result | Source location |
|---|---|---:|---|
| DA-2K, ViT-S relative-depth model | Accuracy / parameters / V100 latency | 95.3% / about 25M / 60 ms | Figure 1 and Table 3, paper pp. 1 and 8 |
| DA-2K, ViT-L relative-depth model | Accuracy / parameters / V100 latency | 97.1% / about 335M / 213 ms | Figure 1 and Table 3, paper pp. 1 and 8 |
| DA-2K, Marigold comparison | Accuracy / parameters / V100 latency | 86.8% / 948M / 5.2 s | Figure 1 and Table 3, paper pp. 1 and 8 |
| ViT-S, synthetic-only versus pseudo-real-only training | DA-2K accuracy | 89.8% versus 95.3% | Table 5, paper p. 9 |
| Metric fine-tuning, ViT-L | NYU-D delta1 / Abs Rel; KITTI delta1 / Abs Rel | 0.984 / 0.056; 0.983 / 0.045 | Table 4, paper p. 9 |

기존 zero-shot 테이블은 여러 데이터 세트에서 Depth Anything V1와 비슷한 결과를 보고한다. 이 논문은 이러한 시끄러운 벤치마크가 미세한 구조, 투명한 표면 및 복잡한 레이아웃에 대한 주장된 이점을 완전히 측정하지 못한다고 명시적으로 주장한다.

## What this supports here

**논문 주장:** 상대적으로 작은 단안 모델은 강력한 프레임당 깊이를 생성할 수 있으며 교사 대 학생 의사 라벨링은 모델 크기 전반에 걸쳐 기하학적 기능을 전송한다.

**프로젝트 추론:** Depth Anything V2-Small은 전체 multi-view 교사가 필요하지 않은 경우 실용적인 경량 형상 기준선 또는 학생 기능 소스이다. 메트릭 변형은 신뢰할 수 있는 VIO 포즈와의 융합을 위한 프레임별 깊이 후보를 제공할 수 있다.

교사/의사 라벨/학생 파이프라인은 또한 더 작은 온디바이스 후보 encoder를 배포하는 동시에 대규모 오프라인 geometry provider 사용을 지원한다.

## What it does not prove

- 주요 상대 깊이 모델은 절대적인 미터법 척도를 제공하지 않는다.
- 프레임당 단안 정확도는 시간적 일관성, 공통 세계 좌표, camera pose 또는 인스턴스 연관을 설정하지 않는다.
- 메트릭 결과는 범용 1Hz 스트리밍 재구성 테스트가 아닌 도메인별 fine-tuning에서 나온 것이다.
- persistent memory, 실제 저장 바이트, SuperMemory-VQA 또는 기하학 기반 QA는 평가하지 않는다.
- 보고된 V100 대기 시간은 AI-안경의 전력 또는 열적 타당성을 확립하지 못한다.

## Project reproduction status

Depth Anything V2는 로컬로 설치, 다운로드 또는 실행되지 않는다. 이는 계획된 경량 깊이 제공자 기준선이다. 현재 시간적 일관성, 형식화된 레코드 유틸리티 또는 SuperMemory-VQA 영향을 측정하는 프로젝트 결과는 없다.

## References

- [Official NeurIPS 2024 record](https://proceedings.neurips.cc/paper_files/paper/2024/hash/26cfdcd8fe6fd75cc53e92963a656c58-Abstract-Conference.html)
- [Official project page](https://depth-anything-v2.github.io/)
- [Official code](https://github.com/DepthAnything/Depth-Anything-V2)
- [Official arXiv record](https://arxiv.org/abs/2406.09414)

[Back to paper index](README.md)
