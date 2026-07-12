# Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces

| Field | Value |
|---|---|
| Page ID | SM-PAPER-VSI-BENCH |
| Status | Reviewed; code and dataset available |
| Publication | CVPR 2025 Oral, pp. 10632–10643; arXiv:2412.14171 v2 |
| Primary source | [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Thinking_in_Space_How_Multimodal_Large_Language_Models_See_Remember_CVPR_2025_paper.html) |
| Official code | [vision-x-nyu/thinking-in-space](https://github.com/vision-x-nyu/thinking-in-space) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-003, C-008 |

## 30-second summary

VSI-Bench는 다중 모드 모델이 순차 비디오에서 실내 공간을 재구성하고 불러올 수 있는지 여부를 평가한다. 프롬프트 전용 추론은 격차를 좁히지 못하는 반면, 명시적 인지 지도는 상대 거리 정확도를 향상시킵니다. 이는 전역 공간 표현 및 기하학 실행에 대한 직접적인 증거이지만 저장 압축에 대한 증거는 아니다.

## Problem addressed

강력한 비디오 언어 모델은 안정적인 동종 중심 모델을 구축하지 않고도 콘텐츠를 인식할 수 있다. VSI-Bench는 8가지 공간 작업을 사용하여 자기중심적 실내 스캔 비디오에서 구성, 측정항목 추정 및 시공간 회상을 테스트한다.

## Relevant method

- ScanNet, ScanNet++ 및 ARKitScenes의 288개 비디오를 통해 5,000개 이상의 QA 쌍을 구축한다.
- 테스트 개체 수, 상대 거리, 상대 방향, 경로 계획, 절대 거리, 개체 크기, 공간 크기 및 모양 순서.
- 객관식 작업에는 정확도를 사용하고 수치 작업에는 평균 상대 정확도를 사용한다.
- 설명과 모델 생성 인지 지도를 이용한 프로브 모델 추론.

## Paper-reported evidence

| Dataset | Condition | Metric | Reported result | Location |
|---|---|---|---|---|
| VSI-Bench | Gemini-1.5 Pro | Average | 45.4 | Table 1 |
| VSI-Bench | Best reported open model, LLaVA-Video-72B | Average | 40.9 | Table 1 |
| VSI-Bench tiny | Human → Gemini-1.5 Pro | Average | 79.2 → 48.8 | Table 1 |
| Error analysis | 163 incorrect samples | Errors attributed to spatial reasoning | About 71% | Figure 7 and Section 5.1 |
| Relative-distance subset | No map → model-generated cognitive map | Accuracy | 46.0 → 56.0 | Table 3 |
| Relative-distance subset | Ground-truth 10×10 → 20×20 map | Accuracy | 66.0 → 78.0 | Table 3 |

논문에서는 zero-shot 생각의 사슬과 생각의 나무가 평균 성능을 약 4포인트 감소시킨 반면, 모델 생성 인지 지도는 상대 거리 작업을 10포인트 향상시켰다고 보고한다.

## What this supports here

- 단절된 국소 인상보다 통일된 동종 중심 표현이 더 좋다.
- 상대 방향, 거리, 경로 및 미터법 관련 질문은 별도로 평가해야 한다.
- 명시적인 공간 구조는 언어 전용 추론 프롬프트가 지원되지 않는 경우 도움이 될 수 있다.
- 프로젝트에서는 결정론적 기하학 작업이 language model에 기하학 추정을 요청하는 대신 명시적 레코드를 사용해야 한다고 추론한다.

## What it does not prove

- 특정 유형의 레코드 스키마 또는 학습된 쓰기 정책이다.
- 스토리지 감소, 실제 바이트 제한, 평생 메모리 또는 1Hz 샘플링.
- 단안 RGB의 미터법 교정 또는 포즈 uncertainty 처리.
- SuperMemory-VQA 전송.

## Project reproduction status

재현되지 않았다. VSI-Bench는 외부 공간 오류 분류 및 평가 참조로 유지된다.

## References

- 양지한 외. [Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Thinking_in_Space_How_Multimodal_Large_Language_Models_See_Remember_CVPR_2025_paper.html). CVPR 2025.
- [Official project page](https://vision-x-nyu.github.io/thinking-in-space.github.io/).
- [Official repository](https://github.com/vision-x-nyu/thinking-in-space).
- [Official dataset](https://huggingface.co/datasets/nyu-visionx/VSI-Bench).
- [Back to paper index](README.md).
