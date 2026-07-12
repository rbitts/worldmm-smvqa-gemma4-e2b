# OpenEQA: Embodied Question Answering in the Era of Foundation Models

| Field | Value |
|---|---|
| Page ID | SM-PAPER-OPENEQA |
| Status | Reviewed; code and benchmark available |
| Publication | CVPR 2024, pp. 16488–16498 |
| Primary source | [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2024/html/Majumdar_OpenEQA_Embodied_Question_Answering_in_the_Era_of_Foundation_Models_CVPR_2024_paper.html) |
| Official code | [facebookresearch/open-eqa](https://github.com/facebookresearch/open-eqa) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-003, C-008, C-009 |

## 30-second summary

OpenEQA는 에피소드 메모리 기록과 활성 탐색 모두에 대해 open-vocabulary 구현된 QA를 정의한다. 스마트 안경 설정과 강력한 기초 모델 기준의 빈약한 공간 성능은 시각적 기록을 유지하는 것만으로는 충분하지 않다는 직접적인 증거가 된다. 메모리는 쿼리 가능한 공간 구조를 보존해야 한다. Spatial-memory compression 방식이 아닌 벤치마크 논문이다.

## Problem addressed

구체화된 에이전트는 자연어 질문에 답할 수 있을 만큼 실제 환경을 잘 이해해야 한다. EM-EQA에서는 스마트 안경 메모리와 유사한 과거 관찰 시퀀스를 수신한다. A-EQA에서는 적극적인 탐색을 통해 증거를 수집해야 한다. Open-vocabulary 답변에는 동등한 표현을 허용하는 평가 방법이 필요하다.

## Relevant method

- 실제 비디오 둘러보기 및 스캔을 통해 구현된 7가지 QA 카테고리에 걸쳐 사람이 작성한 질문을 수집한다.
- 시각 장애인 언어 모델, 캡션 기반 에이전트, 장면 그래프 에이전트 및 다중 프레임 비전 언어 모델을 평가한다.
- LLM-Match를 사용하여 공개 답변에 점수를 매깁니다. 평가자는 1~5를 할당하고 논문은 총계를 0~100으로 정규화한다.
- 독립적인 인간 판단에 대해 LLM-Match를 검증한다.

## Paper-reported evidence

| Dataset | Condition | Metric | Reported result | Location |
|---|---|---|---|---|
| OpenEQA | Dataset | Questions and environments | 1,636 questions; more than 180 environments; seven categories | Figure 2 and Section 2.3 |
| EM-EQA | Multi-frame GPT-4V, 500-question subset | LLM-Match | 49.6 ± 2.0 | Table 2 |
| EM-EQA | Human | LLM-Match | 86.8 ± 0.6 | Table 2 |
| EM-EQA | Blind GPT-4 | LLM-Match | 33.5 ± 1.0 | Table 2 |
| LLM-Match validation | 300 sampled questions | Spearman correlation with human scoring | 0.909; bootstrap CI 0.883–0.928 | Section 5 |

이 논문에서는 또한 객체 위치 파악과 공간 이해가 가장 어려운 범주에 속한다고 보고한다. 장면 그래프 에이전트는 공간 질문에 대해 프레임 캡션 에이전트보다 성능이 우수하지 않았다.

## What this supports here

- 과거 관찰 순서는 유효한 스마트 안경 QA 설정이다.
- 긴 시각적 컨텍스트에는 관련 없는 증거가 포함될 수 있으며 여전히 spatial reasoning에 실패할 수 있다.
- 공간 평가는 하나의 총점에 숨겨지기보다는 문제 유형별로 분리되어야 한다.
- 기하학 실행자와 근거 증명은 공간적 격차에 대한 프로젝트 대응이다. OpenEQA는 이를 처방하지 않는다.

## What it does not prove

- 입력된 객체, 평면, 포털, 랜드마크 또는 이벤트 기록이다.
- Metric geometry 증명 또는 결정론적 공간 실행.
- 바이트 예산 작성기, 1Hz 희소 감지, 평생 저장 또는 기기 내 실행.
- SuperMemory-VQA의 성능.

## Project reproduction status

재현되지 않았다. 이 저장소는 SuperMemory-VQA를 대상으로 하며 OpenEQA를 외부 작업 설계 및 평가 참조로 사용한다. OpenEQA 데이터 또는 모델 아티팩트가 로컬로 다운로드되지 않았다.

## References

- Arjun Majumdaret al. [OpenEQA: Embodied Question Answering in the Era of Foundation Models](https://openaccess.thecvf.com/content/CVPR2024/html/Majumdar_OpenEQA_Embodied_Question_Answering_in_the_Era_of_Foundation_Models_CVPR_2024_paper.html). CVPR 2024.
- [Official project page](https://open-eqa.github.io/).
- [Official benchmark, code, and evaluator](https://github.com/facebookresearch/open-eqa).
- [Back to paper index](README.md).
