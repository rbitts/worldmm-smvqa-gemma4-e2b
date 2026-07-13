# SuperMemory-VQA: An Egocentric Visual Question-Answering Benchmark for Long-Horizon Memory

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-SUPERMEMORY-VQA |
| 상태 | Primary source 검토 완료; project benchmark 로컬 미실행 |
| 출판 | arXiv:2606.00825v1, 2026 |
| 1차 출처 | [공식 arXiv record](https://arxiv.org/abs/2606.00825) |
| 공식 code | [AIoT-MLSys-Lab/supermemory-vqa](https://github.com/AIoT-MLSys-Lab/supermemory-vqa) |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [논문 목록](README.md), [문제 정의](../problem.md), [아키텍처](../architecture.md), [현재 상태](../status.md) |
| 프로젝트 claim | [추적성](../traceability.md): C-008, C-009 |

## 핵심 결론

**논문 주장:** 장기적인 자기중심적 기억에는 인과관계 검색, 다중 증거 추론, 명시적인 기권 평가가 필요하다.

**프로젝트 추론:** persistent memory는 일반적인 요약만 유지하기보다는 정확한 개체 상태, 시간적 유효성, 개수, 공간 증거 및 provenance를 유지해야 한다. 따라서 저장소는 4방향 QA-Acc 및 QA-MRR를 기본 벤치마크 출력으로 처리하고 질문 컷오프 이후 증거를 거부한다.

데이터 세트의 RGB, IMU, 궤적 및 SLAM 스트림은 시스템을 캡션으로 제한하는 대신 명시적 공간 메모리 분기를 평가하는 것을 정당화한다.

## 근거 상태

저장소는 준비된 데이터 계약, 인과관계 사전 확인, 4가지 선택 지표, 증거 검증 및 소규모 합성 연기 테스트를 구현한다. 공식 SuperMemory-VQA 데이터 세트는 로컬로 복사되지 않았으며 이 호스트에서는 공식 벤치마크 실행이 수행되지 않았다. 현재 지역 모의 점수는 건전성 점검이므로 종이 재생산 결과로 보고해서는 안 된다.

## 논문 핵심

SuperMemory-VQA는 AI-glasses 메모리 보조 장치가 증거가 불충분할 때 기권하면서 긴 수평, 다중 모달, 자기 중심적 녹음을 검색하고 추론할 수 있는지 여부를 평가한다. 여기에는 10명의 참가자가 참여한 52.9시간과 6가지 실제 기억 과제에 걸쳐 사람이 검증한 4지선다형 질문 4,853개가 포함되어 있다. 모든 질문에는 대답할 수 없는 명시적인 옵션이 포함되어 있다. 벤치마크는 검색, 시간적 통합, 정확한 상태 추적 및 응답 가능성의 격차를 드러냅니다.

## 근거

이는 이 저장소의 결과가 아닌 논문 결과이다.

| Dataset 또는 조건 | Metric | 논문 보고 결과 | 출처 위치 |
|---|---|---:|---|
| Full dataset | Duration / participants / QA pairs | 52.9 hours / 10 / 4,853 | Section 3.1 and Table 1, paper pp. 4–5 |
| Full dataset | Questions requiring multiple evidence items | 34% | Table 1, paper p. 5 |
| Gemini-3-Flash with Video-RAG | Ans-F1 / QA-Acc / QA-MRR | 83.9 / 61.0 / 76.0 | Table 2, paper p. 8 |
| Mean over ten evaluated VLMs, Video-RAG versus EgoButler | QA-Acc | 46.6 versus 41.4 | Section 5.1, paper p. 9 |
| Qwen3-8B, text-only, Person 1, 1,017 questions | QA-Acc | 23.8%, versus 25% chance | Table 3, paper p. 11 |

가장 강력하게 보고된 Video-RAG 구성은 83.9 Ans-F1에도 불구하고 61.0 QA-Acc에 불과한다. 논문에서는 이를 응답성 탐지가 필요하지만 불충분하다는 증거로 해석한다. 정확한 증거 검색과 근거 있는 추론은 여전히 ​​어렵다.

## 판단 한계

- 명시적 유형의 지오메트리 메모리가 벤치마크 점수를 향상시킨다는 것을 보여주지는 않는다.
- G-CUT3R, CUT3R 또는 특정 형상 encoder의 유효성을 검사하지 않는다.
- 보고된 기준선에는 응시, 궤적, IMU 또는 SLAM 입력이 포함되지 않는다.
- 기기 내 저장, 대기 시간 또는 전력 목표를 설정하지 않는다.
- 벤치마크 정확도만으로는 metric geometry 정확성을 입증할 수 없다. 별도의 증명, uncertainty 및 provenance 확인이 여전히 필요하다.

## 문제 배경

기존 자기중심적 벤치마크는 주로 짧은 클립 인식, 동작 인식 또는 일반 QA를 테스트한다. 세션, 요일, 양식 및 분리된 증거 순간에 걸친 실제 기억 문제를 직접 테스트하지 않는다. SuperMemory-VQA는 객체 및 위치 메모리, 대화형 메모리, 시각적 장면 리콜, 컨텍스트 내 검색, 타임라인 재구성 및 의도 리콜에 대한 벤치마크를 정의한다.

## 관련 방법

데이터 세트는 RGB, 오디오, 시선, IMU 및 SLAM 궤적을 포함하는 Meta Aria 녹음을 사용한다. 주석 파이프라인은 근거 있는 설명을 생성하고, QA 쌍을 제안하고, 인과성과 증거를 확인하고, 사람의 검토로 끝납니다. 질문은 정답, 모호함, 틀림, 답할 수 없는 답변을 나타내는 순서 선택을 사용한다.

이 논문는 개방형 및 폐쇄형 VLM을 사용하여 Video-RAG 및 EgoButler를 평가한다. 두 시스템 모두 질문 종료 시간 이전의 증거만 수신한다. 공식 측정항목은 다음과 같다.

- Ans-F1: 응답 가능한 바이너리와 응답 불가능한 F1.
- QA-Acc: 정확한 4방향 객관식 정확도.
- QA-MRR: 정렬된 답변 점수에서 올바른 옵션의 상호 순위.

## 참고문헌

- [Official arXiv record and paper](https://arxiv.org/abs/2606.00825)
- [공식 code](https://github.com/AIoT-MLSys-Lab/supermemory-vqa)
- [공식 dataset](https://huggingface.co/datasets/OSU-AIoT-MLSys-Lab/SuperMemory-VQA)

[논문 근거 목록으로 돌아가기](README.md)
