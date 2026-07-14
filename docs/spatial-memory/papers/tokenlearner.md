# TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-TOKENLEARNER |
| 상태 | 검토 완료 |
| 저자 | Michael S. Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, Anelia Angelova |
| 출판 | NeurIPS 2021; extended arXiv version |
| 확인 version | arXiv:2106.11297v4, 2022-04-03 |
| 1차 출처 | [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/6a30e32e56fce5cf381895dfe6ca7b6f-Abstract.html) · [Extended arXiv paper](https://arxiv.org/abs/2106.11297) |
| 공식 code | [Scenic TokenLearner](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner) |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [상위: 논문 목록](README.md) · [아키텍처](../architecture.md) · [추적성](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-010 |

## 핵심 결론

- 고정된 개수의 학습된 슬롯을 사용하여 후보 decoder 비용을 명시적으로 만듭니다.
- 동일한 downstream QA 및 지오메트리 테스트에서 4, 8, 16, 32와 같은 슬롯 예산을 비교한다.
- 밀도가 높은 공급자 출력을 균일하게 풀링하는 대신 슬롯을 입력 적응형으로 만듭니다.

## 근거 상태

재현되지 않았다. 현재 decoder는 경험적 명시적 개체, 관계 및 영역 후보를 내보냅니다. TokenLearner 모듈이 아니다. 학습된 고정 슬롯 decoder는 계속 계획되어 있으며 일반적인 시각적 기능이 아닌 유형화된 기하학과 uncertainty를 방출해야 한다.

## 논문 핵심

TokenLearner는 학습된 공간 주의 및 풀링을 사용하여 조밀한 이미지와 같은 기능 텐서를 작은 입력 적응형 토큰 세트로 변환한다. 이는 고정 토큰 수를 명시적 컴퓨팅 예산으로 사용하는 것을 지원한다. 해당 토큰은 미터법 엔터티, 좌표 포함 레코드 또는 long-term memory 항목이 아닌 인식 기능이다.

## 근거

이 논문에서는 TokenLearner가 분류 성능을 손상시키지 않으면서 transformer 계산을 절반 이상 줄일 수 있다고 보고한다. 보고된 ImageNet 비교 중 하나에서 ViT-L/16은 87.35 top-1 정확도에 363.1 GFLOP를 사용하는 반면, 레이어 12에 삽입된 16토큰 변형은 87.68에 178.1 GFLOP를 사용한다. 이 논문에서는 Kinetics-400, Kinetics-600, Charades 및 AViD도 평가한다.

이는 이 저장소에서 재현한 결과가 아닌 논문 결과이다.

## 판단 한계

- 8개의 일반 시각적 토큰은 미터법 좌표 또는 인스턴스 ID를 유지한다.
- 고정된 잠재 병목 현상은 알려지지 않은 향후 질문에 충분하다.
- 그 인식 정확도는 공간적 관계나 마지막으로 본 정확도를 예측한다.
- 해당 임시 토큰은 영구 표현으로 저장되어야 한다.

## 문제 배경

비전 변환기는 2차 self-attention 비용으로 수백 또는 수천 개의 패치 토큰을 반복적으로 처리한다. 고정 그리드는 유용하고 중복된 공간 영역에 동일한 컴퓨팅을 사용한다.

## 관련 방법

- 학습된 공간 가중치 맵은 입력 종속 영역을 식별한다.
- 각 가중치 맵은 기능 텐서의 가중치를 다시 적용한다. 전역 공간 풀링은 하나의 학습된 토큰을 생성한다.
- 일반적으로 8 또는 16개의 작은 고정 출력 개수는 후속 transformer 레이어의 훨씬 더 큰 패치 시퀀스를 대체한다.
- 선택적 TokenFuser는 조밀한 출력이 필요할 때 처리된 토큰을 공간 feature map로 다시 매핑한다.

## 참고문헌

- [논문 목록](README.md)
- [arXiv:2106.11297](https://arxiv.org/abs/2106.11297)
- [Scenic 공식 implementation](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner)
