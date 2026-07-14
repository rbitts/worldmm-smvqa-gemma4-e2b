# TTSA3R: Training-Free Temporal-Spatial Adaptive Persistent State for Streaming 3D Reconstruction

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-TTSA3R |
| 상태 | 검토 완료; code 사용 가능 |
| 출판 | arXiv:2601.22615 v3, 2026 preprint |
| 1차 출처 | [arXiv](https://arxiv.org/abs/2601.22615) |
| 공식 code | [anonus2357/ttsa3r](https://github.com/anonus2357/ttsa3r) |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [상위: 논문 목록](README.md) · [아키텍처](../architecture.md) · [추적성](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-006 |

## 핵심 결론

- 작업 상태 게이트에서 공간적 관찰 품질과 시간적 부실을 분리한다.
- 포즈, 광선, 참신함, 역학 신호가 파괴적인 쓰기를 줄이는지 테스트한다.
- 학습된 QA-aware 영구 작성기와 훈련 없는 상태 안정화를 비교한다.

## 근거 상태

재현되지 않았다. 릴리스된 코드를 임시-state update 기준으로 사용한다. 보고된 재구성 안정성을 QA 메모리 품질에 대한 증거로 취급하지 않는다.

## 논문 핵심

TTSA3R는 시간 상태 진화와 공간 관찰 품질을 결합한 훈련이 필요 없는 CUT3R 상태 업데이트 정책이다. 하나의 글로벌 신뢰 게이트보다 더 선택적인 과도 state update를 지원한다. 이는 지속 가능한 명시적 메모리 압축기가 아닌 암시적 반복 상태 방법으로 남아 있다.

## 근거

**보고된 주장.** NRGBD에서 시퀀스 길이가 50에서 250 프레임으로 증가할 때 v3 논문은 CUT3R의 경우 `4×`보다 큰 TTSA3R의 `1.33×` 재구성 오류 증가를 보고한다. 절제에서는 시간적 및 공간적 모듈을 모두 갖춘 CUT3R 및 `0.064/0.026`에 대한 `0.078/0.046`의 Bonn Abs Rel 및 TUM Dynamics ATE를 보고한다. 이 논문에서는 TTSA3R에 대해 하나의 NVIDIA A6000에서 18.0 FPS 및 6 GB를 보고한다.

**프로젝트 추론.** 시간적 및 공간적 마스크는 희소 또는 동적 관찰 후 임시 기하학 상태가 업데이트될 수 있는 위치를 결정하는 데 유용한 기준이다.

**프로젝트 결과.** 없음. 이 저장소는 TTSA3R를 재현하지 않았다.

## 판단 한계

- 심한 교합이나 매우 낮은 시각적 중첩에서도 안정적인 결합.
- 엔터티 ID, 입력된 기하학, 임시 이벤트, 미터법 증명 또는 provenance.
- 실제 바이트 감소; 고정 잠재 상태 메모리는 직렬화된 평생 저장 장치가 아니다.
- SuperMemory-VQA 개선, 1Hz의 넓은 기준선 작동, 며칠간 보존 또는 AI-glass 타당성.

## 문제 배경

영구 recurrent state는 모든 토큰이 유사하게 업데이트되면 기록을 잊어버립니다. 시간 전용 또는 공간 전용 신호는 관찰이 오래되었거나 동적이거나 잘못 정렬된 위치에서 누락될 수 있다. TTSA3R는 두 신호를 결합하여 부적절한 영역을 더 적게 업데이트한다.

## 관련 방법

- 임시 적응형 업데이트 모듈은 상태 변화를 측정하고 토큰 업데이트 규모를 규제한다.
- 공간 상황별 업데이트 모듈은 관측 상태 정렬 및 장면 역학을 사용하여 업데이트가 필요한 영역을 찾는다.
- 두 마스크가 융합되어 시간적, 공간적 증거가 지속 상태 전환을 공동으로 제어한다.
- 개입에는 훈련이 필요 없으며 사전 훈련된 반복 재구성 모델을 사용한다.

## 참고문헌

- Zhijie Zheng, Xinhao Xiang, Jiawei Zhang. [TTSA3R: Training-Free Temporal-Spatial Adaptive Persistent State for Streaming 3D Reconstruction](https://arxiv.org/abs/2601.22615). arXiv v3, 2026.
- [공식 repository](https://github.com/anonus2357/ttsa3r).

[논문 근거 목록으로 돌아가기](README.md)
