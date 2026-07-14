# SQA3D: Situated Question Answering in 3D Scenes

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-SQA3D |
| 상태 | 검토 완료; code와 dataset 사용 가능 |
| 출판 | ICLR 2023; arXiv:2210.07474 v5 |
| 1차 출처 | [OpenReview](https://openreview.net/forum?id=IDJx97BC38) · [arXiv](https://arxiv.org/abs/2210.07474) |
| 공식 code | [SilongYong/SQA3D](https://github.com/SilongYong/SQA3D) |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [상위: 논문 목록](README.md) · [아키텍처](../architecture.md) · [추적성](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-003 |

## 핵심 결론

- 자기중심적 방향 질문에 대해서는 좌표계 동일성이 명시적이어야 한다.
- 포즈 감독은 기하학적 기반 QA에 유용한 보조 목표이다.
- 상황 접지는 답변 분류와 별도로 평가되어야 한다.
- 프로젝트는 명시적인 프레임 메타데이터와 착용자 포즈가 압축 후에도 유지되어야 한다고 추론한다.

## 근거 상태

재현되지 않았다. SQA3D는 현재 메모리 기록기가 완전하다는 증거가 아니라 외부 위치 QA 참조 및 가능한 보조 평가로 유지된다.

## 논문 핵심

SQA3D는 완전한 3D 장면 내에서 설명된 에이전트 위치와 방향에서 질문을 한다. 상황 입력 및 보조 포즈 감독은 QA를 개선하여 명시적인 착용자 좌표 프레임 및 포즈 인식 학습을 지원한다. 인과적인 희소 스트림 메모리가 아닌 정적, 전체 스캔된 실내 장면을 평가한다.

## 근거

| Dataset | 조건 | Metric | 논문 보고 결과 | 위치 |
|---|---|---|---|---|
| SQA3D | Dataset | Scale | 650 scenes; 6.8k situations; 20,369 descriptions; 33,403 questions | Section 2.2 and Table 2 |
| SQA3D | Blind situation + question | Accuracy | 43.65% | Table 3 |
| SQA3D | ScanQA without situation | Accuracy | 45.27% | Table 3 |
| SQA3D | ScanQA with situation | Accuracy | 46.58% | Table 3 |
| SQA3D | ScanQA with situation and pose auxiliary losses | Accuracy | 47.20% | Table 3 |
| SQA3D | Amateur human | Accuracy | 90.06% | Table 3 |

## 판단 한계

- RGB에서 희박한 1Hz 재구성 또는 포즈 드리프트 처리.
- Long-term memory, 임시 ID, 변경 이벤트 또는 인과적 쓰기 정책.
- 형식화된 레코드 압축, 실제 바이트 예산 또는 장치 내 실행 가능성.
- 동적 장면 또는 SuperMemory-VQA에 대한 결과이다.

## 문제 배경

장면 수준 QA는 구현된 에이전트가 서 있고 향하는 위치를 무시할 수 있다. SQA3D는 텍스트 상황을 이해하고, 3D 스캔에서 암시된 위치와 방향을 기반으로 하며, 해당 상황의 관점에서 공간, 탐색, 상식 및 다중 홉 질문에 대답하는 모델이 필요하다.

## 관련 방법

- 650개의 ScanNet 장면에서 에이전트 상황에 대한 설명과 질문을 수집한다.
- 상황과 질문을 별도로 인코딩한다.
- VoteNet 개체 토큰을 사용한 다음 상황과 질문에 교차 참여한다.
- 선택적으로 답변 분류와 함께 위치 및 회전 헤드를 감독한다.
- 706개의 답변 후보에 대한 정확한 일치 정확도를 평가한다.

## 참고문헌

- Xiaojian Maet al. [SQA3D: Situated Question Answering in 3D Scenes](https://openreview.net/forum?id=IDJx97BC38). ICLR 2023.
- [공식 project page](https://sqa3d.github.io/).
- [공식 repository](https://github.com/SilongYong/SQA3D).
- [공식 dataset release](https://zenodo.org/records/7792397).
- [논문 근거 목록으로 돌아가기](README.md).
