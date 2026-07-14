# ScanQA: 3D Question Answering for Spatial Scene Understanding

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-SCANQA |
| 상태 | 검토 완료; code와 dataset 사용 가능 |
| 출판 | CVPR 2022, pp. 19129–19139; arXiv:2112.10482 v3 |
| 1차 출처 | [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2022/html/Azuma_ScanQA_3D_Question_Answering_for_Spatial_Scene_Understanding_CVPR_2022_paper.html) |
| 공식 code | [ATR-DBI/ScanQA](https://github.com/ATR-DBI/ScanQA) |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [상위: 논문 목록](README.md) · [아키텍처](../architecture.md) · [추적성](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-003 |

## 핵심 결론

- 답변은 지속적인 엔터티 ID 및 형상을 기반으로 해야 한다.
- 엔터티 현지화 및 의미 분류에는 명시적인 훈련 손실이 필요하다.
- 관계 질문에 대해 여러 참조 개체가 지원되어야 한다.
- 프로젝트는 작은 엔터티 수준의 기하학 기록이 일반적인 포인트별 기능보다 이 유틸리티를 더 직접적으로 보존할 수 있다고 추론한다.

## 근거 상태

재현되지 않았다. ScanQA는 접지 손실 및 외부 평가 기준이다. ScanQA 데이터 또는 모델 아티팩트가 로컬로 다운로드되지 않았다.

## 논문 핵심

ScanQA는 자유 형식 3D question answering와 참조 객체의 지역화를 결합한다. 절제는 객체 위치 파악과 의미 분류가 모두 답변을 향상시킨다는 것을 보여준다. 이는 명시적인 엔터티 형상 및 접지 손실을 지원하지만 입력은 압축된 인과 스트림이 아닌 이미 완전한 RGB-D 스캔이다.

## 근거

| Dataset | 조건 | Metric | 논문 보고 결과 | 위치 |
|---|---|---|---|---|
| ScanQA | Dataset | Scale | 41,363 questions; 58,191 answers; 800 scenes | Section 3 and Table 2 |
| Test with objects | ScanQA | EM@1 / EM@10 | 23.45 / 56.51 | Table 3 |
| Test without objects | ScanQA | EM@1 / EM@10 | 20.90 / 54.11 | Table 3 |
| Test with objects | Answer loss only | EM@1 / EM@10 | 12.16 / 42.77 | Table 4 |
| Test with objects | Answer + localization | EM@1 / EM@10 | 20.46 / 51.67 | Table 4 |
| Test with objects | Answer + classification + localization | EM@1 / EM@10 | 23.45 / 56.51 | Table 4 |

## 판단 한계

- 밀도가 높은 포인트 클라우드는 long-term memory로 유지되어야 한다.
- 시간적 정체성, 마지막으로 본 쿼리, 이동 이벤트 또는 인과 타당성.
- 1Hz 희소 RGB 감지, 포즈 uncertainty, 실제 바이트 예산 또는 평생 성장.
- 온디바이스 또는 SuperMemory-VQA 성능.

## 문제 배경

이미지 QA는 3D 정렬, 방향 또는 객체 접지를 직접 모델링하지 않는다. ScanQA는 모델에게 전체 실내 3D 스캔에 대한 질문에 답하고 질문에서 참조되는 객체의 3D 경계 상자를 식별하도록 요청한다.

## 관련 방법

- VoteNet 및 PointNet++를 사용하여 컬러 포인트 클라우드에서 개체 제안을 추출한다.
- 양방향 LSTM로 질문을 인코딩한다.
- 객체 제안과 질문 기능을 transformer 레이어와 융합한다.
- 답변 분류, 객체 위치 파악, 객체 분류 및 검출기 손실을 공동으로 훈련한다.
- 자유 형식 답변을 하나 이상의 참조 개체 ID와 연결한다.

## 참고문헌

- Daichi Azumaet al. [ScanQA: 3D Question Answering for Spatial Scene Understanding](https://openaccess.thecvf.com/content/CVPR2022/html/Azuma_ScanQA_3D_Question_Answering_for_Spatial_Scene_Understanding_CVPR_2022_paper.html). CVPR 2022.
- [공식 arXiv record](https://arxiv.org/abs/2112.10482).
- [공식 project, repository, dataset 안내](https://github.com/ATR-DBI/ScanQA).
- [논문 근거 목록으로 돌아가기](README.md).
