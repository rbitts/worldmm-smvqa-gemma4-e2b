# LONG3R: Long Sequence Streaming 3D Reconstruction

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-LONG3R |
| 상태 | 검토 완료; code stub만 존재 |
| 출판 | ICCV 2025; arXiv:2507.18255 v1 |
| 1차 출처 | [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_LONG3R_Long_Sequence_Streaming_3D_Reconstruction_ICCV_2025_paper.html) · [arXiv](https://arxiv.org/abs/2507.18255) · [Project page](https://zgchen33.github.io/LONG3R/) |
| 공식 code | [zgchen33/LONG3R](https://github.com/zgchen33/LONG3R/) — repository에 code 공개 예정으로 표기 |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [상위: 논문 목록](README.md) · [아키텍처](../architecture.md) · [추적성](../traceability.md) |
| 프로젝트 claim | [추적성](../traceability.md): C-002, C-005 |

## 핵심 결론

- 고정 복셀을 적응형 해상도 공간 슬롯과 비교한다.
- 작업 메모리 쓰기 게이트에서 기하학적 관련성 및 중복성을 사용한다.
- 반복 방문으로 인해 동등한 공간 상태 추가가 중단되는지 평가한다.

이는 SuperMemory-VQA에 대한 LONG3R 결과가 아니라 방법에서 파생된 프로젝트 가설이다.

## 근거 상태

재현되지 않았다. 설계 참조 및 향후 기준으로만 사용한다. 코드, 체크포인트, 데이터세트 다이제스트, 평가 출력이 실험 페이지에 기록될 때까지 LONG3R 파생 성능을 프로젝트 결과로 보고하지 마세요.

## 논문 핵심

LONG3R는 긴 이미지 시퀀스에 대한 반복 스트리밍 재구성 모델이다. 디코딩하기 전에 메모리를 게이트화하고 장면 전체에서 해상도를 변경하면서 중복된 공간 정보를 정리하는 3D 시공간 메모리를 유지한다. 이는 일시적 재구성 프런트엔드에서 적응형 기하학 메모리를 테스트한다는 증거이지, 잠재 메모리가 충분한 장기 QA 데이터베이스라는 증거는 아니다.

## 근거

**신고된 주장.** 기본 실험에서는 10프레임 short-term memory와 3,000개의 장기 토큰을 사용한다. 7-Scenes에서 메모리 게이팅이 토큰을 27% 줄이고 처리량을 18.0에서 21.4 FPS로 증가시키는 것으로 보고서에 나와 있다. 200프레임 복제 시퀀스에서는 Spann3R의 경우 `16.29/4.02`, CUT3R의 경우 `28.30/6.61`와 비교하여 `11.93/2.73`의 Acc/Comp를 보고한다. 이 방법은 `t+1`의 기능을 사용하여 프레임 `t`를 예측하므로 이는 제로 예측 처리가 아닌 1프레임 지연 스트리밍이다.

**프로젝트 추론.** 메모리 게이팅 및 적응형 공간 해상도는 고정 복셀 및 고정 슬롯 과도 형상 메모리에 대한 합리적인 비교 지점이다.

**프로젝트 결과.** 없음. 이 저장소는 LONG3R를 재현하지 않았다.

## 판단 한계

- 그 잠재된 재건 관심은 미래-QA-중요한 사실을 식별한다.
- 메모리는 명시적인 엔터티, 관계, 이벤트 또는 provenance 데이터베이스이다.
- 평생 메모리에 대한 실제 직렬화된 바이트 절약.
- 1Hz 단안 AI-유리 감지, 며칠 간의 재방문 또는 SuperMemory-VQA 미만의 정확도.
- 기기 내 타당성. 공식 코드 저장소에는 확인 시 릴리스된 구현이 포함되어 있지 않는다.
- 제로 예측 인과성; 세련된 예측은 다음 프레임의 특징을 사용한다.

## 문제 배경

기존 multi-view 재구성 시스템은 오프라인 전역 최적화가 필요하거나 recurrent state가 짧은 시퀀스 이상으로 사용되면 품질이 저하된다. LONG3R는 모든 이전 프레임에서 추론 비용을 늘리지 않고 더 긴 스트림에 대한 온라인 재구성을 목표로 한다.

## 관련 방법

- recurrent model는 최근 프레임을 단기 임시 메모리로 사용하여 모든 새로운 관찰에 대한 메모리를 업데이트한다.
- 주의 기반 메모리 게이팅은 세련된 decoder가 관련 항목을 사용하기 전에 관련 항목을 선택한다.
- 개선된 듀얼 소스 decoder는 선택된 메모리와 인접 프레임 기능을 결합한다.
- 장기 포인트맵 패치는 적응형 복셀로 그룹화된다. 복셀당 가장 높은 누적 관심을 가진 하나의 토큰이 유지된다.
- 2단계 커리큘럼 교육은 단기 학습과 장기 학습을 분리한다.

## 참고문헌

- Zhuoguang Chen, Minghui Qin, Tianyuan Yuan, Zhe Liu 및 Hang Zhao. [LONG3R: Long Sequence Streaming 3D Reconstruction](https://arxiv.org/abs/2507.18255). ICCV 2025.
- [ICCV 2025 open-access publication](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_LONG3R_Long_Sequence_Streaming_3D_Reconstruction_ICCV_2025_paper.html).
- [공식 project page](https://zgchen33.github.io/LONG3R/).
- [공식 repository](https://github.com/zgchen33/LONG3R/).

[논문 근거 목록으로 돌아가기](README.md)
