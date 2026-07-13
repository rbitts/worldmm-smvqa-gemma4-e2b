# LongSpace: Exploring Long-Horizon Spatial Memory from Perception to Recall in Video

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-LONGSPACE |
| 상태 | 검토 완료; 최신 preprint; 확인 시 announced code 사용 불가 |
| 출판 | arXiv:2606.05677 v1, 2026-06-04 |
| 1차 출처 | [arXiv](https://arxiv.org/abs/2606.05677) |
| 공식 code | [Announced repository](https://github.com/ShiqiangLang/LongSpace), 확인 시 사용 불가 |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [상위: 논문 목록](README.md) · [아키텍처](../architecture.md) · [추적성](../traceability.md) |
| 프로젝트 claim | [추적성](../traceability.md): C-008 |

## 핵심 결론

- 1fps에서는 최근 프레임만 유지하는 것보다 청크 간 증거가 더 유용하다.
- 메모리 구성 및 검색은 원시 용량과 별도로 제거할 가치가 있다.
- 형상은 초기 모델 레이어에 선택적으로 주입될 수 있다.
- 상태 변화와 시간적 적용 범위는 유용한 후보 작성자 신호이다.

## 근거 상태

재현되지 않았다. 종이 링크 저장소는 확인 시 404를 반환하므로 구현 주장은 공식 릴리스가 제공될 때까지 종이로만 유지된다.

## 논문 핵심

LongSpace는 긴 룸 투어 벤치마크와 3D 구조를 초기 decoder 레이어에 주입하는 동시에 비디오 청크 전체에서 계층적 KV 메모리를 유지하는 모델을 도입한다. 장기 메모리 및 레이어 인식 절제는 조직화된 크로스 청크 증거가 최근 프레임 샘플링보다 더 중요하다는 것을 보여준다. 해당 메모리는 감사 가능한 유형의 지오메트리 데이터베이스가 아닌 잠재 KV 상태로 유지된다.

## 근거

| Dataset | 조건 | Metric | 논문 보고 결과 | 위치 |
|---|---|---|---|---|
| LongSpace-Bench | Dataset | Scale | 445 videos; about 159 hours; 4,073 QA pairs; 21.4-minute average | Table 1 and Section 3.2 |
| LongSpace-Bench | LongSpace | Overall | 49.2 | Table 2 |
| LongSpace-Bench | Uniform 32 frames → recent windows → long memory | Overall | 36.1 → 37.7 → 49.2 | Figure 5 and Section 5 |
| LongSpace-Bench | Layer-agnostic → layer-aware memory | Overall | 41.8 → 49.2 | Table 5 |
| LongSpace-Bench | Long memory over uniform, short / medium / long videos | Improvement | +4.8 / +12.8 / +15.1 points | Figure 5 and Section 5 |

## 판단 한계

- 명시적인 객체 ID, 좌표계, uncertainty, provenance 또는 기하학 증명.
- 실제 직렬화된 바이트 최적화; KV 용량은 영구 데이터베이스 크기가 아니다.
- 활성 탐색, 웨어러블 IMU/VIO, 며칠 간의 재방문 또는 기기 내 실행.
- SuperMemory-VQA 성능.

## 문제 배경

장편 비디오 모델은 몇 분 단위로 구분된 관찰에서 레이아웃, 경로, 시점 변경 및 객체 상태를 유지해야 한다. 균일한 프레임 샘플링과 최근 창은 먼 공간 증거를 잃거나 중복 토큰으로 희석한다.

## 관련 방법

- 연속되는 실제 룸 투어 동영상을 통해 LongSpace-Bench를 구축한다.
- 4프레임이 겹치는 32프레임 청크에서 1fps로 비디오를 처리한다.
- 3D 형상 특징을 정렬하고 처음 8개 decoder 레이어에 구조적 잔차를 주입한다.
- 계층적 KV 기억을 감각 기억, 작업 기억, 장기 기억 역할로 나눕니다.
- 현저성, 상태 변경, 최신성, 임시 적용 범위 및 역할별 예산을 사용하여 메모리를 선택하고 압축한다.
- 각 질문에 대해 세그먼트 수준 증거를 검색한 다음 토큰 수준 증거를 검색한다.

## 참고문헌

- Shiqiang Langet al. [LongSpace: Exploring Long-Horizon Spatial Memory from Perception to Recall in Video](https://arxiv.org/abs/2606.05677). arXiv:2606.05677 v1.
- [Paper-announced repository](https://github.com/ShiqiangLang/LongSpace), 2026년 7월 11일에는 사용할 수 없다.
- [논문 근거 목록으로 돌아가기](README.md).
