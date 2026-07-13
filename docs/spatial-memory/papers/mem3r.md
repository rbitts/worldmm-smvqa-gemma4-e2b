# Mem3R: Streaming 3D Reconstruction with Hybrid Memory via Test-Time Training

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-MEM3R |
| 상태 | 검토 완료; code와 checkpoint 공개 대기 |
| 출판 | arXiv:2604.07279 v1, 2026 preprint |
| 1차 출처 | [arXiv](https://arxiv.org/abs/2604.07279) · [Project page](https://lck666666.github.io/Mem3R/) |
| 공식 code | [lck666666/Mem3R](https://github.com/lck666666/Mem3R) — repository에 code와 checkpoint 공개 예정으로 표기 |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [상위: 논문 목록](README.md) · [아키텍처](../architecture.md) · [추적성](../traceability.md) |
| 프로젝트 claim | [추적성](../traceability.md): C-006 |

## 핵심 결론

- 별도의 빠른 자세, 작업 형상 및 지속적인 QA 메모리 수명 주기를 유지한다.
- 추적 및 장기 검색을 제공해야 하는 단일 전역 recurrent state를 피한다.
- 공급자와 형식화된 decoder가 독립적으로 검증된 후 하이브리드 암시적-명시적 메모리를 테스트한다.

## 근거 상태

재현되지 않았으며 공식 저장소는 현재 자리 표시자이다. 하이브리드 포즈/맵 분리는 디자인 참조로 남아 있다. geometry provider 및 typed record decoder가 독립적인 프로젝트 기준선을 갖기 전에는 통합해서는 안 된다.

## 논문 핵심

Mem3R는 스트리밍 카메라 추적을 기하학적 매핑에서 분리한다. 경량의 암시적 빠른 가중치 MLP는 포즈 관련 상태를 처리하는 반면 고정 크기의 명시적 토큰 상태는 형상을 전달한다. 이는 이 프로젝트의 영구 맵 레코드에서 일시적 포즈 메모리를 분리하는 것을 지원하지만 Mem3R의 "명시적" 토큰은 쿼리 가능한 QA 데이터베이스가 아닌 여전히 모델 상태이다.

## 근거

**보고된 주장.** 이 논문에서는 모델 크기를 793M에서 644M 매개변수로 줄인다고 보고한다. 공식 프로젝트 페이지에는 CUT3R 및 Mem3R 모두에 대해 26개의 FPS가 나열되어 있으며 GPU 메모리는 7,930MiB에서 7,340MiB로 감소한다. 또한 이 논문에서는 TTT3R를 추가하면 500~1,000프레임 시퀀스의 해당 기본 구현에 비해 절대 궤도 오류가 최대 39%까지 감소한다고 보고한다.

**프로젝트 추론.** QA에 필요한 형상은 명시적 영구 레코드로 컴파일되어야 하는 반면 포즈 추적은 암시적이고 수명이 짧은 상태로 유지될 수 있다.

**프로젝트 결과.** 없음. 이 저장소는 Mem3R를 재현하지 않았다.

## 판단 한계

- 해당 토큰 상태는 객체 ID, 좌표 프레임, uncertainty, provenance 또는 결정론적 증명 작업을 노출한다.
- 평생 공간 데이터베이스의 실제 바이트 압축.
- 미래-QA-aware 선택 또는 개체 중심 통합.
- 1Hz AI-glass 스트림, SuperMemory-VQA 또는 기기 내 하드웨어의 성능이다.
- 현재 저장소의 재현성 확인 시 구현 및 체크포인트가 해제되지 않았다.

## 문제 배경

단일 압축 recurrent state는 전역 형상을 유지하고 현재 카메라를 추적해야 한다. 이러한 경쟁적인 역할은 일시적인 망각과 긴 시퀀스의 표류를 유발한다. Mem3R는 순환 추론 메모리를 제한하면서 이들을 분리한다.

## 관련 방법

- 암시적 고속 가중치 MLP 메모리는 카메라 추적을 수행하고 test-time training을 통해 업데이트된다.
- 별도의 고정 크기 토큰 상태는 기하학적 컨텍스트를 유지한다.
- 채널별 모듈은 후보 형상 상태를 이전 상태와 융합한다.
- TTT3R 및 TTSA3R와 같은 기존 CUT3R 업데이트 정책은 하이브리드 분할을 교체하지 않고도 추가할 수 있다.

## 참고문헌

- Changkun Liu, Jiezhi Yang, Zeman Li, Yuan Deng, Jiancong Guo 및 Luca Ballan. [Mem3R: Streaming 3D Reconstruction with Hybrid Memory via Test-Time Training](https://arxiv.org/abs/2604.07279). 2026.
- [공식 project page](https://lck666666.github.io/Mem3R/).
- [공식 repository](https://github.com/lck666666/Mem3R).

[논문 근거 목록으로 돌아가기](README.md)
