# MeMix: Writing Less, Remembering More for Streaming 3D Reconstruction

| Field | Value |
|---|---|
| Page ID | SM-PAPER-MEMIX |
| Status | reviewed; code available |
| Publication | arXiv:2603.15330 v1, 2026 preprint |
| Primary source | [arXiv](https://arxiv.org/abs/2603.15330) · [Project page](https://dongjiacheng06.github.io/MeMix/) |
| Official code | [dongjiacheng06/MeMix](https://github.com/dongjiacheng06/MeMix) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-002, C-005, C-011 |

## 30-second summary

MeMix는 반복 스트리밍 3D 재구성을 위한 훈련 없는 업데이트 정책이다. 하나의 recurrent state를 메모리 패치로 나누고, 가장 적게 정렬된 패치를 업데이트하고, 나머지는 정확하게 보존한다. 선택적 작업 상태 업데이트를 위한 강력한 기준을 제공하지만 형식화된 명시적 레코드를 생성하지도 않고 저장된 바이트당 향후 QA 값을 최적화하지도 않는다.

## Problem addressed

고정된 recurrent state를 균일하게 다시 작성하면 드리프트가 누적되고 긴 스트림에 걸쳐 유용한 기록이 삭제될 수 있다. MeMix는 fine-tuning, 추가 학습 매개변수 또는 시퀀스 길이 종속 추론 메모리 없이 파괴적인 쓰기를 줄이는 것을 목표로 한다.

## Relevant method

- recurrent state를 독립 메모리 패치로 분할한다.
- 후보 상태 패치와 현재 이미지 토큰 간의 점수 정렬.
- 바이너리 마스크를 사용하여 하단 k, 최소 정렬 패치만 업데이트한다.
- 선택되지 않은 패치를 정확하게 보존한다.
- 백본을 재교육하지 않고 모듈을 CUT3R, TTT3R 또는 TTSA3R에 적용한다.

## Paper-reported evidence

**보고된 주장.** 논문의 표준 재구성 벤치마크 및 일치하는 백본 설정 전반에 걸쳐 MeMix는 300~500프레임 7장면 스트림에서 완전성 오류가 평균 15.3% 감소하고 최대 40.0% 감소했다고 보고한다. 공식 프로젝트 페이지는 또한 테스트된 백본에 대해 변경되지 않은 최대 GPU 메모리를 보고한다. CUT3R의 경우 MeMix가 있거나 없는 경우 모두 14.39 대 14.13 FPS 및 5.31 GB를 나열한다.

기본 설정은 768개의 상태 토큰 중 708개, 즉 약 92.2%를 업데이트한다. 따라서 파괴적인 재작성은 줄어들지만 해당 설정에서는 강력한 희소 쓰기 정책이 아니다.

**프로젝트 추론.** Bottom-k 패치 업데이트는 작업별 이벤트 또는 바이트당 값 게이트에 대해 훈련이 필요 없는 유용한 기준선이다.

**프로젝트 결과.** 없음. 이 저장소는 MeMix를 재현하지 않았다.

## What this supports here

- 최종 재구성 품질만 측정하는 것이 아니라 분당 쓰기 수를 측정한다.
- 관찰 내용이 유지된 상태와 일치하지 않는 경우에만 건너뛰기, 업데이트, 삽입 및 만료 결정을 테스트한다.
- 전체 recurrent state를 다시 작성하는 대신 안정적인 작업 메모리 영역을 보존한다.

이 프로젝트에서 제안한 학습된 QA-aware 게이트는 MeMix 기여가 아닌 확장이다.

## What it does not prove

- 최소한의 정렬은 기하학의 참신함, 의미론적 중요성 또는 미래의 QA 유틸리티와 같다.
- 영구 메모리 바이트 감소; 상수 GPU 상태는 직렬화된 저장소 측정이 아니다.
- 엔터티 ID, 메트릭 증명, uncertainty, 임시 유효성 또는 provenance 지원.
- 1Hz 감지, 수일 메모리, AI-유리 하드웨어 또는 SuperMemory-VQA 미만의 성능이다.
- 수천 프레임 및 킬로미터 규모의 동작; 이 논문은 두 정권 모두 평가하지 않는다.

## Project reproduction status

재현되지 않았다. 릴리스된 구현을 향후 선택적 업데이트 기준으로 처리한다. 유효한 프로젝트 결과를 얻으려면 경쟁 작성자와 동일한 교사, 프레임 매니페스트, byte budget 및 QA 평가가 필요하다.

## References

- Jiacheng Dong, Huan Li, Sicheng Zhou, Wenhao Hu, Weili Xu 및 Yan Wang. [MeMix: Writing Less, Remembering More for Streaming 3D Reconstruction](https://arxiv.org/abs/2603.15330). 2026.
- [Official project page](https://dongjiacheng06.github.io/MeMix/).
- [Official repository](https://github.com/dongjiacheng06/MeMix).

[Back to paper index](README.md)
