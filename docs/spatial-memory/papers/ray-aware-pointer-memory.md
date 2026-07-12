# Ray-Aware Pointer Memory with Adaptive Updates for Streaming 3D Reconstruction

| Field | Value |
|---|---|
| Page ID | SM-PAPER-RAY-AWARE-POINTER |
| Status | reviewed; code unavailable |
| Publication | arXiv:2605.05749 v3, 2026 preprint |
| Primary source | [arXiv](https://arxiv.org/abs/2605.05749) |
| Official code | Not linked by the paper or arXiv record as of 2026-07-11 |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-007 |

## 30-second summary

본 논문에서는 공간 포인터에 보기 방향과 소스 시간을 추가한 다음 공간 및 각도 불일치를 사용하여 중복성, 신규성 및 루프 재방문을 구별한다. 유지 또는 교체 정책은 기능 평균화를 방지하고 중복 증가를 제한한다. 이는 1Hz 연관에 대한 광선 인식 랜드마크에 직접적으로 동기를 부여하지만 무작위 교체는 가치 인식 작성자의 기준일 뿐이다.

## Problem addressed

위치 전용 또는 모양 기반 포인터 융합은 서로 다른 표면을 병합하고, 동일한 뷰를 복제하고, 시점이 변경될 때 형상을 불안정하게 만들 수 있다. 또한 긴 스트림에서는 모든 관찰 내용을 유지하지 않고 재방문을 인식해야 한다.

## Relevant method

- 3D 위치, 단위 관측 광선, 특징 embedding 및 포인터당 소스 프레임 인덱스를 저장한다.
- 유클리드 위치와 광선 각도를 함께 비교한다.
- 근접/유사 광선 관측을 중복으로, 근접/다른 광선 관측을 재방문 후보로, 원거리 관측을 새로운 기하학으로 분류한다.
- 중복 기능을 평균화하는 대신 이전 포인터나 새 포인터를 유지한다.
- 공간, 각도, 시간 및 정보 기준으로 선택된 루프 후보에 대한 트리거 포즈 개선.

## Paper-reported evidence

**보고된 주장.** Point3R에 대해 논문에서는 7-Scene Acc/Comp가 `0.085/0.087`에서 `0.035/0.025`로 변경되고 NC가 `0.739`에서 `0.685`로 변경된다고 보고한다. NRGBD에서는 Acc/Comp가 `0.077/0.069`에서 `0.061/0.022`로 변경되고 NC가 `0.835`에서 `0.771`로 변경되는 것으로 보고된다. 따라서 해당 테이블에서는 거리와 완전성이 향상되지만 일반적인 일관성은 감소한다. 보고된 GPU- 메모리 수치는 직렬화된 포인터 바이트가 아닌 예약된 런타임 메모리이다.

**프로젝트 추론.** 광범위한 기준선 연관 및 재방문 감지를 위해 위치, viewing ray 및 타임스탬프를 함께 테스트해야 한다.

**프로젝트 결과.** 없음. 이 저장소는 해당 방법을 재현하지 않았으며 확인 시 공식 코드가 링크되지 않았다.

## What this supports here

- relocalization 랜드마크에 광선 또는 뷰콘 정보를 추가한다.
- 다른 각도에서 드러난 새로운 표면과 동일한 위치의 중복성을 구별한다.
- 학습된 바이트당 값 대체에 대한 간단한 기준으로 유지 또는 대체를 사용한다.
- 근처의 모든 포인터를 동일한 관찰로 처리하는 대신 루프 검증을 트리거한다.

## What it does not prove

- 무작위 유지 또는 교체는 QA-중요한 증거를 보존한다.
- 더 나은 정상적인 일관성; 위에서 보고된 NC 값은 불리한 방향으로 이동한다.
- 엔터티 수준 유형의 레코드, 결정론적 기하학 증명 또는 provenance-완전 QA.
- 직렬화된 고정 바이트 메모리, 1Hz AI-유리 성능, 며칠간 보존 또는 SuperMemory-VQA 개선.
- 공개 코드의 재현성 공식적인 구현이 연결되지 않았다.

## Project reproduction status

공식 구현 또는 별도로 검증된 클린룸 기준에서는 재현되지 않았으며 현재 차단되어 있다. 저장소 URL를 추론하지 마세요. 안전한 첫 번째 프로젝트 실험은 논문의 전체 포즈 미세 조정 시스템이 아니라 포인터 튜플과 결정론적 연관 규칙이다.

## References

- Feifei Li, Qi Song, Chi Zhang 및 Rui Huang. [Ray-Aware Pointer Memory with Adaptive Updates for Streaming 3D Reconstruction](https://arxiv.org/abs/2605.05749). arXiv v3, 2026.

[Back to paper index](README.md)
