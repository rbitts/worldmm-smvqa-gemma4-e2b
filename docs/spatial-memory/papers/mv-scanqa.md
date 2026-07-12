# Advancing 3D Scene Understanding with MV-ScanQA Multi-View Reasoning Evaluation and TripAlign Pre-training Dataset

| Field | Value |
|---|---|
| Page ID | SM-PAPER-MV-SCANQA |
| Status | Reviewed; code and dataset available |
| Publication | ACM Multimedia 2025, pp. 12973–12980; DOI 10.1145/3746027.3758244 |
| Primary source | [ACM DOI](https://doi.org/10.1145/3746027.3758244) · [arXiv](https://arxiv.org/abs/2508.11058) |
| Official code | [matthewdm0816/MVScanQA](https://github.com/matthewdm0816/MVScanQA) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-008 |

## 30-second summary

MV-ScanQA는 일반적으로 하나의 보기에서 관련 개체를 다룰 수 없도록 질문을 작성한다. 분석 및 LEGO 기준선은 보완적인 관점을 유지하고 눈에 보이는 개체 그룹을 정렬함으로써 측정 가능한 이득을 보여준다. 이는 임시 또는 평생 메모리 벤치마크가 아닌 정적 ScanNet multi-view 벤치마크이다.

## Problem addressed

대부분의 3D 비전 언어 벤치마크는 하나의 유리한 관점에서 해결될 수 있으므로 여러 관점에서 통합을 테스트하지 않는다. MV-ScanQA는 관련 개체에 여러 보기가 필요한 구성 질문을 생성하고 보기 종속 다중 개체 정렬을 위해 TripAlign을 제공한다.

## Relevant method

- 투영된 이미지 중첩이 0.5보다 큰 IoSA를 충족할 때 객체를 목격된 것으로 정의한다.
- 객체 앵커를 공유하는 쌍 소스 질문은 각각 비하위 객체 정보를 제공한다.
- LLM를 사용하여 쌍에서 검증 가능한 질문을 작성한다.
- 백만 개의 `<2D view, visible 3D-object set, text>` 세 쌍으로 TripAlign을 구축한다.
- 2D vision-language model과 3D 감지기에 LEGO를 빌드하고 선택한 각 뷰에 표시되지 않는 개체 제안을 삭제한다.

## Paper-reported evidence

| Dataset | Condition | Metric | Reported result | Location |
|---|---|---|---|---|
| ScanQA / ScanRefer / Nr3D | Questions requiring more than one view | Share | 6% / 4% / 7% | Section 3.1 |
| MV-ScanQA | Questions requiring at least two / at least three views | Share | 68% / 13% | Section 3 |
| MV-ScanQA | LEGO single-view → four-view input | Exact match, all | 30.0 → 34.1 | Table 2 |
| MV-ScanQA | LEGO single-view → four-view input, N≥4 | Exact match | 23.3 → 30.2 | Table 2 |
| ScanQA | From scratch → egocentric extension → TripAlign | Exact match | 25.13 → 27.22 → 28.43 | Table 6 |

승인된 논문에서는 multi-view LEGO에 대해 34.1개의 정확한 일치가 보고됐다. 현재 공식 저장소는 정리된 평가 스크립트에서 33.7을 별도로 보고한다. 이는 별개의 provenance 레코드이므로 병합하면 안 된다.

## What this supports here

- 시야 범위와 광선 다양성은 증거 선택에 속한다.
- 보완 뷰는 공간 복제로 폐기되어서는 안 된다.
- 개체 그룹 정렬은 다중 개체 질문에 대한 관련 교육 목표이다.
- 프로젝트는 광선 인식 랜드마크와 provenance가 각 엔터티를 지원하는 뷰를 유지해야 한다고 추론한다.

## What it does not prove

- 시간 순서, 개체 지속성 또는 인과 스트림 처리.
- 1Hz 단안 기하학, IMU/VIO 안내 또는 평생 재방문 동작.
- 실제 바이트 압축 또는 고정 용량 persistent memory.
- SuperMemory-VQA 전송.

## Project reproduction status

재현되지 않았다. 보고된 정확한 일치 값이 다르기 때문에 향후 비교에서는 종이 지표와 공식 저장소 평가 스크립트를 모두 고정해야 한다.

## References

- Wentao Moet al. [Advancing 3D Scene Understanding with MV-ScanQA Multi-View Reasoning Evaluation and TripAlign Pre-training Dataset](https://doi.org/10.1145/3746027.3758244). ACM 멀티미디어 2025.
- [Official arXiv record](https://arxiv.org/abs/2508.11058).
- [Official project and datasets](https://matthewdm0816.github.io/tripalign-mvscanqa/).
- [Official repository](https://github.com/matthewdm0816/MVScanQA).
- [Back to paper index](README.md).
