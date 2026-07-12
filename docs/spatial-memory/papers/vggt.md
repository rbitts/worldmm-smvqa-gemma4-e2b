# VGGT: Visual Geometry Grounded Transformer

| Field | Value |
|---|---|
| Page ID | SM-PAPER-VGGT |
| Status | Reviewed from primary sources; not reproduced in this project |
| Publication | CVPR 2025 Best Paper, pp. 5294–5306 |
| Primary source | [CVPR Open Access paper](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html) |
| Official code | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) |
| Last checked | 2026-07-11 |
| Project links | [Paper index](README.md), [Architecture](../architecture.md), [Roadmap](../roadmap.md) |
| Project claims | [Traceability](../traceability.md): C-001 |

## 30-second summary

VGGT는 약 12억 개의 매개변수 feed-forward transformer로, camera parameters, 깊이 맵, 포인트 맵 및 포인트 트랙을 하나부터 수백 개의 이미지까지 공동으로 예측한다. 이는 프레임별 관심과 글로벌 관심을 번갈아 가며 필수 test-time optimization 없이 직접 사용 가능한 multi-view 형상을 생성한다.

## Problem addressed

전통적인 재구성에는 다단계 시각적 기하학과 반복적인 최적화가 필요하다. 쌍으로 학습된 모델에는 여전히 많은 이미지에 대한 전역 정렬이 필요하다. VGGT는 하나의 대규모 다중 작업 네트워크가 단일 정방향 전달에서 장면의 핵심 3D 속성을 공동으로 추론할 수 있는지 묻는다.

## Relevant method

DINOv2 패치 토큰, 카메라 토큰 및 등록 토큰은 24개의 교대로 프레임별 및 글로벌 어텐션 레이어에 들어갑니다. 별도의 카메라와 DPT 헤드는 내장, 외부, 깊이, 포인트 맵 및 uncertainty를 예측한다. 추적 헤드는 교차 시점 대응을 위해 조밀한 기능을 사용한다. 첫 번째 카메라는 world frame을 정의한다.

모델은 2~24개의 샘플링된 뷰에 대한 카메라, 깊이, 포인트 맵 및 추적 손실에 대해 공동으로 훈련된다. 이 논문은 9일 동안 64개의 A100 GPU에서 160,000번의 반복에 대한 교육을 보고한다.

## Paper-reported evidence

이는 이 저장소의 결과가 아닌 논문 결과이다.

| Dataset or condition | Metric | Reported result | Source location |
|---|---|---:|---|
| RealEstate10K, unseen, ten views, one H100 | AUC at 30 / runtime | 85.3 / about 0.2 s | Table 1, paper p. 7 |
| CO3Dv2, ten views, one H100 | AUC at 30 / runtime | 88.2 / about 0.2 s | Table 1, paper p. 7 |
| ETH3D point-map estimation, depth plus camera heads | Accuracy / completeness / overall / runtime | 0.873 / 0.482 / 0.677 / about 0.2 s | Table 3, paper p. 7 |
| Ten 336 by 518 frames, H100, Flash Attention 3 | Backbone runtime / peak memory | 0.14 s / 3.63 GB | Table 9, paper p. 10 |
| One hundred 336 by 518 frames, same setting | Backbone runtime / peak memory | 3.12 s / 21.15 GB | Table 9, paper p. 10 |

표 9는 또한 8.75초 및 40.63 GB에서 200프레임을 보고한다. 이는 강력한 짧은 시퀀스 처리량에도 불구하고 뷰 세트가 커짐에 따라 비반복 전체 뷰 디자인이 비용이 많이 든다는 직접적인 증거이다.

## What this supports here

**논문 주장:** 공동 feed-forward 예측은 필수 전역 정렬 없이 희박한 multi-view 입력에서 고품질 카메라, 깊이, 포인트 맵 및 트랙을 제공할 수 있다.

**프로젝트 추론:** VGGT는 CUT3R 파생 기하학에 대한 유용한 비반복 교사 비교기이다. 입력된 레코드 품질이 반복 제공자에 따라 달라지는지 아니면 주로 제공자의 multi-view 기하학 정확도에 따라 달라지는지 여부를 테스트할 수 있다.

출력 헤드는 또한 포즈, 미터법에 일치하는 깊이, 포인트 맵, 트랙 및 uncertainty와 같은 유용한 교사 대상을 정의한다.

## What it does not prove

- 스트리밍 고정 상태 메모리가 아니며 선택한 뷰 세트를 공동으로 처리한다.
- 평생 입력에 대해 제한된 메모리를 설정하지 않는다. 표 9는 더 많은 프레임으로 인해 런타임 및 GPU 메모리가 증가하는 것을 보여준다.
- 객체 ID, 시간적 유효성, 유형화된 공간 레코드 또는 QA 증명을 제공하지 않는다.
- 1Hz AI-안경 감지, SuperMemory-VQA 또는 기기 내 사용은 평가하지 않는다.
- 강력한 재구성 메트릭은 그 자체로 미래 QA 유틸리티 또는 저장된 바이트당 값을 설정하지 않는다.

## Project reproduction status

VGGT는 이 저장소에 설치, 다운로드 또는 실행되지 않는다. 모델 가중치, 예측 또는 벤치마크 아티팩트가 로컬에 존재하지 않는다. 이는 구현된 레인이 아니라 계획된 교사/제공자 비교로 남아 있다.

## References

- [CVPR 2025 Open Access record](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html)
- [Official project page](https://vggt.robots.ox.ac.uk/)
- [Official code](https://github.com/facebookresearch/vggt)
- [Official arXiv record](https://arxiv.org/abs/2503.11651)

[Back to paper index](README.md)
