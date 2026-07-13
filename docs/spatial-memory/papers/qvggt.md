# QVGGT: Post-Training Quantized Visual Geometry Grounded Transformer

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-QVGGT |
| 상태 | 검토 완료; official implementation 없음 |
| 출판 | CVPR 2026, pages 7536-7545 |
| 1차 출처 | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_QVGGT_Post-Training_Quantized_Visual_Geometry_Grounded_Transformer_CVPR_2026_paper.html) |
| 공식 code | [official project page]에 미공개(https://ddsacu.github.io/QVGGT/) as of 2026-07-11 |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [논문 목록](README.md) · [프로젝트 홈](../README.md) · [문제 정의](../problem.md) · [추적성](../traceability.md) |
| 프로젝트 claim | [추적성](../traceability.md): C-012 |

## 핵심 결론

**프로젝트 추론.** 대규모 시각적 geometry teacher 또는 소형 학생이 배포를 위해 양자화되는 경우 보정은 transformer 레이어 재구성뿐만 아니라 형상 헤드 출력과 크로스 헤드 일관성을 보존해야 한다. 혼합 정밀도는 측정된 블록 감도를 따라야 하며 특수 포즈 관련 토큰은 명시적인 처리를 받아야 한다.

따라서 QVGGT는 이후 모델 배포 단계와 관련이 있다. 이는 현재 우선순위를 대체하지 않는다. 명시적인 유형의 레코드를 생성하고, 인과 기하학 증명을 검증하고, 직렬화된 바이트를 측정한다.

## 근거 상태

**프로젝트 결과.** 재현되지 않는다. QVGGT 코드, 교정 데이터, 양자화된 체크포인트, 하드웨어 프로필 또는 벤치마크 아티팩트가 로컬에 존재하지 않는다. 이 논문은 현재 미래의 형상 인식 PTQ 절제만을 정당화할 수 있다. 주장된 배포 결과를 정당화할 수는 없다.

## 논문 핵심

QVGGT는 12억 매개변수 VGGT 기하학 모델에 사후 훈련 양자화를 적용한다. 이는 블록 감도 기반 혼합 정밀도, 활성화 교정 중 변동이 큰 카메라 및 레지스터 토큰 필터링, PCA- 파생 카메라 정보 보상 토큰, 카메라, 깊이 및 포인트 맵 헤드 전반에 걸친 작업 인식 양자화 스케일 검색을 결합한다.

이 프로젝트에서 QVGGT는 기하학 모델에 기하학 인식 양자화 교정이 필요하다는 증거이다. 기하학 모델을 양자화하면 해당 모델에 의해 생성된 영구 spatial memory가 압축된다는 증거는 아니다.

## 근거

**논문 보고 결과.** 저자는 카메라 포즈 및 재구성 벤치마크 전반에 걸쳐 거의 무손실 W4A16 결과를 보고한다. FP32와 비교하여 초록에서는 세 가지 지오메트리 헤드 모두의 정확성을 유지하면서 메모리가 3~4.9배 감소하고 실제 하드웨어 속도가 최대 2.8배 향상되었다고 보고한다. 평가에는 CO3Dv2 및 RealEstate10K의 camera pose와 7-Scenes 및 Neural RGB-D의 재구성이 포함된다. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_QVGGT_Post-Training_Quantized_Visual_Geometry_Grounded_Transformer_CVPR_2026_paper.pdf) 및 해당 보충 자료를 참조한다.

이 숫자는 저자가 보고한 것이다. 이 저장소에는 QVGGT 결과가 없다.

## 판단 한계

- 모델 가중치 및 활성화 압축은 지속적이지 않는다.-memory compression.
- 이 논문은 사물, 평면, 포털, 랜드마크, 여유 공간, 관계 또는 이벤트를 선택하거나 직렬화하지 않는다.
- 향후 QA 유틸리티 또는 반복 방문 메모리 증가를 최적화하지 않는다.
- G-CUT3R, CUT3R 또는 프로젝트-학생 호환성을 설정하지 않는다.
- 1Hz 평생 스트림, SuperMemory-VQA 또는 대상 AI-glass 하드웨어는 평가하지 않는다.
- 공식 프로젝트 페이지는 현재 코드를 제공하지 않으므로 독립적인 재생산 세부 사항은 불완전한 상태로 남아 있다.

## 문제 배경

일반적인 사후 훈련 양자화는 transformer 블록과 교정 토큰을 너무 균일하게 처리한다. VGGT는 이종 블록 감도와 특수 카메라 및 고분산 활성화 기능을 갖춘 레지스터 토큰을 갖추고 있다. 이상값은 보정을 왜곡하고 camera pose, 깊이 및 포인트 맵 예측 전반에 걸쳐 오류를 전파할 수 있다.

## 관련 방법

**서류 청구.** QVGGT는 세 가지 구성 요소를 사용한다.

1. 블록별 민감도 분석은 취약한 부분에 더 높은 정밀도를 할당한다.
2. 활성화를 수집하는 동안 카메라 및 등록 토큰이 생략된다.
3. 양자화 스케일은 객관적인 결합 레이어로 선택된다.

이는 훈련 후 모델 양자화이다. 새로운 장기 장면 메모리 스키마를 정의하지 않고 가중치와 활성화 정밀도를 변경한다.

## 참고문헌

- 판, 왕, 왕. [QVGGT: Post-Training Quantized Visual Geometry Grounded Transformer](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_QVGGT_Post-Training_Quantized_Visual_Geometry_Grounded_Transformer_CVPR_2026_paper.html). CVPR 2026.
- 저자의 [official QVGGT project page](https://ddsacu.github.io/QVGGT/).
- 저자의 [arXiv record, version 1](https://arxiv.org/abs/2605.31124).
