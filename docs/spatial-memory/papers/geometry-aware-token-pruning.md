# Seeing Once is Enough? Online Geometry-Aware Token Pruning for 3D Question Answering

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-GEOMETRY-AWARE-TOKEN-PRUNING |
| 상태 | 검토 완료; 최신 workshop preprint |
| 저자 | Ruei-Chi Lai, Bolivar Solarte, Chin-Hsuan Wu, Yi-Hsuan Tsai, Min Sun |
| 출판 | ICLR 2026 Workshop on Efficient Spatial Reasoning |
| 확인 version | arXiv:2607.04079v1, 2026-07-05 |
| 1차 출처 | [arXiv](https://arxiv.org/abs/2607.04079) · [OpenReview](https://openreview.net/forum?id=jnDbE6cV2D) |
| 공식 code | 확인한 primary source에 미공개 또는 link 없음 |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [상위: 논문 목록](README.md) · [아키텍처](../architecture.md) · [추적성](../traceability.md) |
| 프로젝트 claim | [추적성](../traceability.md): C-011 |

## 핵심 결론

- 시각적 증거를 유지하거나 중복된 다운스트림 후보를 생성하기 전에 형상의 참신함을 계산한다.
- 이미 관찰된 영역을 거부할 수 있는 인과 범위 맵을 유지한다.
- QA 유틸리티에 대한 쿼리 독립적 보완으로 형상 중첩 선택을 테스트한다.
- 온라인 처리에서 토큰 비용과 형상에 민감한 QA를 모두 평가한다.

## 근거 상태

재현되지 않았다. 소스 ID가 확인되었지만 이 저장소에는 RGB-D 복셀 정리기 또는 보고된 벤치마크 실행이 없다. 현재 작성자는 명시적 레코드와 실제 바이트 예산을 사용하므로 비교에는 영구 작성자를 교체하는 대신 별도의 임시 입력 제거가 필요하다.

## 논문 핵심

이 작업 프로젝트는 RGB-D 관찰을 공유 복셀 공간에 배치하고 이전 뷰에서 이미 다룬 영역에 해당하는 이미지 토큰을 제거한다. 이는 전체 장면에 대한 오프라인 패스를 요구하지 않고도 지오메트리 중첩이 온라인 3D-QA 입력 토큰을 줄일 수 있다는 직접적인 증거이다. 영구 spatial memory 레코드가 아닌 VLM 입력을 정리한다.

## 근거

저자는 토큰 사용량이 최대 50% 감소했다고 보고했다. Qwen2.5-VL-7B 및 Qwen3-VL-8B에 적용된 이 방법은 ScanQA, SQA3D 및 OpenEQA-HM3D에서 향상된 결과를 보고한다. arXiv 초록은 깊이와 포즈 없이 단안 입력에 대한 결과를 설정하지 않는다.

이는 이 저장소에서 재현한 결과가 아닌 논문 결과이다.

## 판단 한계

- 중복 제거된 이미지 토큰은 향후 알려지지 않은 질문에 충분하다.
- 이 방법은 신뢰할 수 있는 깊이와 자세 없이 약 1Hz 단안 RGB에서 작동한다.
- VLM 토큰 감소는 간결하고 명시적인 영구 맵을 생성한다.
- 이러한 정적 중첩만으로도 개체 이동 및 상호 작용 이벤트가 보존된다.
- 워크숍 결과가 SuperMemory-VQA로 전송된다.

## 문제 배경

Multi-view 3D question answering는 여러 개의 겹치는 이미지 토큰을 다중 모드 language model로 보냅니다. 기존 프레임 선택 및 토큰 병합 방법은 종종 전체 시퀀스를 오프라인으로 사전 처리하므로 온라인 스트림에 맞지 않는다.

## 관련 방법

- 깊이와 camera pose를 사용하여 각 프레임을 공유 복셀 공간에 투영한다.
- 현재 뷰와 이전에 관찰된 영역 간의 공간적 중복을 감지한다.
- language model에 들어가기 전에 중복 이미지 토큰을 정리한다.
- 기존 다중 모드 모델을 사용하여 교육 없이 온라인으로 운영할 수 있다.

## 참고문헌

- [논문 목록](README.md)
- [arXiv:2607.04079](https://arxiv.org/abs/2607.04079)
- [OpenReview](https://openreview.net/forum?id=jnDbE6cV2D)
