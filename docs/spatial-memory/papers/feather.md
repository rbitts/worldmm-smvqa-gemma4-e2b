# Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-FEATHER |
| 상태 | 검토 완료 |
| Method 명칭 | FEATHER: Fast and Effective Acceleration wiTH Ensemble cRiteria |
| 저자 | Mark Endo, Xiaohan Wang, Serena Yeung-Levy |
| 출판 | ICCV 2025 |
| 확인 version | arXiv:2412.13180v2, 2025-07-31 |
| 1차 출처 | [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Endo_Feather_the_Throttle_Revisiting_Visual_Token_Pruning_for_Vision-Language_Model_ICCV_2025_paper.html) · [arXiv](https://arxiv.org/abs/2412.13180) |
| 공식 code | [markendo/FEATHER](https://github.com/markendo/FEATHER) |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [상위: 논문 목록](README.md) · [아키텍처](../architecture.md) · [추적성](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-011 |

## 핵심 결론

- 지역, 개체 및 좌표 적용 범위를 작성자 가드레일로 처리한다.
- QA 점수와 함께 현지화, 관계 회상 및 좌표 오류를 보고한다.
- 선택기를 수용하기 전에 가지치기 실험에 공간 유지 히트맵 또는 적용 범위 진단을 포함한다.
- 학습된 유틸리티가 낮은 경우에도 쿼리에 구애받지 않는 지오메트리 코어를 유지한다.

## 근거 상태

재현되지 않았다. 현재 테스트에서는 명시적인 ID, 기하학, 인과관계 및 바이트 제한을 적용하지만 FEATHER 토큰 정리 구현 또는 현지화 벤치마크가 실행되지 않았다.

## 논문 핵심

FEATHER는 공격적인 초기 시각적 token pruning이 광범위한 VLM 벤치마크에서 성공적으로 보일 수 있지만 현지화에 민감한 작업은 실패하고 이미지 일부에서 대부분의 토큰을 삭제할 수 있음을 보여준다. 광범위한 적용 범위를 보존하기 위해 초기 균일 샘플링과 함께 다단계 가지치기를 사용한다. 이 논문은 평균 QA 성능만으로는 spatial memory에 대한 안전하지 않은 선택 목표라는 경고이다.

## 근거

저자는 이전 가속 접근 방식이 많은 벤치마크에서 높은 점수를 유지함에도 불구하고 이미지 상단 근처의 대부분의 토큰을 잘라낸다고 보고한다. 유사한 컴퓨팅 절감 효과로 FEATHER는 비전 중심 현지화 벤치마크에서 해당 접근 방식에 비해 5배 이상의 성능 향상을 보고한다.

이는 이 저장소에서 재현한 결과가 아닌 논문 결과이다.

## 판단 한계

- 균일한 이미지 적용 범위는 3D 전 세계 적용 범위에 최적이다.
- VLM 가지치기 정책은 일시적인 변경 사항이나 인스턴스 ID를 유지한다.
- 해당 현지화는 SuperMemory-VQA로 이전된다.
- 일시적인 토큰 적용 범위는 지속적인 직렬화된 메모리 증가를 제어한다.

## 문제 배경

초기 token pruning은 비전 언어 모델을 가속화하지만 공통 기준은 공간적으로 편향된 보존을 생성할 수 있다. 세분화된 현지화를 요구하지 않는 벤치마크는 이러한 실패를 숨길 수 있다.

## 관련 방법

- 보유된 시각적 토큰의 공간적 분포를 진단한다.
- 광범위한 이미지 적용 범위를 보장하려면 초기 균일 샘플링을 사용한다.
- 관련 세부 사항을 보존하려면 여러 기준을 사용하여 이후 가지치기 단계를 적용한다.
- VLM 작업을 집계하는 것뿐만 아니라 비전 중심의 현지화 작업으로 평가한다.

## 참고문헌

- [논문 목록](README.md)
- [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/html/Endo_Feather_the_Throttle_Revisiting_Visual_Token_Pruning_for_Vision-Language_Model_ICCV_2025_paper.html)
- [arXiv:2412.13180](https://arxiv.org/abs/2412.13180)
- [공식 project page](https://web.stanford.edu/~markendo/projects/feather)
- [공식 code](https://github.com/markendo/FEATHER)
