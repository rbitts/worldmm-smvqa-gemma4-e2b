# Good Token Hunting: A Hitchhiker's Guide to Token Selection for Visual Geometry Transformers

| Field | Value |
|---|---|
| Page ID | SM-PAPER-GOOD-TOKEN-HUNTING |
| Status | Reviewed; recent preprint |
| Authors | Shuhong Zheng, Michael Oechsle, Erik Sandström, Marie-Julie Rakotosaona, Federico Tombari, Igor Gilitschenski |
| Publication | arXiv preprint |
| Version checked | arXiv:2605.23892v1, 2026-05-22 |
| Primary source | [arXiv](https://arxiv.org/abs/2605.23892) · [Project page](https://zsh2000.github.io/good-token-hunting.github.io/) |
| Official code | [zsh2000/gotohunt](https://github.com/zsh2000/gotohunt) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-011 |

## 30-Second Summary

Good Token Hunting는 두 단계(다양한 프레임을 먼저 선택한 다음 보유된 프레임 내의 중복 토큰)에서 키/값 토큰을 선택하여 시각적 기하학 변환기에서 전역 주의 비용을 줄이다. 이는 multi-view 지오메트리 계산에 장면 적용 범위와 레이어 인식 선택이 중요하다는 직접적인 증거이다. 선택한 토큰이 내구성이 있거나 QA-충분한 메모리를 형성한다는 증거는 아니다.

## Problem Addressed

긴 multi-view 시퀀스에 대한 글로벌 관심은 입력 길이에 따라 2차적으로 증가한다. 균일한 가지치기를 수행하면 이후 레이어에 필요한 개별 관측점이나 형상을 삭제할 수 있다.

## Relevant Method

- 프레임 간 선택을 통해 장면을 포괄하는 다양한 보기 세트가 유지된다.
- 프레임 내 선택은 유지된 뷰에서 추가 중복성을 제거한다.
- 프레임 내 정책은 레이어를 인식하며 글로벌 어텐션 엔트로피를 사용한다.
- 선택은 각 쿼리가 처리하는 키/값 토큰을 제한한다. 지속적인 세계 모델을 직렬화하지 않는다.

## Paper-Reported Evidence

저자는 기본 모델의 형상 성능을 유지하거나 때로는 향상시키면서 500개의 이미지가 있는 장면에서 85% 이상의 가속을 보고한다. 공식 프로젝트 페이지에는 7-Scenes, Neural RGB-D 및 TUM-Dynamics에 대한 카메라 포즈 실험과 최대 500개 프레임 중 25개 프레임의 토큰에 참여하는 각 쿼리가 포함된 재구성 예제도 보고되어 있다.

이는 이 저장소에서 재현한 결과가 아닌 논문 결과이다.

## What This Supports Here

- 뷰 내 토큰 감소와 뷰 수준 다양성을 분리한다.
- 지역적 신뢰도나 주목도 점수에만 의존하기보다는 후보자의 유용성에 공간적 적용 범위와 참신성을 포함시키십시오.
- 처리량뿐만 아니라 형상 및 포즈 측정 기준을 기준으로 선택을 평가한다.
- 임시 기하학 특징을 잘라낼 때 레이어 또는 표현 단계를 관련성 있게 처리한다.

## What It Does Not Prove

- 유지된 주의 토큰은 미래의 기하학적 기반 QA를 보존한다.
- 이 방법은 약 1Hz 단안 AI-유리 입력에서 작동한다.
- 이러한 주의 계산 감소는 지속적으로 직렬화된 바이트를 줄이다.
- 보고된 서버 측 속도 향상이 기기 내 학생에게 전송된다.

## Project Reproduction Status

재현되지 않았다. 저장소에는 독립적인 인과, 실제 바이트 작성자 및 선형 선택기 기준이 있지만 GoToHunt 구현이나 벤치마크 실행은 없다. 일시적 주의 선택에서 persistent memory로의 전환은 프로젝트 가설로 남아 있다.

## References

- [Paper index](README.md)
- [arXiv:2605.23892](https://arxiv.org/abs/2605.23892)
- [Official project page](https://zsh2000.github.io/good-token-hunting.github.io/)
- [Official code](https://github.com/zsh2000/gotohunt)
