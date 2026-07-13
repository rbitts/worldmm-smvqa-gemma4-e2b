# Perceiver: General Perception with Iterative Attention

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-PERCEIVER |
| 상태 | 검토 완료 |
| 저자 | Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, Joao Carreira |
| 출판 | ICML 2021, PMLR 139 |
| 확인 version | arXiv:2103.03206v2, 2021-06-23 |
| 1차 출처 | [PMLR](https://proceedings.mlr.press/v139/jaegle21a.html) · [arXiv](https://arxiv.org/abs/2103.03206) |
| 공식 code | [DeepMind Perceiver](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver) |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [상위: 논문 목록](README.md) · [아키텍처](../architecture.md) · [추적성](../traceability.md) |
| 프로젝트 claim | [추적성](../traceability.md): C-010 |

## 핵심 결론

- 제한된 학습된 잠재 배열을 통해 대규모 일시적 기능 세트를 디코딩한다.
- 원시 포인트 또는 패치 수와 별도로 후보 수를 유지한다.
- 슬롯 자체를 데이터베이스 스키마로 처리하는 대신 별도의 개체, 구조, 랜드마크 및 이벤트 헤드를 잠재 슬롯에 ​​연결한다.

## 근거 상태

재현되지 않았다. Perceiver decoder가 구현되지 않았다. 저장소는 현재 명시적인 경험적 후보와 형식화된 레코드 교육 스캐폴딩을 사용한다. Perceiver 스타일 decoder는 공급자-레코드 추론 경로가 존재하고 경험적 기준선이 후보 병목 현상을 노출한 후에만 추가해야 한다.

## 논문 핵심

Perceiver는 비대칭 cross-attention 작업을 사용하여 매우 큰 입력 배열을 훨씬 작은 학습된 잠재 배열로 읽은 다음 latent space에서 대부분의 처리를 수행한다. 이는 dense geometry 제공자 기능을 읽는 고정된 학습 쿼리의 아키텍처 아이디어를 지원한다. 명시적인 공간 기록이나 평생 보존 정책을 정의하지 않는다.

## 근거

이 논문는 수십만 개의 입력으로 확장된다고 보고한다. 2D 컨볼루션 없이 50,000픽셀에 직접 참여하면서 ResNet-50 및 ViT에 필적하는 ImageNet 성능을 얻고 포인트 클라우드, 오디오, 비디오 및 비디오 + 오디오에 대한 경쟁력 있는 결과를 보고한다.

이는 이 저장소에서 재현한 결과가 아닌 논문 결과이다.

## 판단 한계

- 잠재 슬롯은 시간이 지나도 안정적이거나 지속성 개체에 해당한다.
- 병목 현상이 정확한 거리, 포즈 또는 토폴로지를 유지한다.
- 고정된 잠재 개수는 반복 방문에 걸쳐 직렬화된 메모리의 경계를 정한다.
- 분류를 위해 훈련된 Perceiver가 SuperMemory-VQA로 이전된다.

## 문제 배경

표준 변환기는 self-attention를 모든 입력 요소에 직접 적용하므로 고차원 픽셀, 포인트 클라우드, 오디오 및 비디오 비용이 많이 들고 양식별 전처리가 권장된다.

## 관련 방법

- 고정 크기로 학습된 잠재 배열이 입력 배열에 교차 참여한다.
- 잠재 self-attention는 더 작은 잠재 크기에서 더 깊은 처리를 수행한다.
- Cross-attention는 반복될 수 있으며, 입력 간 전체 주의 없이 큰 입력에서 정보를 반복적으로 추출할 수 있다.
- 위치 및 양식 정보는 입력 표현을 통해 입력된다. 잠재 슬롯은 본질적으로 메트릭 또는 의미 엔터티가 아니다.

## 참고문헌

- [논문 목록](README.md)
- [ICML paper](https://proceedings.mlr.press/v139/jaegle21a.html)
- [arXiv:2103.03206](https://arxiv.org/abs/2103.03206)
- [공식 code](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver)
