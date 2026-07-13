# Finite Scalar Quantization: VQ-VAE Made Simple

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-FSQ |
| 상태 | 검토 완료; project 미재현 |
| 출판 | ICLR 2024 |
| 1차 출처 | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e2dd53601de57c773343a7cdf09fae1c-Abstract-Conference.html) |
| 공식 code | [Google Research FSQ](https://github.com/google-research/google-research/tree/master/fsq) |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [논문 목록](README.md) · [프로젝트 홈](../README.md) · [문제 정의](../problem.md) · [추적성](../traceability.md) |
| 프로젝트 claim | [추적성](../traceability.md): C-012 |

## 핵심 결론

**프로젝트 추론.** FSQ는 typed record에 학습된 코드가 여전히 필요한 경우 압축 연관 설명자 또는 기타 잠재 필드에 대한 간단한 스칼라 이산 기준선 유지를 지원한다. 또한 교육 및 유지 관리 비용을 수용하기 전에 학습된 VQ codebook가 필요한지 테스트하는 것도 지원한다.

여기에서 사용하려면 실제 직렬화된 바이트와 다운스트림 기하학-QA 품질을 측정해야 한다. Codebook 크기만으로는 저장 결과가 아니다.

## 근거 상태

**프로젝트 결과.** 재현되지 않는다. FSQ 모듈, 체크포인트 또는 벤치마크 아티팩트는 현재 프로젝트의 일부가 아니다. FSQ는 명시적 유형 레코드 기준이 측정된 후 압축 학습 설명자가 필요한 경우에만 추가할 선택적 압축 기준으로 유지된다.

## 논문 핵심

Finite Scalar Quantization(FSQ)는 학습된 최근접이웃 vector quantization을 개별 차원이 고정된 유한 수준으로 반올림된 저차원 벡터로 대체한다. 해당 수준의 데카르트 곱은 암시적 codebook이다. 이 논문은 학습된 codebook 붕괴 및 VQ-특정 보조 기계를 피하면서 이미지 생성 및 조밀한 예측에서 VQ와 경쟁할 수 있는 성능을 보고한다.

이 프로젝트의 경우 FSQ는 유용한 컴팩트 잠재 기준선이다. 이미 생성된 기능을 더 작게 만들기 때문에 평생 spatial memory에 대한 주요 답변은 아니다. 어떤 기하학 기반 사실이 존재해야 하는지는 결정하지 않는다.

## 근거

**논문 보고 결과.** MaskGIT를 사용한 256x256 ImageNet 생성에서 이 논문은 FSQ 및 VQ에 대해 유사한 FID, 정밀도, 재현율 및 정성적 샘플을 보고한다. UViM에서 FSQ는 NYU 깊이, 색상화 및 팬옵틱 분할에서 VQ와 경쟁하는 것으로 보고됐다. codebook 분할이 없는 NYU 깊이 비교에서 논문은 99% FSQ 코드 사용을 보고한다. 해당 VQ 변형은 0.78%를 사용하고 RMSE가 더 나쁩니다. [conference paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/e2dd53601de57c773343a7cdf09fae1c-Paper-Conference.pdf)의 섹션 5.2, 5.3 및 표 2를 참조한다.

이는 저자가 보고한 결과이다. 이 저장소는 이를 재현하지 않았다.

## 판단 한계

- 객체, 평면, 포털, 랜드마크 또는 변경 이벤트는 선택되지 않는다.
- 엔터티 ID, 좌표 프레임, uncertainty, provenance, 시간적 타당성 또는 인과적 증거를 보존하지 않는다.
- SuperMemory-VQA에는 스칼라 양자화된 일반 기능이 충분하다는 것을 보여주지 않는다.
- 1Hz, 반복 방문 또는 AI-glass 장치에서는 성능을 설정하지 않는다.
- 프로젝트의 실제 바이트 기록기나 결정론적 기하학 실행기를 대체하지 않는다.

## 문제 배경

VQ-VAE 이산화는 일반적으로 codebook를 학습하고 모든 encoder 벡터를 가장 가까운 항목에 매핑한다. 크게 학습된 codebooks는 제대로 사용되지 않을 수 있으며 약정 손실, codebook 업데이트, 재시드, 분할 또는 엔트로피 정규화가 필요한 경우가 많다. FSQ는 이러한 최적화 및 codebook 활용 문제를 대상으로 한다.

## 관련 방법

**논문 주장.** encoder는 먼저 각 잠재 벡터를 일반적으로 10개 미만의 작은 차원으로 투영한다. 차원 `i`는 직선 그라데이션을 사용하여 `L_i` 고정 레벨 중 하나로 제한되고 반올림된다. 암시적 codebook에는 `product(L_i)` 조합이 포함되어 있지만 embedding 테이블은 학습되지 않는다. 혼합 기수 매핑은 각 조합을 개별 인덱스로 변환한다.

VQ 토큰과 함께 사용되는 동일한 다운스트림 모델은 FSQ 지수를 사용할 수 있다. 이 논문에서는 새로운 공간 메모리 표현을 제안하기보다는 MaskGIT 및 UViM에서 이러한 대체를 평가한다.

## 참고문헌

- Mentzer, Minnen, Agustsson 및 Tschannen. [Finite Scalar Quantization: VQ-VAE Made Simple](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e2dd53601de57c773343a7cdf09fae1c-Abstract-Conference.html). ICLR 2024.
- 저자의 [official FSQ JAX implementation](https://github.com/google-research/google-research/tree/master/fsq).
- [Paper appendix and reference implementation](https://proceedings.iclr.cc/paper_files/paper/2024/file/e2dd53601de57c773343a7cdf09fae1c-Supplementary-Conference.pdf).
