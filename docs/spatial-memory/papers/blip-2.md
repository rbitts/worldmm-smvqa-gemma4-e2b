# BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

| Field | Value |
|---|---|
| Page ID | SM-PAPER-BLIP-2 |
| Status | Reviewed |
| Authors | Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi |
| Publication | ICML 2023, PMLR 202 |
| Version checked | arXiv:2301.12597v3, 2023-06-15 |
| Primary source | [PMLR](https://proceedings.mlr.press/v202/li23q.html) · [arXiv](https://arxiv.org/abs/2301.12597) |
| Official code | [LAVIS BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-010 |

## 30-Second Summary

BLIP-2는 경량 Querying Transformer 또는 Q-Former를 통해 frozen image encoder를 frozen language model에 연결한다. 학습된 쿼리는 cross-attention을 통해 작은 visual representation을 추출한다. 이는 대형 geometry teacher와 작업별 decoder 사이의 컴팩트 adapter를 지원하지만 BLIP-2 자체는 metric geometry 또는 persistent memory를 유지하지 않는다.

## Problem Addressed

End-to-end 비전 언어 pre-training은 시각적 encoder와 language model이 모두 성장함에 따라 비용이 많이 듭니다. 고정된 사전 훈련된 구성요소에는 훈련 가능한 브리지가 필요한 modality gap도 있다.

## Relevant Method

- 경량 Q-Former에는 learned query embeddings가 포함되어 있다.
- 고정된 이미지-encoder 기능에 대한 쿼리가 교차 참여된다.
- 1단계에서는 frozen image encoder에서 학습하는 image-text representation를 부트스트랩한다. frozen language model를 통해 2단계 부트스트랩 생성을 수행한다.
- 브리지 및 관련 projections만 중앙 설정에서 작업별 교육이 필요하다.

## Paper-Reported Evidence

BLIP-2는 이전 시스템보다 훨씬 적은 수의 trainable parameters를 사용하여 여러 비전 언어 작업에 걸쳐 state-of-the-art 결과를 보고한다. 이 논문에서는 zero-shot VQAv2에서 Flamingo-80B보다 8.7% 개선되었으며 trainable parameters를 54배 더 적게 사용했다고 보고한다.

이는 이 저장소에서 재현한 결과가 아닌 논문 결과이다.

## What This Supports Here

- 작은 query adapter 및 입력된 출력 헤드를 훈련하는 동안 미리 훈련된 큰 geometry provider를 고정된 상태로 유지한다.
- 제한된 학습 쿼리를 사용하여 밀집된 공급자 기능에서 작업 관련 정보를 추출한다.
- 표현 정렬을 위한 단계 훈련이 downstream QA 유틸리티보다 우선한다.

## What It Does Not Prove

- Q-Former는 인코딩 좌표 프레임 uncertainty 또는 provenance를 출력한다.
- 이러한 비전-언어 정렬은 낮은 수준의 측정 정확도를 유지한다.
- 해당 쿼리 토큰은 며칠 동안 안정적인 개체 ID를 유지한다.
- 아키텍처가 실제 serialized-byte budget을 충족한다.

## Project Reproduction Status

재현되지 않았다. 저장소에 Q-Former가 없다. 현재 코드는 provider cache contract 및 유형 candidate head를 정의하지만 G-CUT3R 기능과 영구 레코드 간의 learned query 브리지는 정의하지 않는다.

## References

- [Paper index](README.md)
- [ICML paper](https://proceedings.mlr.press/v202/li23q.html)
- [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)
- [Official code](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
