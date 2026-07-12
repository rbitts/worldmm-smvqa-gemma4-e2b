# Stop Looking for Important Tokens in Multimodal Language Models: Duplication Matters More

| Field | Value |
|---|---|
| Page ID | SM-PAPER-DART |
| Status | Reviewed |
| Method name | DART: Duplication-Aware Reduction of Tokens |
| Authors | Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, Linfeng Zhang |
| Publication | EMNLP 2025 main, pages 9961-9980 |
| Version checked | arXiv:2502.11494v2, 2025-06-08 |
| Primary source | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.505/) · [arXiv](https://arxiv.org/abs/2502.11494) |
| Official code | [ZichenWen1/DART](https://github.com/ZichenWen1/DART) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-011 |

## 30-Second Summary

DART는 다중 모드 언어 모델에 대해 추정된 토큰 중요도보다 중복성이 더 나은 정리 신호라고 주장한다. 작은 피벗 세트와 유사성이 낮은 토큰을 유지하며 추가 교육이 필요하지 않는다. 이는 비용이 많이 드는 학습된 메모리 기록기 전에 중복 후보를 제거하는 것을 지원하지만 어떤 공간적 사실이 장기 저장에서 살아남아야 하는지 식별하지 못한다.

## Problem Addressed

긴 시각적 토큰 시퀀스는 다중 모드 모델 추론 비용을 지배한다. 기존 중요도 기반 가지치기는 무작위 가지치기 성능을 저하할 수 있으며 효율적인 주의 커널과 제대로 상호 작용할 수 없다.

## Relevant Method

- 피벗 토큰의 작은 하위 집합을 선택한다.
- 피벗을 기준으로 다른 토큰의 중복을 측정한다.
- 중복이 적은 토큰을 유지하여 고유한 정보를 보존한다.
- 소스 모델을 재교육하지 않고 추론 시 축소를 적용한다.

## Paper-Reported Evidence

저자는 유사한 성능을 유지하면서 시각적 토큰의 88.9%를 정리한다고 보고한다. 효율적인 주의 연산자와의 호환성과 함께 총 시간 속도가 1.99배 향상되고 사전 채우기 속도가 2.99배 향상되었다고 보고한다.

이는 이 저장소에서 재현한 결과가 아닌 논문 결과이다.

## What This Supports Here

- 바이트당 값을 선택하기 전에 동일하거나 거의 동일한 좌표, 인스턴스, 관계 및 관찰 후보를 중복 제거한다.
- 높은 관심을 가치의 유일한 척도로 여기지 말고 새로운 후보를 보존한다.
- 중복 제거가 동일한 serialized-byte budget에서 다운스트림 작성자의 파레토 곡선을 향상하는지 여부를 측정한다.

## What It Does Not Prove

- 낮은 기능 중복은 높은 미래 QA 가치를 의미한다.
- 일반적인 VLM 유사성은 metric geometry 레코드에 안전하다.
- 추론 토큰 절약은 영구 메모리 절약으로 이어집니다.
- DART는 임시 상태 변경 또는 object identity를 보존한다.

## Project Reproduction Status

재현되지 않았다. 현재 작성자는 결정론적 동일 키 보존 및 실제 바이트 승인을 수행하지만 DART가 아니며 피벗 토큰 단계가 없다. 중복 우선 감소는 보고된 프로젝트 결과가 아닌 후보 절제로 남아 있다.

## References

- [Paper index](README.md)
- [EMNLP paper](https://aclanthology.org/2025.emnlp-main.505/)
- [arXiv:2502.11494](https://arxiv.org/abs/2502.11494)
- [Official code](https://github.com/ZichenWen1/DART)
