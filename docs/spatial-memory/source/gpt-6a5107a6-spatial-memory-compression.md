# GPT 출처: Spatial Memory 압축 방법

| 항목 | 값 |
|---|---|
| Page ID | SM-SOURCE-GPT-6A5107A6 |
| Source 유형 | 원문 요청과 도출된 연구 방향 |
| Conversation ID | `6a5107a6-2808-83e8-8e93-e63a1ca64ed4` |
| 원제 | Spatial Memory 압축 방법 |
| Import 일자 | 2026-07-11 |
| Confluence parent | SM-SOURCE |
| Canonical 해석 | [Spatial Memory 프로젝트](../README.md) |

## 핵심 해석

원문은 sparse 1 Hz sensing, 제한된 long-term storage, spatial-memory model 개발의
세 가지 제약 아래 explicit geometry-grounded QA를 요구한다. 도출된 방향은 generic
feature를 사후 quantization하는 것이 아니라 저장 전에 무엇을 쓸지 결정하는 것이다.

## 도출된 방향

```text
transient dense geometry
    -> typed candidates
    -> future-QA and geometry value per actual byte
    -> explicit persistent spatial database
```

- G-CUT3R-like geometry를 transient teacher/front-end로 사용한다.
- Object, structure, free space, landmark, change event, uncertainty, validity,
  provenance, bounded exceptional evidence를 persist한다.
- Derivable relation은 query time에 계산한다.
- Geometry answer에는 deterministic proof를 요구한다.
- Future question을 모르므로 query-agnostic core를 유지한다.

## 원문

> AI 글래스는 Spatial memory 를 적용하는데 explicit한 geometry grounded QA가
> 가능해야 한다. 하지만 다음 몇가지 제약을 가진다.
>
> 1. Sparse sensing이라 1hz 수준의 영상 촬영
> 2. Spatial memory를 장기간 저장하기에는 AI 디바이스 용량이 크지 않다.
> 3. AI 모델을 개발하여야 한다.(Spatial memory 를 만들기 위한 encoder나 encoder
> decoder 같은)
>
> 이를 해결하기 위한 explicit한 spatial 메모리를 구축하는데 spatial 메모리를
> compression 관점으로 만들어진 feature를 양자화 하는등의 압축이 아니라 만들어
> 질때 geometry grounded QA에 중요한 feature만 생성하는 압축방법을 적용하려 한다.
> 상세하게 가능성을 찾아보고 spatial 정보를 잘 표현하는 g cut3r같은 기술에
> 메모리를 압축할 수 있는 상세 조사 수행

## Provenance 범위

이 page는 원 요청과 도출된 thesis를 보존한다. Implementation specification이나
experiment result가 아니다. 채택한 선택은 [아키텍처 결정](../decisions/README.md),
측정 근거는 [실험](../experiments/README.md)을 사용한다.

[출처 목록으로 돌아가기](README.md)
