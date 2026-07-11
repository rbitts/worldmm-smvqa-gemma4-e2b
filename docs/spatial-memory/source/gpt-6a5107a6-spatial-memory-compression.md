# GPT Source: Spatial Memory Compression Method

| Field | Value |
|---|---|
| Page ID | SM-SOURCE-GPT-6A5107A6 |
| Source type | Verbatim user request plus conversation-derived research summary |
| Conversation ID | `6a5107a6-2808-83e8-8e93-e63a1ca64ed4` |
| Original title | Spatial Memory 압축 방법 |
| Imported | 2026-07-11 |
| Parent | [Source and provenance](README.md) |
| Canonical interpretation | [Spatial Memory project](../README.md) |

## Verbatim User Request

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

## Source Problem Summary

AI glasses need explicit geometry-grounded QA but have three constraints:

1. Sparse sensing at approximately 1 Hz RGB capture.
2. Limited device capacity for long-term spatial memory.
3. A spatial-memory encoder or encoder-decoder must be developed.

The proposed research direction is not post-hoc feature quantization. It is to
compress at memory creation time by producing only features and records that
matter for future geometry-grounded QA, with particular attention to
G-CUT3R-like geometry representations.

## Imported Research Thesis

The source conversation concluded that the target should be a learned spatial
compiler:

```text
transient dense world reasoning
    -> explicit typed candidates
    -> future-QA and geometry value per actual byte
    -> persistent spatial database
```

The proposed persistent core includes:

- local-frame place and submap records;
- structural planes, portals, and coarse free space;
- object identity, centroid, extent, support, containment, and validity;
- ray-aware relocalization landmarks;
- meaningful change events;
- uncertainty and observation provenance;
- a small surprise and uncertainty evidence reservoir.

G-CUT3R is treated as a transient teacher or front-end, not as the long-term
memory itself.

## Imported Design Principles

1. Do not persist CUT3R recurrent states or dense point maps as lifelong memory.
2. Generate object, plane, portal, free-space, landmark, and event records
   directly.
3. Learn write decisions from future QA loss, geometry information gain,
   uncertainty reduction, event surprise, redundancy, and serialized byte cost.
4. Compute spatial relations at query time when coordinates and base facts are
   sufficient.
5. Use pose guidance and ray-aware association to compensate for sparse RGB.
6. Preserve a query-agnostic geometry core because future questions are unknown.
7. Require answer proofs with entity IDs, coordinate frame, uncertainty,
   provenance, and evidence references.

## Provenance Boundary

This page preserves the originating task and its research thesis. It is not the
current implementation specification and must not be cited as experimental
evidence. Current decisions live in [Architecture Decisions](../decisions/README.md),
and measured project results live in [Experiments](../experiments/README.md).

[Back to source index](README.md)
