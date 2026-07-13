# Spatial Token Compression 연구 로드맵

| 항목 | 값 |
|---|---|
| Page ID | SM-LEGACY-RESEARCH-ROADMAP |
| 문서 역할 | 레거시 안내 |
| Canonical 로드맵 | [Spatial Memory 로드맵](spatial-memory/roadmap.md) |
| Evidence catalog | [논문 근거 목록](spatial-memory/papers/README.md) |

## 실행 방향

Learned checkpoint-to-QA path를 검증하기 전에는 literature plan을 확장하지 않는다.
External paper는 candidate design을 정당화하지만 project result를 대체하지 않는다.

## 우선순위

| 우선순위 | 결정 |
|---|---|
| P0 | G-CUT3R-compatible teacher와 typed inference bridge 검증 |
| P1 | Learned E1, matched E2/E3, actual-byte curve 확립 |
| P2 | QA deletion utility를 deployed write gate에 연결 |
| P3 | Long-term identity, ray-aware association, relocalization 검증 |
| P4 | Bounded evidence reservoir로 unknown-question coverage 보호 |
| P5 | On-device 가능성 distillation·측정 |

## 연구 portfolio 결정

- **즉시 사용:** Explicit scene graph, transient/persistent state 분리,
  deterministic spatial operation, causal long-horizon evaluation.
- **후속 비교:** Streaming-state retention, geometry-aware pruning, duplicate
  removal, fixed-slot bottleneck을 equal byte에서 비교.
- **보류:** Explicit record가 measured bottleneck이 되기 전 VQ/FSQ codec과
  model quantization.

논문별 evidence, limit, reproduction status는
[canonical 논문 목록](spatial-memory/papers/README.md)에만 둔다. 새 roadmap
decision은 [canonical 로드맵](spatial-memory/roadmap.md)에 기록한다.
