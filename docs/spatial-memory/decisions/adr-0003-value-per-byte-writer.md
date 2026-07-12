# ADR-0003: Select persistent records by value per actual serialized byte

| Metadata | Value |
|---|---|
| Page ID | SM-ADR-0003 |
| ADR | ADR-0003 |
| Project claims | C-005 |
| Status | Accepted; partially implemented |
| Date | 2026-07-11 |
| Traceability | [Bounded storage and QA utility requirements](../traceability.md) |
| Architecture | [Budgeted memory writer](../architecture.md) |

## Context and problem

Feature quantization만으로는 중복 patch와 불필요한 candidate 생성을 막지 못한다.
또한 token count나 fixed record count는 record type마다 다른 실제 저장 비용을
반영하지 않는다. 미래 질문을 모두 알 수 없으므로 QA utility만 극대화하면
query-agnostic geometry core를 삭제할 위험도 있다.

## Decision

Persistent writer는 candidate score를 canonical JSONL의 실제 serialized byte
cost로 나눈 순서로 record를 선택하고 hard byte cap을 넘는 record를 쓰지 않는다.
`no_write`는 영구 artifact에 기록하지 않는다. 동률은 decoder 순서를 보존해
재현성을 유지한다.

최종 learned score는 QA deletion utility, geometry novelty, uncertainty reduction,
pose information, event surprise, redundancy를 결합해야 한다. 현재 repository는
actual-byte hard writer와 별도 selector-training scaffold를 갖지만, 이 utility가
typed student의 deployed write gate에 end-to-end 연결되지는 않았다.

## Evidence

- [LONG3R](../papers/long3r.md)는 fixed-capacity long-term memory selection의 관련
  근거다. 이 결정은 reconstruction attention 대신 deployment utility와 byte cost를
  명시한다.
- [Point3R](../papers/point3r.md)는 generic pointer memory의 저장 단위와 explicit
  indexing을 비교할 관련 근거다.
- `src/worldmm_smvqa/worldmm/typed_memory.py`는 canonical
  serialization 비용과 hard-budget writer를 구현한다.
- `src/worldmm_smvqa/worldmm/spatial_compression.py`는
  heuristic lane에서 causal window별 actual-byte admission을 구현한다.
- `src/worldmm_smvqa/spatial_selector_train.py`는
  explicit utility와 split manifest를 요구하는 별도 selector-training 경로다.
- [local readiness review](../reviews/2026-07-11-local-readiness.md)는 selector와 typed student의
  candidate space가 분리된 상태를 당시 P0 blocker로 기록했다. 현재
  [status](../status.md)와 roadmap은 production bridge 실행과 분리해 P2 research
  gap으로 분류한다.

## Alternatives considered

| Alternative | Why not selected |
|---|---|
| 모든 feature 생성 후 quantization | 불필요한 candidate 수와 시간 중복을 줄이지 못함 |
| Token 수 또는 record 수만 제한 | 실제 JSONL byte 비용 차이를 반영하지 못함 |
| Attention top-k | 미래 geometry QA 기여와 삭제 손실을 직접 나타내지 않음 |
| Surprise-only gate | 안정적이지만 중요한 wall, portal, free space를 삭제할 수 있음 |
| QA utility-only gate | 학습 질문 분포 밖의 geometry core를 잃을 수 있음 |
| Greedy score만 사용하고 byte cost 무시 | 큰 record가 budget을 지배하고 비교 가능한 rate constraint가 사라짐 |

## Implementation

- `serialized_byte_cost`는 canonical JSONL bytes를 직접 센다.
- `write_typed_memory_artifact`는 duplicate ID와 non-finite score를 거부하고,
  score-per-byte로 정렬한 뒤 budget 내 record만 atomic하게 쓴다.
- 작성 후 artifact를 다시 parse하고 file size, memory ID 순서, byte recount를
  검증한다.
- Heuristic source-compact lane은 causal window별 budget을 적용한다.
- Student training의 rate loss는 differentiable regularizer다. Hard cap은 writer가
  책임진다.
- 미구현: counterfactual deletion utility producer, generic geometry core reserve,
  learned typed candidate와 selector의 공통 candidate ID/feature contract, global 또는
  submap lifetime budget.

## Verification

- `tests/test_typed_memory_writer.py`는 duplicate,
  `no_write`, deterministic selection, serialized byte cap, artifact verification을
  검사한다.
- `tests/test_spatial_compression.py`는 causal
  window budget과 freshness/dedup behavior를 검사한다.
- `tests/test_spatial_selector_train.py`는
  explicit utility와 train/validation contract를 검사한다.
- 검증은 local fixture 수준이다. Official dataset의 QA-vs-bytes Pareto, bytes/hour,
  repeated-visit growth, learned utility 개선은 아직 측정하지 않았다.

## Consequences

- 실험 간 rate를 실제 artifact byte로 비교할 수 있다.
- Budget 위반은 training loss가 아니라 persistent write boundary에서 차단된다.
- Greedy value-per-byte는 일반 knapsack 최적해를 보장하지 않는다. 현재 필요한
  deterministic baseline에는 충분하며, 측정된 성능 차이가 있을 때만 더 복잡한
  solver를 검토한다.
- 현재 per-window budget은 lifetime storage bound가 아니다. 장기 current-state
  consolidation은 별도 결정과 실험이 필요하다.
- Utility estimator 오류가 중요한 geometry를 삭제할 수 있으므로 core reserve와
  abstention policy가 필요하다.

## Supersession

None.
