# EXP-0001: Source-Compact Baseline

| Metadata | Value |
| --- | --- |
| Page ID | SM-EXP-0001 |
| Experiment ID | EXP-0001 |
| Confluence parent | SM-EXPERIMENTS |
| Status | Local sanity completed |
| Evidence level | Tiny synthetic fixture and mock QA; not benchmark |
| Last reviewed | 2026-07-11 |
| Supersedes | None |

## Hypothesis

`source-compact-v1`이 causal source record를 compact spatial record로 만들고
actual-byte window budget을 지키면서 deterministic geometry QA에 필요한 증거를
남기는지 tiny fixture에서 검증한다. Spatial store를 제거하면 fixture의 spatial
question 성능이 낮아져야 한다.

## Linked claims, decisions, and papers

| Type | Link | Relevance |
| --- | --- | --- |
| Claim | [C-002: bounded long-term memory](../traceability.md) | frame 수가 아닌 선택된 explicit record에 저장량을 연결 |
| Claim | [C-003: verifiable geometry QA](../traceability.md) | 답변을 deterministic geometry proof에 연결 |
| Claim | [C-005: actual-byte accounting](../traceability.md) | 직렬화 byte 기준 writer와 compression sanity 검증 |
| Claim | [C-009: causal proof boundary](../traceability.md) | future evidence와 불완전 proof를 차단 |
| Decision | [ADR-0001: explicit typed memory](../decisions/adr-0001-explicit-typed-memory.md) | persistent geometry를 explicit record로 저장 |
| Decision | [ADR-0003: value per actual byte](../decisions/adr-0003-value-per-byte-writer.md) | token 수가 아닌 직렬화 byte를 제한 |
| Decision | [ADR-0004: deterministic geometry proof](../decisions/adr-0004-deterministic-geometry-proof.md) | language model 밖에서 거리와 관계를 계산 |
| Paper context | [LONG3R](../papers/long3r.md) | fixed-capacity long-term memory의 연구 근거; 본 실험은 LONG3R 재현이 아님 |
| Paper context | [Point3R](../papers/point3r.md) | explicit spatial indexing의 연구 근거; 본 실험은 Point3R 재현이 아님 |
| Benchmark contract | [SuperMemory-VQA](../papers/supermemory-vqa.md) | four-choice metric 계약; tiny fixture 수치는 공식 benchmark가 아님 |

## Fixed contract

| Item | Fixed value |
| --- | --- |
| Code state | 2026-07-11 local working tree; commit not captured |
| Fixture | `tests/fixtures/tiny_smvqa` |
| Input size | 2 source examples, 6 questions |
| Spatial config | `configs/spatial/source_compact_v1.json` |
| Encoder / projection / decoder | `structured-v1` / `identity-v1` / `delta-topk-v1` |
| Codec / selector | `compact-json-v1` / `linear-v1` |
| Window contract | 30 seconds, at most 16 tokens and 4096 actual JSONL bytes per complete-token window |
| Geometry quantization | 0.25 m |
| QA backend | Deterministic local mock; no Gemma or downloaded model |
| Baseline stores | episodic, semantic, visual, spatial |
| Ablation stores | episodic, semantic, visual |
| Retrieval protocol | `worldmm-smvqa` for both variants |

Local command:

```bash
uv run --offline worldmm-smvqa smoke \
  --fixture tests/fixtures/tiny_smvqa \
  --out /tmp/worldmm-exp-doc-smoke \
  --ablation-stores episodic,semantic,visual
```

## Compared variants

| Variant | Only changed factor | Inputs held constant |
| --- | --- | --- |
| With spatial | Spatial store enabled | fixture, mock QA, retrieval protocol, non-spatial stores |
| Without spatial | Spatial store disabled | fixture, mock QA, retrieval protocol, non-spatial stores |
| Legacy serialization diagnostic | Unselected raw source-derived spatial records serialized for size comparison | fixture and JSONL measurement method; not a QA retrieval variant |

## Metrics and go/no-go

| Metric or invariant | Go condition |
| --- | --- |
| Output contract | 6 parseable predictions and 6 matching evidence packs |
| Causality | 0 causal violations |
| Compression sanity | compressed JSONL bytes less than legacy diagnostic JSONL bytes |
| Spatial diagnostics | categorical and metric relation F1 equal to 1.0 on the checked-in fixture expectations |
| Ablation sensitivity | removing spatial lowers QA-Acc and QA-MRR |
| Scope guard | all values labelled local mock, never official benchmark |

## Results

Local sanity only; not a SuperMemory-VQA benchmark result.

| Metric | With spatial | Without spatial | Without-minus-with delta |
| --- | ---: | ---: | ---: |
| Ans-F1 | 100.0000 | 100.0000 | 0.0000 |
| QA-Acc | 66.6667 | 50.0000 | -16.6667 |
| QA-MRR | 83.3333 | 72.2222 | -11.1111 |

| Spatial artifact diagnostic | Value |
| --- | ---: |
| Legacy diagnostic records | 216 |
| Legacy diagnostic JSONL bytes | 96,456 |
| Persisted compact records | 15 |
| Persisted compact JSONL bytes | 6,050 |
| Byte ratio | 15.94x smaller |
| Categorical relation F1 | 1.0 |
| Metric relation F1 | 1.0 |
| Causal violations | 0 |

모든 local go condition을 통과했다. `legacy diagnostic`은 source-derived record의
JSONL 비교값이다. Dense G-CUT3R feature, binary device encoding, latency, energy,
실제 장기 revisit growth를 측정한 값이 아니다.

**C-002 local result:** bounded source-compact memory retained 15 records and
6,050 JSONL bytes versus 216 records and 96,456 bytes in the legacy diagnostic,
a 15.94x reduction. This verifies only the heuristic tiny-fixture path; it does
not verify learned typed memory or lifelong convergence.

## Run provenance

| Item | Value |
| --- | --- |
| Run date | 2026-07-11 |
| Run ID | Local ad-hoc sanity; no canonical run ID |
| Code revision | Dirty local working tree; commit not captured |
| Process reference | Local command above |
| Temporary output | `/tmp/worldmm-exp-doc-smoke` |
| Remote job ID | None; no SSH or Slurm submission |
| Company artifact path | None |
| Models or datasets copied locally | None; checked-in synthetic fixture only |
| Retained artifacts | None; `/tmp` output is ephemeral |

## Conclusion

Source-compact pipeline, causal retrieval, actual-byte accounting, geometry proof, metric
serialization, spatial ablation 경로는 local sanity 기준 동작한다. Learned spatial
compiler 또는 공식 SuperMemory-VQA 성능은 검증되지 않았다.

## Decision impact

Source-compact를 heuristic E0 baseline으로 유지한다. ADR-0001, ADR-0003,
ADR-0004의 구현 계약은 local-verified 상태로 둘 수 있지만 benchmark 채택 근거는
아니다. 다음 결정은 [EXP-0002](exp-0002-typed-memory-bridge.md)가 checkpoint를
typed artifact와 QA evidence에 실제로 연결한 뒤 내린다.
