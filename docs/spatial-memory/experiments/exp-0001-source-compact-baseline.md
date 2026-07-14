# EXP-0001: Source-compact baseline

| 항목 | 값 |
| --- | --- |
| Page ID | SM-EXP-0001 |
| Experiment ID | EXP-0001 |
| Confluence parent | SM-EXPERIMENTS |
| 상태 | Local sanity 완료 |
| 근거 수준 | Tiny synthetic fixture와 mock QA, benchmark 아님 |
| 최종 검토 | 2026-07-11 |
| 대체 대상 | 없음 |

## 핵심 결론

Source-compact pipeline, causal retrieval, actual-byte accounting, geometry proof,
metric serialization, spatial ablation 경로는 local sanity 기준으로 동작한다.
Learned spatial compiler와 공식 SuperMemory-VQA 성능은 검증되지 않았다.

## 다음 결정

Source-compact를 heuristic E0 baseline으로 유지한다. ADR-0001, ADR-0003,
ADR-0004의 구현 계약은 local-verified 상태로 둘 수 있지만 benchmark 채택 근거는
아니다. 다음 결정은 [EXP-0005](exp-0005-teacher-oracle-ceiling.md)가 동일 byte
budget의 object/location teacher-oracle utility를 측정한 뒤 내린다.

## 근거

Local sanity 결과이며 SuperMemory-VQA benchmark result가 아니다.

| Metric | Spatial 사용 | Spatial 미사용 | 미사용-사용 delta |
| --- | ---: | ---: | ---: |
| Ans-F1 | 100.0000 | 100.0000 | 0.0000 |
| QA-Acc | 66.6667 | 50.0000 | -16.6667 |
| QA-MRR | 83.3333 | 72.2222 | -11.1111 |

| Spatial artifact diagnostic | 값 |
| --- | ---: |
| Legacy diagnostic record | 216 |
| Legacy diagnostic JSONL byte | 96,456 |
| Persisted compact record | 15 |
| Persisted compact JSONL byte | 6,050 |
| Byte ratio | 15.94× 축소 |
| Categorical relation F1 | 1.0 |
| Metric relation F1 | 1.0 |
| Causal violations | 0 |

모든 local go condition을 통과했다. `legacy diagnostic`은 source-derived record의
JSONL 비교값이다. Dense G-CUT3R feature, binary device encoding, latency, energy,
실제 장기 revisit growth를 측정한 값이 아니다.

**C-002 local result:** Bounded source-compact memory는 legacy diagnostic의 216개
record·96,456 JSONL byte 대신 15개 record·6,050 byte를 유지해 15.94× 축소됐다
(`15.94x reduction`).
이는 heuristic tiny-fixture path만 검증하며 learned typed memory나 lifelong
convergence를 검증하지 않는다.

## 의사결정 gate

| Metric 또는 invariant | Go 조건 |
| --- | --- |
| Output contract | Parse 가능한 prediction 6개와 matching evidence pack 6개 |
| Causality | Causal violation 0 |
| Compression sanity | Compressed JSONL byte가 legacy diagnostic보다 작음 |
| Spatial diagnostic | Checked-in fixture expectation에서 categorical/metric relation F1 1.0 |
| Ablation sensitivity | Spatial 제거 시 QA-Acc와 QA-MRR 하락 |
| Scope guard | 모든 값을 local mock으로 표시하고 official benchmark로 사용 금지 |

## 비교안

| Variant | 변경 요소 | 고정 input |
| --- | --- | --- |
| Spatial 사용 | Spatial store 활성 | Fixture, mock QA, retrieval protocol, non-spatial store |
| Spatial 미사용 | Spatial store 비활성 | Fixture, mock QA, retrieval protocol, non-spatial store |
| Legacy serialization diagnostic | 선택되지 않은 raw source-derived spatial record의 크기 비교용 serialization | Fixture와 JSONL measurement method, QA retrieval variant 아님 |

## 가설

`source-compact-v1`이 causal source record를 compact spatial record로 만들고
actual-byte window budget을 지키면서 deterministic geometry QA에 필요한 증거를
남기는지 tiny fixture에서 검증한다. Spatial store를 제거하면 fixture의 spatial
question 성능이 낮아져야 한다.

## 실행 contract

| 항목 | 고정값 |
| --- | --- |
| Code state | 2026-07-11 local working tree, commit 미기록 |
| Fixture | `tests/fixtures/tiny_smvqa` |
| Input size | Source example 2개, question 6개 |
| Spatial config | `configs/spatial/source_compact_v1.json` |
| Encoder / projection / decoder | `structured-v1` / `identity-v1` / `delta-topk-v1` |
| Codec / selector | `compact-json-v1` / `linear-v1` |
| Window contract | 30초, complete-token window당 최대 16 token과 actual JSONL 4,096 byte |
| Geometry quantization | 0.25 m |
| QA backend | Deterministic local mock, Gemma/downloaded model 미사용 |
| Baseline stores | episodic, semantic, visual, spatial |
| Ablation stores | episodic, semantic, visual |
| Retrieval protocol | 두 variant 모두 `worldmm-smvqa` |

로컬 command:

```bash
uv run --offline worldmm-smvqa smoke \
  --fixture tests/fixtures/tiny_smvqa \
  --out /tmp/worldmm-exp-doc-smoke \
  --ablation-stores episodic,semantic,visual
```

## 추적성

| 유형 | Link | 관련성 |
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

## 실행 provenance

| 항목 | 값 |
| --- | --- |
| Run 일자 | 2026-07-11 |
| Run ID | Local ad-hoc sanity, canonical run ID 없음 |
| Code revision | Dirty local working tree, commit 미기록 |
| Process reference | 위 local command |
| Temporary output | `/tmp/worldmm-exp-doc-smoke` |
| Remote job ID | 없음, SSH/Slurm submission 미실행 |
| Company artifact path | 없음 |
| 로컬 복사 model/dataset | 없음, checked-in synthetic fixture만 사용 |
| Retained artifact | 없음, `/tmp` output은 ephemeral |
