# EXP-0004: G-CUT3R Provider Comparison

| Metadata | Value |
| --- | --- |
| Page ID | SM-EXP-0004 |
| Experiment ID | EXP-0004 |
| Confluence parent | SM-EXPERIMENTS |
| Status | Planned |
| Evidence level | External-provider and cache contract checks only |
| Last reviewed | 2026-07-11 |
| Depends on | Approved company GPU run and external G-CUT3R implementation |

## Hypothesis

같은 causal 1 Hz frame manifest에서 pose/depth guidance를 사용하는 external G-CUT3R
teacher가 fallback teacher보다 low-overlap geometry를 개선하고, causal typed
supervision으로 materialize될 수 있다.

## Linked claims, decisions, and papers

| Type | Link | Relevance |
| --- | --- | --- |
| Claim | [C-001: sparse low-overlap geometry](../traceability.md) | 1 Hz RGB 사이의 큰 viewpoint 변화 보완 |
| Claim | [C-002: bounded long-term memory](../traceability.md) | teacher output 전체가 아닌 typed sufficient records만 보존 |
| Claim | [C-006: pose and map separation](../traceability.md) | pose-guided transient teacher와 persistent record 경계 검증 |
| Decision | [ADR-0002: G-CUT3R as teacher](../decisions/adr-0002-gcut3r-as-teacher.md) | 대형 geometry model을 persistent memory가 아닌 offline teacher로 사용 |
| Decision | [ADR-0001: explicit typed memory](../decisions/adr-0001-explicit-typed-memory.md) | teacher geometry를 student record supervision으로 변환 |
| Paper | [G-CUT3R](../papers/g-cut3r.md) | pose/depth-guided sparse-view geometry 근거 |
| Paper | [CUT3R](../papers/cut3r.md) | recurrent geometry teacher의 fallback context |
| Benchmark | [SuperMemory-VQA](../papers/supermemory-vqa.md) | downstream episodic QA target; G-CUT3R 논문이 이 benchmark 개선을 입증한 것은 아님 |

## Fixed contract

| Item | Fixed value |
| --- | --- |
| Execution location | Company GPU resources only after explicit approval |
| Repository behavior | G-CUT3R code, weights, model download을 번들하거나 자동 수행하지 않음 |
| Extractor ownership | approved trusted wrapper owns G-CUT3R code/checkpoint loading and provenance |
| Distributed output | every rank emits exactly one non-empty rank shard; no extra JSONL; merged requests contain no cross-rank duplicate |
| Frame source | One run-scoped 1 Hz sensor-frame manifest shared by all variants |
| Cache coverage | Request `(video_id, frame_ref, timestamp)` set must exactly equal selected sensor observations; no missing or extra request |
| Causality | provider request cutoff and returned observation times must not exceed question/source cutoff |
| Pose provenance | production cache/shards allow `imu`, `vio`, or `slam`; `ground_truth` is rejected |
| Provider provenance | provider ID, code revision, checkpoint digest, config digest, guidance modalities required |
| Cache provenance | request digest, response digest, manifest digest, source/split digest required |
| Teacher variants | `gcut3r_external` and a declared fallback/cache variant under identical frames |
| Dataset, split, provider path, checkpoint, run ID | TBD before execution |
| Raw RGB / IMU / VIO / depth availability | Must be preflighted and recorded; currently unknown |

## Compared variants

| Variant | Only changed factor | Inputs held constant |
| --- | --- | --- |
| A: Fallback teacher | Declared CUT3R-compatible cached fallback without G-CUT3R guidance | split, 1 Hz frames, target codec, materializer |
| B: G-CUT3R external | External G-CUT3R provider with recorded available guidance | split, 1 Hz frames, target codec, materializer |

Pose-only, depth-only, and pose-plus-depth variants are added only if the external provider
supports them without changing frames, checkpoint family, or evaluation data.

## Metrics and go/no-go

| Metric or invariant | Go condition |
| --- | --- |
| Provider/cache validation | 100% of accepted rows match request, manifest, cutoff, and provenance digests; rank shards are one-per-rank and merged request multiset exactly matches selected frame/timestamp inventory |
| Causal violations | 0 |
| Geometry | ATE/RPE and type-specific geometry errors reported only where ground truth exists |
| Materialization | train/validation groups remain disjoint; no missing required targets |
| Typed supervision | every accepted teacher record produces valid, causal record-derived targets |
| Adoption | G-CUT3R improves a predeclared geometry target without violating causality or materialization contracts |

No geometry metric is fabricated when prepared SuperMemory-VQA data lacks the required
ground truth. In that case provider execution, causal cache, and materialization can be
validated, but the teacher-quality comparison remains inconclusive.

## Results

Not run.

Current local evidence covers JSONL request/response encoding, provider/cache provenance,
causal cutoff validation, path resolution, rank-specific cache planning, and teacher-row
materialization. No G-CUT3R model was installed, downloaded, invoked, or evaluated. No
teacher-quality or SuperMemory-VQA result exists.

## Run provenance

| Item | Value |
| --- | --- |
| Run ID | Not assigned |
| Code revision | Not pinned |
| External provider ID and revision | None |
| G-CUT3R checkpoint digest | None |
| Sensor-frame manifest digest | None |
| Slurm job ID or process reference | None |
| Company artifact path | None |
| Metrics artifact | None |
| Copied locally | None |

## Conclusion

Pending. ADR-0002는 연구 가설과 interface 결정이며 아직 empirical reproduction
근거가 아니다.

## Decision impact

Go이면 검증된 provider/checkpoint 조합을 ADR-0002의 채택 근거로 연결하고
EXP-0002 training input으로 고정한다. No-go이면 provider를 교체하거나 fallback을
유지한다. G-CUT3R를 persistent memory로 직접 저장하는 별도 경로는 추가하지 않는다.
