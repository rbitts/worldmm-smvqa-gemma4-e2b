# EXP-0004: G-CUT3R provider 비교

| 항목 | 값 |
| --- | --- |
| Page ID | SM-EXP-0004 |
| Experiment ID | EXP-0004 |
| Confluence parent | SM-EXPERIMENTS |
| 상태 | 계획 |
| 근거 수준 | External-provider/cache contract check만 완료 |
| 최종 검토 | 2026-07-14 |
| 선행 조건 | 승인된 company GPU run과 external G-CUT3R implementation |

## 핵심 결론

대기. ADR-0002는 연구 가설과 interface 결정이며 아직 empirical reproduction
근거가 아니다.

## 다음 결정

Go이면 검증된 provider/checkpoint 조합을
[EXP-0005](exp-0005-teacher-oracle-ceiling.md)의 offline oracle로 고정한다. No-go이면
provider를 교체하거나 fallback을 유지한다. G-CUT3R를 device runtime 또는
persistent memory로 직접 저장하는 경로는 추가하지 않는다.

## 근거

미실행.

현재 local evidence는 JSONL request/response encoding, provider/cache provenance,
causal cutoff validation, path resolution, rank-specific cache planning,
teacher-row materialization만 포함한다. G-CUT3R model을 설치, download, invoke,
evaluate하지 않았으며 teacher-quality 또는 SuperMemory-VQA result가 없다.

## 의사결정 gate

| Metric 또는 invariant | Go 조건 |
| --- | --- |
| Provider/cache validation | 채택된 row 100%가 request, manifest, cutoff, provenance digest와 일치하고 rank당 shard 1개이며 merged request multiset이 selected frame/timestamp inventory와 정확히 일치 |
| Causal violations | 0 |
| Geometry | Ground truth가 있는 경우에만 ATE/RPE와 type-specific geometry error 보고 |
| Materialization | Train/validation group이 disjoint하고 required target 누락 없음 |
| Typed supervision | 모든 accepted teacher record가 valid causal record-derived target 생성 |
| Adoption | G-CUT3R가 causality/materialization contract 위반 없이 predeclared geometry target 개선 |

Prepared SuperMemory-VQA data에 필요한 ground truth가 없으면 geometry metric을
만들어내지 않는다. 이 경우 provider execution, causal cache, materialization은
검증할 수 있지만 teacher-quality comparison은 결론을 내리지 않는다.

## 비교안

| Variant | 변경 요소 | 고정 input |
| --- | --- | --- |
| A: Fallback teacher | G-CUT3R guidance가 없는 declared CUT3R-compatible cached fallback | Split, 1 Hz frame, target codec, materializer |
| B: G-CUT3R external | Available guidance가 기록된 external G-CUT3R provider | Split, 1 Hz frame, target codec, materializer |

External provider가 frame, checkpoint family, evaluation data 변경 없이 지원하는
경우에만 pose-only, depth-only, pose-plus-depth variant를 추가한다.

## 가설

같은 causal 1 Hz frame manifest에서 pose/depth guidance를 사용하는 external G-CUT3R
teacher가 fallback teacher보다 low-overlap geometry를 개선하고, causal typed
supervision으로 materialize될 수 있다.

## 실행 contract

| 항목 | 고정값 |
| --- | --- |
| Execution location | 명시적 승인 후 company GPU resource에서만 실행 |
| Repository behavior | G-CUT3R code, weights, model download을 번들하거나 자동 수행하지 않음 |
| Extractor ownership | Approved trusted wrapper가 G-CUT3R code/checkpoint loading과 provenance 소유 |
| Distributed output | 모든 rank가 non-empty rank shard 하나만 생성하고 extra JSONL/cross-rank duplicate가 없으며 merged request가 selected inventory와 일치 |
| Frame source | 모든 variant가 하나의 run-scoped 1 Hz sensor-frame manifest 공유 |
| Cache coverage | Request `(video_id, frame_ref, timestamp)` set이 selected sensor observation과 정확히 일치하고 missing/extra request 없음 |
| Causality | Provider request cutoff와 returned observation time이 question/source cutoff를 초과하지 않음 |
| Pose provenance | Production cache/shard는 `imu`, `vio`, `slam`만 허용하고 `ground_truth` 거부 |
| Provider provenance | Provider ID, code revision, checkpoint digest, config digest, guidance modality 필수 |
| Cache provenance | Request/response/manifest/source/split digest 필수 |
| Teacher variant | 동일 frame 아래 `gcut3r_external`과 declared fallback/cache variant |
| Dataset, split, provider path, checkpoint, run ID | 실행 전 TBD 해소 |
| Raw RGB / camera intrinsics / IMU / VIO / depth availability | Preflight·실제 coverage 기록 필수, 현재 미확인; intrinsics는 depth와 독립 |

## 추적성

| 유형 | Link | 관련성 |
| --- | --- | --- |
| Claim | [C-001: sparse low-overlap geometry](../traceability.md) | 1 Hz RGB 사이의 큰 viewpoint 변화 보완 |
| Claim | [C-002: bounded long-term memory](../traceability.md) | teacher output 전체가 아닌 typed sufficient records만 보존 |
| Claim | [C-006: pose and map separation](../traceability.md) | pose-guided transient teacher와 persistent record 경계 검증 |
| Decision | [ADR-0002: G-CUT3R as teacher](../decisions/adr-0002-gcut3r-as-teacher.md) | 대형 geometry model을 persistent memory가 아닌 offline teacher로 사용 |
| Decision | [ADR-0005: hybrid device compiler](../decisions/adr-0005-hybrid-on-device-compiler.md) | Provider 결과를 device architecture claim과 분리 |
| Decision | [ADR-0001: explicit typed memory](../decisions/adr-0001-explicit-typed-memory.md) | teacher geometry를 student record supervision으로 변환 |
| Paper | [G-CUT3R](../papers/g-cut3r.md) | pose/depth-guided sparse-view geometry 근거 |
| Paper | [CUT3R](../papers/cut3r.md) | recurrent geometry teacher의 fallback context |
| Benchmark | [SuperMemory-VQA](../papers/supermemory-vqa.md) | downstream episodic QA target; G-CUT3R 논문이 이 benchmark 개선을 입증한 것은 아님 |

## 실행 provenance

| 항목 | 값 |
| --- | --- |
| Run ID | 미할당 |
| Code revision | 미고정 |
| External provider ID/revision | 없음 |
| G-CUT3R checkpoint digest | 없음 |
| Sensor-frame manifest digest | 없음 |
| Slurm job ID 또는 process reference | 없음 |
| Company artifact path | 없음 |
| Metrics artifact | 없음 |
| 로컬 복사 | 없음 |
