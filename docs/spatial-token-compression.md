# Spatial Token Compression

## 1. 문서 목적

이 문서는 1 Hz sparse image sensing을 사용하는 AI Glass 환경에서
geometry-grounded QA를 지원하기 위한 explicit spatial memory의 목표, 현재
구조, 데이터 계약, 학습 경로, 평가 기준을 정의한다.

핵심 목표는 모든 dense geometry를 보존하는 것이 아니다. 미래 질문에 필요한
최소한의 공간 정보를 causal하게 저장하고, 기존 WorldMM의 episodic,
semantic, visual memory와 함께 하나의 QA 경로에서 사용하는 것이다.

관련 논문, 후속 접근법, 실험 우선순위는
[spatial-token-research-roadmap.md](spatial-token-research-roadmap.md)에
분리했다.

## 2. 목표와 비목표

### 2.1 목표

1. **Explicit geometry 유지**
   - object 위치, zone, object-object relation, wearer trajectory를 QA에서
     다시 해석할 수 있는 형태로 저장한다.
2. **생성 단계 압축**
   - 반복 관측, 역관계 중복, 불필요한 frame reference를 메모리에 쓰기 전에
     제거한다.
3. **고정 예산**
   - 시간 구간별 spatial token 수와 저장 byte를 제한할 수 있어야 한다.
4. **Geometry-grounded QA 보존**
   - 압축률만 높이는 것이 아니라 spatial retrieval recall과 최종 QA 성능을
     보존해야 한다.
5. **구성요소 교체**
   - CUT3R 같은 geometry provider, projection head, token decoder, selector,
     codec을 독립 실험할 수 있어야 한다.
6. **WorldMM 통합**
   - spatial memory가 별도 QA 시스템으로 끝나지 않고 E/S/V/S retrieval과
     Gemma QA decoder에 연결되어야 한다.
7. **Causal memory**
   - 질문 시각 이후의 geometry가 retrieval 또는 QA에 노출되면 안 된다.

### 2.2 비목표

- 전체 장면의 photorealistic reconstruction 보존
- 렌더링 품질을 위한 dense point cloud 또는 Gaussian 전체 저장
- spatial 전용 LLM 답변기 구축
- 현재 단계에서 Gemma와 geometry encoder를 end-to-end 공동 학습
- 측정 없이 binary codec이나 복잡한 ANN infrastructure 선도입

## 3. 용어

### Spatial geometry feature

Geometry provider가 이미지, pose, gaze, object detection으로부터 만든
중간 표현이다. 현재 구현에서는 `zone`, `anchor`, `relation`,
`extra_features`로 구성된다.

### Spatial token

Spatial memory에 영속 저장되는 application-level record다. Gemma tokenizer의
vocabulary token ID와 다르다. 현재 token은 zone, object anchor,
object relation 중 하나를 compact payload로 표현한다.

### Spatial token decoder

Projected geometry로부터 저장 후보를 생성하는 memory writer다. 최종 QA
answer decoder가 아니다.

### QA decoder

Episodic, semantic, visual, spatial evidence가 합쳐진 prompt를 받아 답을
생성하는 Gemma backend다.

## 4. 최적화 문제

입력 stream을 `X_1:T`, geometry provider를 `G`, 압축 memory를 `M_T`,
질문과 정답을 `(q, y)`, retrieval을 `R`, QA decoder를 `D`라고 두면 목표는
다음처럼 정리할 수 있다.

```text
minimize
    E[L_QA(D(R(q, E, S, V, M_T)), y)]
  + lambda_bytes * bytes(M_T)
  + lambda_write * writes(M_T)
  + lambda_geometry * L_geometry
  + lambda_temporal * L_temporal

subject to
    token.end_time <= question_time
    bytes(M_window) <= device_budget
    latency_per_frame <= device_latency_budget
    memory construction is query-independent
```

중요한 점:

- 압축 objective는 reconstruction loss만으로 정의하면 안 된다.
- 미래 질문을 알 수 없으므로 생성 단계는 query-independent여야 한다.
- QA utility, geometry fidelity, storage, write frequency 사이 Pareto frontier를
  측정해야 한다.

## 5. 전체 구조

```text
1 Hz image / pose / gaze / object observations
                  |
                  v
        SpatialGeometryEncoder
                  |
                  v
        SpatialProjectionHead
                  |
                  v
         SpatialTokenDecoder
          | candidate record
          | candidate features
          v
       SpatialMemoryCodec
          |
          v
 learned selector gate + causal window cap
          |
          v
         spatial store
          |
          +-------------------------------+
                                          |
episodic store ----------------------------|
semantic store ----------------------------|--> WorldMM retrieval
visual store ------------------------------|       |
                                                  v
                                        one EvidencePack
                                                  |
                                      sampled video frames
                                                  |
                                                  v
                                          Gemma QA decoder
```

코드상 `SpatialMemoryModel`은 encoder, projection head, token decoder, codec,
selector와 옵션을 묶는 experiment-level composite model이다.

- 구현:
  [spatial_compression.py](../src/worldmm_smvqa/worldmm/spatial_compression.py)
- 저장 schema:
  [spatial_types.py](../src/worldmm_smvqa/worldmm/spatial_types.py)
- 기본 experiment:
  [source_compact_v1.json](../configs/spatial/source_compact_v1.json)

## 6. 구성요소 계약

| 구성요소 | 역할 | 현재 구현 | 교체 시 유지할 계약 |
|---|---|---|---|
| `SpatialGeometryEncoder` | 원시 sensing을 geometry feature로 변환 | `structured-v1` | source별 causal geometry와 provenance |
| `SpatialProjectionHead` | provider별 feature를 decoder 공간으로 정규화 | `identity-v1` | `SpatialGeometryFeatureSet` 반환 |
| `SpatialTokenDecoder` | 변화와 관계를 저장 후보로 변환 | `delta-topk-v1` | record와 selector feature 반환 |
| `SpatialTokenSelector` | 후보의 보존 중요도 계산 | `linear-v1` | `[0, 1]` keep score |
| `SpatialMemoryCodec` | semantic token을 저장 payload로 encode/decode | `compact-json-v1` | versioned reversible token payload |

현재 runtime에서는 token decoder가 codec을 호출해 encoded candidate record를
만든 뒤 selector gate와 causal window cap이 저장 여부를 결정한다. 표의
component 분리는 experiment 교체 경계를 뜻한다.

Plugin output의 `encoder`, `projection_head`, 모든 geometry record의
`video_id`는 실행 중인 component/source와 일치해야 한다. 불일치하면 artifact를
쓰기 전에 실패한다.

### 6.1 현재 projection boundary의 정확한 의미

현재 baseline `identity-v1`은 PyTorch tensor projection layer가 아니다.
그러나 interface는 `zones`, `anchors`, `relations`, scalar
`extra_features`뿐 아니라 in-process `latent_state`도 전달한다. CUT3R state,
pointmap tensor, learned slot 같은 provider 전용 객체를 encoder에서 projection과
decoder까지 복사하거나 직렬화하지 않고 넘길 수 있다. `latent_state`는 memory
artifact에 저장되지 않으며 plugin이 type, shape, device, lifetime을 책임진다.

CUT3R adapter의 첫 구현은 다음처럼 연결할 수 있다.

```text
CUT3R state / pointmaps / confidence
  -> object and zone aggregation
  -> projection head over transient latent_state
  -> projected selector scalars in extra_features
  -> existing token decoder or new decoder plugin
```

배포 관점에서는 이 전체 composite가 하나의 spatial memory model이다. Component
경계는 encoder, projection, decoder를 따로 배포한다는 뜻이 아니라 checkpoint와
ablation 교체 지점이다. Encoder와 projection을 한 `nn.Module`이 공유해도 된다.

다만 현재 core에는 tensor batch, shape, dtype, device, gradient, checkpoint
계약이 없다. End-to-end neural training은 plugin 내부에서 이를 정의해야 하며,
여러 neural plugin이 공통 trainer를 공유해야 할 때만 typed batch 계약을 core에
추가한다.

## 7. 현재 baseline 동작

### 7.1 Structured geometry encoder

`structured-v1`은 기존 source schema의 다음 정보를 사용한다.

- object detection의 metric `x, y, z`
- gaze sample
- wearer pose 또는 SLAM-style pose
- frame metadata

처리 순서:

1. 2 m XY grid로 zone을 생성한다.
2. object detection을 spatial anchor로 변환한다.
3. 같은 시간대의 anchor pair로 `near`, `left_of`, `in_front_of`, `above`와
   역관계를 계산한다.
4. object geometry, gaze, pose provenance를 기록한다.

Geometry binding 구현:
[geometry_binding.py](../src/worldmm_smvqa/worldmm/geometry_binding.py)

### 7.2 Delta token decoder

`delta-topk-v1`은 다음 후보를 생성한다.

- `ZoneToken`
- `ObjectToken`
- `RelationToken`

Decoder는 후보마다 state key, state signature, object delta radius를 붙인다.
실제 retained state는 selector admission 단계에서 다음 규칙으로 갱신한다.

1. Zone은 provider가 반환한 `zone_id`마다 한 후보를 만들고, aggregate centroid의
   causal availability를 마지막 visit interval의 종료시각으로 둔다.
2. Object는 마지막으로 저장된 같은 label의 zone, 0.1 단위 confidence,
   provenance,
   frame-grounding이 같고 위치 거리가 기본 `2 * quantization_m` 이하이면
   반복 관측으로 제거한다.
3. Relation은 역관계를 canonical direction으로 변환한다.
   - `right_of` -> swapped `left_of`
   - `behind` -> swapped `in_front_of`
   - `below` -> swapped `above`
4. Relation의 canonical pair/kind state에서 zone과 quantized distance까지 같은
   저장 상태면 중복 제거한다.
5. Gate에서 drop됐거나 window cap 때문에 쓰지 못한 후보는 retained state를
   갱신하지 않는다. 같은 상태의 후속 관측은 다시 admission될 수 있다.

이 단계는 학습 가능한 score gate와 explicit delta state를 결합한 event-driven
write baseline이다. Neural recurrent write policy는 아직 후속 실험 대상이다.

### 7.3 Selector

기본 selector는 작은 logistic head다.

기본 feature:

- token kind: object, relation, zone
- detection confidence
- geometry provenance reliability
- frame-grounded 여부
- metric relation 여부
- recency
- provider 또는 projection이 추가한 dynamic feature

후보는 `(video_id, floor(end_time / window_seconds))`로 묶인다. 기본
`min_keep_score=0.5`를 통과한 후보를 `end_time` 순서로 admission하고, 같은
시각 후보만 keep score로 정렬한다. 한 window에서 최대 `token_budget`개를
저장하며 기본값은 30초당 16개다.

Admission은 irrevocable이다. 같은 window의 미래 고점 후보가 과거 token을
소급 제거하지 않으므로 어떤 question cutoff에서도 prefix memory가 미래 관측에
의존하지 않는다. 더 높은 offline top-K 품질이 필요하면 window-close latency를
명시하거나 replacement history를 저장하는 별도 정책으로 실험해야 한다.

주의: 현재 budget은 `SpatialTokenRecord`에만 적용된다. trajectory summary는
별도로 추가되므로 전체 spatial record 또는 byte에 대한 strict upper bound는
아니다.

### 7.4 Codec

`compact-json-v1`은 inspectable JSON array를 저장한다.

```text
Zone:
["Z", scale_cm, zone_id, qx, qy, qz]

Object:
["O", scale_cm, object_label, zone_id, qx, qy, qz,
 confidence_percent, provenance_code]

Relation:
["R", scale_cm, subject, relation_code, object, zone_id,
 quantized_distance_or_minus_one]
```

기본 `quantization_m=0.25`다. nearest-grid quantization이므로 이상적인 입력에서
축별 최대 오차는 `0.125 m`, 3D Euclidean 최대 오차는 약 `0.217 m`다.

Codec은 payload만 압축한다. JSONL field name, memory ID, provenance 등 record
metadata 비용은 그대로 남는다. 실제 device storage profiling 전에는 binary
codec을 추가하지 않는다.

### 7.5 저장 record

`SpatialTokenRecord`에는 다음이 포함된다.

- `memory_id`, `video_id`
- `encoder`
- `projection_head`
- `token_decoder`
- `codec`
- `start_time`, `end_time`
- encoded `token`
- `importance`
- 최대 한 개의 frame reference

이 provenance는 experiment 재현과 artifact decoder 선택에 사용된다.

## 8. Retrieval 및 QA 통합

Spatial token은 저장 후 retrieval adapter에서 다음으로 변환된다.

- human-readable `snippet`
- typed `geometry` dictionary
- `importance` 기반 base score
- causal time span
- optional frame reference

구현:
[retrieval.py](../src/worldmm_smvqa/retrieval.py)

그 후 E/S/V/S가 동일한 `RetrievalMemoryRecord` 목록에 들어간다.

```text
episodic + semantic + visual + spatial
  -> causal shard filtering
  -> WorldMM store routing and scoring
  -> evidence budget selection
  -> one EvidencePack
  -> one Gemma prompt
```

QA prompt는 spatial evidence도 다른 store와 같은 JSON evidence item으로
전달한다. `geometry` field만 spatial grounding 정보를 추가한다.

구현:
[qa_prompt.py](../src/worldmm_smvqa/qa_prompt.py)

통합 테스트:
[test_smoke_pipeline.py](../tests/test_smoke_pipeline.py)

따라서 spatial token decoder는 memory construction component이며, 별도
spatial QA decoder가 아니다.

## 9. Selector 학습 경로

구현:
[spatial_selector_train.py](../src/worldmm_smvqa/spatial_selector_train.py)

현재 학습 row 생성:

1. question 또는 answer choice에 geometry term이 있는 QA만 사용한다.
2. `candidate.end_time <= question_time`인 causal 후보만 사용한다.
3. label evidence span과 시간이 겹치는 후보를 positive로 둔다.
4. 기본 selector가 높게 평가한 나머지 후보를 hard negative로 선택한다.
5. class-weighted logistic regression을 학습한다.

중요한 anti-leakage 경계:

- production memory builder는 `labels.jsonl`을 읽지 않는다.
- evaluator-only trainer가 QA label을 읽어 selector weight를 만든다.
- 학습된 selector는 질문을 직접 입력받지 않고 memory 생성 시 사용된다.

현재 label은 "해당 token이 답에 필수인가"가 아니라 "evidence 시간 구간과
겹치는가"를 근사한다. 후속 단계에서는 counterfactual token removal로
token necessity label을 만들어야 한다.

## 10. Experiment 설정

```json
{
  "name": "source-compact-v1",
  "encoder": "structured-v1",
  "projection_head": "identity-v1",
  "token_decoder": "delta-topk-v1",
  "codec": "compact-json-v1",
  "selector": "linear-v1",
  "selector_path": null,
  "token_budget": 16,
  "quantization_m": 0.25,
  "window_seconds": 30.0,
  "min_keep_score": 0.5,
  "plugins": [],
  "encoder_options": {},
  "projection_options": {},
  "decoder_options": {"object_delta_multiplier": 2.0},
  "codec_options": {},
  "selector_options": {}
}
```

새 geometry provider, projection, decoder, codec, selector는 registry에
등록하고 experiment JSON에서 이름을 선택한다. Built-in component는 알 수 없는
option을 즉시 거부한다. QA, retrieval 코드는 변경하지 않는다.

Retrieval process는 `memory_manifest.json`의 `spatial_experiment`를 먼저 읽어
plugin과 codec option을 복원한 뒤 token을 decode한다. Plugin module import는
임의 코드를 실행하므로 experiment manifest는 신뢰한 run artifact만 사용한다.
Remote manifest 단계는 config만 해석하며 model plugin을 생성하지 않는다.
CUT3R 같은 GPU provider는 distributed memory worker 안에서만 생성된다.

주요 환경 변수:

- `WORLDMM_SPATIAL_EXPERIMENT_CONFIG`
- `WORLDMM_SPATIAL_SELECTOR_PATH`
- `WORLDMM_SPATIAL_TOKEN_BUDGET`
- `WORLDMM_SPATIAL_QUANTIZATION_M`

## 11. 평가 기준

### 11.1 압축

- total spatial bytes
- bytes per minute
- token records per minute
- write count per minute
- frame reference count
- legacy 대비 compression ratio

### 11.2 Geometry fidelity

- object position error: mean, median, P95
- zone assignment accuracy
- relation precision, recall, F1
- metric relation error
- moved object update recall
- stale object rate
- duplicate object and relation rate

### 11.3 Retrieval

- spatial Memory-Recall@K
- geometry question subset Recall@K
- E/S/V/S store contribution
- causal violation count
- retrieved evidence bytes와 prompt tokens

### 11.4 최종 QA

- Ans-F1
- QA-Acc
- QA-MRR
- spatial subset QA-Acc
- `without_spatial` ablation delta
- 동일 byte budget에서 encoder, decoder, codec 비교

### 11.5 Device

- encoder latency/frame
- projection 및 decoder latency/frame
- peak RAM
- persistent write bytes/min
- energy/frame
- thermal throttling 이후 steady-state throughput

QA 성능 하나만으로 device 적합성을 판단하지 않는다. 최소한
`QA utility vs bytes vs latency` Pareto curve가 필요하다.

## 12. 현재 local smoke 결과

Tiny fixture에서 legacy comparison을 명시적으로 활성화한 결과:

| 항목 | Legacy | Compressed |
|---|---:|---:|
| spatial records | 151 | 11 |
| JSONL bytes | 62,417 | 3,850 |
| compression ratio | 1.0x | 16.21x smaller |

이 값은 구조 검증용 tiny fixture 결과다. 실제 dataset, CUT3R feature,
on-device filesystem, binary encoding의 benchmark가 아니다.

현재 검증된 동작:

- repeated static object collapse
- inverse relation canonicalization
- quantization error bound
- moved object causal delta
- relation return-state delta and causal confidence
- causal per-window admission cap
- malformed token rejection
- custom selector loading
- encoder/projection/decoder/selector independent swap
- transient neural state handoff
- custom codec artifact reload
- decoder option behavior
- distributed compression statistics
- E/S/V/S unified evidence pack

## 13. 알려진 한계

### 13.1 Object identity

현재 object delta state는 `object_label`로 keying한다. 같은 scene에 mug가 두
개 있으면 하나로 합쳐질 수 있다. 실제 적용 전 instance tracking 또는
track ID가 필요하다.

### 13.2 Coordinate frame

현재 schema는 좌표계 ID, scale uncertainty, camera calibration version을
저장하지 않는다. Provider 교체 시 world/camera coordinate 혼합을 막는 typed
coordinate frame 계약이 필요하다.

### 13.3 Uncertainty

Detection confidence와 provenance heuristic만 사용한다. Depth variance,
pose covariance, triangulation confidence가 codec과 selector에 직접 반영되지
않는다.

### 13.4 Budget

Token admission cap은 record 수만 제한한다. Variable-length label, metadata,
trajectory record 때문에 byte budget이 보장되지 않는다.

### 13.5 Relation cost

Anchor pair relation 생성은 현재 O(n^2)이다. Dense provider를 연결한 뒤 실제
bottleneck이 확인되면 voxel hash 또는 spatial index로 교체한다.

### 13.6 Learned model 범위

현재 learned component는 linear selector뿐이다. Encoder, projection,
token decoder, codec은 baseline 구현이다. "end-to-end learned spatial token
compressor"는 아직 후속 실험 대상이다.

### 13.7 Label quality

Evidence time overlap은 token necessity의 약한 proxy다. Counterfactual QA
utility 또는 token-level geometry annotation이 필요하다.

### 13.8 Global lifetime과 supersession

현재 cap은 window별 write 수만 제한하며 전체 lifetime storage bound는 아니다.
변경 전 delta도 historical QA를 위해 JSONL에 남고 relation disappearance를
표현하는 tombstone도 없다. 실제 device current-state store에는 stable instance
key, supersession/tombstone, TTL 또는 compaction 정책이 필요하다. Historical QA
artifact와 on-device mutable state의 평가 계약도 분리해야 한다.

## 14. 필수 불변식

구현 변경 시 다음을 깨면 안 된다.

1. Production memory builder는 QA label을 읽지 않는다.
2. Token의 `end_time`은 관측 가능 시각보다 앞설 수 없다.
3. Retrieval은 `question_time` 이후 token을 제거한다.
4. Codec decode는 unknown version과 malformed payload를 명시적으로 거부한다.
5. Encoder, projection, decoder, codec provenance를 artifact에 남긴다.
6. Spatial output은 E/S/V와 동일한 retrieval 및 QA 경로를 사용한다.
7. 실험 비교는 동일 evidence budget, frame budget, causal cutoff를 사용한다.
8. 압축률 보고에는 저장 record metadata를 포함한 실제 artifact byte를 쓴다.
9. 같은 window의 미래 후보는 이미 admission된 과거 token을 제거하지 않는다.

## 15. 실행 예시

Local smoke:

```bash
WORLDMM_SPATIAL_EXPERIMENT_CONFIG=configs/spatial/source_compact_v1.json \
uv run worldmm-smvqa smoke \
  --fixture tests/fixtures/tiny_smvqa \
  --out /tmp/worldmm-spatial-smoke
```

Selector row 준비:

```bash
uv run python -m worldmm_smvqa.spatial_selector_train prepare \
  --fixture tests/fixtures/tiny_smvqa \
  --experiment configs/spatial/source_compact_v1.json \
  --out /tmp/spatial-selector-rows.jsonl
```

실제 selector training, memory build, retrieval, Gemma QA, benchmark evaluation은
company resource에서 실행한다. Remote run은
`manifests/spatial_experiment.json`에 effective config를 기록한다. Spatial build는
video 단위로 distributed partition되며 rank별 `source_count`, `record_count`,
`token_count`, `candidate_count`, `raw_bytes`, `compressed_bytes`를
`memory/worldmm_sv/spatial.stats.jsonl`에 남긴다. 이 CLI 통계의 `raw_*`는
동일 source에서 explicit snapshot baseline을 추가 생성한 비교값이다. Library의
기본 on-device build는 이 측정용 baseline을 생성하지 않는다.
Remote final report는 rank 통계를 합산해 raw/compressed bytes, compression
ratio, spatial token count를 QA metric과 함께 기록한다.
