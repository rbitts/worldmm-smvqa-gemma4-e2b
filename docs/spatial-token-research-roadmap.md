# Spatial Token Compression Research Roadmap

> Legacy migration source. Canonical research navigation now lives in
> [spatial-memory/roadmap.md](spatial-memory/roadmap.md), with one page per paper
> under [spatial-memory/papers](spatial-memory/papers/README.md).

## 1. 목적

이 문서는 spatial token compression을 고도화할 때 검토할 논문과 접근법을
정리한다. 단순 논문 목록이 아니라 각 아이디어를 현재 `SpatialMemoryModel`의
교체 지점, 실험 가설, 평가 기준, 중단 조건에 연결한다.

현재 구조와 구현 계약은
[spatial-token-compression.md](spatial-token-compression.md)를 기준으로 한다.

## 2. 연구 질문

핵심 질문은 하나다.

> 미래 geometry QA 성능을 유지하면서 1 Hz AI Glass가 저장하고 갱신해야 하는
> spatial state를 얼마나 작게 만들 수 있는가?

이를 다음 하위 질문으로 나눈다.

1. Dense geometry provider의 어떤 state가 장기 기억에 실제 필요한가?
2. Frame마다 쓰지 않고 언제 memory를 갱신해야 하는가?
3. Object, voxel, latent slot 중 어떤 token 단위가 QA에 효율적인가?
4. Token 중요도를 reconstruction이 아니라 QA utility로 학습할 수 있는가?
5. Explicit geometry와 learned latent를 어떤 비율로 섞어야 하는가?
6. 압축 memory가 E/S/V retrieval과 결합될 때 최종 QA 이득이 있는가?

## 3. 현재 판단

첫 production 후보는 다음 조합이다.

```text
object/zone/relation explicit core
  + event-gated writes
  + QA-supervised fixed-budget selector
  + coarse quantization
  + provider provenance and uncertainty
```

이 조합을 우선하는 이유:

- Object-centric graph는 language QA와 직접 연결하기 쉽다.
- Event-driven write는 1 Hz 반복 관측의 저장 비용을 직접 줄인다.
- Static token의 window별 fixed budget은 write rate upper bound를 관리하기
  쉽다. Trajectory와 lifetime storage까지 포함한 full-artifact bound는 별도다.
- Explicit token은 failure analysis와 geometry metric 계산이 가능하다.
- Dense neural scene representation 전체를 저장하는 것보다 구현 및 검증
  비용이 작다.

Latent-only memory는 첫 단계가 아니다. Explicit baseline보다 같은 byte에서
QA와 geometry fidelity가 높다는 증거가 생긴 뒤 추가한다.

## 4. 논문별 근거

### 4.1 Geometry provider와 persistent reconstruction

#### CUT3R

[CUT3R: Continuous 3D Perception Model with Persistent State](https://arxiv.org/abs/2501.12387)

논문 아이디어:

- 이미지 stream을 recurrent state에 통합한다.
- camera pose와 pointmap을 unified representation으로 예측한다.
- online processing과 persistent state를 제공한다.

우리 적용:

- `SpatialGeometryEncoder`의 teacher 또는 high-quality provider 후보
- recurrent state에서 object/zone/relation token 추출
- 기존 structured provider 대비 geometry oracle 역할

주의:

- CUT3R state 자체를 memory artifact로 저장하면 device compression 목표를
  달성하지 못한다.
- 첫 실험은 CUT3R 전체 on-device 배치보다 remote teacher feature 추출과
  adapter 검증이 적절하다.

#### G-CUT3R

[G-CUT3R: Guided 3D Reconstruction with Camera and Depth Prior Integration](https://arxiv.org/abs/2508.11379)

논문 아이디어:

- external camera와 depth prior를 plug-and-play 방식으로 CUT3R에 결합한다.
- original weight를 크게 바꾸지 않고 geometry prior를 수용한다.

우리 적용:

- AI Glass의 VIO/SLAM pose, depth sensor, calibration prior를 provider에
  주입하는 설계 근거
- camera/depth prior 유무 ablation
- geometry provider를 전부 재학습하지 않는 adapter 경로

#### VGGT

[VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.11651)

논문 아이디어:

- 여러 view에서 camera, depth, point map, track을 feed-forward 방식으로
  공동 예측한다.

우리 적용:

- sparse multi-view geometry teacher
- CUT3R와 다른 non-recurrent provider 비교군
- provider 종류가 token decoder 성능에 미치는 영향 분리

#### Depth Anything V2와 UniDepthV2

- [Depth Anything V2](https://arxiv.org/abs/2406.09414)
- [UniDepthV2](https://arxiv.org/abs/2502.20110)

우리 적용:

- CUT3R보다 작은 monocular depth provider 후보
- device pose와 결합한 lightweight geometry baseline
- teacher-student distillation의 student backbone 후보

제약:

- monocular depth의 scale과 temporal consistency를 별도 보정해야 한다.
- 단일 frame depth 성능만으로 persistent geometry 품질을 판단하면 안 된다.

### 4.2 Memory state와 write policy

#### LONG3R

[LONG3R: Long Sequence Streaming 3D Reconstruction](https://arxiv.org/abs/2507.18255)

논문 아이디어:

- recurrent 3D spatio-temporal memory를 유지한다.
- memory gating으로 현재 observation에 관련된 state를 고른다.
- 중복 spatial information을 동적으로 줄이고 scene에 따라 resolution을
  조절한다.

우리 적용:

- 현재 `zone + object` token과 비교할 adaptive spatial-memory decoder
- 같은 공간 반복 방문의 gating 및 redundancy 제거
- fixed voxel grid와 adaptive-resolution slot의 비교 근거

가설:

- object detection이 약한 장면에서는 adaptive spatial token이 explicit object
  graph보다 geometry recall을 높일 수 있다.
- object-heavy QA에서는 semantic label 없는 voxel token 때문에 prompt 효율이
  낮을 수 있다.

#### MeMix

[MeMix: Writing Less, Remembering More for Streaming 3D Reconstruction](https://arxiv.org/abs/2603.15330)

논문 아이디어:

- recurrent state를 여러 memory patch로 나눈다.
- 새 observation과 가장 덜 정렬된 patch만 갱신하고 나머지는 보존한다.
- training-free, plug-and-play update로 O(1) inference memory를 유지한다.

우리 적용:

- 현재 거리 threshold 기반 delta rule과 selective patch update 비교
- 새 object, zone transition, relation change, uncertainty 감소, stale state
  correction에만 write하는 task-specific extension

가설:

- selector top-K보다 먼저 selective update를 적용하면 candidate 생성과 저장을
  동시에 줄일 수 있다. Learned gate는 MeMix 자체가 아니라 후속 실험이다.

#### Mem3R

[Mem3R: Streaming 3D Reconstruction with Hybrid Memory via Test-Time Training](https://arxiv.org/abs/2604.07279)

논문 아이디어:

- camera tracking과 geometry mapping memory를 분리한다.
- tracking은 test-time training으로 갱신되는 lightweight fast-weight MLP를,
  mapping은 explicit fixed-size token state를 사용한다.

우리 적용:

- wearer pose/state는 implicit recurrent state
- object, relation, zone은 explicit QA memory
- 전체 geometry를 record로 저장하지 않는 hybrid design

가설:

- QA에서 직접 인용할 정보만 explicit하게 남기고 tracking state는 hidden
  memory로 유지하면 bytes와 drift를 함께 줄일 수 있다.

#### Point3R

[Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory](https://arxiv.org/abs/2507.02863)

논문 아이디어:

- implicit global token 대신 explicit spatial pointer memory를 쓴다.
- memory entry가 spatial region을 직접 가리킨다.

우리 적용:

- `zone_id`보다 정밀한 pointer 또는 key-value spatial index
- pointer distribution과 local retrieval
- token decoder의 pointer-slot 버전

#### Emerging watchlist

- [Mono-Hydra++](https://arxiv.org/abs/2605.17661): monocular RGB와 IMU로
  metric semantic map과 hierarchical 3D scene graph를 구축하는 방향
- [Ray-Aware Pointer Memory with Adaptive Updates](https://arxiv.org/abs/2605.05749):
  position, viewing ray, feature를 가진 pointer를 retain-or-replace 방식으로
  갱신하는 방향

두 논문은 2026 preprint다. 핵심 실험이 재현되기 전 production 우선순위로
올리지 않는다.

### 4.3 Token selection과 fixed-size decoder

#### Good Token Hunting

[Good Token Hunting: A Hitchhiker's Guide to Token Selection for Visual Geometry Transformers](https://arxiv.org/abs/2605.23892)

논문 아이디어:

- frame-level diversity selection과 frame 내부 token selection을 분리한다.
- inter-frame selection에는 scene coverage를 위한 diversity가 중요하고,
  intra-frame selection에는 layer-aware sparsification이 필요하다고 보고한다.

우리 적용:

- selector feature에 confidence뿐 아니라 coverage와 novelty 추가
- voxel/zone coverage regularization
- 같은 object와 region token의 redundancy penalty

이 논문은 geometry-specific selection 근거로 직접적이지만 2026 preprint이므로
우리 dataset에서 독립 검증한다. 또한 논문의 대상은 transformer attention
compute이며 persistent QA memory storage로의 전이는 우리 가설이다.

#### TokenLearner

[TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?](https://arxiv.org/abs/2106.11297)

논문 아이디어:

- dense spatial features를 작은 수의 adaptive token으로 압축한다.

우리 적용:

- projected CUT3R feature를 fixed-K latent slot으로 pooling하는 decoder
- K를 4, 8, 16, 32로 고정해 device budget을 직접 제어

제약:

- generic visual token은 metric coordinate를 보장하지 않는다.
- coordinate, relation, temporal delta auxiliary loss가 필수다.

#### Perceiver와 BLIP-2 Q-Former

- [Perceiver](https://arxiv.org/abs/2103.03206)
- [BLIP-2](https://arxiv.org/abs/2301.12597)

논문 아이디어:

- 작은 learned query 또는 latent array가 큰 입력 feature 집합에서 필요한
  정보를 cross-attention으로 읽는다.

우리 적용:

- `SpatialTokenDecoder`를 fixed learned queries로 구현
- query type을 object, relation, zone, motion slot으로 분리
- output slot에 explicit geometry head와 discrete codec 연결

#### DART와 VisionZip

- [Stop Looking for Important Tokens in Multimodal Language Models: Duplication
  Matters More (DART)](https://arxiv.org/abs/2502.11494)
- [VisionZip](https://arxiv.org/abs/2412.04467)

논문 아이디어:

- DART는 pivot과 중복도가 낮은 token을 보존한다.
- VisionZip은 dominant token을 선택하고 contextual token 정보를 병합한다.
- 둘 다 visual-token redundancy를 줄여 LLM 입력 비용을 낮춘다.

우리 적용:

- 동일 quantized coordinate, object instance, relation signature의
  duplicate-first 제거
- spatial memory 생성 전 후보 수 감소

제약:

- 주로 VLM inference token reduction을 다룬다.
- 장기 persistent geometry 보존 여부는 별도 검증이 필요하다.

#### FEATHER

[Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](https://arxiv.org/abs/2412.13180)

보고된 관찰:

- aggressive token reduction은 localization-sensitive task에서 손실을 만들 수
  있다.

우리 적용:

- generic attention score만으로 spatial token을 제거하지 않는다.
- object coverage, coordinate error, relation recall guardrail을 둔다.

### 4.4 Object-centric scene graph

#### ConceptGraphs

[ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning](https://arxiv.org/abs/2309.16650)

논문 아이디어:

- object-centric 3D map과 open-vocabulary relation을 결합한다.
- language query와 planning에 쓸 explicit scene graph를 만든다.

우리 적용:

- object anchor와 relation token을 spatial memory의 explicit core로 유지
- visual/semantic memory와 object identity 공유
- QA에서 object-relation evidence 직접 인용

제약:

- full object graph는 장기 stream에서 계속 증가한다.
- instance merge, stale expiration, graph compaction 정책이 필요하다.

#### GraphEQA

[GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied Question Answering](https://arxiv.org/abs/2412.14480)

논문 아이디어:

- online semantic scene graph를 embodied QA의 explicit world model로 쓴다.

우리 적용:

- spatial token을 QA evidence로 연결하는 architecture 근거
- object/room/relation graph와 text retrieval 결합 비교
- supporting memory ID를 이용한 evidence attribution

### 4.5 Discrete codec와 rate-distortion

#### Finite Scalar Quantization

[Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505)

논문 아이디어:

- learned representation을 finite scalar level 조합으로 discrete code화한다.
- 별도 codebook lookup 없이 compact discrete latent를 만든다.

우리 적용:

- projected geometry latent를 작은 integer tuple로 저장
- codebook collapse 위험이 적은 첫 learned codec 후보
- decoder plugin과 codec plugin을 분리해 비교

#### VQ-VAE

[Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)

우리 적용:

- object 또는 voxel latent를 codebook ID로 저장
- repeated geometry가 동일 code를 공유하도록 학습

제약:

- codebook collapse와 unknown geometry 일반화 문제가 있다.
- explicit coordinate를 모두 latent ID로 대체하지 말고 residual 또는 hybrid
  codec부터 검토한다.

#### Rate-distortion training

[End-to-end Optimized Image Compression](https://arxiv.org/abs/1611.01704)

우리 적용:

```text
L = L_QA
  + alpha * L_evidence
  + beta * L_coordinate
  + gamma * L_relation
  + delta * L_temporal
  + lambda * estimated_bits
```

`estimated_bits` 또는 actual serialized bytes를 rate term으로 쓴다.
Reconstruction PSNR 대신 QA와 geometry task loss를 distortion으로 둔다.

#### QVGGT

[QVGGT: Post-Training Quantized Visual Geometry Grounded Transformer](https://arxiv.org/abs/2605.31124)

논문 아이디어:

- VGGT의 geometry task별 sensitivity를 고려해 mixed-precision quantization을
  적용한다.

우리 적용:

- geometry provider 자체의 device memory와 latency를 줄이는 별도 축
- token memory compression과 encoder weight quantization을 분리 측정

주의:

- 2026 preprint다.
- model weight compression과 persistent spatial memory compression은 다른
  문제이므로 결과를 섞어 보고하지 않는다.

### 4.6 QA와 평가 benchmark

#### OpenEQA

- [OpenEQA project](https://open-eqa.github.io/)
- [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Majumdar_OpenEQA_Embodied_Question_Answering_in_the_Era_of_Foundation_Models_CVPR_2024_paper.html)

적용:

- episodic memory와 embodied spatial understanding 평가 참고
- human episodic memory와 model gap을 보는 외부 benchmark

#### Memory-Centric Embodied Question Answering

[Memory-Centric Embodied Question Answering](https://arxiv.org/abs/2505.13948)

적용:

- viewpoint comparison을 이용한 redundant observation write 억제
- entropy 기반 adaptive retrieval로 module별 최소 memory 선택
- multi-target, multi-region QA에서 update와 retrieval 평가
- memory가 answering뿐 아니라 planning과 stopping에도 기여하는 구조 참고

#### SQA3D와 ScanQA

- [SQA3D](https://arxiv.org/abs/2210.07474)
- [ScanQA](https://arxiv.org/abs/2112.10482)

적용:

- situated pose를 포함한 3D QA
- object relation과 metric grounding 평가
- explicit scene representation의 QA utility 확인

제약:

- static indoor scan과 1 Hz egocentric stream의 domain 차이가 크다.
- 외부 benchmark 결과를 on-device long-term memory 성능으로 직접 해석하지
  않는다.

#### VSI-Bench

[Thinking in Space: How Multimodal Large Language Models See, Remember, and
Recall Spaces](https://arxiv.org/abs/2412.14171)

적용:

- relative position, distance, direction, route 같은 spatial subset 정의
- geometry reasoning 유형별 error taxonomy

## 5. 권장 실험 순서

### P0. 현재 baseline 측정

모델 추가 전에 수행한다.

#### P0-A. Rate-quality sweep

변수:

- `token_budget`: 4, 8, 16, 32
- `byte_budget`: 1024, 2048, 4096, 8192
- `quantization_m`: 0.10, 0.25, 0.50, 1.00
- object delta threshold: 1x, 2x, 4x quantization step
- trajectory summary 포함/제외

측정:

- bytes/min
- writes/min
- object position P95
- relation F1
- spatial Recall@K
- spatial QA-Acc
- full QA-Acc

초기 수용 기준:

- baseline 대비 최소 4x byte 감소
- spatial QA-Acc 하락 1 percentage point 이하
- causal violation 0

실제 dataset 결과 전에는 이 기준을 제품 SLA로 고정하지 않는다.

#### P0-B. Instance identity

구현:

- provider track ID가 있으면 우선 사용
- 없으면 label + zone + geometry gating으로 short track 생성
- object token key를 label에서 instance ID로 변경

수용 기준:

- duplicate-label scene에서 object merge error 감소
- token 증가량이 QA 이득보다 크면 instance expiration 추가

#### P0-C. Byte budget

상태: static spatial-token baseline 구현 완료.

```text
default per 30s window:
  records(SpatialTokenRecord) <= 16
  sum(actual JSONL bytes(SpatialTokenRecord)) <= 4096
```

각 candidate의 전체 `SpatialTokenRecord` JSON 직렬화와 newline을 실제 byte
비용으로 측정한다. 관측시각 순서로 admission하고 같은 시각 후보만 score/byte
greedy로 정렬한다. 미래 token은 이미 admission된 과거 token을 퇴출하지 않아
prefix causality를 유지한다. 기존 16-token cap도 byte cap과 병행한다.

남은 한계: trajectory summary가 admission 뒤 추가되므로 현재 4096-byte cap은
full spatial artifact cap이 아니다. Full-artifact budget은 trajectory 비용 정책과
lifetime compaction 계약을 정한 뒤 추가한다. Knapsack 또는 differentiable
selection은 greedy가 SuperMemory-VQA Pareto에서 부족할 때만 검토한다.

### P1. Selector supervision 개선

#### P1-A. Counterfactual token utility

현재 evidence-overlap positive를 다음 label로 교체한다.

```text
utility(token_i)
  = QA_score(all_tokens)
  - QA_score(all_tokens without token_i)
```

비용 절감:

- geometry question subset에서만 계산
- retrieval top-N 후보만 제거 평가
- Gemma teacher output cache

수용 기준:

- 같은 byte budget에서 overlap-label selector보다 spatial QA-Acc와
  Memory-Recall@K가 모두 높아야 한다.

#### P1-B. Coverage-aware selector

추가 feature:

- minimum distance to selected token
- zone novelty
- object instance novelty
- relation novelty
- uncertainty reduction
- age and stale-state risk

비교:

- current linear score
- score/byte greedy
- facility-location greedy
- small MLP

MLP는 linear selector가 같은 feature로 부족하다는 결과가 나온 뒤 추가한다.

### P2. Geometry provider 비교

#### P2-A. Provider oracle

같은 source stream에서 다음을 비교한다.

- structured source geometry
- CUT3R
- G-CUT3R with pose/depth priors
- VGGT
- lightweight depth + device pose

모든 provider는 같은 projection, decoder, selector, codec을 사용한다. Provider
효과와 compression model 효과를 섞지 않는다.

측정:

- provider geometry fidelity
- candidate count
- compressed bytes
- spatial QA
- GPU latency와 peak memory

#### P2-B. Teacher-student

1. CUT3R 또는 G-CUT3R로 remote teacher geometry를 생성한다.
2. Lightweight depth + pose student가 object anchor와 uncertainty를 예측한다.
3. Student output을 동일 token decoder에 연결한다.
4. Teacher와 student의 final QA Pareto를 비교한다.

On-device 후보 선정은 geometry benchmark가 아니라 final compressed-memory QA
결과로 결정한다.

### P3. Selective write policy

MeMix의 training-free selective patch update와 task-specific learned gate를
분리해 비교한다.

입력:

- current projected geometry
- previous memory state
- pose delta
- object novelty
- relation change
- uncertainty change

출력:

- skip
- update existing token
- insert token
- expire token

Learned-gate 비교군의 학습 loss:

```text
L_write = L_QA_or_proxy + lambda_write * write_count
```

수용 기준:

- 현재 deterministic delta보다 writes/min 감소
- moved-object recall과 relation-change recall 유지
- stale state 증가 없음

### P4. Token decoder 비교

#### P4-A. Adaptive spatial-slot decoder

LONG3R의 gated 3D spatio-temporal memory와 adaptive-resolution 아이디어를
간단한 fixed-voxel baseline부터 비교한다.

비교 대상:

- object graph token
- fixed voxel token
- adaptive spatial slot
- object + coarse voxel hybrid

질문 유형별 예상:

- named-object QA: object graph 우세 가능
- free-space, distance, route QA: voxel 또는 hybrid 우세 가능

#### P4-B. Fixed-K cross-attention decoder

TokenLearner, Perceiver, Q-Former 형태의 learned query를 사용한다.

권장 slot:

- object slots
- relation slots
- zone slots
- motion/change slots

필수 auxiliary head:

- coordinate reconstruction
- relation classification
- token type
- uncertainty
- temporal validity

Latent slot만 저장하지 않고 최소한 type, time, coordinate-frame provenance를
함께 저장한다.

#### P4-C. Pointer memory

Point3R 방향으로 bounded pointer slot을 유지한다.

검증 질문:

- pointer가 explicit object ID보다 compact한가?
- pointer 재사용이 stale aliasing을 만들지 않는가?
- retrieval이 pointer neighborhood를 효율적으로 찾는가?

### P5. Learned discrete codec

순서:

1. 현재 scalar quantization
2. FSQ residual code
3. VQ codebook
4. mixed explicit-latent codec

권장 hybrid payload:

```text
explicit:
  token type, time, coordinate frame, zone/object identity

discrete latent:
  local shape, appearance-free geometry context, uncertainty pattern
```

수용 기준:

- actual serialized byte 감소
- coordinate 및 relation fidelity 유지
- unknown scene에서 catastrophic decode failure 없음
- codec version migration 가능

### P6. Hybrid implicit-explicit memory

Mem3R 방향:

- implicit: pose tracking, local temporal state
- explicit: object, zone, relation, evidence provenance
- optional latent: local geometry residual

이 단계는 P2 provider와 P4 decoder 결과가 나온 뒤 진행한다. 먼저 도입하면
어느 component가 성능을 만들었는지 분리하기 어렵다.

## 6. 실험 matrix

모든 experiment는 다음 축을 manifest에 남긴다.

| 축 | Baseline | 후보 |
|---|---|---|
| Encoder | structured-v1 | CUT3R, G-CUT3R, VGGT, lightweight depth |
| Projection | identity-v1 | scalar MLP, tensor projection |
| Decoder | delta-topk-v1 | event gate, voxel slot, fixed-K query, pointer |
| Selector | linear-v1 | score/byte, coverage greedy, MLP |
| Codec | compact-json-v1 | packed integer, FSQ, VQ |
| Budget | 16 records + 4096 B / 30s static tokens | token, byte, adaptive |

최소 ablation:

```text
full E/S/V/S
without spatial
spatial only
without relation token
without object token
without trajectory
without learned selector
without quantization
```

## 7. 학습 objective

권장 multi-task loss:

```text
L_total =
    L_QA
  + alpha * L_evidence_retrieval
  + beta * L_object_position
  + gamma * L_relation
  + delta * L_temporal_change
  + epsilon * L_uncertainty_calibration
  + lambda_rate * actual_or_estimated_bits
  + lambda_write * write_count
```

- `L_QA`: 최종 answer 또는 teacher distillation loss
- `L_evidence_retrieval`: 정답 evidence가 top-K에 남도록 하는 loss
- `L_object_position`: metric coordinate 또는 quantized-bin loss
- `L_relation`: relation type과 distance
- `L_temporal_change`: moved object, enter/leave, relation transition
- `L_uncertainty_calibration`: confidence와 실제 geometry error의 일치
- `actual_or_estimated_bits`: 저장 비용
- `write_count`: flash write 및 update 비용 proxy

초기에는 모든 loss를 end-to-end로 묶지 않는다. Selector, decoder, codec
순서로 독립 검증한 뒤 joint fine-tuning을 검토한다.

## 8. 평가 프로토콜

### 8.1 Split

- scene-disjoint
- user 또는 trajectory-disjoint
- object-category tail split
- long-duration split
- duplicate-instance stress split
- pose/depth-noise stress split

### 8.2 Causality

- 모든 token에 observation time과 validity interval 기록
- `token.end_time <= question_time`
- moved-object test에서 future state가 early question에 노출되지 않아야 함
- offline teacher feature도 causal window만 사용

### 8.3 비교 원칙

- 동일 question set
- 동일 E/S/V stores
- 동일 evidence budget
- 동일 frame sampling
- 동일 Gemma checkpoint와 decoding
- 동일 device byte budget
- 세 개 이상 seed 또는 bootstrap confidence interval

### 8.4 Pareto 보고

단일 최고 QA score 대신 다음 frontier를 보고한다.

```text
QA-Acc vs bytes/min
spatial QA-Acc vs writes/min
relation F1 vs decoder latency
Memory-Recall@K vs prompt tokens
```

## 9. On-device 배치 구조 후보

권장 단계적 구조:

```text
Device, 1 Hz:
  frame + VIO pose
    -> lightweight geometry student
    -> projection
    -> deterministic or learned write gate
    -> compact explicit tokens
    -> local spatial store

Remote training:
  CUT3R/G-CUT3R/VGGT teacher
    -> geometry targets
    -> selector utility labels
    -> student and decoder training

QA time:
  E/S/V/S retrieval
    -> one evidence pack
    -> Gemma QA
```

Provider가 device에 직접 올라가지 못해도 teacher로서 가치가 있다. 반대로
provider latency만 낮고 compressed-memory QA가 나쁘면 채택하지 않는다.

## 10. 당장 하지 않을 것

1. Full dense point cloud를 장기 memory artifact로 저장
2. 첫 실험부터 Gemma, CUT3R, decoder를 end-to-end 공동 학습
3. Generic VLM attention score만 사용한 spatial pruning
4. Device profiling 전 custom binary format 구현
5. 실사용 질문으로 검증되지 않은 대형 relation ontology 구축
6. Linear selector 비교 없이 Transformer selector 추가
7. Actual serialized bytes 대신 token count만 compression metric으로 보고

## 11. 추천 다음 세 실험

### Experiment 1: Rate-quality baseline

코드 변경 최소. `token_budget`, `byte_budget`, `quantization_m`, delta threshold를
sweep한다.
전체 dataset의 Pareto curve와 spatial subset QA를 만든다.

### Experiment 2: Instance-aware delta memory

Object label keying을 track ID로 교체한다. Duplicate-object stress test와
moved-object test를 추가한다.

### Experiment 3: Counterfactual selector

Top-N token removal로 QA utility label을 만들고 현재 overlap-label linear
selector와 비교한다.

이 세 실험이 끝나기 전에는 neural token decoder를 추가하지 않는다. 현재
압축 bottleneck이 provider, identity, selector 중 어디인지 먼저 확인해야
한다.

## 12. 논문 상태와 해석 원칙

- 2025-2026 arXiv-only 결과는 emerging evidence로 취급한다.
- 논문이 보고한 reconstruction 또는 VLM latency 개선을 우리 QA 개선으로
  간주하지 않는다.
- 모든 논문 아이디어는 SuperMemory-VQA의 causal E/S/V/S full pipeline에서
  재검증한다.
- 채택 기준은 novelty가 아니라 같은 device budget에서의 QA와 geometry
  Pareto 개선이다.
