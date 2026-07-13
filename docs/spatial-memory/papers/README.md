# 논문 근거 목록

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPERS |
| 상태 | 활성 evidence catalog |
| 최종 확인 | 2026-07-11 |
| 논문 page | 42 |
| Confluence parent | SM-ROOT |

## 핵심 결론

- **P0 적용:** G-CUT3R/CUT3R는 external teacher 후보로만 사용하고 dense state는
  저장하지 않는다.
- **현재 채택:** Explicit scene graph, query-time relation, causal evaluation,
  deterministic proof의 설계 근거를 유지한다.
- **후속 비교:** Streaming memory와 token-selection 기법은 동일 byte budget의
  baseline으로만 평가한다.
- **보류:** Explicit baseline 측정 전 VQ/FSQ와 model quantization에 투자하지 않는다.
- **근거 수준:** 42편 모두 external evidence이며 project reproduction result는 없다.

각 page는 `핵심 결론 → 근거 상태 → 논문 핵심` 순서로 project decision을 먼저
제시한다. 출판·code 상태는 `최종 확인` 날짜 기준이다.

## 목표 benchmark와 QA 평가

| 논문 | 프로젝트에서 사용하는 근거 | 프로젝트 판단 |
|---|---|---|
| [SuperMemory-VQA](supermemory-vqa.md) | Target data와 four-choice metric contract | Primary target, official run 미시작 |
| [OpenEQA](openeqa.md) | Smart-glass historical EQA와 spatial-reasoning gap | Evaluation context로 사용 |
| [Memory-Centric EQA](memory-centric-embodied-question-answering.md) | EQA용 explicit memory construction | Comparison candidate로 평가 |
| [SQA3D](sqa3d.md) | Situated position/orientation reasoning | Geometry evaluation source로 사용 |
| [ScanQA](scanqa.md) | Object-grounded 3D QA | Geometry evaluation source로 사용 |
| [MV-ScanQA](mv-scanqa.md) | Multi-view evidence reasoning | Multi-evidence evaluation source로 사용 |
| [VSI-Bench](vsi-bench.md) | Spatial perception, memory, recall | External spatial benchmark로 사용 |
| [LongSpace](longspace.md) | 1 fps long-horizon spatial video memory | External long-horizon 비교에 사용 |

## Geometry provider 분류

| 논문 | 프로젝트에서 사용하는 근거 | 프로젝트 판단 |
|---|---|---|
| [CUT3R](cut3r.md) | Recurrent continuous 3D state | Teacher context, state는 persistent DB가 아님 |
| [G-CUT3R](g-cut3r.md) | Pose, intrinsics, depth guidance | 선호 external teacher, 로컬 차단 |
| [VGGT](vggt.md) | Feed-forward multi-view geometry | Provider comparator로 사용 |
| [Depth Anything V2](depth-anything-v2.md) | Lightweight relative-depth | Baseline candidate로 평가 |
| [UniDepthV2](unidepth-v2.md) | Monocular metric depth | Baseline candidate로 평가 |

## Streaming memory와 association

| 논문 | 프로젝트에서 사용하는 근거 | 프로젝트 판단 |
|---|---|---|
| [Spann3R](spann3r.md) | Working/long-term memory 분리 | Retention baseline으로 사용 |
| [LONG3R](long3r.md) | Fixed-capacity adaptive spatial memory | Retention baseline으로 사용 |
| [MeMix](memix.md) | Selective recurrent-state update | Working-state baseline만 사용 |
| [Mem3R](mem3r.md) | Tracking/geometry memory 분리 | Architecture evidence로 사용 |
| [Point3R](point3r.md) | Position-indexed explicit pointer memory | Pointer baseline으로 사용 |
| [TTT3R](ttt3r.md) | Test-time recurrent-state stabilization | Working-state candidate만 사용 |
| [TTSA3R](ttsa3r.md) | Temporal-spatial adaptive state update | Working-state candidate만 사용 |
| [Ray-Aware Pointer Memory](ray-aware-pointer-memory.md) | Position, viewing ray, adaptive replacement | Association baseline으로 사용 |

## Explicit scene graph와 spatial database

| 논문 | 프로젝트에서 사용하는 근거 | 프로젝트 판단 |
|---|---|---|
| [ConceptGraphs](conceptgraphs.md) | Object-centric open-vocabulary 3D graph | Typed-object evidence로 사용 |
| [GraphEQA](grapheqa.md) | Scene graph와 small visual memory | Geometry core/reservoir evidence로 사용 |
| [HOV-SG](hov-sg.md) | Hierarchical floor/room/object graph | Submap hierarchy evidence로 사용 |
| [Hydra](hydra.md) | Real-time hierarchical 3D scene graph | Place/submap, loop-closure evidence로 사용 |
| [FARM](farm.md) | Compact relational object memory | Query-time relation evidence로 사용 |
| [DAAAM](daaam.md) | Hierarchical 4D graph와 tool-based QA | Temporal executor evidence로 사용 |
| [FOUND-IT](found-it.md) | Task-driven granularity on demand | Evidence-reservoir candidate로 평가 |
| [Mono-Hydra++](mono-hydra-plus-plus.md) | Monocular RGB/IMU metric scene graph | Pose-guidance evidence로 사용 |

## Token selection과 bottleneck

| 논문 | 프로젝트에서 사용하는 근거 | 프로젝트 판단 |
|---|---|---|
| [Good Token Hunting](good-token-hunting.md) | Coverage/diversity-aware geometry token selection | Selection baseline으로 사용 |
| [TokenLearner](tokenlearner.md) | Direct fixed-count learned token | Decoder baseline으로 사용 |
| [Perceiver](perceiver.md) | Fixed latent slot cross-attention | Decoder baseline으로 사용 |
| [BLIP-2](blip-2.md) | Frozen-model query bottleneck | Teacher-student bridge context |
| [DART](dart.md) | Duplicate-first visual token removal | Redundancy baseline으로 사용 |
| [VisionZip](visionzip.md) | Dominant-token selection/merge | Visual-token baseline으로 사용 |
| [FEATHER](feather.md) | Localization-aware pruning failure | Coverage protection evidence로 사용 |
| [Geometry-Aware Token Pruning](geometry-aware-token-pruning.md) | 3D QA 전 voxel-overlap pruning | Geometry-novelty baseline으로 사용 |

이 논문군은 주로 transient reasoning token을 줄이며 persistent typed spatial
memory의 byte budget을 설정하지 않는다.

## Discrete/model compression 분류

| 논문 | 프로젝트에서 사용하는 근거 | 프로젝트 판단 |
|---|---|---|
| [Finite Scalar Quantization](finite-scalar-quantization.md) | Simple discrete latent baseline | Typed baseline 측정 전 보류 |
| [VQ-VAE](vq-vae.md) | Learned discrete representation baseline | 보류 |
| [End-to-end Optimized Image Compression](end-to-end-optimized-image-compression.md) | Rate-distortion objective | Objective context만 사용 |
| [QVGGT](qvggt.md) | Geometry-aware model quantization | 후속 deployment baseline이며 memory compression 아님 |

## 미지 질문용 evidence reservoir

| 논문 | 프로젝트에서 사용하는 근거 | 프로젝트 판단 |
|---|---|---|
| [Worth Remembering](surprise-gated-robot-episodic-memory.md) | Equal budget surprise-gated episodic write | Reservoir candidate로 평가; static core 대체 불가 |

## 상태 정의

| 상태 | 의미 |
|---|---|
| 검토 완료 | Primary source와 relevant claim 확인 완료 |
| 재현 완료 (Reproduced) | Pinned contract의 project experiment가 named claim 재현 |
| 채택 (Adopted) | Evidence가 accepted ADR을 지원하나 reproduction과 동일하지 않음 |
| 후보 (Candidate) | Equal-budget project experiment 가치 있음 |
| 보류 (Deferred) | 더 단순한 baseline 측정 전 정당화되지 않음 |
| 검증 필요 (Needs verification) | Canonical source 또는 numerical claim 미확인 |

## 근거 관리 원칙

- Traceable claim은 primary-source page, index row, reverse `C-*` mapping이
  필요하다.
- Secondary source는 discovery에만 사용하며 decision/result에 쓰기 전에
  primary page로 승격한다.
- Paper-reported value, project inference, reproduction을 분리한다.
- PDF, 긴 abstract, recursive bibliography를 commit하지 않는다.
- 새 page는 `docs/spatial-memory/papers/TEMPLATE.md`를 사용한다.

[프로젝트 홈으로 돌아가기](../README.md)
