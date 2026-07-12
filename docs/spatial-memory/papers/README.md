# Paper Evidence Index

| Field | Value |
|---|---|
| Page ID | SM-PAPERS |
| Status | Active evidence catalog |
| Last checked | 2026-07-11 |
| Paper pages | 42 |
| Confluence parent | SM-ROOT |

각 논문에는 하나의 페이지가 있으므로 하나의 Confluence 하위 페이지가 될 수 있다. 이 페이지는 작성자가 보고한 증거, 프로젝트 추론 및 프로젝트 재현 상태를 구분한다. 출판 또는 코드 가용성은 `Last checked` 날짜로 고정된다.

## Target Benchmark and QA Evaluation

| Paper | Evidence used here | Project position |
|---|---|---|
| [SuperMemory-VQA](supermemory-vqa.md) | Target data and four-choice metric contract | Primary target; official run not started |
| [OpenEQA](openeqa.md) | Smart-glass historical EQA and spatial-reasoning gap | Evaluation context |
| [Memory-Centric EQA](memory-centric-embodied-question-answering.md) | Explicit memory construction for EQA | Comparison candidate |
| [SQA3D](sqa3d.md) | Situated position and orientation reasoning | Geometry evaluation source |
| [ScanQA](scanqa.md) | Object-grounded 3D QA | Geometry evaluation source |
| [MV-ScanQA](mv-scanqa.md) | Multi-view evidence reasoning | Multi-evidence evaluation source |
| [VSI-Bench](vsi-bench.md) | Spatial perception, memory, and recall | External spatial benchmark |
| [LongSpace](longspace.md) | 1 fps long-horizon spatial video memory | External long-horizon comparison |

## Geometry Providers

| Paper | Evidence used here | Project position |
|---|---|---|
| [CUT3R](cut3r.md) | Recurrent continuous 3D state | Teacher context; state is not persistent DB |
| [G-CUT3R](g-cut3r.md) | Pose, intrinsics, and depth guidance | Preferred external teacher; blocked locally |
| [VGGT](vggt.md) | Feed-forward multi-view geometry | Provider comparator |
| [Depth Anything V2](depth-anything-v2.md) | Lightweight relative-depth candidate | Baseline candidate |
| [UniDepthV2](unidepth-v2.md) | Monocular metric-depth candidate | Baseline candidate |

## Streaming Memory and Association

| Paper | Evidence used here | Project position |
|---|---|---|
| [Spann3R](spann3r.md) | Working and sparse long-term memory split | Retention baseline |
| [LONG3R](long3r.md) | Fixed-capacity adaptive spatial memory | Retention baseline |
| [MeMix](memix.md) | Selective recurrent-state updates | Working-state baseline only |
| [Mem3R](mem3r.md) | Tracking and geometry memory separation | Architecture evidence |
| [Point3R](point3r.md) | Position-indexed explicit pointer memory | Pointer baseline |
| [TTT3R](ttt3r.md) | Test-time recurrent-state stabilization | Working-state candidate only |
| [TTSA3R](ttsa3r.md) | Temporal-spatial adaptive state updates | Working-state candidate only |
| [Ray-Aware Pointer Memory](ray-aware-pointer-memory.md) | Position, viewing ray, and adaptive replacement | Association baseline |

## Explicit Scene Graphs and Spatial Databases

| Paper | Evidence used here | Project position |
|---|---|---|
| [ConceptGraphs](conceptgraphs.md) | Object-centric open-vocabulary 3D graph | Typed-object evidence |
| [GraphEQA](grapheqa.md) | Scene graph plus small visual memory for EQA | Geometry core and reservoir evidence |
| [HOV-SG](hov-sg.md) | Hierarchical floor, room, and object graph | Submap hierarchy evidence |
| [Hydra](hydra.md) | Real-time hierarchical 3D scene graph | Place/submap and loop-closure evidence |
| [FARM](farm.md) | Compact relational object memory | Query-time relation evidence |
| [DAAAM](daaam.md) | Hierarchical 4D graph and tool-based QA | Temporal executor evidence |
| [FOUND-IT](found-it.md) | Task-driven granularity on demand | Evidence-reservoir candidate |
| [Mono-Hydra++](mono-hydra-plus-plus.md) | Monocular RGB and IMU metric scene graph | Pose-guidance evidence |

## Token Selection and Bottlenecks

| Paper | Evidence used here | Project position |
|---|---|---|
| [Good Token Hunting](good-token-hunting.md) | Coverage and diversity-aware geometry token selection | Selection baseline |
| [TokenLearner](tokenlearner.md) | Direct fixed-count learned tokens | Decoder baseline |
| [Perceiver](perceiver.md) | Cross-attention into fixed latent slots | Decoder baseline |
| [BLIP-2](blip-2.md) | Frozen-model query bottleneck | Teacher-student bridge context |
| [DART](dart.md) | Duplicate-first visual token removal | Redundancy baseline |
| [VisionZip](visionzip.md) | Dominant-token selection and merging | Visual-token baseline |
| [FEATHER](feather.md) | Localization-aware pruning failure analysis | Coverage protection evidence |
| [Geometry-Aware Token Pruning](geometry-aware-token-pruning.md) | Voxel-overlap pruning before 3D QA | Geometry-novelty baseline |

이 논문은 주로 일시적 추론 토큰을 줄이다. 영구 유형의 공간 메모리 byte budget를 설정하지 않는다.

## Discrete and Model Compression

| Paper | Evidence used here | Project position |
|---|---|---|
| [Finite Scalar Quantization](finite-scalar-quantization.md) | Simple discrete latent baseline | Deferred until typed baseline is measured |
| [VQ-VAE](vq-vae.md) | Learned discrete representation baseline | Deferred |
| [End-to-end Optimized Image Compression](end-to-end-optimized-image-compression.md) | Rate-distortion objective | Objective context only |
| [QVGGT](qvggt.md) | Geometry-aware model quantization | Later deployment baseline, not memory compression |

## Unknown-Question Evidence Reservoir

| Paper | Evidence used here | Project position |
|---|---|---|
| [Worth Remembering](surprise-gated-robot-episodic-memory.md) | Surprise-gated episodic writes under equal budget | Reservoir candidate; cannot replace static core |

## Status Vocabulary

| Status | Meaning |
|---|---|
| Reviewed | Primary source and relevant claims checked |
| Reproduced | Project experiment reproduces a named claim under a pinned contract |
| Adopted | Evidence supports an accepted ADR; not equivalent to reproduction |
| Candidate | Worth an equal-budget project experiment |
| Deferred | Not justified before a simpler baseline is measured |
| Needs verification | Canonical source or numerical claim is unresolved |

## Adding a Paper

1. Repository `docs/spatial-memory/papers/TEMPLATE.md`를 복사한다.
2. 주요 논문, 출판 현황, 버전, 공식 코드를 확인한다.
3. 데이터세트, 측정항목, 테이블 또는 그림 위치와 함께 작성자가 보고한 결과를 추가한다.
4. 이 프로젝트에 대해 논문이 증명하지 못한 점을 기술한다.
5. 지원되는 주장, ADR 및 실험을 연결한다.
6. 이 인덱스에 행을 하나 추가한다.

PDF를 커밋하거나 긴 초록을 복사하지 않는다. 기본 링크와 간결한 프로젝트 관련 증거를 저장한다.

## Secondary Bibliography Policy

- A primary source used to support, reject, or scope a project claim requires
  its own paper page, index row, and reverse `C-*` mapping.
- A secondary source used only for terminology, discovery, or background may
  remain an external link inside the relevant paper page. Label it
  `Secondary context`; do not use it as traceability evidence or reproduce its
  numerical claim.
- Promote a secondary source to a paper page before citing it in an ADR,
  experiment hypothesis, result interpretation, or traceability row.
- Survey and benchmark bibliographies are not recursively imported. The paper
  index is a claim-evidence catalog, not an exhaustive bibliography.

[Back to project home](../README.md)
