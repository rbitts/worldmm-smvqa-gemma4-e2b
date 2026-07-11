# Paper Evidence Index

| Field | Value |
|---|---|
| Page ID | SM-PAPERS |
| Status | Active evidence catalog |
| Last checked | 2026-07-11 |
| Paper pages | 42 |
| Parent | [Spatial Memory project](../README.md) |

Each paper has one page so it can become one Confluence child page. The page
separates author-reported evidence, project inference, and project reproduction
status. Publication or code availability is pinned to the `Last checked` date.

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

These papers mainly reduce transient inference tokens. They do not establish a
persistent typed spatial-memory byte budget.

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

1. Copy [the paper template](TEMPLATE.md).
2. Verify the primary paper, publication status, version, and official code.
3. Add author-reported results with dataset, metric, and table or figure location.
4. State what the paper does not prove for this project.
5. Link supported claims, ADRs, and experiments.
6. Add one row to this index.

Do not commit PDFs or copy long abstracts. Store primary links and concise
project-relevant evidence.

[Back to project home](../README.md)
