# Evidence and Implementation Traceability

| Field | Value |
|---|---|
| Page ID | SM-TRACEABILITY |
| Status | Living index |
| Last updated | 2026-07-11 |
| Purpose | Connect problem, evidence, decision, implementation, experiment, and result |

## How to Read This Page

Each claim is a project claim, not a paper claim. Linked paper pages state the
conditions actually evaluated by their authors. ADRs record project decisions.
Experiments record project results. A row marked `Blocked` or `Research` must not
be described as implemented evidence.

| ID | Problem or project claim | Paper evidence | Decision | Implementation | Experiment and result | Status |
|---|---|---|---|---|---|---|
| C-001 | Sparse 1 Hz RGB needs pose or depth guidance for stable low-overlap geometry. | [G-CUT3R](papers/g-cut3r.md), [Mono-Hydra++](papers/mono-hydra-plus-plus.md) | [ADR-0002](decisions/adr-0002-gcut3r-as-teacher.md) | `src/worldmm_smvqa/worldmm/gcut3r_teacher.py` defines only the provider/cache contract. | [EXP-0004](experiments/exp-0004-gcut3r-provider.md): not run. | Blocked |
| C-002 | Lifelong memory should persist bounded explicit typed records rather than dense point maps or recurrent-state snapshots. | [Point3R](papers/point3r.md), [ConceptGraphs](papers/conceptgraphs.md), [HOV-SG](papers/hov-sg.md), [FARM](papers/farm.md) | [ADR-0001](decisions/adr-0001-explicit-typed-memory.md) | `src/worldmm_smvqa/worldmm/typed_memory.py`, `src/worldmm_smvqa/worldmm/spatial.py` | [EXP-0002](experiments/exp-0002-typed-memory-bridge.md): checkpoint bridge not run. | Partially implemented |
| C-003 | Geometry answers require deterministic operations and proof-to-answer agreement. | [GraphEQA](papers/grapheqa.md), [OpenEQA](papers/openeqa.md), [SQA3D](papers/sqa3d.md) motivate explicit spatial reasoning but do not directly prove this implementation. | [ADR-0004](decisions/adr-0004-deterministic-geometry-proof.md) | `src/worldmm_smvqa/worldmm/geometry_executor.py`, `src/worldmm_smvqa/qa.py` | [EXP-0001](experiments/exp-0001-source-compact-baseline.md): tiny distance proof and causal QA path passed. | Locally verified |
| C-004 | Unknown future questions require a stable geometry core plus a small bounded evidence reservoir. | [GraphEQA](papers/grapheqa.md), [FOUND-IT](papers/found-it.md), [Worth Remembering](papers/surprise-gated-robot-episodic-memory.md) | No accepted reservoir ADR. | Surprise features exist in selector preparation; no persistent visual reservoir exists. | [EXP-0003](experiments/exp-0003-byte-pareto.md): reservoir variant not run. | Research |
| C-005 | Writer decisions and comparisons must use actual serialized bytes, not token count alone. | [LONG3R](papers/long3r.md), [MeMix](papers/memix.md), and [rate-distortion](papers/end-to-end-optimized-image-compression.md) provide related but not equivalent evidence. | [ADR-0003](decisions/adr-0003-value-per-byte-writer.md) | Compact and typed writers measure canonical JSONL bytes. | [EXP-0001](experiments/exp-0001-source-compact-baseline.md): local sanity passed. [EXP-0003](experiments/exp-0003-byte-pareto.md): not run. | Local contract verified; benchmark pending |
| C-006 | Fast pose/tracking state must remain separate from persistent map memory. | [Mem3R](papers/mem3r.md), [Hydra](papers/hydra.md), [Mono-Hydra++](papers/mono-hydra-plus-plus.md) | [ADR-0001](decisions/adr-0001-explicit-typed-memory.md), [ADR-0002](decisions/adr-0002-gcut3r-as-teacher.md) | Provider state is transient; typed records are persistent. No raw pose student exists. | [EXP-0004](experiments/exp-0004-gcut3r-provider.md): not run. | Partially specified |
| C-007 | Wide-baseline association should combine position, viewing ray, semantics, time, and geometry compatibility. | [Ray-Aware Pointer Memory](papers/ray-aware-pointer-memory.md), [Point3R](papers/point3r.md) | No accepted learned-association ADR. | Current baseline uses explicit IDs or one-to-one heuristic spatial association. | No open-world learned-association experiment. | Research |
| C-008 | Long-horizon evaluation must use one causal frame inventory and measure memory across disjoint evidence moments. | [SuperMemory-VQA](papers/supermemory-vqa.md), [LongSpace](papers/longspace.md), [MV-ScanQA](papers/mv-scanqa.md) | Covered by the evaluation contract, not a standalone ADR. | `src/worldmm_smvqa/sensor_frames.py`, `src/worldmm_smvqa/preflight.py`, causal retrieval. | [EXP-0003](experiments/exp-0003-byte-pareto.md): official run not started. | Local contract verified |
| C-009 | Causal scope, provenance, uncertainty, and completeness are answer correctness requirements. | [SuperMemory-VQA](papers/supermemory-vqa.md), [OpenEQA](papers/openeqa.md) provide benchmark context; the strict proof boundary is a project decision. | [ADR-0004](decisions/adr-0004-deterministic-geometry-proof.md) | Preflight, evidence-pack validation, proof hashes, complete-index gates, and QA resume manifests. | [EXP-0001](experiments/exp-0001-source-compact-baseline.md): zero tiny-fixture causal violations. | Locally verified |
| C-010 | Fixed latent bottlenecks are useful decoder baselines but cannot replace explicit geometry without proof of geometry sufficiency. | [TokenLearner](papers/tokenlearner.md), [Perceiver](papers/perceiver.md), [BLIP-2](papers/blip-2.md) | No accepted latent-memory ADR. | `src/worldmm_smvqa/spatial_train.py` is a candidate head over supplied vectors, not a raw encoder. | No equal-byte typed-versus-latent experiment. | Deferred |
| C-011 | Geometry-aware novelty and duplicate removal are stronger baselines than generic attention-only pruning. | [Good Token Hunting](papers/good-token-hunting.md), [DART](papers/dart.md), [FEATHER](papers/feather.md), [Geometry-Aware Token Pruning](papers/geometry-aware-token-pruning.md) | No separate ADR; treated as writer baselines. | Geometry novelty and redundancy features exist in selector preparation. | [EXP-0003](experiments/exp-0003-byte-pareto.md): not run. | Planned comparison |
| C-012 | Quantizing model or latent values is a later deployment optimization, not the primary persistent-memory method. | [FSQ](papers/finite-scalar-quantization.md), [VQ-VAE](papers/vq-vae.md), [QVGGT](papers/qvggt.md) | Explicit-first direction recorded in [ADR-0001](decisions/adr-0001-explicit-typed-memory.md). | No learned codec or model PTQ path is deployed. | No project experiment. | Deferred |

## Evidence Labels

| Label | Meaning |
|---|---|
| Paper-reported | Verified in a primary paper under its stated dataset and condition |
| Project inference | Design transfer from paper conditions to this problem; still a hypothesis |
| Locally verified | Tiny fixture, contract, or unit behavior only |
| Benchmark verified | Reproduced under pinned official data, model, configuration, and run provenance |
| Blocked | Required implementation or approved compute artifact does not exist |

## Update Rule

- New research evidence updates a paper page and the relevant row.
- A changed design creates or supersedes an ADR.
- A measured result updates only its experiment page, then changes this row's
  status and result link.
- Architecture pages do not store transient metric values.
- Current blockers are maintained in [Current Status](status.md).

[Back to project home](README.md)
