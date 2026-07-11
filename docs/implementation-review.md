# SuperMemory-VQA Spatial Memory Implementation Review

> Historical 2026-07-11 review source. The immutable review page is
> [spatial-memory/reviews/2026-07-11-local-readiness.md](spatial-memory/reviews/2026-07-11-local-readiness.md),
> and current status lives in [spatial-memory/status.md](spatial-memory/status.md).

## Verdict

Current repository is a strong local-preparation scaffold plus a working
heuristic explicit-memory baseline. It is not yet an end-to-end reproduction of
the learned G-CUT3R-derived method.

The source-compact baseline runs:

```text
prepared source streams
  -> causal 1 Hz inventory
  -> explicit object/relation/zone/trajectory tokens
  -> actual-byte per-window writer
  -> causal retrieval
  -> deterministic geometry proofs
  -> four-choice QA and 0-100 benchmark metrics
```

The learned lane currently stops at a checkpoint:

```text
external teacher cache + external supervision
  -> materialized rows
  -> DDP typed candidate head
  -> spatial_student.pt
  -X-> typed inference / association / persistent evidence
```

## Implemented And Locally Verified

- Prepared-dataset preflight checks IDs, semantic question/label parity,
  four-choice/N/A contract, timebase, causal ranges, evidence scope, 1 Hz frame
  selection, optional frame files, and spatial sensor coverage.
- Object identity, change type, previous-state validity closure, latest-state
  selection, one-to-one heuristic association, coordinate-frame preservation,
  and quantization uncertainty are explicit.
- Compact records enforce token and actual serialized-byte limits per causal
  window. Typed records also have a total actual-byte hard writer.
- Flat typed object, plane, portal, free-space, landmark, and event artifacts
  can enter retrieval. `no_write` records cannot enter persistent memory.
- Geometry execution supports distance, near, relative direction, last-seen,
  and count. Count and last-seen abstain without a complete-index certificate.
- Proofs retain subject/object roles, frame, uncertainty, provenance, evidence,
  and all query parameters in a stable proof hash.
- QA rejects unknown, missing, duplicate, off-scope, or future evidence packs.
  Spatial snippets and geometry dictionaries are removed from model prompts.
- QA resume artifacts are bound to evidence, question, source, backend, model,
  and prompt/schema hashes.
- Evaluation implements four-way QA-Acc and QA-MRR plus answerability F1 on the
  paper's 0-100 scale. CLI evaluation rejects causal evidence violations.
- Counterfactual selector rows require explicit utility and split manifests;
  lexical evidence-overlap supervision is legacy-only.
- DDP losses and validation use global numerators/denominators. Checkpointing is
  atomic and configuration/cache-bound.
- Generated Slurm DAG separates CPU/GPU work, uses 10 nodes x 8 GPUs by default,
  enforces explicit approval and run-scoped output, passes one 1 Hz manifest
  through ingest and teacher stages, and writes rank-specific teacher shards.

## P0 Blockers Before Final Reproduction

1. No installed or repository-owned G-CUT3R extractor. The current adapter and
   cache validator are contracts only.
2. No raw RGB/IMU/VIO student encoder. Current PyTorch model is a candidate head
   over externally supplied vectors.
3. No type-specific checkpoint inference decoder for variable geometry such as
   plane/free-space polygons.
4. No learned open-world pointer association. Current student association is a
   closed-set class head.
5. No internal checkpoint-to-typed-memory path. The preferred DAG still accepts
   externally produced student evidence, so checkpoint changes need not change
   QA metrics.
6. QA-aware selector and typed student are separate candidate spaces. Deletion
   utility does not supervise the deployed typed write gate.
7. No repository-owned counterfactual deletion job that produces the utility
   cache.
8. Preferred typed DAG runs the main lane only. Spatial/protocol ablations,
   byte Pareto curves, diagnostics, and final report still belong to the legacy
   lane or operator steps.

## P1 Research Gaps

- Long-term source-compact memory is bounded per window, not by global/submap
  current-state capacity. Repeated unchanged observations can still grow over
  time to preserve historical benchmark queries.
- Multi-evidence retrieval keeps a one-clip-per-video default. Questions needing
  several disjoint moments require configurable diverse clip selection.
- Query-time intervals such as "during placement" are not parsed into validity
  constraints.
- Allocentric and egocentric direction need separate operators and wearer-pose
  binding; missing yaw correctly causes abstention today.
- Submap graph optimization, loop closure, ray-aware landmark replacement, and
  surprise evidence reservoir are schemas/design targets, not runtime systems.
- External geometry and association supervision remain trusted inputs; teacher
  record-derived type-specific target encoding is still required.
- Official raw-data ingest is outside this repository. Prepared files must have
  their split, time normalization, and dataset digest pinned before evaluation.

## Company-Side Go/No-Go Gates

Do not report final benchmark numbers until all gates pass:

1. Prepared dataset preflight has zero errors; warnings are reviewed.
2. Official evaluation split and source/question/label digests are recorded.
3. G-CUT3R provider provenance, checkpoint, 1 Hz manifest, and causal cache
   digests validate.
4. Student checkpoint produces flat typed records internally; no external
   evidence symlink bypass remains.
5. Typed artifact file size satisfies the configured hard byte cap after actual
   serialization.
6. Every geometry answer cites a matching deterministic proof; count uses a
   completeness certificate.
7. Main, without-spatial, and retrieval-protocol lanes share model, frames,
   split, and budgets.
8. Causal violation count is zero and QA resume manifests match current inputs.
9. Report includes bytes/hour, bytes/new area, bytes/object, bytes/change,
   repeated-visit growth, and QA-vs-bytes Pareto curves.

No real training, model download, benchmark evaluation, SSH session, or Slurm
submission was performed during this local review.
