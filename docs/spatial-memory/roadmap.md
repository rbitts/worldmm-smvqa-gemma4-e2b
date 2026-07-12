# Research Roadmap

| Field | Value |
|---|---|
| Page ID | SM-ROADMAP |
| Status | Active |
| Last updated | 2026-07-12 |
| Ordering principle | Close end-to-end correctness gaps before adding codecs |

## P0: Close the Learned End-to-End Path

Exercise the implemented production bridge as one causal path:

```text
teacher extraction
    -> record-derived supervision
    -> DDP student training
    -> WORLDMM_SPATIAL_INFER_EXE (worldmm-spatial-infer-v1)
    -> typed decode, open-world association, actual-byte writer
    -> repository artifact and lineage validation
    -> retrieval
    -> real-frame proof-grounded QA
    -> immutable profile-bound PROBE or learned-E1 identity and report
```

Required outcomes:

- install or provide a pinned G-CUT3R-compatible extractor;
- derive geometry targets from teacher records rather than trusting arbitrary
  external vectors;
- provide a pinned production inference executable that decodes every supported
  type, including variable polygons, and supports existing-pointer or `NEW`
  association in unseen scenes;
- emit canonical typed JSONL grouped into 30-second windows under the approved
  per-window byte budget;
- prove checkpoint, typed-memory, inference-manifest, evidence, config, data,
  sensor, model, frame, prompt, prediction, and metric lineage;
- pass the approved 1-node x 1-GPU probe before requesting full scale.

The probe's valid claim is `contract_probe` / `PROBE`; it is not renamed to E1
after success. A new approved `full` run is required for `student` / `E1`.

Primary experiments: [EXP-0002](experiments/exp-0002-typed-memory-bridge.md) and
[EXP-0004](experiments/exp-0004-gcut3r-provider.md).

## P1: Measure the Explicit Baseline

After P0, run learned E1 with a fixed causal frame inventory and report:

- QA-Acc, QA-MRR, and answerability F1;
- geometry proof and grounding accuracy;
- bytes per hour, area, object, and event;
- repeated-visit memory growth;
- peak memory, latency, and energy where available;
- actual-byte Pareto curves.

Primary experiment: [EXP-0003](experiments/exp-0003-byte-pareto.md).

Do not label E1 official yet. Add immutable E2 spatial-ablation and E3 retrieval-
protocol-ablation identities, then validate that every non-ablated split, model,
frame, prompt, and checkpoint-derived input digest matches. Official E1/E2/E3
completion remains blocked until this exists.

## P2: Connect QA Utility to the Deployed Writer

- Generate counterfactual deletion utility from a detailed offline reference
  memory.
- Protect a query-agnostic geometry core.
- Train the same candidate gate that is used during typed inference.
- Compare geometry novelty, QA utility, uncertainty, pose information, and event
  surprise under equal byte budgets.

Do not build a new selector abstraction if the existing typed write logit can be
supervised directly.

## P3: Long-Term Identity and Relocalization

- replace closed-set association with a causal pointer or metric key;
- add position and viewing-ray compatibility;
- measure identity across days, lighting changes, and moved objects;
- add submap loop correction and landmark retain-or-replace policy;
- measure false merge, duplicate identity, and loop-closure precision/recall.

## P4: Unknown-Question Safety

- keep structural topology, object extent, validity, uncertainty, and provenance
  independent of training question templates;
- add a small bounded evidence reservoir for text, appearance, surprise, and
  detector uncertainty;
- hold out operators and object categories during writer training;
- evaluate abstention on unavailable or inferred-only evidence.

## P5: On-Device Student

Only after P0-P4 produce a stable record contract:

- distill the geometry candidate encoder;
- benchmark one-node server inference before multi-node training;
- profile realistic device latency and energy;
- calibrate pose, camera, depth, and uncertainty on hardware;
- retain an offline consolidation path for expensive corrections.

## Deferred Until Evidence Exists

- VQ or FSQ codec for generic latent memory;
- dense neural scene storage;
- custom ANN infrastructure;
- end-to-end joint training with the QA language model;
- photorealistic reconstruction optimization.

These additions become justified only when the explicit baseline is the measured
bottleneck under equal actual-byte budgets.

## Go/No-Go Order

1. P0 produces checkpoint-backed typed evidence, loads real frames, and closes
   the generated profile-bound run lineage.
2. P1 establishes a reproducible learned-E1 baseline and byte curve, then adds
   matched E2/E3 identities before any official claim.
3. P2 demonstrates utility beyond geometry novelty.
4. P3 maintains identity and localization across repeated visits.
5. P4 preserves unknown-question coverage.
6. P5 measures device feasibility.

[Back to project home](README.md)

Operational execution and approval details live in repository `HANDOFF.md`,
imported under the [Operations](operations/README.md) parent in Confluence.
