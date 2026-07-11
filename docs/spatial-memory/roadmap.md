# Research Roadmap

| Field | Value |
|---|---|
| Page ID | SM-ROADMAP |
| Status | Active |
| Last updated | 2026-07-11 |
| Ordering principle | Close end-to-end correctness gaps before adding codecs |

## P0: Close the Learned End-to-End Path

Deliver one causal path:

```text
teacher extraction
    -> record-derived supervision
    -> DDP student training
    -> checkpoint inference
    -> typed decode and open-world association
    -> actual-byte writer
    -> retrieval
    -> proof-grounded QA
```

Required outcomes:

- install or provide a pinned G-CUT3R-compatible extractor;
- derive geometry targets from teacher records rather than trusting arbitrary
  external vectors;
- decode every supported type, including variable polygons;
- support existing-pointer or `NEW` association in unseen scenes;
- bind evidence manifests to checkpoint, selector, data, and frame digests;
- remove the external-evidence bypass from the final learned lane.

Primary experiment: [EXP-0004](experiments/exp-0004-gcut3r-provider.md).

## P1: Measure the Explicit Baseline

Run the official split with fixed causal frame inventory and report:

- QA-Acc, QA-MRR, and answerability F1;
- geometry proof and grounding accuracy;
- bytes per hour, area, object, and event;
- repeated-visit memory growth;
- peak memory, latency, and energy where available;
- actual-byte Pareto curves.

Primary experiment: [EXP-0003](experiments/exp-0003-byte-pareto.md).

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

1. P0 produces checkpoint-backed typed evidence.
2. P1 establishes a reproducible baseline and byte curve.
3. P2 demonstrates utility beyond geometry novelty.
4. P3 maintains identity and localization across repeated visits.
5. P4 preserves unknown-question coverage.
6. P5 measures device feasibility.

[Back to project home](README.md)
