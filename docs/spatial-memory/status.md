# Current Status

| Field | Value |
|---|---|
| Page ID | SM-STATUS |
| Status date | 2026-07-11 |
| Local preparation | Ready |
| Heuristic explicit baseline | Working on tiny fixture |
| Learned G-CUT3R reproduction | Blocked before typed inference |
| Real training or evaluation | Not run |

## Verdict

The repository is a strong local-preparation scaffold and a working heuristic
explicit-memory baseline. It is not an end-to-end reproduction of the learned
G-CUT3R-derived method.

```text
external teacher cache + external supervision
    -> materialized rows
    -> DDP typed candidate head
    -> spatial_student.pt
    -X-> typed inference / association / persistent evidence
```

## Implemented

- prepared-dataset preflight and causal at-most-1-Hz frame selection;
- explicit object identity, one-to-one observation-time association, change
  states, validity closure, coordinate frames, and uncertainty;
- per-window compact memory and total-budget typed-memory writers using actual
  serialized bytes;
- object, plane, portal, free-space, landmark, event, and no-write schemas;
- flat typed-record retrieval and deterministic geometry proofs;
- proof-to-choice verification and strict evidence-pack validation;
- four-choice evaluation on a 0-100 scale with causal diagnostics;
- globally normalized DDP losses, validation, atomic checkpoints, and resume;
- approval-gated 10-node by 8-GPU Slurm plan generation.

## P0 Blockers

1. No repository-owned G-CUT3R extractor.
2. No raw RGB/IMU/VIO student encoder.
3. No type-specific checkpoint inference decoder.
4. No learned open-world pointer association.
5. Student checkpoints do not yet produce QA evidence internally.
6. Counterfactual QA utility is not connected to the deployed typed write gate.
7. Utility-cache generation remains external.
8. The preferred typed DAG does not yet run all ablations and Pareto reporting.

## Local Verification

| Check | Result |
|---|---|
| Test suite | 342 passed, 1 intentionally skipped |
| Ruff lint | Passed |
| basedpyright | 0 errors, 0 warnings |
| Diff whitespace check | Passed |
| Tiny preflight | 0 errors, 4 expected coverage warnings |
| Tiny smoke causal violations | 0 |

Tiny mock results:

| Metric | With spatial | Without spatial |
|---|---:|---:|
| Ans-F1 | 100.00 | 100.00 |
| QA-Acc | 66.67 | 50.00 |
| QA-MRR | 83.33 | 72.22 |
| Relation F1 | 1.00 | Not applicable |

These values are synthetic fixture sanity checks, not benchmark results and not
evidence of production model quality. The source-compact spatial artifact was
6,050 bytes across multiple causal windows; its budget is per window, not a
lifelong global cap.

See [EXP-0001](experiments/exp-0001-source-compact-baseline.md) for the complete
contract.

## Remote State

- SSH sessions: none.
- Slurm submissions: none.
- Remote job IDs: none.
- Company artifacts: none.
- Dataset, model, and checkpoint copies to this host: none.

Only local dry-run plan generation was performed.

## Next Gate

Do not launch a large training run until [EXP-0004](experiments/exp-0004-gcut3r-provider.md)
can produce checkpoint-backed typed records without an external evidence
symlink.

[Back to project home](README.md)
