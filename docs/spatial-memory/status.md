# Current Status

| Field | Value |
|---|---|
| Page ID | SM-STATUS |
| Status date | 2026-07-12 |
| Local preparation | Ready |
| Heuristic explicit baseline | Working on tiny fixture |
| Learned typed handoff | Prepared; external production inference required |
| Real training or evaluation | Not run |

## Verdict

The repository now prepares a fail-closed, checkpoint-to-E1 production path, but
has not executed it. The learned boundary is an operator-supplied executable;
the repository does not contain a production G-CUT3R encoder, type-specific
decoder, or open-world association model.

```text
external teacher cache + external supervision
    -> materialized rows
    -> DDP typed candidate head
    -> spatial_student.pt
    -> WORLDMM_SPATIAL_INFER_EXE (worldmm-spatial-infer-v1)
    -> canonical typed JSONL + production manifest
    -> validated student evidence lineage
    -> real-frame Gemma QA -> profile-bound PROBE or E1 identity/report
```

## Implemented

- prepared-dataset preflight and causal at-most-1-Hz frame selection;
- explicit object identity, one-to-one observation-time association, change
  states, validity closure, coordinate frames, and uncertainty;
- per-window compact memory and typed-memory writers using actual serialized
  bytes;
- object, plane, portal, free-space, landmark, event, and no-write schemas;
- flat typed-record retrieval and deterministic geometry proofs;
- proof-to-choice verification and strict evidence-pack validation;
- four-choice evaluation on a 0-100 scale with causal diagnostics;
- globally normalized DDP losses, validation, atomic checkpoints, and resume;
- phased preflight/approval plus seven-stage Slurm DAG;
- sanitized production inference input, per-30-second canonical JSONL budget
  validation, frame/model content and filename-inventory fingerprints,
  video-namespaced frame audits, QA v4 completion marker, and
  checkpoint-to-evidence lineage that also binds the memory manifest and exact
  episodic/semantic/visual store bytes;
- generated profile-bound run identity, remote manifest, and final report
  contract with profile-neutral filenames and a finalization input seal
  covering QA, lineage, typed/non-spatial memory artifacts, config, sensor, and
  split inputs (`probe` = `contract_probe`/`PROBE`; `full` = `student`/`E1`).

## P0 Blockers

1. No repository-owned G-CUT3R extractor or raw RGB/IMU/VIO student encoder.
2. No repository-owned type-specific inference decoder or learned open-world
   pointer association; production execution requires
   `WORLDMM_SPATIAL_INFER_EXE`.
3. The external executable and its `worldmm-spatial-infer-v1` outputs have not
   been exercised against company checkpoints, frames, or data.

## Official Reporting Gap

The typed DAG produces a bounded contract probe or learned E1 only. Immutable
matched E2/E3 identities and official completion remain blocked. Byte-Pareto
reporting is a later measurement milestone, not part of the P0 execution gate.

## P2 Research Gap

Counterfactual QA utility is not connected to the deployed typed write gate.
The utility/split preparation path is a separate experiment, not a P0 production
input or a substitute for checkpoint-backed inference.

## Local Verification

| Check | Result |
|---|---|
| Test suite | Run live before transfer; no benchmark execution |
| Ruff and basedpyright | Run live before transfer |
| Generated shell syntax | Run live before transfer |
| Diff whitespace check | Run live before transfer |
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

Run the approved 1-node x 1-GPU contract probe first. It must prove that the
external inference executable produces non-empty canonical typed records within
the 4,096-byte 30-second-window budget, that QA loads real frames, and that the
generated run identity closes every checkpoint/model/frame/evidence digest. Do
not scale or claim official results before that probe passes. A successful probe
is recorded as `result_class=contract_probe`, experiment `PROBE`; only a newly
approved `full` profile may produce `result_class=student`, experiment `E1`.

Operational source of truth: repository `HANDOFF.md`, imported under the
[Operations](operations/README.md) parent in Confluence.

[Back to project home](README.md)
