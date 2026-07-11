# Spatial Memory for SuperMemory-VQA

| Field | Value |
|---|---|
| Page ID | SM-ROOT |
| Status | Active local preparation |
| Last updated | 2026-07-11 |
| Target | Explicit compressed spatial memory for geometry-grounded QA |
| Current verdict | Heuristic baseline works locally; learned G-CUT3R path is not end-to-end |

## Five-Minute Summary

AI glasses observe sparse RGB frames, approximately 1 Hz, but must answer later
questions about metric position, direction, distance, containment, reachability,
and object history. Keeping frames, dense point maps, or recurrent states for
months exceeds the device budget and does not provide directly auditable QA
evidence.

The project hypothesis is:

> Use dense geometry only as transient reasoning, then persist the smallest
> typed records sufficient for future geometry-grounded QA.

The intended memory contains explicit objects, planes, portals, coarse free
space, relocalization landmarks, meaningful change events, uncertainty, and
provenance. A deterministic geometry executor computes spatial answers; the
language model explains or selects the result rather than inventing geometry.

## System Shape

```text
1 Hz RGB + native-rate IMU/VIO + optional depth
    -> guided transient geometry state
    -> typed record candidates
    -> QA utility / geometry novelty / uncertainty / actual-byte writer
    -> explicit persistent spatial memory
    -> causal retrieval
    -> deterministic geometry proof
    -> four-choice QA
```

## Current State

- Source-compact explicit memory, causal retrieval, actual-byte limits, and
  deterministic geometry proofs run on the tiny local fixture.
- Typed record schemas, teacher-cache contracts, DDP candidate-head training,
  and a hard typed-record writer exist.
- No repository-owned G-CUT3R extractor, raw RGB/IMU student encoder,
  checkpoint-to-record inference decoder, or open-world learned association
  exists yet.
- No real dataset, model download, training, benchmark evaluation, SSH session,
  or Slurm job has run on this development host.

See [Current Status](status.md) for the live implementation verdict and
[Experiments](experiments/README.md) for results.

## Page Tree

- [Source and provenance](source/README.md)
- [Problem and research questions](problem.md)
- [Architecture](architecture.md)
- [Evidence and implementation traceability](traceability.md)
- [Research roadmap](roadmap.md)
- [Current status](status.md)
- [Paper index](papers/README.md)
- [Architecture decisions](decisions/README.md)
- [Experiments and results](experiments/README.md)
- [Dated reviews](reviews/README.md)

## Reading Paths

For the research idea:

```text
Problem -> Traceability -> Papers -> Decisions -> Architecture
```

For implementation readiness:

```text
Status -> Experiments -> Architecture -> Company handoff
```

For a new paper:

```text
Paper page -> supported claim -> decision -> experiment -> result
```

## Confluence Import Rules

- Import this page as the parent page and preserve the directory hierarchy.
- Use the `Page ID` metadata value as the stable migration key.
- Keep page titles unique within this tree.
- Convert relative Markdown links only after every page has been created.
- Keep repository code references as inline paths, or convert them to
  commit-pinned source-control URLs after the target repository URL is known.
- Preserve completed experiment and dated review pages as immutable records.
- Keep living pages limited to this page, `problem.md`, `architecture.md`,
  `traceability.md`, `roadmap.md`, and `status.md`.

[Back to documentation index](../README.md)
