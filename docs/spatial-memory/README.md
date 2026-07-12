# Spatial Memory for SuperMemory-VQA

| Field | Value |
|---|---|
| Page ID | SM-ROOT |
| Confluence parent | SM-DOCS |
| Status | Active local preparation |
| Last updated | 2026-07-12 |
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
- [Operations](operations/README.md): imports repository `HANDOFF.md` as child
  Page ID `SM-OPERATIONS-HANDOFF`

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
- Import [Operations](operations/README.md) as `SM-OPERATIONS`, then import
  repository `HANDOFF.md` as its `SM-OPERATIONS-HANDOFF` child; do not duplicate
  the runbook in this tree.
- Use the `Page ID` metadata value as the stable migration key.
- Keep page titles unique within this tree.
- Convert relative Markdown links only after every page has been created.
- Keep repository code references as inline paths, or convert them to
  commit-pinned source-control URLs after the target repository URL is known.
- Preserve completed experiment and dated review pages as immutable records.
- Keep living pages limited to this page, `problem.md`, `architecture.md`,
  `traceability.md`, `roadmap.md`, and `status.md`.

## Confluence Import Manifest

This table is the canonical import scope, stable Page ID resolver, and parent
map. A patterned row resolves each page's own ID from its `Page ID` metadata;
its parent resolves to the fixed ID in this table. This replaces per-file parent
metadata where a whole directory has one parent.

| Source scope | Imported Page ID | Confluence parent ID | Import rule |
|---|---|---|---|
| `docs/README.md` | `SM-DOCS` | `SPACE-HOME` | Resolve the external sentinel to the configured space landing page |
| `docs/spatial-memory/README.md` | `SM-ROOT` | `SM-DOCS` | Import project home |
| `docs/spatial-memory/{problem,architecture,traceability,roadmap,status}.md` | Page metadata | `SM-ROOT` | Import living pages directly |
| `docs/spatial-memory/source/README.md` | `SM-SOURCE` | `SM-ROOT` | Import source index |
| Non-template `docs/spatial-memory/source/*.md` | Page metadata | `SM-SOURCE` | Import as source-index children |
| `docs/spatial-memory/papers/README.md` | `SM-PAPERS` | `SM-ROOT` | Import paper index |
| Non-template `docs/spatial-memory/papers/*.md` | Page metadata | `SM-PAPERS` | Import as paper-index children |
| `docs/spatial-memory/decisions/README.md` | `SM-DECISIONS` | `SM-ROOT` | Import decision index |
| Non-template `docs/spatial-memory/decisions/*.md` | Page metadata | `SM-DECISIONS` | Import as decision-index children |
| `docs/spatial-memory/experiments/README.md` | `SM-EXPERIMENTS` | `SM-ROOT` | Import experiment index |
| Non-template `docs/spatial-memory/experiments/*.md` | Page metadata | `SM-EXPERIMENTS` | Import as experiment-index children |
| `docs/spatial-memory/reviews/README.md` | `SM-REVIEWS` | `SM-ROOT` | Import review index |
| Non-template `docs/spatial-memory/reviews/*.md` | Page metadata | `SM-REVIEWS` | Import as review-index children |
| `docs/spatial-memory/operations/README.md` | `SM-OPERATIONS` | `SM-ROOT` | Import operations parent |
| Repository `HANDOFF.md` | `SM-OPERATIONS-HANDOFF` | `SM-OPERATIONS` | Import the canonical runbook as the operations child |
| Any `TEMPLATE.md` | Excluded | Excluded | Repository authoring template, not a Confluence page |
| `docs/implementation-review.md`, `docs/spatial-token-compression.md`, `docs/spatial-token-research-roadmap.md` | Excluded | Excluded | Legacy migration sources; canonical content is already in this tree |

All relative page links in an imported page must resolve to another imported
page. Refer to excluded templates and legacy sources as inline repository paths,
not Markdown page links.

[Back to documentation index](../README.md)
