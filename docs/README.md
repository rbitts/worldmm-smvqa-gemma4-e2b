# Documentation

| Field | Value |
|---|---|
| Page ID | SM-DOCS |
| Confluence parent | SPACE-HOME |
| Page role | Repository documentation index |
| Status | Active |

This directory contains the repository's canonical design and research pages.
Canonical import scope is defined by the Spatial Memory import manifest; legacy
sources and authoring templates are excluded.

`SPACE-HOME` is the only external parent sentinel. Resolve it to the selected
Confluence space landing page at import time; every other parent is a stable
`SM-*` Page ID defined inside the canonical import set.

## Projects

- [Spatial Memory for SuperMemory-VQA](spatial-memory/README.md)

## Operational Documents

- Repository-only quick start: `README.md` at repository root (not imported)
- [Company-compute handoff](../HANDOFF.md)

## Legacy Documents

The following pages are retained as migration sources. New research and design
updates belong under `docs/spatial-memory/`.

- `docs/spatial-token-compression.md`
- `docs/spatial-token-research-roadmap.md`
- `docs/implementation-review.md` (2026-07-11 snapshot)

## Markdown Contract

- Every file in the canonical import manifest maps to one Confluence page.
- Every file has exactly one level-one heading.
- Metadata uses ordinary Markdown tables, not YAML front matter.
- Repository links are relative.
- Diagrams use plain text code blocks; Mermaid and GitHub-only macros are not
  canonical.
- PDFs, model files, datasets, and large run artifacts are not committed.
- Paper-reported evidence, project inference, and project results remain
  separate.
