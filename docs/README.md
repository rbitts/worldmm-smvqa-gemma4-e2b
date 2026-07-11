# Documentation

This directory contains the repository's canonical design and research pages.
Each Markdown file is intended to map to one Confluence page later.

## Projects

- [Spatial Memory for SuperMemory-VQA](spatial-memory/README.md)

## Operational Documents

- [Repository quick start](../README.md)
- [Company-compute handoff](../HANDOFF.md)

## Legacy Documents

The following pages are retained as migration sources. New research and design
updates belong under `docs/spatial-memory/`.

- [Legacy spatial-token architecture](spatial-token-compression.md)
- [Legacy research roadmap](spatial-token-research-roadmap.md)
- [2026-07-11 implementation review](implementation-review.md)

## Markdown Contract

- One file maps to one Confluence page.
- Every file has exactly one level-one heading.
- Metadata uses ordinary Markdown tables, not YAML front matter.
- Repository links are relative.
- Diagrams use plain text code blocks; Mermaid and GitHub-only macros are not
  canonical.
- PDFs, model files, datasets, and large run artifacts are not committed.
- Paper-reported evidence, project inference, and project results remain
  separate.
