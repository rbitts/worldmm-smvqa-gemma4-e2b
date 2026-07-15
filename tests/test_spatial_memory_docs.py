from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).parents[1]
DOC_ROOT = ROOT / "docs" / "spatial-memory"
DOCS_INDEX = ROOT / "docs" / "README.md"
HANDOFF = ROOT / "HANDOFF.md"
LINK_RE: re.Pattern[str] = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
ID_RE: re.Pattern[str] = re.compile(
    r"^\| Page ID \| ([^|]+?) \|$",
    re.MULTILINE,
)
PARENT_RE: re.Pattern[str] = re.compile(
    r"^\| Confluence parent \| ([^|]+?) \|$",
    re.MULTILINE,
)

SECTION_PARENT_IDS = {
    "decisions": "SM-DECISIONS",
    "experiments": "SM-EXPERIMENTS",
    "operations": "SM-OPERATIONS",
    "papers": "SM-PAPERS",
    "reviews": "SM-REVIEWS",
    "source": "SM-SOURCE",
}


def _canonical_pages() -> tuple[Path, ...]:
    tree_pages = (path for path in DOC_ROOT.rglob("*.md") if path.name != "TEMPLATE.md")
    return tuple(sorted((DOCS_INDEX, HANDOFF, *tree_pages)))


def _manifest_parent_id(path: Path) -> str:
    if path == DOCS_INDEX:
        return "SPACE-HOME"
    if path == HANDOFF:
        return "SM-OPERATIONS"
    relative = path.relative_to(DOC_ROOT)
    if relative == Path("README.md"):
        return "SM-DOCS"
    if len(relative.parts) == 1 or relative.name == "README.md":
        return "SM-ROOT"
    parent_id = SECTION_PARENT_IDS.get(relative.parts[0])
    assert parent_id is not None, f"{path}: canonical import scope has no parent rule"
    return parent_id


def _prose_text(text: str) -> str:
    prose: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            prose.append(line)
    assert not in_fence, "unclosed Markdown code fence"
    return "\n".join(prose)


def test_spatial_memory_markdown_is_confluence_importable() -> None:
    page_ids: dict[str, Path] = {}
    page_titles: dict[str, Path] = {}
    canonical_pages = _canonical_pages()
    canonical_targets = {path.resolve() for path in canonical_pages}

    for directory in (path for path in DOC_ROOT.rglob("*") if path.is_dir()):
        if any(directory.glob("*.md")):
            assert (directory / "README.md").is_file(), (
                f"{directory}: Confluence parent README is missing"
            )

    for path in canonical_pages:
        text = path.read_text(encoding="utf-8")
        prose = _prose_text(text)
        h1s = [line for line in prose.splitlines() if line.startswith("# ")]
        assert len(h1s) == 1, f"{path}: expected one H1, found {len(h1s)}"
        assert not text.startswith("---\n"), f"{path}: YAML front matter is forbidden"
        assert "```mermaid" not in text.lower(), f"{path}: Mermaid is not canonical"

        title = h1s[0][2:].strip()
        assert title not in page_titles, (
            f"duplicate page title {title}: {page_titles[title]} and {path}"
        )
        page_titles[title] = path
        match = ID_RE.search(text)
        assert match is not None, f"{path}: stable page ID is missing"
        page_id = match.group(1).strip("` ")
        assert page_id.startswith("SM-"), f"{path}: page ID must start with SM-"
        assert page_id not in page_ids, (
            f"duplicate page ID {page_id}: {page_ids[page_id]} and {path}"
        )
        page_ids[page_id] = path

        expected_parent = _manifest_parent_id(path)
        explicit_parent = PARENT_RE.search(text)
        if explicit_parent is not None:
            assert explicit_parent.group(1).strip("` ") == expected_parent, (
                f"{path}: explicit parent disagrees with import manifest"
            )

        for link_match in LINK_RE.finditer(prose):
            raw_target = link_match.group(1)
            target = raw_target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = unquote(target.split("#", 1)[0])
            if not relative:
                continue
            resolved = (path.parent / relative).resolve()
            assert resolved.exists(), f"{path}: missing relative link {target}"
            assert resolved in canonical_targets, (
                f"{path}: link targets a page excluded from the import manifest: "
                f"{target}"
            )

    for path in canonical_pages:
        parent_id = _manifest_parent_id(path)
        if parent_id != "SPACE-HOME":
            assert parent_id in page_ids, (
                f"{path}: parent Page ID {parent_id} is absent from import scope"
            )

    assert page_ids["SM-DOCS"] == DOCS_INDEX
    assert page_ids["SM-OPERATIONS-HANDOFF"] == HANDOFF

    import_manifest = (DOC_ROOT / "README.md").read_text(encoding="utf-8")
    assert "| `docs/README.md` | `SM-DOCS` | `SPACE-HOME` |" in import_manifest
    assert (
        f"""| Repository `{HANDOFF.name}` | `SM-OPERATIONS-HANDOFF` | \
`SM-OPERATIONS` |"""
        in import_manifest
    )

    paper_root = DOC_ROOT / "papers"
    paper_index = (paper_root / "README.md").read_text(encoding="utf-8")
    paper_pages = sorted(
        path
        for path in paper_root.glob("*.md")
        if path.name not in {"README.md", "TEMPLATE.md"}
    )
    assert f"| Paper pages | {len(paper_pages)} |" in paper_index
    for path in paper_pages:
        assert f"({path.name})" in paper_index, f"{path}: missing from paper index"
        text = path.read_text(encoding="utf-8")
        assert "| Project claims | [Traceability](../traceability.md): C-" in text, (
            f"{path}: reverse claim mapping is missing"
        )


def test_research_and_handoff_documentation_contracts_are_explicit() -> None:
    traceability = (DOC_ROOT / "traceability.md").read_text(encoding="utf-8")
    for number in range(1, 7):
        rq_id = f"RQ-{number:03d}"
        assert re.search(
            f"""^\\| \\[{rq_id}:[^\\]]+\\]\\(problem\\.md#rq-\
{number:03d}-[^)]+\\) \\| C-""",
            traceability,
            re.MULTILINE,
        ), f"{rq_id}: claim mapping is missing"

    exp_0001 = (
        DOC_ROOT / "experiments" / "exp-0001-source-compact-baseline.md"
    ).read_text(encoding="utf-8")
    assert "**C-002 local result:**" in exp_0001
    assert "15.94x reduction" in exp_0001
    c_002_row = next(
        line for line in traceability.splitlines() if line.startswith("| C-002 |")
    )
    assert "EXP-0001" in c_002_row
    assert "15.94x smaller" in c_002_row

    paper_index = (DOC_ROOT / "papers" / "README.md").read_text(encoding="utf-8")
    assert "## Secondary Bibliography Policy" in paper_index
    assert "before citing it in an ADR" in paper_index

    handoff = HANDOFF.read_text(encoding="utf-8")
    env_requirement = (
        "The untracked company-side `$WORLDMM_REMOTE_REPO/.env.worldmm` is mandatory"
    )
    assert env_requirement in handoff
    assert "export WORLDMM_RUN_ID=REPLACE_WITH_APPROVED_RUN_ID" in handoff
    assert "WORLDMM_RUN_ID:-$(date" not in handoff
    assert "use the same unchanged file for both phases" in handoff
    assert "${BASTION_HOST:?set local ProxyJump host}" in handoff
    assert "${HEAD_NODE:?set local Slurm head host}" in handoff
    assert "spatial_infer_clean()" in handoff
    assert "-u WORLDMM_SPATIAL_INFER_EXE" in handoff
    assert (
        'export SMVQA_FRAME_ROOT="$WORLDMM_OUTPUT_ROOT/inference_inputs/frames"'
        in handoff
    )
    assert 'test -s "$WORLDMM_SENSOR_FRAME_MANIFEST"' in handoff
    assert 'test "$(stat -c %a "$env_file")" = 600' in handoff
    assert "env_contract.json.effective_teacher_resources" in handoff
    assert "These operational fields are not validated by comparing them to" in handoff


def test_traceability_contains_every_paper_and_experiment_claim_edge() -> None:
    traceability = (DOC_ROOT / "traceability.md").read_text(encoding="utf-8")
    claim_rows = {
        match.group(1): line
        for line in traceability.splitlines()
        if (match := re.match(r"^\| (C-\d{3}) \|", line)) is not None
    }

    paper_root = DOC_ROOT / "papers"
    for path in paper_root.glob("*.md"):
        if path.name in {"README.md", "TEMPLATE.md"}:
            continue
        text = path.read_text(encoding="utf-8")
        paper_claim_pattern = r"^\| Project claims \| \[Traceability\]\(\.\./traceability\.md\): ([^|]+) \|$"  # noqa: E501
        match = re.search(
            paper_claim_pattern,
            text,
            re.MULTILINE,
        )
        assert match is not None, f"{path}: project claim metadata missing"
        for claim_match in re.finditer(r"C-\d{3}", match.group(1)):
            claim_id = claim_match.group(0)
            assert f"(papers/{path.name})" in claim_rows[claim_id], (
                f"{path}: {claim_id} missing reverse traceability edge"
            )

    experiment_root = DOC_ROOT / "experiments"
    for path in experiment_root.glob("exp-*.md"):
        text = path.read_text(encoding="utf-8")
        for claim_match in re.finditer(
            r"^\| Claim \| \[(C-\d{3})[^\]]*\]",
            text,
            re.MULTILINE,
        ):
            claim_id = claim_match.group(1)
            assert f"(experiments/{path.name})" in claim_rows[claim_id], (
                f"{path}: {claim_id} missing reverse traceability edge"
            )
