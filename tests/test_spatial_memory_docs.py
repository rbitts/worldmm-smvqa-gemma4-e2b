from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).parents[1]
DOC_ROOT = ROOT / "docs" / "spatial-memory"
LINK_RE: re.Pattern[str] = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
ID_RE: re.Pattern[str] = re.compile(
    r"^\| Page ID \| ([^|]+?) \|$",
    re.MULTILINE,
)


def test_spatial_memory_markdown_is_confluence_importable() -> None:
    page_ids: dict[str, Path] = {}
    page_titles: dict[str, Path] = {}

    for directory in (path for path in DOC_ROOT.rglob("*") if path.is_dir()):
        if any(directory.glob("*.md")):
            assert (directory / "README.md").is_file(), (
                f"{directory}: Confluence parent README is missing"
            )

    for path in sorted(DOC_ROOT.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        h1s = [line for line in text.splitlines() if line.startswith("# ")]
        assert len(h1s) == 1, f"{path}: expected one H1, found {len(h1s)}"
        assert not text.startswith("---\n"), f"{path}: YAML front matter is forbidden"
        assert "```mermaid" not in text.lower(), f"{path}: Mermaid is not canonical"

        if path.name != "TEMPLATE.md":
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

        for link_match in LINK_RE.finditer(text):
            raw_target = link_match.group(1)
            target = raw_target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = unquote(target.split("#", 1)[0])
            if not relative:
                continue
            resolved = (path.parent / relative).resolve()
            assert resolved.exists(), f"{path}: missing relative link {target}"
            is_document_page = resolved.is_relative_to(DOC_ROOT) or resolved == (
                ROOT / "docs/README.md"
            )
            assert is_document_page, (
                f"{path}: Confluence page link escapes canonical tree: {target}"
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
