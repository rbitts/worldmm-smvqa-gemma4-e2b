from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_MANIFEST = "tests/fixtures/tiny_smvqa/remote_manifest.example.json"
REQUIRED_SECTIONS = (
    "## Local code/config changed",
    "## Remote command used",
    "## Remote job ID or process reference",
    "## Remote artifact path on company storage",
    "## Key metrics or failure reason",
    "## What was not copied locally",
)
EXPECTED_HEADINGS = ("# WorldMM-SMVQA Final Result Handoff", *REQUIRED_SECTIONS)


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["UV_NO_NETWORK"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        ["uv", "run", "--offline", "worldmm-smvqa", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_report_writes_required_handoff_sections_when_manifest_complete(
    tmp_path: Path,
) -> None:
    # Given: a complete remote manifest with pending remote execution status.
    report_path = tmp_path / "report.md"

    # When: report is generated through the CLI surface.
    result = run_cli(
        "report",
        "--run-manifest",
        FIXTURE_MANIFEST,
        "--out",
        str(report_path),
    )

    # Then: every AGENTS.md handoff section is present and no metric is claimed.
    assert result.returncode == 0, result.stderr
    report = report_path.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in report
    assert "remote_status: pending" in report
    assert "No remote metrics reported" in report
    assert "WorldMM-SMVQA" in report
    assert "Video-RAG exact reproduction" not in report


def test_report_fails_when_manifest_missing_remote_artifact_path(
    tmp_path: Path,
) -> None:
    # Given: untrusted manifest text omits the remote artifact path.
    manifest = tmp_path / "missing-artifact.json"
    _ = manifest.write_text(
        """{
  "baseline_name": "WorldMM-SMVQA",
  "remote_status": "pending",
  "local_changes": ["src/worldmm_smvqa/report.py"],
  "remote_command": "ssh \\"$BASTION_HOST\\"",
  "remote_job_reference": "not-run",
  "metrics": [],
  "failure_reason": "Remote benchmark not run.",
  "not_copied_locally": ["full datasets", "model weights", "checkpoints"]
}
""",
        encoding="utf-8",
    )

    # When: report generation parses the manifest at the boundary.
    result = run_cli(
        "report",
        "--run-manifest",
        str(manifest),
        "--out",
        str(tmp_path / "report.md"),
    )

    # Then: typed manifest validation fails before any report is written.
    assert result.returncode != 0
    assert "IncompleteRemoteManifest" in result.stderr
    assert not (tmp_path / "report.md").exists()


def test_report_escapes_manifest_markdown_when_values_spoof_structure(
    tmp_path: Path,
) -> None:
    # Given: untrusted manifest values try to create Markdown headings and links.
    manifest = tmp_path / "adversarial.json"
    report_path = tmp_path / "report.md"
    _ = manifest.write_text(
        json.dumps(
            {
                "baseline_name": "WorldMM-SMVQA",
                "remote_status": "failed",
                "local_changes": [
                    "## Remote command used",
                    "# fake heading",
                    "- fake bullet",
                    "```fence",
                    "[spoof](https://example.invalid)",
                ],
                "remote_command": "```bash\n## injected heading\n```",
                "remote_job_reference": "## Remote artifact path on company storage",
                "remote_artifact_path": "# fake artifact heading",
                "metrics": [{"name": "## Key metrics or failure reason", "value": 1.0}],
                "failure_reason": "- fake failure bullet",
                "not_copied_locally": ["[full dataset](https://example.invalid)"],
            },
        ),
        encoding="utf-8",
    )

    # When: report is generated through the CLI surface.
    result = run_cli(
        "report",
        "--run-manifest",
        str(manifest),
        "--out",
        str(report_path),
    )

    # Then: manifest content cannot add headings or spoof required sections.
    assert result.returncode == 0, result.stderr
    report = report_path.read_text(encoding="utf-8")
    headings = tuple(line for line in report.splitlines() if line.startswith("#"))
    assert headings == EXPECTED_HEADINGS
    assert r"\#\# Remote command used" in report
    assert r"\# fake heading" in report
    assert r"\- fake bullet" in report
    assert r"\[spoof\]\(https://example\.invalid\)" in report
