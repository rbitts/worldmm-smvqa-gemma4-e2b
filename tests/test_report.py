from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import cast

import pytest

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


def _fixture_payload() -> dict[str, object]:
    return cast(
        "dict[str, object]",
        json.loads((ROOT / FIXTURE_MANIFEST).read_text(encoding="utf-8")),
    )


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
  "result_class": "contract_probe",
  "experiment_id": "missing-artifact-contract-probe",
  "lane": "contract_probe",
  "split_id": "not-run",
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


@pytest.mark.parametrize(
    "field",
    ["remote_command", "remote_job_reference", "remote_artifact_path"],
)
def test_report_rejects_blank_remote_execution_identity(
    tmp_path: Path,
    field: str,
) -> None:
    # Given: a required remote execution identity contains only whitespace.
    payload = _fixture_payload()
    payload[field] = "   "
    manifest = tmp_path / f"blank-{field}.json"
    _ = manifest.write_text(json.dumps(payload), encoding="utf-8")

    # When: report generation validates the manifest.
    result = run_cli(
        "report",
        "--run-manifest",
        str(manifest),
        "--out",
        str(tmp_path / "report.md"),
    )

    # Then: whitespace cannot satisfy the remote identity contract.
    assert result.returncode != 0
    assert "remote execution identity must not be blank" in result.stderr
    assert not (tmp_path / "report.md").exists()


@pytest.mark.parametrize(
    ("remote_status", "failure_reason", "expected_error"),
    [
        ("failed", "   ", "failed result requires failure_reason"),
        ("complete", None, "heuristic result cannot claim remote completion"),
    ],
)
def test_report_rejects_status_without_required_payload(
    tmp_path: Path,
    remote_status: str,
    failure_reason: str | None,
    expected_error: str,
) -> None:
    # Given: status claims failure/completion without its required payload.
    payload = _fixture_payload()
    payload.update(
        remote_status=remote_status,
        result_class="heuristic",
        failure_reason=failure_reason,
        metrics=[],
    )
    manifest = tmp_path / f"invalid-{remote_status}.json"
    _ = manifest.write_text(json.dumps(payload), encoding="utf-8")

    # When: report generation validates the status claim.
    result = run_cli(
        "report",
        "--run-manifest",
        str(manifest),
        "--out",
        str(tmp_path / "report.md"),
    )

    # Then: an incomplete status payload fails closed.
    assert result.returncode != 0
    assert expected_error in result.stderr
    assert not (tmp_path / "report.md").exists()


def test_report_rejects_metrics_on_noncomplete_status(tmp_path: Path) -> None:
    payload = _fixture_payload()
    payload.update(
        remote_status="pending",
        result_class="official",
        metrics=[{"experiment": "E1", "name": "QA-Acc", "value": 99.0}],
    )
    manifest = tmp_path / "pending-metrics.json"
    _ = manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = run_cli(
        "report",
        "--run-manifest",
        str(manifest),
        "--out",
        str(tmp_path / "report.md"),
    )

    assert result.returncode != 0
    assert "pending or failed result must not contain metrics" in result.stderr


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
                "result_class": "mock",
                "experiment_id": "adversarial-mock",
                "lane": "mock",
                "split_id": "test",
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
                "evidence_lineage_sha256": "a" * 64,
                "metrics": [],
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


@pytest.mark.parametrize(
    ("result_class", "expected_error"),
    [
        ("student", "complete student result missing digests"),
        (
            "official",
            "official completion requires per-experiment immutable manifests",
        ),
    ],
)
def test_report_rejects_complete_learned_result_without_lineage(
    tmp_path: Path,
    result_class: str,
    expected_error: str,
) -> None:
    # Given: a learned result claims completion without immutable lineage.
    manifest = tmp_path / f"unbound-{result_class}.json"
    _ = manifest.write_text(
        json.dumps(
            {
                "baseline_name": "WorldMM-SMVQA",
                "remote_status": "complete",
                "result_class": result_class,
                "experiment_id": "official-run",
                "execution_profile": "full",
                "lane": "student",
                "split_id": "official-test",
                "local_changes": ["src/worldmm_smvqa/report.py"],
                "remote_command": "sbatch official.sh",
                "remote_job_reference": "123",
                "remote_artifact_path": "/approved/run",
                "metrics": [],
                "not_copied_locally": ["model weights"],
            },
        ),
        encoding="utf-8",
    )

    # When: report generation validates the completion claim.
    result = run_cli(
        "report",
        "--run-manifest",
        str(manifest),
        "--out",
        str(tmp_path / "report.md"),
    )

    # Then: learned completion fails closed before report creation.
    assert result.returncode != 0
    assert expected_error in result.stderr
    assert not (tmp_path / "report.md").exists()


def test_report_accepts_complete_student_lineage(
    tmp_path: Path,
) -> None:
    # Given: a complete digest-bound student result with one experiment.
    digest = "a" * 64
    manifest = tmp_path / "official.json"
    report_path = tmp_path / "report.md"
    metrics: list[dict[str, object]] = [
        {"experiment": "E1", "name": "Ans-F1", "value": 1.0},
        {"experiment": "E1", "name": "QA-Acc", "value": 1.0},
        {"experiment": "E1", "name": "QA-MRR", "value": 1.0},
    ]
    payload: dict[str, object] = {
        "baseline_name": "WorldMM-SMVQA",
        "remote_status": "complete",
        "result_class": "student",
        "experiment_id": "official-run",
        "execution_profile": "full",
        "lane": "student",
        "split_id": digest,
        "code_sha256": digest,
        "checkpoint_sha256": digest,
        "typed_memory_sha256": digest,
        "inference_manifest_sha256": digest,
        "evidence_sha256": digest,
        "evidence_lineage_sha256": digest,
        "model_sha256": digest,
        "prompt_sha256": digest,
        "predictions_sha256": digest,
        "metrics_sha256": digest,
        "qa_resume_manifest_sha256": digest,
        "run_identity_sha256": digest,
        "finalization_inputs_sha256": digest,
        "local_changes": ["src/worldmm_smvqa/report.py"],
        "remote_command": "sbatch official.sh",
        "remote_job_reference": "123",
        "remote_artifact_path": "/approved/run",
        "metrics": metrics,
        "not_copied_locally": ["model weights"],
    }
    _ = manifest.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    # When: report generation validates the student result.
    student = run_cli(
        "report",
        "--run-manifest",
        str(manifest),
        "--out",
        str(report_path),
    )

    # Then: student completion needs lineage, not the official experiment matrix.
    assert student.returncode == 0, student.stderr
    report = report_path.read_text(encoding="utf-8")
    assert "result_class: student" in report
    assert f"evidence_lineage_sha256: {digest}" in report
    assert f"run_identity_sha256: {digest}" in report
    assert f"finalization_inputs_sha256: {digest}" in report
    assert "- [E1] QA\\-Acc: 1.000000" in report

    _ = manifest.write_text(
        json.dumps({**payload, "evidence_lineage_sha256": None}),
        encoding="utf-8",
    )
    missing_lineage = run_cli(
        "report",
        "--run-manifest",
        str(manifest),
        "--out",
        str(tmp_path / "missing-evidence-lineage.md"),
    )
    assert missing_lineage.returncode != 0
    assert "evidence_lineage_sha256" in missing_lineage.stderr

    invalid_cases: tuple[tuple[dict[str, object], str], ...] = (
        (
            {"run_identity_sha256": None},
            "complete student result missing digests: run_identity_sha256",
        ),
        ({"run_identity_sha256": "not-a-digest"}, "run_identity_sha256"),
        (
            {"finalization_inputs_sha256": None},
            "complete student result missing digests: finalization_inputs_sha256",
        ),
        (
            {"finalization_inputs_sha256": "not-a-digest"},
            "finalization_inputs_sha256",
        ),
        ({"lane": "heuristic"}, "requires lane=student"),
        ({"split_id": "not-a-digest"}, "requires a SHA-256 split_id"),
        (
            {"metrics": metrics[:2]},
            "requires Ans-F1, QA-Acc, and QA-MRR",
        ),
        (
            {
                "metrics": [
                    {**metric, "experiment": "E3"}
                    for metric in metrics
                ],
            },
            "may contain only E1 metrics",
        ),
        (
            {
                "metrics": [
                    {**metrics[0], "value": 101.0},
                    *metrics[1:],
                ],
            },
            "must be between 0 and 100",
        ),
    )
    for index, (update, expected_error) in enumerate(invalid_cases):
        _ = manifest.write_text(
            json.dumps({**payload, **update}),
            encoding="utf-8",
        )
        rejected = run_cli(
            "report",
            "--run-manifest",
            str(manifest),
            "--out",
            str(tmp_path / f"rejected-{index}.md"),
        )
        assert rejected.returncode != 0
        assert expected_error in rejected.stderr


def test_report_requires_valid_contract_probe_evidence_lineage(
    tmp_path: Path,
) -> None:
    payload = _fixture_payload()
    digest = "a" * 64
    payload.update(
        remote_status="complete",
        execution_profile="probe",
        lane="student",
        split_id=digest,
        code_sha256=digest,
        checkpoint_sha256=digest,
        typed_memory_sha256=digest,
        inference_manifest_sha256=digest,
        evidence_sha256=digest,
        model_sha256=digest,
        prompt_sha256=digest,
        predictions_sha256=digest,
        metrics_sha256=digest,
        qa_resume_manifest_sha256=digest,
        run_identity_sha256=digest,
        finalization_inputs_sha256=digest,
        metrics=[
            {"experiment": "PROBE", "name": "Ans-F1", "value": 1.0},
            {"experiment": "PROBE", "name": "QA-Acc", "value": 1.0},
            {"experiment": "PROBE", "name": "QA-MRR", "value": 1.0},
        ],
        failure_reason=None,
    )
    manifest = tmp_path / "contract-probe.json"

    # Every completed probe is bound to the emitted evidence-lineage artifact.
    _ = manifest.write_text(json.dumps(payload), encoding="utf-8")
    missing = run_cli(
        "report",
        "--run-manifest",
        str(manifest),
        "--out",
        str(tmp_path / "contract-probe.md"),
    )
    assert missing.returncode != 0
    assert "evidence_lineage_sha256" in missing.stderr

    # The emitted artifact digest must be a valid SHA-256 value.
    _ = manifest.write_text(
        json.dumps({**payload, "evidence_lineage_sha256": "not-a-digest"}),
        encoding="utf-8",
    )
    rejected = run_cli(
        "report",
        "--run-manifest",
        str(manifest),
        "--out",
        str(tmp_path / "invalid-contract-probe.md"),
    )
    assert rejected.returncode != 0
    assert "evidence_lineage_sha256" in rejected.stderr

    _ = manifest.write_text(
        json.dumps({**payload, "evidence_lineage_sha256": digest}),
        encoding="utf-8",
    )
    accepted = run_cli(
        "report",
        "--run-manifest",
        str(manifest),
        "--out",
        str(tmp_path / "valid-contract-probe.md"),
    )
    assert accepted.returncode == 0, accepted.stderr
