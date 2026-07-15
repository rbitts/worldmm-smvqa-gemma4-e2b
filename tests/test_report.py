from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from worldmm_smvqa.report import (
    OracleRunManifest,
    StudentRunManifest,
    render_report,
)
from worldmm_smvqa.retrieval_types import (
    CanonicalOracleEvidencePack,
    EvidencePack,
    OracleEvidenceLineage,
    OracleVariantLineage,
    SharedQALineage,
    load_legacy_evidence_pack,
)

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


def _oracle_shared_qa_digest() -> str:
    digest = "a" * 64
    return SharedQALineage(
        approved_salt="approved-salt",
        world_size=1,
        question_map_sha256=digest,
        model_sha256=digest,
        prompt_sha256=digest,
        decoding_sha256=digest,
        runtime_sha256=digest,
        python_inventory_sha256=digest,
        torch_inventory_sha256=digest,
        transformers_inventory_sha256=digest,
        seed=7,
    ).sha256


def _oracle_variants(digest: str, qa_digest: str) -> tuple[OracleVariantLineage, ...]:
    return tuple(
        OracleVariantLineage(
            variant=variant,
            memory_sha256=digest,
            evidence_sha256=digest,
            predictions_sha256=digest,
            metrics_sha256=digest,
            pre_evaluation_sha256=qa_digest,
            finalization_receipt_sha256=digest,
            finalization_receipt_file_sha256=digest,
        )
        for variant in ("E0", "T0", "T1")
    )


def _oracle_lineage(
    digest: str, variants: tuple[OracleVariantLineage, ...]
) -> OracleEvidenceLineage:
    return OracleEvidenceLineage(
        producer="offline-teacher",
        sensor_audit_sha256=digest,
        object_semantic_sha256=digest,
        geometry_sha256=digest,
        place_sha256=digest,
        typed_memory_sha256=digest,
        shared_input_sha256=digest,
        variants=variants,
        shared_qa_lineage=SharedQALineage(
            approved_salt="approved-salt",
            world_size=1,
            question_map_sha256=digest,
            model_sha256=digest,
            prompt_sha256=digest,
            decoding_sha256=digest,
            runtime_sha256=digest,
            python_inventory_sha256=digest,
            torch_inventory_sha256=digest,
            transformers_inventory_sha256=digest,
            seed=7,
        ),
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
        (
            "complete",
            "Remote benchmark failed.",
            "complete result must not contain failure_reason",
        ),
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
                "metrics": [{**metric, "experiment": "E3"} for metric in metrics],
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


def test_oracle_manifest_is_discriminated_and_requires_complete_lineage() -> None:

    qa_digest = _oracle_shared_qa_digest()
    digest = "a" * 64
    variants = _oracle_variants(digest, qa_digest)
    common: dict[str, object] = {
        "baseline_name": "WorldMM-SMVQA",
        "remote_status": "complete",
        "result_class": "teacher_oracle",
        "manifest_schema": "oracle_variants_terminal_v1",
        "execution_profile": "teacher-oracle",
        "experiment_id": "EXP-0005",
        "split_id": digest,
        "code_sha256": digest,
        "local_changes": ["report"],
        "remote_command": "sbatch oracle.sh",
        "remote_job_reference": "123",
        "remote_artifact_path": "/approved/oracle",
        "not_copied_locally": ["data"],
        "operational_decision": "accepted",
        "provider_result_sha256": digest,
        "provider_outcome": "provider_result",
        "producer_status_sha256": digest,
        "approval_sha256": digest,
        "preflight_sha256": digest,
        "payload_sha256": digest,
        "outcome_sha256": digest,
        "continuation_sha256": digest,
        "continuation_consume_sha256": digest,
        "operational_artifact_sha256": digest,
        "scientific_artifact_sha256": digest,
        "scientific_decision": "not_measurable",
        "scientific_terminal_reason": "Required measurement is unavailable.",
        "sensor_audit_sha256": digest,
        "object_semantic_sha256": digest,
        "geometry_sha256": digest,
        "place_sha256": digest,
        "typed_memory_sha256": digest,
        "shared_input_sha256": digest,
        "qa_model_sha256": digest,
        "qa_prompt_sha256": digest,
        "qa_seed": 7,
        "qa_question_map_sha256": digest,
        "qa_approved_salt": "approved-salt",
        "qa_world_size": 1,
        "qa_decoding_sha256": digest,
        "qa_runtime_sha256": digest,
        "qa_pre_evaluation_sha256": qa_digest,
        "qa_python_inventory_sha256": digest,
        "qa_torch_inventory_sha256": digest,
        "qa_transformers_inventory_sha256": digest,
        "oracle_lineage_sha256": digest,
        "metrics": [],
        "variants": variants,
    }
    lineage = _oracle_lineage(digest, variants)
    common["oracle_lineage"] = lineage.model_dump(mode="json")
    common["oracle_lineage_sha256"] = lineage.sha256
    oracle = OracleRunManifest.model_validate(common)
    assert oracle.lane == "teacher_oracle"
    with pytest.raises(ValidationError, match="checkpoint_sha256"):
        _ = OracleRunManifest.model_validate({**common, "checkpoint_sha256": digest})
    with pytest.raises(ValidationError, match="E0, T0, and T1"):
        _ = OracleRunManifest.model_validate({**common, "variants": variants[:2]})
    with pytest.raises(ValidationError, match="contract_probe"):
        _ = StudentRunManifest.model_validate(common)
    incomplete = {
        **common,
        "sensor_audit_sha256": None,
    }
    with pytest.raises(ValidationError, match="sensor_audit_sha256"):
        _ = OracleRunManifest.model_validate(incomplete)
    tampered_values: tuple[tuple[str, object], ...] = (
        ("qa_approved_salt", "other-approved-salt"),
        ("qa_world_size", 2),
        ("qa_question_map_sha256", "b" * 64),
        ("qa_model_sha256", "b" * 64),
        ("qa_prompt_sha256", "b" * 64),
        ("qa_decoding_sha256", "b" * 64),
        ("qa_runtime_sha256", "b" * 64),
        ("qa_seed", 8),
        ("qa_pre_evaluation_sha256", "b" * 64),
        ("qa_python_inventory_sha256", "b" * 64),
        ("qa_torch_inventory_sha256", "b" * 64),
        ("qa_transformers_inventory_sha256", "b" * 64),
    )
    for field, value in tampered_values:
        with pytest.raises(
            ValidationError,
            match=r"canonical shared QA lineage|same pre-evaluation QA lineage",
        ):
            _ = OracleRunManifest.model_validate({**common, field: value})
    with pytest.raises(ValidationError, match="top-level evidence digests"):
        _ = OracleRunManifest.model_validate(
            {**common, "sensor_audit_sha256": "b" * 64}
        )


def test_provider_audit_is_terminal_without_variant_artifacts() -> None:

    payload = {
        "baseline_name": "WorldMM-SMVQA",
        "remote_status": "complete",
        "result_class": "teacher_oracle",
        "manifest_schema": "provider_gate_terminal_v1",
        "execution_profile": "provider-audit",
        "experiment_id": "EXP-0004",
        "split_id": "provider-only",
        "local_changes": ["report"],
        "remote_command": "sbatch audit.sh",
        "remote_job_reference": "123",
        "remote_artifact_path": "/approved/audit",
        "not_copied_locally": ["data"],
        "operational_decision": "accepted",
        "provider_result_sha256": "a" * 64,
        "provider_outcome": "provider_result",
        "producer_status_sha256": "a" * 64,
        "approval_sha256": "a" * 64,
        "preflight_sha256": "a" * 64,
        "payload_sha256": "a" * 64,
        "outcome_sha256": "a" * 64,
        "operational_artifact_sha256": "a" * 64,
        "scientific_artifact_sha256": "a" * 64,
    }
    assert OracleRunManifest.model_validate(payload).variants == ()
    assert "Provider audit completed: accepted." in render_report(
        OracleRunManifest.model_validate(payload)
    )
    with pytest.raises(ValidationError, match="scientific decision or evidence"):
        _ = OracleRunManifest.model_validate(
            {**payload, "scientific_decision": "no_go"}
        )
    with pytest.raises(ValidationError, match="scientific decision or evidence"):
        _ = OracleRunManifest.model_validate(
            {
                **payload,
                "variants": [
                    {
                        "variant": "E0",
                        **dict.fromkeys(
                            (
                                "memory_sha256",
                                "evidence_sha256",
                                "predictions_sha256",
                                "metrics_sha256",
                                "pre_evaluation_sha256",
                                "finalization_receipt_sha256",
                                "finalization_receipt_file_sha256",
                            ),
                            "a" * 64,
                        ),
                    }
                ],
            }
        )
    with pytest.raises(ValidationError, match="provider"):
        _ = OracleRunManifest.model_validate(
            {
                key: value
                for key, value in payload.items()
                if key != "provider_result_sha256"
            }
        )


@pytest.mark.parametrize(
    ("remote_status", "scientific_decision", "provider_outcome", "provider_field"),
    [
        ("complete", "no_go", "provider_result", "provider_result_sha256"),
        (
            "complete",
            "not_measurable",
            "provider_result",
            "provider_result_sha256",
        ),
        (
            "failed",
            "not_decidable",
            "contract_ineligible",
            "provider_result_sha256",
        ),
        (
            "failed",
            "not_decidable",
            "producer_failure",
            "provider_failure_sha256",
        ),
        (
            "failed",
            "not_decidable",
            "gate_failure",
            "provider_failure_sha256",
        ),
        (
            "cancelled",
            "not_decidable",
            "cancelled_pre_continuation",
            "provider_failure_sha256",
        ),
    ],
)
def test_early_exp0005_terminal_truth_table(
    remote_status: str,
    scientific_decision: str,
    provider_outcome: str,
    provider_field: str,
) -> None:
    digest = "a" * 64
    payload: dict[str, object] = {
        "baseline_name": "WorldMM-SMVQA",
        "remote_status": remote_status,
        "result_class": "teacher_oracle",
        "manifest_schema": "provider_gate_terminal_v1",
        "execution_profile": "teacher-oracle",
        "experiment_id": "EXP-0005",
        "split_id": "terminal",
        "local_changes": ["report"],
        "remote_command": "sbatch terminal.sh",
        "remote_job_reference": "123",
        "remote_artifact_path": "/approved/terminal",
        "not_copied_locally": ["data"],
        "operational_decision": "rejected",
        "scientific_decision": scientific_decision,
        "provider_outcome": provider_outcome,
        "producer_status_sha256": digest,
        "approval_sha256": digest,
        "preflight_sha256": digest,
        "payload_sha256": digest,
        "outcome_sha256": digest,
        "operational_artifact_sha256": digest,
        "scientific_artifact_sha256": digest,
        provider_field: digest,
    }
    if remote_status != "complete":
        payload["failure_reason"] = "provider failed"
    assert OracleRunManifest.model_validate(payload).remote_status == remote_status
    wrong_provider = (
        "provider_failure_sha256"
        if provider_field == "provider_result_sha256"
        else "provider_result_sha256"
    )
    with pytest.raises(ValidationError, match="matching provider"):
        _ = OracleRunManifest.model_validate(
            {key: value for key, value in payload.items() if key != provider_field}
            | {wrong_provider: digest}
        )
    with pytest.raises(ValidationError, match="matching provider"):
        _ = OracleRunManifest.model_validate({**payload, wrong_provider: digest})
    with pytest.raises(ValidationError, match="provider/gate truth table"):
        _ = OracleRunManifest.model_validate(
            {**payload, "provider_outcome": "provider_result"}
            if provider_outcome != "provider_result"
            else {**payload, "provider_outcome": "contract_ineligible"}
        )


@pytest.mark.parametrize("remote_status", ["failed", "cancelled"])
def test_full_downstream_terminal_preserves_provider_result_and_continuation(
    remote_status: str,
) -> None:
    digest = "a" * 64
    payload = {
        "baseline_name": "WorldMM-SMVQA",
        "remote_status": remote_status,
        "result_class": "teacher_oracle",
        "manifest_schema": "oracle_variants_terminal_v1",
        "execution_profile": "teacher-oracle",
        "experiment_id": "EXP-0005",
        "split_id": "terminal",
        "local_changes": ["report"],
        "remote_command": "sbatch terminal.sh",
        "remote_job_reference": "123",
        "remote_artifact_path": "/approved/terminal",
        "not_copied_locally": ["data"],
        "failure_reason": "downstream failure",
        "operational_decision": "rejected",
        "scientific_decision": "not_decidable",
        "provider_outcome": "provider_result",
        "provider_result_sha256": digest,
        "producer_status_sha256": digest,
        "approval_sha256": digest,
        "preflight_sha256": digest,
        "payload_sha256": digest,
        "outcome_sha256": digest,
        "continuation_sha256": digest,
        "continuation_consume_sha256": digest,
        "operational_artifact_sha256": digest,
        "scientific_artifact_sha256": digest,
    }
    assert OracleRunManifest.model_validate(payload).remote_status == remote_status
    for omitted in (
        "provider_result_sha256",
        "continuation_sha256",
        "continuation_consume_sha256",
    ):
        with pytest.raises(
            ValidationError, match=r"mandatory digests|matching provider"
        ):
            _ = OracleRunManifest.model_validate(
                {key: value for key, value in payload.items() if key != omitted}
            )
    with pytest.raises(ValidationError, match="matching provider"):
        _ = OracleRunManifest.model_validate(
            {
                **payload,
                "provider_result_sha256": None,
                "provider_failure_sha256": digest,
            }
        )


@pytest.mark.parametrize(
    ("remote_status", "provider_outcome", "provider_field", "operational_decision"),
    [
        ("complete", "provider_result", "provider_result_sha256", "accepted"),
        ("failed", "producer_failure", "provider_failure_sha256", "rejected"),
        ("failed", "gate_failure", "provider_failure_sha256", "rejected"),
        (
            "cancelled",
            "cancelled_pre_continuation",
            "provider_failure_sha256",
            "rejected",
        ),
    ],
)
def test_provider_audit_terminals_require_every_mandatory_lineage_digest(
    remote_status: str,
    provider_outcome: str,
    provider_field: str,
    operational_decision: str,
) -> None:
    digest = "a" * 64
    payload: dict[str, object] = {
        "baseline_name": "WorldMM-SMVQA",
        "remote_status": remote_status,
        "result_class": "teacher_oracle",
        "manifest_schema": "provider_gate_terminal_v1",
        "execution_profile": "provider-audit",
        "experiment_id": "EXP-0004",
        "split_id": "provider-only",
        "local_changes": ["report"],
        "remote_command": "sbatch audit.sh",
        "remote_job_reference": "123",
        "remote_artifact_path": "/approved/audit",
        "operational_decision": operational_decision,
        "provider_outcome": provider_outcome,
        "producer_status_sha256": digest,
        "approval_sha256": digest,
        "preflight_sha256": digest,
        "payload_sha256": digest,
        "outcome_sha256": digest,
        "operational_artifact_sha256": digest,
        "scientific_artifact_sha256": digest,
        "not_copied_locally": ["data"],
        provider_field: digest,
    }
    if remote_status != "complete":
        payload["failure_reason"] = "provider failed"
    assert OracleRunManifest.model_validate(payload).remote_status == remote_status
    for omitted in (
        "provider_outcome",
        provider_field,
        "producer_status_sha256",
        "approval_sha256",
        "preflight_sha256",
        "payload_sha256",
        "outcome_sha256",
        "operational_artifact_sha256",
        "scientific_artifact_sha256",
    ):
        with pytest.raises(
            ValidationError,
            match=r"mandatory digests|matching provider|provider truth table",
        ):
            _ = OracleRunManifest.model_validate(
                {key: value for key, value in payload.items() if key != omitted}
            )


@pytest.mark.parametrize(
    ("remote_status", "provider_outcome", "operational_decision"),
    [
        ("complete", "provider_result", "rejected"),
        ("failed", "producer_failure", "accepted"),
        ("cancelled", "cancelled_pre_continuation", "accepted"),
    ],
)
def test_provider_audit_rejects_terminal_decision_contradictions(
    remote_status: str, provider_outcome: str, operational_decision: str
) -> None:
    digest = "a" * 64
    payload: dict[str, object] = {
        "baseline_name": "WorldMM-SMVQA",
        "remote_status": remote_status,
        "result_class": "teacher_oracle",
        "manifest_schema": "provider_gate_terminal_v1",
        "execution_profile": "provider-audit",
        "experiment_id": "EXP-0004",
        "split_id": "provider-only",
        "local_changes": ["report"],
        "remote_command": "sbatch audit.sh",
        "remote_job_reference": "123",
        "remote_artifact_path": "/approved/audit",
        "not_copied_locally": ["data"],
        "operational_decision": operational_decision,
        "provider_outcome": provider_outcome,
        "producer_status_sha256": digest,
        "approval_sha256": digest,
        "preflight_sha256": digest,
        "payload_sha256": digest,
        "outcome_sha256": digest,
        "operational_artifact_sha256": digest,
        "scientific_artifact_sha256": digest,
        (
            "provider_result_sha256"
            if remote_status == "complete"
            else "provider_failure_sha256"
        ): digest,
    }
    if remote_status != "complete":
        payload["failure_reason"] = "provider failed"
    with pytest.raises(ValidationError, match="provider truth table"):
        _ = OracleRunManifest.model_validate(payload)


def test_provider_audit_pending_rejects_decisions() -> None:
    payload = {
        "baseline_name": "WorldMM-SMVQA",
        "remote_status": "pending",
        "result_class": "teacher_oracle",
        "manifest_schema": "provider_gate_terminal_v1",
        "execution_profile": "provider-audit",
        "experiment_id": "EXP-0004",
        "split_id": "not-run",
        "local_changes": ["report"],
        "remote_command": "sbatch audit.sh",
        "remote_job_reference": "not-run",
        "remote_artifact_path": "/approved/audit",
        "not_copied_locally": ["data"],
        "operational_decision": "pending",
    }
    with pytest.raises(ValidationError, match="oracle decisions require"):
        _ = OracleRunManifest.model_validate(payload)


@pytest.mark.parametrize(
    ("execution_profile", "experiment_id", "expected_error"),
    [
        (
            "teacher-oracle",
            "EXP-0004",
            "teacher-oracle requires experiment_id=EXP-0005",
        ),
        (
            "teacher-oracle",
            "EXP-9999",
            "teacher-oracle requires experiment_id=EXP-0005",
        ),
        (
            "provider-audit",
            "EXP-0005",
            "provider-audit requires experiment_id=EXP-0004",
        ),
        (
            "provider-audit",
            "EXP-9999",
            "provider-audit requires experiment_id=EXP-0004",
        ),
    ],
)
def test_oracle_profile_requires_its_canonical_experiment(
    execution_profile: str,
    experiment_id: str,
    expected_error: str,
) -> None:
    payload = {
        "baseline_name": "WorldMM-SMVQA",
        "remote_status": "pending",
        "result_class": "teacher_oracle",
        "execution_profile": execution_profile,
        "experiment_id": experiment_id,
        "split_id": "not-run",
        "local_changes": ["report"],
        "remote_command": "sbatch oracle.sh",
        "remote_job_reference": "not-run",
        "remote_artifact_path": "/approved/oracle",
        "not_copied_locally": ["data"],
    }
    with pytest.raises(ValidationError, match=expected_error):
        _ = OracleRunManifest.model_validate(payload)


def test_terminal_oracle_reason_is_stripped_rendered_and_not_blank() -> None:
    qa_digest = _oracle_shared_qa_digest()
    digest = "a" * 64
    payload: dict[str, object] = {
        "baseline_name": "WorldMM-SMVQA",
        "remote_status": "complete",
        "result_class": "teacher_oracle",
        "execution_profile": "teacher-oracle",
        "manifest_schema": "oracle_variants_terminal_v1",
        "experiment_id": "EXP-0005",
        "split_id": digest,
        "code_sha256": digest,
        "local_changes": ["report"],
        "remote_command": "sbatch oracle.sh",
        "remote_job_reference": "123",
        "remote_artifact_path": "/approved/oracle",
        "not_copied_locally": ["data"],
        "operational_decision": "accepted",
        "provider_result_sha256": digest,
        "provider_outcome": "provider_result",
        "producer_status_sha256": digest,
        "approval_sha256": digest,
        "preflight_sha256": digest,
        "payload_sha256": digest,
        "outcome_sha256": digest,
        "continuation_sha256": digest,
        "continuation_consume_sha256": digest,
        "operational_artifact_sha256": digest,
        "scientific_artifact_sha256": digest,
        "scientific_decision": "not_measurable",
        "scientific_terminal_reason": "  Required measurement unavailable.  ",
        "sensor_audit_sha256": digest,
        "object_semantic_sha256": digest,
        "geometry_sha256": digest,
        "place_sha256": digest,
        "typed_memory_sha256": digest,
        "shared_input_sha256": digest,
        "qa_model_sha256": digest,
        "qa_prompt_sha256": digest,
        "qa_seed": 7,
        "qa_question_map_sha256": digest,
        "qa_approved_salt": "approved-salt",
        "qa_world_size": 1,
        "oracle_lineage_sha256": digest,
        "qa_decoding_sha256": digest,
        "qa_runtime_sha256": digest,
        "qa_pre_evaluation_sha256": qa_digest,
        "qa_python_inventory_sha256": digest,
        "qa_torch_inventory_sha256": digest,
        "qa_transformers_inventory_sha256": digest,
        "variants": _oracle_variants(digest, qa_digest),
        "metrics": [],
    }
    lineage = _oracle_lineage(
        digest, cast("tuple[OracleVariantLineage, ...]", payload["variants"])
    )
    payload["oracle_lineage"] = lineage.model_dump(mode="json")
    payload["oracle_lineage_sha256"] = lineage.sha256
    manifest = OracleRunManifest.model_validate(payload)
    report = render_report(manifest)
    assert manifest.scientific_terminal_reason == "Required measurement unavailable."
    assert r"Required measurement unavailable\." in report
    assert "Remote benchmark not run." not in report
    with pytest.raises(
        ValidationError,
        match="scientific_terminal_reason must not be blank",
    ):
        _ = OracleRunManifest.model_validate(
            {**payload, "scientific_terminal_reason": "   "}
        )


def test_pending_report_labels_command_as_not_run() -> None:
    payload = _fixture_payload()

    report = render_report(StudentRunManifest.model_validate(payload))
    assert "Remote command not run." in report
    assert "Planned remote command:" in report


def test_legacy_evidence_loader_is_explicit_and_oracle_validation_is_strict() -> None:

    payload: dict[str, object] = {
        "question_id": "q1",
        "variant": "E0",
        "video_id": "v1",
        "requested_stores": [],
        "selected_stores": [],
        "evidence_budget": 0,
        "evidence": [],
        "causal_filtered_count": 0,
    }
    legacy_payload = {key: value for key, value in payload.items() if key != "variant"}
    assert EvidencePack.model_validate(legacy_payload).retrieval_trace.policy_route == (
        "legacy-missing-trace"
    )
    assert load_legacy_evidence_pack(legacy_payload).retrieval_trace.policy_route == (
        "legacy-missing-trace"
    )
    with pytest.raises(ValidationError, match="retrieval_trace"):
        _ = CanonicalOracleEvidencePack.model_validate(payload)


def test_canonical_oracle_evidence_requires_matching_video_and_nonblank_route() -> None:
    payload: dict[str, object] = {
        "question_id": "q1",
        "variant": "E0",
        "video_id": "v1",
        "requested_stores": ["semantic"],
        "selected_stores": ["semantic"],
        "evidence_budget": 1,
        "evidence": [
            {
                "memory_id": "m1",
                "video_id": "v2",
                "snippet": "evidence",
                "frame_refs": [],
                "source_store": "semantic",
                "start_time": 0.0,
                "end_time": 1.0,
                "retrieval_score": 1.0,
            }
        ],
        "causal_filtered_count": 0,
        "retrieval_trace": {
            "protocols": [],
            "eligible_shard_ids": [],
            "selected_clip_ids": [],
            "policy_route": "balanced",
            "store_order": ["semantic"],
            "candidate_counts": [],
            "causal_filtered_count": 0,
            "frame_ref_count": 0,
        },
    }
    evidence_rows = cast("list[dict[str, object]]", payload["evidence"])
    retrieval_trace = cast("dict[str, object]", payload["retrieval_trace"])
    with pytest.raises(ValidationError, match="video_ids must match"):
        _ = CanonicalOracleEvidencePack.model_validate(payload)
    with pytest.raises(ValidationError, match="nonblank retrieval_trace policy_route"):
        _ = CanonicalOracleEvidencePack.model_validate(
            {
                **payload,
                "evidence": [{**evidence_rows[0], "video_id": "v1"}],
                "retrieval_trace": {
                    **retrieval_trace,
                    "policy_route": "   ",
                },
            }
        )
    legacy = EvidencePack.model_validate(
        {
            **{key: value for key, value in payload.items() if key != "variant"},
            "retrieval_trace": {
                **retrieval_trace,
                "policy_route": "   ",
            },
        }
    )
    assert legacy.evidence[0].video_id == "v2"
    assert legacy.retrieval_trace.policy_route == "   "


def test_oracle_metric_matrix_rejects_under_bound() -> None:
    qa_digest = _oracle_shared_qa_digest()
    digest = "a" * 64
    base: dict[str, object] = {
        "baseline_name": "WorldMM-SMVQA",
        "remote_status": "complete",
        "result_class": "teacher_oracle",
        "execution_profile": "teacher-oracle",
        "manifest_schema": "oracle_variants_terminal_v1",
        "experiment_id": "EXP-0005",
        "split_id": digest,
        "code_sha256": digest,
        "local_changes": ["report"],
        "remote_command": "sbatch oracle.sh",
        "remote_job_reference": "123",
        "remote_artifact_path": "/approved/oracle",
        "not_copied_locally": ["data"],
        "operational_decision": "accepted",
        "provider_result_sha256": digest,
        "provider_outcome": "provider_result",
        "producer_status_sha256": digest,
        "approval_sha256": digest,
        "preflight_sha256": digest,
        "payload_sha256": digest,
        "outcome_sha256": digest,
        "continuation_sha256": digest,
        "continuation_consume_sha256": digest,
        "operational_artifact_sha256": digest,
        "scientific_artifact_sha256": digest,
        "scientific_decision": "go",
        "scientific_thresholds_sha256": digest,
        "scientific_evidence_sha256": digest,
        "sensor_audit_sha256": digest,
        "object_semantic_sha256": digest,
        "geometry_sha256": digest,
        "place_sha256": digest,
        "typed_memory_sha256": digest,
        "shared_input_sha256": digest,
        "qa_model_sha256": digest,
        "qa_prompt_sha256": digest,
        "qa_seed": 7,
        "qa_question_map_sha256": digest,
        "qa_approved_salt": "approved-salt",
        "qa_world_size": 1,
        "oracle_lineage_sha256": digest,
        "qa_decoding_sha256": digest,
        "qa_runtime_sha256": digest,
        "qa_pre_evaluation_sha256": qa_digest,
        "qa_python_inventory_sha256": digest,
        "qa_torch_inventory_sha256": digest,
        "qa_transformers_inventory_sha256": digest,
        "variants": _oracle_variants(digest, qa_digest),
        "metrics": [{"experiment": "E0", "name": "QA-Acc", "value": 1.0}],
    }
    lineage = _oracle_lineage(
        digest, cast("tuple[OracleVariantLineage, ...]", base["variants"])
    )
    base["oracle_lineage"] = lineage.model_dump(mode="json")
    base["oracle_lineage_sha256"] = lineage.sha256
    with pytest.raises(ValidationError, match="complete unique bounded metric matrix"):
        _ = OracleRunManifest.model_validate(base)
