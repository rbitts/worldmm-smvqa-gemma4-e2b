from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from worldmm_smvqa.qa_shards import QAShardError
from worldmm_smvqa.qa_transformers import (
    QA_PROMPT_SCHEMA_VERSION,
    QA_RESUME_MANIFEST_VERSION,
    TransformersCliArgs,
    qa_resume_manifest,
    qa_resume_manifest_path,
    run_transformers_cli,
)
from worldmm_smvqa.retrieval_types import EvidenceLineage
from worldmm_smvqa.smoke import run_smoke_pipeline

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/tiny_smvqa"


def test_qa_resume_manifest_binds_inputs_backend_model_and_schema(
    tmp_path: Path,
) -> None:
    args = _completed_mock_run(tmp_path)

    manifest = cast(
        "dict[str, str]",
        json.loads(
            qa_resume_manifest_path(args.out).read_text(encoding="utf-8"),
        ),
    )

    assert manifest == {
        "backend": "mock",
        "checkpoint_sha256": "",
        "evidence_lane": "heuristic",
        "evidence_lineage_sha256": "",
        "model_contract_sha256": "",
        "student_architecture_sha256": "",
        "model_load_consensus_payload_sha256": "",
        "model_load_consensus_file_sha256": "",
        "evidence_sha256": _sha256(args.evidence),
        "expected_variant": "",
        "frame_assets_sha256": "",
        "inference_manifest_sha256": "",
        "inference_producer_sha256": "",
        "inference_sources_sha256": "",
        "lineage_config_sha256": "",
        "manifest_version": QA_RESUME_MANIFEST_VERSION,
        "memory_manifest_sha256": "",
        "model": "mock-model-v1",
        "model_fingerprint_sha256": "",
        "prompt_schema_version": QA_PROMPT_SCHEMA_VERSION,
        "questions_sha256": _sha256(FIXTURE / "questions.jsonl"),
        "require_frames": "false",
        "sensor_audit_sha256": "",
        "sensor_frame_manifest_sha256": "",
        "sources_sha256": _sha256(FIXTURE / "sources.jsonl"),
        "typed_memory_sha256": "",
    }


def test_student_resume_manifest_propagates_contract_and_consensus(
    tmp_path: Path,
) -> None:
    args = _completed_mock_run(tmp_path)
    trust = {
        "model_contract_sha256": "a" * 64,
        "student_architecture_sha256": "b" * 64,
        "model_load_consensus_payload_sha256": "c" * 64,
        "model_load_consensus_file_sha256": "d" * 64,
    }
    lineage = EvidenceLineage(
        lane="student",
        producer="spatial-student",
        evidence_sha256=_sha256(args.evidence),
        checkpoint_sha256="1" * 64,
        typed_memory_sha256="2" * 64,
        inference_manifest_sha256="3" * 64,
        config_sha256="4" * 64,
        sensor_sha256="5" * 64,
        data_sha256="6" * 64,
        memory_manifest_sha256="7" * 64,
        episodic_memory_sha256="8" * 64,
        semantic_memory_sha256="9" * 64,
        visual_memory_sha256="0" * 64,
        **trust,
    )
    lineage_path = tmp_path / "student.lineage.json"
    _ = lineage_path.write_text(lineage.model_dump_json(), encoding="utf-8")

    manifest = qa_resume_manifest(
        replace(
            args,
            evidence_lane="student",
            evidence_lineage=lineage_path,
        ),
    )

    assert {name: manifest[name] for name in trust} == trust


def test_qa_resume_rejects_changed_evidence(tmp_path: Path) -> None:
    args = _completed_mock_run(tmp_path)
    original = args.evidence.read_text(encoding="utf-8")
    _ = args.evidence.write_text(f"{original}\n", encoding="utf-8")

    with pytest.raises(QAShardError, match="evidence_sha256"):
        _ = run_transformers_cli(args, env={})


def test_qa_resume_rejects_changed_model(tmp_path: Path) -> None:
    args = _completed_mock_run(tmp_path)
    changed = TransformersCliArgs(
        model="mock-model-v2",
        fixture=args.fixture,
        evidence=args.evidence,
        evidence_lane=args.evidence_lane,
        evidence_lineage=args.evidence_lineage,
        checkpoint=args.checkpoint,
        typed_memory=args.typed_memory,
        inference_manifest=args.inference_manifest,
        require_frames=args.require_frames,
        out=args.out,
        backend=args.backend,
    )

    with pytest.raises(QAShardError, match=r"manifest mismatch \(model\)"):
        _ = run_transformers_cli(changed, env={})


def test_qa_resume_rejects_predictions_without_manifest(tmp_path: Path) -> None:
    args = _completed_mock_run(tmp_path)
    qa_resume_manifest_path(args.out).unlink()

    with pytest.raises(QAShardError, match="manifest missing"):
        _ = run_transformers_cli(args, env={})


def _completed_mock_run(tmp_path: Path) -> TransformersCliArgs:
    smoke_dir = tmp_path / "smoke"
    _ = run_smoke_pipeline(FIXTURE, smoke_dir, {})
    args = TransformersCliArgs(
        model="mock-model-v1",
        fixture=FIXTURE,
        evidence=smoke_dir / "evidence_packs.jsonl",
        evidence_lane="heuristic",
        evidence_lineage=None,
        checkpoint=None,
        typed_memory=None,
        inference_manifest=None,
        require_frames=False,
        out=tmp_path / "qa/predictions.jsonl",
        backend="mock",
    )
    _ = run_transformers_cli(args, env={})
    return args


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
