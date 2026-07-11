from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from worldmm_smvqa.qa_shards import QAShardError
from worldmm_smvqa.qa_transformers import (
    QA_PROMPT_SCHEMA_VERSION,
    QA_RESUME_MANIFEST_VERSION,
    TransformersCliArgs,
    qa_resume_manifest_path,
    run_transformers_cli,
)
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
        "evidence_sha256": _sha256(args.evidence),
        "manifest_version": QA_RESUME_MANIFEST_VERSION,
        "model": "mock-model-v1",
        "prompt_schema_version": QA_PROMPT_SCHEMA_VERSION,
        "questions_sha256": _sha256(FIXTURE / "questions.jsonl"),
        "sources_sha256": _sha256(FIXTURE / "sources.jsonl"),
    }


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
        out=tmp_path / "qa/predictions.jsonl",
        backend="mock",
    )
    _ = run_transformers_cli(args, env={})
    return args


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
