from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from worldmm_smvqa import smoke as smoke_module
from worldmm_smvqa.retrieval_types import EvidencePack
from worldmm_smvqa.schema import FrozenModel, MemoryBuilderCandidate, PredictionRecord
from worldmm_smvqa.smoke import SmokeMemoryManifest
from worldmm_smvqa.worldmm.semantic import (
    SemanticTripleRecord,
    build_semantic_memory,
)

if TYPE_CHECKING:
    import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "tests/fixtures/tiny_smvqa"


class SmokeSummaryStats(FrozenModel):
    count: int
    total: int
    min: int
    max: int
    mean: float


class SmokeMetricDiagnostics(FrozenModel):
    causal_violation_count: int
    prompt_tokens: SmokeSummaryStats
    memory_size: SmokeSummaryStats


class SmokeMetricRecord(FrozenModel):
    ans_f1: float = Field(validation_alias="Ans-F1")
    qa_acc: float = Field(validation_alias="QA-Acc")
    qa_mrr: float = Field(validation_alias="QA-MRR")
    memory_recall_at_1: float = Field(validation_alias="Memory-Recall@1")
    memory_recall_at_3: float = Field(validation_alias="Memory-Recall@3")
    memory_recall_at_5: float = Field(validation_alias="Memory-Recall@5")
    diagnostics: SmokeMetricDiagnostics


def run_cli(*args: str, disable_mock: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    _ = env.pop("WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST", None)
    if disable_mock:
        env["WORLDMM_SMVQA_DISABLE_MOCK"] = "1"
    return subprocess.run(
        ["uv", "run", "--offline", "worldmm-smvqa", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_smoke_cli_writes_parseable_artifacts_and_replaces_rerun(
    tmp_path: Path,
) -> None:
    # Given: a tiny fixture and a clean smoke output directory.
    out_dir = tmp_path / "smoke"

    # When: smoke runs through the CLI twice into the same output directory.
    first = run_cli("smoke", "--fixture", FIXTURE, "--out", str(out_dir))
    second = run_cli("smoke", "--fixture", FIXTURE, "--out", str(out_dir))

    # Then: required artifacts parse and rerun did not append stale rows.
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "predictions=4" in second.stdout
    metrics = SmokeMetricRecord.model_validate_json(
        (out_dir / "metrics.json").read_text(encoding="utf-8"),
    )
    assert metrics.ans_f1 >= 0.0
    assert metrics.qa_acc >= 0.0
    assert metrics.qa_mrr >= 0.0
    assert metrics.diagnostics.prompt_tokens.total > 0

    predictions = [
        PredictionRecord.model_validate_json(line)
        for line in (out_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    packs = [
        EvidencePack.model_validate_json(line)
        for line in (out_dir / "evidence_packs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    manifest = SmokeMemoryManifest.model_validate_json(
        (out_dir / "memory_manifest.json").read_text(encoding="utf-8"),
    )
    assert len(predictions) == 4
    assert len(packs) == 4
    assert {pack.question_id for pack in packs} == {
        prediction.question_id for prediction in predictions
    }
    assert manifest.counts_by_store.retrieval.episodic > 0
    assert manifest.counts_by_store.retrieval.semantic > 0
    assert manifest.counts_by_store.retrieval.visual > 0


def test_smoke_cli_fails_when_mock_backend_disabled(tmp_path: Path) -> None:
    # Given: a smoke output directory and mock backend disabled by env.
    out_dir = tmp_path / "smoke-disabled"

    # When: smoke is invoked locally.
    result = run_cli(
        "smoke",
        "--fixture",
        FIXTURE,
        "--out",
        str(out_dir),
        disable_mock=True,
    )

    # Then: it fails before producing model-backed smoke artifacts.
    assert result.returncode != 0
    assert "NoLocalModelBackend" in f"{result.stdout}\n{result.stderr}"
    assert not (out_dir / "predictions.jsonl").exists()


def test_smoke_retrieval_uses_current_worldmm_store_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: semantic memory built by smoke includes a current-run record.
    out_dir = tmp_path / "smoke-current-store"
    injected_id = "semantic:current-smoke-store-record"

    def build_semantic_with_current_record(
        candidates: Sequence[MemoryBuilderCandidate],
    ) -> tuple[SemanticTripleRecord, ...]:
        records = build_semantic_memory(candidates)
        return (
            *records,
            SemanticTripleRecord(
                memory_id=injected_id,
                video_id="fake_video_001",
                subject="fake_video_001",
                predicate="current_smoke_store",
                object="mug",
                text="fake mug placed beside notebook current smoke store",
                support_memory_ids=("current-smoke-store-support",),
                support_event_count=1,
                start_time=5.0,
                end_time=12.0,
                confidence=100.0,
                text_embedding_id=f"embedding:{injected_id}:text",
            ),
        )

    monkeypatch.setattr(
        smoke_module,
        "build_semantic_memory",
        build_semantic_with_current_record,
    )

    # When: the smoke pipeline builds stores and runs retrieval.
    result = smoke_module.run_smoke_pipeline(Path(FIXTURE), out_dir, {})

    # Then: retrieval evidence came from the current built store state.
    packs = [
        EvidencePack.model_validate_json(line)
        for line in (out_dir / "evidence_packs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result.evidence_packs == 4
    assert any(
        item.memory_id == injected_id
        for pack in packs
        for item in pack.evidence
    )
