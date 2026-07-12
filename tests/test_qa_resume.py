from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from worldmm_smvqa.qa_shards import checkpoint_rank, partial_output_path
from worldmm_smvqa.qa_transformers import TransformersCliArgs, run_transformers_cli
from worldmm_smvqa.schema import PredictionRecord
from worldmm_smvqa.smoke import run_smoke_pipeline

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/tiny_smvqa"


def test_transformers_qa_resumes_partial_rank_checkpoint(tmp_path: Path) -> None:
    # Given: a provenance-bound run leaves two predictions before interruption.
    smoke_dir = tmp_path / "smoke"
    _ = run_smoke_pipeline(FIXTURE, smoke_dir, {})
    output = tmp_path / "qa" / "predictions.jsonl"
    args = TransformersCliArgs(
        model="unused-mock-model",
        fixture=FIXTURE,
        evidence=smoke_dir / "evidence_packs.jsonl",
        evidence_lane="heuristic",
        evidence_lineage=None,
        checkpoint=None,
        typed_memory=None,
        inference_manifest=None,
        require_frames=False,
        out=output,
        backend="mock",
    )
    _ = run_transformers_cli(args, env={})
    first_predictions = tuple(
        PredictionRecord.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    )
    output.unlink()
    checkpoint_rank(output, first_predictions[:2])
    raw_root = output.parent / "predictions_raw_model_outputs"
    shutil.rmtree(raw_root)

    # When: the same rank restarts.
    result = run_transformers_cli(args, env={})

    # Then: final shard is complete, partial marker is removed, completed rows skip.
    rows = tuple(
        PredictionRecord.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    )
    assert result.predictions == 6
    assert len(rows) == 6
    assert not partial_output_path(output).exists()
    for prediction in first_predictions[:2]:
        digest = hashlib.sha256(prediction.question_id.encode()).hexdigest()[:16]
        assert not (raw_root / f"q_{digest}.json").exists()
