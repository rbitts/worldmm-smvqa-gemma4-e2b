from __future__ import annotations

import hashlib
from pathlib import Path

from worldmm_smvqa.qa_shards import checkpoint_rank, partial_output_path
from worldmm_smvqa.qa_transformers import TransformersCliArgs, run_transformers_cli
from worldmm_smvqa.schema import PredictionRecord
from worldmm_smvqa.smoke import run_smoke_pipeline

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/tiny_smvqa"


def test_transformers_qa_resumes_partial_rank_checkpoint(tmp_path: Path) -> None:
    # Given: two completed predictions saved before an interrupted rank exits.
    smoke_dir = tmp_path / "smoke"
    _ = run_smoke_pipeline(FIXTURE, smoke_dir, {})
    smoke_predictions = tuple(
        PredictionRecord.model_validate_json(line)
        for line in (smoke_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    output = tmp_path / "qa" / "predictions.jsonl"
    checkpoint_rank(output, smoke_predictions[:2])

    # When: the same rank restarts.
    result = run_transformers_cli(
        TransformersCliArgs(
            model="unused-mock-model",
            fixture=FIXTURE,
            evidence=smoke_dir / "evidence_packs.jsonl",
            out=output,
            backend="mock",
        ),
        env={},
    )

    # Then: final shard is complete, partial marker is removed, completed rows skip.
    rows = tuple(
        PredictionRecord.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    )
    assert result.predictions == 6
    assert len(rows) == 6
    assert not partial_output_path(output).exists()
    raw_root = output.parent / "predictions_raw_model_outputs"
    for prediction in smoke_predictions[:2]:
        digest = hashlib.sha256(prediction.question_id.encode()).hexdigest()[:16]
        assert not (raw_root / f"q_{digest}.json").exists()
