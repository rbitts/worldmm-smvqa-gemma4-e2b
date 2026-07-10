from __future__ import annotations

import os
import subprocess
from pathlib import Path

from worldmm_smvqa.spatial_selector_train import (
    SelectorTrainingRow,
    build_selector_training_rows,
    train_selector_model,
)
from worldmm_smvqa.worldmm.spatial_compression import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/tiny_smvqa"


def test_geometry_qa_builds_balanced_selector_rows() -> None:
    # Given: labels are evaluator-only but available to the offline trainer.
    # When: selector supervision is prepared from geometry-QA evidence spans.
    rows = build_selector_training_rows(FIXTURE)

    # Then: source-derived features receive both keep and drop labels.
    assert rows
    assert {row.label for row in rows} == {0, 1}
    assert all(len(row.features) == len(FEATURE_NAMES) for row in rows)
    assert {row.question_id for row in rows} == {"q_fake_001", "q_fake_005"}


def test_linear_selector_trains_without_a_framework_dependency() -> None:
    # Given: tiny QA-supervised selector rows.
    rows = build_selector_training_rows(FIXTURE)

    # When: the on-device linear scorer is fitted.
    result = train_selector_model(rows, epochs=100, learning_rate=2.0)

    # Then: training is deterministic, small, and separates the fixture rows.
    assert result.rows == len(rows)
    assert result.positives > 0
    assert result.accuracy == 1.0
    assert result.loss < 0.6
    assert result.model.weights != (
        1.6,
        1.8,
        0.3,
        1.0,
        0.8,
        0.2,
        0.4,
        0.2,
    )


def test_selector_accepts_projected_features_from_future_encoders() -> None:
    # Given: a projection head adds one CUT3R-derived latent feature.
    rows = (
        SelectorTrainingRow(
            question_id="q1",
            candidate_id="keep-1",
            features={"kind_object": 1.0, "cut3r_latent_0": 1.0},
            label=1,
        ),
        SelectorTrainingRow(
            question_id="q1",
            candidate_id="drop-1",
            features={"kind_object": 1.0, "cut3r_latent_0": 0.0},
            label=0,
        ),
    )

    # When: the same selector trainer fits the expanded feature schema.
    result = train_selector_model(rows, epochs=100)

    # Then: no core trainer change is needed for the new encoder dimension.
    assert "cut3r_latent_0" in result.model.feature_names


def test_selector_prepare_cli_writes_jsonl(tmp_path: Path) -> None:
    # Given: a tiny fixture and output path.
    output = tmp_path / "selector_rows.jsonl"

    # When: the standalone preparation command runs locally.
    result = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "python",
            "-m",
            "worldmm_smvqa.spatial_selector_train",
            "prepare",
            "--fixture",
            str(FIXTURE),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then: compact supervised rows are written without model downloads.
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert output.read_text(encoding="utf-8").count("\n") > 0
    assert '"positives":' in result.stdout


def test_selector_train_cli_is_remote_only(tmp_path: Path) -> None:
    # Given: prepared rows but no remote-host approval.
    rows = tmp_path / "rows.jsonl"
    prepare = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "python",
            "-m",
            "worldmm_smvqa.spatial_selector_train",
            "prepare",
            "--fixture",
            str(FIXTURE),
            "--out",
            str(rows),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepare.returncode == 0, prepare.stderr
    output = tmp_path / "selector.json"
    env = os.environ.copy()
    _ = env.pop("WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST", None)

    # When: real selector training is requested on the development host.
    result = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "python",
            "-m",
            "worldmm_smvqa.spatial_selector_train",
            "train",
            "--config",
            "configs/remote.example.yaml",
            "--input",
            str(rows),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then: guardrails stop training before writing weights.
    assert result.returncode == 2
    assert "remote-only" in result.stderr
    assert not output.exists()
