from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from worldmm_smvqa.chunking import read_source_streams
from worldmm_smvqa.spatial_selector_train import (
    SelectorCounterfactualUtility,
    SelectorModelManifest,
    SelectorSplitAssignment,
    SelectorSplitManifest,
    SelectorTrainingError,
    SelectorUtilityCacheRecord,
    build_selector_training_rows,
    selector_model_manifest_path,
    train_selector_model,
    write_selector_model,
)
from worldmm_smvqa.worldmm.spatial_compression import (
    FEATURE_NAMES,
    SpatialTokenCandidate,
    build_spatial_token_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/tiny_smvqa"


def _serialized_bytes(candidate: SpatialTokenCandidate) -> int:
    return len(candidate.record.model_dump_json().encode("utf-8")) + 1


def _counterfactual(
    question_id: str,
    candidate: SpatialTokenCandidate,
    *,
    useful: bool,
) -> SelectorUtilityCacheRecord:
    return SelectorUtilityCacheRecord(
        question_id=question_id,
        candidate_id=candidate.record.memory_id,
        baseline_qa_loss=0.2,
        deleted_qa_loss=1.0 if useful else 0.2,
        baseline_qa_score=0.9,
        deleted_qa_score=0.3 if useful else 0.9,
        geometry_coverage_gain=0.2 if useful else 0.0,
        uncertainty_reduction=0.1 if useful else 0.0,
        pose_information_gain=0.1 if useful else 0.0,
        surprise=0.1 if useful else 0.0,
        redundancy=0.0 if useful else 0.5,
        actual_serialized_bytes=_serialized_bytes(candidate),
    )


def _write_contract(tmp_path: Path) -> tuple[Path, Path]:
    candidates = build_spatial_token_candidates(
        read_source_streams(FIXTURE),
    ).candidates
    by_id = {candidate.record.memory_id: candidate for candidate in candidates}
    selected = (
        (
            "q_fake_001",
            next(value for key, value in by_id.items() if ":object:mug:" in key),
            True,
        ),
        (
            "q_fake_001",
            next(
                value
                for key, value in by_id.items()
                if "fake_video_001:zone:" in key and value.record.end_time <= 45.0
            ),
            False,
        ),
        (
            "q_fake_003",
            next(value for key, value in by_id.items() if ":object:box:" in key),
            True,
        ),
        (
            "q_fake_003",
            next(
                value
                for key, value in by_id.items()
                if "fake_video_002:zone:" in key and value.record.end_time <= 60.0
            ),
            False,
        ),
    )
    utility_cache = tmp_path / "utility.jsonl"
    _ = utility_cache.write_text(
        "".join(
            f"{_counterfactual(question, candidate, useful=useful).model_dump_json()}\n"
            for question, candidate, useful in selected
        ),
        encoding="utf-8",
    )
    split_manifest = tmp_path / "split.json"
    manifest = SelectorSplitManifest(
        assignments=(
            SelectorSplitAssignment(
                question_id="q_fake_001",
                participant_id="participant_train",
                session_id="session_train",
                split="train",
            ),
            SelectorSplitAssignment(
                question_id="q_fake_003",
                participant_id="participant_validation",
                session_id="session_validation",
                split="validation",
            ),
        ),
    )
    _ = split_manifest.write_text(manifest.model_dump_json(), encoding="utf-8")
    return utility_cache, split_manifest


def test_counterfactual_cache_builds_disjoint_selector_rows(tmp_path: Path) -> None:
    # Given: cached deletion outcomes and an explicit participant split.
    utility_cache, split_manifest = _write_contract(tmp_path)

    # When: selector supervision is prepared.
    rows = build_selector_training_rows(
        FIXTURE,
        utility_cache=utility_cache,
        split_manifest=split_manifest,
    )

    # Then: labels come from signed utility, not evidence-time overlap.
    assert {row.label for row in rows} == {0, 1}
    assert {row.split for row in rows} == {"train", "validation"}
    assert {row.question_id for row in rows} == {"q_fake_001", "q_fake_003"}
    assert all(len(row.features) == len(FEATURE_NAMES) for row in rows)
    assert all(row.utility is not None for row in rows)
    assert len({row.split_manifest_sha256 for row in rows}) == 1
    assert len({row.utility_cache_sha256 for row in rows}) == 1


def test_default_supervision_fails_closed_without_cached_utility() -> None:
    # Given / When / Then: evidence overlap is never the implicit fallback.
    with pytest.raises(SelectorTrainingError, match="split manifest required"):
        _ = build_selector_training_rows(FIXTURE)


def test_evidence_overlap_requires_explicit_legacy_mode(tmp_path: Path) -> None:
    # Given: only a split manifest; no counterfactual cache.
    _, split_manifest = _write_contract(tmp_path)

    # When: the caller explicitly requests the old supervision contract.
    rows = build_selector_training_rows(
        FIXTURE,
        split_manifest=split_manifest,
        supervision_mode="legacy-evidence-overlap",
    )

    # Then: every row and later model manifest can expose the weaker source.
    assert rows
    assert {row.supervision_mode for row in rows} == {"legacy-evidence-overlap"}
    assert all(row.utility is None for row in rows)


def test_actual_bytes_are_verified_and_reduce_utility(tmp_path: Path) -> None:
    # Given: equal gross benefit at two serialized sizes.
    small = SelectorCounterfactualUtility(
        baseline_qa_loss=0.0,
        deleted_qa_loss=1.0,
        baseline_qa_score=1.0,
        deleted_qa_score=0.0,
        geometry_coverage_gain=0.0,
        uncertainty_reduction=0.0,
        pose_information_gain=0.0,
        surprise=0.0,
        redundancy=0.0,
        actual_serialized_bytes=256,
    )
    large = small.model_copy(update={"actual_serialized_bytes": 1024})
    assert small.value_per_kib == pytest.approx(4 * large.value_per_kib)
    assert small.target_probability > large.target_probability

    # Cached byte counts must also match the actual candidate JSONL record.
    utility_cache, split_manifest = _write_contract(tmp_path)
    records = [
        SelectorUtilityCacheRecord.model_validate_json(line)
        for line in utility_cache.read_text().splitlines()
    ]
    records[0] = records[0].model_copy(
        update={
            "actual_serialized_bytes": records[0].actual_serialized_bytes + 1,
        },
    )
    _ = utility_cache.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(SelectorTrainingError, match="actual_serialized_bytes"):
        _ = build_selector_training_rows(
            FIXTURE,
            utility_cache=utility_cache,
            split_manifest=split_manifest,
        )


def test_linear_selector_reports_held_out_metrics_and_manifest(
    tmp_path: Path,
) -> None:
    # Given: counterfactual rows with disjoint train and validation sessions.
    utility_cache, split_manifest = _write_contract(tmp_path)
    rows = build_selector_training_rows(
        FIXTURE,
        utility_cache=utility_cache,
        split_manifest=split_manifest,
    )

    # When: the tiny linear scorer is fitted and exported.
    result = train_selector_model(rows, epochs=100, learning_rate=2.0)
    output = tmp_path / "selector.json"
    write_selector_model(result, output)

    # Then: training and held-out metrics stay separate and sources are hashed.
    assert result.rows == len(rows)
    assert result.training_rows == 2
    assert result.validation_rows == 2
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.validation_accuracy <= 1.0
    manifest = SelectorModelManifest.model_validate_json(
        selector_model_manifest_path(output).read_text(encoding="utf-8"),
    )
    assert manifest.supervision_mode == "counterfactual"
    assert manifest.utility_cache_sha256 == rows[0].utility_cache_sha256
    assert manifest.split_manifest_sha256 == rows[0].split_manifest_sha256


def test_selector_rejects_participant_or_video_split_leakage(tmp_path: Path) -> None:
    # Given: otherwise valid rows whose validation group impersonates train.
    utility_cache, split_manifest = _write_contract(tmp_path)
    base_rows = list(
        build_selector_training_rows(
            FIXTURE,
            utility_cache=utility_cache,
            split_manifest=split_manifest,
        ),
    )
    train_row = next(row for row in base_rows if row.split == "train")
    validation_index = next(
        index for index, row in enumerate(base_rows) if row.split == "validation"
    )
    rows = list(base_rows)
    rows[validation_index] = rows[validation_index].model_copy(
        update={"participant_id": train_row.participant_id},
    )

    # When / Then: training rejects cross-split identity leakage.
    with pytest.raises(SelectorTrainingError, match=r"participant.*crosses"):
        _ = train_selector_model(rows)

    rows = list(base_rows)
    rows[validation_index] = rows[validation_index].model_copy(
        update={"video_id": train_row.video_id},
    )
    with pytest.raises(SelectorTrainingError, match=r"video.*crosses"):
        _ = train_selector_model(rows)


def test_selector_accepts_projected_features_from_future_encoders(
    tmp_path: Path,
) -> None:
    # Given: a projection head adds one teacher-derived feature.
    utility_cache, split_manifest = _write_contract(tmp_path)
    rows = tuple(
        row.model_copy(
            update={
                "features": {
                    **row.features,
                    "cut3r_latent_0": float(row.label),
                },
            },
        )
        for row in build_selector_training_rows(
            FIXTURE,
            utility_cache=utility_cache,
            split_manifest=split_manifest,
        )
    )

    # When: the same selector trainer fits the expanded feature schema.
    result = train_selector_model(rows, epochs=100)

    # Then: no trainer framework change is needed for the new dimension.
    assert "cut3r_latent_0" in result.model.feature_names


def test_selector_prepare_cli_writes_counterfactual_jsonl(tmp_path: Path) -> None:
    # Given: a tiny cache, split manifest, and output path.
    utility_cache, split_manifest = _write_contract(tmp_path)
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
            "--utility-cache",
            str(utility_cache),
            "--split-manifest",
            str(split_manifest),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert '"supervision_mode": "counterfactual"' in result.stdout


def test_selector_train_cli_is_remote_only(tmp_path: Path) -> None:
    # Given: prepared counterfactual rows but no remote-host approval.
    utility_cache, split_manifest = _write_contract(tmp_path)
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
            "--utility-cache",
            str(utility_cache),
            "--split-manifest",
            str(split_manifest),
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

    assert result.returncode == 2
    assert "remote-only" in result.stderr
    assert not output.exists()
