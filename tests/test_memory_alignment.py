from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import ValidationError

from worldmm_smvqa.memory_alignment import (
    AlignmentValidationError,
    IntervalLabelV1,
    MemoryAlignmentReportV1,
    MemoryCoverageManifestV1,
    MemoryProjectionV1,
    MemoryRequestV2,
    VisualPointLabelV1,
    atomic_write_report_no_clobber,
    coverage_report_row,
    paired_bootstrap_comparison,
    parse_evaluator_microseconds,
    rank_at_6,
    recall_at_6,
    validate_relative_path,
)


def _request(**updates: object) -> MemoryRequestV2:
    values: dict[str, object] = {
        "request_id": "q1",
        "store": "visual",
        "video_id": "video-1",
        "query_text": "What is on the red table?",
        "query_time_us": "100",
        "expected_evidence_ids": ["e1"],
    }
    values.update(updates)
    return MemoryRequestV2.model_validate(values)


def _point(**updates: object) -> MemoryProjectionV1:
    values: dict[str, object] = {
        "projection_id": "p1",
        "projection_kind": "visual_point_projection",
        "role": "baseline",
        "store_kind": "visual",
        "source_id": "source-1",
        "native_id": "native-1",
        "video_id": "video-1",
        "snippet": "red table",
        "base_score_ppm": "1000000",
        "frame_ref": "frame-1",
        "timestamp_us": "90",
    }
    values.update(updates)
    return MemoryProjectionV1.model_validate(values)


@pytest.mark.parametrize(
    "value",
    [1, True, None, "", "00", "01", "+1", "-1", "1.0", " 1", "9223372036854775808"],
)
def test_evaluator_microseconds_reject_noncanonical_values(value: object) -> None:
    with pytest.raises(ValueError):
        parse_evaluator_microseconds(value)


def test_relative_paths_reject_escape_and_links_are_not_normalized() -> None:
    for value in ("", "/absolute", "../escape", "a/../b", "a\\b", "./a"):
        with pytest.raises(ValueError):
            validate_relative_path(value)
    assert validate_relative_path("stores/visual.jsonl").as_posix() == "stores/visual.jsonl"


def test_rank_at_6_is_store_isolated_causal_exact_and_stably_ordered() -> None:
    request = _request()
    candidates = [
        _point(projection_id="other-role", role="candidate"),
        _point(projection_id="future", timestamp_us="101"),
        _point(projection_id="other-video", video_id="video-2"),
        *[
            _point(
                projection_id=f"p{index}",
                source_id=f"s{index}",
                native_id=f"native-{index:02d}",
                timestamp_us=str(90 - index),
            )
            for index in range(8)
        ],
    ]

    ranked = rank_at_6(request, candidates, role="baseline", store_kind="visual")

    assert len(ranked) == 6
    assert [item.native_id for item in ranked] == [
        "native-00",
        "native-01",
        "native-02",
        "native-03",
        "native-04",
        "native-05",
    ]


def test_recall_at_6_counts_each_expected_label_once() -> None:
    request = _request(expected_evidence_ids=["e1", "e2"])
    labels = {
        "e1": VisualPointLabelV1(
            label_kind="visual_point_label",
            evidence_id="e1",
            store="visual",
            video_id="video-1",
            frame_ref="frame-1",
            timestamp_us="90",
        ),
        "e2": VisualPointLabelV1(
            label_kind="visual_point_label",
            evidence_id="e2",
            store="visual",
            video_id="video-1",
            frame_ref="frame-2",
            timestamp_us="80",
        ),
    }
    duplicate_matches = [_point(), _point(projection_id="p2", native_id="native-2")]

    assert recall_at_6(request, labels, duplicate_matches, logical_store="visual") == Fraction(
        1, 2
    )


def test_interval_match_is_strict_half_open() -> None:
    request = _request(
        store="semantic",
        expected_evidence_ids=["e1"],
    )
    label = IntervalLabelV1(
        label_kind="interval_label",
        evidence_id="e1",
        store="semantic",
        video_id="video-1",
        start_us="20",
        end_us="30",
    )
    touching = MemoryProjectionV1(
        projection_id="p1",
        projection_kind="interval_projection",
        role="candidate",
        store_kind="semantic",
        source_id="source-1",
        native_id="native-1",
        video_id="video-1",
        snippet="red table",
        base_score_ppm="1",
        start_us="10",
        end_us="20",
    )

    assert recall_at_6(request, {"e1": label}, [touching], logical_store="semantic") == 0


def test_paired_bootstrap_is_deterministic_and_exact() -> None:
    pairs = [("q2", Fraction(1), Fraction(1)), ("q1", Fraction(0), Fraction(1))]

    first = paired_bootstrap_comparison("visual_primary", pairs)
    second = paired_bootstrap_comparison("visual_primary", tuple(reversed(pairs)))

    assert first == second
    assert first.baseline_mean_recall_at_6 == "1/2"
    assert first.candidate_mean_recall_at_6 == "1/1"
    assert first.mean_delta == "1/2"
    assert first.decision == "pass"
    assert first.accepted_random_block_count == "20000"


def test_paired_bootstrap_fails_below_noninferiority_bound() -> None:
    row = paired_bootstrap_comparison(
        "semantic_primary", [("q1", Fraction(1), Fraction(0))]
    )

    assert row.ci95_low_delta == "-1/1"
    assert row.decision == "fail"
    assert row.scientific_failure == "ci_lower_below_threshold"


def test_zero_pair_count_has_public_validation_code() -> None:
    with pytest.raises(AlignmentValidationError) as caught:
        paired_bootstrap_comparison("episodic_primary", [])
    assert caught.value.code == "zero_pair_count"


def test_coverage_is_scientific_failure_not_validation_failure() -> None:
    manifest = MemoryCoverageManifestV1(
        schema_version="memory-coverage-manifest-v1",
        store_kind="visual",
        store_path="stores/visual.jsonl",
        source_manifest_file_sha256="1" * 64,
        store_file_sha256="2" * 64,
        expected_count="2",
        attempted_count="2",
        schema_valid_count="2",
        written_count="1",
        coverage_basis_points="5000",
        producer_assertion="trusted_external_seal",
    )

    row = coverage_report_row("candidate", manifest)

    assert row.decision == "fail"
    assert row.scientific_failure == "coverage_below_100_percent"


def test_public_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _ = _request(unknown="forbidden")


def test_report_write_is_canonical_and_never_clobbers(tmp_path: Path) -> None:
    report = MemoryAlignmentReportV1(
        status="validation_fail",
        baseline_bundle_sha256=None,
        candidate_bundle_sha256=None,
        cohort_sha256=None,
        coverage=(),
        comparisons=(),
        validation_error="contract_invalid",
    )
    target = tmp_path / "report.json"

    atomic_write_report_no_clobber(target, report)
    first = target.read_bytes()
    assert json.loads(first)["schema_version"] == "memory-alignment-report-v1"
    assert first.endswith(b"\n")

    with pytest.raises(FileExistsError):
        atomic_write_report_no_clobber(target, report)
    assert target.read_bytes() == first
    assert list(tmp_path.iterdir()) == [target]
