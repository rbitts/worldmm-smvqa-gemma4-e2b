from __future__ import annotations

import math
from pathlib import Path

import pytest

from worldmm_smvqa.schema import FrameMetadata, SourceStreamExample
from worldmm_smvqa.sensor_frames import build_sensor_frame_manifest
from worldmm_smvqa.worldmm.spatial_teacher_targets import (
    TeacherObjectTarget,
    compile_teacher_object_record,
)
from worldmm_smvqa.worldmm.typed_memory import (
    canonical_jsonl_bytes,
    validate_typed_memory_artifact,
)


def _target(**updates: object) -> TeacherObjectTarget:
    payload: dict[str, object] = {
        "memory_id": "teacher:obs-1:mug-1",
        "observation_id": "obs-1",
        "source_video_id": "video-1",
        "frame_ref": "frame-1.jpg",
        "local_frame_id": "room-1",
        "timestamp": 2.0,
        "observed_through_time": 2.0,
        "entity_id": "entity-mug-1",
        "instance_id": "mug-1",
        "semantic_label": "mug",
        "place_label": "kitchen counter",
        "points_m": ((0.0, 0.0, 0.0), (2.0, 4.0, 6.0)),
        "confidence": 0.8,
    }
    payload.update(updates)
    return TeacherObjectTarget.model_validate(payload)


def test_teacher_points_compile_to_evidence_bound_object_record() -> None:
    record = compile_teacher_object_record(
        _target(),
        min_extent_m=0.01,
        uncertainty_floor_m=0.02,
    )

    assert record.geometry.centroid == pytest.approx((1.0, 2.0, 3.0))
    assert record.geometry.extent == pytest.approx((2.0, 4.0, 6.0))
    assert record.geometry_uncertainty.covariance_xyz[0][1] == pytest.approx(2.0)
    assert record.geometry_uncertainty.standard_deviation_m == pytest.approx(
        math.sqrt(9.0004),
    )
    assert record.validity.start_time == 2.0
    assert record.provenance == "model_inferred"
    assert record.evidence_refs == ("frame-1.jpg",)
    assert record.place_label == "kitchen counter"


def test_single_teacher_point_uses_explicit_extent_and_uncertainty_floors() -> None:
    record = compile_teacher_object_record(
        _target(points_m=((1.0, 2.0, 3.0),)),
    )

    assert record.geometry.extent == pytest.approx((0.01, 0.01, 0.01))
    assert record.geometry_uncertainty.standard_deviation_m == pytest.approx(0.02)


def test_compiled_target_passes_contextual_frame_grounding(tmp_path: Path) -> None:
    source = SourceStreamExample(
        video_id="video-1",
        start_time=0.0,
        end_time=3.0,
        frame_metadata=(
            FrameMetadata(
                frame_ref="frame-1.jpg",
                timestamp=2.0,
                description="teacher input",
            ),
        ),
    )
    output = tmp_path / "typed-memory.jsonl"
    record = compile_teacher_object_record(_target())
    _ = output.write_bytes(canonical_jsonl_bytes(record))

    summary = validate_typed_memory_artifact(
        output,
        sources=(source,),
        sensor_records=build_sensor_frame_manifest((source,)),
    )

    assert summary.record_count == 1


@pytest.mark.parametrize(
    ("updates", "kwargs", "error"),
    [
        ({"observed_through_time": 1.5}, {}, "must equal timestamp"),
        ({}, {"min_extent_m": 0.0}, "min_extent_m must be positive"),
        (
            {},
            {"uncertainty_floor_m": float("nan")},
            "uncertainty_floor_m must be non-negative",
        ),
    ],
)
def test_teacher_target_compiler_fails_closed(
    updates: dict[str, object],
    kwargs: dict[str, float],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _ = compile_teacher_object_record(_target(**updates), **kwargs)
