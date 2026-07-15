from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldmm_smvqa.spatial_train import TeacherCacheDataset
from worldmm_smvqa.teacher_materializer import (
    TeacherMaterializationError,
    main,
    materialize_teacher_cache,
)
from worldmm_smvqa.worldmm.gcut3r_teacher import (
    EMPTY_PREFIX_SHA256,
    TeacherRequest,
    TeacherResponse,
    build_teacher_cache_record,
    write_teacher_cache,
)
from worldmm_smvqa.worldmm.typed_memory import (
    NoWriteMemoryRecord,
    ObjectGeometry,
    ObjectMemoryRecord,
    SpatialUncertainty,
    ValidityInterval,
    serialized_byte_cost,
)


def _uncertainty() -> SpatialUncertainty:
    return SpatialUncertainty(
        covariance_xyz=((0.04, 0.0, 0.0), (0.0, 0.04, 0.0), (0.0, 0.0, 0.04)),
        standard_deviation_m=0.2,
    )


def _teacher_cache(path: Path) -> tuple[ObjectMemoryRecord, NoWriteMemoryRecord]:
    object_record = ObjectMemoryRecord(
        memory_id="memory-object",
        source_video_id="video-1",
        entity_id="cup",
        instance_id="cup-1",
        local_frame_id="room-1",
        geometry_uncertainty=_uncertainty(),
        validity=ValidityInterval(start_time=1.0, end_time=1.0),
        first_seen_time=1.0,
        last_seen_time=1.0,
        observation_count=1,
        confidence=0.9,
        provenance="observed",
        evidence_refs=("frame-1.jpg",),
        geometry=ObjectGeometry(
            centroid=(1.0, 2.0, 0.5),
            extent=(0.1, 0.1, 0.2),
        ),
        semantic_label="cup",
    )
    no_write = NoWriteMemoryRecord(
        memory_id="memory-no-write",
        source_video_id="video-1",
        entity_id="candidate",
        instance_id="candidate-1",
        local_frame_id="room-1",
        geometry_uncertainty=_uncertainty(),
        validity=ValidityInterval(start_time=2.0, end_time=2.0),
        first_seen_time=2.0,
        last_seen_time=2.0,
        observation_count=1,
        confidence=0.2,
        provenance="model_inferred",
        candidate_type="landmark",
        reason="below threshold",
    )
    first_request = TeacherRequest(
        observation_id="obs-1",
        video_id="video-1",
        timestamp=1.0,
        frame_ref="frame-1.jpg",
        local_frame_id="room-1",
        sequence_index=0,
        prefix_before_sha256=EMPTY_PREFIX_SHA256,
    )
    first_response = TeacherResponse(
        observation_id="obs-1",
        video_id="video-1",
        timestamp=1.0,
        observed_through_time=1.0,
        state_ref="state-1",
        records=(object_record,),
    )
    first = build_teacher_cache_record(
        teacher_backend="gcut3r_external",
        provider_id="test-provider",
        request=first_request,
        response=first_response,
    )
    second_request = TeacherRequest(
        observation_id="obs-2",
        video_id="video-1",
        timestamp=2.0,
        frame_ref="frame-2.jpg",
        local_frame_id="room-1",
        sequence_index=1,
        previous_state_ref="state-1",
        prefix_before_sha256=first.prefix_sha256,
    )
    second_response = TeacherResponse(
        observation_id="obs-2",
        video_id="video-1",
        timestamp=2.0,
        observed_through_time=2.0,
        state_ref="state-2",
        records=(no_write,),
    )
    second = build_teacher_cache_record(
        teacher_backend="gcut3r_external",
        provider_id="test-provider",
        request=second_request,
        response=second_response,
    )
    write_teacher_cache(path, (first, second))
    return object_record, no_write


def _supervision_rows() -> list[dict[str, object]]:
    return [
        {
            "observation_id": "obs-2",
            "memory_id": "memory-no-write",
            "group_id": "participant-validation",
            "split": "validation",
            "features": [0.3, 0.4],
            "teacher_embedding": [0.5, 0.6, 0.7],
            "geometry_target": [0.0, 0.0, 0.0, 0.0],
            "association_target": 0,
        },
        {
            "observation_id": "obs-1",
            "memory_id": "memory-object",
            "group_id": "participant-train",
            "split": "train",
            "features": [0.1, 0.2],
            "teacher_embedding": [0.8, 0.9, 1.0],
            "geometry_target": [1.0, 2.0, 0.5, 0.2],
            "association_target": 0,
        },
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    _ = path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def test_materializer_joins_in_causal_order_and_derives_trusted_targets(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "teacher-cache.jsonl"
    supervision = tmp_path / "supervision.jsonl"
    out = tmp_path / "student-cache.jsonl"
    object_record, _ = _teacher_cache(cache)
    _write_jsonl(supervision, _supervision_rows())

    rows = materialize_teacher_cache(cache, supervision, out)
    assert len(rows) == 2
    loaded = TeacherCacheDataset(out)
    encoded = tuple(json.loads(line) for line in out.read_text().splitlines())

    assert [row.sample_id for row in loaded.rows] == [
        "obs-1:memory-object",
        "obs-2:memory-no-write",
    ]
    assert encoded[0]["type_label"] == "object"
    assert encoded[0]["group_id"] == "participant-train"
    assert encoded[0]["uncertainty_target"] == 0.2
    assert encoded[0]["byte_cost"] == serialized_byte_cost(object_record)
    assert encoded[1]["type_label"] == "no_write"
    assert encoded[1]["byte_cost"] == 0
    assert not tuple(tmp_path.glob(".*.tmp"))
    with pytest.raises(SystemExit, match="2"):
        _ = main(
            [
                "--teacher-cache",
                str(cache),
                "--supervision",
                str(supervision),
                "--out",
                str(out),
            ]
        )


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing", "missing"),
        ("duplicate", "duplicate"),
        ("dimensions", "dimensions"),
        ("group_split", "crosses splits"),
        ("association", "contiguous"),
    ],
)
def test_materializer_rejects_incomplete_or_inconsistent_supervision(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    cache = tmp_path / "teacher-cache.jsonl"
    supervision = tmp_path / "supervision.jsonl"
    out = tmp_path / "student-cache.jsonl"
    _ = _teacher_cache(cache)
    rows = _supervision_rows()
    if failure == "missing":
        rows[0]["memory_id"] = "missing-memory"
    elif failure == "duplicate":
        rows[1] = dict(rows[0])
    elif failure == "dimensions":
        rows[0]["features"] = [0.3]
    elif failure == "association":
        rows[1]["association_target"] = 1
    else:
        rows[0]["group_id"] = rows[1]["group_id"]
    _write_jsonl(supervision, rows)

    with pytest.raises(TeacherMaterializationError, match=message):
        _ = materialize_teacher_cache(cache, supervision, out)

    assert not out.exists()
