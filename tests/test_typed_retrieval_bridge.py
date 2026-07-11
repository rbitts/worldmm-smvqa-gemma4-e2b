from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldmm_smvqa.retrieval import (
    InvalidRetrievalStoreError,
    RetrievalOptions,
    build_retrieval_records,
    read_retrieval_memory_artifacts,
    retrieve_evidence,
)
from worldmm_smvqa.retrieval_types import RetrievalMemoryRecord
from worldmm_smvqa.schema import QuestionRequest
from worldmm_smvqa.worldmm.geometry_executor import geometry_proofs_for_question
from worldmm_smvqa.worldmm.typed_memory import (
    EventGeometry,
    EventMemoryRecord,
    FreeSpaceGeometry,
    FreeSpaceMemoryRecord,
    LandmarkGeometry,
    LandmarkMemoryRecord,
    NoWriteMemoryRecord,
    ObjectGeometry,
    ObjectMemoryRecord,
    PlaneGeometry,
    PlaneMemoryRecord,
    PortalGeometry,
    PortalMemoryRecord,
    SpatialUncertainty,
    TypedMemoryRecordBase,
    ValidityInterval,
    canonical_jsonl_bytes,
)


def _base(
    name: str,
    *,
    start_time: float = 1.0,
    end_time: float = 2.0,
) -> dict[str, object]:
    return {
        "memory_id": f"memory-{name}",
        "source_video_id": "video-1",
        "entity_id": f"entity-{name}",
        "instance_id": f"instance-{name}",
        "local_frame_id": "room-1",
        "geometry_uncertainty": SpatialUncertainty(
            covariance_xyz=(
                (0.01, 0.0, 0.0),
                (0.0, 0.01, 0.0),
                (0.0, 0.0, 0.01),
            ),
            standard_deviation_m=0.1,
        ),
        "validity": ValidityInterval(
            start_time=start_time,
            end_time=end_time,
        ),
        "first_seen_time": start_time,
        "last_seen_time": end_time,
        "observation_count": 2,
        "confidence": 0.9,
        "provenance": "observed",
        "evidence_refs": (f"frame-{name}",),
    }


def _object(
    name: str,
    label: str,
    centroid: tuple[float, float, float],
    *,
    start_time: float = 1.0,
    end_time: float = 2.0,
) -> ObjectMemoryRecord:
    return ObjectMemoryRecord.model_validate(
        {
            **_base(name, start_time=start_time, end_time=end_time),
            "geometry": ObjectGeometry(
                centroid=centroid,
                extent=(0.2, 0.2, 0.2),
            ),
            "semantic_label": label,
        },
    )


def _typed_records() -> tuple[TypedMemoryRecordBase, ...]:
    return (
        _object("object", "mug", (1.0, 2.0, 3.0)),
        PlaneMemoryRecord.model_validate(
            {
                **_base("plane"),
                "geometry": PlaneGeometry(
                    normal=(0.0, 0.0, 1.0),
                    offset_m=0.0,
                    boundary=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
                ),
            },
        ),
        PortalMemoryRecord.model_validate(
            {
                **_base("portal"),
                "geometry": PortalGeometry(
                    centroid=(0.0, 0.0, 1.0),
                    normal=(1.0, 0.0, 0.0),
                    width_m=1.0,
                    height_m=2.0,
                ),
                "connects_frame_ids": ("room-1", "room-2"),
            },
        ),
        FreeSpaceMemoryRecord.model_validate(
            {
                **_base("free-space"),
                "geometry": FreeSpaceGeometry(
                    floor_polygon=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
                    height_m=2.0,
                ),
            },
        ),
        LandmarkMemoryRecord.model_validate(
            {
                **_base("landmark"),
                "geometry": LandmarkGeometry(
                    position=(1.0, 2.0, 3.0),
                    ray_direction=(0.0, 0.0, 1.0),
                    view_cone_degrees=30.0,
                ),
                "descriptor_ref": "descriptor-1",
            },
        ),
        EventMemoryRecord.model_validate(
            {
                **_base("event"),
                "geometry": EventGeometry(
                    before_position=(0.0, 0.0, 0.0),
                    after_position=(1.0, 0.0, 0.0),
                ),
                "event_kind": "moved",
                "involved_entity_ids": ("entity-object",),
            },
        ),
        NoWriteMemoryRecord.model_validate(
            {
                **_base("no-write"),
                "candidate_type": "object",
                "reason": "duplicate observation",
            },
        ),
    )


def _manifest(tmp_path: Path, spatial_payload: bytes) -> Path:
    empty = tmp_path / "empty.jsonl"
    _ = empty.write_text("", encoding="utf-8")
    spatial = tmp_path / "spatial.jsonl"
    _ = spatial.write_bytes(spatial_payload)
    manifest = tmp_path / "manifest.json"
    _ = manifest.write_text(
        json.dumps(
            {
                "episodic_memory": str(empty),
                "semantic_memory": str(empty),
                "visual_memory": str(empty),
                "spatial_memory": {"path": str(spatial)},
            },
        ),
        encoding="utf-8",
    )
    return manifest


def test_flat_typed_jsonl_is_retrieval_ready_and_no_write_is_skipped(
    tmp_path: Path,
) -> None:
    records = _typed_records()
    manifest = _manifest(
        tmp_path,
        b"".join(canonical_jsonl_bytes(record) for record in records),
    )

    loaded = read_retrieval_memory_artifacts(manifest)

    assert tuple(record.memory_id for record in loaded) == tuple(
        record.memory_id for record in records[:-1]
    )
    assert {record.geometry["record_type"] for record in loaded if record.geometry} == {
        "object",
        "plane",
        "portal",
        "free_space",
        "landmark",
        "event",
    }
    object_record = loaded[0]
    assert object_record.video_id == "video-1"
    assert (object_record.start_time, object_record.end_time) == (1.0, 2.0)
    assert object_record.frame_refs == ("frame-object",)
    assert object_record.geometry == {
        "record_type": "object",
        "entity_id": "entity-object",
        "instance_id": "instance-object",
        "label": "mug",
        "coordinate_frame": "room-1",
        "uncertainty_m": 0.1,
        "last_seen_time": 2.0,
        "provenance": "observed",
        "evidence_refs": "frame-object",
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "extent_x": 0.2,
        "extent_y": 0.2,
        "extent_z": 0.2,
    }


def test_unknown_typed_record_fails_closed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, b'{"record_type":"future_primitive"}\n')

    with pytest.raises(InvalidRetrievalStoreError, match="future_primitive"):
        _ = read_retrieval_memory_artifacts(manifest)


def test_two_typed_objects_form_geometry_bundle_without_stored_relation() -> None:
    mug = _object("mug", "mug", (0.0, 0.0, 0.0))
    notebook = _object("notebook", "notebook", (3.0, 4.0, 0.0))
    future = _object(
        "future",
        "spoon",
        (9.0, 9.0, 9.0),
        start_time=3.0,
        end_time=4.0,
    )
    memories = build_retrieval_records((), (), (), (mug, notebook, future))
    question = QuestionRequest(
        question_id="q-distance",
        video_id="video-1",
        question="How far was the mug from the notebook?",
        question_time=2.5,
        answer_choices=(),
    )

    pack = retrieve_evidence(
        question,
        memories,
        enabled_stores=frozenset({"spatial"}),
        options=RetrievalOptions(evidence_budget=2),
    )
    proofs = geometry_proofs_for_question(
        question,
        pack,
        coordinate_frame="room-1",
    )

    assert {item.memory_id for item in pack.evidence} == {
        "memory-mug",
        "memory-notebook",
    }
    assert pack.causal_filtered_count == 1
    assert len(proofs) == 1
    assert proofs[0].answerable
    assert proofs[0].value == pytest.approx(5.0)
    assert proofs[0].entity_ids == ("entity-mug", "entity-notebook")
    assert proofs[0].evidence_refs == (
        "frame-mug",
        "frame-notebook",
        "memory-mug",
        "memory-notebook",
    )


def test_existing_relation_bundle_stays_relation_first() -> None:
    mug = _object("mug", "mug", (0.0, 0.0, 0.0))
    notebook = _object("notebook", "notebook", (1.0, 0.0, 0.0))
    typed = build_retrieval_records((), (), (), (mug, notebook))
    relation = RetrievalMemoryRecord(
        memory_id="relation-mug-notebook",
        source_store="spatial",
        video_id="video-1",
        start_time=1.0,
        end_time=2.0,
        snippet="mug near notebook distance_m=1",
        frame_refs=(),
        base_score=1.0,
        geometry={
            "relation": "near",
            "distance_m": 1.0,
            "subject_instance_id": "instance-mug",
            "object_instance_id": "instance-notebook",
        },
    )
    question = QuestionRequest(
        question_id="q-related",
        video_id="video-1",
        question="How far was the mug from the notebook?",
        question_time=3.0,
        answer_choices=(),
    )

    pack = retrieve_evidence(
        question,
        (*typed, relation),
        enabled_stores=frozenset({"spatial"}),
        options=RetrievalOptions(evidence_budget=3),
    )

    assert tuple(item.memory_id for item in pack.evidence) == (
        "relation-mug-notebook",
        "memory-mug",
        "memory-notebook",
    )
