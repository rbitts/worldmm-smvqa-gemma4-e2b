from __future__ import annotations

import pytest
from pydantic import ValidationError

from worldmm_smvqa.worldmm.geometry_executor import (
    GeometryEntityFact,
    GeometryQuery,
    execute_geometry,
    plan_geometry_query,
)
from worldmm_smvqa.worldmm.typed_memory import (
    ObjectGeometry,
    ObjectMemoryRecord,
    SpatialUncertainty,
    ValidityInterval,
)


def _entity(entity_id: str, x: float) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "label": "mug",
        "x": x,
        "y": 0.0,
        "z": 0.0,
        "coordinate_frame": "room:1",
        "uncertainty_m": 0.1,
        "provenance": "observed",
        "evidence_refs": [f"memory:{entity_id}"],
    }


def _typed_object(name: str, label: str, x: float) -> ObjectMemoryRecord:
    return ObjectMemoryRecord(
        memory_id=f"memory:{name}",
        source_video_id="video:1",
        entity_id=f"entity:{name}",
        instance_id=f"instance:{name}",
        local_frame_id="room:1",
        geometry_uncertainty=SpatialUncertainty(
            covariance_xyz=(
                (0.01, 0.0, 0.0),
                (0.0, 0.01, 0.0),
                (0.0, 0.0, 0.01),
            ),
            standard_deviation_m=0.1,
        ),
        validity=ValidityInterval(start_time=1.0, end_time=2.0),
        first_seen_time=1.0,
        last_seen_time=2.0,
        observation_count=1,
        confidence=0.9,
        provenance="observed",
        evidence_refs=(f"frame:{name}",),
        geometry=ObjectGeometry(
            centroid=(x, 0.0, 0.0),
            extent=(0.2, 0.2, 0.2),
        ),
        semantic_label=label,
    )


def test_count_requires_explicit_complete_entity_index() -> None:
    top_k_records = (_entity("mug:1", 0.0), _entity("mug:2", 1.0))
    incomplete_query = GeometryQuery(
        operation="count",
        coordinate_frame="room:1",
        entity_label="mug",
    )

    incomplete = execute_geometry(top_k_records, incomplete_query)
    complete = execute_geometry(
        top_k_records,
        incomplete_query.model_copy(update={"entity_index_complete": True}),
    )

    assert not incomplete.answerable
    assert incomplete.value is None
    assert incomplete.reason == "count requires a complete entity index"
    assert complete.answerable
    assert complete.value == 2


def test_last_seen_requires_explicit_complete_entity_index() -> None:
    records = ({**_entity("mug:1", 0.0), "last_seen_time": 2.0},)
    incomplete_query = GeometryQuery(
        operation="last_seen",
        coordinate_frame="video-time",
        subject="mug:1",
    )

    incomplete = execute_geometry(records, incomplete_query)
    complete = execute_geometry(
        records,
        incomplete_query.model_copy(update={"entity_index_complete": True}),
    )

    assert not incomplete.answerable
    assert incomplete.reason == "last-seen requires a complete entity index"
    assert complete.answerable
    assert complete.value == 2.0


def test_production_planner_never_certifies_count_completeness() -> None:
    query = plan_geometry_query(
        "How many mugs were visible?",
        (_entity("mug:1", 0.0),),
        coordinate_frame="room:1",
    )

    assert query is not None
    assert not query.entity_index_complete


def test_proof_hash_covers_every_behavior_option() -> None:
    records = (_entity("mug:1", 0.0), _entity("mug:2", 1.0))
    baseline = GeometryQuery(
        operation="relative_direction",
        coordinate_frame="room:1",
        subject="mug:1",
        object="mug:2",
        wearer_yaw_degrees=0.0,
    )
    queries = (
        baseline,
        baseline.model_copy(update={"wearer_yaw_degrees": 1.0}),
        baseline.model_copy(update={"near_threshold_m": 2.0}),
        baseline.model_copy(update={"max_uncertainty_m": 0.6}),
        baseline.model_copy(update={"entity_index_complete": True}),
    )

    proof_ids = {execute_geometry(records, query).proof_id for query in queries}

    assert len(proof_ids) == len(queries)


def test_direct_nested_typed_objects_are_normalized() -> None:
    proof = execute_geometry(
        (
            _typed_object("mug", "mug", 0.0),
            _typed_object("notebook", "notebook", 3.0),
        ),
        GeometryQuery(
            operation="distance",
            coordinate_frame="room:1",
            subject="entity:mug",
            object="entity:notebook",
        ),
    )

    assert proof.answerable
    assert proof.value == pytest.approx(3.0)
    assert proof.uncertainty == pytest.approx(2**0.5 * 0.1)
    assert proof.evidence_refs == (
        "frame:mug",
        "frame:notebook",
        "memory:mug",
        "memory:notebook",
    )


def test_missing_coordinate_frame_never_matches_requested_frame() -> None:
    missing_frame = _entity("mug:1", 0.0)
    del missing_frame["coordinate_frame"]
    records = (missing_frame, _entity("desk:1", 1.0))

    distance = execute_geometry(
        records,
        GeometryQuery(
            operation="distance",
            coordinate_frame="room:1",
            subject="mug:1",
            object="desk:1",
        ),
    )
    count = execute_geometry(
        (missing_frame,),
        GeometryQuery(
            operation="count",
            coordinate_frame="room:1",
            entity_label="mug",
            entity_index_complete=True,
        ),
    )

    assert not distance.answerable
    assert distance.reason == "coordinate frame mismatch: mug:1"
    assert not count.answerable
    assert count.reason == "entity coordinate frames do not match query frame"


@pytest.mark.parametrize(
    "conflicting_update",
    [{"x": 2.0}, {"coordinate_frame": "room:2"}],
)
def test_conflicting_latest_entity_records_abstain_order_independently(
    conflicting_update: dict[str, object],
) -> None:
    first = {**_entity("mug:1", 0.0), "last_seen_time": 5.0}
    conflicting = {**first, **conflicting_update}
    desk = {**_entity("desk:1", 4.0), "label": "desk"}
    query = GeometryQuery(
        operation="distance",
        coordinate_frame="room:1",
        subject="mug:1",
        object="desk:1",
    )

    forward = execute_geometry((first, conflicting, desk), query)
    reversed_ = execute_geometry((conflicting, first, desk), query)

    assert not forward.answerable
    assert forward.reason == "conflicting latest records: mug:1"
    assert forward.subject_entity_id == "mug:1"
    assert forward.object_entity_id == "desk:1"
    assert reversed_.reason == forward.reason
    assert reversed_.proof_id == forward.proof_id


def test_time_uncertainty_must_be_finite() -> None:
    with pytest.raises(ValidationError, match="geometry values must be finite"):
        _ = GeometryEntityFact(
            entity_id="mug:1",
            label="mug",
            provenance="observed",
            evidence_refs=(),
            time_uncertainty_s=float("inf"),
        )
