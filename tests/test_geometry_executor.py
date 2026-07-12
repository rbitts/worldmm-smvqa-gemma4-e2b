from __future__ import annotations

import math

import pytest

from worldmm_smvqa.worldmm.geometry_executor import (
    GeometryQuery,
    execute_geometry,
    plan_geometry_query,
)


def entity(  # noqa: PLR0913
    entity_id: str,
    label: str,
    x: float,
    y: float,
    *,
    uncertainty_m: float = 0.05,
    last_seen_time: float = 1.0,
    provenance: str = "object_geometry",
) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "label": label,
        "x": x,
        "y": y,
        "z": 0.0,
        "coordinate_frame": "room:1",
        "uncertainty_m": uncertainty_m,
        "provenance": provenance,
        "evidence_refs": [f"memory:{entity_id}"],
        "last_seen_time": last_seen_time,
    }


def test_distance_returns_stable_auditable_proof() -> None:
    records = (entity("mug:1", "mug", 0.0, 0.0), entity("desk:1", "desk", 3.0, 4.0))
    query = GeometryQuery(
        operation="distance",
        coordinate_frame="room:1",
        subject="mug:1",
        object="desk:1",
    )

    proof = execute_geometry(records, query)

    assert proof.answerable
    assert proof.value == pytest.approx(5.0)
    assert proof.uncertainty == pytest.approx(math.hypot(0.05, 0.05))
    assert proof.entity_ids == ("desk:1", "mug:1")
    assert proof.subject_entity_id == "mug:1"
    assert proof.object_entity_id == "desk:1"
    assert proof.provenance == ("object_geometry",)
    assert proof.evidence_refs == ("memory:desk:1", "memory:mug:1")
    assert proof.proof_id == execute_geometry(records, query).proof_id

    reversed_proof = execute_geometry(
        records,
        query.model_copy(update={"subject": "desk:1", "object": "mug:1"}),
    )
    assert reversed_proof.entity_ids == proof.entity_ids
    assert reversed_proof.subject_entity_id == "desk:1"
    assert reversed_proof.object_entity_id == "mug:1"
    assert reversed_proof.proof_id != proof.proof_id


@pytest.mark.parametrize(
    ("x", "y", "yaw", "expected"),
    [
        (-2.0, 0.0, 0.0, "left"),
        (2.0, 0.0, 0.0, "right"),
        (0.0, 2.0, 0.0, "front"),
        (0.0, -2.0, 0.0, "behind"),
        (2.0, 0.0, 90.0, "front"),
    ],
)
def test_relative_direction_uses_wearer_yaw(
    x: float,
    y: float,
    yaw: float,
    expected: str,
) -> None:
    proof = execute_geometry(
        (entity("target:1", "target", x, y), entity("origin:1", "origin", 0, 0)),
        GeometryQuery(
            operation="relative_direction",
            coordinate_frame="room:1",
            subject="target:1",
            object="origin:1",
            wearer_yaw_degrees=yaw,
        ),
    )

    assert proof.value == expected


def test_near_last_seen_and_count_use_grounded_instances() -> None:
    records = (
        entity("mug:1", "mug", 0.0, 0.0, last_seen_time=2.0),
        entity("mug:1", "mug", 0.1, 0.0, last_seen_time=8.0),
        entity("mug:2", "mug", 0.5, 0.0, last_seen_time=3.0),
    )

    near = execute_geometry(
        records,
        GeometryQuery(
            operation="near",
            coordinate_frame="room:1",
            subject="mug:1",
            object="mug:2",
            near_threshold_m=1.0,
        ),
    )
    last_seen = execute_geometry(
        records,
        GeometryQuery(
            operation="last_seen",
            coordinate_frame="video-time",
            subject="mug:1",
            entity_index_complete=True,
        ),
    )
    count = execute_geometry(
        records,
        GeometryQuery(
            operation="count",
            coordinate_frame="room:1",
            entity_label="mug",
            entity_index_complete=True,
        ),
    )

    assert near.value is True
    assert last_seen.value == 8.0
    assert last_seen.uncertainty_unit == "seconds"
    assert count.value == 2


@pytest.mark.parametrize(
    "record",
    [
        entity("mug:1", "mug", 0, 0, provenance="pose"),
        entity("mug:1", "mug", 0, 0, uncertainty_m=2.0),
    ],
)
def test_metric_operation_abstains_on_non_grounded_or_uncertain_input(
    record: dict[str, object],
) -> None:
    proof = execute_geometry(
        (record, entity("desk:1", "desk", 2, 0)),
        GeometryQuery(
            operation="distance",
            coordinate_frame="room:1",
            subject="mug:1",
            object="desk:1",
        ),
    )

    assert not proof.answerable
    assert proof.value is None
    assert proof.reason is not None


def test_typed_record_dict_uses_instance_id_and_observed_provenance() -> None:
    proof = execute_geometry(
        (
            {
                "record_type": "object",
                "instance_id": "cup:7",
                "object_label": "cup",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "coordinate_frame": "submap:2",
                "uncertainty_m": 0.1,
                "provenance": "observed",
                "memory_id": "memory:cup:7",
            },
            {
                "instance_id": "table:3",
                "object_label": "table",
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
                "coordinate_frame": "submap:2",
                "uncertainty_m": 0.1,
                "provenance": "observed",
                "memory_id": "memory:table:3",
            },
        ),
        GeometryQuery(
            operation="distance",
            coordinate_frame="submap:2",
            subject="cup:7",
            object="table:3",
        ),
    )

    assert proof.value == pytest.approx(1.0)
    assert proof.entity_ids == ("cup:7", "table:3")


def test_keyword_planner_requires_unambiguous_mentioned_entities() -> None:
    records = (
        entity("mug:1", "mug", 0, 0),
        entity("desk:1", "desk", 1, 0),
    )

    query = plan_geometry_query(
        "How far is mug:1 from desk:1?",
        records,
        coordinate_frame="room:1",
    )
    uncertified_labels = plan_geometry_query(
        "How far is the mug from the desk?",
        records,
        coordinate_frame="room:1",
    )
    certified_ambiguous = plan_geometry_query(
        "How far is the mug from the desk?",
        (*records, entity("mug:2", "mug", 2, 0)),
        coordinate_frame="room:1",
        entity_index_complete=True,
    )

    assert query == GeometryQuery(
        operation="distance",
        coordinate_frame="room:1",
        subject="mug:1",
        object="desk:1",
    )
    assert uncertified_labels is None
    assert certified_ambiguous is None
