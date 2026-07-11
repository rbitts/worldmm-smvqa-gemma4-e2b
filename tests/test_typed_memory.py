from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

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
    TypedMemoryRecord,
    ValidityInterval,
    canonical_jsonl_bytes,
    serialized_byte_cost,
)


def _common() -> dict[str, object]:
    return {
        "memory_id": "memory-1",
        "source_video_id": "video-1",
        "entity_id": "entity-1",
        "instance_id": "instance-1",
        "local_frame_id": "room-1",
        "geometry_uncertainty": SpatialUncertainty(
            covariance_xyz=((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.2)),
            standard_deviation_m=0.2,
        ),
        "validity": ValidityInterval(start_time=1.0, end_time=2.0),
        "first_seen_time": 1.0,
        "last_seen_time": 2.0,
        "observation_count": 2,
        "confidence": 0.9,
        "provenance": "observed",
        "evidence_refs": ("frame-1",),
    }


def test_typed_memory_union_accepts_every_record_type() -> None:
    records = (
        ObjectMemoryRecord.model_validate(
            {
                **_common(),
                "geometry": ObjectGeometry(centroid=(1, 2, 3), extent=(1, 1, 1)),
                "semantic_label": "mug",
            }
        ),
        PlaneMemoryRecord.model_validate(
            {
                **_common(),
                "geometry": PlaneGeometry(
                    normal=(0, 0, 1),
                    offset_m=0,
                    boundary=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
                ),
            }
        ),
        PortalMemoryRecord.model_validate(
            {
                **_common(),
                "geometry": PortalGeometry(
                    centroid=(0, 0, 1),
                    normal=(1, 0, 0),
                    width_m=1,
                    height_m=2,
                ),
                "connects_frame_ids": ("room-1", "room-2"),
            }
        ),
        FreeSpaceMemoryRecord.model_validate(
            {
                **_common(),
                "geometry": FreeSpaceGeometry(
                    floor_polygon=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
                    height_m=2,
                ),
            }
        ),
        LandmarkMemoryRecord.model_validate(
            {
                **_common(),
                "geometry": LandmarkGeometry(
                    position=(1, 2, 3),
                    ray_direction=(0, 0, 1),
                    view_cone_degrees=30,
                ),
                "descriptor_ref": "descriptor-1",
            }
        ),
        EventMemoryRecord.model_validate(
            {
                **_common(),
                "geometry": EventGeometry(
                    before_position=(0, 0, 0),
                    after_position=(1, 0, 0),
                ),
                "event_kind": "moved",
                "involved_entity_ids": ("entity-1",),
            }
        ),
        NoWriteMemoryRecord.model_validate(
            {
                **_common(),
                "candidate_type": "object",
                "reason": "duplicate observation",
            }
        ),
    )

    parsed = TypeAdapter(tuple[TypedMemoryRecord, ...]).validate_python(records)

    assert tuple(record.record_type for record in parsed) == (
        "object",
        "plane",
        "portal",
        "free_space",
        "landmark",
        "event",
        "no_write",
    )
    assert all(record.validity.end_time == 2.0 for record in parsed)
    assert serialized_byte_cost(records[-1]) == 0


def test_canonical_jsonl_byte_cost_is_exact_and_deterministic() -> None:
    record = ObjectMemoryRecord.model_validate(
        {
            **_common(),
            "geometry": ObjectGeometry(centroid=(1, 2, 3), extent=(1, 1, 1)),
            "semantic_label": "머그컵",
        }
    )

    encoded = canonical_jsonl_bytes(record)

    assert encoded.endswith(b"\n")
    assert encoded == canonical_jsonl_bytes(record)
    assert serialized_byte_cost(record) == len(encoded)
    assert "머그컵" in encoded.decode()


def test_object_requires_persistent_instance_identity() -> None:
    payload = {
        **_common(),
        "geometry": ObjectGeometry(centroid=(1, 2, 3), extent=(1, 1, 1)),
        "semantic_label": "mug",
    }
    del payload["instance_id"]

    with pytest.raises(ValidationError, match="instance_id"):
        _ = ObjectMemoryRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("model", "kwargs", "error"),
    [
        (
            ValidityInterval,
            {"start_time": 2.0, "end_time": 1.0},
            "end_time must be greater than or equal",
        ),
        (
            SpatialUncertainty,
            {
                "covariance_xyz": (
                    (1.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                "standard_deviation_m": 1.0,
            },
            "covariance must be symmetric",
        ),
        (
            LandmarkGeometry,
            {
                "position": (0.0, 0.0, 0.0),
                "ray_direction": (0.0, 0.0, 0.0),
                "view_cone_degrees": 30.0,
            },
            "ray_direction must be non-zero",
        ),
    ],
)
def test_invalid_geometry_contract_is_rejected(
    model: type[ValidityInterval | SpatialUncertainty | LandmarkGeometry],
    kwargs: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(ValidationError, match=error):
        _ = model.model_validate(kwargs)


@pytest.mark.parametrize(
    "covariance",
    [
        ((1.0, 2.0, 0.0), (2.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 0.9, 0.9), (0.9, 1.0, -0.9), (0.9, -0.9, 1.0)),
    ],
)
def test_covariance_rejects_non_psd_principal_minors(
    covariance: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> None:
    with pytest.raises(ValidationError, match="positive semidefinite"):
        _ = SpatialUncertainty(
            covariance_xyz=covariance,
            standard_deviation_m=1.0,
        )


def test_covariance_accepts_numerical_roundoff_at_psd_boundary() -> None:
    uncertainty = SpatialUncertainty(
        covariance_xyz=(
            (1.0, 1.0 + 1e-12, 0.0),
            (1.0 + 1e-12, 1.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        standard_deviation_m=1.0,
    )

    assert uncertainty.covariance_xyz[0][1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("event_kind", "geometry", "error"),
    [
        ("moved", EventGeometry(before_position=(0, 0, 0)), "moved event"),
        ("appeared", EventGeometry(), "appeared event"),
        (
            "appeared",
            EventGeometry(
                before_position=(0, 0, 0),
                after_position=(1, 0, 0),
            ),
            "appeared event",
        ),
        ("disappeared", EventGeometry(), "disappeared event"),
        (
            "disappeared",
            EventGeometry(
                before_position=(0, 0, 0),
                after_position=(1, 0, 0),
            ),
            "disappeared event",
        ),
    ],
)
def test_event_kind_requires_semantically_valid_positions(
    event_kind: str,
    geometry: EventGeometry,
    error: str,
) -> None:
    with pytest.raises(ValidationError, match=error):
        _ = EventMemoryRecord.model_validate(
            {
                **_common(),
                "geometry": geometry,
                "event_kind": event_kind,
                "involved_entity_ids": ("entity-1",),
            }
        )


@pytest.mark.parametrize(
    ("event_kind", "geometry"),
    [
        (
            "moved",
            EventGeometry(
                before_position=(0, 0, 0),
                after_position=(1, 0, 0),
            ),
        ),
        ("appeared", EventGeometry(after_position=(1, 0, 0))),
        ("disappeared", EventGeometry(before_position=(0, 0, 0))),
    ],
)
def test_event_kind_accepts_required_positions(
    event_kind: str,
    geometry: EventGeometry,
) -> None:
    record = EventMemoryRecord.model_validate(
        {
            **_common(),
            "geometry": geometry,
            "event_kind": event_kind,
            "involved_entity_ids": ("entity-1",),
        }
    )

    assert record.event_kind == event_kind
