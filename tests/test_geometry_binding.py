from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from worldmm_smvqa.fixtures import tiny_fixture_examples
from worldmm_smvqa.worldmm.geometry_binding import (
    DeterministicGeometryBinder,
    EmptyGeometryPrimitivesError,
    GeometryPrimitive,
    NoopBinder,
    SemanticGeometryBinder,
    build_geometry_primitives,
)
from worldmm_smvqa.worldmm.spatial import build_object_anchors

if TYPE_CHECKING:
    from worldmm_smvqa.worldmm.spatial_types import SpatialAnchorRecord


def _fixture_anchor() -> SpatialAnchorRecord:
    sources, _labels = tiny_fixture_examples()
    return build_object_anchors(sources[0])[0]


def test_noop_binder_selects_nearest_frame_deterministically() -> None:
    # Given: fixture anchor evidence and geometry primitives in different orders.
    anchor = _fixture_anchor()
    later = GeometryPrimitive(
        frame_ref="fake_video_001_frame_0072",
        x=7.2,
        y=0.0,
        z=1.0,
        source="slam_pose",
    )
    nearest = GeometryPrimitive(
        frame_ref="fake_video_001_frame_0008",
        x=0.4,
        y=1.4,
        z=1.0,
        source="gaze",
    )
    binder = NoopBinder()

    # When: the same primitives are bound in opposite input orders.
    first = binder.bind(anchor, (later, nearest))
    second = binder.bind(anchor, (nearest, later))

    # Then: the no-op binder picks the same nearest-in-time primitive.
    assert first == second
    assert first.anchor_memory_id == anchor.memory_id
    assert first.embedding_ref == "geometry:gaze:fake_video_001_frame_0008"
    assert first.primitive == nearest
    assert first.distance_m == 0.0


def test_deterministic_binder_prefers_matching_object_geometry() -> None:
    # Given: an exact object primitive and a closer-in-time pose primitive.
    anchor = _fixture_anchor()
    exact = GeometryPrimitive(
        frame_ref="fake_video_001_frame_0008",
        x=anchor.x,
        y=anchor.y,
        z=anchor.z,
        source="object_geometry",
        object_label=anchor.object_label,
        timestamp=anchor.start_time,
    )
    pose = GeometryPrimitive(
        frame_ref="fake_video_001_frame_0007",
        x=anchor.x + 0.1,
        y=anchor.y + 0.1,
        z=anchor.z,
        source="slam_pose",
        timestamp=(anchor.start_time + anchor.end_time) / 2.0,
    )

    # When: binding runs.
    bound = DeterministicGeometryBinder().bind(anchor, (pose, exact))

    # Then: typed object geometry wins over time-only proximity.
    assert bound.primitive == exact
    assert bound.embedding_ref == "geometry:object_geometry:fake_video_001_frame_0008"
    assert bound.distance_m == 0.0


def test_build_geometry_primitives_exports_object_gaze_and_pose_sources() -> None:
    # Given: the tiny fixture has gaze and pose samples.
    sources, _labels = tiny_fixture_examples()

    # When: primitives are built from source streams.
    primitives = build_geometry_primitives(sources[0])

    # Then: every available geometry source is represented deterministically.
    assert {primitive.source for primitive in primitives} == {"gaze", "slam_pose"}
    assert all(primitive.frame_ref for primitive in primitives)


def test_semantic_geometry_binder_is_runtime_checkable() -> None:
    # Given: the no-op binder implements the binder shape.
    binder = NoopBinder()

    # When / Then: runtime checks can guard plugin boundaries.
    assert isinstance(binder, SemanticGeometryBinder)


def test_noop_binder_rejects_empty_primitives_with_typed_error() -> None:
    # Given: a fixture anchor and no geometry primitives.
    anchor = _fixture_anchor()

    # When / Then: binding fails with a typed, inspectable error.
    with pytest.raises(EmptyGeometryPrimitivesError, match=anchor.memory_id):
        _ = NoopBinder().bind(anchor, ())
