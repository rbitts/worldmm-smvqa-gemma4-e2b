from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from worldmm_smvqa.fixtures import tiny_fixture_examples
from worldmm_smvqa.worldmm.geometry_binding import (
    EmptyGeometryPrimitivesError,
    GeometryPrimitive,
    NoopBinder,
    SemanticGeometryBinder,
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
    assert first.embedding_ref is None
    assert first.primitive == nearest


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
