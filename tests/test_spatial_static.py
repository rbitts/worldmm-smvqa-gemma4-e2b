from __future__ import annotations

import pytest

from worldmm_smvqa.schema import (
    FrameMetadata,
    GazeSample,
    ObjectMetadata,
    PoseSample,
    SourceStreamExample,
)
from worldmm_smvqa.worldmm.spatial import (
    build_object_anchors,
    build_zones,
    derive_relations,
)
from worldmm_smvqa.worldmm.spatial_types import InvalidSpatialInputError


def _source() -> SourceStreamExample:
    return SourceStreamExample(
        video_id="spatial_video",
        start_time=0.0,
        end_time=10.0,
        pose_samples=(
            PoseSample(timestamp=0.0, x=0.0, y=0.0, z=0.0),
            PoseSample(timestamp=2.0, x=0.8, y=0.2, z=0.0),
            PoseSample(timestamp=4.0, x=2.2, y=0.3, z=0.0),
            PoseSample(timestamp=6.0, x=2.4, y=0.4, z=0.0),
            PoseSample(timestamp=8.0, x=0.5, y=0.5, z=0.0),
        ),
        gaze_samples=(
            GazeSample(timestamp=2.0, x=0.4, y=0.4, z=1.0),
        ),
        object_detections=(
            ObjectMetadata(label="mug", confidence=0.9, start_time=1.0, end_time=3.0),
            ObjectMetadata(
                label="notebook",
                confidence=0.8,
                start_time=2.2,
                end_time=3.2,
            ),
        ),
        frame_metadata=(
            FrameMetadata(
                frame_ref="spatial_video_frame_0001",
                timestamp=0.5,
                description="before detections",
            ),
            FrameMetadata(
                frame_ref="spatial_video_frame_0002",
                timestamp=2.5,
                description="mug beside notebook",
            ),
            FrameMetadata(
                frame_ref="spatial_video_frame_0003",
                timestamp=4.0,
                description="after detections",
            ),
        ),
    )


def test_build_zones_merges_consecutive_pose_samples_by_grid_cell() -> None:
    # Given: a pose track that visits one grid cell, leaves, then returns.
    source = _source()

    # When: static spatial zones are built.
    zones = build_zones(source)

    # Then: zone ids and visit intervals are deterministic and hand-computed.
    assert tuple(zone.zone_id for zone in zones) == (
        "zone_spatial_video_0_0",
        "zone_spatial_video_1_0",
    )
    assert zones[0].cell == (0, 0)
    assert zones[0].visit_intervals == ((0.0, 2.0), (8.0, 8.0))
    assert zones[0].centroid_x == pytest.approx(1.3 / 3.0)
    assert zones[0].centroid_y == pytest.approx(0.7 / 3.0)
    assert zones[1].cell == (1, 0)
    assert zones[1].visit_intervals == ((4.0, 6.0),)


def test_build_object_anchors_prefers_gaze_then_interpolated_pose() -> None:
    # Given: one detection with an in-interval gaze target and one without.
    source = _source()

    # When: object anchors are built.
    anchors = build_object_anchors(source)

    # Then: gaze provenance wins, pose fallback is linearly interpolated.
    assert tuple(anchor.memory_id for anchor in anchors) == (
        "spatial_anchor:spatial_video:mug:1",
        "spatial_anchor:spatial_video:notebook:2.2",
    )
    mug, notebook = anchors
    assert mug.object_label == "mug"
    assert mug.zone_id == "zone_spatial_video_0_0"
    assert (mug.x, mug.y, mug.z) == (0.4, 0.4, 1.0)
    assert mug.provenance == "gaze"
    assert mug.frame_refs == ("spatial_video_frame_0002",)
    assert notebook.object_label == "notebook"
    assert notebook.zone_id == "zone_spatial_video_0_0"
    assert notebook.x == pytest.approx(1.29)
    assert notebook.y == pytest.approx(0.235)
    assert notebook.z == pytest.approx(0.0)
    assert notebook.provenance == "pose"
    assert notebook.frame_refs == ("spatial_video_frame_0002",)


def test_build_object_anchors_prefers_object_geometry_over_gaze() -> None:
    # Given: a detection has exact object geometry and an in-window gaze sample.
    source = SourceStreamExample(
        video_id="geo_video",
        start_time=0.0,
        end_time=10.0,
        pose_samples=(PoseSample(timestamp=1.0, x=0.0, y=0.0, z=1.0),),
        gaze_samples=(GazeSample(timestamp=2.0, x=9.0, y=9.0, z=9.0),),
        object_detections=(
            ObjectMetadata(
                label="mug",
                confidence=0.9,
                start_time=1.0,
                end_time=3.0,
                x=1.0,
                y=2.0,
                z=0.5,
            ),
        ),
    )

    # When: anchors are built.
    (anchor,) = build_object_anchors(source)

    # Then: exact object geometry wins over proxy gaze/pose coordinates.
    assert (anchor.x, anchor.y, anchor.z) == (1.0, 2.0, 0.5)
    assert anchor.provenance == "object_geometry"
    assert anchor.geometry_frame_ref == "geo_video_mug_1"
    assert anchor.geometry_source == "object_geometry"
    assert anchor.geometry_distance_m == 0.0
    assert anchor.geometry_embedding_ref == "geometry:object_geometry:geo_video_mug_1"


def test_derive_relations_emits_one_ordered_near_relation() -> None:
    # Given: two overlapping anchors in the same zone under the near threshold.
    anchors = build_object_anchors(_source())

    # When: static spatial relations are derived.
    relations = derive_relations(anchors)

    # Then: one lexicographically ordered near relation covers the overlap.
    assert len(relations) == 1
    relation = relations[0]
    assert relation.memory_id == "spatial_relation:spatial_video:mug:near:notebook:2.2"
    assert relation.subject == "mug"
    assert relation.relation == "near"
    assert relation.object == "notebook"
    assert relation.zone_id == "zone_spatial_video_0_0"
    assert relation.start_time == 2.2
    assert relation.end_time == 3.0


def test_derive_relations_emits_directional_geometry_relations() -> None:
    # Given: exact object coordinates, not gaze/pose proxy coordinates.
    source = SourceStreamExample(
        video_id="geo_video",
        start_time=0.0,
        end_time=10.0,
        object_detections=(
            ObjectMetadata(
                label="mug",
                confidence=0.9,
                start_time=1.0,
                end_time=4.0,
                x=0.0,
                y=1.0,
                z=0.7,
            ),
            ObjectMetadata(
                label="notebook",
                confidence=0.9,
                start_time=2.0,
                end_time=5.0,
                x=1.0,
                y=0.0,
                z=0.2,
            ),
        ),
    )

    # When: static spatial relations are derived.
    relations = derive_relations(build_object_anchors(source))

    # Then: metric direction survives as structured relation records.
    relation_keys = {
        (relation.subject, relation.relation, relation.object)
        for relation in relations
    }
    assert ("mug", "left_of", "notebook") in relation_keys
    assert ("notebook", "right_of", "mug") in relation_keys
    assert ("mug", "in_front_of", "notebook") in relation_keys
    assert ("notebook", "behind", "mug") in relation_keys
    assert ("mug", "above", "notebook") in relation_keys
    assert ("notebook", "below", "mug") in relation_keys
    left_relation = next(
        relation for relation in relations if relation.relation == "left_of"
    )
    assert left_relation.distance_m == pytest.approx(1.5)
    assert (
        left_relation.delta_x,
        left_relation.delta_y,
        left_relation.delta_z,
    ) == pytest.approx((1.0, -1.0, -0.5))


def test_build_zones_rejects_unsorted_pose_samples() -> None:
    # Given: source pose samples with decreasing timestamps.
    source = SourceStreamExample(
        video_id="bad_spatial_video",
        start_time=0.0,
        end_time=10.0,
        pose_samples=(
            PoseSample(timestamp=2.0, x=0.0, y=0.0, z=0.0),
            PoseSample(timestamp=1.0, x=1.0, y=1.0, z=0.0),
        ),
    )

    # When / Then: spatial zone building rejects malformed input.
    with pytest.raises(InvalidSpatialInputError, match="pose_samples must be sorted"):
        _ = build_zones(source)


def test_build_zones_rejects_non_positive_cell_size() -> None:
    # Given: a valid source and an invalid grid size.
    source = _source()

    # When / Then: spatial zone building rejects malformed input.
    with pytest.raises(InvalidSpatialInputError, match="cell_size must be positive"):
        _ = build_zones(source, cell_size=0.0)
