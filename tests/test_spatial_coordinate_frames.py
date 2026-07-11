from __future__ import annotations

import pytest

from worldmm_smvqa.schema import (
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
from worldmm_smvqa.worldmm.spatial_types import (
    InvalidSpatialInputError,
    SpatialAnchorRecord,
    ZoneRecord,
)


def test_pose_anchor_and_zone_preserve_custom_coordinate_frame() -> None:
    source = SourceStreamExample(
        video_id="custom-frame",
        start_time=0.0,
        end_time=6.0,
        pose_samples=(
            PoseSample(
                timestamp=0.0,
                x=0.0,
                y=0.0,
                z=1.5,
                coordinate_frame="slam_world",
            ),
            PoseSample(
                timestamp=4.0,
                x=2.0,
                y=0.0,
                z=1.5,
                coordinate_frame="slam_world",
            ),
        ),
        object_detections=(
            ObjectMetadata(
                label="mug",
                confidence=0.9,
                start_time=2.0,
                end_time=4.0,
            ),
        ),
    )

    (anchor,) = build_object_anchors(source)
    zones = build_zones(source)

    assert anchor.provenance == "pose"
    assert anchor.coordinate_frame == "slam_world"
    assert {zone.coordinate_frame for zone in zones} == {"slam_world"}


def test_pose_anchor_rejects_interpolation_across_incompatible_frames() -> None:
    source = SourceStreamExample(
        video_id="frame-transition",
        start_time=0.0,
        end_time=6.0,
        pose_samples=(
            PoseSample(
                timestamp=0.0,
                x=0.0,
                y=0.0,
                z=1.5,
                coordinate_frame="submap:a",
            ),
            PoseSample(
                timestamp=4.0,
                x=2.0,
                y=0.0,
                z=1.5,
                coordinate_frame="submap:b",
            ),
        ),
        object_detections=(
            ObjectMetadata(
                label="mug",
                confidence=0.9,
                start_time=2.0,
                end_time=4.0,
            ),
        ),
    )

    with pytest.raises(
        InvalidSpatialInputError,
        match="cannot interpolate across incompatible pose frames",
    ):
        _ = build_object_anchors(source)


def test_pose_anchor_at_frame_transition_uses_exact_sample_frame() -> None:
    source = SourceStreamExample(
        video_id="exact-frame-transition",
        start_time=0.0,
        end_time=10.0,
        pose_samples=(
            PoseSample(
                timestamp=0.0,
                x=0.0,
                y=0.0,
                z=1.5,
                coordinate_frame="submap:a",
            ),
            PoseSample(
                timestamp=4.0,
                x=2.0,
                y=0.0,
                z=1.5,
                coordinate_frame="submap:b",
            ),
            PoseSample(
                timestamp=8.0,
                x=4.0,
                y=0.0,
                z=1.5,
                coordinate_frame="submap:b",
            ),
        ),
        object_detections=(
            ObjectMetadata(
                label="mug",
                confidence=0.9,
                start_time=0.0,
                end_time=8.0,
            ),
        ),
    )

    (anchor,) = build_object_anchors(source)

    assert (anchor.x, anchor.y, anchor.z) == (2.0, 0.0, 1.5)
    assert anchor.coordinate_frame == "submap:b"


def test_zone_build_rejects_mixed_pose_frames() -> None:
    source = SourceStreamExample(
        video_id="mixed-zone-frame",
        start_time=0.0,
        end_time=2.0,
        pose_samples=(
            PoseSample(
                timestamp=0.0,
                x=0.0,
                y=0.0,
                z=1.5,
                coordinate_frame="submap:a",
            ),
            PoseSample(
                timestamp=1.0,
                x=0.5,
                y=0.0,
                z=1.5,
                coordinate_frame="submap:b",
            ),
        ),
    )

    with pytest.raises(
        InvalidSpatialInputError,
        match="pose_samples use incompatible coordinate frames",
    ):
        _ = build_zones(source)


def test_object_geometry_and_gaze_keep_source_world_default() -> None:
    source = SourceStreamExample(
        video_id="implicit-source-world",
        start_time=0.0,
        end_time=5.0,
        gaze_samples=(GazeSample(timestamp=3.0, x=1.0, y=1.0, z=1.0),),
        object_detections=(
            ObjectMetadata(
                label="mug",
                confidence=0.9,
                start_time=0.5,
                end_time=1.5,
                x=0.0,
                y=0.0,
                z=1.0,
            ),
            ObjectMetadata(
                label="notebook",
                confidence=0.8,
                start_time=2.5,
                end_time=3.5,
            ),
        ),
    )

    object_anchor, gaze_anchor = build_object_anchors(source)

    assert object_anchor.provenance == "object_geometry"
    assert object_anchor.coordinate_frame == "source_world"
    assert gaze_anchor.provenance == "gaze"
    assert gaze_anchor.coordinate_frame == "source_world"
    assert (
        ZoneRecord(
            zone_id="legacy-zone",
            video_id="implicit-source-world",
            centroid_x=0.0,
            centroid_y=0.0,
            centroid_z=0.0,
            cell=(0, 0),
            visit_intervals=((0.0, 1.0),),
        ).coordinate_frame
        == "source_world"
    )


def test_relations_require_matching_frames_and_preserve_custom_frame() -> None:
    left = _anchor("mug", x=0.0, coordinate_frame="room:desk")
    right = _anchor("notebook", x=0.5, coordinate_frame="room:desk")

    relations = derive_relations((left, right))

    assert relations
    assert {relation.coordinate_frame for relation in relations} == {"room:desk"}
    assert (
        derive_relations(
            (left, right.model_copy(update={"coordinate_frame": "room:kitchen"})),
        )
        == ()
    )


def _anchor(
    label: str,
    *,
    x: float,
    coordinate_frame: str,
) -> SpatialAnchorRecord:
    return SpatialAnchorRecord(
        memory_id=f"anchor:{label}",
        video_id="coordinate-frame-video",
        object_label=label,
        instance_id=f"{label}-1",
        x=x,
        y=0.0,
        z=1.0,
        zone_id="zone_coordinate-frame-video_0_0",
        start_time=0.0,
        end_time=2.0,
        frame_refs=(),
        confidence=1.0,
        provenance="object_geometry",
        coordinate_frame=coordinate_frame,
    )
