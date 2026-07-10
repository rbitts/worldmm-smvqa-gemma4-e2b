from __future__ import annotations

import pytest
from pydantic import ValidationError

from worldmm_smvqa.chunking import TemporalOrderError, build_chunks
from worldmm_smvqa.schema import (
    GazeSample,
    ObjectMetadata,
    PoseSample,
    SourceStreamExample,
    StreamChunk,
)


def _clip(chunk_id: str, chunks: tuple[StreamChunk, ...]) -> StreamChunk:
    for chunk in chunks:
        if chunk.chunk_id == chunk_id:
            return chunk
    raise AssertionError(chunk_id)


def test_pose_and_gaze_samples_are_sliced_when_chunks_cross_boundary() -> None:
    # Given: pose and gaze samples before, at, and after a 30s boundary.
    source = SourceStreamExample(
        video_id="video-spatial",
        start_time=0.0,
        end_time=60.0,
        pose_samples=(
            PoseSample(timestamp=0.0, x=1.0, y=2.0, z=0.5),
            PoseSample(timestamp=29.999, x=1.5, y=2.5, z=0.6, yaw=90.0),
            PoseSample(timestamp=30.0, x=3.0, y=4.0, z=0.7),
            PoseSample(timestamp=59.999, x=3.5, y=4.5, z=0.8),
        ),
        gaze_samples=(
            GazeSample(timestamp=10.0, x=10.0, y=20.0, z=1.0),
            GazeSample(timestamp=30.0, x=30.0, y=40.0, z=1.5),
        ),
    )

    # When: source streams are split into 30s clips.
    chunks = build_chunks((source,))
    first = _clip("video-spatial:0:30:clip_30s", chunks)
    second = _clip("video-spatial:30:60:clip_30s", chunks)

    # Then: each clip contains only samples inside its time window.
    assert tuple(sample.timestamp for sample in first.pose_samples) == (0.0, 29.999)
    assert tuple(sample.timestamp for sample in second.pose_samples) == (30.0, 59.999)
    assert tuple(sample.timestamp for sample in first.gaze_samples) == (10.0,)
    assert tuple(sample.timestamp for sample in second.gaze_samples) == (30.0,)


def test_pose_sample_preserves_coordinate_convention() -> None:
    # Given: coordinates following the spatial-memory convention.
    docstring = PoseSample.__doc__

    # When: the sample is parsed by the schema.
    sample = PoseSample(timestamp=1.0, x=2.0, y=3.0, z=4.0)

    # Then: x/y remain horizontal meters and z remains vertical meters.
    assert sample.x == 2.0
    assert sample.y == 3.0
    assert sample.z == 4.0
    assert docstring is not None
    assert "x/y" in docstring
    assert "horizontal" in docstring
    assert "z" in docstring
    assert "vertical" in docstring


def test_object_geometry_requires_complete_xyz() -> None:
    # Given / When / Then: partial object geometry is rejected at the schema edge.
    with pytest.raises(ValidationError, match="object geometry requires x, y, and z"):
        _ = ObjectMetadata(
            label="mug",
            confidence=0.9,
            start_time=1.0,
            end_time=2.0,
            x=1.0,
        )


def test_out_of_order_pose_samples_raise_temporal_order_error() -> None:
    # Given: pose samples with decreasing timestamps.
    source = SourceStreamExample(
        video_id="video-unsorted-pose",
        start_time=0.0,
        end_time=60.0,
        pose_samples=(
            PoseSample(timestamp=20.0, x=1.0, y=1.0, z=0.0),
            PoseSample(timestamp=10.0, x=2.0, y=2.0, z=0.0),
        ),
    )

    # When / Then: chunking rejects it through the existing temporal path.
    with pytest.raises(TemporalOrderError, match="pose_samples"):
        _ = build_chunks((source,))


def test_out_of_order_gaze_samples_raise_temporal_order_error() -> None:
    # Given: gaze samples with decreasing timestamps.
    source = SourceStreamExample(
        video_id="video-unsorted-gaze",
        start_time=0.0,
        end_time=60.0,
        gaze_samples=(
            GazeSample(timestamp=20.0, x=1.0, y=1.0, z=0.0),
            GazeSample(timestamp=10.0, x=2.0, y=2.0, z=0.0),
        ),
    )

    # When / Then: chunking rejects it through the existing temporal path.
    with pytest.raises(TemporalOrderError, match="gaze_samples"):
        _ = build_chunks((source,))
