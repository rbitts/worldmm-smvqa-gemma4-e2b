from __future__ import annotations

import pytest

from worldmm_smvqa.schema import StreamChunk
from worldmm_smvqa.worldmm.spatial import (
    build_object_state_snapshots,
    build_trajectory_summaries,
)
from worldmm_smvqa.worldmm.spatial_types import (
    InvalidSpatialInputError,
    SpatialAnchorRecord,
    ZoneRecord,
)


def _clip_chunk(start_time: float, end_time: float) -> StreamChunk:
    return StreamChunk(
        chunk_id=f"dyn_video:{int(start_time)}:{int(end_time)}:clip_30s",
        video_id="dyn_video",
        start_time=start_time,
        end_time=end_time,
        granularity="clip_30s",
    )


def _chunks() -> tuple[StreamChunk, ...]:
    return (
        _clip_chunk(0.0, 30.0),
        _clip_chunk(30.0, 60.0),
        _clip_chunk(60.0, 90.0),
    )


def _anchors() -> tuple[SpatialAnchorRecord, ...]:
    return (
        SpatialAnchorRecord(
            memory_id="spatial_anchor:dyn_video:mug:10",
            video_id="dyn_video",
            object_label="mug",
            x=1.2,
            y=0.4,
            z=0.0,
            zone_id="zone_dyn_video_0_0",
            start_time=10.0,
            end_time=12.0,
            frame_refs=(),
            confidence=0.9,
            provenance="pose",
        ),
        SpatialAnchorRecord(
            memory_id="spatial_anchor:dyn_video:key:65",
            video_id="dyn_video",
            object_label="key",
            x=4.0,
            y=0.0,
            z=0.0,
            zone_id="zone_dyn_video_2_0",
            start_time=65.0,
            end_time=66.0,
            frame_refs=(),
            confidence=0.8,
            provenance="gaze",
        ),
    )


def _zones() -> tuple[ZoneRecord, ...]:
    return (
        ZoneRecord(
            zone_id="zone_dyn_video_0_0",
            video_id="dyn_video",
            centroid_x=1.0,
            centroid_y=0.0,
            centroid_z=0.0,
            cell=(0, 0),
            visit_intervals=((0.0, 20.0), (70.0, 80.0)),
        ),
        ZoneRecord(
            zone_id="zone_dyn_video_2_0",
            video_id="dyn_video",
            centroid_x=4.0,
            centroid_y=0.0,
            centroid_z=0.0,
            cell=(2, 0),
            visit_intervals=((31.0, 50.0),),
        ),
    )


def test_object_state_snapshots_are_causal_and_recency_scored() -> None:
    # Given: three clip chunks and anchors where key first appears in chunk 3.
    chunks = _chunks()
    anchors = _anchors()

    # When: dynamic object state snapshots are built.
    snapshots = build_object_state_snapshots(chunks, anchors)

    # Then: each chunk only sees anchors whose evidence ended by chunk end.
    assert tuple(snapshot.memory_id for snapshot in snapshots) == (
        "spatial_snapshot:dyn_video:mug:30",
        "spatial_snapshot:dyn_video:mug:60",
        "spatial_snapshot:dyn_video:key:90",
        "spatial_snapshot:dyn_video:mug:90",
    )
    assert tuple(snapshot.object_label for snapshot in snapshots[:2]) == ("mug", "mug")
    assert snapshots[0].snippet == (
        "as of t=30s, mug last seen in zone_dyn_video_0_0 at t=12s near (1.2,0.4)"
    )
    assert snapshots[2].snippet == (
        "as of t=90s, key last seen in zone_dyn_video_2_0 at t=66s near (4,0)"
    )
    assert tuple(snapshot.end_time for snapshot in snapshots) == (
        30.0,
        60.0,
        90.0,
        90.0,
    )
    assert tuple(snapshot.base_score for snapshot in snapshots[:2]) == pytest.approx(
        (1.0 / 3.0, 2.0 / 3.0),
    )
    assert snapshots[2].base_score == pytest.approx(1.0)
    assert snapshots[3].base_score == pytest.approx(1.0)


def test_trajectory_summaries_render_chunk_local_zone_snippets() -> None:
    # Given: zone visits that overlap three clip chunks.
    chunks = _chunks()
    zones = _zones()

    # When: wearer trajectory summaries are built.
    summaries = build_trajectory_summaries(chunks, zones)

    # Then: every chunk has deterministic zone text and recency score.
    assert tuple(summary.memory_id for summary in summaries) == (
        "spatial_trajectory:dyn_video:30",
        "spatial_trajectory:dyn_video:60",
        "spatial_trajectory:dyn_video:90",
    )
    assert tuple(summary.snippet for summary in summaries) == (
        "wearer in zone_dyn_video_0_0 during [0,30]s",
        "wearer in zone_dyn_video_2_0 during [30,60]s",
        "wearer in zone_dyn_video_0_0 during [60,90]s",
    )
    assert tuple(summary.base_score for summary in summaries) == pytest.approx(
        (1.0 / 3.0, 2.0 / 3.0, 1.0),
    )


def test_trajectory_summaries_only_grow_when_zone_changes() -> None:
    chunks = _chunks()
    stable_zone = ZoneRecord(
        zone_id="zone_dyn_video_0_0",
        video_id="dyn_video",
        centroid_x=0.0,
        centroid_y=0.0,
        centroid_z=0.0,
        cell=(0, 0),
        visit_intervals=((0.0, 90.0),),
    )

    summaries = build_trajectory_summaries(chunks, (stable_zone,))

    assert tuple(summary.memory_id for summary in summaries) == (
        "spatial_trajectory:dyn_video:30",
    )


def test_dynamic_builders_reject_malformed_inputs() -> None:
    # Given: unsorted chunks and no zone coverage for a chunk.
    unsorted_chunks = (_clip_chunk(30.0, 60.0), _clip_chunk(0.0, 30.0))
    missing_zone_chunks = (_clip_chunk(90.0, 120.0),)

    # When / Then: dynamic builders reject malformed spatial inputs.
    with pytest.raises(InvalidSpatialInputError, match="clip_chunks must be sorted"):
        _ = build_object_state_snapshots(unsorted_chunks, _anchors())
    with pytest.raises(InvalidSpatialInputError, match="no zone visit overlaps"):
        _ = build_trajectory_summaries(missing_zone_chunks, _zones())
