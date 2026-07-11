from __future__ import annotations

from worldmm_smvqa.chunking import build_chunks
from worldmm_smvqa.schema import (
    FrameMetadata,
    GazeSample,
    ObjectMetadata,
    OCRMetadata,
    PoseSample,
    SourceStreamExample,
    TranscriptSpan,
)


def test_source_end_samples_are_kept_once_per_granularity() -> None:
    # Given: every timed stream has an internal-boundary and source-end item.
    source = SourceStreamExample(
        video_id="video-source-end",
        start_time=0.0,
        end_time=1830.0,
        transcript_spans=(
            TranscriptSpan(start_time=1800.0, end_time=1800.1, text="boundary"),
            TranscriptSpan(start_time=1830.0, end_time=1830.1, text="source end"),
        ),
        ocr_entries=(
            OCRMetadata(
                start_time=1800.0,
                end_time=1800.1,
                text="BOUNDARY",
                frame_ref="frame-1800",
            ),
            OCRMetadata(
                start_time=1830.0,
                end_time=1830.1,
                text="SOURCE-END",
                frame_ref="frame-1830",
            ),
        ),
        object_detections=(
            ObjectMetadata(
                start_time=1800.0,
                end_time=1800.1,
                label="boundary-object",
                confidence=1.0,
            ),
            ObjectMetadata(
                start_time=1830.0,
                end_time=1830.1,
                label="source-end-object",
                confidence=1.0,
            ),
        ),
        pose_samples=(
            PoseSample(timestamp=1800.0, x=1.0, y=0.0, z=0.0),
            PoseSample(timestamp=1830.0, x=2.0, y=0.0, z=0.0),
        ),
        gaze_samples=(
            GazeSample(timestamp=1800.0, x=1.0, y=0.0, z=0.0),
            GazeSample(timestamp=1830.0, x=2.0, y=0.0, z=0.0),
        ),
        frame_metadata=(
            FrameMetadata(
                frame_ref="frame-1800",
                timestamp=1800.0,
                description="boundary",
            ),
            FrameMetadata(
                frame_ref="frame-1830",
                timestamp=1830.0,
                description="source end",
            ),
        ),
    )

    # When: independent 30-second and 30-minute windows are built.
    chunks = build_chunks((source,))

    # Then: source-end data lands in each final window exactly once.
    for granularity in ("clip_30s", "shard_30m"):
        selected = tuple(chunk for chunk in chunks if chunk.granularity == granularity)
        final = selected[-1]
        assert final.start_time == 1800.0
        assert final.end_time == 1830.0
        assert tuple(span.start_time for span in final.transcript_spans) == (
            1800.0,
            1830.0,
        )
        assert tuple(entry.start_time for entry in final.ocr_entries) == (
            1800.0,
            1830.0,
        )
        assert tuple(item.start_time for item in final.object_detections) == (
            1800.0,
            1830.0,
        )
        assert tuple(sample.timestamp for sample in final.pose_samples) == (
            1800.0,
            1830.0,
        )
        assert tuple(sample.timestamp for sample in final.gaze_samples) == (
            1800.0,
            1830.0,
        )
        assert tuple(frame.timestamp for frame in final.frame_metadata) == (
            1800.0,
            1830.0,
        )

        assert (
            sum(
                span.start_time == 1830.0
                for chunk in selected
                for span in chunk.transcript_spans
            )
            == 1
        )
        assert (
            sum(
                entry.start_time == 1830.0
                for chunk in selected
                for entry in chunk.ocr_entries
            )
            == 1
        )
        assert (
            sum(
                item.start_time == 1830.0
                for chunk in selected
                for item in chunk.object_detections
            )
            == 1
        )
        assert (
            sum(
                sample.timestamp == 1830.0
                for chunk in selected
                for sample in chunk.pose_samples
            )
            == 1
        )
        assert (
            sum(
                sample.timestamp == 1830.0
                for chunk in selected
                for sample in chunk.gaze_samples
            )
            == 1
        )
        assert (
            sum(
                frame.timestamp == 1830.0
                for chunk in selected
                for frame in chunk.frame_metadata
            )
            == 1
        )
