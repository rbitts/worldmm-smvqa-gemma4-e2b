from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from worldmm_smvqa.chunking import (
    TemporalOrderError,
    build_chunks,
)
from worldmm_smvqa.schema import (
    PROHIBITED_MEMORY_FIELDS,
    FrameMetadata,
    ObjectMetadata,
    OCRMetadata,
    SourceStreamExample,
    StreamChunk,
    TranscriptSpan,
)

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    _ = env.pop("WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST", None)
    return subprocess.run(
        ["uv", "run", "worldmm-smvqa", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def boundary_source() -> SourceStreamExample:
    return SourceStreamExample(
        video_id="video-boundary",
        start_time=0.0,
        end_time=1830.0,
        captions=("wide shot", "counter close-up"),
        ocr=("MENU", "OPEN"),
        objects=("lamp", "mug"),
        frame_refs=("source_frame_0000", "source_frame_1800"),
        transcript_spans=(
            TranscriptSpan(start_time=0.0, end_time=0.5, text="at 0"),
            TranscriptSpan(start_time=29.999, end_time=30.5, text="before 30"),
            TranscriptSpan(start_time=30.0, end_time=31.0, text="at 30"),
            TranscriptSpan(
                start_time=1799.999,
                end_time=1800.5,
                text="before 1800",
            ),
            TranscriptSpan(start_time=1800.0, end_time=1801.0, text="at 1800"),
        ),
        ocr_entries=(
            OCRMetadata(
                start_time=29.999,
                end_time=30.2,
                text="OCR-BEFORE-30",
                frame_ref="frame_0029",
            ),
            OCRMetadata(
                start_time=30.0,
                end_time=30.2,
                text="OCR-AT-30",
                frame_ref="frame_0030",
            ),
        ),
        object_detections=(
            ObjectMetadata(
                start_time=1799.999,
                end_time=1800.1,
                label="lamp",
                confidence=0.9,
            ),
            ObjectMetadata(
                start_time=1800.0,
                end_time=1800.1,
                label="mug",
                confidence=0.8,
            ),
        ),
        frame_metadata=(
            FrameMetadata(frame_ref="frame_0000", timestamp=0.0, description="zero"),
            FrameMetadata(
                frame_ref="frame_1799",
                timestamp=1799.999,
                description="pre shard",
            ),
            FrameMetadata(
                frame_ref="frame_1800",
                timestamp=1800.0,
                description="next shard",
            ),
        ),
    )


def texts(chunk_id: str, chunks: tuple[StreamChunk, ...]) -> tuple[str, ...]:
    for chunk in chunks:
        if chunk.chunk_id == chunk_id:
            return tuple(span.text for span in chunk.transcript_spans)
    raise AssertionError(chunk_id)


def test_chunk_boundaries_are_start_inclusive_end_exclusive() -> None:
    # Given: source timestamps exactly around clip and shard boundaries.
    source = boundary_source()

    # When: source streams are split into 30s clips and 30m shards.
    chunks = build_chunks((source,))

    # Then: boundary timestamps land in the expected stable interval IDs.
    assert texts("video-boundary:0:30:clip_30s", chunks) == ("at 0", "before 30")
    assert texts("video-boundary:30:60:clip_30s", chunks) == ("at 30",)
    assert texts("video-boundary:1770:1800:clip_30s", chunks) == ("before 1800",)
    assert texts("video-boundary:1800:1830:clip_30s", chunks) == ("at 1800",)
    assert texts("video-boundary:0:1800:shard_30m", chunks) == (
        "at 0",
        "before 30",
        "at 30",
        "before 1800",
    )
    assert texts("video-boundary:1800:1830:shard_30m", chunks) == ("at 1800",)


def test_chunks_preserve_modality_refs_and_exclude_labels() -> None:
    # Given: source rows with flat and timestamped modality metadata.
    source = boundary_source()

    # When: chunks are built.
    chunks = build_chunks((source,))
    chunk = next(
        item for item in chunks if item.chunk_id == "video-boundary:30:60:clip_30s"
    )

    # Then: source summaries and time-local refs are preserved without labels.
    assert chunk.captions == ("wide shot", "counter close-up")
    assert chunk.ocr == ("MENU", "OPEN")
    assert chunk.objects == ("lamp", "mug")
    assert chunk.frame_refs == ()
    assert chunk.ocr_entries[0].text == "OCR-AT-30"
    assert chunk.ocr_entries[0].frame_ref == "frame_0030"
    assert chunk.ocr_entries[0].start_time == 30.0
    chunk_json = chunk.model_dump_json()
    for field in PROHIBITED_MEMORY_FIELDS:
        key = field.rsplit(".", maxsplit=1)[-1]
        assert f'"{key}":' not in chunk_json


def test_chunks_do_not_copy_future_source_frame_refs() -> None:
    # Given: a source has one frame before and one frame after a pre-question chunk.
    source = boundary_source()

    # When: chunks are built.
    chunks = build_chunks((source,))
    early = next(
        item for item in chunks if item.chunk_id == "video-boundary:0:30:clip_30s"
    )

    # Then: the early chunk contains only frame refs local to its time window.
    assert early.frame_refs == ("frame_0000",)
    assert "source_frame_1800" not in early.frame_refs
    assert "frame_1800" not in early.frame_refs


def test_unsorted_timestamps_raise_temporal_order_error() -> None:
    # Given: a source stream with transcript spans out of timestamp order.
    source = SourceStreamExample(
        video_id="video-unsorted",
        start_time=0.0,
        end_time=60.0,
        transcript_spans=(
            TranscriptSpan(start_time=20.0, end_time=21.0, text="later"),
            TranscriptSpan(start_time=10.0, end_time=11.0, text="earlier"),
        ),
    )

    # When / Then: chunking rejects it with the typed temporal error.
    with pytest.raises(TemporalOrderError, match="video-unsorted"):
        _ = build_chunks((source,))


def test_build_memory_chunk_cli_writes_both_granularities(tmp_path: Path) -> None:
    # Given: the checked-in tiny fixture and an output JSONL path.
    output = tmp_path / "chunks.jsonl"

    # When: chunk stage is driven through the CLI.
    result = run_cli(
        "build-memory",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--stage",
        "chunk",
        "--out",
        str(output),
    )

    # Then: the CLI succeeds and writes both requested granularities.
    assert result.returncode == 0, result.stderr
    chunks = [
        StreamChunk.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {chunk.granularity for chunk in chunks} == {"clip_30s", "shard_30m"}
    assert "clip_30s" in result.stdout
    assert "shard_30m" in result.stdout


def test_build_memory_chunk_cli_fails_on_unsorted_timestamps(tmp_path: Path) -> None:
    # Given: a fixture whose source transcript spans are out of order.
    fixture = tmp_path / "fixture"
    prepare = run_cli("prepare-fixture", "--out", str(fixture))
    assert prepare.returncode == 0
    lines = (fixture / "sources.jsonl").read_text(encoding="utf-8").splitlines()
    first = SourceStreamExample.model_validate_json(lines[0])
    lines[0] = first.model_copy(
        update={"transcript_spans": tuple(reversed(first.transcript_spans))},
    ).model_dump_json()
    _ = (fixture / "sources.jsonl").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # When: chunk stage is driven through the CLI.
    result = run_cli(
        "build-memory",
        "--fixture",
        str(fixture),
        "--stage",
        "chunk",
        "--out",
        str(tmp_path / "chunks.jsonl"),
    )

    # Then: it fails with the typed temporal order error.
    assert result.returncode != 0
    assert "TemporalOrderError" in result.stderr
