from __future__ import annotations

from pathlib import Path

import pytest

from worldmm_smvqa.chunking import read_source_streams
from worldmm_smvqa.cli import run
from worldmm_smvqa.schema import (
    FrameMetadata,
    ObjectMetadata,
    OCRMetadata,
    SourceStreamExample,
)
from worldmm_smvqa.sensor_frames import (
    SENSOR_FRAME_MANIFEST_ENV,
    SENSOR_FRAME_MANIFEST_NAME,
    SensorFrameManifestError,
    configured_sensor_frame_manifest,
    read_sensor_frame_manifest,
    write_sensor_frame_manifest,
)


def test_sensor_manifest_applies_one_shared_1hz_rgb_inventory(tmp_path: Path) -> None:
    # Given: dense frames and visual metadata spanning three one-second windows.
    source = _dense_source()
    _write_sources(tmp_path, source)
    manifest_path = tmp_path / SENSOR_FRAME_MANIFEST_NAME

    # When: one manifest is written and loaded through the shared source reader.
    summary = write_sensor_frame_manifest((source,), manifest_path)
    sensed = read_source_streams(tmp_path)[0]
    record = read_sensor_frame_manifest(manifest_path)[0]

    # Then: only the first timestamp-only sample per second reaches memory and QA.
    assert summary.source_frame_count == 5
    assert summary.selected_frame_count == 3
    assert tuple(frame.sample_index for frame in record.selected_frames) == (0, 1, 2)
    assert sensed.frame_refs == ("f-0a", "f-1a", "f-2a")
    assert tuple(frame.timestamp for frame in sensed.frame_metadata) == (0.1, 1.2, 2.4)
    assert sensed.captions == ()
    assert sensed.ocr == ("KEEP-0", "KEEP-1")
    assert tuple(item.label for item in sensed.object_detections) == (
        "kept-zero",
        "kept-one",
    )
    assert sensed.objects == ("kept-zero", "kept-one")


def test_sensor_manifest_rejects_changed_source_frame_inventory(tmp_path: Path) -> None:
    # Given: a manifest generated from one raw frame inventory.
    source = _dense_source()
    _write_sources(tmp_path, source)
    _ = write_sensor_frame_manifest((source,), tmp_path / SENSOR_FRAME_MANIFEST_NAME)
    changed = source.model_copy(
        update={"frame_metadata": source.frame_metadata[:-1]},
    )
    _write_sources(tmp_path, changed)

    # When / Then: stale manifests cannot silently change the experiment input.
    with pytest.raises(SensorFrameManifestError, match="stale or modified"):
        _ = read_source_streams(tmp_path)


def test_sensor_manifest_cli_writes_auditable_counts(tmp_path: Path) -> None:
    # Given / When: the checked-in source inventory is sampled through the CLI.
    output = tmp_path / "sensor_frames.jsonl"
    result = run(
        (
            "build-memory",
            "--stage",
            "sensor-frames",
            "--fixture",
            "tests/fixtures/tiny_smvqa",
            "--out",
            str(output),
        ),
    )

    # Then: the run artifact records the fixed rate and before/after counts.
    assert output.is_file()
    assert "sensor_rate_hz=1" in result.stdout
    assert "source_frames=6 selected_frames=6" in result.stdout


def test_remote_source_read_requires_sensor_manifest(tmp_path: Path) -> None:
    with pytest.raises(SensorFrameManifestError, match=SENSOR_FRAME_MANIFEST_ENV):
        _ = configured_sensor_frame_manifest(
            tmp_path,
            env={"WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST": "1"},
        )


def _dense_source() -> SourceStreamExample:
    frames = (
        FrameMetadata(frame_ref="f-0a", timestamp=0.1, description="selected 0"),
        FrameMetadata(frame_ref="f-0b", timestamp=0.8, description="dropped 0"),
        FrameMetadata(frame_ref="f-1a", timestamp=1.2, description="selected 1"),
        FrameMetadata(frame_ref="f-1b", timestamp=1.9, description="dropped 1"),
        FrameMetadata(frame_ref="f-2a", timestamp=2.4, description="selected 2"),
    )
    return SourceStreamExample(
        video_id="dense-video",
        start_time=0.0,
        end_time=3.0,
        captions=("ungrounded dense caption",),
        ocr=("KEEP-0", "DROP-0", "KEEP-1"),
        ocr_entries=(
            OCRMetadata(
                start_time=0.1,
                end_time=0.2,
                text="KEEP-0",
                frame_ref="f-0a",
            ),
            OCRMetadata(
                start_time=0.8,
                end_time=0.9,
                text="DROP-0",
                frame_ref="f-0b",
            ),
            OCRMetadata(
                start_time=1.2,
                end_time=1.3,
                text="KEEP-1",
                frame_ref="f-1a",
            ),
        ),
        objects=("kept-zero", "dropped-zero", "kept-one"),
        object_detections=(
            ObjectMetadata(
                start_time=0.0,
                end_time=0.2,
                label="kept-zero",
                confidence=0.9,
            ),
            ObjectMetadata(
                start_time=0.7,
                end_time=0.9,
                label="dropped-zero",
                confidence=0.8,
            ),
            ObjectMetadata(
                start_time=1.0,
                end_time=1.3,
                label="kept-one",
                confidence=0.7,
            ),
        ),
        frame_refs=tuple(frame.frame_ref for frame in frames),
        frame_metadata=frames,
    )


def _write_sources(path: Path, source: SourceStreamExample) -> None:
    _ = (path / "sources.jsonl").write_text(
        f"{source.model_dump_json()}\n",
        encoding="utf-8",
    )
