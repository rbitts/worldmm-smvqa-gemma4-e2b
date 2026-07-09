from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from worldmm_smvqa.fixtures import tiny_fixture_examples
from worldmm_smvqa.schema import (
    PROHIBITED_MEMORY_FIELDS,
    FrameMetadata,
    SourceStreamExample,
)
from worldmm_smvqa.worldmm.semantic import SemanticTripleRecord, build_semantic_memory
from worldmm_smvqa.worldmm.spatial_types import SpatialAnchorRecord
from worldmm_smvqa.worldmm.visual import (
    MissingGroundingError,
    VisualMemoryRecord,
    build_visual_memory,
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


def test_semantic_memory_builds_repeated_source_triples() -> None:
    # Given: tiny fixture streams with repeated fake/synthetic source events.
    sources, _labels = tiny_fixture_examples()

    # When: semantic memory is built from source-only stream data.
    records = build_semantic_memory(sources)

    # Then: deterministic relation/habit triples are written without labels.
    assert records
    triples = {(record.subject, record.predicate, record.object) for record in records}
    assert ("fake_video_001", "habitually_mentions", "fake") in triples
    assert ("fake_video_002", "habitually_mentions", "synthetic") in triples
    for record in records:
        assert record.store == "semantic"
        assert record.support_event_count >= 2
        assert record.support_memory_ids
        assert record.text_embedding_id == f"embedding:{record.memory_id}:text"


def test_visual_memory_records_frame_grounding_and_fake_embedding_refs() -> None:
    # Given: tiny fixture streams with frame, OCR, and object source metadata.
    sources, _labels = tiny_fixture_examples()

    # When: visual memory is built in deterministic fixture mode.
    records = build_visual_memory(sources)

    # Then: each visual record has timestamp grounding and source refs.
    assert records
    first = records[0]
    assert first.store == "visual"
    assert first.frame_ref == "fake_video_001_frame_0008"
    assert first.timestamp == 8.0
    assert first.embedding_ref == "fixture-embedding:fake_video_001_frame_0008:8"
    assert first.ocr_refs == ("NOTE-42",)
    assert first.object_refs == ("mug",)
    for record in records:
        assert record.start_time <= record.timestamp <= record.end_time
        assert record.frame_ref
        assert record.embedding_ref


def test_visual_memory_rejects_frame_without_timestamp() -> None:
    # Given: a source stream with a frame ref but no timestamped frame metadata.
    source = SourceStreamExample(
        video_id="ungrounded",
        start_time=0.0,
        end_time=30.0,
        frame_refs=("ungrounded_frame_0001",),
    )

    # When / Then: visual memory construction rejects missing grounding.
    with pytest.raises(MissingGroundingError, match="ungrounded_frame_0001"):
        _ = build_visual_memory((source,))


def test_build_memory_semantic_visual_cli_writes_directory_without_labels(
    tmp_path: Path,
) -> None:
    # Given: the checked-in tiny fixture and a WorldMM S/V output directory.
    output = tmp_path / "worldmm_sv"

    # When: semantic and visual stores are driven through the CLI.
    result = run_cli(
        "build-memory",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--store",
        "semantic,visual",
        "--out",
        str(output),
    )

    # Then: deterministic JSONL files are parseable and exclude labels.
    assert result.returncode == 0, result.stderr
    assert "semantic_records=" in result.stdout
    assert "visual_records=" in result.stdout
    semantic_lines = (
        (output / "semantic.jsonl").read_text(encoding="utf-8").splitlines()
    )
    visual_lines = (output / "visual.jsonl").read_text(encoding="utf-8").splitlines()
    assert semantic_lines
    assert visual_lines
    payload = "\n".join((*semantic_lines, *visual_lines))
    for field in PROHIBITED_MEMORY_FIELDS:
        key = field.rsplit(".", maxsplit=1)[-1]
        assert f'"{key}":' not in payload
    _ = tuple(SemanticTripleRecord.model_validate_json(line) for line in semantic_lines)
    _ = tuple(VisualMemoryRecord.model_validate_json(line) for line in visual_lines)


def test_build_memory_semantic_visual_spatial_cli_writes_directory_without_labels(
    tmp_path: Path,
) -> None:
    # Given: the checked-in tiny fixture and all non-episodic WorldMM stores.
    output = tmp_path / "worldmm_svs"

    # When: semantic, visual, and spatial stores are driven through the CLI.
    result = run_cli(
        "build-memory",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--stores",
        "semantic,visual,spatial",
        "--out",
        str(output),
    )

    # Then: the spatial JSONL is written with source-only memory records.
    assert result.returncode == 0, result.stderr
    assert "spatial_records=" in result.stdout
    spatial_lines = (output / "spatial.jsonl").read_text(encoding="utf-8").splitlines()
    assert spatial_lines
    payload = "\n".join(spatial_lines)
    for field in PROHIBITED_MEMORY_FIELDS:
        key = field.rsplit(".", maxsplit=1)[-1]
        assert f'"{key}":' not in payload
    anchors = tuple(
        SpatialAnchorRecord.model_validate_json(line)
        for line in spatial_lines
        if '"record_type":"spatial_anchor"' in line
    )
    assert anchors


def test_build_memory_visual_cli_fails_without_timestamp(tmp_path: Path) -> None:
    # Given: a fixture with a frame ref whose timestamp metadata was removed.
    fixture = tmp_path / "fixture"
    prepare = run_cli("prepare-fixture", "--out", str(fixture))
    assert prepare.returncode == 0
    lines = (fixture / "sources.jsonl").read_text(encoding="utf-8").splitlines()
    first = SourceStreamExample.model_validate_json(lines[0])
    lines[0] = first.model_copy(
        update={
            "frame_metadata": (
                FrameMetadata(
                    frame_ref="fake_video_001_frame_0072",
                    timestamp=72.0,
                    description="Fake desk frame with lamp switched on.",
                ),
            ),
        },
    ).model_dump_json()
    _ = (fixture / "sources.jsonl").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # When: visual memory build is driven through the CLI.
    result = run_cli(
        "build-memory",
        "--fixture",
        str(fixture),
        "--store",
        "visual",
        "--out",
        str(tmp_path / "visual"),
    )

    # Then: it fails with the typed grounding error.
    assert result.returncode != 0
    assert "MissingGroundingError" in result.stderr
