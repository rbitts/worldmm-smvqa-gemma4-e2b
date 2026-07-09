from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from worldmm_smvqa.chunking import build_chunks
from worldmm_smvqa.fixtures import tiny_fixture_examples
from worldmm_smvqa.memory_sources import build_source_memories
from worldmm_smvqa.schema import (
    PROHIBITED_MEMORY_FIELDS,
    SourceStreamExample,
    TranscriptSpan,
)
from worldmm_smvqa.worldmm.episodic import (
    EpisodicEdgeRecord,
    EpisodicNodeRecord,
    InvalidTemporalGraphError,
    build_episodic_graph,
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


def test_episodic_graph_links_clips_to_shards_and_temporal_neighbors() -> None:
    # Given: tiny source streams chunked into 30s clips and 30m shards.
    sources, _labels = tiny_fixture_examples()
    chunks = build_chunks(sources)
    memories = build_source_memories(
        tuple(chunk for chunk in chunks if chunk.granularity == "clip_30s"),
    )

    # When: the episodic graph is built from allowed chunks and source memories.
    records = build_episodic_graph(chunks, memories)

    # Then: graph records include stable multi-scale nodes and deterministic edges.
    nodes = {
        record.node_id: record
        for record in records
        if isinstance(record, EpisodicNodeRecord)
    }
    edges = tuple(
        record for record in records if isinstance(record, EpisodicEdgeRecord)
    )
    assert nodes["episodic:fake_video_001:0:30:clip_30s"].text_embedding_id == (
        "embedding:episodic:fake_video_001:0:30:clip_30s:text"
    )
    assert nodes["episodic:fake_video_001:0:1800:shard_30m"].source_modalities == (
        "caption",
        "transcript",
        "ocr",
        "object",
        "frame",
    )
    assert any(
        edge.edge_type == "contains"
        and edge.source_node_id == "episodic:fake_video_001:0:1800:shard_30m"
        and edge.target_node_id == "episodic:fake_video_001:0:30:clip_30s"
        for edge in edges
    )
    assert any(
        edge.edge_type == "contains"
        and edge.source_node_id == "episodic:fake_video_001:1800:1900:shard_30m"
        and edge.target_node_id == "episodic:fake_video_001:1800:1830:clip_30s"
        for edge in edges
    )
    assert any(
        edge.edge_type == "temporal_next"
        and edge.source_node_id == "episodic:fake_video_001:0:30:clip_30s"
        and edge.target_node_id == "episodic:fake_video_001:30:60:clip_30s"
        for edge in edges
    )
    assert any(
        edge.edge_type == "temporal_next"
        and edge.source_node_id == "episodic:fake_video_001:1800:1830:clip_30s"
        and edge.target_node_id == "episodic:fake_video_001:1830:1860:clip_30s"
        for edge in edges
    )
    for edge in edges:
        source = nodes[edge.source_node_id]
        target = nodes[edge.target_node_id]
        assert (edge.source_start_time, edge.source_end_time) == (
            source.start_time,
            source.end_time,
        )
        assert (edge.target_start_time, edge.target_end_time) == (
            target.start_time,
            target.end_time,
        )
        assert edge.source_start_time <= edge.start_time
        assert edge.end_time <= edge.source_end_time


def test_episodic_graph_rejects_overlapping_event_spans() -> None:
    # Given: sorted transcript spans that overlap inside one clip.
    source = SourceStreamExample(
        video_id="video-overlap",
        start_time=0.0,
        end_time=30.0,
        transcript_spans=(
            TranscriptSpan(start_time=5.0, end_time=12.0, text="first"),
            TranscriptSpan(start_time=10.0, end_time=14.0, text="second"),
        ),
    )
    chunks = build_chunks((source,))
    memories = build_source_memories(
        tuple(chunk for chunk in chunks if chunk.granularity == "clip_30s"),
    )

    # When / Then: graph construction rejects the invalid event timeline.
    with pytest.raises(InvalidTemporalGraphError, match="overlap"):
        _ = build_episodic_graph(chunks, memories)


def test_build_memory_episodic_cli_writes_jsonl_without_labels(tmp_path: Path) -> None:
    # Given: the checked-in tiny fixture and an episodic output path.
    output = tmp_path / "episodic.jsonl"

    # When: episodic build is driven through the CLI.
    result = run_cli(
        "build-memory",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--store",
        "episodic",
        "--out",
        str(output),
    )

    # Then: JSONL records are parseable and exclude evaluator-only labels.
    assert result.returncode == 0, result.stderr
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines
    payload = "\n".join(lines)
    assert "nodes=" in result.stdout
    assert "edges=" in result.stdout
    assert "contains=" in result.stdout
    for field in PROHIBITED_MEMORY_FIELDS:
        key = field.rsplit(".", maxsplit=1)[-1]
        assert f'"{key}":' not in payload
    node_lines = tuple(line for line in lines if '"record_type":"node"' in line)
    edge_lines = tuple(line for line in lines if '"record_type":"edge"' in line)
    assert node_lines
    assert edge_lines
    _ = tuple(EpisodicNodeRecord.model_validate_json(line) for line in node_lines)
    _ = tuple(EpisodicEdgeRecord.model_validate_json(line) for line in edge_lines)


def test_build_memory_episodic_cli_fails_on_overlapping_events(tmp_path: Path) -> None:
    # Given: a fixture with sorted but overlapping transcript event spans.
    fixture = tmp_path / "fixture"
    prepare = run_cli("prepare-fixture", "--out", str(fixture))
    assert prepare.returncode == 0
    lines = (fixture / "sources.jsonl").read_text(encoding="utf-8").splitlines()
    first = SourceStreamExample.model_validate_json(lines[0])
    lines[0] = first.model_copy(
        update={
            "transcript_spans": (
                first.transcript_spans[0],
                TranscriptSpan(start_time=10.0, end_time=13.0, text="overlap"),
                first.transcript_spans[1],
            ),
        },
    ).model_dump_json()
    _ = (fixture / "sources.jsonl").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # When: episodic build is driven through the CLI.
    result = run_cli(
        "build-memory",
        "--fixture",
        str(fixture),
        "--store",
        "episodic",
        "--out",
        str(tmp_path / "episodic.jsonl"),
    )

    # Then: it fails with the typed graph error.
    assert result.returncode != 0
    assert "InvalidTemporalGraphError" in result.stderr
