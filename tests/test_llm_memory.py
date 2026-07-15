from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from worldmm_smvqa.chunking import build_chunks
from worldmm_smvqa.memory_sources import build_source_memories
from worldmm_smvqa.retrieval import build_retrieval_records
from worldmm_smvqa.schema import (
    FrameMetadata,
    SourceStreamExample,
    TranscriptSpan,
)
from worldmm_smvqa.worldmm.episodic_types import EpisodicNodeRecord
from worldmm_smvqa.worldmm.llm_errors import LLMMemoryError
from worldmm_smvqa.worldmm.llm_memory import (
    TextGenerator,
    build_llm_episodic_graph,
    build_llm_semantic_memory,
    build_llm_visual_memory,
)
from worldmm_smvqa.worldmm.llm_memory_io import (
    write_llm_episodic_memory,
    write_llm_semantic_memory,
)

ROOT = Path(__file__).resolve().parents[1]


def _source() -> SourceStreamExample:
    return SourceStreamExample(
        video_id="llm_video",
        start_time=0.0,
        end_time=60.0,
        transcript_spans=(
            TranscriptSpan(start_time=5.0, end_time=10.0, text="mug on desk"),
            TranscriptSpan(start_time=35.0, end_time=40.0, text="mug on shelf"),
        ),
        captions=("desk scene",),
        frame_refs=("llm_video_frame_0005",),
        frame_metadata=(
            FrameMetadata(
                frame_ref="llm_video_frame_0005",
                timestamp=5.0,
                description="desk frame",
            ),
        ),
    )


def _fake_generate(prompt: str) -> str:
    if "clip summaries" in prompt:
        return json.dumps({"summary": "shard: desk routine"})
    if "Observations" in prompt:
        return f"```json\n{json.dumps({'summary': 'clip event'})}\n```"
    detail = f"unexpected prompt: {prompt[:80]}"
    raise AssertionError(detail)


def test_llm_episodic_nodes_carry_clip_and_shard_summaries() -> None:
    # Given: chunks and source memories from one small stream.
    chunks = build_chunks((_source(),))
    memories = build_source_memories(
        tuple(chunk for chunk in chunks if chunk.granularity == "clip_30s"),
    )

    # When: the LLM episodic graph is built with a fake generator.
    records = build_llm_episodic_graph(chunks, memories, _fake_generate)

    # Then: clip and shard nodes carry LLM summaries and edges survive.
    nodes = [item for item in records if isinstance(item, EpisodicNodeRecord)]
    clip_nodes = [node for node in nodes if node.granularity == "clip_30s"]
    shard_nodes = [node for node in nodes if node.granularity == "shard_30m"]
    assert all(node.summary == "clip event" for node in clip_nodes)
    assert all(node.summary == "shard: desk routine" for node in shard_nodes)
    assert any(not isinstance(item, EpisodicNodeRecord) for item in records)


def test_llm_episodic_io_partitions_by_rank_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: two distributed ranks building a two-video fixture.
    output = tmp_path / "episodic.jsonl"
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    _ = write_llm_episodic_memory(
        ROOT / "tests/fixtures/tiny_smvqa",
        output,
        _fake_generate,
    )

    # When: completed rank restarts, then rank zero finishes and merges.
    def fail_if_called(_prompt: str) -> str:
        detail = "completed rank must not regenerate memory"
        raise AssertionError(detail)

    _ = write_llm_episodic_memory(
        ROOT / "tests/fixtures/tiny_smvqa",
        output,
        fail_if_called,
    )
    monkeypatch.setenv("RANK", "0")
    _ = write_llm_episodic_memory(
        ROOT / "tests/fixtures/tiny_smvqa",
        output,
        _fake_generate,
    )

    # Then: merged artifact contains nodes from both videos.
    nodes = tuple(
        EpisodicNodeRecord.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if '"record_type":"node"' in line
    )
    assert {node.video_id for node in nodes} == {
        "fake_video_001",
        "fake_video_002",
    }


def test_episodic_retrieval_snippet_prefers_llm_summary() -> None:
    # Given: an episodic graph whose nodes carry LLM summaries.
    chunks = build_chunks((_source(),))
    memories = build_source_memories(
        tuple(chunk for chunk in chunks if chunk.granularity == "clip_30s"),
    )
    records = build_llm_episodic_graph(chunks, memories, _fake_generate)

    # When: retrieval candidates are built.
    candidates = build_retrieval_records(records, (), ())

    # Then: episodic snippets lead with the LLM summary text.
    episodic = [item for item in candidates if item.source_store == "episodic"]
    assert episodic
    assert all(
        item.snippet.startswith("clip event")
        or item.snippet.startswith("shard: desk routine")
        for item in episodic
    )


def _consolidating_generate(decision: str) -> tuple[list[str], TextGenerator]:
    calls: list[str] = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        if "existing triplet" in prompt:
            return json.dumps({"action": decision})
        if "mug on shelf" in prompt:
            return json.dumps(
                {
                    "triplets": [
                        {
                            "subject": "mug",
                            "predicate": "located_on",
                            "object": "shelf",
                        },
                    ]
                },
            )
        return json.dumps(
            {
                "triplets": [
                    {"subject": "mug", "predicate": "located_on", "object": "desk"},
                ]
            },
        )

    return calls, generate


def _clip_node(node_id: str, start: float, summary: str) -> EpisodicNodeRecord:
    return EpisodicNodeRecord(
        node_id=node_id,
        video_id="llm_video",
        granularity="clip_30s",
        start_time=start,
        end_time=start + 30.0,
        source_chunk_id=node_id.removeprefix("episodic:"),
        source_memory_ids=(),
        source_modality_refs=(),
        source_modalities=(),
        frame_refs=(),
        confidence=1.0,
        text_embedding_id=f"embedding:{node_id}:text",
        summary=summary,
    )


def test_llm_semantic_consolidation_replaces_object_on_llm_decision() -> None:
    # Given: two time-ordered clip summaries that contradict each other.
    nodes = (
        _clip_node("episodic:llm_video:0:30:clip_30s", 0.0, "mug on desk"),
        _clip_node("episodic:llm_video:30:60:clip_30s", 30.0, "mug on shelf"),
    )
    _calls, generate = _consolidating_generate("replace")

    # When: semantic memory is consolidated.
    triples = build_llm_semantic_memory(nodes, generate)

    # Then: one triple survives with the revised object and both supports.
    assert len(triples) == 1
    triple = triples[0]
    assert triple.object == "shelf"
    assert triple.support_event_count == 2
    assert triple.memory_id == "semantic:llm_video:mug:located_on"


def test_llm_semantic_keep_decision_preserves_existing_object() -> None:
    # Given: the same contradiction but a keep decision.
    nodes = (
        _clip_node("episodic:llm_video:0:30:clip_30s", 0.0, "mug on desk"),
        _clip_node("episodic:llm_video:30:60:clip_30s", 30.0, "mug on shelf"),
    )
    _calls, generate = _consolidating_generate("keep")

    # When: semantic memory is consolidated.
    triples = build_llm_semantic_memory(nodes, generate)

    # Then: the original object stays.
    assert len(triples) == 1
    assert triples[0].object == "desk"


def test_llm_semantic_io_reads_episodic_jsonl_with_extra_fields(
    tmp_path: Path,
) -> None:
    # Given: LLM episodic memory persisted as full node/edge JSONL records.
    episodic_path = tmp_path / "episodic.jsonl"
    semantic_path = tmp_path / "semantic.jsonl"
    _ = write_llm_episodic_memory(
        Path("tests/fixtures/tiny_smvqa"),
        episodic_path,
        _fake_generate,
    )

    # When: semantic memory reads that episodic artifact.
    summary = write_llm_semantic_memory(
        episodic_path,
        semantic_path,
        lambda _prompt: json.dumps(
            {
                "triplets": [
                    {
                        "subject": "mug",
                        "predicate": "placed",
                        "object": "desk",
                    },
                ],
            },
        ),
    )

    # Then: header parsing ignores node extras and emits semantic JSONL.
    assert summary.records > 0
    assert semantic_path.read_text(encoding="utf-8").strip()


def test_llm_visual_memory_captions_frames_without_unused_vectors(
    tmp_path: Path,
) -> None:
    # Given: a real frame file under the frame root.
    frame_dir = tmp_path / "llm_video"
    frame_dir.mkdir()
    _ = (frame_dir / "llm_video_frame_0005.jpg").write_bytes(b"fake")

    # When: LLM visual memory is built with fake caption/embedding models.
    records = build_llm_visual_memory(
        (_source(),),
        frame_root=tmp_path,
        caption=lambda path: f"caption for {path.stem}",
    )

    # Then: each record carries VLM caption and frame grounding only.
    assert len(records) == 1
    assert records[0].source_frame_description == "caption for llm_video_frame_0005"
    assert records[0].embedding_ref == "vlm-caption:llm_video_frame_0005"


def test_llm_visual_memory_requires_frame_assets(tmp_path: Path) -> None:
    # Given: an empty frame root.
    # When / Then: building visual memory fails on the missing frame file.
    with pytest.raises(LLMMemoryError, match="llm_video_frame_0005"):
        _ = build_llm_visual_memory(
            (_source(),),
            frame_root=tmp_path,
            caption=lambda _path: "caption",
        )


def test_llm_visual_memory_rejects_frame_path_escape(tmp_path: Path) -> None:
    # Given: a source whose frame ref tries to escape frame_root.
    source = _source().model_copy(
        update={
            "frame_refs": ("../outside",),
            "frame_metadata": (
                FrameMetadata(
                    frame_ref="../outside",
                    timestamp=5.0,
                    description="escape",
                ),
            ),
        },
    )

    # When / Then: visual memory rejects the path before captioning.
    with pytest.raises(LLMMemoryError, match="escapes frame root"):
        _ = build_llm_visual_memory(
            (source,),
            frame_root=tmp_path,
            caption=lambda _path: "caption",
        )


def test_build_memory_cli_rejects_local_qwen_backend(tmp_path: Path) -> None:
    # Given: a local host without remote approval.
    env = os.environ.copy()
    _ = env.pop("WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST", None)

    # When: the LLM memory lane is requested locally.
    result = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "worldmm-smvqa",
            "build-memory",
            "--config",
            "configs/remote.example.yaml",
            "--store",
            "episodic",
            "--fixture",
            "tests/fixtures/tiny_smvqa",
            "--backend",
            "qwen",
            "--out",
            str(tmp_path / "episodic.jsonl"),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then: it fails as remote-only before doing any work.
    assert result.returncode != 0
    assert "remote-only" in f"{result.stdout}\n{result.stderr}"
    assert not (tmp_path / "episodic.jsonl").exists()


def test_build_memory_cli_reports_missing_qwen_paths_without_traceback(
    tmp_path: Path,
) -> None:
    # Given: remote approval but no local model path bindings.
    env = os.environ.copy()
    env["WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST"] = "1"
    _ = env.pop("WORLDMM_MEMORY_MODEL_PATH", None)

    # When: the Qwen backend is requested.
    result = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "worldmm-smvqa",
            "build-memory",
            "--config",
            "configs/remote.example.yaml",
            "--store",
            "episodic",
            "--fixture",
            "tests/fixtures/tiny_smvqa",
            "--backend",
            "qwen",
            "--out",
            str(tmp_path / "episodic.jsonl"),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then: CLI reports a concise typed error and writes nothing.
    assert result.returncode == 2
    assert "LLMMemoryError: backend: missing WORLDMM_MEMORY_MODEL_PATH" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "episodic.jsonl").exists()
