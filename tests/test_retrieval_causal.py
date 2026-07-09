from __future__ import annotations

import os
import subprocess
from pathlib import Path

from worldmm_smvqa.chunking import build_chunks, read_source_streams
from worldmm_smvqa.fixtures import read_fixture_questions, tiny_fixture_examples
from worldmm_smvqa.retrieval import (
    RetrievalOptions,
    build_fixture_retrieval_stores,
    retrieve_evidence,
)
from worldmm_smvqa.retrieval_types import (
    EvidencePack,
    RetrievalMemoryRecord,
)
from worldmm_smvqa.schema import QuestionRequest

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


def test_retrieve_evidence_excludes_post_question_high_score_memory() -> None:
    # Given: a source-only question request and a future candidate with exact terms.
    _sources, labels = tiny_fixture_examples()
    label = labels[0]
    question = QuestionRequest(
        question_id=label.question_id,
        video_id=label.video_id,
        question=label.question,
        question_time=label.question_time,
        answer_choices=label.answer_choices,
    )
    stores = build_fixture_retrieval_stores(Path("tests/fixtures/tiny_smvqa"))
    future = RetrievalMemoryRecord(
        memory_id="future-perfect-match",
        source_store="semantic",
        video_id=question.video_id,
        start_time=50.0,
        end_time=60.0,
        snippet="Where fake mug placed beside notebook exact future answer",
        frame_refs=(),
        base_score=100.0,
    )

    # When: adaptive retrieval runs with all WorldMM stores.
    pack = retrieve_evidence(
        question,
        (*stores, future),
        enabled_stores=frozenset({"episodic", "semantic", "visual"}),
        options=RetrievalOptions(evidence_budget=6),
    )

    # Then: causal cutoff removes the future match and reports it.
    memory_ids = {item.memory_id for item in pack.evidence}
    assert "future-perfect-match" not in memory_ids
    assert pack.causal_filtered_count > 0
    assert all(item.end_time <= question.question_time for item in pack.evidence)


def test_retrieve_evidence_respects_store_subset_ablation() -> None:
    # Given: tiny fixture memories and a source-only question request.
    _sources, labels = tiny_fixture_examples()
    label = labels[0]
    question = QuestionRequest(
        question_id=label.question_id,
        video_id=label.video_id,
        question=label.question,
        question_time=label.question_time,
        answer_choices=label.answer_choices,
    )
    stores = build_fixture_retrieval_stores(Path("tests/fixtures/tiny_smvqa"))

    # When: retrieval is restricted to the visual store.
    pack = retrieve_evidence(
        question,
        stores,
        enabled_stores=frozenset({"visual"}),
        options=RetrievalOptions(evidence_budget=3),
    )

    # Then: only visual evidence appears.
    assert pack.evidence
    assert pack.selected_stores == ("visual",)
    assert {item.source_store for item in pack.evidence} == {"visual"}


def test_retrieve_evidence_returns_spatial_for_q_fake_005() -> None:
    # Given: q_fake_005 asks for pre-question spatial placement evidence.
    _sources, labels = tiny_fixture_examples()
    label = next(item for item in labels if item.question_id == "q_fake_005")
    question = QuestionRequest(
        question_id=label.question_id,
        video_id=label.video_id,
        question=label.question,
        question_time=label.question_time,
        answer_choices=label.answer_choices,
    )
    fixture_dir = Path("tests/fixtures/tiny_smvqa")
    chunks = build_chunks(read_source_streams(fixture_dir))
    stores = build_fixture_retrieval_stores(fixture_dir)

    # When: retrieval runs with all stores.
    pack = retrieve_evidence(
        question,
        stores,
        enabled_stores=frozenset({"episodic", "semantic", "visual", "spatial"}),
        options=RetrievalOptions(evidence_budget=6, chunks=chunks),
    )

    # Then: protocol-aware retrieval surfaces spatial memory and trace.
    assert "spatial" in pack.selected_stores
    assert any(item.source_store == "spatial" for item in pack.evidence)
    assert pack.retrieval_trace.protocols == (
        "smvqa-video-rag",
        "egobutler",
        "worldmm",
    )
    assert pack.retrieval_trace.eligible_shard_ids
    assert pack.retrieval_trace.selected_clip_ids
    assert pack.retrieval_trace.policy_route == "location"
    assert pack.retrieval_trace.store_order[0] == "spatial"
    assert pack.retrieval_trace.frame_ref_count <= 32


def test_retrieve_evidence_has_no_candidates_before_first_shard_closes() -> None:
    # Given: q_fake_001 is asked before any real 30m shard has closed.
    fixture_dir = Path("tests/fixtures/tiny_smvqa")
    question = next(
        item for item in read_fixture_questions(fixture_dir)
        if item.question_id == "q_fake_001"
    )
    chunks = build_chunks(read_source_streams(fixture_dir))
    stores = build_fixture_retrieval_stores(fixture_dir)
    early_memory = RetrievalMemoryRecord(
        memory_id="injected-early-perfect",
        source_store="semantic",
        video_id=question.video_id,
        start_time=10.0,
        end_time=11.0,
        snippet="Where fake mug placed beside notebook exact answer",
        frame_refs=(),
        base_score=1000.0,
    )

    # When: protocol-aware retrieval runs with real chunks.
    pack = retrieve_evidence(
        question,
        (*stores, early_memory),
        enabled_stores=frozenset({"episodic", "semantic", "visual", "spatial"}),
        options=RetrievalOptions(evidence_budget=6, chunks=chunks),
    )

    # Then: no synthetic first shard admits pre-45s memories.
    assert pack.evidence == ()
    assert pack.retrieval_trace.eligible_shard_ids == ()
    assert pack.retrieval_trace.selected_clip_ids == ()
    assert pack.causal_filtered_count > 0


def test_retrieve_evidence_excludes_future_spatial_and_future_shard_records() -> None:
    # Given: q_fake_006 has only post-question magnet evidence.
    _sources, labels = tiny_fixture_examples()
    label = next(item for item in labels if item.question_id == "q_fake_006")
    question = QuestionRequest(
        question_id=label.question_id,
        video_id=label.video_id,
        question=label.question,
        question_time=label.question_time,
        answer_choices=label.answer_choices,
    )
    fixture_dir = Path("tests/fixtures/tiny_smvqa")
    chunks = build_chunks(read_source_streams(fixture_dir))
    stores = build_fixture_retrieval_stores(fixture_dir)
    future_spatial_snapshot = RetrievalMemoryRecord(
        memory_id="spatial_snapshot:fake_video_002:magnet:150",
        source_store="spatial",
        video_id=question.video_id,
        start_time=132.0,
        end_time=137.0,
        snippet="magnet fridge zone blue exact answer",
        frame_refs=("fake_video_002_frame_0132",),
        base_score=100.0,
    )
    future_shard = RetrievalMemoryRecord(
        memory_id="future-shard-magnet",
        source_store="semantic",
        video_id=question.video_id,
        start_time=132.0,
        end_time=137.0,
        snippet="magnet fridge zone blue exact answer",
        frame_refs=(),
        base_score=100.0,
    )

    # When: retrieval runs with injected high-score future evidence.
    pack = retrieve_evidence(
        question,
        (*stores, future_spatial_snapshot, future_shard),
        enabled_stores=frozenset({"episodic", "semantic", "visual", "spatial"}),
        options=RetrievalOptions(evidence_budget=6, chunks=chunks),
    )

    # Then: future spatial/shard evidence is excluded before scoring.
    memory_ids = {item.memory_id for item in pack.evidence}
    assert "spatial_snapshot:fake_video_002:magnet:150" not in memory_ids
    assert "future-shard-magnet" not in memory_ids
    assert all("magnet" not in item.snippet.casefold() for item in pack.evidence)
    assert pack.causal_filtered_count > 0
    assert pack.retrieval_trace.causal_filtered_count == pack.causal_filtered_count


def test_retrieve_evidence_omits_spatial_when_store_disabled() -> None:
    # Given: spatial records exist but the caller disables the spatial store.
    _sources, labels = tiny_fixture_examples()
    label = next(item for item in labels if item.question_id == "q_fake_005")
    question = QuestionRequest(
        question_id=label.question_id,
        video_id=label.video_id,
        question=label.question,
        question_time=label.question_time,
        answer_choices=label.answer_choices,
    )
    stores = build_fixture_retrieval_stores(Path("tests/fixtures/tiny_smvqa"))

    # When: retrieval is restricted to non-spatial stores.
    pack = retrieve_evidence(
        question,
        stores,
        enabled_stores=frozenset({"episodic", "semantic", "visual"}),
        options=RetrievalOptions(evidence_budget=6),
    )

    # Then: spatial is absent from evidence and policy trace.
    assert "spatial" not in pack.selected_stores
    assert "spatial" not in pack.requested_stores
    assert "spatial" not in pack.retrieval_trace.store_order


def test_retrieve_cli_writes_evidence_pack(tmp_path: Path) -> None:
    # Given: the checked-in tiny fixture and requested CLI shape.
    output = tmp_path / "evidence_pack.json"

    # When: retrieval is driven through the CLI.
    result = run_cli(
        "retrieve",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--question",
        "q_object_001",
        "--stores",
        "episodic,semantic,visual",
        "--retrieval-protocol",
        "legacy-round-robin",
        "--out",
        str(output),
    )

    # Then: the output pack has causal, scored evidence from requested stores.
    assert result.returncode == 0, result.stderr
    pack = EvidencePack.model_validate_json(output.read_text(encoding="utf-8"))
    assert pack.question_id == "q_fake_001"
    assert pack.causal_filtered_count >= 1
    assert pack.evidence
    assert "stores=episodic,semantic,visual" in result.stdout
    assert "causal_filtered_count=" in result.stdout


def test_retrieve_cli_worldmm_smvqa_protocol_emits_trace(
    tmp_path: Path,
) -> None:
    # Given: the spatial fixture question has an eligible closed shard.
    output = tmp_path / "spatial_pack.json"

    # When: retrieval uses the default WorldMM-SMVQA protocol explicitly.
    result = run_cli(
        "retrieve",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--question",
        "q_fake_005",
        "--stores",
        "episodic,semantic,visual,spatial",
        "--retrieval-protocol",
        "worldmm-smvqa",
        "--max-frame-refs",
        "32",
        "--out",
        str(output),
    )

    # Then: chunk-backed protocol trace and spatial evidence are persisted.
    assert result.returncode == 0, result.stderr
    pack = EvidencePack.model_validate_json(output.read_text(encoding="utf-8"))
    assert pack.retrieval_trace.protocols == (
        "smvqa-video-rag",
        "egobutler",
        "worldmm",
    )
    assert pack.retrieval_trace.eligible_shard_ids
    assert pack.retrieval_trace.selected_clip_ids
    assert pack.retrieval_trace.store_order[0] == "spatial"
    assert any(item.source_store == "spatial" for item in pack.evidence)
    assert "protocol=worldmm-smvqa" in result.stdout


def test_retrieve_cli_bogus_protocol_fails_nonzero(tmp_path: Path) -> None:
    # Given: a misspelled retrieval protocol flag.
    output = tmp_path / "bad_protocol.json"

    # When: retrieval is driven through the CLI.
    result = run_cli(
        "retrieve",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--question",
        "q_fake_005",
        "--retrieval-protocol",
        "bogus",
        "--out",
        str(output),
    )

    # Then: usage fails before writing an evidence pack.
    assert result.returncode != 0
    assert "UsageError" in result.stderr
    assert "unsupported retrieval protocol: bogus" in result.stderr
    assert not output.exists()


def test_retrieve_cli_negative_future_memory_is_filtered(tmp_path: Path) -> None:
    # Given: a fixture plus CLI-injected future high-score memory.
    output = tmp_path / "negative_pack.json"

    # When: retrieval runs with the injection flag used by fixture QA.
    result = run_cli(
        "retrieve",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--question",
        "q_object_001",
        "--stores",
        "episodic,semantic,visual",
        "--out",
        str(output),
        "--inject-future-memory",
    )

    # Then: the injected post-question memory is absent and counted as causal.
    assert result.returncode == 0, result.stderr
    pack = EvidencePack.model_validate_json(output.read_text(encoding="utf-8"))
    memory_ids = {item.memory_id for item in pack.evidence}
    assert "injected-post-question-high-score" not in memory_ids
    assert pack.causal_filtered_count > 0


def test_retrieve_cli_pre_question_evidence_has_no_future_frame_refs(
    tmp_path: Path,
) -> None:
    # Given: q_fake_001 is asked at 45s and fixture frames include a 72s frame.
    output = tmp_path / "causal_pack.json"

    # When: retrieval is driven through the CLI.
    result = run_cli(
        "retrieve",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--question",
        "q_fake_001",
        "--stores",
        "episodic,semantic,visual",
        "--out",
        str(output),
    )

    # Then: selected evidence has no frame refs from after question_time.
    assert result.returncode == 0, result.stderr
    pack = EvidencePack.model_validate_json(output.read_text(encoding="utf-8"))
    refs = {
        frame_ref
        for item in pack.evidence
        for frame_ref in item.frame_refs
    }
    assert "fake_video_001_frame_0072" not in refs
