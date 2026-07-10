from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from worldmm_smvqa.chunking import build_chunks, read_source_streams
from worldmm_smvqa.fixtures import read_fixture_questions, tiny_fixture_examples
from worldmm_smvqa.retrieval import (
    RetrievalOptions,
    build_fixture_retrieval_stores,
    build_retrieval_records,
    read_retrieval_memory_artifacts,
    retrieve_evidence,
)
from worldmm_smvqa.retrieval_types import (
    EvidencePack,
    RetrievalMemoryRecord,
)
from worldmm_smvqa.schema import QuestionRequest, StreamChunk
from worldmm_smvqa.worldmm.semantic import SemanticTripleRecord
from worldmm_smvqa.worldmm.spatial_compression import SpatialExperimentConfig
from worldmm_smvqa.worldmm.spatial_types import SpatialTokenRecord
from worldmm_smvqa.worldmm.visual import VisualMemoryRecord

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


def test_visual_retrieval_interval_is_frame_timestamp() -> None:
    # Given: a visual memory whose source stream spans much more than one frame.
    visual = VisualMemoryRecord(
        memory_id="visual:v1:f42:42",
        video_id="v1",
        frame_ref="f42",
        timestamp=42.0,
        start_time=0.0,
        end_time=100.0,
        embedding_ref="embedding:f42",
        ocr_refs=(),
        object_refs=("mug",),
        timestamp_grounding="v1@42",
        source_frame_description="mug on desk",
    )

    # When: retrieval records are built.
    record = build_retrieval_records((), (), (visual,))[0]

    # Then: clip/shard assignment uses the frame point, not the full source span.
    assert record.start_time == record.end_time == 42.0


def test_retrieve_evidence_round_robins_available_stores() -> None:
    # Given: the first routed store has enough records to consume the whole budget.
    question = QuestionRequest(
        question_id="q-fair",
        video_id="v1",
        question="Where was the mug?",
        question_time=1900.0,
        answer_choices=(),
    )
    records = (
        RetrievalMemoryRecord(
            memory_id="episodic-1",
            source_store="episodic",
            video_id="v1",
            start_time=1.0,
            end_time=2.0,
            snippet="mug in room",
            frame_refs=(),
            base_score=10.0,
        ),
        RetrievalMemoryRecord(
            memory_id="episodic-2",
            source_store="episodic",
            video_id="v1",
            start_time=3.0,
            end_time=4.0,
            snippet="mug in room",
            frame_refs=(),
            base_score=9.0,
        ),
        RetrievalMemoryRecord(
            memory_id="semantic-1",
            source_store="semantic",
            video_id="v1",
            start_time=5.0,
            end_time=6.0,
            snippet="mug beside notebook",
            frame_refs=(),
            base_score=1.0,
        ),
    )

    # When: evidence budget can hold one candidate from each routed store.
    pack = retrieve_evidence(
        question,
        records,
        enabled_stores=frozenset({"episodic", "semantic"}),
        options=RetrievalOptions(evidence_budget=2),
    )

    # Then: the first store cannot starve the next store.
    assert tuple(item.source_store for item in pack.evidence) == (
        "episodic",
        "semantic",
    )


def test_where_survives_retrieval_tokenization() -> None:
    # Given: geometry wording is the only lexical overlap.
    question = QuestionRequest(
        question_id="q-where",
        video_id="v1",
        question="Where?",
        question_time=1900.0,
        answer_choices=(),
    )
    record = RetrievalMemoryRecord(
        memory_id="spatial-where",
        source_store="spatial",
        video_id="v1",
        start_time=1.0,
        end_time=2.0,
        snippet="where",
        frame_refs=(),
    )

    # When: spatial evidence is scored.
    pack = retrieve_evidence(
        question,
        (record,),
        enabled_stores=frozenset({"spatial"}),
    )

    # Then: "where" contributes lexical and geometry score.
    assert pack.evidence[0].retrieval_score > 1.0


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


def test_retrieve_evidence_uses_causal_records_from_current_shard() -> None:
    # Given: q_fake_001 is asked inside the first 30m shard.
    fixture_dir = Path("tests/fixtures/tiny_smvqa")
    question = next(
        item
        for item in read_fixture_questions(fixture_dir)
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

    # Then: current shard is allowed, but only records ending before 45s survive.
    assert any(item.memory_id == early_memory.memory_id for item in pack.evidence)
    assert pack.retrieval_trace.eligible_shard_ids
    assert pack.retrieval_trace.selected_clip_ids
    assert all(item.end_time <= question.question_time for item in pack.evidence)
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


def test_retrieve_evidence_scopes_to_question_video_ids() -> None:
    # Given: a SuperMemory-style question whose evidence can live in another session.
    question = QuestionRequest(
        question_id="q-cross-video",
        video_id="primary-video",
        video_ids=("primary-video", "support-video"),
        question="Where was the red mug?",
        question_time=1900.0,
        answer_choices=(),
    )
    support = RetrievalMemoryRecord(
        memory_id="support-hit",
        source_store="semantic",
        video_id="support-video",
        start_time=120.0,
        end_time=130.0,
        snippet="red mug on shelf",
        frame_refs=(),
        base_score=10.0,
    )
    outside = RetrievalMemoryRecord(
        memory_id="outside-hit",
        source_store="semantic",
        video_id="outside-video",
        start_time=120.0,
        end_time=130.0,
        snippet="red mug on shelf",
        frame_refs=(),
        base_score=100.0,
    )
    future = RetrievalMemoryRecord(
        memory_id="future-hit",
        source_store="semantic",
        video_id="support-video",
        start_time=1901.0,
        end_time=1902.0,
        snippet="red mug on shelf",
        frame_refs=(),
        base_score=1000.0,
    )
    chunks = (
        StreamChunk(
            chunk_id="support-video:0:1800:shard_30m",
            video_id="support-video",
            start_time=0.0,
            end_time=1800.0,
            granularity="shard_30m",
        ),
        StreamChunk(
            chunk_id="support-video:120:150:clip_30s",
            video_id="support-video",
            start_time=120.0,
            end_time=150.0,
            granularity="clip_30s",
        ),
    )

    # When: protocol-aware retrieval runs.
    pack = retrieve_evidence(
        question,
        (support, outside, future),
        enabled_stores=frozenset({"semantic"}),
        options=RetrievalOptions(evidence_budget=3, chunks=chunks),
    )

    # Then: listed sessions are searched, unlisted sessions and future evidence are not.
    memory_ids = {item.memory_id for item in pack.evidence}
    assert "support-hit" in memory_ids
    assert "outside-hit" not in memory_ids
    assert "future-hit" not in memory_ids
    support_item = next(
        item for item in pack.evidence if item.memory_id == "support-hit"
    )
    assert support_item.video_id == "support-video"
    assert pack.causal_filtered_count == 1


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


def test_retrieve_cli_uses_manifest_memory_artifacts(tmp_path: Path) -> None:
    # Given: a memory manifest with one semantic artifact not present in fixture memory.
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    episodic = memory_dir / "episodic.jsonl"
    semantic = memory_dir / "semantic.jsonl"
    visual = memory_dir / "visual.jsonl"
    spatial = memory_dir / "spatial.jsonl"
    _ = episodic.write_text("", encoding="utf-8")
    _ = visual.write_text("", encoding="utf-8")
    _ = spatial.write_text("", encoding="utf-8")
    artifact_memory_id = "semantic:artifact-only:mug"
    _ = semantic.write_text(
        SemanticTripleRecord(
            memory_id=artifact_memory_id,
            video_id="fake_video_001",
            subject="mug",
            predicate="placed",
            object="beside notebook",
            text="mug placed beside notebook",
            support_memory_ids=("artifact-source",),
            support_event_count=1,
            start_time=0.0,
            end_time=30.0,
            confidence=1.0,
            text_embedding_id="embedding:artifact-only",
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "memory_manifest.json"
    _ = manifest.write_text(
        json.dumps(
            {
                "episodic_memory": str(episodic),
                "semantic_memory": str(semantic),
                "visual_memory": str(visual),
                "spatial_memory": {"path": str(spatial), "count": 0},
            },
        ),
        encoding="utf-8",
    )
    output = tmp_path / "artifact_pack.json"

    # When: retrieval is pointed at the manifest.
    result = run_cli(
        "retrieve",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--input",
        str(manifest),
        "--question",
        "q_fake_001",
        "--stores",
        "semantic",
        "--retrieval-protocol",
        "legacy-round-robin",
        "--out",
        str(output),
    )

    # Then: the evidence comes from the artifact, not rebuilt fixture stores.
    assert result.returncode == 0, result.stderr
    pack = EvidencePack.model_validate_json(output.read_text(encoding="utf-8"))
    assert tuple(item.memory_id for item in pack.evidence) == (artifact_memory_id,)


def test_artifact_reader_restores_custom_spatial_codec(tmp_path: Path) -> None:
    # Given: a persisted custom codec whose decoder needs experiment options.
    plugin = tmp_path / "artifact_codec_plugin.py"
    _ = plugin.write_text(
        """
from worldmm_smvqa.worldmm.spatial_compression import (
    ZoneToken,
    register_spatial_memory_codec,
)

class ArtifactCodec:
    name = "artifact-test-v1"

    def __init__(self, options):
        self.zone_id = str(options["zone_id"])

    def encode(self, token):
        return self.zone_id

    def decode(self, record):
        return ZoneToken(
            scale_m=0.25,
            zone_id=self.zone_id,
            x=1.0,
            y=2.0,
            z=3.0,
        )

register_spatial_memory_codec("artifact-test-v1", ArtifactCodec)
""".lstrip(),
        encoding="utf-8",
    )
    experiment_path = tmp_path / "spatial_experiment.json"
    _ = experiment_path.write_text(
        SpatialExperimentConfig(
            codec="artifact-test-v1",
            plugins=("artifact_codec_plugin",),
            codec_options={"zone_id": "plugin-zone"},
        ).model_dump_json(),
        encoding="utf-8",
    )
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    for name in ("episodic", "semantic", "visual"):
        _ = (memory_dir / f"{name}.jsonl").write_text("", encoding="utf-8")
    spatial = memory_dir / "spatial.jsonl"
    _ = spatial.write_text(
        SpatialTokenRecord(
            memory_id="custom-zone",
            video_id="video",
            codec="artifact-test-v1",
            start_time=0.0,
            end_time=1.0,
            token="self-describing-payload",  # noqa: S106
            importance=1.0,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "memory_manifest.json"
    _ = manifest.write_text(
        json.dumps(
            {
                "episodic_memory": str(memory_dir / "episodic.jsonl"),
                "semantic_memory": str(memory_dir / "semantic.jsonl"),
                "visual_memory": str(memory_dir / "visual.jsonl"),
                "spatial_memory": {"path": str(spatial)},
                "spatial_experiment": str(experiment_path),
            },
        ),
        encoding="utf-8",
    )

    # When: retrieval opens artifacts in a fresh artifact-loading path.
    sys.path.insert(0, str(tmp_path))
    try:
        (record,) = read_retrieval_memory_artifacts(manifest)
    finally:
        _ = sys.path.pop(0)
        _ = sys.modules.pop("artifact_codec_plugin", None)

    # Then: plugin and codec options restore before token decoding.
    assert record.memory_id == "custom-zone"
    assert "plugin-zone" in record.snippet
    assert record.geometry == {
        "codec": "artifact-test-v1",
        "encoder": "structured-v1",
        "projection_head": "identity-v1",
        "token_decoder": "delta-topk-v1",
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
    }


def test_retrieve_batch_cli_atomically_writes_all_packs(tmp_path: Path) -> None:
    # Given: a stale output at the production evidence-pack path.
    output = tmp_path / "evidence_packs.jsonl"
    _ = output.write_text("stale\n", encoding="utf-8")

    # When: batch retrieval processes the full question fixture.
    result = run_cli(
        "retrieve-batch",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--stores",
        "episodic,semantic,visual,spatial",
        "--retrieval-protocol",
        "legacy-round-robin",
        "--out",
        str(output),
    )

    # Then: one complete JSONL replacement contains every question exactly once.
    assert result.returncode == 0, result.stderr
    packs = tuple(
        EvidencePack.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    )
    assert len(packs) == 6
    assert len({pack.question_id for pack in packs}) == 6
    assert not tuple(tmp_path.glob(f".{output.name}.*.tmp"))
    assert "evidence_packs=6" in result.stdout


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
    refs = {frame_ref for item in pack.evidence for frame_ref in item.frame_refs}
    assert "fake_video_001_frame_0072" not in refs
