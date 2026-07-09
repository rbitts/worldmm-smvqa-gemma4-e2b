from __future__ import annotations

import os
import subprocess
from pathlib import Path

from worldmm_smvqa.fixtures import tiny_fixture_examples
from worldmm_smvqa.retrieval import (
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
        evidence_budget=6,
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
        evidence_budget=3,
    )

    # Then: only visual evidence appears.
    assert pack.evidence
    assert pack.selected_stores == ("visual",)
    assert {item.source_store for item in pack.evidence} == {"visual"}


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
