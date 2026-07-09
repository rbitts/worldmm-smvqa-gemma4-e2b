from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from worldmm_smvqa.memory_sources import build_source_memories
from worldmm_smvqa.schema import (
    PROHIBITED_MEMORY_FIELDS,
    AnswerChoice,
    FrameMetadata,
    LeakageError,
    MemoryRecord,
    ObjectMetadata,
    OCRMetadata,
    QALabelExample,
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


def memory_chunk() -> StreamChunk:
    return StreamChunk(
        chunk_id="video-memory:0:30:clip_30s",
        video_id="video-memory",
        start_time=0.0,
        end_time=30.0,
        granularity="clip_30s",
        transcript_spans=(
            TranscriptSpan(
                start_time=5.0,
                end_time=9.0,
                text="The cup is next to the notebook.",
            ),
        ),
        captions=("Synthetic caption of a desk.",),
        ocr_entries=(
            OCRMetadata(
                start_time=6.0,
                end_time=7.0,
                text="NOTE-7",
                frame_ref="frame_0006",
            ),
        ),
        object_detections=(
            ObjectMetadata(start_time=8.0, end_time=10.0, label="cup", confidence=0.9),
        ),
        frame_metadata=(
            FrameMetadata(
                frame_ref="frame_0008",
                timestamp=8.0,
                description="Desk frame with cup.",
            ),
        ),
    )


def label_example() -> QALabelExample:
    return QALabelExample(
        question_id="q-leak",
        video_id="video-memory",
        question="Where is the cup?",
        question_time=40.0,
        answer_choices=(
            AnswerChoice(choice_id="A", text="notebook", choice_ltype="place"),
            AnswerChoice(choice_id="B", text="sink", choice_ltype="place"),
        ),
        answer="A",
        is_answerable=True,
        evidence_list=("SECRET_EVIDENCE_ID",),
        verification_score=1.0,
    )


def test_source_memory_builders_emit_allowed_modalities() -> None:
    # Given: one source chunk containing only source-stream modality fields.
    chunk = memory_chunk()

    # When: source memories are built.
    memories = build_source_memories((chunk,))

    # Then: each allowed source modality produces a grounded memory record.
    assert {memory.store for memory in memories} == {
        "caption",
        "transcript",
        "ocr",
        "object",
        "frame",
    }
    assert {memory.source_chunk_id for memory in memories} == {chunk.chunk_id}
    assert any(memory.text == "Synthetic caption of a desk." for memory in memories)
    assert any(memory.text == "The cup is next to the notebook." for memory in memories)
    assert any(memory.text == "OCR NOTE-7" for memory in memories)
    assert any(memory.text == "object cup confidence=0.900" for memory in memories)
    assert any(memory.frame_refs == ("frame_0008",) for memory in memories)


def test_source_memory_builders_reject_label_examples() -> None:
    # Given: evaluator-only labels at the memory-builder boundary.
    label = label_example()

    # When / Then: source memory construction rejects the label object.
    with pytest.raises(LeakageError, match="QALabelExample"):
        _ = build_source_memories((label,))


def test_source_memory_output_excludes_eval_only_labels() -> None:
    # Given: a malicious label value that must never become memory text or refs.
    label = label_example()

    # When: source memories are built from a valid source chunk.
    output = "\n".join(
        memory.model_dump_json() for memory in build_source_memories((memory_chunk(),))
    )

    # Then: prohibited label keys and values are absent from output.
    for field in PROHIBITED_MEMORY_FIELDS:
        key = field.rsplit(".", maxsplit=1)[-1]
        assert f'"{key}":' not in output
    assert label.answer not in output
    assert label.evidence_list[0] not in output
    assert label.answer_choices[0].choice_ltype not in output


def test_build_memory_source_memories_cli_writes_memory_records(tmp_path: Path) -> None:
    # Given: the checked-in tiny fixture and an output JSONL path.
    output = tmp_path / "source_memories.jsonl"

    # When: source-memory stage is driven through the CLI.
    result = run_cli(
        "build-memory",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--stage",
        "source-memories",
        "--out",
        str(output),
    )

    # Then: the CLI succeeds and writes all allowed memory source types.
    assert result.returncode == 0, result.stderr
    memories = [
        MemoryRecord.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {"caption", "transcript", "ocr", "object", "frame"} <= {
        memory.store for memory in memories
    }
    assert all(memory.source_chunk_id is not None for memory in memories)
    assert "source_memories" in result.stdout


def test_build_memory_source_memories_rejects_eval_fields(
    tmp_path: Path,
) -> None:
    # Given: a fixture whose source row has injected evaluator-only labels.
    fixture = tmp_path / "fixture"
    prepare = run_cli("prepare-fixture", "--out", str(fixture))
    assert prepare.returncode == 0
    lines = (fixture / "sources.jsonl").read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0][:-1] + ',"answer":"SECRET","evidence_list":["SECRET"]}'
    _ = (fixture / "sources.jsonl").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # When: source-memory stage is driven through the CLI.
    result = run_cli(
        "build-memory",
        "--fixture",
        str(fixture),
        "--stage",
        "source-memories",
        "--out",
        str(tmp_path / "source_memories.jsonl"),
    )

    # Then: schema parsing rejects the injected eval-only data.
    assert result.returncode != 0
    assert "Extra inputs are not permitted" in result.stderr
