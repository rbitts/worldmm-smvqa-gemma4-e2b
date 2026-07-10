from __future__ import annotations

import os
import subprocess
from pathlib import Path

from worldmm_smvqa.schema import QALabelExample, SourceStreamExample
from worldmm_smvqa.worldmm.spatial import build_object_anchors

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


def test_validate_schema_reports_tiny_fixture_counts() -> None:
    # Given: the checked-in tiny SuperMemory-VQA fixture.
    fixture = "tests/fixtures/tiny_smvqa"

    # When: the user validates the fixture through the CLI.
    result = run_cli("validate-schema", "--input", fixture)

    # Then: validation succeeds and reports the source/QA counts.
    assert result.returncode == 0
    assert "source_examples=2" in result.stdout
    assert "qa_examples=6" in result.stdout


def test_prepare_fixture_recreates_valid_tiny_fixture(tmp_path: Path) -> None:
    # Given: an empty output directory for generated fake fixture data.
    fixture = tmp_path / "tiny_smvqa.generated"

    # When: the user generates then validates the fixture through the CLI.
    prepare = run_cli("prepare-fixture", "--out", str(fixture))
    validate = run_cli("validate-schema", "--input", str(fixture))

    # Then: generated files exist and validate with the expected counts.
    assert prepare.returncode == 0
    assert validate.returncode == 0
    assert (fixture / "sources.jsonl").is_file()
    assert (fixture / "questions.jsonl").is_file()
    assert (fixture / "labels.jsonl").is_file()
    assert "source_examples=2" in validate.stdout
    assert "qa_examples=6" in validate.stdout


def test_tiny_fixture_includes_spatial_and_unanswerable_questions() -> None:
    # Given: the checked-in tiny fixture source and labels.
    fixture = ROOT / "tests/fixtures/tiny_smvqa"
    sources = [
        SourceStreamExample.model_validate_json(line)
        for line in (fixture / "sources.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    labels = [
        QALabelExample.model_validate_json(line)
        for line in (fixture / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    # When: the new spatial and unanswerable rows are selected.
    video_001 = next(
        source for source in sources if source.video_id == "fake_video_001"
    )
    video_002 = next(
        source for source in sources if source.video_id == "fake_video_002"
    )
    q_fake_005 = next(label for label in labels if label.question_id == "q_fake_005")
    q_fake_006 = next(label for label in labels if label.question_id == "q_fake_006")
    video_002_anchors = build_object_anchors(video_002)
    magnet_anchor = next(
        anchor for anchor in video_002_anchors if anchor.object_label == "magnet"
    )

    # Then: long-horizon spatial evidence and causal unanswerability are pinned.
    assert video_001.end_time >= 1900.0
    assert video_001.pose_samples[-1].timestamp >= 1900.0
    assert video_001.gaze_samples[-1].timestamp >= 1900.0
    assert video_001.frame_metadata[-1].timestamp >= 1900.0
    assert q_fake_005.question_time == 1850.0
    assert q_fake_005.answer == "A"
    assert q_fake_005.evidence_list == ("fake_video_001:5:12:spatial",)
    assert q_fake_006.question_time == 15.0
    assert q_fake_006.answer == ""
    assert not q_fake_006.is_answerable
    assert q_fake_006.evidence_list == ()
    assert video_002.pose_samples
    assert video_002.gaze_samples
    assert magnet_anchor.memory_id == "spatial_anchor:fake_video_002:magnet:132"
    assert magnet_anchor.provenance == "gaze"
    assert magnet_anchor.start_time > q_fake_006.question_time
    assert magnet_anchor.end_time > q_fake_006.question_time


def test_validate_schema_fails_when_question_time_missing(tmp_path: Path) -> None:
    # Given: a malformed fixture with a QA label missing question_time.
    fixture = tmp_path / "malformed"
    prepare = run_cli("prepare-fixture", "--out", str(fixture))
    assert prepare.returncode == 0
    malformed_label = (
        '{"question_id":"q_missing_time","video_id":"fake_video_001",'
        '"question":"Where is the fake mug?",'
        '"answer_choices":[{"choice_id":"A","text":"desk","choice_ltype":"place"}],'
        '"answer":"A","is_answerable":true,"evidence_list":["fake_video_001:0:30"],'
        '"verification_score":1.0}\n'
    )
    _ = (fixture / "labels.jsonl").write_text(malformed_label, encoding="utf-8")

    # When: the user validates the malformed fixture.
    result = run_cli("validate-schema", "--input", str(fixture))

    # Then: validation fails and names the missing boundary field.
    assert result.returncode != 0
    assert "question_time" in result.stderr


def test_validate_schema_rejects_question_label_drift(tmp_path: Path) -> None:
    # Given: valid fixture whose public question text no longer matches its label.
    fixture = tmp_path / "drifted"
    prepare = run_cli("prepare-fixture", "--out", str(fixture))
    assert prepare.returncode == 0
    questions = (fixture / "questions.jsonl").read_text(encoding="utf-8")
    _ = (fixture / "questions.jsonl").write_text(
        questions.replace(
            "Where is the fake mug placed?",
            "Where did the fake mug move?",
            1,
        ),
        encoding="utf-8",
    )

    # When: schema validation checks all three files together.
    result = run_cli("validate-schema", "--input", str(fixture))

    # Then: cross-file drift fails before memory generation.
    assert result.returncode != 0
    assert "question fields differ from label" in result.stderr


def test_validate_schema_rejects_empty_required_file(tmp_path: Path) -> None:
    # Given: generated fixture whose questions file was truncated.
    fixture = tmp_path / "empty"
    prepare = run_cli("prepare-fixture", "--out", str(fixture))
    assert prepare.returncode == 0
    _ = (fixture / "questions.jsonl").write_text("", encoding="utf-8")

    # When / Then: preflight fails before expensive remote stages.
    result = run_cli("validate-schema", "--input", str(fixture))
    assert result.returncode != 0
    assert "questions.jsonl: file has no records" in result.stderr
