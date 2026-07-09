from __future__ import annotations

import os
import subprocess
from pathlib import Path

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
    assert "qa_examples=4" in result.stdout


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
    assert "qa_examples=4" in validate.stdout


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
