from __future__ import annotations

from pathlib import Path

from worldmm_smvqa.config import load_config
from worldmm_smvqa.fixtures import FixtureCounts, validate_fixture, write_tiny_fixture


def prepare_fixture_stdout(config: Path, output: Path | None) -> str:
    _config = load_config(config)
    output_dir = output or Path("tests/fixtures/tiny_smvqa.generated")
    counts = write_tiny_fixture(output_dir)
    return f"wrote {output_dir}\n{format_fixture_counts(counts)}\n"


def validate_schema_stdout(config: Path, input_dir: Path | None) -> str:
    _config = load_config(config)
    fixture_dir = input_dir or Path("tests/fixtures/tiny_smvqa")
    counts = validate_fixture(fixture_dir)
    return f"{format_fixture_counts(counts)}\n"


def format_fixture_counts(counts: FixtureCounts) -> str:
    return f"source_examples={counts.source_examples} qa_examples={counts.qa_examples}"
