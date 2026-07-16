from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from worldmm_smvqa.memory_alignment_config import (
    MemoryAlignmentConfigError,
    load_memory_alignment_config,
)
from worldmm_smvqa.worldmm.llm_memory_io import memory_bindings

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/memory_alignment.example.yaml"


def test_reviewed_memory_config_binds_exact_contract() -> None:
    config = load_memory_alignment_config(CONFIG, ROOT)

    assert config.model_variant == "Gemma-4-E2B-IT"
    assert config.contract_relative_path == (
        "configs/spatial/model_boundary_contract_v2.json"
    )
    assert config.contract_path.is_file()


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("location: remote", "location: local", "must be remote"),
        (
            "${WORLDMM_MEMORY_MODEL_PATH}",
            "${GEMMA_MODEL_PATH}",
            "model_path mismatch",
        ),
        (
            "6d7ad8546e63bd5d25260c8a6b5fe04e9bb23a424e726a09a44946435016fe77",
            "1" * 64,
            "must be reviewed",
        ),
    ],
)
def test_memory_config_rejects_unreviewed_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    config = tmp_path / "memory.yaml"
    _ = config.write_text(
        CONFIG.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )

    with pytest.raises(MemoryAlignmentConfigError, match=message):
        _ = load_memory_alignment_config(config, ROOT)


def test_memory_binding_uses_only_reviewed_model_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, int]] = []

    def generate(prompt: str, model_ref: str, frames: tuple[object, ...]) -> str:
        calls.append((prompt, model_ref, len(frames)))
        return "generated"

    monkeypatch.setattr(
        "worldmm_smvqa.transformers_backend.generate_transformers_multimodal",
        generate,
    )
    bindings = memory_bindings({"WORLDMM_MEMORY_MODEL_PATH": "/reviewed/gemma"})

    assert bindings.generate("prompt") == "generated"
    assert calls == [("prompt", "/reviewed/gemma", 0)]


def test_memory_cli_fails_before_output_when_model_binding_is_missing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "memory.jsonl"
    env = os.environ.copy()
    env["WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST"] = "1"
    _ = env.pop("WORLDMM_MEMORY_MODEL_PATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "worldmm_smvqa.cli",
            "build-memory",
            "--config",
            str(CONFIG),
            "--backend",
            "memory",
            "--store",
            "episodic",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "missing WORLDMM_MEMORY_MODEL_PATH" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_memory_cli_rejects_spatial_before_config_or_output(tmp_path: Path) -> None:
    output = tmp_path / "memory"
    missing_config = tmp_path / "missing.yaml"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "worldmm_smvqa.cli",
            "build-memory",
            "--config",
            str(missing_config),
            "--backend",
            "memory",
            "--store",
            "spatial",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "unsupported build-memory --backend memory store: spatial" in result.stderr
    assert "ConfigNotFound" not in result.stderr
    assert not output.exists()
