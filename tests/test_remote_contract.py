from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from worldmm_smvqa.schema import PredictionRecord

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ENV_NAMES = frozenset(
    {
        "SMVQA_DATA_ROOT",
        "GEMMA_MODEL_PATH",
        "WORLDMM_OUTPUT_ROOT",
        "BASTION_HOST",
        "HEAD_NODE",
        "REMOTE_JOB_LAUNCHER",
    },
)


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in REMOTE_ENV_NAMES | {
        "WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST",
        "WORLDMM_SMVQA_REMOTE_APPROVED",
    }:
        _ = env.pop(name, None)
    return subprocess.run(
        ["uv", "run", "--offline", "worldmm-smvqa", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_qa_transformers_mock_cli_writes_predictions_from_evidence(
    tmp_path: Path,
) -> None:
    # Given: evidence packs generated locally without model weights.
    smoke_dir = tmp_path / "smoke"
    smoke = run_cli(
        "smoke",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--out",
        str(smoke_dir),
    )
    assert smoke.returncode == 0, smoke.stderr
    predictions = tmp_path / "predictions.jsonl"

    # When: the remote QA module command is probed in mock mode.
    result = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "python",
            "-m",
            "worldmm_smvqa.qa_transformers",
            "--model",
            "remote-gemma-placeholder",
            "--fixture",
            "tests/fixtures/tiny_smvqa",
            "--evidence",
            str(smoke_dir / "evidence_packs.jsonl"),
            "--out",
            str(predictions),
            "--backend",
            "mock",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "UV_NO_NETWORK": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    # Then: predictions are actually written.
    assert result.returncode == 0, result.stderr
    rows = [
        PredictionRecord.model_validate_json(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 4
    assert {row.question_id for row in rows} == {
        "q_fake_001",
        "q_fake_002",
        "q_fake_003",
        "q_fake_004",
    }


def test_qa_transformers_mock_cli_shards_and_merges_ddp_predictions(
    tmp_path: Path,
) -> None:
    # Given: evidence packs generated locally without model weights.
    smoke_dir = tmp_path / "smoke"
    smoke = run_cli(
        "smoke",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--out",
        str(smoke_dir),
    )
    assert smoke.returncode == 0, smoke.stderr
    predictions = tmp_path / "predictions.jsonl"
    command = [
        "uv",
        "run",
        "--offline",
        "python",
        "-m",
        "worldmm_smvqa.qa_transformers",
        "--model",
        "remote-gemma-placeholder",
        "--fixture",
        "tests/fixtures/tiny_smvqa",
        "--evidence",
        str(smoke_dir / "evidence_packs.jsonl"),
        "--out",
        str(predictions),
        "--backend",
        "mock",
    ]

    # When: nonzero rank writes first, then rank zero writes and merges.
    rank_one = subprocess.run(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "RANK": "1",
            "UV_NO_NETWORK": "1",
            "WORLD_SIZE": "2",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    rank_zero = subprocess.run(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "RANK": "0",
            "UV_NO_NETWORK": "1",
            "WORLD_SIZE": "2",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    # Then: each rank used a distinct shard and rank zero merged final output.
    assert rank_one.returncode == 0, rank_one.stderr
    assert rank_zero.returncode == 0, rank_zero.stderr
    assert (tmp_path / "predictions.rank00001-of00002.jsonl").is_file()
    assert (tmp_path / "predictions.rank00000-of00002.jsonl").is_file()
    rows = [
        PredictionRecord.model_validate_json(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    assert [row.question_id for row in rows] == [
        "q_fake_001",
        "q_fake_002",
        "q_fake_003",
        "q_fake_004",
    ]


def test_remote_plan_script_compacts_jsonl_and_writes_memory_manifest(
    tmp_path: Path,
) -> None:
    # Given: a generated remote plan.
    out_dir = tmp_path / "remote_plan"
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )
    assert result.returncode == 0, result.stderr

    # When: the remote script text is inspected.
    script_text = (out_dir / "run_worldmm_smvqa.sh").read_text(encoding="utf-8")

    # Then: it writes one compact JSON object per line and the declared manifest.
    assert "json.dumps(payload, separators=(',', ':'))" in script_text
    raw_cat = 'cat "$tmp" >> "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl"'
    assert raw_cat not in script_text
    assert "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json" in script_text
    assert "source_memories.jsonl" in script_text
    assert "worldmm_sv/semantic.jsonl" in script_text
    assert "worldmm_sv/visual.jsonl" in script_text


def test_remote_plan_uses_hashed_question_tmp_paths(tmp_path: Path) -> None:
    # Given: a malicious question id that would escape retrieval with raw paths.
    question_id = "../escape/id"
    digest = subprocess.run(
        [
            "python3",
            "-c",
            (
                "import hashlib,sys; "
                "print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:16])"
            ),
            question_id,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert digest.returncode == 0, digest.stderr

    # When: the generated script is read.
    out_dir = tmp_path / "remote_plan"
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )
    assert result.returncode == 0, result.stderr
    script_text = (out_dir / "run_worldmm_smvqa.sh").read_text(encoding="utf-8")

    # Then: temp retrieval paths are derived from a hash, not raw question text.
    assert "hashlib.sha256" in script_text
    assert 'tmp="$WORLDMM_OUTPUT_ROOT/retrieval/${question_id}.json"' not in script_text
    safe_path = Path("retrieval") / f"q_{digest.stdout.strip()}.json"
    assert ".." not in safe_path.parts
    assert "/" not in safe_path.name


def test_plan_stdout_shell_quotes_script_path(tmp_path: Path) -> None:
    # Given: an out path with quotes and shell metacharacters.
    out_dir = tmp_path / "remote plan' ; touch bad $(uname)"

    # When: launch-remote prints the copy/paste command.
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    # Then: the rendered command parses to a safe argv without eval.
    assert result.returncode == 0, result.stderr
    script = out_dir / "run_worldmm_smvqa.sh"
    command_line = result.stdout.splitlines()[-1]
    argv = shlex.split(command_line)
    assert argv[:4] == [
        "ssh",
        "$BASTION_HOST",
        "$REMOTE_JOB_LAUNCHER",
        "$HEAD_NODE",
    ]
    assert shlex.split(argv[4]) == ["bash", str(script)]
    assert f"bash {script}" not in result.stdout


def test_checked_in_remote_script_delegates_without_printing_paths() -> None:
    # Given: the checked-in remote helper is user-facing.
    script_text = (ROOT / "scripts/remote/run_worldmm_smvqa.sh").read_text(
        encoding="utf-8",
    )

    # Then: it delegates to the generated plan and does not print sensitive paths.
    assert "worldmm-smvqa launch-remote --dry-run" in script_text
    assert "printf 'WORLDMM_OUTPUT_ROOT=%s" not in script_text
    assert "printf 'GEMMA_MODEL_PATH=%s" not in script_text
