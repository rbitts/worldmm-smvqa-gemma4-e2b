from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ENV_NAMES = frozenset(
    {
        "SMVQA_DATA_ROOT",
        "GEMMA_MODEL_PATH",
        "WORLDMM_OUTPUT_ROOT",
        "BASTION_HOST",
        "HEAD_NODE",
    },
)


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in REMOTE_ENV_NAMES | {"WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST"}:
        _ = env.pop(name, None)
    return subprocess.run(
        ["uv", "run", "worldmm-smvqa", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launch_remote_dry_run_prints_bastion_ssh(tmp_path: Path) -> None:
    # Given: remote dry-run is configured with an output artifact directory.
    out_dir = tmp_path / "remote_plan"

    # When: the launch command is rendered.
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    # Then: CLI prints the staged DAG command and opens no connection.
    assert result.returncode == 0
    assert 'ssh -J "$BASTION_HOST" "$HEAD_NODE"' in result.stdout
    assert "bash remote-plan/submit_worldmm_smvqa_dag.sh" in result.stdout
    assert "WORLDMM_SMVQA_REMOTE_APPROVED=1" not in result.stdout
    assert "legacy single-job compatibility" in result.stdout
    assert "dry-run" in result.stdout
    default_repo = "/repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b"
    assert (
        f'"$HEAD_NODE:${{WORLDMM_REMOTE_REPO:-{default_repo}}}/"'
        in result.stdout
    )
    assert (
        f'"$HEAD_NODE:${{WORLDMM_REMOTE_REPO:-{default_repo}}}/remote-plan/"'
        in result.stdout
    )
    assert result.stdout.count("--exclude '.env*'") == 2
    assert (out_dir / "run_worldmm_smvqa.sh").is_file()


def test_launch_remote_dry_run_requires_worldmm_output_root_config(
    tmp_path: Path,
) -> None:
    # Given: WORLDMM_OUTPUT_ROOT is absent from the remote config template.
    config = tmp_path / "remote.yaml"
    _ = config.write_text(
        """runtime:
  location: remote
remote:
  bastion_host: ${BASTION_HOST}
  head_node: ${HEAD_NODE}
  data_root: ${SMVQA_DATA_ROOT}
  model_path: ${GEMMA_MODEL_PATH}
""",
        encoding="utf-8",
    )

    # When: the launch dry-run is requested.
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        str(config),
        "--out",
        str(tmp_path / "remote_plan"),
    )

    # Then: the boundary reports the missing remote config value.
    assert result.returncode != 0
    assert "MissingRemoteConfig: WORLDMM_OUTPUT_ROOT" in result.stderr


def test_remote_example_uses_only_expected_remote_env_placeholders() -> None:
    # Given: the remote example config is the user-facing template.
    # When: the placeholder names are read.
    config_text = (ROOT / "configs/remote.example.yaml").read_text(encoding="utf-8")

    # Then: only Todo 9 env placeholders are present.
    placeholders = {
        part.split("}", 1)[0]
        for part in config_text.split("${")[1:]
    }
    assert placeholders == REMOTE_ENV_NAMES
