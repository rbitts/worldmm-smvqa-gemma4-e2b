from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cli(
    *args: str,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    _ = env.pop("WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST", None)
    if env_overrides is not None:
        env.update(env_overrides)
    return subprocess.run(
        ["uv", "run", "worldmm-smvqa", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_lists_scaffold_commands() -> None:
    # Given: a newly scaffolded benchmark CLI.
    # When: the user asks for top-level help.
    result = run_cli("--help")

    # Then: every planned command surface is discoverable.
    assert result.returncode == 0
    for command in (
        "prepare-fixture",
        "validate-schema",
        "preflight",
        "audit-sensors",
        "build-memory",
        "retrieve",
        "retrieve-batch",
        "qa",
        "evaluate",
        "diagnose-spatial",
        "report",
        "mock-dag",
        "smoke",
        "launch-remote",
        "validate-teacher-oracle-inputs",
    ):
        assert command in result.stdout


def test_command_help_lists_exact_guarded_contracts() -> None:
    expected_help = {
        "audit-sensors": (
            "usage: worldmm-smvqa audit-sensors --sensor-manifest SENSOR_MANIFEST "
            "--observations OBSERVATIONS --frame-root FRAME_ROOT --out OUT\n\n"
            "Audit causal sensor coverage.\n"
        ),
        "validate-teacher-oracle-inputs": (
            "usage: worldmm-smvqa validate-teacher-oracle-inputs "
            "--sensor-audit SENSOR_AUDIT --experiment-config EXPERIMENT_CONFIG "
            "--out OUT\n\nValidate teacher-oracle production inputs.\n"
        ),
        "mock-dag": (
            "usage: worldmm-smvqa mock-dag [--config CONFIG] --fixture FIXTURE "
            "[--student-architecture STUDENT_ARCHITECTURE]\n\n"
            "Validate the model-free production-consumer DAG.\n"
        ),
        "launch-remote": (
            "usage: worldmm-smvqa launch-remote [--config CONFIG] --out OUT "
            "--profile PROFILE --experiment-config EXPERIMENT_CONFIG "
            "(--dry-run | --submit)\n\nPrint remote commands.\n"
        ),
    }
    for command, help_text in expected_help.items():
        result = run_cli(command, "--help")

        assert result.returncode == 0
        assert result.stdout == help_text


def test_launch_remote_dry_run_prints_commands_without_submit(tmp_path: Path) -> None:
    # Given: remote config that uses environment placeholders.
    out_dir = tmp_path / "remote_plan"

    # When: launch-remote is run in dry-run mode.
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--profile",
        "teacher-oracle",
        "--experiment-config",
        "configs/spatial/exp_0005_teacher_oracle.example.json",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    # Then: it prints the remote command and does not submit work.
    assert result.returncode == 0
    assert "ssh -J" not in result.stdout
    assert "dry-run" in result.stdout
    assert (out_dir / "expected_outputs.json").is_file()


def test_launch_remote_requires_reviewed_teacher_oracle_arguments(
    tmp_path: Path,
) -> None:
    output = tmp_path / "remote-plan"
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(output),
    )

    assert result.returncode == 2
    assert "--profile teacher-oracle" in result.stderr
    assert not output.exists()


def test_qa_real_model_is_remote_only_on_local_config() -> None:
    # Given: local config and no explicit remote override.
    # When: real-model QA is requested.
    result = run_cli("qa", "--config", "configs/local.example.yaml", "--real-model")

    # Then: the guardrail refuses before any model work can start.
    assert result.returncode != 0
    assert "remote-only" in result.stderr


def test_missing_config_path_fails_before_command_work() -> None:
    # Given: a config path that does not exist.
    # When: a command tries to load it.
    result = run_cli("smoke", "--config", "configs/missing.yaml")

    # Then: the CLI reports the bad path and exits nonzero.
    assert result.returncode != 0
    assert "ConfigNotFound" in result.stderr


def test_build_memory_invalid_stage_fails_nonzero(tmp_path: Path) -> None:
    # Given: an unsupported build-memory stage and requested artifact path.
    output = tmp_path / "bad.jsonl"

    # When: the command is run.
    result = run_cli("build-memory", "--stage", "bad", "--out", str(output))

    # Then: it fails with a typed usage error and writes no artifact.
    assert result.returncode != 0
    assert "UsageError" in result.stderr
    assert not output.exists()


def test_build_memory_invalid_store_fails_nonzero(tmp_path: Path) -> None:
    # Given: an unsupported build-memory store and requested artifact directory.
    output = tmp_path / "bad"

    # When: the command is run.
    result = run_cli("build-memory", "--store", "bad", "--out", str(output))

    # Then: it fails with a typed usage error and writes no artifact.
    assert result.returncode != 0
    assert "UsageError" in result.stderr
    assert not output.exists()


def test_teacher_oracle_validation_requires_all_inputs(tmp_path: Path) -> None:
    result = run_cli(
        "validate-teacher-oracle-inputs",
        "--sensor-audit",
        str(tmp_path / "audit.json"),
        "--out",
        str(tmp_path / "report.json"),
    )

    assert result.returncode != 0
    assert "experiment-config" in result.stderr
    assert not (tmp_path / "report.json").exists()


def test_guarded_commands_reject_cross_command_options_before_work(
    tmp_path: Path,
) -> None:
    cases = (
        ("audit-sensors", ("--sensor-audit", str(tmp_path / "audit.json"))),
        (
            "validate-teacher-oracle-inputs",
            ("--sensor-manifest", str(tmp_path / "sensor-manifest.json")),
        ),
        ("launch-remote", ("--frame-root", str(tmp_path / "frames"))),
    )
    for command, irrelevant_option in cases:
        result = run_cli(command, *irrelevant_option)

        assert result.returncode == 2
        assert f"{command} does not accept {irrelevant_option[0]}" in result.stderr
        assert "Traceback" not in result.stderr


def test_launch_remote_publication_failure_is_a_typed_cli_diagnostic(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "remote-plan"
    _ = output_file.write_text("not a plan directory", encoding="utf-8")

    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--profile",
        "teacher-oracle",
        "--experiment-config",
        "configs/spatial/exp_0005_teacher_oracle.example.json",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(output_file),
    )

    assert result.returncode == 2
    assert "CliPublicationError: launch-remote could not publish" in result.stderr
    assert "Traceback" not in result.stderr
    assert output_file.read_text(encoding="utf-8") == "not a plan directory"


def test_value_options_reject_following_option_tokens_before_command_work(
    tmp_path: Path,
) -> None:
    for trailing_options in (
        ("--dry-run", "--submit"),
        ("--submit", "--dry-run"),
    ):
        output = tmp_path / "-".join(
            option.removeprefix("--") for option in trailing_options
        )
        result = run_cli(
            "launch-remote",
            "--out",
            trailing_options[0],
            trailing_options[1],
        )

        assert result.returncode == 2
        assert "missing value for --out" in result.stderr
        assert not output.exists()


def test_build_memory_rejects_empty_normalized_store_before_creating_output(
    tmp_path: Path,
) -> None:
    for index, store in enumerate(("", " ", ",,,")):
        output = tmp_path / f"empty-store-{index}"
        result = run_cli("build-memory", "--store", store, "--out", str(output))

        assert result.returncode == 2
        assert "at least one non-empty" in result.stderr
        assert not output.exists()


def test_preflight_subprocess_writes_report_and_missing_input_writes_nothing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "preflight.json"
    result = run_cli(
        "preflight",
        "--input",
        "tests/fixtures/tiny_smvqa",
        "--out",
        str(output),
    )

    assert result.returncode == 0
    assert output.is_file()
    assert '"status"' in output.read_text(encoding="utf-8")

    missing_output = tmp_path / "missing-preflight.json"
    missing = run_cli("preflight", "--out", str(missing_output))
    assert missing.returncode == 2
    assert not missing_output.exists()


def test_audit_sensors_subprocess_writes_report_and_missing_input_writes_nothing(
    tmp_path: Path,
) -> None:
    manifest, observations, frame_root = _write_ready_sensor_inputs(tmp_path)
    output = tmp_path / "sensor-audit.json"
    result = run_cli(
        "audit-sensors",
        "--sensor-manifest",
        str(manifest),
        "--observations",
        str(observations),
        "--frame-root",
        str(frame_root),
        "--out",
        str(output),
    )

    assert result.returncode == 0
    assert output.is_file()
    assert '"operational_state": "ready"' in output.read_text(encoding="utf-8")

    missing_output = tmp_path / "missing-sensor-audit.json"
    missing = run_cli(
        "audit-sensors",
        "--sensor-manifest",
        str(manifest),
        "--out",
        str(missing_output),
    )
    assert missing.returncode == 2
    assert not missing_output.exists()


def test_teacher_oracle_validation_subprocess_writes_failure_report(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit.json"
    config = tmp_path / "experiment.json"
    output = tmp_path / "oracle-preflight.json"
    _ = audit.write_text("{}", encoding="utf-8")
    _ = config.write_text("{}", encoding="utf-8")

    result = run_cli(
        "validate-teacher-oracle-inputs",
        "--sensor-audit",
        str(audit),
        "--experiment-config",
        str(config),
        "--out",
        str(output),
    )

    assert result.returncode == 1
    assert output.is_file()
    assert '"status": "fail"' in output.read_text(encoding="utf-8")


def test_teacher_oracle_plan_requirement_is_a_typed_cli_diagnostic(
    tmp_path: Path,
) -> None:
    output = tmp_path / "remote-plan"
    result = run_cli(
        "launch-remote",
        "--submit",
        "--profile",
        "teacher-oracle",
        "--experiment-config",
        "configs/spatial/exp_0005_teacher_oracle.example.json",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(output),
        env_overrides={"WORLDMM_SMVQA_REMOTE_APPROVED": "1"},
    )

    assert result.returncode == 2
    assert "TeacherOraclePlanRequired" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def _write_ready_sensor_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    rgb = b"P6\n1 1\n255\n\x00\x00\x00"
    manifest = tmp_path / "manifest.jsonl"
    observations = tmp_path / "observations.jsonl"
    frame_root = tmp_path / "frames"
    _ = (frame_root / "video-a").mkdir(parents=True)
    _ = (frame_root / "video-a" / "frame.ppm").write_bytes(rgb)
    _ = manifest.write_text(
        json.dumps(
            {
                "video_id": "video-a",
                "cadence_origin": 0.0,
                "source_frame_count": 1,
                "source_frame_sha256": "0" * 64,
                "selected_frames": [
                    {"sample_index": 0, "frame_ref": "frame.ppm", "timestamp": 0.0}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _ = observations.write_text(
        json.dumps(
            {
                "observation_id": "observation-1",
                "video_id": "video-a",
                "timestamp": 0.0,
                "frame_ref": "frame.ppm",
                "local_frame_id": "device",
                "intrinsics": {
                    "width_px": 1,
                    "height_px": 1,
                    "fx": 1.0,
                    "fy": 1.0,
                    "cx": 0.0,
                    "cy": 0.0,
                },
                "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, observations, frame_root
