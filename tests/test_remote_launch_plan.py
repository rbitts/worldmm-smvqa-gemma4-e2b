from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic import TypeAdapter

from worldmm_smvqa.remote_plan import canonical_student_run_graph
from worldmm_smvqa.remote_script import (
    student_stage_script_text,
    student_submit_script_text,
)

if TYPE_CHECKING:
    import pytest

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
type JsonValue = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
_JSON_OBJECT_ADAPTER: TypeAdapter[object] = TypeAdapter(object)
_JSON_MAPPING_ADAPTER: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])
_JSON_SEQUENCE_ADAPTER: TypeAdapter[list[object]] = TypeAdapter(list[object])


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        mapping = _JSON_MAPPING_ADAPTER.validate_python(value)
        return {key: _json_value(item) for key, item in mapping.items()}
    if isinstance(value, list):
        sequence = _JSON_SEQUENCE_ADAPTER.validate_python(value)
        return [_json_value(item) for item in sequence]
    detail = f"unsupported JSON value type: {type(value).__name__}"
    raise AssertionError(detail)


def _read_json_object(path: Path) -> dict[str, JsonValue]:
    raw = _JSON_OBJECT_ADAPTER.validate_json(path.read_text())
    value = _json_value(raw)
    assert isinstance(value, dict), "expected a JSON object"
    return value


def _string_mapping(value: JsonValue) -> dict[str, str]:
    assert isinstance(value, dict), "expected a JSON object with string values"
    result: dict[str, str] = {}
    for key, item in value.items():
        assert isinstance(item, str), "expected a JSON object with string values"
        result[key] = item
    return result


def _json_mapping(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict), "expected a JSON object"
    return value


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in REMOTE_ENV_NAMES | {
        "WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST",
        "WORLDMM_SMVQA_REMOTE_APPROVED",
    }:
        _ = env.pop(name, None)
    return subprocess.run(
        ["uv", "run", "worldmm-smvqa", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launch_remote_writes_only_phased_dag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLDMM_EXECUTION_PROFILE", "teacher-oracle")
    out_dir = tmp_path / "remote_plan"

    result = run_cli(
        "launch-remote",
        "--profile",
        "teacher-oracle",
        "--experiment-config",
        "configs/spatial/exp_0005_teacher_oracle.example.json",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    assert result.returncode == 0, result.stderr
    _assert_generated_scripts(out_dir)
    _assert_expected_outputs(out_dir)
    _assert_operator_contract(out_dir)
    _assert_dry_run_artifacts(out_dir, result)


def _assert_generated_scripts(out_dir: Path) -> None:
    scripts = (
        out_dir / "submit_teacher_oracle_preflight.sh",
        out_dir / "submit_teacher_oracle_provider_gate.sh",
        out_dir / "submit_teacher_oracle_downstream.sh",
        out_dir / "run_teacher_oracle_stage.sh",
    )
    for script in scripts:
        assert script.is_file()
        proof = subprocess.run(
            ["bash", "-n", str(script)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert proof.returncode == 0, proof.stderr
    assert not (out_dir / "run_worldmm_smvqa.sh").exists()
    assert not (out_dir / "submit_worldmm_smvqa_dag.sh").exists()
    assert not (out_dir / "run_worldmm_smvqa_stage.sh").exists()


def _assert_expected_outputs(out_dir: Path) -> None:
    expected = _read_json_object(out_dir / "expected_outputs.json")
    outputs = _string_mapping(expected["outputs"])
    conditional_outputs = _string_mapping(expected["conditional_outputs"])
    assert outputs["sensor_audit"].endswith("/diagnostics/sensor_audit.json")
    assert conditional_outputs["sealed_continue_receipt_if_provider_go"].endswith(
        "/summary/teacher_oracle_continue.json",
    )
    assert conditional_outputs["provider_gate_terminal_on_every_branch"].endswith(
        "/summary/teacher_oracle_terminal.json",
    )
    assert conditional_outputs["phase_b_report_after_second_approval"].endswith(
        "/summary/final_report.md"
    )
    assert {"teacher_job_manifest", "preflight_job_manifest"} <= outputs.keys()
    assert "phase_b_outputs_after_second_approval" in conditional_outputs
    assert cast("str", expected["remote_job_reference"]).endswith("#FINALIZE_JOB_ID")
    assert outputs["preflight_job_manifest"] != outputs["teacher_job_manifest"]
    assert conditional_outputs["phase_b_outputs_after_second_approval"].endswith(
        "metrics.json"
    )
    assert "phase_b_job_manifest_after_second_approval" in conditional_outputs
    assert "student" not in json.dumps(outputs).lower()
    assert "E1" not in json.dumps(outputs)


def _assert_operator_contract(out_dir: Path) -> None:
    contract = _read_json_object(out_dir / "operator_contract.json")
    assert contract["execution_profile"] == "teacher-oracle"
    assert contract["variants"] == ["E0", "T0", "T1"]
    sensor_audit = _json_mapping(contract["sensor_audit"])
    assert sensor_audit["window_microseconds"] == 30_000_000
    stage_specs = contract["stage_specs"]
    assert isinstance(stage_specs, list)
    assert len(stage_specs) == 17
    assert contract["resource_schema"] == "ExperimentGraphV1.stage_specs[].resources"
    assert contract["validation"] == {
        "graph_model": "ExperimentGraphV1",
        "capability_model": "CapabilitySpecV1",
        "resource_model": "ResourceSpecV1",
        "dependency_model": "DependencySpecV1",
        "stage_model": "StageSpecV1",
    }
    generated_inputs = _json_mapping(contract["generated_script_inputs"])
    assert generated_inputs["stages"] == stage_specs
    operations = contract["operations"]
    assert isinstance(operations, list)
    _assert_operation_rows(operations)
    go_branch = _json_mapping(contract["go_branch_artifacts"])
    assert "both artifacts" in cast("str", go_branch["rule"])
    _assert_handoff_operations(operations)


def _assert_operation_rows(operations: list[JsonValue]) -> None:
    assert [
        operation["step_id"] for operation in operations if isinstance(operation, dict)
    ] == [
        "local-dry-run",
        "immutable-deployment",
        "preflight-submit",
        "phase-a-submit",
        "phase-b-submit",
        "preflight",
        "producer-geometry",
        "producer-semantic",
        "producer-place",
        "gate",
        "terminal",
        "e0-materialize",
        "e0-retrieve",
        "e0-qa",
        "t0-materialize",
        "t0-retrieve",
        "t0-qa",
        "t1-materialize",
        "t1-retrieve",
        "t1-qa",
        "evaluator",
        "finalizer",
        "monitoring",
        "continuation-verification",
        "cancellation",
        "recovery",
        "copyback",
    ]
    required_keys = {
        "host",
        "argv",
        "prerequisites",
        "expected_artifacts",
        "retry",
        "cancellation_intent_before_scancel",
        "monitor",
        "early_copyback",
        "full_copyback",
    }
    for operation in operations:
        assert isinstance(operation, dict)
        assert required_keys <= operation.keys()


def _assert_handoff_operations(operations: list[JsonValue]) -> None:
    handoff = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
    operation_by_id = {
        cast("str", operation["step_id"]): operation
        for operation in operations
        if isinstance(operation, dict)
    }
    assert {
        "monitoring",
        "continuation-verification",
        "cancellation",
        "recovery",
        "copyback",
    } <= set(operation_by_id)
    for operation in operation_by_id.values():
        assert isinstance(operation["argv"], list)
    assert (
        "$PROVIDER_GEOMETRY_JOB_ID,$PROVIDER_SEMANTIC_JOB_ID,"
        "$PROVIDER_PLACE_JOB_ID" in handoff
    )
    assert "$MATERIALIZE_E0_JOB_ID,$RETRIEVE_E0_JOB_ID,$QA_E0_JOB_ID" in handoff


def _assert_dry_run_artifacts(
    out_dir: Path, result: subprocess.CompletedProcess[str]
) -> None:
    blockers = _read_json_object(out_dir / "approval_blockers.json")
    assert blockers["runnable"] is False
    assert blockers["blockers"]
    copyback_policy = (out_dir / "copyback_policy.txt").read_text()
    assert "teacher caches" in copyback_policy
    assert "raw evidence packs" in copyback_policy
    assert "legacy single-job compatibility" not in result.stdout


def test_launch_remote_submit_requires_explicit_env_approval(tmp_path: Path) -> None:
    out_dir = tmp_path / "remote_plan"

    result = run_cli(
        "launch-remote",
        "--profile",
        "teacher-oracle",
        "--experiment-config",
        "configs/spatial/exp_0005_teacher_oracle.example.json",
        "--submit",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    assert result.returncode != 0
    assert "ExplicitApprovalRequired" in result.stderr
    assert not out_dir.exists()
    assert "ssh " not in f"{result.stdout}\n{result.stderr}"


def test_launch_remote_approved_plan_rejects_unresolved_placeholders(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    for name in REMOTE_ENV_NAMES | {
        "WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST",
        "WORLDMM_SMVQA_REMOTE_APPROVED",
    }:
        _ = env.pop(name, None)
    env["WORLDMM_SMVQA_REMOTE_APPROVED"] = "1"

    result = subprocess.run(
        [
            "uv",
            "run",
            "worldmm-smvqa",
            "launch-remote",
            "--profile",
            "teacher-oracle",
            "--experiment-config",
            "configs/spatial/exp_0005_teacher_oracle.example.json",
            "--submit",
            "--config",
            "configs/remote.example.yaml",
            "--out",
            str(tmp_path / "remote_plan"),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "TeacherOraclePlanRequired" in result.stderr
    assert "REPLACE_" in result.stderr
    assert not (tmp_path / "remote_plan").exists()


def test_launch_remote_removes_stale_legacy_runner(tmp_path: Path) -> None:
    out_dir = tmp_path / "remote_plan"
    out_dir.mkdir()
    stale = out_dir / "run_worldmm_smvqa.sh"
    _ = stale.write_text("unsafe legacy runner\n", encoding="utf-8")

    result = run_cli(
        "launch-remote",
        "--profile",
        "teacher-oracle",
        "--experiment-config",
        "configs/spatial/exp_0005_teacher_oracle.example.json",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    assert result.returncode == 0, result.stderr
    assert not stale.exists()


def test_launch_remote_config_requires_remote_placeholders(tmp_path: Path) -> None:
    config = tmp_path / "missing-output-root.yaml"
    _ = config.write_text(
        """runtime:
  location: remote
remote:
  bastion_host: ${BASTION_HOST}
  head_node: ${HEAD_NODE}
  data_root: ${SMVQA_DATA_ROOT}
  model_path: ${GEMMA_MODEL_PATH}
  execution_profile: teacher-oracle
  experiment_config: configs/spatial/exp_0005_teacher_oracle.example.json
""",
        encoding="utf-8",
    )

    result = run_cli(
        "launch-remote",
        "--profile",
        "teacher-oracle",
        "--experiment-config",
        "configs/spatial/exp_0005_teacher_oracle.example.json",
        "--dry-run",
        "--config",
        str(config),
        "--out",
        str(tmp_path / "remote_plan"),
    )

    assert result.returncode != 0
    assert "MissingRemoteConfig: WORLDMM_OUTPUT_ROOT" in result.stderr


def test_teacher_oracle_example_and_wrapper_are_dry_run_only() -> None:
    experiment = _read_json_object(
        ROOT / "configs/spatial/exp_0005_teacher_oracle.example.json"
    )
    assert experiment["execution_profile"] == "teacher-oracle"
    sensor_audit = _json_mapping(experiment["sensor_audit"])
    assert sensor_audit["window_microseconds"] == 30_000_000
    assert experiment["variants"] == ["E0", "T0", "T1"]
    capabilities = _json_mapping(experiment["capabilities"])
    assert {"provider", "semantic", "ontology", "signing", "accounting"} <= set(
        capabilities
    )
    variants = experiment["variants"]
    assert isinstance(variants, list)
    assert "E1" not in variants
    accounting = _json_mapping(experiment["accounting"])
    assert accounting["fields"] == [
        "JobIDRaw",
        "Cluster",
        "State%64",
        "ExitCode",
        "Restarts",
        "SLUID",
        "OriginalSLUID",
    ]
    assert accounting["version"] == "23.11"
    assert accounting["job_id_injection"] == {
        "flag": "--jobs",
        "placeholder": "REPLACE_JOB_IDS",
    }
    assert experiment["manifest_job_keys"] == [
        "PREFLIGHT_JOB_ID",
        "PROVIDER_GEOMETRY_JOB_ID",
        "PROVIDER_SEMANTIC_JOB_ID",
        "PROVIDER_PLACE_JOB_ID",
        "PROVIDER_GATE_JOB_ID",
        "PROVIDER_GATE_TERMINAL_JOB_ID",
        "MATERIALIZE_E0_JOB_ID",
        "RETRIEVE_E0_JOB_ID",
        "QA_E0_JOB_ID",
        "MATERIALIZE_T0_JOB_ID",
        "RETRIEVE_T0_JOB_ID",
        "QA_T0_JOB_ID",
        "MATERIALIZE_T1_JOB_ID",
        "RETRIEVE_T1_JOB_ID",
        "QA_T1_JOB_ID",
        "EVALUATE_JOB_ID",
        "FINALIZE_JOB_ID",
    ]
    assert "resources" not in experiment
    assert "stages" not in experiment
    stage_specs = experiment["stage_specs"]
    assert isinstance(stage_specs, list)
    assert len(stage_specs) == 17
    assert {
        "preflight",
        "geometry",
        "semantic",
        "place",
        "gate",
        "terminal",
        "evaluator",
        "finalizer",
    } <= {
        cast("str", stage["name"])
        for stage in stage_specs
        if isinstance(stage, dict) and isinstance(stage.get("name"), str)
    }
    wrapper = (ROOT / "scripts/remote/run_worldmm_smvqa.sh").read_text()
    assert "WORLDMM_EXECUTION_PROFILE=teacher-oracle is required" in wrapper
    assert "launch-remote --dry-run" in wrapper
    assert "--submit" not in wrapper


def test_student_graph_closes_fork_join_resources_and_outputs() -> None:

    graph = canonical_student_run_graph(
        execution_profile="full",
        model_contract_sha256="1" * 64,
        provider_lock_sha256="2" * 64,
        student_architecture_sha256="3" * 64,
        train_time_limit_minutes=1440,
        global_deadline_minutes=1800,
    )
    stages = {stage.stage_id: stage for stage in graph.stages}
    assert len(stages) == 17
    assert (
        stages["model_load_workers"].nodes,
        stages["model_load_workers"].gpus_per_node,
    ) == (
        10,
        8,
    )
    assert stages["student_watchdog"].output_keys["student_terminal"] == (
        "summary/student_terminal.json"
    )
    assert not {
        "control_primary",
        "control_backup",
        "control_actuator",
        "student_watchdog",
    } & {edge.to_stage for edge in graph.edges}
    retrieval_parents = {
        edge.from_stage for edge in graph.edges if edge.to_stage == "retrieval_join"
    }
    assert retrieval_parents == {"spatial_infer", "qwen_semantic_visual"}


def test_student_probe_renderer_is_held_and_has_native_join() -> None:

    graph = canonical_student_run_graph(
        execution_profile="probe",
        model_contract_sha256="1" * 64,
        provider_lock_sha256="2" * 64,
        student_architecture_sha256="3" * 64,
        train_time_limit_minutes=30,
        global_deadline_minutes=60,
    )
    assert all(
        (stage.nodes, stage.gpus_per_node) == (1, 1)
        for stage in graph.stages
        if stage.host_class == "gpu"
    )
    submit = student_submit_script_text(graph)
    runner = student_stage_script_text(graph)
    assert "--hold" in submit
    assert "--export=NONE" in submit
    assert (
        'EDGE["retrieval_join"]="afterok:spatial_infer;qwen_semantic_visual"' in submit
    )
    assert "only control_actuator may call scontrol/scancel" in submit
    assert "worldmm_smvqa.model_load actuator" in runner
    proof = subprocess.run(
        ["bash", "-n"],
        input=submit,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proof.returncode == 0, proof.stderr
