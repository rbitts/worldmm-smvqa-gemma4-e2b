from __future__ import annotations

import json
import shlex
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Literal, TypedDict, override

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from worldmm_smvqa.config import AppConfig, MissingRemoteConfigError
from worldmm_smvqa.remote_script import (
    teacher_oracle_downstream_submit_script_text,
    teacher_oracle_preflight_submit_script_text,
    teacher_oracle_provider_gate_submit_script_text,
    teacher_oracle_stage_script_text,
)

type JsonValue = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

_DEPENDENCY_REQUIREMENTS_MESSAGE: Final = "dependencies require both kind and stages"
_CANONICAL_GRAPH_MESSAGE: Final = "stage_specs is not the canonical EXP-0005 graph"
_UNKNOWN_DEPENDENCY_MESSAGE: Final = "stage_specs contains an unknown dependency"
_VARIANTS_ORDER_MESSAGE: Final = "variants must be E0, T0, T1 in canonical order"
_MANIFEST_KEYS_MESSAGE: Final = (
    "manifest_job_keys is not the canonical EXP-0005 manifest"
)
_STAGE_VARIANT_REQUIRED_MESSAGE: Final = (
    "materialize, retrieve, and qa stages require a variant"
)


class CapabilitySpecV1(BaseModel):
    """Immutable capability binding; ``kind`` is its on-wire discriminator."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["capability"] = "capability"
    artifact: str
    sha256: str
    policy: str | None = None


class ResourceSpecV1(BaseModel):
    """Immutable Slurm allocation; ``kind`` is its on-wire discriminator."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["slurm"] = "slurm"
    partition: str
    nodes: int
    gpus_per_node: int
    cpus: int
    memory: str
    time: str


class DependencySpecV1(BaseModel):
    """Immutable dependency edge; ``kind`` discriminates the edge semantics."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["afterok", "afterany"] | None = None
    stages: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _empty_edges_are_untyped_only(self) -> DependencySpecV1:
        if (self.kind is None) != (not self.stages):
            raise ValueError(_DEPENDENCY_REQUIREMENTS_MESSAGE)
        return self


StageRole = Literal[
    "preflight",
    "geometry",
    "semantic",
    "place",
    "gate",
    "terminal",
    "materialize",
    "retrieve",
    "qa",
    "evaluator",
    "finalizer",
]


class StageSpecV1(BaseModel):
    """Immutable graph node; ``role`` is the discriminated stage kind."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    name: str
    role: StageRole
    variant: Literal["E0", "T0", "T1"] | None
    dependencies: DependencySpecV1
    retries: int
    resources: ResourceSpecV1


class ExperimentGraphV1(BaseModel):
    """The one parsed, immutable source for EXP-0005 operator rendering."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", frozen=True)

    experiment_id: Literal["EXP-0005"]
    execution_profile: Literal["teacher-oracle"]
    lane: str
    result_class: str
    variants: tuple[Literal["E0", "T0", "T1"], ...]
    capabilities: dict[str, CapabilitySpecV1]
    stage_specs: tuple[StageSpecV1, ...]
    manifest_job_keys: tuple[str, ...]
    sensor_audit: dict[str, object]
    provider_policy: dict[str, object]
    qa: dict[str, object]
    accounting: dict[str, object]
    signer_registry: dict[str, object]

    _CANONICAL_STAGE_TOPOLOGY: ClassVar[
        tuple[tuple[str, str | None, tuple[str, ...], int], ...]
    ] = (
        ("preflight", None, (), 0),
        ("geometry", "afterok", ("preflight",), 0),
        ("semantic", "afterok", ("preflight",), 0),
        ("place", "afterok", ("preflight",), 0),
        ("gate", "afterany", ("geometry", "semantic", "place"), 0),
        ("terminal", "afterany", ("gate",), 0),
        ("e0_materialize", "afterok", ("gate",), 0),
        ("e0_retrieve", "afterok", ("e0_materialize",), 0),
        ("e0_qa", "afterok", ("e0_retrieve",), 0),
        ("t0_materialize", "afterok", ("gate",), 0),
        ("t0_retrieve", "afterok", ("t0_materialize",), 0),
        ("t0_qa", "afterok", ("t0_retrieve",), 0),
        ("t1_materialize", "afterok", ("gate",), 0),
        ("t1_retrieve", "afterok", ("t1_materialize",), 0),
        ("t1_qa", "afterok", ("t1_retrieve",), 0),
        ("evaluator", "afterok", ("e0_qa", "t0_qa", "t1_qa"), 0),
        ("finalizer", "afterany", ("evaluator", "terminal"), 0),
    )

    @model_validator(mode="after")
    def _canonical_graph(self) -> ExperimentGraphV1:
        expected = (
            ("preflight", "preflight", None),
            ("geometry", "geometry", None),
            ("semantic", "semantic", None),
            ("place", "place", None),
            ("gate", "gate", None),
            ("terminal", "terminal", None),
            ("e0_materialize", "materialize", "E0"),
            ("e0_retrieve", "retrieve", "E0"),
            ("e0_qa", "qa", "E0"),
            ("t0_materialize", "materialize", "T0"),
            ("t0_retrieve", "retrieve", "T0"),
            ("t0_qa", "qa", "T0"),
            ("t1_materialize", "materialize", "T1"),
            ("t1_retrieve", "retrieve", "T1"),
            ("t1_qa", "qa", "T1"),
            ("evaluator", "evaluator", None),
            ("finalizer", "finalizer", None),
        )
        actual = tuple(
            (stage.name, stage.role, stage.variant) for stage in self.stage_specs
        )
        if actual != expected:
            raise ValueError(_CANONICAL_GRAPH_MESSAGE)
        names = {stage.name for stage in self.stage_specs}
        if any(
            dependency not in names
            for stage in self.stage_specs
            for dependency in stage.dependencies.stages
        ):
            raise ValueError(_UNKNOWN_DEPENDENCY_MESSAGE)
        topology = tuple(
            (
                stage.name,
                stage.dependencies.kind,
                stage.dependencies.stages,
                stage.retries,
            )
            for stage in self.stage_specs
        )
        if topology != self._CANONICAL_STAGE_TOPOLOGY:
            raise ValueError(_CANONICAL_GRAPH_MESSAGE)
        if self.variants != ("E0", "T0", "T1"):
            raise ValueError(_VARIANTS_ORDER_MESSAGE)
        expected_keys = (
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
        )
        if self.manifest_job_keys != expected_keys:
            raise ValueError(_MANIFEST_KEYS_MESSAGE)
        return self


_JSON_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_JSON_OBJECT_ADAPTER: Final[TypeAdapter[object]] = TypeAdapter(object)
_JSON_MAPPING_ADAPTER: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(
    dict[str, object]
)
_JSON_SEQUENCE_ADAPTER: Final[TypeAdapter[list[object]]] = TypeAdapter(list[object])

APPROVAL_ENV: Final = "WORLDMM_SMVQA_REMOTE_APPROVED"
TEACHER_ORACLE_PREFLIGHT_SUBMIT_SCRIPT_NAME: Final = (
    "submit_teacher_oracle_preflight.sh"
)
TEACHER_ORACLE_PROVIDER_GATE_SUBMIT_SCRIPT_NAME: Final = (
    "submit_teacher_oracle_provider_gate.sh"
)
TEACHER_ORACLE_DOWNSTREAM_SUBMIT_SCRIPT_NAME: Final = (
    "submit_teacher_oracle_downstream.sh"
)
TEACHER_ORACLE_STAGE_SCRIPT_NAME: Final = "run_teacher_oracle_stage.sh"
EXPECTED_OUTPUTS_NAME: Final = "expected_outputs.json"
COPYBACK_POLICY_NAME: Final = "copyback_policy.txt"
OPERATOR_CONTRACT_NAME: Final = "operator_contract.json"
APPROVAL_BLOCKERS_NAME: Final = "approval_blockers.json"
EXPERIMENT_GRAPH_NAME: Final = "experiment_graph.json"
DEFAULT_REMOTE_REPO: Final = "/repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b"
_OUTPUT_ROOT: Final = "$WORLDMM_OUTPUT_ROOT"
_SUMMARY_DIR: Final = f"{_OUTPUT_ROOT}/summary"
_ORACLE_DIR: Final = f"{_OUTPUT_ROOT}/oracle"
_PREFLIGHT_JOB_MANIFEST: Final = "summary/dag_jobs.preflight.env"
_PROVIDER_JOB_MANIFEST: Final = "summary/dag_jobs.provider.env"
_TERMINAL_ARTIFACT: Final = "summary/teacher_oracle_terminal.json"
_CONTINUE_RECEIPT_ARTIFACT: Final = "summary/teacher_oracle_continue.json"
_FINAL_REPORT_ARTIFACT: Final = "summary/final_report.md"
_REMOTE_PLAN_DIR: Final = "$WORLDMM_REMOTE_REPO/remote-plan"
_MONITOR_COMMAND: Final = (
    "/opt/slurm/bin/sacct",
    "-D",
    "-X",
    "-n",
    "-P",
    '--clusters="$WORLDMM_SLURM_CLUSTER"',
    '--jobs="$JOB_IDS"',
    "--format=JobIDRaw,Cluster,State%64,ExitCode,Restarts,SLUID,OriginalSLUID",
)
_EARLY_COPYBACK_COMMAND: Final = (
    "rsync",
    "-av",
    "--files-from=<(printf '%s\\n' summary/teacher_oracle_terminal.json "
    "summary/dag_jobs.provider.env)",
    '"$HEAD_NODE:$WORLDMM_OUTPUT_ROOT/"',
    '"./exp-0005-$WORLDMM_RUN_ID-early/"',
)
_FULL_COPYBACK_COMMAND: Final = (
    "rsync",
    "-av",
    "--files-from=<(printf '%s\\n' summary/teacher_oracle_terminal.json "
    "summary/final_report.md summary/remote_manifest.json summary/dag_jobs.env "
    "oracle/E0/metrics.json oracle/T0/metrics.json oracle/T1/metrics.json)",
    '"$HEAD_NODE:$WORLDMM_OUTPUT_ROOT/"',
    '"./exp-0005-$WORLDMM_RUN_ID-full/"',
)

REMOTE_FIELDS: Final = {
    "bastion_host": "BASTION_HOST",
    "head_node": "HEAD_NODE",
    "data_root": "SMVQA_DATA_ROOT",
    "model_path": "GEMMA_MODEL_PATH",
    "output_root": "WORLDMM_OUTPUT_ROOT",
}


class ExpectedOutputs(TypedDict):
    remote_job_reference: str
    metrics: list[str]
    outputs: dict[str, str]
    conditional_outputs: dict[str, str]
    copyback_allowed: list[str]
    copyback_forbidden: list[str]
    operator_contract: dict[str, object]


SubmitMode = Literal["dry-run", "approved-plan-only"]


@dataclass(frozen=True, slots=True)
class ExplicitApprovalRequiredError(Exception):
    flag: str

    @override
    def __str__(self) -> str:
        return f"ExplicitApprovalRequired: set {self.flag}=1 to use --submit"


type PlanProfile = Literal["teacher-oracle"]


@dataclass(frozen=True, slots=True)
class TeacherOraclePlanRequiredError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"TeacherOraclePlanRequired: {self.detail}"


@dataclass(frozen=True, slots=True)
class RemotePlanResult:
    script: Path
    expected_outputs: Path
    copyback_policy: Path
    operator_contract: Path
    approval_blockers: Path
    mode: SubmitMode
    execution_profile: PlanProfile


def write_remote_plan(
    config: AppConfig,
    out_dir: Path,
    env: Mapping[str, str],
    *,
    submit: bool,
    plan_profile: PlanProfile = "teacher-oracle",
) -> RemotePlanResult:
    if submit and env.get(APPROVAL_ENV) != "1":
        raise ExplicitApprovalRequiredError(flag=APPROVAL_ENV)

    _require_remote_placeholders(config)
    configured_profile = config.values["remote"]["execution_profile"]
    if configured_profile != "teacher-oracle":
        raise TeacherOraclePlanRequiredError(
            detail="remote.execution_profile must be teacher-oracle"
        )
    _ = plan_profile
    effective_profile: PlanProfile = "teacher-oracle"
    requested_profile = env.get("WORLDMM_EXECUTION_PROFILE")
    if requested_profile not in (None, configured_profile):
        raise TeacherOraclePlanRequiredError(
            detail=(
                "WORLDMM_EXECUTION_PROFILE conflicts with the validated "
                f"remote.execution_profile: {requested_profile}"
            )
        )
    experiment = _reviewed_experiment(config)
    blockers = _approval_blockers(config, experiment)
    if submit and blockers:
        raise TeacherOraclePlanRequiredError(detail="; ".join(blockers))
    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=parent))
    try:
        script = staged_dir / TEACHER_ORACLE_PREFLIGHT_SUBMIT_SCRIPT_NAME
        provider_submit_script = (
            staged_dir / TEACHER_ORACLE_PROVIDER_GATE_SUBMIT_SCRIPT_NAME
        )
        downstream_submit_script = (
            staged_dir / TEACHER_ORACLE_DOWNSTREAM_SUBMIT_SCRIPT_NAME
        )
        dag_stage_script = staged_dir / TEACHER_ORACLE_STAGE_SCRIPT_NAME
        expected_outputs = staged_dir / EXPECTED_OUTPUTS_NAME
        copyback_policy = staged_dir / COPYBACK_POLICY_NAME
        operator_contract = staged_dir / OPERATOR_CONTRACT_NAME
        approval_blockers = staged_dir / APPROVAL_BLOCKERS_NAME

        _write_experiment_graph(staged_dir, experiment)
        _ = script.write_text(
            teacher_oracle_preflight_submit_script_text(experiment), encoding="utf-8"
        )
        script.chmod(0o755)
        _ = provider_submit_script.write_text(
            teacher_oracle_provider_gate_submit_script_text(experiment),
            encoding="utf-8",
        )
        provider_submit_script.chmod(0o755)
        _ = downstream_submit_script.write_text(
            teacher_oracle_downstream_submit_script_text(experiment), encoding="utf-8"
        )
        downstream_submit_script.chmod(0o755)
        _ = dag_stage_script.write_text(
            teacher_oracle_stage_script_text(experiment), encoding="utf-8"
        )
        dag_stage_script.chmod(0o755)
        _ = expected_outputs.write_text(
            json.dumps(_expected_outputs(experiment), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _ = copyback_policy.write_text(_copyback_policy_text(), encoding="utf-8")
        _ = operator_contract.write_text(
            json.dumps(_operator_contract(experiment), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _ = approval_blockers.write_text(
            json.dumps(
                {"blockers": blockers, "runnable": not blockers},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _publish_plan_directory(staged_dir, out_dir)
    except BaseException:
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise

    script = out_dir / TEACHER_ORACLE_PREFLIGHT_SUBMIT_SCRIPT_NAME
    expected_outputs = out_dir / EXPECTED_OUTPUTS_NAME
    copyback_policy = out_dir / COPYBACK_POLICY_NAME
    operator_contract = out_dir / OPERATOR_CONTRACT_NAME
    approval_blockers = out_dir / APPROVAL_BLOCKERS_NAME

    mode: SubmitMode = "approved-plan-only" if submit else "dry-run"
    return RemotePlanResult(
        script=script,
        expected_outputs=expected_outputs,
        copyback_policy=copyback_policy,
        operator_contract=operator_contract,
        approval_blockers=approval_blockers,
        mode=mode,
        execution_profile=effective_profile,
    )


def _publish_plan_directory(staged_dir: Path, out_dir: Path) -> None:
    """Atomically replace a generated plan without exposing partial contents."""
    if not staged_dir.is_dir():
        message = f"staged remote plan is not a directory: {staged_dir}"
        raise OSError(message)
    if out_dir.is_symlink() or (out_dir.exists() and not out_dir.is_dir()):
        message = f"remote plan target is not a safe directory: {out_dir}"
        raise OSError(message)
    if not out_dir.exists():
        _ = staged_dir.replace(out_dir)
        return

    backup_dir = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.previous.", dir=out_dir.parent)
    )
    backup_dir.rmdir()
    _ = out_dir.replace(backup_dir)
    try:
        _ = staged_dir.replace(out_dir)
    except BaseException:
        _ = backup_dir.replace(out_dir)
        raise
    shutil.rmtree(backup_dir)


def plan_stdout(result: RemotePlanResult) -> str:
    if result.mode == "dry-run":
        return (
            "remote plan mode=dry-run\n"
            f"wrote {result.operator_contract}\n"
            f"wrote {result.approval_blockers}\n"
            "runnable=false\n"
            "# dry-run/plan only; no ssh, remote shell, or job submission "
            "opened locally\n"
        )
    plan_dir = shlex.quote(str(result.script.parent))
    snapshot_ref = (
        f"${{WORLDMM_REMOTE_SNAPSHOT_ROOT:-{DEFAULT_REMOTE_REPO}.snapshots}}/"
        "${WORLDMM_CODE_SHA}"
    )
    tracked_deploy = shlex.quote(
        'approved_sha="$1" && '
        'snapshot_root="${WORLDMM_REMOTE_SNAPSHOT_ROOT:-'
        '/repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b.snapshots}" && '
        'snapshot="$snapshot_root/$approved_sha" && '
        'mkdir -p "$(dirname "$snapshot")" && '
        'test ! -e "$snapshot" && '
        'mkdir "$snapshot" && '
        'tar -xf - -C "$snapshot"'
    )
    sync_repo = (
        ': "${WORLDMM_CODE_SHA:?set the approved git SHA}"; '
        'local_repo="$(git rev-parse --show-toplevel)"; '
        'test -z "$(git -C "$local_repo" status --porcelain)"; '
        'test "$(git -C "$local_repo" rev-parse HEAD)" = "$WORLDMM_CODE_SHA"; '
        'git -C "$local_repo" archive --format=tar "$WORLDMM_CODE_SHA" '
        f'| ssh -J "$BASTION_HOST" "$HEAD_NODE" {tracked_deploy} '
        'sh "$WORLDMM_CODE_SHA"'
    )
    remote_plan_dir = f"{snapshot_ref}/remote-plan"
    sync_plan = (
        "rsync -az --delete -e \"ssh -J $BASTION_HOST\" --exclude '.env*' "
        f"{plan_dir}/ "
        f'"$HEAD_NODE:{remote_plan_dir}/"'
    )
    approval_assignment = (
        f"{APPROVAL_ENV}=1 " if result.mode == "approved-plan-only" else ""
    )
    remote_shell_command = (
        "set -euo pipefail; umask 077; "
        'approved_sha="$1"; environment_contract="$2"; '
        'environment_sha256="$3"; '
        'environment_copy="$(mktemp)"; trap \'rm -f "$environment_copy"\' EXIT; '
        'python3 - "$environment_contract" "$environment_sha256" '
        "\"$environment_copy\" <<'PY'\n"
        "import hashlib, os, stat, sys\n"
        "source, expected, target = sys.argv[1:]\n"
        "fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)\n"
        "try:\n"
        "    metadata = os.fstat(fd)\n"
        "    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()\n"
        "            or metadata.st_mode & 0o077):\n"
        '        raise SystemExit("unsafe environment contract")\n'
        '    data = b""\n'
        "    while len(data) < metadata.st_size:\n"
        "        chunk = os.read(fd, metadata.st_size - len(data))\n"
        "        if not chunk: break\n"
        "        data += chunk\n"
        "    if len(data) != metadata.st_size or "
        "hashlib.sha256(data).hexdigest() != expected:\n"
        '        raise SystemExit("environment contract digest mismatch")\n'
        "finally:\n"
        "    os.close(fd)\n"
        "out = os.open(target, os.O_WRONLY | os.O_TRUNC | os.O_CLOEXEC)\n"
        "try: os.write(out, data)\n"
        "finally: os.close(out)\n"
        "os.chmod(target, 0o600)\n"
        "PY\n"
        'set -a; . "$environment_copy"; set +a; rm -f "$environment_copy"; '
        'snapshot_root="${WORLDMM_REMOTE_SNAPSHOT_ROOT:-'
        '/repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b.snapshots}"; '
        'snapshot="$snapshot_root/$approved_sha"; '
        'test -d "$snapshot"; '
        'cd "$snapshot"; mkdir -p remote-plan/logs; '
        f'{approval_assignment}WORLDMM_REMOTE_REPO="$snapshot" '
        'WORLDMM_CODE_SHA="$approved_sha" '
        "WORLDMM_EXECUTION_PROFILE=teacher-oracle "
        f"bash remote-plan/{TEACHER_ORACLE_PREFLIGHT_SUBMIT_SCRIPT_NAME}"
    )
    remote_command = (
        f'ssh -J "$BASTION_HOST" "$HEAD_NODE" {shlex.quote(remote_shell_command)} '
        'sh "$WORLDMM_CODE_SHA" "$WORLDMM_REMOTE_ENV_FILE" '
        '"$WORLDMM_REMOTE_ENV_SHA256"'
    )
    return (
        f"remote plan mode={result.mode}\n"
        f"wrote {result.script}\n"
        f"wrote {result.expected_outputs}\n"
        f"wrote {result.copyback_policy}\n"
        f"wrote {result.operator_contract}\n"
        f"wrote {result.approval_blockers}\n"
        "wrote "
        f"{result.script.parent / TEACHER_ORACLE_PROVIDER_GATE_SUBMIT_SCRIPT_NAME}\n"
        f"wrote {result.script.parent / TEACHER_ORACLE_DOWNSTREAM_SUBMIT_SCRIPT_NAME}\n"
        f"wrote {result.script.parent / TEACHER_ORACLE_STAGE_SCRIPT_NAME}\n"
        "# dry-run/plan only; no ssh, remote shell, or job submission opened "
        "locally\n"
        f"{sync_repo}\n"
        f"{sync_plan}\n"
        f"{remote_command}\n"
    )


def _require_remote_placeholders(config: AppConfig) -> None:
    remote = config.values.get("remote", {})
    for key, env_name in REMOTE_FIELDS.items():
        if remote.get(key) != f"${{{env_name}}}":
            raise MissingRemoteConfigError(name=env_name)
    if remote.get("execution_profile") != "teacher-oracle":
        raise TeacherOraclePlanRequiredError(
            detail="remote.execution_profile must be teacher-oracle"
        )
    if remote.get("experiment_config") != (
        "configs/spatial/exp_0005_teacher_oracle.example.json"
    ):
        raise TeacherOraclePlanRequiredError(
            detail="remote.experiment_config must name the reviewed EXP-0005 config"
        )


def _reviewed_experiment(config: AppConfig) -> ExperimentGraphV1:
    experiment_config = Path(config.values["remote"]["experiment_config"])
    try:
        return ExperimentGraphV1.model_validate_json(
            experiment_config.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        raise TeacherOraclePlanRequiredError(
            detail=f"reviewed experiment config is unreadable or invalid: {exc}"
        ) from exc


def _approval_blockers(config: AppConfig, experiment: ExperimentGraphV1) -> list[str]:
    blockers = _replace_placeholders(config.values, "remote config")
    blockers.extend(
        _replace_placeholders(
            experiment.model_dump(mode="json"), "reviewed experiment config"
        )
    )
    blockers.extend(_strict_oracle_config_blockers(experiment))
    return blockers


def _strict_oracle_config_blockers(experiment: ExperimentGraphV1) -> list[str]:
    """Require every binding consumed by the isolated remote oracle renderer."""
    required_paths = (
        ("sensor_audit", "version"),
        ("sensor_audit", "window_microseconds"),
        ("sensor_audit", "byte_cap_per_window"),
        ("sensor_audit", "slice_id"),
        ("provider_policy", "provider_digest"),
        ("provider_policy", "allowed_provider_ids"),
        ("qa", "model_digest"),
        ("qa", "prompt_digest"),
        ("qa", "seed"),
        ("qa", "shard_salt"),
        ("qa", "world_size"),
        ("accounting", "cluster"),
        ("accounting", "fields"),
        ("signer_registry", "registry_digest"),
    )
    values = _JSON_VALUE_ADAPTER.validate_python(experiment.model_dump(mode="json"))
    if not isinstance(values, dict):
        message = "reviewed experiment config is not a JSON object"
        raise TypeError(message)
    blockers: list[str] = []
    for path in required_paths:
        current: JsonValue = values
        for key in path:
            if not isinstance(current, dict) or key not in current:
                blockers.append(
                    "reviewed experiment config is missing strict binding: "
                    + ".".join(path)
                )
                break
            current = current[key]
    return blockers


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        mapping = _JSON_MAPPING_ADAPTER.validate_python(value)
        return {key: _json_value(child) for key, child in mapping.items()}
    if isinstance(value, list):
        sequence = _JSON_SEQUENCE_ADAPTER.validate_python(value)
        return [_json_value(child) for child in sequence]
    detail = f"unsupported JSON value type: {type(value).__name__}"
    raise ValueError(detail)


def _replace_placeholders(value: object, location: str) -> list[str]:
    if isinstance(value, str):
        return (
            [f"{location} contains unresolved placeholder: {value}"]
            if ("REPLACE_" in value)
            else []
        )
    if isinstance(value, dict):
        mapping = _JSON_MAPPING_ADAPTER.validate_python(value)
        return [
            blocker
            for key, child in mapping.items()
            for blocker in _replace_placeholders(child, f"{location}.{key}")
        ]
    if isinstance(value, list):
        sequence = _JSON_SEQUENCE_ADAPTER.validate_python(value)
        return [
            blocker
            for index, child in enumerate(sequence)
            for blocker in _replace_placeholders(child, f"{location}[{index}]")
        ]
    return []


@dataclass(frozen=True, slots=True)
class _Operation:
    step_id: str
    host: str
    argv: list[str]
    prerequisites: list[str]
    artifacts: list[str]
    retry: str = "none"
    cancellation: str = "not applicable"

    def render(
        self,
        *,
        monitor: list[str],
        early_copyback: list[str],
        full_copyback: list[str],
    ) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "host": self.host,
            "argv": self.argv,
            "prerequisites": self.prerequisites,
            "expected_artifacts": self.artifacts,
            "retry": self.retry,
            "cancellation_intent_before_scancel": self.cancellation,
            "monitor": monitor,
            "early_copyback": early_copyback,
            "full_copyback": full_copyback,
        }


def _expected_outputs(experiment: ExperimentGraphV1) -> ExpectedOutputs:
    contract = _operator_contract(experiment)
    return {
        "remote_job_reference": (f"{_SUMMARY_DIR}/dag_jobs.env#FINALIZE_JOB_ID"),
        "metrics": ["Ans-F1", "QA-Acc", "QA-MRR"],
        "outputs": {
            "code_snapshot": f"{_OUTPUT_ROOT}/code_snapshot",
            "sensor_audit": f"{_OUTPUT_ROOT}/diagnostics/sensor_audit.json",
            "teacher_job_manifest": f"{_OUTPUT_ROOT}/{_PROVIDER_JOB_MANIFEST}",
            "preflight_job_manifest": f"{_OUTPUT_ROOT}/{_PREFLIGHT_JOB_MANIFEST}",
            "stage_logs": f"{_OUTPUT_ROOT}/logs/*-*.out",
            "stage_errors": f"{_OUTPUT_ROOT}/logs/*-*.err",
        },
        "conditional_outputs": {
            "sealed_continue_receipt_if_provider_go": (
                f"{_OUTPUT_ROOT}/{_CONTINUE_RECEIPT_ARTIFACT}"
            ),
            "provider_gate_terminal_on_every_branch": (
                f"{_OUTPUT_ROOT}/{_TERMINAL_ARTIFACT}"
            ),
            "phase_b_job_manifest_after_second_approval": (
                f"{_SUMMARY_DIR}/dag_jobs.env"
            ),
            "phase_b_outputs_after_second_approval": (
                f"{_ORACLE_DIR}/{{E0,T0,T1}}/"
                "{typed_memory,evidence_packs,predictions}.jsonl,"
                f"{_ORACLE_DIR}/{{E0,T0,T1}}/metrics.json"
            ),
            "phase_b_report_after_second_approval": (
                f"{_OUTPUT_ROOT}/{_FINAL_REPORT_ARTIFACT}"
            ),
        },
        "copyback_allowed": [
            "metrics",
            "reviewed non-sensitive sensor and provider diagnostics",
            "redacted lightweight logs",
            "plots",
            "summaries",
            "explicitly approved small sample outputs",
        ],
        "copyback_forbidden": [
            "full datasets",
            "frame assets",
            "model weights",
            "checkpoints",
            "teacher caches",
            "raw evidence packs",
            "sensitive artifacts",
        ],
        "operator_contract": contract,
    }


def _stage_job_key(stage: StageSpecV1) -> str:
    if stage.role in ("geometry", "semantic", "place"):
        return f"PROVIDER_{stage.role.upper()}_JOB_ID"
    if stage.role in ("materialize", "retrieve", "qa"):
        if stage.variant is None:
            raise ValueError(_STAGE_VARIANT_REQUIRED_MESSAGE)
        return f"{stage.role.upper()}_{stage.variant}_JOB_ID"
    special_keys = {
        "gate": "PROVIDER_GATE_JOB_ID",
        "terminal": "PROVIDER_GATE_TERMINAL_JOB_ID",
        "evaluator": "EVALUATE_JOB_ID",
        "finalizer": "FINALIZE_JOB_ID",
    }
    return special_keys.get(stage.role, f"{stage.role.upper()}_JOB_ID")


def _write_experiment_graph(
    staged_dir: Path,
    experiment: ExperimentGraphV1,
) -> None:
    target = staged_dir / EXPERIMENT_GRAPH_NAME
    _ = target.write_text(
        experiment.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def _stage_manifest(stage: StageSpecV1) -> str:
    if stage.role == "preflight":
        return _PREFLIGHT_JOB_MANIFEST
    if stage.role in ("geometry", "semantic", "place", "gate", "terminal"):
        return _PROVIDER_JOB_MANIFEST
    return "summary/dag_jobs.env"


def _stage_rows(experiment: ExperimentGraphV1) -> list[dict[str, object]]:
    return [
        {
            "name": stage.name,
            "role": stage.role,
            "variant": stage.variant,
            "dependencies": stage.dependencies.model_dump(mode="json"),
            "retries": stage.retries,
            "resources": stage.resources.model_dump(mode="json"),
            "job_manifest_key": _stage_job_key(stage),
        }
        for stage in experiment.stage_specs
    ]


def _operator_contract(experiment: ExperimentGraphV1) -> dict[str, object]:
    """Render every operator artifact from the one parsed ExperimentGraphV1."""
    monitor: list[str] = list(_MONITOR_COMMAND)
    early_copyback: list[str] = list(_EARLY_COPYBACK_COMMAND)
    full_copyback: list[str] = list(_FULL_COPYBACK_COMMAND)
    operations: list[_Operation] = [
        _Operation(
            step_id="local-dry-run",
            host="local",
            argv=["bash", "scripts/remote/run_worldmm_smvqa.sh"],
            prerequisites=[
                "WORLDMM_PLAN_OUT",
                "WORLDMM_EXECUTION_PROFILE=teacher-oracle",
            ],
            artifacts=[OPERATOR_CONTRACT_NAME, APPROVAL_BLOCKERS_NAME],
        ),
        _Operation(
            step_id="immutable-deployment",
            host="local-to-head",
            argv=["git", "archive", "$WORLDMM_CODE_SHA"],
            prerequisites=[
                "clean approved checkout",
                "new immutable remote snapshot",
            ],
            artifacts=["$WORLDMM_REMOTE_REPO", _REMOTE_PLAN_DIR],
        ),
    ]
    operations.extend(
        [
            _Operation(
                step_id="preflight-submit",
                host="head",
                argv=[
                    "bash",
                    f"{_REMOTE_PLAN_DIR}/{TEACHER_ORACLE_PREFLIGHT_SUBMIT_SCRIPT_NAME}",
                ],
                prerequisites=[
                    "separate preflight approval",
                    "verified immutable environment",
                ],
                artifacts=[_PREFLIGHT_JOB_MANIFEST],
            ),
            _Operation(
                step_id="phase-a-submit",
                host="head",
                argv=[
                    "bash",
                    f"{_REMOTE_PLAN_DIR}/"
                    f"{TEACHER_ORACLE_PROVIDER_GATE_SUBMIT_SCRIPT_NAME}",
                ],
                prerequisites=[
                    "separate Phase-A approval",
                    "verified preflight receipt",
                ],
                artifacts=[_PROVIDER_JOB_MANIFEST, _TERMINAL_ARTIFACT],
            ),
            _Operation(
                step_id="phase-b-submit",
                host="head",
                argv=[
                    "bash",
                    f"{_REMOTE_PLAN_DIR}/"
                    f"{TEACHER_ORACLE_DOWNSTREAM_SUBMIT_SCRIPT_NAME}",
                ],
                prerequisites=[
                    "separate Phase-B approval",
                    "signed Go continuation and terminal",
                ],
                artifacts=["summary/dag_jobs.env", _FINAL_REPORT_ARTIFACT],
            ),
        ]
    )
    for stage in experiment.stage_specs:
        step_id = (
            f"producer-{stage.role}"
            if stage.role in ("geometry", "semantic", "place")
            else (
                f"{stage.variant.lower()}-{stage.role}"
                if stage.variant is not None
                else stage.role
            )
        )
        submitter = (
            TEACHER_ORACLE_PREFLIGHT_SUBMIT_SCRIPT_NAME
            if stage.role == "preflight"
            else TEACHER_ORACLE_PROVIDER_GATE_SUBMIT_SCRIPT_NAME
            if stage.role in ("geometry", "semantic", "place", "gate", "terminal")
            else TEACHER_ORACLE_DOWNSTREAM_SUBMIT_SCRIPT_NAME
        )
        operations.append(
            _Operation(
                step_id=step_id,
                host="head",
                argv=["bash", f"{_REMOTE_PLAN_DIR}/{submitter}"],
                prerequisites=[
                    f"dependencies={','.join(stage.dependencies.stages) or 'none'}",
                    f"manifest_key={_stage_job_key(stage)}",
                ],
                artifacts=[f"{_stage_manifest(stage)}#{_stage_job_key(stage)}"],
                retry=(
                    "immutable attempt-N only; changed resources/config/world-size "
                    "require a new run"
                    if stage.role
                    in ("materialize", "retrieve", "qa", "evaluator", "finalizer")
                    else "none"
                ),
                cancellation=(
                    "one immutable O_EXCL/fsynced CancellationIntentV1 per producer "
                    "exact run/stage/job/attempt before /opt/slurm/bin/scancel; "
                    "accounting accepts only the matching intent and emits "
                    "cancelled/not_decidable"
                    if stage.role in ("geometry", "semantic", "place")
                    else "gate and terminal cancellation is forbidden"
                    if stage.role in ("gate", "terminal")
                    else "not applicable"
                ),
            )
        )
    operations.extend(
        [
            _Operation(
                step_id="monitoring",
                host="head",
                argv=monitor,
                prerequisites=[
                    "source the phase manifest",
                    "JOB_IDS from manifest keys",
                ],
                artifacts=["summary/accounting.json"],
            ),
            _Operation(
                step_id="continuation-verification",
                host="head",
                argv=[
                    "sha256sum",
                    _CONTINUE_RECEIPT_ARTIFACT,
                    _TERMINAL_ARTIFACT,
                ],
                prerequisites=[
                    "ProviderGateTerminalV1 Go",
                    "separate Phase-B approval",
                ],
                artifacts=[_CONTINUE_RECEIPT_ARTIFACT, _TERMINAL_ARTIFACT],
            ),
            _Operation(
                step_id="cancellation",
                host="head",
                argv=["/opt/slurm/bin/scancel", "$JOB_ID"],
                prerequisites=[
                    "one immutable O_EXCL/fsynced CancellationIntentV1 per producer "
                    "exact run/stage/job/attempt",
                    "gate and terminal jobs are forbidden cancellation targets",
                ],
                artifacts=["summary/cancellation_intent.<stage>.<job>.<attempt>.json"],
                cancellation=(
                    "only matching intent accounts as cancelled/not_decidable; "
                    "unknown sbatch requires immutable descriptor-stable reconciliation"
                ),
            ),
            _Operation(
                step_id="recovery",
                host="local",
                argv=early_copyback,
                prerequisites=["non-Go, cancellation, or failure terminal"],
                artifacts=[
                    "teacher_oracle_terminal.json",
                    "dag_jobs.provider.env",
                ],
            ),
            _Operation(
                step_id="copyback",
                host="local",
                argv=full_copyback,
                prerequisites=["full terminal and reviewed lightweight artifacts"],
                artifacts=[
                    "teacher_oracle_terminal.json",
                    "final_report.md",
                    "dag_jobs.env",
                ],
            ),
        ]
    )
    sensor_audit = experiment.sensor_audit
    accounting = experiment.accounting
    return {
        "experiment_id": experiment.experiment_id,
        "execution_profile": experiment.execution_profile,
        "lane": experiment.lane,
        "result_class": experiment.result_class,
        "variants": list(experiment.variants),
        "stage_specs": _stage_rows(experiment),
        "manifest_job_keys": list(experiment.manifest_job_keys),
        "generated_script_inputs": {
            "graph_model": "ExperimentGraphV1",
            "stages": _stage_rows(experiment),
            "manifest_job_keys": list(experiment.manifest_job_keys),
        },
        "validation": {
            "graph_model": "ExperimentGraphV1",
            "capability_model": "CapabilitySpecV1",
            "resource_model": "ResourceSpecV1",
            "dependency_model": "DependencySpecV1",
            "stage_model": "StageSpecV1",
        },
        "sensor_audit": {
            "version": sensor_audit["version"],
            "window_microseconds": sensor_audit["window_microseconds"],
        },
        "resource_schema": "ExperimentGraphV1.stage_specs[].resources",
        "accounting": {
            "command": accounting["command"],
            "fields": accounting["fields"],
            "producer_no_requeue": True,
        },
        "go_branch_artifacts": {
            "continue_receipt": f"{_OUTPUT_ROOT}/{_CONTINUE_RECEIPT_ARTIFACT}",
            "terminal": f"{_OUTPUT_ROOT}/{_TERMINAL_ARTIFACT}",
            "rule": (
                "Go writes both artifacts: the signed continue receipt and the "
                "ProviderGateTerminalV1 Go record. Phase-B approval binds SHA-256 "
                "of their exact file bytes."
            ),
        },
        "operations": [
            operation.render(
                monitor=monitor,
                early_copyback=early_copyback,
                full_copyback=full_copyback,
            )
            for operation in operations
        ],
        "phase_b_bindings": [
            "qa_shard_map_sha256",
            "qa_lineage_sha256",
            "qa_finalization_receipt_sha256",
            "qa_predictions_sha256",
            "terminal_sha256",
            "continue_receipt_sha256",
        ],
        "forbidden_claims": ["student", "E1"],
    }


def _copyback_policy_text() -> str:
    return (
        "Only metrics, reviewed non-sensitive sensor/provider diagnostics, redacted "
        "lightweight logs, plots, summaries, and explicitly approved small samples "
        "may be copied locally.\n"
        "Forbidden copyback: no full datasets, frame assets, model weights, "
        "checkpoints, teacher caches, raw evidence packs, or sensitive artifacts.\n"
        "Store all benchmark data, model artifacts, checkpoints, and raw evidence "
        "on approved company storage under $WORLDMM_OUTPUT_ROOT.\n"
    )
