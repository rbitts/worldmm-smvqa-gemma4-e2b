from __future__ import annotations

import json
import shlex
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Literal, TypedDict, Unpack, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
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


StudentStageRole = Literal[
    "preflight",
    "load",
    "gate",
    "terminal",
    "teacher",
    "merge",
    "train",
    "spatial_infer",
    "qwen",
    "retrieval",
    "qa",
    "report",
    "controller",
    "actuator",
    "watchdog",
]
StudentHostClass = Literal["cpu", "gpu"]
StudentDependencyKind = Literal["afterok", "afterany"]


class StudentStageSpecV1(BaseModel):
    """One immutable stage in the approved student workload graph."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    stage_id: str
    role: StudentStageRole
    host_class: StudentHostClass
    nodes: int = Field(ge=1)
    gpus_per_node: int = Field(ge=0)
    cpus_per_task: int = Field(ge=1)
    memory_gb: int = Field(ge=1)
    time_limit_minutes: int = Field(ge=1)
    command_key: str
    output_keys: dict[str, str]

    @model_validator(mode="after")
    def _host_resources_agree(self) -> StudentStageSpecV1:
        if (self.host_class == "gpu") != (self.gpus_per_node > 0):
            msg = "student stage host class and GPU resources disagree"
            raise ValueError(msg)
        return self


class StudentEdgeSpecV1(BaseModel):
    """A native Slurm edge plus its required durable authorization receipt."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    from_stage: str
    to_stage: str
    dependency_kind: StudentDependencyKind
    requires_receipt_kind: str | None


_STUDENT_OUTPUT_CATALOGUE: Final[dict[str, dict[str, str]]] = {
    "preflight_ingest": {
        "preflight_contract": "manifests/preflight_contract.json",
        "preflight_inputs": "manifests/preflight_inputs.json",
        "preflight_inventory": "manifests/pre_load_artifact_inventory.json",
    },
    "model_load_workers": {
        "rank_receipts": "diagnostics/model_load/ranks/",
        "worker_log": "logs/model_load_workers.log",
    },
    "model_load_gate": {
        "model_load_consensus": "diagnostics/model_load/model_load_consensus.json",
        "model_load_continue": "diagnostics/model_load/model_load_continue.json",
        "gate_log": "logs/model_load_gate.log",
    },
    "model_load_terminal": {
        "model_load_terminal": "diagnostics/model_load/model_load_terminal.json",
        "terminal_log": "logs/model_load_terminal.log",
    },
    "teacher_extract": {
        "teacher_shards": "teacher/shards/",
        "teacher_manifest": "teacher/teacher_manifest.json",
    },
    "merge_materialize": {
        "teacher_cache": "training/student_teacher_cache.jsonl",
        "teacher_cache_contract": "training/teacher_cache.contract.json",
    },
    "train": {
        "student_checkpoint": "checkpoints/spatial_student.pt",
        "training_metrics": "training/training_metrics.json",
    },
    "spatial_infer": {
        "spatial_inference_load_receipt": (
            "diagnostics/model_load/spatial_inference_load.json"
        ),
        "typed_memory": "memory/spatial.jsonl",
        "typed_memory_manifest": "memory/spatial_manifest.json",
    },
    "qwen_episodic": {
        "episodic_memory": "memory/episodic.jsonl",
        "episodic_manifest": "memory/episodic_manifest.json",
    },
    "qwen_semantic_visual": {
        "semantic_memory": "memory/semantic.jsonl",
        "visual_memory": "memory/visual.jsonl",
        "qwen_store_envelope": "memory/qwen_store_envelope.json",
    },
    "retrieval_join": {
        "memory_manifest": "memory/memory_manifest.json",
        "retrieval_records": "retrieval/retrieval_records.jsonl",
        "evidence_packs": "retrieval/evidence_packs.jsonl",
    },
    "qa": {
        "qa_rank_shards": "qa/ranks/",
        "qa_manifest": "qa/qa_manifest.json",
    },
    "metrics_report": {
        "metrics": "metrics/metrics.json",
        "student_run_manifest_provisional": (
            "summary/student_run_manifest.provisional.json"
        ),
        "report": "summary/report.md",
    },
    "control_primary": {
        "controller_primary_arbitration": "summary/controller/arbitration/",
        "controller_primary_proposals": "summary/controller/proposals/primary/",
    },
    "control_backup": {
        "controller_backup_arbitration": "summary/controller/arbitration/",
        "controller_backup_proposals": "summary/controller/proposals/backup/",
    },
    "control_actuator": {
        "control_actions": "summary/controller/actions/",
    },
    "student_watchdog": {
        "student_terminal": "summary/student_terminal.json",
        "accounting": "summary/stage_accounting.json",
        "control_summary": "summary/control_summary.json",
    },
}

_STUDENT_STAGE_DEFINITIONS: Final[
    tuple[tuple[str, StudentStageRole, StudentHostClass, int, int, int, int, str], ...]
] = (
    ("preflight_ingest", "preflight", "cpu", 1, 0, 32, 128, "student.preflight"),
    (
        "model_load_workers",
        "load",
        "gpu",
        10,
        8,
        16,
        160,
        "student.model_load_workers",
    ),
    (
        "model_load_gate",
        "gate",
        "cpu",
        1,
        0,
        32,
        128,
        "student.model_load_gate",
    ),
    (
        "model_load_terminal",
        "terminal",
        "cpu",
        1,
        0,
        32,
        128,
        "student.model_load_terminal",
    ),
    (
        "teacher_extract",
        "teacher",
        "gpu",
        10,
        8,
        16,
        160,
        "student.teacher_extract",
    ),
    (
        "merge_materialize",
        "merge",
        "cpu",
        1,
        0,
        32,
        128,
        "student.merge_materialize",
    ),
    ("train", "train", "gpu", 10, 8, 16, 160, "student.train"),
    (
        "spatial_infer",
        "spatial_infer",
        "gpu",
        1,
        1,
        32,
        128,
        "student.spatial_infer",
    ),
    (
        "qwen_episodic",
        "qwen",
        "gpu",
        10,
        8,
        16,
        160,
        "student.qwen_episodic",
    ),
    (
        "qwen_semantic_visual",
        "qwen",
        "gpu",
        10,
        8,
        16,
        160,
        "student.qwen_semantic_visual",
    ),
    (
        "retrieval_join",
        "retrieval",
        "cpu",
        1,
        0,
        32,
        128,
        "student.retrieval_join",
    ),
    ("qa", "qa", "gpu", 10, 8, 16, 160, "student.qa"),
    (
        "metrics_report",
        "report",
        "cpu",
        1,
        0,
        32,
        128,
        "student.metrics_report",
    ),
    (
        "control_primary",
        "controller",
        "cpu",
        1,
        0,
        4,
        16,
        "student.control_primary",
    ),
    (
        "control_backup",
        "controller",
        "cpu",
        1,
        0,
        4,
        16,
        "student.control_backup",
    ),
    (
        "control_actuator",
        "actuator",
        "cpu",
        1,
        0,
        4,
        16,
        "student.control_actuator",
    ),
    (
        "student_watchdog",
        "watchdog",
        "cpu",
        1,
        0,
        8,
        32,
        "student.student_watchdog",
    ),
)

_STUDENT_TIME_LIMITS: Final[dict[str, int]] = {
    "preflight_ingest": 30,
    "model_load_workers": 75,
    "model_load_gate": 30,
    "model_load_terminal": 30,
    "teacher_extract": 120,
    "merge_materialize": 30,
    "spatial_infer": 60,
    "qwen_episodic": 120,
    "qwen_semantic_visual": 120,
    "retrieval_join": 60,
    "qa": 60,
    "metrics_report": 30,
}

_STUDENT_EDGES: Final[
    tuple[tuple[str, str, StudentDependencyKind, str | None], ...]
] = (
    ("model_load_workers", "model_load_gate", "afterany", None),
    ("model_load_gate", "model_load_terminal", "afterany", None),
    (
        "model_load_gate",
        "teacher_extract",
        "afterok",
        "model_load_continue_v1",
    ),
    ("teacher_extract", "merge_materialize", "afterok", "teacher_manifest_v1"),
    (
        "merge_materialize",
        "train",
        "afterok",
        "teacher_cache_contract_v1",
    ),
    ("train", "spatial_infer", "afterok", "checkpoint_v2"),
    (
        "model_load_gate",
        "qwen_episodic",
        "afterok",
        "model_load_continue_v1",
    ),
    (
        "qwen_episodic",
        "qwen_semantic_visual",
        "afterok",
        "episodic_manifest_v1",
    ),
    (
        "spatial_infer",
        "retrieval_join",
        "afterok",
        "spatial_typed_memory_manifest_v1",
    ),
    (
        "qwen_semantic_visual",
        "retrieval_join",
        "afterok",
        "qwen_store_envelope_v1",
    ),
    ("retrieval_join", "qa", "afterok", "retrieval_manifest_v1"),
    ("qa", "metrics_report", "afterok", "qa_manifest_v1"),
)


class StudentRunGraphV1(BaseModel):
    """Closed, immutable source of truth for the remote student run."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["student_run_graph_v1"] = "student_run_graph_v1"
    graph_id: str
    plan_profile: Literal["student"]
    execution_profile: Literal["full", "probe"]
    model_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    student_architecture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stages: tuple[StudentStageSpecV1, ...]
    edges: tuple[StudentEdgeSpecV1, ...]
    terminal_stage_id: Literal["student_watchdog"]

    @model_validator(mode="after")
    def _closed_catalogue(self) -> StudentRunGraphV1:
        expected_ids = tuple(row[0] for row in _STUDENT_STAGE_DEFINITIONS)
        actual_ids = tuple(stage.stage_id for stage in self.stages)
        if actual_ids != expected_ids:
            msg = "stages are not the complete canonical student catalogue"
            raise ValueError(msg)
        by_id = {stage.stage_id: stage for stage in self.stages}
        if any(
            stage.output_keys != _STUDENT_OUTPUT_CATALOGUE[stage.stage_id]
            for stage in self.stages
        ):
            msg = "student stage output catalogue does not match"
            raise ValueError(msg)
        expected_edges = tuple(
            StudentEdgeSpecV1(
                from_stage=source,
                to_stage=target,
                dependency_kind=kind,
                requires_receipt_kind=receipt,
            )
            for source, target, kind, receipt in _STUDENT_EDGES
        )
        if self.edges != expected_edges:
            msg = "edges are not the canonical student fork/join graph"
            raise ValueError(msg)
        gpu_nodes = 10 if self.execution_profile == "full" else 1
        gpu_count = 8 if self.execution_profile == "full" else 1
        for stage in self.stages:
            if (
                stage.host_class == "gpu"
                and stage.stage_id != "spatial_infer"
                and (stage.nodes, stage.gpus_per_node) != (gpu_nodes, gpu_count)
            ):
                msg = "GPU stage does not cover the execution matrix"
                raise ValueError(msg)
            if (
                self.execution_profile == "probe"
                and stage.stage_id == "spatial_infer"
                and (stage.nodes, stage.gpus_per_node) != (1, 1)
            ):
                msg = "probe spatial inference is not pinned 1x1"
                raise ValueError(msg)
        incoming = {edge.to_stage for edge in self.edges}
        if any(
            stage_id in incoming
            for stage_id in (
                "control_primary",
                "control_backup",
                "control_actuator",
                "student_watchdog",
            )
        ):
            msg = "control and watchdog stages must be unheld and independent"
            raise ValueError(msg)
        if by_id[self.terminal_stage_id].role != "watchdog":
            msg = "terminal stage must be the independent watchdog"
            raise ValueError(msg)
        return self


_MAX_TRAIN_TIME_LIMIT_MINUTES: Final = 10_080


class StudentGraphBuildOptions(TypedDict):
    student_architecture_sha256: str
    train_time_limit_minutes: int
    global_deadline_minutes: int


def canonical_student_run_graph(
    *,
    execution_profile: Literal["full", "probe"],
    model_contract_sha256: str,
    provider_lock_sha256: str,
    **options: Unpack[StudentGraphBuildOptions],
) -> StudentRunGraphV1:
    """Build the only admitted full/probe graph from reviewed resource constants."""
    student_architecture_sha256 = options["student_architecture_sha256"]
    train_time_limit_minutes = options["train_time_limit_minutes"]
    global_deadline_minutes = options["global_deadline_minutes"]
    if not 1 <= train_time_limit_minutes <= _MAX_TRAIN_TIME_LIMIT_MINUTES:
        msg = "train time limit must be in 1..10080 minutes"
        raise ValueError(msg)
    if global_deadline_minutes < train_time_limit_minutes:
        msg = "global deadline cannot precede training timeout"
        raise ValueError(msg)
    stages: list[StudentStageSpecV1] = []
    for (
        stage_id,
        role,
        host_class,
        nodes,
        gpus,
        cpus,
        memory,
        command_key,
    ) in _STUDENT_STAGE_DEFINITIONS:
        effective_nodes = nodes
        effective_gpus = gpus
        if execution_profile == "probe" and host_class == "gpu":
            effective_nodes, effective_gpus = 1, 1
        time_limit = _STUDENT_TIME_LIMITS.get(stage_id, global_deadline_minutes)
        if stage_id == "train":
            time_limit = train_time_limit_minutes
        if stage_id == "student_watchdog":
            time_limit = global_deadline_minutes + 10
        stages.append(
            StudentStageSpecV1(
                stage_id=stage_id,
                role=role,
                host_class=host_class,
                nodes=effective_nodes,
                gpus_per_node=effective_gpus,
                cpus_per_task=cpus,
                memory_gb=memory,
                time_limit_minutes=time_limit,
                command_key=command_key,
                output_keys=_STUDENT_OUTPUT_CATALOGUE[stage_id],
            )
        )
    edges = tuple(
        StudentEdgeSpecV1(
            from_stage=source,
            to_stage=target,
            dependency_kind=kind,
            requires_receipt_kind=receipt,
        )
        for source, target, kind, receipt in _STUDENT_EDGES
    )
    return StudentRunGraphV1(
        graph_id=f"student-{execution_profile}-v1",
        plan_profile="student",
        execution_profile=execution_profile,
        model_contract_sha256=model_contract_sha256,
        provider_lock_sha256=provider_lock_sha256,
        student_architecture_sha256=student_architecture_sha256,
        stages=tuple(stages),
        edges=edges,
        terminal_stage_id="student_watchdog",
    )


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
