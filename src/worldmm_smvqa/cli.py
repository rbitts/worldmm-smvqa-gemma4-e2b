from __future__ import annotations

import sys
from collections.abc import Sequence

from worldmm_smvqa.chunking import TemporalOrderError
from worldmm_smvqa.cli_args import (
    CliPublicationError,
    CliUsageError,
    CommandResult,
    CommandSpec,
    find_command,
    parse_args,
)
from worldmm_smvqa.cli_commands import (
    handle_audit_sensors,
    handle_build_memory,
    handle_diagnose_spatial,
    handle_evaluate,
    handle_launch_remote,
    handle_mock_dag,
    handle_preflight,
    handle_prepare_fixture,
    handle_qa,
    handle_report,
    handle_retrieve,
    handle_retrieve_batch,
    handle_smoke,
    handle_validate_schema,
    handle_validate_teacher_oracle_inputs,
)
from worldmm_smvqa.config import (
    ConfigNotFoundError,
    MalformedConfigError,
    MissingRemoteConfigError,
    RemoteOnlyError,
)
from worldmm_smvqa.fixtures import FixtureValidationError
from worldmm_smvqa.metrics import (
    InvalidPredictionError,
)
from worldmm_smvqa.qa import QABackendUnavailableError, QAParseError
from worldmm_smvqa.remote_plan import (
    ExplicitApprovalRequiredError,
    TeacherOraclePlanRequiredError,
)
from worldmm_smvqa.report import IncompleteRemoteManifestError
from worldmm_smvqa.retrieval import (
    InvalidRetrievalStoreError,
)
from worldmm_smvqa.sensor_frames import SensorFrameManifestError
from worldmm_smvqa.smoke import NoLocalModelBackendError
from worldmm_smvqa.transformers_backend import TransformersGenerationError
from worldmm_smvqa.worldmm.episodic import (
    InvalidTemporalGraphError,
)
from worldmm_smvqa.worldmm.llm_errors import LLMMemoryError
from worldmm_smvqa.worldmm.spatial_compression import SpatialCompressionError
from worldmm_smvqa.worldmm.spatial_diagnostics import SpatialDiagnosticsError
from worldmm_smvqa.worldmm.visual import (
    MissingGroundingError,
)


def command_specs() -> tuple[CommandSpec, ...]:
    return (
        CommandSpec("prepare-fixture", "Prepare tiny fixture.", handle_prepare_fixture),
        CommandSpec("validate-schema", "Validate schema.", handle_validate_schema),
        CommandSpec("preflight", "Inspect prepared dataset.", handle_preflight),
        CommandSpec(
            "audit-sensors",
            "Audit causal sensor coverage.",
            handle_audit_sensors,
            allowed_options=frozenset(
                {"--sensor-manifest", "--observations", "--frame-root", "--out"}
            ),
            required_options=(
                "--sensor-manifest",
                "--observations",
                "--frame-root",
                "--out",
            ),
        ),
        CommandSpec(
            "validate-teacher-oracle-inputs",
            "Validate teacher-oracle production inputs.",
            handle_validate_teacher_oracle_inputs,
            allowed_options=frozenset(
                {"--sensor-audit", "--experiment-config", "--out"}
            ),
            required_options=("--sensor-audit", "--experiment-config", "--out"),
        ),
        CommandSpec(
            "build-memory",
            "Build benchmark memory artifacts.",
            handle_build_memory,
        ),
        CommandSpec("retrieve", "Retrieve causal evidence packs.", handle_retrieve),
        CommandSpec(
            "retrieve-batch",
            "Retrieve all evidence packs.",
            handle_retrieve_batch,
        ),
        CommandSpec("qa", "Run QA over evidence packs.", handle_qa),
        CommandSpec("evaluate", "Evaluate prediction files.", handle_evaluate),
        CommandSpec(
            "diagnose-spatial",
            "Write spatial retrieval diagnostics.",
            handle_diagnose_spatial,
        ),
        CommandSpec("report", "Write a run handoff report.", handle_report),
        CommandSpec(
            "mock-dag",
            "Validate the model-free production-consumer DAG.",
            handle_mock_dag,
            allowed_options=frozenset(
                {"--config", "--fixture", "--student-architecture"},
            ),
            required_options=("--fixture",),
        ),
        CommandSpec("smoke", "Run the tiny local pipeline.", handle_smoke),
        CommandSpec(
            "launch-remote",
            "Print remote commands.",
            handle_launch_remote,
            allowed_options=frozenset(
                {
                    "--config",
                    "--out",
                    "--profile",
                    "--experiment-config",
                    "--dry-run",
                    "--submit",
                }
            ),
            required_options=("--out", "--profile", "--experiment-config"),
            mutually_exclusive_options=(frozenset({"--dry-run", "--submit"}),),
        ),
    )


def top_help() -> str:
    specs = command_specs()
    command_list = ",".join(spec.name for spec in specs)
    lines = [
        "usage: worldmm-smvqa [-h]",
        f"                     {{{command_list}}} ...",
        "",
        "WorldMM-SMVQA benchmark scaffold.",
        "",
        "commands:",
    ]
    lines.extend(f"  {spec.name:<18} {spec.help_text}" for spec in specs)
    return "\n".join(lines) + "\n"


def command_help(command: str) -> str:
    spec = find_command(command, command_specs())
    usage_by_command = {
        "audit-sensors": (
            "worldmm-smvqa audit-sensors --sensor-manifest SENSOR_MANIFEST "
            "--observations OBSERVATIONS --frame-root FRAME_ROOT --out OUT"
        ),
        "validate-teacher-oracle-inputs": (
            "worldmm-smvqa validate-teacher-oracle-inputs "
            "--sensor-audit SENSOR_AUDIT --experiment-config EXPERIMENT_CONFIG "
            "--out OUT"
        ),
        "mock-dag": (
            "worldmm-smvqa mock-dag [--config CONFIG] --fixture FIXTURE "
            "[--student-architecture STUDENT_ARCHITECTURE]"
        ),
        "launch-remote": (
            "worldmm-smvqa launch-remote [--config CONFIG] --out OUT "
            "--profile PROFILE --experiment-config EXPERIMENT_CONFIG "
            "(--dry-run | --submit)"
        ),
    }
    usage = usage_by_command.get(
        spec.name,
        f"worldmm-smvqa {spec.name} [--config CONFIG] [--help]",
    )
    return "\n".join(["usage: " + usage, "", spec.help_text]) + "\n"


def run(argv: Sequence[str]) -> CommandResult:
    spec, args = parse_args(argv, command_specs())
    if spec is None:
        return CommandResult(stdout=top_help())
    if args is None:
        return CommandResult(stdout=command_help(spec.name))
    return spec.handler(args)


def main() -> int:
    try:
        result = run(sys.argv[1:])
    except (
        CliPublicationError,
        CliUsageError,
        ConfigNotFoundError,
        FixtureValidationError,
        InvalidPredictionError,
        InvalidTemporalGraphError,
        InvalidRetrievalStoreError,
        QABackendUnavailableError,
        QAParseError,
        MissingGroundingError,
        MissingRemoteConfigError,
        ExplicitApprovalRequiredError,
        TeacherOraclePlanRequiredError,
        MalformedConfigError,
        RemoteOnlyError,
        TemporalOrderError,
        NoLocalModelBackendError,
        IncompleteRemoteManifestError,
        LLMMemoryError,
        SpatialCompressionError,
        SpatialDiagnosticsError,
        SensorFrameManifestError,
        TransformersGenerationError,
    ) as exc:
        _ = sys.stderr.write(f"{exc}\n")
        return 2
    _ = sys.stdout.write(result.stdout)
    _ = sys.stderr.write(result.stderr)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
