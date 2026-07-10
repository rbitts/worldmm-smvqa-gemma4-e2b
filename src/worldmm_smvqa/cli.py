from __future__ import annotations

import sys
from collections.abc import Sequence

from worldmm_smvqa.chunking import TemporalOrderError
from worldmm_smvqa.cli_args import (
    CliUsageError,
    CommandResult,
    CommandSpec,
    find_command,
    parse_args,
)
from worldmm_smvqa.cli_commands import (
    handle_build_memory,
    handle_diagnose_spatial,
    handle_evaluate,
    handle_launch_remote,
    handle_prepare_fixture,
    handle_qa,
    handle_report,
    handle_retrieve,
    handle_retrieve_batch,
    handle_smoke,
    handle_validate_schema,
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
from worldmm_smvqa.remote_plan import ExplicitApprovalRequiredError
from worldmm_smvqa.report import IncompleteRemoteManifestError
from worldmm_smvqa.retrieval import (
    InvalidRetrievalStoreError,
)
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
        CommandSpec("build-memory", "Build memory placeholder.", handle_build_memory),
        CommandSpec("retrieve", "Retrieve evidence placeholder.", handle_retrieve),
        CommandSpec(
            "retrieve-batch",
            "Retrieve all evidence packs.",
            handle_retrieve_batch,
        ),
        CommandSpec("qa", "Run QA placeholder.", handle_qa),
        CommandSpec("evaluate", "Evaluate predictions placeholder.", handle_evaluate),
        CommandSpec(
            "diagnose-spatial",
            "Write spatial retrieval diagnostics.",
            handle_diagnose_spatial,
        ),
        CommandSpec("report", "Write report placeholder.", handle_report),
        CommandSpec("smoke", "Run local smoke placeholder.", handle_smoke),
        CommandSpec("launch-remote", "Print remote commands.", handle_launch_remote),
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
    return "\n".join(
        [
            f"usage: worldmm-smvqa {spec.name} [--config CONFIG] [--help]",
            "",
            spec.help_text,
            "",
            "common options:",
            "  --config CONFIG",
            "  --out OUT",
            "  --run-manifest RUN_MANIFEST",
            "  --input INPUT",
            "  --fixture FIXTURE",
            "",
            "command options:",
            "  --stage STAGE",
            "  --store STORE",
            "  --stores STORES",
            "  --question QUESTION",
            "  --pred PRED",
            "  --labels LABELS",
            "  --real-model",
            "  --backend BACKEND",
            "  --local",
            "  --dry-run",
            "  --submit",
            "  --inject-future-memory",
        ],
    ) + "\n"


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
        MalformedConfigError,
        RemoteOnlyError,
        TemporalOrderError,
        NoLocalModelBackendError,
        IncompleteRemoteManifestError,
        LLMMemoryError,
        SpatialCompressionError,
        SpatialDiagnosticsError,
        TransformersGenerationError,
    ) as exc:
        _ = sys.stderr.write(f"{exc}\n")
        return 2
    _ = sys.stdout.write(result.stdout)
    _ = sys.stderr.write(result.stderr)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
