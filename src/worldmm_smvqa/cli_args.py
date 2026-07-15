from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import override


@dataclass(frozen=True, slots=True)
class CommandResult:
    stdout: str
    stderr: str = ""
    exit_code: int = 0


@dataclass(frozen=True, slots=True)
class ParsedArgs:
    config: Path
    out: Path | None
    run_manifest: Path | None
    input: Path | None
    fixture: Path | None
    question: str | None
    pred: Path | None
    labels: Path | None
    stage: str | None
    store: str | None
    real_model: bool
    backend: str
    retrieval_protocol: str
    max_frame_refs: int
    ablation_stores: str | None
    ablation_protocol: str | None
    local: bool
    dry_run: bool
    submit: bool
    inject_future_memory: bool
    sensor_manifest: Path | None
    observations: Path | None
    frame_root: Path | None
    sensor_audit: Path | None
    experiment_config: Path | None
    profile: str | None


@dataclass(frozen=True, slots=True)
class ParsedValueArgs:
    config: Path
    out: Path | None
    run_manifest: Path | None
    input: Path | None
    fixture: Path | None
    question: str | None
    pred: Path | None
    labels: Path | None
    stage: str | None
    store: str | None
    backend: str
    retrieval_protocol: str
    max_frame_refs: int
    ablation_stores: str | None
    ablation_protocol: str | None
    sensor_manifest: Path | None
    observations: Path | None
    frame_root: Path | None
    sensor_audit: Path | None
    experiment_config: Path | None
    profile: str | None


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    help_text: str
    handler: Callable[[ParsedArgs], CommandResult]
    allowed_options: frozenset[str] | None = None
    required_options: tuple[str, ...] = ()
    mutually_exclusive_options: tuple[frozenset[str], ...] = ()


@dataclass(frozen=True, slots=True)
class CliUsageError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"UsageError: {self.detail}"


@dataclass(frozen=True, slots=True)
class CliPublicationError(Exception):
    command: str
    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        return (
            f"CliPublicationError: {self.command} could not publish "
            f"{self.path}: {self.reason}"
        )


def find_command(command: str, specs: Sequence[CommandSpec]) -> CommandSpec:
    for spec in specs:
        if spec.name == command:
            return spec
    raise CliUsageError(detail=f"unknown command: {command}")


def parse_args(  # noqa: PLR0912, PLR0915
    argv: Sequence[str],
    specs: Sequence[CommandSpec],
) -> tuple[CommandSpec | None, ParsedArgs | None]:
    if not argv or argv[0] in {"-h", "--help"}:
        return None, None

    command = argv[0]
    spec = find_command(command, specs)
    option_tokens = argv[1:]

    value_args = ParsedValueArgs(
        Path("configs/local.example.yaml"),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "mock",
        "worldmm-smvqa",
        32,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    real_model = False
    local = False
    dry_run = False
    submit = False
    inject_future_memory = False
    options_with_values = {
        "--config",
        "--out",
        "--input",
        "--run-manifest",
        "--fixture",
        "--stage",
        "--store",
        "--stores",
        "--question",
        "--pred",
        "--labels",
        "--backend",
        "--retrieval-protocol",
        "--max-frame-refs",
        "--ablation-stores",
        "--ablation-protocol",
        "--sensor-manifest",
        "--observations",
        "--frame-root",
        "--sensor-audit",
        "--experiment-config",
        "--profile",
    }

    provided_options: set[str] = set()
    index = 0
    while index < len(option_tokens):
        option_name = option_tokens[index]
        if option_name == "--real-model":
            real_model = True
            provided_options.add(option_name)
            index += 1
        elif option_name == "--local":
            local = True
            provided_options.add(option_name)
            index += 1
        elif option_name == "--dry-run":
            dry_run = True
            provided_options.add(option_name)
            index += 1
        elif option_name == "--submit":
            submit = True
            provided_options.add(option_name)
            index += 1
        elif option_name == "--inject-future-memory":
            inject_future_memory = True
            provided_options.add(option_name)
            index += 1
        elif option_name in {"-h", "--help"}:
            return spec, None
        elif option_name in options_with_values:
            if index + 1 >= len(option_tokens) or option_tokens[index + 1].startswith(
                "-"
            ):
                raise CliUsageError(detail=f"missing value for {option_name}")
            provided_options.add(option_name)
            value = option_tokens[index + 1]
            value_args = parse_value_option(value_args, option_name, value)
            index += 2
        else:
            raise CliUsageError(detail=f"unknown option: {option_name}")

    if spec.allowed_options is not None:
        irrelevant_options = provided_options - spec.allowed_options
        if irrelevant_options:
            raise CliUsageError(
                detail=(
                    f"{spec.name} does not accept "
                    f"{','.join(sorted(irrelevant_options))}"
                )
            )
    for required_option in spec.required_options:
        if required_option not in provided_options:
            required_detail = required_option
            if spec.name == "launch-remote" and required_option == "--profile":
                required_detail = "--profile teacher-oracle"
            raise CliUsageError(detail=f"{spec.name} requires {required_detail}")
    for mutually_exclusive_options in spec.mutually_exclusive_options:
        if mutually_exclusive_options <= provided_options:
            raise CliUsageError(
                detail=(
                    f"{spec.name} accepts only one of "
                    f"{'/'.join(sorted(mutually_exclusive_options))}"
                )
            )

    parsed = ParsedArgs(
        value_args.config,
        value_args.out,
        value_args.run_manifest,
        value_args.input,
        value_args.fixture,
        value_args.question,
        value_args.pred,
        value_args.labels,
        value_args.stage,
        value_args.store,
        real_model,
        value_args.backend,
        value_args.retrieval_protocol,
        value_args.max_frame_refs,
        value_args.ablation_stores,
        value_args.ablation_protocol,
        local,
        dry_run,
        submit,
        inject_future_memory,
        value_args.sensor_manifest,
        value_args.observations,
        value_args.frame_root,
        value_args.sensor_audit,
        value_args.experiment_config,
        value_args.profile,
    )
    return spec, parsed


def parse_value_option(
    args: ParsedValueArgs,
    option: str,
    value: str,
) -> ParsedValueArgs:
    path_options = {
        "--config",
        "--out",
        "--run-manifest",
        "--input",
        "--fixture",
        "--pred",
        "--labels",
        "--sensor-manifest",
        "--observations",
        "--frame-root",
        "--sensor-audit",
        "--experiment-config",
    }
    text_options = {
        "--question",
        "--stage",
        "--store",
        "--stores",
        "--backend",
        "--retrieval-protocol",
        "--ablation-stores",
        "--profile",
        "--ablation-protocol",
    }
    if option in path_options:
        return _parse_path_option(args, option, value)
    if option in text_options:
        return _parse_text_option(args, option, value)
    if option == "--max-frame-refs":
        return _parse_max_frame_refs(args, value)
    return args


def _parse_path_option(
    args: ParsedValueArgs,
    option: str,
    value: str,
) -> ParsedValueArgs:
    path = Path(value)
    updated = args
    if option == "--config":
        updated = replace(args, config=path)
    elif option == "--out":
        updated = replace(args, out=path)
    elif option == "--run-manifest":
        updated = replace(args, run_manifest=path)
    elif option == "--input":
        updated = replace(args, input=path)
    elif option == "--fixture":
        updated = replace(args, fixture=path)
    elif option == "--pred":
        updated = replace(args, pred=path)
    elif option == "--labels":
        updated = replace(args, labels=path)
    elif option == "--sensor-manifest":
        updated = replace(args, sensor_manifest=path)
    elif option == "--observations":
        updated = replace(args, observations=path)
    elif option == "--frame-root":
        updated = replace(args, frame_root=path)
    elif option == "--sensor-audit":
        updated = replace(args, sensor_audit=path)
    elif option == "--experiment-config":
        updated = replace(args, experiment_config=path)
    return updated


def _parse_text_option(
    args: ParsedValueArgs,
    option: str,
    value: str,
) -> ParsedValueArgs:
    updated = args
    if option == "--question":
        updated = replace(args, question=value)
    elif option == "--stage":
        updated = replace(args, stage=value)
    elif option in {"--store", "--stores"}:
        updated = replace(args, store=value)
    elif option == "--backend":
        updated = replace(args, backend=value)
    elif option == "--retrieval-protocol":
        updated = replace(args, retrieval_protocol=value)
    elif option == "--ablation-stores":
        updated = replace(args, ablation_stores=value)
    elif option == "--ablation-protocol":
        updated = replace(args, ablation_protocol=value)
    elif option == "--profile":
        updated = replace(args, profile=value)
    return updated


def _parse_max_frame_refs(args: ParsedValueArgs, value: str) -> ParsedValueArgs:
    try:
        max_frame_refs = int(value)
    except ValueError as exc:
        raise CliUsageError(detail=f"invalid --max-frame-refs: {value}") from exc
    return replace(args, max_frame_refs=max_frame_refs)
