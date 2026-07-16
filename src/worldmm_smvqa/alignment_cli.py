# ruff: noqa: BLE001, EM101, EM102, PLC0415, PLR0913, T201, TRY003
# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnusedCallResult=false
from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, cast

from worldmm_smvqa.memory_alignment_plan import render_comparison_plan


class _Report(Protocol):
    status: str


class AlignmentCliUsageError(ValueError):
    """Raised for command syntax and option errors."""


@dataclass(frozen=True, slots=True)
class _Command:
    options: dict[str, str]


_COMMAND_OPTIONS: Final = {
    "validate-contract": frozenset(
        {
            "--repository-root",
            "--contract-path",
            "--expected-contract-sha256",
            "--version",
        }
    ),
    "evaluate": frozenset(
        {
            "--repository-root",
            "--baseline-contract-path",
            "--baseline-contract-sha256",
            "--candidate-contract-path",
            "--candidate-contract-sha256",
            "--baseline",
            "--candidate",
            "--cohort",
            "--out",
        }
    ),
    "render-plan": frozenset(
        {
            "--config",
            "--repository-root",
            "--baseline-manifest",
            "--candidate-manifest",
            "--cohort",
            "--out",
        }
    ),
}


class _Evaluate(Protocol):
    def __call__(
        self,
        *,
        repository_root: Path,
        baseline_contract_path: str,
        baseline_contract_sha256: str,
        candidate_contract_path: str,
        candidate_contract_sha256: str,
        baseline_bundle: Path,
        candidate_bundle: Path,
        cohort_path: Path,
    ) -> _Report: ...


class _WriteReport(Protocol):
    def __call__(self, path: Path, report: _Report) -> None: ...


def main(argv: Sequence[str] | None = None) -> int:
    """Run the isolated alignment contract, evaluator, or renderer command."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        name, command = _parse(arguments)
        if name == "validate-contract":
            return _validate_contract(command)
        if name == "evaluate":
            return _evaluate(command)
        return _render_plan(command)
    except (ValueError, OSError) as exc:
        print(f"alignment-cli: {exc}", file=sys.stderr)
        return 2


def _parse(argv: tuple[str, ...]) -> tuple[str, _Command]:
    if not argv or argv[0] not in _COMMAND_OPTIONS:
        raise AlignmentCliUsageError(
            "expected validate-contract, evaluate, or render-plan command"
        )
    name = argv[0]
    allowed = _COMMAND_OPTIONS[name]
    values: dict[str, str] = {}
    index = 1
    while index < len(argv):
        option = argv[index]
        if not option.startswith("--"):
            raise AlignmentCliUsageError(f"positional argument is forbidden: {option}")
        if option not in allowed:
            raise AlignmentCliUsageError(f"unknown option for {name}: {option}")
        if option in values:
            raise AlignmentCliUsageError(f"duplicate option: {option}")
        if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            raise AlignmentCliUsageError(f"missing value for {option}")
        value = argv[index + 1]
        if not value:
            raise AlignmentCliUsageError(f"empty value for {option}")
        values[option] = value
        index += 2
    missing = allowed - values.keys()
    if missing:
        raise AlignmentCliUsageError(
            "missing required options: " + ", ".join(sorted(missing))
        )
    return name, _Command(values)


def _validate_contract(command: _Command) -> int:
    version = command.options["--version"]
    if version not in {"v1", "v2"}:
        raise AlignmentCliUsageError("--version must be v1 or v2")
    try:
        from worldmm_smvqa.memory_alignment import load_contract_selection
    except ImportError as exc:
        raise AlignmentCliUsageError("alignment evaluator is unavailable") from exc
    try:
        load_contract_selection(
            Path(command.options["--repository-root"]),
            command.options["--contract-path"],
            command.options["--expected-contract-sha256"],
            version=cast("Literal['v1', 'v2']", version),
        )
    except (FileNotFoundError, OSError):
        raise
    except Exception as exc:  # Contract schema/digest failure is an observed result.
        print(f"contract_invalid: {exc}", file=sys.stderr)
        return 1
    return 0


def _evaluate(command: _Command) -> int:
    try:
        from worldmm_smvqa.memory_alignment import (
            atomic_write_report_no_clobber,
            evaluate_memory_alignment,
        )
    except ImportError as exc:
        raise AlignmentCliUsageError("alignment evaluator is unavailable") from exc
    evaluate = cast("_Evaluate", evaluate_memory_alignment)
    write_report = cast("_WriteReport", atomic_write_report_no_clobber)
    report = evaluate(
        repository_root=Path(command.options["--repository-root"]),
        baseline_contract_path=command.options["--baseline-contract-path"],
        baseline_contract_sha256=command.options["--baseline-contract-sha256"],
        candidate_contract_path=command.options["--candidate-contract-path"],
        candidate_contract_sha256=command.options["--candidate-contract-sha256"],
        baseline_bundle=Path(command.options["--baseline"]),
        candidate_bundle=Path(command.options["--candidate"]),
        cohort_path=Path(command.options["--cohort"]),
    )
    write_report(Path(command.options["--out"]), report)
    return 0 if report.status == "pass" else 1


def _render_plan(command: _Command) -> int:
    render_comparison_plan(
        config=Path(command.options["--config"]),
        repository_root=Path(command.options["--repository-root"]),
        baseline_manifest=Path(command.options["--baseline-manifest"]),
        candidate_manifest=Path(command.options["--candidate-manifest"]),
        cohort=Path(command.options["--cohort"]),
        out=Path(command.options["--out"]),
    )
    return 0


_ENTRY_POINT: Callable[[Sequence[str] | None], int] = main


if __name__ == "__main__":
    raise SystemExit(main())
