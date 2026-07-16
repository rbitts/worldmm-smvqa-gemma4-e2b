# ruff: noqa: EM101, TRY003
# pyright: reportAny=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from worldmm_smvqa.alignment_cli import main

_EVALUATE_OPTIONS = (
    "--repository-root",
    "/repo",
    "--baseline-contract-path",
    "configs/v1.json",
    "--baseline-contract-sha256",
    "a" * 64,
    "--candidate-contract-path",
    "configs/v2.json",
    "--candidate-contract-sha256",
    "b" * 64,
    "--baseline",
    "/baseline",
    "--candidate",
    "/candidate",
    "--cohort",
    "/cohort.json",
    "--out",
    "/report.json",
)


def test_unknown_and_duplicate_options_fail_before_evaluator_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = ModuleType("worldmm_smvqa.memory_alignment")

    def fail_import(_name: str) -> None:
        pytest.fail("evaluator imported before CLI validation")

    sentinel.__getattr__ = fail_import  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "worldmm_smvqa.memory_alignment", sentinel)

    assert main(("evaluate", *_EVALUATE_OPTIONS, "--submit", "yes")) == 2
    assert (
        main(
            (
                "evaluate",
                *_EVALUATE_OPTIONS,
                "--repository-root",
                "/other",
            )
        )
        == 2
    )
    assert main(("evaluate", *_EVALUATE_OPTIONS, "positional")) == 2


def test_evaluate_routes_exact_options_and_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, object]] = []
    fake = ModuleType("worldmm_smvqa.memory_alignment")

    def evaluate_memory_alignment(**kwargs: object) -> object:
        calls.append(("evaluate", kwargs))
        return SimpleNamespace(status="scientific_fail")

    def atomic_write_report_no_clobber(path: Path, report: object) -> None:
        calls.append(("write", (path, report)))

    fake.evaluate_memory_alignment = evaluate_memory_alignment  # type: ignore[attr-defined]
    fake.atomic_write_report_no_clobber = atomic_write_report_no_clobber  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "worldmm_smvqa.memory_alignment", fake)
    options = list(_EVALUATE_OPTIONS)
    options[options.index("/report.json")] = str(tmp_path / "report.json")

    assert main(("evaluate", *options)) == 1
    assert [name for name, _value in calls] == ["evaluate", "write"]
    kwargs = calls[0][1]
    assert isinstance(kwargs, dict)
    assert kwargs == {
        "repository_root": Path("/repo"),
        "baseline_contract_path": "configs/v1.json",
        "baseline_contract_sha256": "a" * 64,
        "candidate_contract_path": "configs/v2.json",
        "candidate_contract_sha256": "b" * 64,
        "baseline_bundle": Path("/baseline"),
        "candidate_bundle": Path("/candidate"),
        "cohort_path": Path("/cohort.json"),
    }


def test_evaluate_preflight_error_exits_two_without_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = ModuleType("worldmm_smvqa.memory_alignment")
    wrote_report = False

    def evaluate_memory_alignment(**_kwargs: object) -> object:
        raise ValueError("contract path mismatch")

    def atomic_write_report_no_clobber(_path: Path, _report: object) -> None:
        nonlocal wrote_report
        wrote_report = True

    fake.evaluate_memory_alignment = evaluate_memory_alignment  # type: ignore[attr-defined]
    fake.atomic_write_report_no_clobber = atomic_write_report_no_clobber  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "worldmm_smvqa.memory_alignment", fake)

    assert main(("evaluate", *_EVALUATE_OPTIONS)) == 2
    assert wrote_report is False


def test_validate_contract_maps_invalid_contract_to_exit_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = ModuleType("worldmm_smvqa.memory_alignment")

    def load_contract_selection(*_args: object, **_kwargs: object) -> object:
        raise ValueError("digest mismatch")

    fake.load_contract_selection = load_contract_selection  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "worldmm_smvqa.memory_alignment", fake)

    assert (
        main(
            (
                "validate-contract",
                "--repository-root",
                "/repo",
                "--contract-path",
                "configs/v1.json",
                "--expected-contract-sha256",
                "a" * 64,
                "--version",
                "v1",
            )
        )
        == 1
    )


def test_render_plan_rejects_submission_option_before_reading_inputs() -> None:
    assert (
        main(
            (
                "render-plan",
                "--config",
                "/missing/config",
                "--repository-root",
                "/missing/repo",
                "--baseline-manifest",
                "/missing/baseline",
                "--candidate-manifest",
                "/missing/candidate",
                "--cohort",
                "/missing/cohort",
                "--out",
                "/missing/out",
                "--execute",
                "yes",
            )
        )
        == 2
    )


def test_alignment_cli_fresh_import_excludes_operational_modules() -> None:
    script = """
import sys
import worldmm_smvqa.alignment_cli
forbidden = {
    "worldmm_smvqa.cli",
    "worldmm_smvqa.cli_commands",
    "worldmm_smvqa.qa_transformers",
    "worldmm_smvqa.remote_plan",
    "worldmm_smvqa.report",
    "transformers",
    "torch",
    "requests",
    "socket",
    "subprocess",
}
loaded = sorted(forbidden & set(sys.modules))
raise SystemExit("forbidden imports: " + ",".join(loaded) if loaded else 0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
