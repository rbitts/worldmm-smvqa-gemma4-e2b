# ruff: noqa: EM101, EM102, PLR0912, PLR0913, TRY003
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

PLAN_SCHEMA: Final = "memory-alignment-plan-v1"
SUITE_ID: Final = "memory-alignment-four-store-suite-v1"
PROTOCOL_ID: Final = "memory-recall6-paired-bootstrap-v1"
SHA256_LENGTH: Final = 64


class PlanRenderError(ValueError):
    """Raised when render-only planning inputs are invalid."""


@dataclass(frozen=True, slots=True)
class RenderedPlan:
    root: Path
    plan_path: Path
    review_path: Path


_COMPARISONS: Final = (
    {
        "comparison_id": "visual_primary",
        "baseline": {"role": "baseline", "store": "visual"},
        "candidate": {"role": "candidate", "store": "visual"},
        "request_store": "visual",
    },
    {
        "comparison_id": "episodic_primary",
        "baseline": {"role": "baseline", "store": "episodic"},
        "candidate": {"role": "candidate", "store": "episodic"},
        "request_store": "episodic",
    },
    {
        "comparison_id": "semantic_primary",
        "baseline": {"role": "baseline", "store": "semantic"},
        "candidate": {"role": "candidate", "store": "semantic"},
        "request_store": "semantic",
    },
    {
        "comparison_id": "semantic_rebuild",
        "baseline": {"role": "baseline", "store": "semantic"},
        "candidate": {
            "role": "candidate_semantic_rebuild",
            "store": "semantic_rebuild",
        },
        "request_store": "semantic",
    },
)

_BLOCKERS: Final = (
    "trusted sealed baseline and candidate bundles are required",
    "complete trusted coverage manifests are required",
    "a supplied fixed comparison cohort is required",
    "separate approval is required for execution or submission",
)

_DEFERRALS: Final = (
    "bundle production and raw provenance",
    "remote execution and scheduler submission",
    "model loading, generation, training, and evaluation",
    "QA identity attestation and scoring",
    "artifact promotion, rollback, and Android deployment",
)


def render_comparison_plan(
    *,
    config: Path,
    repository_root: Path,
    baseline_manifest: Path,
    candidate_manifest: Path,
    cohort: Path,
    out: Path,
) -> RenderedPlan:
    """Render review artifacts without executing or describing an execution path."""
    root = _regular_directory(repository_root, "repository root")
    config_path = _regular_file(config, "config")
    baseline_path = _regular_file(baseline_manifest, "baseline manifest")
    candidate_path = _regular_file(candidate_manifest, "candidate manifest")
    cohort_path = _regular_file(cohort, "cohort")

    config_bytes = _read(config_path, "config")
    baseline_bytes = _read(baseline_path, "baseline manifest")
    candidate_bytes = _read(candidate_path, "candidate manifest")
    cohort_bytes = _read(cohort_path, "cohort")

    config_values = _parse_simple_yaml(config_bytes, config_path)
    baseline = _parse_json_object(baseline_bytes, baseline_path)
    candidate = _parse_json_object(candidate_bytes, candidate_path)
    cohort_value = _parse_json_object(cohort_bytes, cohort_path)
    contract = _validate_inputs(
        root=root,
        config_values=config_values,
        baseline=baseline,
        candidate=candidate,
        cohort=cohort_value,
    )

    plan = {
        "schema_version": PLAN_SCHEMA,
        "submission": False,
        "inputs": {
            "config_sha256": _sha256(config_bytes),
            "baseline_manifest_sha256": _sha256(baseline_bytes),
            "candidate_manifest_sha256": _sha256(candidate_bytes),
            "cohort_sha256": _sha256(cohort_bytes),
        },
        "contracts": contract,
        "comparison_suite_id": SUITE_ID,
        "comparisons": list(_COMPARISONS),
        "scientific_protocol": {
            "protocol_id": PROTOCOL_ID,
            "k": "6",
            "paired_bootstrap_replicates": "10000",
            "confidence_interval": "paired_bootstrap_95_percentile",
            "confidence_interval_indexes": ["249", "9749"],
            "non_inferiority_threshold": "-0.05",
            "decision": "pass only when every CI lower bound is at least -0.05",
        },
        "blockers": list(_BLOCKERS),
        "deferred": list(_DEFERRALS),
    }
    _assert_non_executable(plan)
    plan_bytes = (
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    review_bytes = _render_review(plan).encode()
    return _publish_no_clobber(out, plan_bytes, review_bytes)


def _validate_inputs(
    *,
    root: Path,
    config_values: Mapping[str, Mapping[str, str]],
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    cohort: Mapping[str, object],
) -> dict[str, object]:
    alignment = config_values.get("memory_alignment")
    if alignment is None:
        raise PlanRenderError("config missing memory_alignment section")
    required = {"contract_id", "contract_path", "contract_sha256"}
    if not required <= alignment.keys():
        missing = ", ".join(sorted(required - alignment.keys()))
        raise PlanRenderError(f"config missing memory_alignment fields: {missing}")
    if alignment["contract_id"] != "worldmm-smvqa-memory-v2":
        raise PlanRenderError("config candidate contract ID is not reviewed v2")
    candidate_contract_path = alignment["contract_path"]
    candidate_contract_sha = alignment["contract_sha256"]
    _validate_sha(candidate_contract_sha, "config candidate contract digest")
    contract_path = _repository_file(root, candidate_contract_path)
    if _sha256(_read(contract_path, "candidate contract")) != candidate_contract_sha:
        raise PlanRenderError("config candidate contract digest mismatch")

    baseline_selection = _contract_selection(baseline, "baseline")
    candidate_selection = _contract_selection(candidate, "candidate")
    if baseline.get("schema_version") != "sealed-memory-bundle-v1":
        raise PlanRenderError("baseline manifest has wrong schema_version")
    if candidate.get("schema_version") != "sealed-memory-bundle-v1":
        raise PlanRenderError("candidate manifest has wrong schema_version")
    if cohort.get("schema_version") != "memory-comparison-cohort-v1":
        raise PlanRenderError("cohort has wrong schema_version")
    if baseline.get("role") != "baseline" or candidate.get("role") != "candidate":
        raise PlanRenderError("bundle roles must be baseline and candidate")
    if candidate_selection["contract_id"] != alignment["contract_id"]:
        raise PlanRenderError("candidate manifest contract ID differs from config")
    if candidate_selection["contract_path"] != candidate_contract_path:
        raise PlanRenderError("candidate manifest contract path differs from config")
    if candidate_selection["expected_contract_file_sha256"] != candidate_contract_sha:
        raise PlanRenderError("candidate manifest contract digest differs from config")
    if baseline_selection["version"] != "v1" or candidate_selection["version"] != "v2":
        raise PlanRenderError(
            "bundle contract versions must be baseline v1 and candidate v2"
        )

    for label, selection in (
        ("baseline", baseline_selection),
        ("candidate", candidate_selection),
    ):
        _validate_sha(
            selection["expected_contract_file_sha256"],
            f"{label} contract digest",
        )

    return {
        "baseline": baseline_selection,
        "candidate": candidate_selection,
    }


def _contract_selection(manifest: Mapping[str, object], label: str) -> dict[str, str]:
    value = manifest.get("contract_selection")
    if not isinstance(value, dict):
        raise PlanRenderError(f"{label} manifest missing contract_selection")
    fields = (
        "schema_version",
        "version",
        "contract_id",
        "contract_path",
        "expected_contract_file_sha256",
    )
    if set(value) != set(fields) or not all(
        isinstance(value[key], str) for key in fields
    ):
        raise PlanRenderError(f"{label} manifest has invalid contract_selection")
    if value["schema_version"] != "contract-selection-v1":
        raise PlanRenderError(f"{label} manifest has invalid contract selection schema")
    return {key: value[key] for key in fields}


def _parse_simple_yaml(data: bytes, path: Path) -> dict[str, dict[str, str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PlanRenderError(f"config is not UTF-8: {path}") from exc
    values: dict[str, dict[str, str]] = {}
    section: str | None = None
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            if not line.endswith(":"):
                key, separator, value = line.partition(":")
                if separator and value.strip():
                    values.setdefault("_root", {})[key.strip()] = _yaml_scalar(value)
                    section = None
                    continue
                raise PlanRenderError(f"config line {line_number} is invalid")
            section = line[:-1].strip()
            if not section or section in values:
                raise PlanRenderError(f"config line {line_number} has invalid section")
            values[section] = {}
            continue
        if section is None:
            raise PlanRenderError(f"config line {line_number} has no section")
        key, separator, value = line.strip().partition(":")
        if not separator or not key or key in values[section]:
            raise PlanRenderError(f"config line {line_number} is invalid")
        values[section][key] = _yaml_scalar(value)
    return values


def _yaml_scalar(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _parse_json_object(data: bytes, path: Path) -> dict[str, object]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanRenderError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PlanRenderError(f"JSON root must be an object: {path}")
    return value


def _repository_file(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or "\\" in relative
    ):
        raise PlanRenderError("candidate contract path is not repository-relative")
    path = root.joinpath(candidate)
    for parent in (path, *path.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise PlanRenderError("candidate contract path contains a symbolic link")
    resolved = _regular_file(path, "candidate contract")
    if not resolved.is_relative_to(root):
        raise PlanRenderError("candidate contract path escapes repository root")
    return resolved


def _regular_directory(path: Path, label: str) -> Path:
    try:
        if path.is_symlink() or not path.is_dir():
            raise PlanRenderError(f"{label} is not a regular directory: {path}")
        return path.resolve(strict=True)
    except OSError as exc:
        raise PlanRenderError(f"cannot open {label}: {path}") from exc


def _regular_file(path: Path, label: str) -> Path:
    try:
        if path.is_symlink() or not path.is_file():
            raise PlanRenderError(f"{label} is not a regular file: {path}")
        return path.resolve(strict=True)
    except OSError as exc:
        raise PlanRenderError(f"cannot open {label}: {path}") from exc


def _read(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PlanRenderError(f"cannot read {label}: {path}") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_sha(value: str, label: str) -> None:
    if len(value) != SHA256_LENGTH or value.lower() != value:
        raise PlanRenderError(f"{label} is not lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise PlanRenderError(f"{label} is not lowercase SHA-256") from exc


def _assert_non_executable(value: object) -> None:
    forbidden = {
        "command",
        "host",
        "url",
        "environment",
        "approval",
        "secret",
        "scheduler",
        "submit_command",
        "execute",
    }

    def walk(item: object) -> None:
        if isinstance(item, dict):
            overlap = forbidden & {str(key).lower() for key in item}
            if overlap:
                detail = f"rendered plan contains forbidden fields: {overlap}"
                raise AssertionError(detail)
            for child in item.values():
                walk(child)
        elif isinstance(item, list | tuple):
            for child in item:
                walk(child)

    walk(value)


def _render_review(plan: Mapping[str, object]) -> str:
    comparisons = plan["comparisons"]
    if not isinstance(comparisons, list):
        raise TypeError("comparisons must be a list")
    comparison_lines = "\n".join(
        f"- {row['comparison_id']}: {row['baseline']['store']} -> "
        f"{row['candidate']['store']}"
        for row in comparisons
    )
    blockers = "\n".join(f"- {item}" for item in cast("list[str]", plan["blockers"]))
    deferred = "\n".join(f"- {item}" for item in cast("list[str]", plan["deferred"]))
    return (
        "# Status\n\nRender-only comparison plan; no work was submitted or "
        "executed.\n\n"
        "# Inputs\n\nInput and contract digests are recorded in "
        "comparison-plan.json.\n\n"
        f"# Comparisons\n\n{comparison_lines}\n\n"
        "# Scientific protocol\n\nRecall@6 with 10,000 paired-bootstrap "
        "replicates and a -0.05 non-inferiority threshold.\n\n"
        f"# Blockers\n\n{blockers}\n\n"
        f"# Deferred\n\n{deferred}\n\n"
        "Submission supported: no\n"
    )


def _publish_no_clobber(out: Path, plan: bytes, review: bytes) -> RenderedPlan:
    try:
        out.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise PlanRenderError(f"output already exists: {out}") from exc
    except OSError as exc:
        raise PlanRenderError(f"cannot create output: {out}") from exc
    plan_path = out / "comparison-plan.json"
    review_path = out / "review.md"
    try:
        _exclusive_write(plan_path, plan)
        _exclusive_write(review_path, review)
    except Exception:
        for path in (review_path, plan_path):
            with suppress(OSError):
                path.unlink(missing_ok=True)
        with suppress(OSError):
            out.rmdir()
        raise
    return RenderedPlan(root=out, plan_path=plan_path, review_path=review_path)


def _exclusive_write(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
