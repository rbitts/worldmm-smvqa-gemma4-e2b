from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shlex
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final, Literal, cast

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from worldmm_smvqa.attestation import (
    AttestationError,
    ImmutableAttestationKeyRegistryV1,
    SignedAttestationEnvelopeV1,
    verify_signed_attestation_envelope,
)
from worldmm_smvqa.schema import (
    ANSWER_CHOICE_COUNT,
    LocalTimedModel,
    QALabelExample,
    QuestionRequest,
    SourceStreamExample,
    is_unanswerable_choice,
)
from worldmm_smvqa.sensor_audit import SensorAuditReport
from worldmm_smvqa.sensor_frames import (
    SensorFrameManifestError,
    build_sensor_frame_manifest,
)
from worldmm_smvqa.worldmm.spatial_diagnostics import STORES

PREFLIGHT_VERSION: Final = "smvqa-preflight-v1"
FRAME_EXTENSIONS: Final = (".jpg", ".jpeg", ".png", ".webp")
STORE_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_-]*$")
TIMESTAMP_EPSILON: Final = 1e-9
_MAX_ACCOUNTING_SETTLE_SECONDS: Final = 3600
_ACCOUNTING_COMMAND_TOKEN_COUNT: Final = 8
EPOCH_SCALE: Final = 100_000_000.0
PREVIEW_LIMIT: Final = 3
EVIDENCE_SPAN_PARTS: Final = 4
PERCENT_FULL: Final = 100.0
_PRODUCER_INPUT_ALLOWLIST: Final = frozenset({"sources.jsonl", "sensor_frames.jsonl"})
_JSON_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


class PreflightIssue(BaseModel):
    code: str
    message: str
    record_id: str | None = None


ORACLE_PREFLIGHT_VERSION: Final = "teacher-oracle-preflight-v1"
ORACLE_PROFILE: Final = "teacher-oracle"
ORACLE_EXPERIMENT_ID: Final = "EXP-0005"
ORACLE_RESULT_CLASS: Final = "teacher_oracle"
ORACLE_VARIANTS: Final = frozenset({"E0", "T0", "T1"})
ORACLE_WINDOW_US: Final = 30_000_000
ORACLE_AUDIT_VERSION: Final = "sensor-audit-v1"
ORACLE_AUDIT_MAX_BYTES: Final = 1_048_576
ORACLE_CONFIG_MAX_BYTES: Final = 1_048_576
_PLACEHOLDER_PATTERN: Final = re.compile(
    r"\$\{[^}]+\}|<[^>]+>|\b(?:TBD|TODO|REPLACE_ME|CHANGEME)\b",
    re.IGNORECASE,
)
_REQUIRED_ORACLE_CAPABILITIES: Final = (
    "code",
    "environment",
    "data",
    "model",
    "provider",
    "semantic",
    "ontology",
    "signing",
    "accounting",
)
_REQUIRED_SELF_CHECKS: Final = (
    "capability_runner",
    "signer_vectors",
    "resolver",
    "quality",
)
_RESOLVER_MASK: Final = [
    "RESOLVE_IN_ROOT",
    "RESOLVE_NO_SYMLINKS",
    "RESOLVE_NO_MAGICLINKS",
    "RESOLVE_NO_XDEV",
]


class OraclePreflightBlocker(BaseModel):
    code: str
    message: str
    field: str | None = None


class OraclePreflightReport(BaseModel):
    version: Literal["teacher-oracle-preflight-v1"] = ORACLE_PREFLIGHT_VERSION
    status: Literal["pass", "fail"]
    profile: str | None
    experiment_id: str | None
    operational_state: str | None
    scientific_state: str | None
    sensor_audit_digest: str | None
    experiment_config_digest: str
    selected_sensor_inventory_digest: str | None
    blockers: tuple[OraclePreflightBlocker, ...]


class PreflightReport(BaseModel):
    version: Literal["smvqa-preflight-v1"] = PREFLIGHT_VERSION
    status: Literal["pass", "warn", "fail"]
    input_dir: str
    frame_root: str | None
    counts: dict[str, int]
    coverage: dict[str, float | int | None]
    distributions: dict[str, dict[str, int]]
    errors: tuple[PreflightIssue, ...]
    warnings: tuple[PreflightIssue, ...]


def inspect_prepared_dataset(
    input_dir: Path,
    *,
    frame_root: Path | None = None,
) -> PreflightReport:
    errors: list[PreflightIssue] = []
    warnings: list[PreflightIssue] = []
    source_rows = _read_jsonl(
        input_dir / "sources.jsonl",
        SourceStreamExample,
        errors,
    )
    question_rows = _read_jsonl(
        input_dir / "questions.jsonl",
        QuestionRequest,
        errors,
    )
    label_rows = _read_jsonl(
        input_dir / "labels.jsonl",
        QALabelExample,
        errors,
    )

    sources = tuple(model for _, model in source_rows)
    questions = tuple(model for _, model in question_rows)
    labels = tuple(model for _, model in label_rows)
    counts: Counter[str] = Counter(
        sources_valid=len(sources),
        questions_valid=len(questions),
        labels_valid=len(labels),
    )
    timestamps: list[float] = []

    sources_by_video: dict[str, SourceStreamExample] = {}
    for source in sources:
        if source.video_id in sources_by_video:
            _issue(
                errors,
                "duplicate_source_id",
                f"duplicate source video_id: {source.video_id}",
                source.video_id,
            )
            continue
        sources_by_video[source.video_id] = source
        _inspect_source(
            source,
            counts=counts,
            timestamps=timestamps,
            errors=errors,
        )

    resolved_frame_root = _resolved_frame_root(input_dir, frame_root)
    _inspect_frame_files(
        sources,
        resolved_frame_root,
        counts=counts,
        errors=errors,
        warnings=warnings,
    )

    questions_by_id = _unique_questions(questions, errors)
    labels_by_id = _unique_labels(labels, errors)
    _inspect_question_label_sets(questions_by_id, labels_by_id, errors)

    task_distribution: Counter[str] = Counter()
    choice_type_distribution: Counter[str] = Counter()
    for raw, question in question_rows:
        task_distribution[_task_name(raw)] += 1
        choice_type_distribution.update(
            choice.choice_ltype for choice in question.answer_choices
        )
        _inspect_question(
            question,
            sources_by_video,
            timestamps=timestamps,
            errors=errors,
        )

    evidence_store_distribution: Counter[str] = Counter()
    for _, label in label_rows:
        _inspect_label_evidence(
            label,
            sources_by_video,
            evidence_store_distribution,
            timestamps=timestamps,
            errors=errors,
            warnings=warnings,
        )

    _add_coverage_warnings(counts, task_distribution, warnings)
    if {_timestamp_scale(value) for value in timestamps} == {"epoch", "relative"}:
        _issue(
            errors,
            "common_timebase_risk",
            "epoch-scale and relative timestamps are mixed",
        )

    status: Literal["pass", "warn", "fail"] = (
        "fail" if errors else "warn" if warnings else "pass"
    )
    return PreflightReport(
        status=status,
        input_dir=str(input_dir),
        frame_root=None if resolved_frame_root is None else str(resolved_frame_root),
        counts=dict(sorted(counts.items())),
        coverage=_coverage(counts),
        distributions={
            "task": dict(sorted(task_distribution.items())),
            "answer_choice_type": dict(sorted(choice_type_distribution.items())),
            "evidence_store": dict(sorted(evidence_store_distribution.items())),
        },
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def write_preflight_report(
    input_dir: Path,
    output: Path,
    *,
    frame_root: Path | None = None,
) -> PreflightReport:
    report = inspect_prepared_dataset(input_dir, frame_root=frame_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def validate_teacher_oracle_inputs(
    sensor_audit_path: Path,
    experiment_config_path: Path,
) -> OraclePreflightReport:
    """Validate immutable teacher-oracle inputs without inferring missing facts."""
    blockers: list[OraclePreflightBlocker] = []
    audit_bytes = _read_oracle_file(
        sensor_audit_path,
        "sensor_audit",
        ORACLE_AUDIT_MAX_BYTES,
        blockers,
    )
    config_bytes = _read_oracle_file(
        experiment_config_path,
        "experiment_config",
        ORACLE_CONFIG_MAX_BYTES,
        blockers,
    )
    config = _read_oracle_json(
        config_bytes,
        "experiment_config",
        blockers,
    )
    audit = _read_sensor_audit(audit_bytes, blockers)

    profile = _oracle_text(config, "profile")
    experiment_id = _oracle_text(config, "experiment_id")
    _oracle_equal(blockers, "profile_mismatch", profile, ORACLE_PROFILE, "profile")
    _oracle_equal(
        blockers,
        "experiment_identity_mismatch",
        experiment_id,
        ORACLE_EXPERIMENT_ID,
        "experiment_id",
    )
    _oracle_equal(
        blockers,
        "result_class_mismatch",
        _oracle_text(config, "result_class"),
        ORACLE_RESULT_CLASS,
        "result_class",
    )
    _oracle_equal(
        blockers,
        "lane_mismatch",
        _oracle_text(config, "lane"),
        ORACLE_RESULT_CLASS,
        "lane",
    )
    variants = _string_list(config.get("variants"))
    if variants != ["E0", "T0", "T1"]:
        _oracle_block(
            blockers,
            "variant_mismatch",
            "variants must be exactly E0, T0, T1",
            "variants",
        )
    _oracle_equal(
        blockers,
        "window_mismatch",
        config.get("window_us"),
        ORACLE_WINDOW_US,
        "window_us",
    )
    byte_budget = config.get("byte_budget")
    if (
        isinstance(byte_budget, bool)
        or not isinstance(byte_budget, int)
        or byte_budget <= 0
    ):
        _oracle_block(
            blockers,
            "byte_budget_missing",
            "byte_budget must be a positive integer",
            "byte_budget",
        )
    if config.get("scientific_state", "not_measured") != "not_measured":
        _oracle_block(
            blockers,
            "scientific_state_caller_asserted",
            "scientific state is owned by preflight and must not be asserted",
            "scientific_state",
        )
    _find_placeholders(config, "experiment_config", blockers)
    _validate_audit_binding(config, audit, audit_bytes, blockers)
    _validate_oracle_paths(config, blockers)
    _validate_oracle_capabilities(config, blockers)
    _validate_oracle_self_checks(config, blockers)
    _validate_sensor_policy(config, blockers)
    _validate_stage_specs(config, blockers)
    _validate_stage_topology(config, blockers)
    _validate_accounting(config, blockers)
    _validate_label_blind_roots(config, blockers)

    manifest_digest = audit.get("manifest_digest")
    selected_sensor_inventory_digest: str | None = (
        manifest_digest if isinstance(manifest_digest, str) else None
    )
    return OraclePreflightReport(
        status="fail" if blockers else "pass",
        profile=profile,
        experiment_id=experiment_id,
        operational_state=_oracle_text(audit, "operational_state"),
        scientific_state="not_measured",
        sensor_audit_digest=(_sha256_bytes(audit_bytes) if audit_bytes else None),
        experiment_config_digest=_sha256_bytes(config_bytes),
        selected_sensor_inventory_digest=selected_sensor_inventory_digest,
        blockers=tuple(blockers),
    )


def write_teacher_oracle_preflight_report(
    sensor_audit_path: Path,
    experiment_config_path: Path,
    output: Path,
) -> OraclePreflightReport:
    report = validate_teacher_oracle_inputs(
        sensor_audit_path,
        experiment_config_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            _ = stream.write(report.model_dump_json(indent=2) + "\n")
        _ = temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report


def _read_oracle_file(  # noqa: PLR0911
    path: Path,
    field: str,
    maximum_bytes: int,
    blockers: list[OraclePreflightBlocker],
) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int) or nofollow == 0:
        _oracle_block(
            blockers,
            "input_nofollow_unavailable",
            "O_NOFOLLOW is required for oracle inputs",
            field,
        )
        return b""
    symlink = _first_symlink_component(path)
    if symlink is not None:
        _oracle_block(blockers, "input_symlink", str(symlink), field)
        return b""
    try:
        descriptor = _open_oracle_file_nofollow(path, nofollow)
    except OSError as exc:
        code = "input_symlink" if exc.errno == errno.ELOOP else "input_unreadable"
        _oracle_block(blockers, code, f"{path}: {exc}", field)
        return b""
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            _oracle_block(blockers, "input_not_regular_file", str(path), field)
            return b""
        if metadata.st_size > maximum_bytes:
            _oracle_block(
                blockers,
                "input_too_large",
                f"{path}: exceeds {maximum_bytes} byte limit",
                field,
            )
            return b""
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > maximum_bytes:
            _oracle_block(
                blockers,
                "input_too_large",
                f"{path}: exceeds {maximum_bytes} byte limit",
                field,
            )
            return b""
        return payload  # noqa: TRY300
    except OSError as exc:
        _oracle_block(blockers, "input_unreadable", f"{path}: {exc}", field)
        return b""
    finally:
        os.close(descriptor)


def _open_oracle_file_nofollow(path: Path, nofollow: int) -> int:
    absolute_path = Path(os.path.abspath(path))  # noqa: PTH100
    parts = absolute_path.parts
    directory = os.open(
        absolute_path.anchor,
        os.O_RDONLY | os.O_DIRECTORY | nofollow,
    )
    try:
        for part in parts[1:-1]:
            next_directory = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | nofollow,
                dir_fd=directory,
            )
            os.close(directory)
            directory = next_directory
        return os.open(parts[-1], os.O_RDONLY | nofollow, dir_fd=directory)
    finally:
        os.close(directory)


def _read_oracle_json(
    payload: bytes,
    field: str,
    blockers: list[OraclePreflightBlocker],
) -> dict[str, object]:
    try:
        value = cast(
            "object",
            json.loads(
                payload,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
                parse_float=_parse_finite_json_float,
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _oracle_block(blockers, "invalid_json", str(exc), field)
        return {}
    result = _object_dict(value)
    if result is None:
        _oracle_block(
            blockers,
            "invalid_json",
            "must be a JSON object",
            field,
        )
        return {}
    return result


def _read_sensor_audit(
    payload: bytes,
    blockers: list[OraclePreflightBlocker],
) -> dict[str, object]:
    audit = _read_oracle_json(payload, "sensor_audit", blockers)
    try:
        report = SensorAuditReport.model_validate(audit)
    except ValidationError as exc:
        _oracle_block(
            blockers,
            "invalid_sensor_audit",
            str(exc),
            "sensor_audit",
        )
        return {}
    audit = cast("dict[str, object]", report.model_dump(mode="json"))
    if report.operational_state != "ready":
        _oracle_block(
            blockers,
            "sensor_audit_not_ready",
            "sensor audit must be ready",
            "sensor_audit.operational_state",
        )
    if report.provider_gate_decision != "go":
        _oracle_block(
            blockers,
            "provider_gate_not_go",
            "sensor provider gate must be go",
            "sensor_audit.provider_gate_decision",
        )
    if report.issues:
        _oracle_block(
            blockers,
            "sensor_audit_has_issues",
            "sensor audit issues must be empty",
            "sensor_audit.issues",
        )
    counts = report.counts
    selected = counts.get("selected_frames")
    joined = counts.get("joined_observations")
    if (
        not isinstance(selected, int)
        or selected < 1
        or joined != selected
        or counts.get("observations") != selected
        or counts.get("rgb_verified") != selected
        or counts.get("intrinsics_available") != selected
    ):
        _oracle_block(
            blockers,
            "sensor_audit_counts_inconsistent",
            "sensor audit counts must completely cover selected frames",
            "sensor_audit.counts",
        )
    if any(
        value != PERCENT_FULL
        for value in (
            report.coverage.get("rgb_percent"),
            report.coverage.get("intrinsics_percent"),
        )
    ):
        _oracle_block(
            blockers,
            "sensor_audit_coverage_incomplete",
            "sensor audit RGB and intrinsics coverage must be complete",
            "sensor_audit.coverage",
        )
    return audit


def _validate_audit_binding(
    config: Mapping[str, object],
    audit: Mapping[str, object],
    audit_bytes: bytes,
    blockers: list[OraclePreflightBlocker],
) -> None:
    _oracle_equal(
        blockers,
        "sensor_audit_digest_mismatch",
        config.get("sensor_audit_digest"),
        _sha256_bytes(audit_bytes),
        "sensor_audit_digest",
    )
    _oracle_equal(
        blockers,
        "sensor_inventory_digest_mismatch",
        config.get("sensor_inventory_digest"),
        audit.get("manifest_digest"),
        "sensor_inventory_digest",
    )
    policy = _object_mapping(config.get("sensor_policy"))
    roots = _string_list(policy.get("approved_roots")) if policy is not None else None
    approved_root_digests: set[str] = set()
    if roots is not None:
        try:
            approved_root_digests = {
                _directory_identity_digest(Path(root)) for root in roots
            }
        except OSError:
            approved_root_digests = set()
    if (
        not isinstance(audit.get("frame_root_digest"), str)
        or audit.get("frame_root_digest") not in approved_root_digests
    ):
        _oracle_block(
            blockers,
            "sensor_frame_root_mismatch",
            "sensor audit frame root must match one approved sensor root",
            "sensor_audit.frame_root_digest",
        )


def _validate_oracle_capabilities(
    config: Mapping[str, object],
    blockers: list[OraclePreflightBlocker],
) -> None:
    capabilities = _object_mapping(config.get("capabilities"))
    if capabilities is None or set(capabilities) != set(_REQUIRED_ORACLE_CAPABILITIES):
        _oracle_block(
            blockers,
            "capability_contract_mismatch",
            "capabilities must contain exactly the canonical capability contracts",
            "capabilities",
        )
        return
    roots = _canonical_non_symlink_directories(
        _string_list(config.get("allowed_company_roots"))
    )
    if roots is None:
        _oracle_block(
            blockers,
            "capability_root_unavailable",
            "capability artifacts require approved non-symlink roots",
            "allowed_company_roots",
        )
        return
    for name in _REQUIRED_ORACLE_CAPABILITIES:
        capability = _object_mapping(capabilities.get(name))
        required_fields = {"artifact", "sha256"}
        if name == "provider":
            required_fields.add("policy")
        if (
            capability is None
            or set(capability) != required_fields
            or not isinstance(capability.get("artifact"), str)
            or not capability["artifact"]
            or not _is_sha256(capability.get("sha256"))
            or (
                name == "provider"
                and capability.get("policy") not in {"required", "pinned"}
            )
        ):
            _oracle_block(
                blockers,
                "missing_capability_contract",
                f"{name} capability requires a canonical artifact contract",
                f"capabilities.{name}",
            )
            continue
        artifact = Path(cast("str", capability["artifact"]))
        inside_approved_root = any(_within(artifact, root) for root in roots)
        if not artifact.is_absolute() or not inside_approved_root:
            _oracle_block(
                blockers,
                "capability_outside_approved_root",
                str(artifact),
                f"capabilities.{name}.artifact",
            )
            continue
        payload = _read_oracle_file(
            artifact,
            f"capabilities.{name}.artifact",
            ORACLE_CONFIG_MAX_BYTES,
            blockers,
        )
        if _sha256_bytes(payload) != capability["sha256"]:
            _oracle_block(
                blockers,
                "capability_digest_mismatch",
                name,
                f"capabilities.{name}.sha256",
            )


def _validate_oracle_self_checks(
    config: Mapping[str, object],
    blockers: list[OraclePreflightBlocker],
) -> None:
    checks = _object_mapping(config.get("self_checks"))
    if checks is None or set(checks) != set(_REQUIRED_SELF_CHECKS):
        _oracle_block(
            blockers,
            "self_check_contract_mismatch",
            "self_checks must contain every canonical typed self-check receipt",
            "self_checks",
        )
        return
    roots = _canonical_non_symlink_directories(
        _string_list(config.get("allowed_company_roots"))
    )
    if roots is None:
        _oracle_block(
            blockers,
            "self_check_root_unavailable",
            "self-check receipts require approved non-symlink roots",
            "allowed_company_roots",
        )
        return
    registry = _load_self_check_registry(config, blockers)
    if registry is None:
        return
    for name in _REQUIRED_SELF_CHECKS:
        check = _object_mapping(checks.get(name))
        if (
            check is None
            or set(check) != {"artifact", "sha256"}
            or not isinstance(check.get("artifact"), str)
            or not check["artifact"]
            or not _is_sha256(check.get("sha256"))
        ):
            _oracle_block(
                blockers,
                "self_check_contract_mismatch",
                f"{name} self-check requires an artifact and SHA-256",
                f"self_checks.{name}",
            )
            continue
        artifact = Path(cast("str", check["artifact"]))
        if not artifact.is_absolute() or not any(
            _within(artifact, root) for root in roots
        ):
            _oracle_block(
                blockers,
                "self_check_outside_approved_root",
                str(artifact),
                f"self_checks.{name}.artifact",
            )
            continue
        payload = _read_oracle_file(
            artifact,
            f"self_checks.{name}.artifact",
            ORACLE_CONFIG_MAX_BYTES,
            blockers,
        )
        if _sha256_bytes(payload) != check["sha256"]:
            _oracle_block(
                blockers,
                "self_check_digest_mismatch",
                name,
                f"self_checks.{name}.sha256",
            )
            continue
        receipt = _read_oracle_json(
            payload,
            f"self_checks.{name}.artifact",
            blockers,
        )
        _validate_self_check_receipt(name, receipt, config, registry, blockers)


def _load_self_check_registry(
    config: Mapping[str, object],
    blockers: list[OraclePreflightBlocker],
) -> ImmutableAttestationKeyRegistryV1 | None:
    capabilities = _object_mapping(config.get("capabilities"))
    signer_registry = _object_mapping(config.get("signer_registry"))
    signing = _object_mapping(capabilities.get("signing")) if capabilities else None
    if (
        signer_registry is None
        or signing is None
        or signer_registry.get("registry_digest") != signing.get("sha256")
        or not isinstance(signing.get("artifact"), str)
    ):
        _oracle_block(
            blockers,
            "self_check_registry_invalid",
            "self-check signer registry must be the authenticated signing capability",
            "signer_registry",
        )
        return None
    payload = _read_oracle_file(
        Path(cast("str", signing["artifact"])),
        "capabilities.signing.artifact",
        ORACLE_CONFIG_MAX_BYTES,
        blockers,
    )
    registry_value = _read_oracle_json(
        payload, "capabilities.signing.artifact", blockers
    )
    try:
        return ImmutableAttestationKeyRegistryV1.model_validate(registry_value)
    except ValidationError:
        _oracle_block(
            blockers,
            "self_check_registry_invalid",
            "signing capability must contain an immutable Ed25519 key registry",
            "capabilities.signing.artifact",
        )
        return None


def _validate_self_check_receipt(
    name: str,
    receipt: Mapping[str, object],
    config: Mapping[str, object],
    registry: ImmutableAttestationKeyRegistryV1,
    blockers: list[OraclePreflightBlocker],
) -> None:
    purpose = f"oracle-preflight-self-check:{name}"
    try:
        envelope = SignedAttestationEnvelopeV1.model_validate(receipt)
        verify_signed_attestation_envelope(envelope, registry, purpose=purpose)
    except (ValidationError, AttestationError):
        _oracle_block(
            blockers,
            "self_check_receipt_invalid",
            f"{name} receipt must be a valid purpose-bound Ed25519 envelope",
            f"self_checks.{name}.artifact",
        )
        return
    if envelope.payload != expected_self_check_payload(name, config):
        _oracle_block(
            blockers,
            "self_check_receipt_invalid",
            f"{name} receipt must attest the current runner, kernel, and contract",
            f"self_checks.{name}.artifact",
        )


def expected_self_check_payload(
    name: str, config: Mapping[str, object]
) -> dict[str, object]:
    capabilities = _object_mapping(config.get("capabilities")) or {}
    capability_digests: dict[str, object] = {}
    for key, value in capabilities.items():
        capability = _object_mapping(value)
        if capability is not None:
            capability_digests[key] = capability.get("sha256")
    environment = {
        "runner_sha256": capability_digests.get("environment"),
        "kernel_release": os.uname().release,
    }
    if name == "capability_runner":
        outcomes: dict[str, object] = {"capability_sha256": capability_digests}
    elif name == "signer_vectors":
        signer_registry = _object_mapping(config.get("signer_registry")) or {}
        outcomes = {
            "signer_registry_sha256": signer_registry.get("registry_digest"),
            "rfc8785_golden": "verified",
            "ed25519_valid": "verified",
            "ed25519_tampered": "rejected",
        }
    elif name == "resolver":
        resolver = _object_mapping(config.get("resolver")) or {}
        mask = resolver.get("openat2_resolve_mask")
        if mask != _RESOLVER_MASK:
            outcomes = {"resolve_mask": mask}
        else:
            outcomes = {
                "resolve_mask": mask,
                "relative_regular": "opened",
                "symlink": "rejected",
                "magiclink": "rejected",
                "mount_crossing": "rejected",
            }
    else:
        quality = _object_mapping(config.get("quality")) or {}
        outcomes = {
            "utility_rule": quality.get("utility_rule"),
            "confidence_interval_rule": quality.get("confidence_interval_rule"),
            "selective_risk_rule": quality.get("selective_risk_rule"),
        }
    return {
        "kind": name,
        "environment": environment,
        "outcomes": outcomes,
    }


def _validate_oracle_paths(
    config: Mapping[str, object],
    blockers: list[OraclePreflightBlocker],
) -> None:
    roots = _string_list(config.get("allowed_company_roots"))
    if roots is None or not roots:
        _oracle_block(
            blockers,
            "allowed_company_roots_missing",
            "allowed_company_roots is required",
            "allowed_company_roots",
        )
        return
    allowed = tuple(Path(root).resolve() for root in roots)
    paths = _object_mapping(config.get("paths"))
    if paths is None:
        _oracle_block(
            blockers,
            "paths_missing",
            "paths binding is required",
            "paths",
        )
        return
    required_paths = {"code", "env", "data", "model"}
    if set(paths) != required_paths:
        _oracle_block(
            blockers,
            "paths_contract_mismatch",
            "paths must be exactly code, env, data, and model",
            "paths",
        )
    for name, value in paths.items():
        if not isinstance(value, str):
            _oracle_block(
                blockers,
                "invalid_path",
                "path must be a string",
                f"paths.{name}",
            )
            continue
        path = Path(value)
        if not path.is_absolute() or not any(_within(path, root) for root in allowed):
            _oracle_block(
                blockers,
                "path_outside_company_root",
                value,
                f"paths.{name}",
            )
            continue
        _check_no_symlink(path, f"paths.{name}", blockers)
        if not path.exists():
            _oracle_block(blockers, "path_missing", value, f"paths.{name}")


def _validate_stage_specs(
    config: Mapping[str, object],
    blockers: list[OraclePreflightBlocker],
) -> None:
    stages = _object_list(config.get("stage_specs"))
    if not stages:
        _oracle_block(
            blockers,
            "stage_specs_missing",
            "a complete canonical StageSpec graph is required",
            "stage_specs",
        )
        return
    for index, stage in enumerate(stages):
        stage_spec = _object_mapping(stage)
        resources = (
            _object_mapping(stage_spec.get("resources"))
            if stage_spec is not None
            else None
        )
        if (
            stage_spec is None
            or set(stage_spec)
            != {"name", "role", "variant", "dependencies", "retries", "resources"}
            or not isinstance(stage_spec.get("name"), str)
            or not stage_spec["name"]
            or not isinstance(stage_spec.get("role"), str)
            or not _is_nonnegative_int(stage_spec.get("retries"))
            or resources is None
            or set(resources)
            != {
                "partition",
                "nodes",
                "gpus_per_node",
                "cpus",
                "memory",
                "time",
            }
            or not isinstance(resources.get("partition"), str)
            or not resources["partition"]
            or any(
                not _is_positive_int(resources.get(key)) for key in ("nodes", "cpus")
            )
            or not _is_nonnegative_int(resources.get("gpus_per_node"))
            or not isinstance(resources.get("memory"), str)
            or not resources["memory"]
            or not isinstance(resources.get("time"), str)
            or not resources["time"]
        ):
            _oracle_block(
                blockers,
                "stage_resources_incomplete",
                "every StageSpec requires partition, retries, and typed resources",
                f"stage_specs.{index}",
            )


def _validate_stage_topology(
    config: Mapping[str, object],
    blockers: list[OraclePreflightBlocker],
) -> None:
    stages = _object_list(config.get("stage_specs"))
    if stages is None:
        return
    typed = [_object_mapping(stage) for stage in stages]
    if any(stage is None for stage in typed):
        return
    by_name = {
        cast("str", stage["name"]): stage
        for stage in cast("list[Mapping[str, object]]", typed)
        if isinstance(stage.get("name"), str) and stage["name"]
    }
    if len(by_name) != len(stages):
        _oracle_block(
            blockers,
            "stage_name_not_unique",
            "StageSpec names must be unique",
            "stage_specs",
        )
        return
    mode = config.get("t1_location_mode")
    producers = ["geometry", "semantic", "place"]
    if mode == "stable_last_location":
        producers.append("identity")
    elif mode != "frame_bound_place":
        _oracle_block(
            blockers,
            "t1_location_mode_invalid",
            "t1_location_mode must be frame_bound_place or stable_last_location",
            "t1_location_mode",
        )
        return
    producer = _object_mapping(config.get("producer"))
    configured = (
        _string_list(producer.get("configured")) if producer is not None else None
    )
    if configured != producers:
        _oracle_block(
            blockers,
            "producer_configured_mismatch",
            "producer.configured must exactly match the ordered canonical producer set",
            "producer.configured",
        )
        return
    expected: dict[str, tuple[str, str | None, dict[str, object]]] = {
        "preflight": ("preflight", None, {}),
        **{
            role: (
                role,
                None,
                {"kind": "afterok", "stages": ["preflight"]},
            )
            for role in producers
        },
        "gate": ("gate", None, {"kind": "afterany", "stages": producers}),
        "terminal": ("terminal", None, {"kind": "afterany", "stages": ["gate"]}),
    }
    qa_stages: list[str] = []
    for variant in ("E0", "T0", "T1"):
        materialize = f"{variant.lower()}_materialize"
        retrieve = f"{variant.lower()}_retrieve"
        qa = f"{variant.lower()}_qa"
        qa_stages.append(qa)
        expected.update(
            {
                materialize: (
                    "materialize",
                    variant,
                    {"kind": "afterok", "stages": ["gate"]},
                ),
                retrieve: (
                    "retrieve",
                    variant,
                    {"kind": "afterok", "stages": [materialize]},
                ),
                qa: ("qa", variant, {"kind": "afterok", "stages": [retrieve]}),
            }
        )
    expected["evaluator"] = (
        "evaluator",
        None,
        {"kind": "afterok", "stages": qa_stages},
    )
    expected["finalizer"] = (
        "finalizer",
        None,
        {"kind": "afterany", "stages": ["evaluator", "terminal"]},
    )
    if set(by_name) != set(expected):
        _oracle_block(
            blockers,
            "stage_graph_incomplete",
            "stage_specs must be the complete canonical execution graph",
            "stage_specs",
        )
        return
    for name, (role, variant, dependencies) in expected.items():
        stage = by_name[name]
        if (
            stage.get("role") != role
            or stage.get("variant") != variant
            or stage.get("dependencies") != dependencies
        ):
            _oracle_block(
                blockers,
                "stage_graph_mismatch",
                (
                    "StageSpec roles, variants, and dependencies must match "
                    "the canonical graph"
                ),
                f"stage_specs.{name}",
            )
    topology = _object_mapping(config.get("stage_topology"))
    if topology != {
        "producer_to_gate": "afterany",
        "gate_to_terminal": "afterany",
    }:
        _oracle_block(
            blockers,
            "stage_topology_mismatch",
            "producer->gate and gate->terminal must both use afterany",
            "stage_topology",
        )


def _validate_accounting(
    config: Mapping[str, object],
    blockers: list[OraclePreflightBlocker],
) -> None:
    accounting = _object_mapping(config.get("accounting"))
    if accounting is None:
        _oracle_block(
            blockers,
            "accounting_contract_missing",
            "accounting must be declared at the canonical root",
            "accounting",
        )
        return
    command = accounting.get("command")
    fields = _string_list(accounting.get("fields"))
    injection = _object_mapping(accounting.get("job_id_injection"))
    try:
        command_tokens = shlex.split(command) if isinstance(command, str) else []
    except ValueError:
        command_tokens = []
    command_valid = (
        set(accounting)
        == {
            "cluster",
            "command",
            "fields",
            "version",
            "job_id_injection",
            "settle_policy",
        }
        and injection is not None
        and set(injection) == {"flag", "placeholder"}
        and injection.get("flag") == "--jobs"
        and isinstance(injection.get("placeholder"), str)
        and bool(injection["placeholder"])
        and len(command_tokens) == _ACCOUNTING_COMMAND_TOKEN_COUNT
        and command_tokens[0] == "/opt/slurm/bin/sacct"
        and command_tokens[1:5] == ["-D", "-X", "-n", "-P"]
        and command_tokens[5].startswith("--clusters=")
        and command_tokens[5] != "--clusters="
        and command_tokens[6] == f"{injection['flag']}={injection['placeholder']}"
        and command_tokens[7]
        == "--format=JobIDRaw,Cluster,State%64,ExitCode,Restarts,SLUID,OriginalSLUID"
    )
    if not command_valid:
        _oracle_block(
            blockers,
            "accounting_command_incomplete",
            "accounting must use exact duplicate-visible lossless sacct flags",
            "accounting.command",
        )
    if fields != [
        "JobIDRaw",
        "Cluster",
        "State%64",
        "ExitCode",
        "Restarts",
        "SLUID",
        "OriginalSLUID",
    ]:
        _oracle_block(
            blockers,
            "accounting_fields_incomplete",
            "accounting fields must be the exact lossless job identity declaration",
            "accounting.fields",
        )
    version = accounting.get("version")
    if (
        not isinstance(version, str)
        or re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version) is None
        or tuple(int(part) for part in version.split(".")[:2]) < (23, 2)
    ):
        _oracle_block(
            blockers,
            "accounting_version_missing",
            "Slurm accounting version must be at least 23.2",
            "accounting.version",
        )
    settle = _object_mapping(accounting.get("settle_policy"))
    max_wait = settle.get("max_wait_seconds") if settle is not None else None
    interval = settle.get("poll_interval_seconds") if settle is not None else None
    if (
        settle is None
        or set(settle) != {"max_wait_seconds", "poll_interval_seconds"}
        or not _is_positive_int(max_wait)
        or not _is_positive_int(interval)
        or cast("int", interval) > cast("int", max_wait)
        or cast("int", max_wait) > _MAX_ACCOUNTING_SETTLE_SECONDS
    ):
        _oracle_block(
            blockers,
            "accounting_settle_policy_invalid",
            "accounting requires a bounded max_wait_seconds and poll_interval_seconds",
            "accounting.settle_policy",
        )
    producer = _object_mapping(config.get("producer"))
    if producer is None or producer.get("requeue") is not False:
        _oracle_block(
            blockers,
            "producer_requeue_not_disabled",
            "producer must declare requeue false",
            "producer.requeue",
        )


def _validate_sensor_policy(
    config: Mapping[str, object],
    blockers: list[OraclePreflightBlocker],
) -> None:
    policy = _object_mapping(config.get("sensor_policy"))
    modalities = _string_list(policy.get("modalities")) if policy is not None else None
    roots = _string_list(policy.get("approved_roots")) if policy is not None else None
    if (
        policy is None
        or set(policy) != {"modalities", "approved_roots"}
        or modalities != ["rgb", "intrinsics"]
        or roots is None
        or not roots
    ):
        _oracle_block(
            blockers,
            "sensor_policy_invalid",
            "sensor policy must bind rgb/intrinsics to unique approved roots",
            "sensor_policy",
        )
        return

    allowed_roots = _canonical_non_symlink_directories(
        _string_list(config.get("allowed_company_roots"))
    )
    paths = _object_mapping(config.get("paths"))
    data_path = paths.get("data") if paths is not None else None
    data_root = (
        _canonical_non_symlink_directory(Path(data_path))
        if isinstance(data_path, str)
        else None
    )
    if not allowed_roots:
        _oracle_block(
            blockers,
            "sensor_root_company_boundary_invalid",
            "allowed_company_roots must contain canonical non-symlink directories",
            "allowed_company_roots",
        )
    if data_root is None:
        _oracle_block(
            blockers,
            "sensor_root_data_boundary_invalid",
            "paths.data must be a canonical non-symlink directory",
            "paths.data",
        )

    canonical_roots: list[Path] = []
    for root in roots:
        root_path = Path(root)
        canonical_root = _canonical_non_symlink_directory(root_path)
        if canonical_root is None:
            code = (
                "sensor_root_symlink"
                if _first_symlink_component(root_path) is not None
                else "sensor_policy_invalid"
            )
            _oracle_block(
                blockers,
                code,
                "approved sensor roots must be absolute non-symlink directories",
                "sensor_policy.approved_roots",
            )
            continue
        canonical_roots.append(canonical_root)
        if allowed_roots and not any(
            _within(canonical_root, company_root) for company_root in allowed_roots
        ):
            _oracle_block(
                blockers,
                "sensor_root_outside_company_root",
                str(root_path),
                "sensor_policy.approved_roots",
            )
        if data_root is not None and not _within(canonical_root, data_root):
            _oracle_block(
                blockers,
                "sensor_root_outside_data_root",
                str(root_path),
                "sensor_policy.approved_roots",
            )
    if len(canonical_roots) != len(set(canonical_roots)):
        _oracle_block(
            blockers,
            "sensor_policy_invalid",
            "approved sensor roots must be unique canonical paths",
            "sensor_policy.approved_roots",
        )


def _validate_label_blind_roots(
    config: Mapping[str, object],
    blockers: list[OraclePreflightBlocker],
) -> None:
    qa = _object_mapping(config.get("qa"))
    production = _string_list(qa.get("production_roots")) if qa is not None else None
    labels = _string_list(qa.get("label_roots")) if qa is not None else None
    producer = _object_mapping(config.get("producer"))
    producer_inputs = (
        _string_list(producer.get("input_manifest")) if producer is not None else None
    )
    if production is None or not production:
        _oracle_block(
            blockers,
            "qa_production_roots_missing",
            "QA production roots are required",
            "qa.production_roots",
        )
        return
    if labels is None or not labels:
        _oracle_block(
            blockers,
            "qa_label_roots_missing",
            "QA label roots must be nonempty",
            "qa.label_roots",
        )
        return
    if qa is None or qa.get("label_root_owner") != "evaluator":
        _oracle_block(
            blockers,
            "qa_label_root_owner_invalid",
            "label roots must be owned exclusively by evaluator",
            "qa.label_root_owner",
        )
    canonical_production = tuple(
        Path(root).resolve(strict=False) for root in production
    )
    canonical_labels = tuple(Path(root).resolve(strict=False) for root in labels)
    if (
        len(canonical_production) != len(set(canonical_production))
        or len(canonical_labels) != len(set(canonical_labels))
        or any(
            not Path(root).is_absolute()
            or not Path(root).is_dir()
            or Path(root).is_symlink()
            for root in (*production, *labels)
        )
    ):
        _oracle_block(
            blockers,
            "qa_root_alias_invalid",
            "QA roots must be unique absolute non-symlink canonical paths",
            "qa",
        )
    if (
        producer_inputs is None
        or not producer_inputs
        or len(producer_inputs) != len(set(producer_inputs))
        or not set(producer_inputs).issubset(_PRODUCER_INPUT_ALLOWLIST)
    ):
        _oracle_block(
            blockers,
            "producer_input_manifest_invalid",
            "producer inputs must be unique allowlisted sensor/source manifests",
            "producer.input_manifest",
        )
    for production_root in canonical_production:
        for label_root in canonical_labels:
            if _within(production_root, label_root) or _within(
                label_root,
                production_root,
            ):
                _oracle_block(
                    blockers,
                    "qa_label_root_exposure",
                    "production roots must be label-blind",
                    "qa.production_roots",
                )


def _oracle_text(
    config: Mapping[str, object],
    field: str,
) -> str | None:
    value = config.get(field)
    return value if isinstance(value, str) else None


def _oracle_equal(
    blockers: list[OraclePreflightBlocker],
    code: str,
    actual: object,
    expected: object,
    field: str,
) -> None:
    if actual != expected:
        _oracle_block(
            blockers,
            code,
            f"expected {expected!r}, got {actual!r}",
            field,
        )


def _oracle_block(
    blockers: list[OraclePreflightBlocker],
    code: str,
    message: str,
    field: str | None,
) -> None:
    blockers.append(
        OraclePreflightBlocker(code=code, message=message, field=field),
    )


def _object_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    raw = cast("dict[object, object]", value)
    if not all(isinstance(key, str) for key in raw):
        return None
    return cast("dict[str, object]", raw)


def _object_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    raw = cast("Mapping[object, object]", value)
    if not all(isinstance(key, str) for key in raw):
        return None
    return cast("Mapping[str, object]", raw)


def _object_list(value: object) -> list[object] | None:
    if not isinstance(value, list):
        return None
    return cast("list[object]", value)


def _string_list(value: object) -> list[str] | None:
    items = _object_list(value)
    if items is None or not all(isinstance(item, str) for item in items):
        return None
    return cast("list[str]", items)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _find_placeholders(
    value: object,
    field: str,
    blockers: list[OraclePreflightBlocker],
) -> None:
    if isinstance(value, str) and _PLACEHOLDER_PATTERN.search(value):
        _oracle_block(
            blockers,
            "unresolved_placeholder",
            f"placeholder in {field}",
            field,
        )
    elif (mapping := _object_mapping(value)) is not None:
        for key, child in mapping.items():
            _find_placeholders(child, f"{field}.{key}", blockers)
    elif (items := _object_list(value)) is not None:
        for index, child in enumerate(items):
            _find_placeholders(child, f"{field}.{index}", blockers)


def _within(path: Path, root: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    try:
        _ = resolved_path.relative_to(root)
    except ValueError:
        return False
    return True


def _canonical_non_symlink_directories(
    roots: list[str] | None,
) -> tuple[Path, ...] | None:
    if roots is None or not roots:
        return None
    canonical_roots: list[Path] = []
    for root in roots:
        canonical_root = _canonical_non_symlink_directory(Path(root))
        if canonical_root is None:
            return None
        canonical_roots.append(canonical_root)
    return tuple(canonical_roots)


def _canonical_non_symlink_directory(path: Path) -> Path | None:
    if not path.is_absolute() or _first_symlink_component(path) is not None:
        return None
    try:
        canonical_path = path.resolve(strict=True)
    except OSError:
        return None
    return canonical_path if canonical_path.is_dir() else None


def _first_symlink_component(path: Path) -> Path | None:
    absolute_path = Path(os.path.abspath(path))  # noqa: PTH100
    current = Path(absolute_path.anchor)
    for part in absolute_path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            return current
    return None


def _check_no_symlink(
    path: Path, field: str, blockers: list[OraclePreflightBlocker]
) -> None:
    symlink = _first_symlink_component(path)
    if symlink is not None:
        _oracle_block(blockers, "path_symlink", str(symlink), field)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            message = f"duplicate JSON key: {key}"
            raise ValueError(message)
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    message = f"non-finite JSON value: {value}"
    raise ValueError(message)


def _parse_finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        message = f"non-finite JSON value: {value}"
        raise ValueError(message)
    return result


def _directory_identity_digest(path: Path) -> str:
    metadata = path.stat(follow_symlinks=False)
    return _sha256_bytes(f"{metadata.st_dev}:{metadata.st_ino}".encode())


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl[ModelT: BaseModel](
    path: Path,
    model: type[ModelT],
    errors: list[PreflightIssue],
) -> tuple[tuple[dict[str, object], ModelT], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _issue(errors, "file_unreadable", f"{path}: {exc}")
        return ()
    rows: list[tuple[dict[str, object], ModelT]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw_object = _JSON_ADAPTER.validate_json(line)
        except ValidationError as exc:
            _issue(
                errors,
                "invalid_jsonl_row",
                f"{path.name}:{line_number}: {exc}",
            )
            continue
        raw = _object_dict(raw_object)
        if raw is None:
            _issue(
                errors,
                "invalid_jsonl_row",
                f"{path.name}:{line_number}: JSONL row must be an object",
            )
            continue
        try:
            rows.append((raw, model.model_validate(raw)))
        except ValidationError as exc:
            _issue(
                errors,
                "invalid_jsonl_row",
                f"{path.name}:{line_number}: {exc}",
            )
    if not rows:
        _issue(errors, "empty_file", f"{path} has no valid records")
    return tuple(rows)


def _inspect_source(
    source: SourceStreamExample,
    *,
    counts: Counter[str],
    timestamps: list[float],
    errors: list[PreflightIssue],
) -> None:
    timestamps.extend((source.start_time, source.end_time))
    _check_finite(source.start_time, "source.start_time", source.video_id, errors)
    _check_finite(source.end_time, "source.end_time", source.video_id, errors)
    for field_name, spans in (
        ("transcript_spans", source.transcript_spans),
        ("ocr_entries", source.ocr_entries),
        ("object_detections", source.object_detections),
    ):
        _inspect_intervals(source, field_name, spans, timestamps, errors)
    for field_name, samples in (
        ("pose_samples", source.pose_samples),
        ("gaze_samples", source.gaze_samples),
        ("frame_metadata", source.frame_metadata),
    ):
        _inspect_timestamps(
            source,
            field_name,
            tuple(sample.timestamp for sample in samples),
            timestamps,
            errors,
        )

    detections = source.object_detections
    counts["object_detections"] += len(detections)
    counts["object_detections_with_xyz"] += sum(
        detection.x is not None and detection.y is not None and detection.z is not None
        for detection in detections
    )
    counts["object_detections_with_instance_id"] += sum(
        bool(getattr(detection, "instance_id", None)) for detection in detections
    )
    counts["sources_with_pose"] += bool(source.pose_samples)
    counts["sources_with_gaze"] += bool(source.gaze_samples)
    counts["pose_samples"] += len(source.pose_samples)
    counts["gaze_samples"] += len(source.gaze_samples)
    counts["source_frames"] += len(source.frame_metadata)

    metadata_refs = tuple(frame.frame_ref for frame in source.frame_metadata)
    if source.frame_refs and set(source.frame_refs) != set(metadata_refs):
        _issue(
            errors,
            "frame_inventory_mismatch",
            "frame_refs and frame_metadata frame_ref values differ",
            source.video_id,
        )
    metadata_ref_set = set(metadata_refs)
    for entry in source.ocr_entries:
        if entry.frame_ref not in metadata_ref_set:
            _issue(
                errors,
                "ocr_frame_ref_missing",
                f"OCR frame_ref absent from frame_metadata: {entry.frame_ref}",
                source.video_id,
            )
    try:
        manifest = build_sensor_frame_manifest((source,))[0]
    except SensorFrameManifestError as exc:
        _issue(errors, "invalid_sensor_frames", exc.detail, source.video_id)
    else:
        counts["selected_1hz_frames"] += len(manifest.selected_frames)


def _inspect_intervals(
    source: SourceStreamExample,
    field_name: str,
    spans: Sequence[LocalTimedModel],
    timestamps: list[float],
    errors: list[PreflightIssue],
) -> None:
    previous_start: float | None = None
    for index, span in enumerate(spans):
        timestamps.extend((span.start_time, span.end_time))
        if not math.isfinite(span.start_time) or not math.isfinite(span.end_time):
            _issue(
                errors,
                "nonfinite_nested_interval",
                f"{field_name}[{index}] has non-finite time",
                source.video_id,
            )
        if (
            span.start_time < source.start_time - TIMESTAMP_EPSILON
            or span.end_time > source.end_time + TIMESTAMP_EPSILON
        ):
            _issue(
                errors,
                "nested_interval_out_of_bounds",
                f"{field_name}[{index}] is outside source interval",
                source.video_id,
            )
        if previous_start is not None and span.start_time < previous_start:
            _issue(
                errors,
                "nested_interval_out_of_order",
                f"{field_name} must be sorted by start_time",
                source.video_id,
            )
            break
        previous_start = span.start_time


def _inspect_timestamps(
    source: SourceStreamExample,
    field_name: str,
    values: Sequence[float],
    timestamps: list[float],
    errors: list[PreflightIssue],
) -> None:
    timestamps.extend(values)
    for index, value in enumerate(values):
        if not math.isfinite(value):
            _issue(
                errors,
                "nonfinite_nested_timestamp",
                f"{field_name}[{index}] has non-finite timestamp",
                source.video_id,
            )
        if (
            value < source.start_time - TIMESTAMP_EPSILON
            or value > source.end_time + TIMESTAMP_EPSILON
        ):
            _issue(
                errors,
                "nested_timestamp_out_of_bounds",
                f"{field_name}[{index}] is outside source interval",
                source.video_id,
            )
    if tuple(values) != tuple(sorted(values)):
        _issue(
            errors,
            "nested_timestamp_out_of_order",
            f"{field_name} must be sorted by timestamp",
            source.video_id,
        )


def _inspect_frame_files(
    sources: Sequence[SourceStreamExample],
    frame_root: Path | None,
    *,
    counts: Counter[str],
    errors: list[PreflightIssue],
    warnings: list[PreflightIssue],
) -> None:
    if counts["source_frames"] and frame_root is None:
        _issue(
            warnings,
            "frame_files_unchecked",
            "no frame root found; frame file existence was not checked",
        )
        return
    if frame_root is None:
        return
    missing: list[str] = []
    for source in sources:
        for frame in source.frame_metadata:
            counts["frame_files_checked"] += 1
            if _frame_exists(frame_root, source.video_id, frame.frame_ref):
                counts["frame_files_found"] += 1
            else:
                missing.append(f"{source.video_id}/{frame.frame_ref}")
    if missing:
        preview = ", ".join(missing[:PREVIEW_LIMIT])
        suffix = (
            ""
            if len(missing) <= PREVIEW_LIMIT
            else f" (+{len(missing) - PREVIEW_LIMIT} more)"
        )
        _issue(
            errors,
            "frame_file_missing",
            f"missing {len(missing)} frame file(s): {preview}{suffix}",
        )


def _inspect_question(
    question: QuestionRequest,
    sources_by_video: dict[str, SourceStreamExample],
    *,
    timestamps: list[float],
    errors: list[PreflightIssue],
) -> None:
    timestamps.append(question.question_time)
    _check_finite(
        question.question_time,
        "question_time",
        question.question_id,
        errors,
    )
    choice_ids = tuple(choice.choice_id for choice in question.answer_choices)
    if len(choice_ids) != ANSWER_CHOICE_COUNT or len(set(choice_ids)) != len(
        choice_ids,
    ):
        _issue(
            errors,
            "invalid_answer_choices",
            "question requires exactly four unique answer choices",
            question.question_id,
        )
    unanswerable_count = sum(
        is_unanswerable_choice(choice) for choice in question.answer_choices
    )
    if unanswerable_count != 1:
        _issue(
            errors,
            "invalid_unanswerable_choice",
            "question requires exactly one unanswerable choice",
            question.question_id,
        )
    scope = question.video_ids or (question.video_id,)
    if question.video_id not in scope:
        _issue(
            errors,
            "primary_video_out_of_scope",
            "primary video_id is absent from video_ids",
            question.question_id,
        )
    missing = sorted(set(scope) - sources_by_video.keys())
    if missing:
        _issue(
            errors,
            "question_unknown_video",
            f"unknown scoped video_id(s): {', '.join(missing)}",
            question.question_id,
        )
        return
    scoped_sources = tuple(sources_by_video[video_id] for video_id in scope)
    if not any(
        source.start_time - TIMESTAMP_EPSILON
        <= question.question_time
        <= source.end_time + TIMESTAMP_EPSILON
        for source in scoped_sources
    ):
        ranges = ", ".join(
            f"{source.video_id}:[{source.start_time}, {source.end_time}]"
            for source in scoped_sources
        )
        _issue(
            errors,
            "question_time_out_of_bounds",
            f"question_time {question.question_time} is outside every scoped "
            f"source interval: {ranges}",
            question.question_id,
        )


def _inspect_label_evidence(  # noqa: PLR0913
    label: QALabelExample,
    sources_by_video: dict[str, SourceStreamExample],
    stores: Counter[str],
    *,
    timestamps: list[float],
    errors: list[PreflightIssue],
    warnings: list[PreflightIssue],
) -> None:
    choice_ids = tuple(choice.choice_id for choice in label.answer_choices)
    unanswerable_ids = tuple(
        choice.choice_id
        for choice in label.answer_choices
        if is_unanswerable_choice(choice)
    )
    if label.answer not in choice_ids:
        _issue(
            errors,
            "invalid_gold_choice",
            "label answer is not one of its choice IDs",
            label.question_id,
        )
    elif len(unanswerable_ids) == 1 and label.is_answerable == (
        label.answer == unanswerable_ids[0]
    ):
        _issue(
            errors,
            "label_answerability_mismatch",
            "label answerability disagrees with its gold choice",
            label.question_id,
        )
    scope = set(label.video_ids or (label.video_id,))
    for raw_span in label.evidence_list:
        parsed = _parse_evidence_span(raw_span)
        if parsed is None:
            _issue(
                errors,
                "invalid_evidence_grammar",
                f"expected video:start:end:store: {raw_span}",
                label.question_id,
            )
            continue
        video_id, start, end, store = parsed
        stores[store] += 1
        timestamps.extend((start, end))
        if not STORE_PATTERN.fullmatch(store):
            _issue(
                errors,
                "invalid_evidence_store",
                f"invalid evidence store name: {store}",
                label.question_id,
            )
        elif store not in STORES:
            _issue(
                warnings,
                "unsupported_evidence_store",
                f"evidence store is not supported by diagnostics: {store}",
                label.question_id,
            )
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            _issue(
                errors,
                "invalid_evidence_interval",
                f"evidence must have finite start < end: {raw_span}",
                label.question_id,
            )
            continue
        if video_id not in scope:
            _issue(
                errors,
                "evidence_video_out_of_scope",
                f"evidence video_id is outside question scope: {video_id}",
                label.question_id,
            )
        source = sources_by_video.get(video_id)
        if source is None:
            _issue(
                errors,
                "evidence_unknown_video",
                f"evidence references unknown video_id: {video_id}",
                label.question_id,
            )
        elif (
            start < source.start_time - TIMESTAMP_EPSILON
            or end > source.end_time + TIMESTAMP_EPSILON
        ):
            _issue(
                errors,
                "evidence_out_of_bounds",
                f"evidence interval is outside source bounds: {raw_span}",
                label.question_id,
            )
        if end > label.question_time + TIMESTAMP_EPSILON:
            _issue(
                errors,
                "future_evidence",
                f"evidence ends after question_time: {raw_span}",
                label.question_id,
            )


def _unique_questions(
    questions: Sequence[QuestionRequest],
    errors: list[PreflightIssue],
) -> dict[str, QuestionRequest]:
    result: dict[str, QuestionRequest] = {}
    for question in questions:
        if question.question_id in result:
            _issue(
                errors,
                "duplicate_question_id",
                f"duplicate question_id: {question.question_id}",
                question.question_id,
            )
        else:
            result[question.question_id] = question
    return result


def _unique_labels(
    labels: Sequence[QALabelExample],
    errors: list[PreflightIssue],
) -> dict[str, QALabelExample]:
    result: dict[str, QALabelExample] = {}
    for label in labels:
        if label.question_id in result:
            _issue(
                errors,
                "duplicate_label_id",
                f"duplicate label question_id: {label.question_id}",
                label.question_id,
            )
        else:
            result[label.question_id] = label
    return result


def _inspect_question_label_sets(
    questions: dict[str, QuestionRequest],
    labels: dict[str, QALabelExample],
    errors: list[PreflightIssue],
) -> None:
    missing = sorted(labels.keys() - questions.keys())
    extra = sorted(questions.keys() - labels.keys())
    if missing or extra:
        _issue(
            errors,
            "question_label_id_mismatch",
            f"question/label IDs differ; missing={missing} extra={extra}",
        )
    for question_id in sorted(questions.keys() & labels.keys()):
        question = questions[question_id]
        label = labels[question_id]
        mismatched_fields = tuple(
            field_name
            for field_name in (
                "video_id",
                "video_ids",
                "question",
                "question_time",
                "answer_choices",
                "task",
                "skill",
            )
            if getattr(question, field_name) != getattr(label, field_name)
        )
        if mismatched_fields:
            _issue(
                errors,
                "question_label_semantic_mismatch",
                "question/label fields differ: " + ", ".join(mismatched_fields),
                question_id,
            )


def _add_coverage_warnings(
    counts: Counter[str],
    task_distribution: Counter[str],
    warnings: list[PreflightIssue],
) -> None:
    detections = counts["object_detections"]
    if detections and counts["object_detections_with_xyz"] < detections:
        _issue(
            warnings,
            "object_xyz_coverage",
            "some object detections lack XYZ geometry",
        )
    if detections and counts["object_detections_with_instance_id"] < detections:
        _issue(
            warnings,
            "object_instance_coverage",
            "some object detections lack instance_id",
        )
    sources = counts["sources_valid"]
    if sources and counts["sources_with_pose"] < sources:
        _issue(warnings, "pose_coverage", "some sources have no pose samples")
    if sources and counts["sources_with_gaze"] < sources:
        _issue(warnings, "gaze_coverage", "some sources have no gaze samples")
    if task_distribution.get("unspecified"):
        _issue(
            warnings,
            "task_metadata_missing",
            "some questions have no task/category metadata",
        )


def _coverage(counts: Counter[str]) -> dict[str, float | int | None]:
    return {
        "object_xyz_percent": _percent(
            counts["object_detections_with_xyz"],
            counts["object_detections"],
        ),
        "object_instance_id_percent": _percent(
            counts["object_detections_with_instance_id"],
            counts["object_detections"],
        ),
        "source_pose_percent": _percent(
            counts["sources_with_pose"],
            counts["sources_valid"],
        ),
        "source_gaze_percent": _percent(
            counts["sources_with_gaze"],
            counts["sources_valid"],
        ),
        "selected_1hz_percent": _percent(
            counts["selected_1hz_frames"],
            counts["source_frames"],
        ),
        "frame_file_percent": _percent(
            counts["frame_files_found"],
            counts["frame_files_checked"],
        ),
    }


def _task_name(raw: dict[str, object]) -> str:
    typed_metadata = _object_mapping(raw.get("metadata"))
    metadata: Mapping[str, object] = typed_metadata or {}
    candidates: tuple[object, ...] = (
        raw.get("task"),
        raw.get("task_type"),
        raw.get("question_type"),
        raw.get("skill"),
        raw.get("category"),
        metadata.get("task"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "unspecified"


def _parse_evidence_span(raw_span: str) -> tuple[str, float, float, str] | None:
    parts = raw_span.rsplit(":", 3)
    if len(parts) != EVIDENCE_SPAN_PARTS:
        return None
    video_id, raw_start, raw_end, store = parts
    if not video_id or not store:
        return None
    try:
        return video_id, float(raw_start), float(raw_end), store
    except ValueError:
        return None


def _resolved_frame_root(input_dir: Path, frame_root: Path | None) -> Path | None:
    if frame_root is not None:
        return frame_root
    local = input_dir / "frames"
    return local if local.is_dir() else None


def _frame_exists(frame_root: Path, video_id: str, frame_ref: str) -> bool:
    reference = Path(frame_ref)
    video = Path(video_id)
    if (
        reference.is_absolute()
        or video.is_absolute()
        or ".." in reference.parts
        or ".." in video.parts
    ):
        return False
    for base in (frame_root / video / reference, frame_root / reference):
        if not _legacy_frame_path_safe(frame_root, base):
            continue
        candidates = (base, *(base.with_suffix(suffix) for suffix in FRAME_EXTENSIONS))
        for candidate in candidates:
            try:
                metadata = candidate.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISREG(metadata.st_mode):
                return True
    return False


def _legacy_frame_path_safe(frame_root: Path, candidate: Path) -> bool:
    try:
        relative = candidate.relative_to(frame_root)
    except ValueError:
        return False
    current = frame_root
    for part in relative.parts:
        current /= part
        try:
            if stat.S_ISLNK(current.stat(follow_symlinks=False).st_mode):
                return False
        except OSError:
            return True
    return True


def _check_finite(
    value: float,
    field_name: str,
    record_id: str,
    errors: list[PreflightIssue],
) -> None:
    if not math.isfinite(value):
        _issue(
            errors,
            "nonfinite_timestamp",
            f"{field_name} must be finite",
            record_id,
        )


def _timestamp_scale(value: float) -> str:
    return "epoch" if abs(value) >= EPOCH_SCALE else "relative"


def _percent(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return round(100.0 * numerator / denominator, 2)


def _issue(
    destination: list[PreflightIssue],
    code: str,
    message: str,
    record_id: str | None = None,
) -> None:
    destination.append(
        PreflightIssue(code=code, message=message, record_id=record_id),
    )
