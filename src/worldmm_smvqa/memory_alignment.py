# ruff: noqa: EM101, TRY301, TRY003
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Literal, Self, cast

from pydantic import Field, TypeAdapter, model_validator

from worldmm_smvqa.attestation import canonicalize
from worldmm_smvqa.schema import FrozenModel
from worldmm_smvqa.worldmm.episodic_types import EpisodicEdgeRecord, EpisodicNodeRecord
from worldmm_smvqa.worldmm.semantic import SemanticTripleRecord
from worldmm_smvqa.worldmm.visual import VisualMemoryRecord

SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
CANONICAL_UINT_RE: Final = re.compile(r"0|[1-9][0-9]*")
MAX_EVALUATOR_MICROSECONDS: Final = 9_223_372_036_854_775_807
BOOTSTRAP_REPLICATES: Final = 10_000
COVERAGE_BASIS_POINTS_FULL: Final = 10_000
BOOTSTRAP_CI_LOW_INDEX: Final = 249
BOOTSTRAP_CI_HIGH_INDEX: Final = 9_749
NON_INFERIORITY_THRESHOLD: Final = Fraction(-5, 100)
PROTOCOL_ID: Final = "memory-recall6-paired-bootstrap-v1"
SUITE_ID: Final = "memory-alignment-four-store-suite-v1"
STOP_WORDS: Final = frozenset({"a", "and", "is", "on", "the", "what", "which", "with"})

ValidationCode = Literal[
    "contract_invalid",
    "baseline_bundle_invalid",
    "candidate_bundle_invalid",
    "cross_bundle_mismatch",
    "coverage_invalid",
    "cohort_invalid",
    "native_projection_invalid",
    "episodic_reference_invalid",
    "semantic_support_invalid",
    "label_invalid",
    "request_invalid",
    "comparison_role_invalid",
    "zero_pair_count",
    "bootstrap_bound_exceeded",
]
StoreKind = Literal["visual", "episodic", "semantic", "semantic_rebuild"]
ProjectionRole = Literal["baseline", "candidate", "candidate_semantic_rebuild"]
ComparisonId = Literal[
    "visual_primary", "episodic_primary", "semantic_primary", "semantic_rebuild"
]


@dataclass(frozen=True, slots=True)
class AlignmentValidationError(Exception):
    code: ValidationCode
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


class ContractSelectionV1(FrozenModel):
    schema_version: Literal["contract-selection-v1"]
    version: Literal["v1", "v2"]
    contract_id: str
    contract_path: str
    expected_contract_file_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def _validate_semantics(self) -> Self:
        require_nfc_nonempty(self.contract_id, "contract_id")
        validate_relative_path(self.contract_path)
        return self


class SealedArtifactReferenceV1(FrozenModel):
    schema_version: Literal["sealed-artifact-reference-v1"]
    artifact_kind: Literal[
        "store", "coverage", "model_config", "source_manifest", "retrieval_config"
    ]
    path: str
    file_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def _validate_path(self) -> Self:
        validate_relative_path(self.path)
        return self


class BundleStoreV1(FrozenModel):
    store_kind: StoreKind
    path: str
    file_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    record_count: str
    record_contract_id: str | None = None

    @model_validator(mode="after")
    def _validate_values(self) -> Self:
        validate_relative_path(self.path)
        parse_canonical_count(self.record_count)
        return self


class BundleCoverageReferenceV1(FrozenModel):
    store_kind: StoreKind
    path: str
    file_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def _validate_path(self) -> Self:
        validate_relative_path(self.path)
        return self


class SealedMemoryBundleV1(FrozenModel):
    schema_version: Literal["sealed-memory-bundle-v1"]
    bundle_id: str
    role: Literal["baseline", "candidate"]
    model_artifact_role: Literal["qwen_memory_baseline", "gemma_memory_candidate"]
    model_artifact_id: str
    contract_selection: ContractSelectionV1
    model_config_path: str
    model_config_file_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_manifest_path: str
    source_manifest_file_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    cohort_id: str
    cohort_file_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    retrieval_config_path: str
    retrieval_config_file_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    k: Literal["6"]
    comparison_suite_id: Literal["memory-alignment-four-store-suite-v1"]
    stores: tuple[BundleStoreV1, ...]
    coverage: tuple[BundleCoverageReferenceV1, ...]
    sealed_artifacts: tuple[SealedArtifactReferenceV1, ...]
    seal_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def _validate_bundle_shape(self) -> Self:
        for value, field in (
            (self.bundle_id, "bundle_id"),
            (self.model_artifact_id, "model_artifact_id"),
            (self.cohort_id, "cohort_id"),
        ):
            require_nfc_nonempty(value, field)
        for path in (
            self.model_config_path,
            self.source_manifest_path,
            self.retrieval_config_path,
        ):
            validate_relative_path(path)
        expected = (
            (
                "baseline",
                "qwen_memory_baseline",
                "v1",
                ("visual", "episodic", "semantic"),
            ),
            (
                "candidate",
                "gemma_memory_candidate",
                "v2",
                ("visual", "episodic", "semantic", "semantic_rebuild"),
            ),
        )
        actual = (
            self.role,
            self.model_artifact_role,
            self.contract_selection.version,
            tuple(item.store_kind for item in self.stores),
        )
        if actual not in expected:
            msg = "bundle role/model/contract/store order is invalid"
            raise ValueError(msg)
        if tuple(item.store_kind for item in self.coverage) != actual[3]:
            msg = "coverage order must match store order"
            raise ValueError(msg)
        if self.role == "baseline" and any(
            item.record_contract_id is not None for item in self.stores
        ):
            msg = "baseline stores must not declare v2 record contracts"
            raise ValueError(msg)
        if self.role == "candidate":
            expected_contracts = {
                "visual": "memory-visual-record-contract-v2",
                "episodic": "memory-episodic-record-contract-v2",
                "semantic": "memory-semantic-record-contract-v2",
                "semantic_rebuild": "memory-semantic-record-contract-v2",
            }
            if any(
                item.record_contract_id != expected_contracts[item.store_kind]
                for item in self.stores
            ):
                msg = "candidate store record-contract mapping is invalid"
                raise ValueError(msg)
        paths = tuple(item.path for item in self.sealed_artifacts)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            msg = "sealed artifact paths must be unique and sorted"
            raise ValueError(msg)
        return self


class MemoryCoverageManifestV1(FrozenModel):
    schema_version: Literal["memory-coverage-manifest-v1"]
    store_kind: StoreKind
    store_path: str
    source_manifest_file_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    store_file_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    expected_count: str
    attempted_count: str
    schema_valid_count: str
    written_count: str
    coverage_basis_points: str
    producer_assertion: Literal["trusted_external_seal"]

    @model_validator(mode="after")
    def _validate_values(self) -> Self:
        validate_relative_path(self.store_path)
        for value in (
            self.expected_count,
            self.attempted_count,
            self.schema_valid_count,
            self.written_count,
            self.coverage_basis_points,
        ):
            parse_canonical_count(value)
        if int(self.coverage_basis_points) > COVERAGE_BASIS_POINTS_FULL:
            msg = "coverage_basis_points must not exceed 10000"
            raise ValueError(msg)
        return self


class MemoryProjectionV1(FrozenModel):
    projection_id: str
    projection_kind: Literal["visual_point_projection", "interval_projection"]
    role: ProjectionRole
    store_kind: StoreKind
    source_id: str
    native_id: str
    video_id: str
    snippet: str
    base_score_ppm: str
    frame_ref: str | None = None
    timestamp_us: str | None = None
    start_us: str | None = None
    end_us: str | None = None

    @model_validator(mode="after")
    def _validate_projection(self) -> Self:
        for value, field in (
            (self.projection_id, "projection_id"),
            (self.source_id, "source_id"),
            (self.native_id, "native_id"),
            (self.video_id, "video_id"),
        ):
            require_nfc_nonempty(value, field)
        parse_canonical_count(self.base_score_ppm)
        point = self.projection_kind == "visual_point_projection"
        if point:
            if (
                self.store_kind != "visual"
                or self.frame_ref is None
                or self.timestamp_us is None
            ):
                msg = "visual projection requires frame_ref and timestamp_us"
                raise ValueError(msg)
            require_nfc_nonempty(self.frame_ref, "frame_ref")
            parse_evaluator_microseconds(self.timestamp_us)
            if self.start_us is not None or self.end_us is not None:
                msg = "visual projection cannot carry an interval"
                raise ValueError(msg)
        else:
            if self.store_kind not in {"episodic", "semantic", "semantic_rebuild"}:
                msg = "interval projection has an invalid store"
                raise ValueError(msg)
            if self.start_us is None or self.end_us is None:
                msg = "interval projection requires start_us and end_us"
                raise ValueError(msg)
            if parse_evaluator_microseconds(
                self.start_us
            ) >= parse_evaluator_microseconds(self.end_us):
                msg = "projection interval must be forward"
                raise ValueError(msg)
            if self.frame_ref is not None or self.timestamp_us is not None:
                msg = "interval projection cannot carry point fields"
                raise ValueError(msg)
        return self


class MemoryRequestV2(FrozenModel):
    request_id: str
    store: Literal["visual", "episodic", "semantic"]
    video_id: str
    query_text: str
    query_time_us: str
    expected_evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_request(self) -> Self:
        for value, field in (
            (self.request_id, "request_id"),
            (self.video_id, "video_id"),
            (self.query_text, "query_text"),
        ):
            require_nfc_nonempty(value, field)
        parse_evaluator_microseconds(self.query_time_us)
        if len(set(self.expected_evidence_ids)) != len(self.expected_evidence_ids):
            msg = "expected evidence IDs must be unique"
            raise ValueError(msg)
        for value in self.expected_evidence_ids:
            require_nfc_nonempty(value, "expected_evidence_id")
        return self


class VisualPointLabelV1(FrozenModel):
    label_kind: Literal["visual_point_label"]
    evidence_id: str
    store: Literal["visual"]
    video_id: str
    frame_ref: str
    timestamp_us: str


class IntervalLabelV1(FrozenModel):
    label_kind: Literal["interval_label"]
    evidence_id: str
    store: Literal["episodic", "semantic"]
    video_id: str
    start_us: str
    end_us: str

    @model_validator(mode="after")
    def _validate_interval(self) -> Self:
        if parse_evaluator_microseconds(self.start_us) >= parse_evaluator_microseconds(
            self.end_us
        ):
            msg = "label interval must be forward"
            raise ValueError(msg)
        return self


MemoryLabelV1 = VisualPointLabelV1 | IntervalLabelV1
MemoryLabelDiscriminatedV1 = Annotated[MemoryLabelV1, Field(discriminator="label_kind")]
_LABEL_ADAPTER: Final[TypeAdapter[MemoryLabelDiscriminatedV1]] = TypeAdapter(
    MemoryLabelDiscriminatedV1
)


class MemoryComparisonCohortV1(FrozenModel):
    schema_version: Literal["memory-comparison-cohort-v1"]
    cohort_id: str
    cohort_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    baseline_contract: ContractSelectionV1
    candidate_contract: ContractSelectionV1
    source_manifest_file_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    question_manifest_file_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    retrieval_config_file_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    k: Literal["6"]
    comparison_suite_id: Literal["memory-alignment-four-store-suite-v1"]
    requests: tuple[MemoryRequestV2, ...]
    labels: tuple[MemoryLabelDiscriminatedV1, ...]
    projections: tuple[MemoryProjectionV1, ...]

    @model_validator(mode="after")
    def _validate_cohort(self) -> Self:
        require_nfc_nonempty(self.cohort_id, "cohort_id")
        request_ids = tuple(item.request_id for item in self.requests)
        label_ids = tuple(item.evidence_id for item in self.labels)
        projection_ids = tuple(item.projection_id for item in self.projections)
        for values, name in (
            (request_ids, "request"),
            (label_ids, "label"),
            (projection_ids, "projection"),
        ):
            if len(values) != len(set(values)):
                msg = f"{name} IDs must be unique"
                raise ValueError(msg)
        if request_ids != tuple(sorted(request_ids)):
            msg = "requests must be ordered by request ID"
            raise ValueError(msg)
        return self


class CoverageReportRowV1(FrozenModel):
    bundle_role: Literal["baseline", "candidate"]
    store_kind: StoreKind
    expected_count: str
    attempted_count: str
    schema_valid_count: str
    written_count: str
    coverage_basis_points: str
    provenance: Literal["trusted_manifest_not_replayed"] = (
        "trusted_manifest_not_replayed"
    )
    decision: Literal["pass", "fail"]
    scientific_failure: Literal["coverage_below_100_percent"] | None


class ComparisonReportRowV1(FrozenModel):
    comparison_id: ComparisonId
    pair_count: str
    baseline_mean_recall_at_6: str
    candidate_mean_recall_at_6: str
    mean_delta: str
    ci95_low_delta: str
    ci95_high_delta: str
    non_inferiority_threshold: Literal["-1/20"] = "-1/20"
    decision: Literal["pass", "fail"]
    scientific_failure: Literal["ci_lower_below_threshold"] | None
    seed_identifier: str
    accepted_random_block_count: str
    rejected_random_block_count: str


class MemoryAlignmentReportV1(FrozenModel):
    schema_version: Literal["memory-alignment-report-v1"] = "memory-alignment-report-v1"
    status: Literal["pass", "scientific_fail", "validation_fail"]
    protocol_id: Literal["memory-recall6-paired-bootstrap-v1"] = PROTOCOL_ID
    comparison_suite_id: Literal["memory-alignment-four-store-suite-v1"] = SUITE_ID
    baseline_bundle_sha256: str | None
    candidate_bundle_sha256: str | None
    cohort_sha256: str | None
    coverage_provenance: Literal["trusted_manifest_not_replayed"] = (
        "trusted_manifest_not_replayed"
    )
    coverage: tuple[CoverageReportRowV1, ...]
    comparisons: tuple[ComparisonReportRowV1, ...]
    validation_error: ValidationCode | None


@dataclass(frozen=True, slots=True)
class ValidatedBundle:
    root: Path
    manifest: SealedMemoryBundleV1
    rows: Mapping[StoreKind, tuple[object, ...]]
    coverage: Mapping[StoreKind, MemoryCoverageManifestV1]
    bundle_sha256: str


def require_nfc_nonempty(value: str, field: str) -> None:
    if not value or unicodedata.normalize("NFC", value) != value:
        msg = f"{field} must be nonempty NFC text"
        raise ValueError(msg)


def parse_canonical_count(value: object) -> int:
    if not isinstance(value, str) or CANONICAL_UINT_RE.fullmatch(value) is None:
        msg = "count must be a canonical nonnegative decimal string"
        raise ValueError(msg)
    return int(value)


def parse_evaluator_microseconds(value: object) -> int:
    parsed = parse_canonical_count(value)
    if parsed > MAX_EVALUATOR_MICROSECONDS:
        msg = "evaluator microseconds exceed signed int64 range"
        raise ValueError(msg)
    return parsed


def validate_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        msg = "path must be a normalized relative POSIX path"
        raise ValueError(msg)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        msg = "path must be a normalized relative POSIX path"
        raise ValueError(msg)
    return path


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_regular_file(root: Path, relative: str) -> Path:
    validate_relative_path(relative)
    current = root.resolve(strict=True)
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            msg = f"links are forbidden: {relative}"
            raise ValueError(msg)
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        msg = f"path escapes root: {relative}"
        raise ValueError(msg) from exc
    if not resolved.is_file():
        msg = f"not a regular file: {relative}"
        raise ValueError(msg)
    return resolved


def load_contract_selection(
    repository_root: Path,
    relative_path: str,
    expected_sha256: str,
    *,
    version: Literal["v1", "v2"],
) -> ContractSelectionV1:
    if SHA256_RE.fullmatch(expected_sha256) is None:
        msg = "expected contract SHA-256 must be lowercase hexadecimal"
        raise ValueError(msg)
    path = resolve_regular_file(repository_root, relative_path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        msg = "contract file digest mismatch"
        raise ValueError(msg)
    try:
        payload = cast("dict[str, object]", json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = "contract file is not valid JSON"
        raise ValueError(msg) from exc
    expected_schema = f"model-boundary-contract-{version}"
    expected_id = (
        "worldmm-smvqa-local-boundaries-v1"
        if version == "v1"
        else "worldmm-smvqa-memory-v2"
    )
    if (
        payload.get("schema_version") != expected_schema
        or payload.get("contract_id") != expected_id
    ):
        msg = "contract schema or ID mismatch"
        raise ValueError(msg)
    return ContractSelectionV1(
        schema_version="contract-selection-v1",
        version=version,
        contract_id=expected_id,
        contract_path=relative_path,
        expected_contract_file_sha256=expected_sha256,
    )


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"invalid JSON: {path}"
        raise ValueError(msg) from exc
    if not isinstance(value, dict):
        msg = f"expected JSON object: {path}"
        raise TypeError(msg)
    return cast("dict[str, object]", value)


def _seal_digest(manifest: Mapping[str, object]) -> str:
    payload = dict(manifest)
    payload.pop("seal_sha256", None)
    return hashlib.sha256(canonicalize(payload)).hexdigest()


def _strict_rows(  # noqa: PLR0912
    path: Path, store: StoreKind
) -> tuple[object, ...]:
    raw = path.read_bytes()
    if not raw:
        return ()
    physical = raw.splitlines(keepends=True)
    rows: list[object] = []
    for number, line in enumerate(physical, start=1):
        if not line.endswith(b"\n") and number != len(physical):
            msg = f"invalid JSONL framing at line {number}"
            raise ValueError(msg)
        content = line[:-1] if line.endswith(b"\n") else line
        if not content:
            msg = f"blank JSONL row at line {number}"
            raise ValueError(msg)
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            msg = f"invalid JSONL row at line {number}"
            raise ValueError(msg) from exc
        if not isinstance(value, dict):
            msg = f"JSONL row {number} is not an object"
            raise TypeError(msg)
        if store == "visual":
            if value.get("record_type") != "visual" or value.get("store") != "visual":
                msg = f"visual row {number} lacks physical discriminators"
                raise ValueError(msg)
            row = VisualMemoryRecord.model_validate(value)
        elif store == "episodic":
            if value.get("record_type") == "node":
                if "summary" not in value:
                    msg = f"episodic node {number} lacks physical summary"
                    raise ValueError(msg)
                row = EpisodicNodeRecord.model_validate(value)
            elif value.get("record_type") == "edge":
                row = EpisodicEdgeRecord.model_validate(value)
            else:
                msg = f"episodic row {number} has invalid discriminator"
                raise ValueError(msg)
        else:
            if (
                value.get("record_type") != "semantic_triple"
                or value.get("store") != "semantic"
            ):
                msg = f"semantic row {number} lacks physical discriminators"
                raise ValueError(msg)
            row = SemanticTripleRecord.model_validate(value)
        rows.append(row)
    return tuple(rows)


def validate_sealed_bundle(  # noqa: PLR0912, PLR0915
    bundle_root: Path,
    expected_contract: ContractSelectionV1,
    *,
    expected_role: Literal["baseline", "candidate"],
) -> ValidatedBundle:
    code: ValidationCode = (
        "baseline_bundle_invalid"
        if expected_role == "baseline"
        else "candidate_bundle_invalid"
    )
    try:
        root = bundle_root.resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            msg = "bundle root must be a real directory"
            raise ValueError(msg)
        manifest_path = resolve_regular_file(root, "bundle.json")
        raw_manifest = _read_json_object(manifest_path)
        manifest = SealedMemoryBundleV1.model_validate(raw_manifest)
        if manifest.role != expected_role:
            msg = "bundle role mismatch"
            raise ValueError(msg)
        if manifest.contract_selection != expected_contract:
            raise AlignmentValidationError(
                "contract_invalid", "bundle contract declaration mismatch"
            )
        if _seal_digest(raw_manifest) != manifest.seal_sha256:
            msg = "bundle seal digest mismatch"
            raise ValueError(msg)
        actual_files: set[str] = set()
        for candidate in root.rglob("*"):
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink() or not (
                candidate.is_dir() or candidate.is_file()
            ):
                msg = f"bundle contains link or special file: {relative}"
                raise ValueError(msg)
            if candidate.is_file() and relative != "bundle.json":
                actual_files.add(relative)
        referenced = {item.path for item in manifest.sealed_artifacts}
        if referenced != actual_files:
            msg = "sealed artifacts do not enumerate exact bundle files"
            raise ValueError(msg)
        artifacts = {item.path: item for item in manifest.sealed_artifacts}
        for relative, reference in artifacts.items():
            if _sha256(resolve_regular_file(root, relative)) != reference.file_sha256:
                msg = f"sealed artifact digest mismatch: {relative}"
                raise ValueError(msg)
        public_refs = {
            manifest.model_config_path: (
                "model_config",
                manifest.model_config_file_sha256,
            ),
            manifest.source_manifest_path: (
                "source_manifest",
                manifest.source_manifest_file_sha256,
            ),
            manifest.retrieval_config_path: (
                "retrieval_config",
                manifest.retrieval_config_file_sha256,
            ),
            **{item.path: ("store", item.file_sha256) for item in manifest.stores},
            **{item.path: ("coverage", item.file_sha256) for item in manifest.coverage},
        }
        if set(public_refs) != referenced:
            msg = "public references do not match sealed artifacts"
            raise ValueError(msg)
        for relative, (kind, digest) in public_refs.items():
            reference = artifacts[relative]
            if reference.artifact_kind != kind or reference.file_sha256 != digest:
                msg = f"sealed artifact kind/reference mismatch: {relative}"
                raise ValueError(msg)
        rows: dict[StoreKind, tuple[object, ...]] = {}
        coverage: dict[StoreKind, MemoryCoverageManifestV1] = {}
        coverage_refs = {item.store_kind: item for item in manifest.coverage}
        for store_entry in manifest.stores:
            store_rows = _strict_rows(
                resolve_regular_file(root, store_entry.path), store_entry.store_kind
            )
            if parse_canonical_count(store_entry.record_count) != len(store_rows):
                msg = f"{store_entry.store_kind} record_count mismatch"
                raise ValueError(msg)
            coverage_ref = coverage_refs[store_entry.store_kind]
            item = MemoryCoverageManifestV1.model_validate(
                _read_json_object(resolve_regular_file(root, coverage_ref.path))
            )
            if (
                item.store_kind != store_entry.store_kind
                or item.store_path != store_entry.path
                or item.store_file_sha256 != store_entry.file_sha256
                or item.source_manifest_file_sha256
                != manifest.source_manifest_file_sha256
            ):
                raise AlignmentValidationError(
                    "coverage_invalid", "coverage references mismatch"
                )
            if parse_canonical_count(item.written_count) != len(store_rows):
                raise AlignmentValidationError(
                    "coverage_invalid", "written_count mismatch"
                )
            rows[store_entry.store_kind] = store_rows
            coverage[store_entry.store_kind] = item
        return ValidatedBundle(
            root=root,
            manifest=manifest,
            rows=rows,
            coverage=coverage,
            bundle_sha256=_sha256(manifest_path),
        )
    except AlignmentValidationError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise AlignmentValidationError(code, str(exc)) from exc


def tokenize(text: str) -> frozenset[str]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in text)
    return frozenset(token for token in normalized.split() if token not in STOP_WORDS)


def _endpoint(projection: MemoryProjectionV1) -> int:
    value = (
        projection.timestamp_us
        if projection.timestamp_us is not None
        else projection.end_us
    )
    if value is None:
        raise AssertionError("validated projection has no endpoint")
    return parse_evaluator_microseconds(value)


def rank_at_6(
    request: MemoryRequestV2,
    projections: Iterable[MemoryProjectionV1],
    *,
    role: ProjectionRole,
    store_kind: StoreKind,
) -> tuple[MemoryProjectionV1, ...]:
    query = tokenize(request.query_text)
    query_count = max(1, len(query))
    query_time = parse_evaluator_microseconds(request.query_time_us)
    eligible = tuple(
        item
        for item in projections
        if item.role == role
        and item.store_kind == store_kind
        and item.video_id == request.video_id
        and _endpoint(item) <= query_time
    )

    def key(item: MemoryProjectionV1) -> tuple[Fraction, int, str]:
        overlap = len(query & tokenize(item.snippet))
        score = Fraction(overlap, query_count) + Fraction(
            parse_canonical_count(item.base_score_ppm), 100_000_000
        )
        return (-score, -_endpoint(item), item.native_id)

    return tuple(sorted(eligible, key=key)[:6])


def recall_at_6(
    request: MemoryRequestV2,
    labels: Mapping[str, MemoryLabelV1],
    ranked: Sequence[MemoryProjectionV1],
    *,
    logical_store: Literal["visual", "episodic", "semantic"],
) -> Fraction:
    expected = tuple(request.expected_evidence_ids)
    if not expected:
        raise AlignmentValidationError("request_invalid", "request labels are empty")
    hits = 0
    for evidence_id in expected:
        label = labels.get(evidence_id)
        if label is None:
            raise AlignmentValidationError(
                "label_invalid", "request references missing label"
            )
        if label.store != logical_store or label.video_id != request.video_id:
            raise AlignmentValidationError(
                "label_invalid", "label request/store mismatch"
            )
        if any(_projection_matches_label(item, label) for item in ranked):
            hits += 1
    return Fraction(hits, len(expected))


def _projection_matches_label(
    projection: MemoryProjectionV1, label: MemoryLabelV1
) -> bool:
    if projection.video_id != label.video_id:
        return False
    if isinstance(label, VisualPointLabelV1):
        return (
            projection.projection_kind == "visual_point_projection"
            and projection.frame_ref == label.frame_ref
            and projection.timestamp_us == label.timestamp_us
        )
    if projection.projection_kind != "interval_projection":
        return False
    if projection.start_us is None or projection.end_us is None:
        return False
    return parse_evaluator_microseconds(
        projection.start_us
    ) < parse_evaluator_microseconds(label.end_us) and parse_evaluator_microseconds(
        label.start_us
    ) < parse_evaluator_microseconds(projection.end_us)


def _bootstrap_indexes(seed: bytes, replicate: int, count: int) -> tuple[int, ...]:
    indexes: list[int] = []
    counter = 0
    while len(indexes) < count:
        block = hashlib.sha256(
            seed + replicate.to_bytes(8, "big") + counter.to_bytes(8, "big")
        ).digest()
        counter += 1
        for offset in range(0, len(block), 8):
            indexes.append(int.from_bytes(block[offset : offset + 8], "big") % count)
            if len(indexes) == count:
                break
    return tuple(indexes)


def paired_bootstrap_comparison(
    comparison_id: ComparisonId,
    pairs: Sequence[tuple[str, Fraction, Fraction]],
    *,
    seed_identifier: str = "memory-recall6-bootstrap-seed-v1",
) -> ComparisonReportRowV1:
    if not pairs:
        raise AlignmentValidationError("zero_pair_count", "comparison has no pairs")
    ordered = tuple(sorted(pairs, key=lambda item: item[0]))
    if len({item[0] for item in ordered}) != len(ordered):
        raise AlignmentValidationError("request_invalid", "duplicate request IDs")
    count = len(ordered)
    baseline_mean = sum((item[1] for item in ordered), Fraction()) / count
    candidate_mean = sum((item[2] for item in ordered), Fraction()) / count
    deltas = tuple(item[2] - item[1] for item in ordered)
    delta_mean = sum(deltas, Fraction()) / count
    seed = hashlib.sha256(
        (seed_identifier + "\x00" + comparison_id).encode("utf-8")
    ).digest()
    replicates = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        indexes = _bootstrap_indexes(seed, replicate, count)
        replicates.append(sum((deltas[index] for index in indexes), Fraction()) / count)
    replicates.sort()
    low = replicates[BOOTSTRAP_CI_LOW_INDEX]
    high = replicates[BOOTSTRAP_CI_HIGH_INDEX]
    decision = "pass" if low >= NON_INFERIORITY_THRESHOLD else "fail"
    return ComparisonReportRowV1(
        comparison_id=comparison_id,
        pair_count=str(count),
        baseline_mean_recall_at_6=fraction_text(baseline_mean),
        candidate_mean_recall_at_6=fraction_text(candidate_mean),
        mean_delta=fraction_text(delta_mean),
        ci95_low_delta=fraction_text(low),
        ci95_high_delta=fraction_text(high),
        decision=decision,
        scientific_failure=None if decision == "pass" else "ci_lower_below_threshold",
        seed_identifier=seed_identifier,
        accepted_random_block_count=str(BOOTSTRAP_REPLICATES * count),
        rejected_random_block_count="0",
    )


def coverage_report_row(
    role: Literal["baseline", "candidate"], manifest: MemoryCoverageManifestV1
) -> CoverageReportRowV1:
    values = (
        parse_canonical_count(manifest.expected_count),
        parse_canonical_count(manifest.attempted_count),
        parse_canonical_count(manifest.schema_valid_count),
        parse_canonical_count(manifest.written_count),
    )
    passed = (
        values[0] > 0
        and len(set(values)) == 1
        and manifest.coverage_basis_points == "10000"
    )
    return CoverageReportRowV1(
        bundle_role=role,
        store_kind=manifest.store_kind,
        expected_count=manifest.expected_count,
        attempted_count=manifest.attempted_count,
        schema_valid_count=manifest.schema_valid_count,
        written_count=manifest.written_count,
        coverage_basis_points=manifest.coverage_basis_points,
        decision="pass" if passed else "fail",
        scientific_failure=None if passed else "coverage_below_100_percent",
    )


def _cohort_semantic_digest(payload: Mapping[str, object]) -> str:
    semantic = dict(payload)
    semantic.pop("cohort_sha256", None)
    return hashlib.sha256(canonicalize(semantic)).hexdigest()


def _validation_report(
    code: ValidationCode,
    *,
    baseline_digest: str | None,
    candidate_digest: str | None,
    cohort_digest: str | None,
    coverage: tuple[CoverageReportRowV1, ...] = (),
) -> MemoryAlignmentReportV1:
    return MemoryAlignmentReportV1(
        status="validation_fail",
        baseline_bundle_sha256=baseline_digest,
        candidate_bundle_sha256=candidate_digest,
        cohort_sha256=cohort_digest,
        coverage=coverage,
        comparisons=(),
        validation_error=code,
    )


def _validate_root_bindings(  # noqa: PLR0913
    baseline_root: SealedMemoryBundleV1,
    candidate_root: SealedMemoryBundleV1,
    cohort: MemoryComparisonCohortV1,
    *,
    baseline_contract: ContractSelectionV1,
    candidate_contract: ContractSelectionV1,
    cohort_file_sha256: str,
) -> None:
    if (
        baseline_root.contract_selection != baseline_contract
        or candidate_root.contract_selection != candidate_contract
        or cohort.baseline_contract != baseline_contract
        or cohort.candidate_contract != candidate_contract
    ):
        raise AlignmentValidationError(
            "contract_invalid", "root contract bindings do not match preflight bytes"
        )
    if (
        baseline_root.cohort_id != cohort.cohort_id
        or candidate_root.cohort_id != cohort.cohort_id
        or baseline_root.cohort_file_sha256 != cohort_file_sha256
        or candidate_root.cohort_file_sha256 != cohort_file_sha256
        or baseline_root.source_manifest_file_sha256
        != candidate_root.source_manifest_file_sha256
        or baseline_root.source_manifest_file_sha256
        != cohort.source_manifest_file_sha256
        or baseline_root.retrieval_config_file_sha256
        != candidate_root.retrieval_config_file_sha256
        or baseline_root.retrieval_config_file_sha256
        != cohort.retrieval_config_file_sha256
    ):
        raise AlignmentValidationError(
            "cross_bundle_mismatch", "bundle/cohort identities do not reconcile"
        )
    if (
        baseline_root.model_artifact_id == candidate_root.model_artifact_id
        or baseline_root.model_config_file_sha256
        == candidate_root.model_config_file_sha256
    ):
        raise AlignmentValidationError(
            "cross_bundle_mismatch",
            "baseline and candidate model identities must differ",
        )


def _require_label_causal(request: MemoryRequestV2, label: MemoryLabelV1) -> None:
    query_time = parse_evaluator_microseconds(request.query_time_us)
    endpoint = (
        parse_evaluator_microseconds(label.timestamp_us)
        if isinstance(label, VisualPointLabelV1)
        else parse_evaluator_microseconds(label.end_us)
    )
    if endpoint > query_time:
        raise AlignmentValidationError(
            "label_invalid", "label is not causal for request"
        )


def evaluate_comparison_suite(
    cohort: MemoryComparisonCohortV1,
) -> tuple[ComparisonReportRowV1, ...]:
    labels = {item.evidence_id: item for item in cohort.labels}
    projections = cohort.projections
    mappings: tuple[
        tuple[
            ComparisonId,
            Literal["visual", "episodic", "semantic"],
            StoreKind,
            ProjectionRole,
        ],
        ...,
    ] = (
        ("visual_primary", "visual", "visual", "candidate"),
        ("episodic_primary", "episodic", "episodic", "candidate"),
        ("semantic_primary", "semantic", "semantic", "candidate"),
        (
            "semantic_rebuild",
            "semantic",
            "semantic_rebuild",
            "candidate_semantic_rebuild",
        ),
    )
    rows: list[ComparisonReportRowV1] = []
    for comparison_id, request_store, candidate_store, candidate_role in mappings:
        requests = tuple(
            item for item in cohort.requests if item.store == request_store
        )
        if not requests:
            raise AlignmentValidationError(
                "zero_pair_count", f"{comparison_id} has no requests"
            )
        baseline_store: StoreKind = request_store
        pairs: list[tuple[str, Fraction, Fraction]] = []
        for request in requests:
            for evidence_id in request.expected_evidence_ids:
                label = labels.get(evidence_id)
                if label is None:
                    raise AlignmentValidationError(
                        "label_invalid", "request references an absent label"
                    )
                _require_label_causal(request, label)
            baseline_ranked = rank_at_6(
                request,
                projections,
                role="baseline",
                store_kind=baseline_store,
            )
            candidate_ranked = rank_at_6(
                request,
                projections,
                role=candidate_role,
                store_kind=candidate_store,
            )
            if not baseline_ranked or not candidate_ranked:
                raise AlignmentValidationError(
                    "comparison_role_invalid",
                    f"{comparison_id} has an empty eligible operand",
                )
            baseline_recall = recall_at_6(
                request, labels, baseline_ranked, logical_store=request_store
            )
            candidate_recall = recall_at_6(
                request, labels, candidate_ranked, logical_store=request_store
            )
            pairs.append((request.request_id, baseline_recall, candidate_recall))
        rows.append(paired_bootstrap_comparison(comparison_id, pairs))
    return tuple(rows)


def build_alignment_report(
    baseline: ValidatedBundle,
    candidate: ValidatedBundle,
    cohort: MemoryComparisonCohortV1,
    *,
    cohort_file_sha256: str,
) -> MemoryAlignmentReportV1:
    coverage = tuple(
        coverage_report_row("baseline", baseline.coverage[store])
        for store in ("visual", "episodic", "semantic")
    ) + tuple(
        coverage_report_row("candidate", candidate.coverage[store])
        for store in ("visual", "episodic", "semantic", "semantic_rebuild")
    )
    comparisons = evaluate_comparison_suite(cohort)
    scientific_failure = any(item.decision == "fail" for item in coverage) or any(
        item.decision == "fail" for item in comparisons
    )
    return MemoryAlignmentReportV1(
        status="scientific_fail" if scientific_failure else "pass",
        baseline_bundle_sha256=baseline.bundle_sha256,
        candidate_bundle_sha256=candidate.bundle_sha256,
        cohort_sha256=cohort_file_sha256,
        coverage=coverage,
        comparisons=comparisons,
        validation_error=None,
    )


def evaluate_memory_alignment(  # noqa: PLR0913
    *,
    repository_root: Path,
    baseline_contract_path: str,
    baseline_contract_sha256: str,
    candidate_contract_path: str,
    candidate_contract_sha256: str,
    baseline_bundle: Path,
    candidate_bundle: Path,
    cohort_path: Path,
) -> MemoryAlignmentReportV1:
    baseline_contract = load_contract_selection(
        repository_root,
        baseline_contract_path,
        baseline_contract_sha256,
        version="v1",
    )
    candidate_contract = load_contract_selection(
        repository_root,
        candidate_contract_path,
        candidate_contract_sha256,
        version="v2",
    )
    if baseline_contract.contract_path == candidate_contract.contract_path:
        msg = "baseline and candidate contract paths must be distinct"
        raise ValueError(msg)

    baseline_manifest_path = resolve_regular_file(baseline_bundle, "bundle.json")
    candidate_manifest_path = resolve_regular_file(candidate_bundle, "bundle.json")
    cohort_file = cohort_path.resolve(strict=True)
    if cohort_file.is_symlink() or not cohort_file.is_file():
        msg = "cohort must be a regular file without links"
        raise ValueError(msg)
    baseline_digest = _sha256(baseline_manifest_path)
    candidate_digest = _sha256(candidate_manifest_path)
    cohort_digest = _sha256(cohort_file)
    try:
        baseline_root = SealedMemoryBundleV1.model_validate(
            _read_json_object(baseline_manifest_path)
        )
        candidate_root = SealedMemoryBundleV1.model_validate(
            _read_json_object(candidate_manifest_path)
        )
        cohort_payload = _read_json_object(cohort_file)
        cohort = MemoryComparisonCohortV1.model_validate(cohort_payload)
        if _cohort_semantic_digest(cohort_payload) != cohort.cohort_sha256:
            raise AlignmentValidationError(
                "cohort_invalid", "cohort semantic digest mismatch"
            )
        _validate_root_bindings(
            baseline_root,
            candidate_root,
            cohort,
            baseline_contract=baseline_contract,
            candidate_contract=candidate_contract,
            cohort_file_sha256=cohort_digest,
        )
        baseline = validate_sealed_bundle(
            baseline_bundle, baseline_contract, expected_role="baseline"
        )
        candidate = validate_sealed_bundle(
            candidate_bundle, candidate_contract, expected_role="candidate"
        )
        return build_alignment_report(
            baseline, candidate, cohort, cohort_file_sha256=cohort_digest
        )
    except AlignmentValidationError as exc:
        return _validation_report(
            exc.code,
            baseline_digest=baseline_digest,
            candidate_digest=candidate_digest,
            cohort_digest=cohort_digest,
        )
    except (TypeError, ValueError) as exc:
        _ = exc
        return _validation_report(
            "cohort_invalid",
            baseline_digest=baseline_digest,
            candidate_digest=candidate_digest,
            cohort_digest=cohort_digest,
        )


def atomic_write_report_no_clobber(path: Path, report: MemoryAlignmentReportV1) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    payload = canonicalize(report.model_dump(mode="json")) + b"\n"
    temporary = parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
