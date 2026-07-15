from __future__ import annotations

import hashlib
import json
import os
import sys
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol, cast, override

from pydantic import Field, FiniteFloat, TypeAdapter, ValidationError

from worldmm_smvqa.attestation import (
    ImmutableAttestationKeyRegistryV1,
    SignedAttestationEnvelopeV1,
)
from worldmm_smvqa.schema import FrozenModel
from worldmm_smvqa.spatial_train import RECORD_TYPES, TeacherCacheRow
from worldmm_smvqa.teacher_oracle import (
    ApprovedCompiledAssignmentsArtifactV1,
    CompiledOracleAssignmentsV1,
    CrossVariantFairnessContractV1,
    DigestRefV1,
    MaterializeVariantMemoryContext,
    SealedManifestPayloadResolver,
    SealedPayloadResolver,
    TeacherOracleContractError,
    TeacherOracleInputManifestV1,
    TeacherOracleOutputManifestV1,
    build_output_manifest,
    canonical_sha256,
    load_approved_compiled_assignments,
    materialize_variant_memory,
    validate_label_blind_payload,
    validate_producer_input_manifest,
    validate_request_manifest_fairness_trio,
    validate_variant_memory_candidates,
)
from worldmm_smvqa.worldmm.gcut3r_teacher import (
    CacheProductionContractV1,
    TeacherCacheRecord,
    TeacherContractError,
    read_teacher_cache,
    validate_cache_production_attestation,
)
from worldmm_smvqa.worldmm.typed_memory import (
    ObjectMemoryRecord,
    ObjectPresenceMemoryRecord,
    ScoredMemoryCandidate,
    SourceCompactMemoryRecord,
    TypedMemoryRecordBase,
    serialized_byte_cost,
)

type FiniteVector = Annotated[tuple[FiniteFloat, ...], Field(min_length=1)]
type NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class _TypedRecordWithType(Protocol):
    record_type: Literal[
        "object",
        "plane",
        "portal",
        "free_space",
        "landmark",
        "event",
        "no_write",
    ]


@dataclass(frozen=True, slots=True)
class TeacherMaterializationError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"TeacherMaterializationError: {self.detail}"


class TeacherSupervisionRow(FrozenModel):
    observation_id: str = Field(min_length=1)
    memory_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    split: Literal["train", "validation"]
    features: FiniteVector
    teacher_embedding: FiniteVector
    geometry_target: FiniteVector
    association_target: NonNegativeInt


@dataclass(frozen=True, slots=True)
class OracleTeacherCacheMaterializationRequest:
    teacher_cache: Path
    supervision: Path
    output: Path
    production_attestation: SignedAttestationEnvelopeV1
    authorized_registry: ImmutableAttestationKeyRegistryV1
    approved_contract: CacheProductionContractV1
    oracle_manifest: TeacherOracleInputManifestV1
    approved_assignments: ApprovedCompiledAssignmentsArtifactV1
    sealed_payload_resolver: SealedPayloadResolver
    fairness_contract: CrossVariantFairnessContractV1
    fairness_registry: ImmutableAttestationKeyRegistryV1
    cross_variant_manifests: tuple[TeacherOracleInputManifestV1, ...]
    identity_registry: ImmutableAttestationKeyRegistryV1 | None = None


@dataclass(frozen=True, slots=True)
class LabelBlindOracleMemoryMaterializationRequest:
    manifest: TeacherOracleInputManifestV1
    candidates: Sequence[ScoredMemoryCandidate]
    output: Path
    producer_input: Mapping[str, object]
    approved_assignments: ApprovedCompiledAssignmentsArtifactV1
    sealed_payload_resolver: SealedPayloadResolver
    fairness_contract: CrossVariantFairnessContractV1
    fairness_registry: ImmutableAttestationKeyRegistryV1
    cross_variant_manifests: tuple[TeacherOracleInputManifestV1, ...]
    identity_registry: ImmutableAttestationKeyRegistryV1 | None = None


def materialize_teacher_rows(
    teacher_cache: Path,
    supervision: Path,
    *,
    cache_snapshot: Sequence[TeacherCacheRecord] | None = None,
) -> tuple[TeacherCacheRow, ...]:
    """Join supervision to a digest-validated causal teacher cache."""
    cache = (
        tuple(cache_snapshot)
        if cache_snapshot is not None
        else read_teacher_cache(teacher_cache)
    )
    labels = _read_supervision(supervision)

    cache_records: dict[tuple[str, str], TypedMemoryRecordBase] = {}
    ordered_keys: list[tuple[str, str]] = []
    for cache_row in cache:
        observation_id = cache_row.request.observation_id
        for record in cache_row.response.records:
            key = (observation_id, record.memory_id)
            if key in cache_records:
                raise TeacherMaterializationError(
                    detail=f"duplicate teacher record: {_format_key(key)}",
                )
            if isinstance(record, SourceCompactMemoryRecord):
                prefix = f"{_format_key(key)}: source_compact records"
                raise TeacherMaterializationError(
                    detail=f"{prefix} cannot materialize teacher supervision",
                )
            if isinstance(record, ObjectPresenceMemoryRecord):
                raise TeacherMaterializationError(
                    detail=(
                        f"{_format_key(key)}: object_presence_v1 records cannot "
                        "materialize teacher supervision"
                    ),
                )
            cache_records[key] = record
            ordered_keys.append(key)
    if not cache_records:
        raise TeacherMaterializationError(detail="teacher cache has no typed records")

    supervision_by_key = {(row.observation_id, row.memory_id): row for row in labels}
    unknown = supervision_by_key.keys() - cache_records.keys()
    if unknown:
        unknown_key = _format_key(min(unknown))
        raise TeacherMaterializationError(
            detail=f"supervision references missing teacher record: {unknown_key}",
        )
    missing = cache_records.keys() - supervision_by_key.keys()
    if missing:
        missing_key = _format_key(min(missing))
        raise TeacherMaterializationError(
            detail=f"missing supervision for teacher record: {missing_key}",
        )

    rows: list[TeacherCacheRow] = []
    sample_ids: set[str] = set()
    for key in ordered_keys:
        label = supervision_by_key[key]
        record = cache_records[key]
        record_type = cast(
            "_TypedRecordWithType",
            cast("object", record),
        ).record_type
        type_index = RECORD_TYPES.index(record_type)
        uncertainty = float(record.geometry_uncertainty.standard_deviation_m)
        if uncertainty <= 0.0:
            raise TeacherMaterializationError(
                detail=f"{_format_key(key)}: uncertainty_target must be positive",
            )
        byte_cost = serialized_byte_cost(record)
        if (record_type == "no_write") != (byte_cost == 0):
            raise TeacherMaterializationError(
                detail=f"{_format_key(key)}: only no_write may have byte_cost 0",
            )
        sample_id = f"{key[0]}:{key[1]}"
        if sample_id in sample_ids:
            raise TeacherMaterializationError(
                detail=f"sample_id collision: {sample_id}",
            )
        sample_ids.add(sample_id)
        rows.append(
            TeacherCacheRow(
                sample_id=sample_id,
                group_id=label.group_id,
                split=label.split,
                features=tuple(float(value) for value in label.features),
                teacher_embedding=tuple(
                    float(value) for value in label.teacher_embedding
                ),
                type_index=type_index,
                geometry_target=tuple(float(value) for value in label.geometry_target),
                association_target=label.association_target,
                uncertainty_target=uncertainty,
                byte_cost=float(byte_cost),
            ),
        )
    return tuple(rows)


def write_teacher_rows(path: Path, rows: Sequence[TeacherCacheRow]) -> None:
    if not rows:
        raise TeacherMaterializationError(detail="no materialized rows")
    payload = "".join(f"{_encode_row(row)}\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _ = temporary.write_text(payload, encoding="utf-8")
        _ = temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_teacher_cache(
    teacher_cache: Path,
    supervision: Path,
    out: Path,
) -> tuple[TeacherCacheRow, ...]:
    rows = materialize_teacher_rows(teacher_cache, supervision)
    write_teacher_rows(out, rows)
    return rows


def materialize_oracle_teacher_cache(
    request: OracleTeacherCacheMaterializationRequest,
) -> tuple[TeacherCacheRow, ...]:
    resolver = request.sealed_payload_resolver
    if not isinstance(resolver, SealedManifestPayloadResolver):
        raise TeacherMaterializationError(
            detail=(
                "strict oracle cache materialization requires a concrete sealed "
                "manifest resolver"
            )
        )
    try:
        validate_request_manifest_fairness_trio(
            request.oracle_manifest,
            request.fairness_contract,
            request.cross_variant_manifests,
            authorized_registry=request.fairness_registry,
        )
        assignments = load_approved_compiled_assignments(
            request.oracle_manifest,
            request.approved_assignments,
            resolver,
        )
    except TeacherOracleContractError as exc:
        raise TeacherMaterializationError(
            detail=f"strict oracle cache materialization rejected authority: {exc}"
        ) from exc
    cache = read_teacher_cache(request.teacher_cache)
    claims = validate_cache_production_attestation(
        cache,
        request.production_attestation,
        authorized_registry=request.authorized_registry,
        approved_contract=request.approved_contract,
    )
    _validate_production_assignment_binding(
        claims.config_sha256,
        request.approved_assignments,
        assignments,
    )
    _validate_assignment_bound_cache(
        cache,
        request.oracle_manifest,
        assignments,
        resolver,
        request.identity_registry,
    )
    rows = materialize_teacher_rows(
        request.teacher_cache,
        request.supervision,
        cache_snapshot=cache,
    )
    write_teacher_rows(request.output, rows)
    return rows


def _validate_production_assignment_binding(
    config_sha256: str,
    artifact: ApprovedCompiledAssignmentsArtifactV1,
    assignments: CompiledOracleAssignmentsV1,
) -> None:
    expected = canonical_sha256(
        {
            "approved_assignment_artifact": artifact.model_dump(mode="json"),
            "assignment_sha256": assignments.assignment_sha256,
        }
    )
    if config_sha256 != expected:
        raise TeacherMaterializationError(
            detail=(
                "strict oracle cache attestation config does not bind the approved "
                "compiled-assignment artifact and digest"
            )
        )


def _validate_assignment_bound_cache(
    cache: Sequence[TeacherCacheRecord],
    manifest: TeacherOracleInputManifestV1,
    assignments: CompiledOracleAssignmentsV1,
    resolver: SealedManifestPayloadResolver,
    identity_registry: ImmutableAttestationKeyRegistryV1 | None,
) -> None:
    candidates: list[ScoredMemoryCandidate] = []
    for cache_row in cache:
        for record in cache_row.response.records:
            score = 0.0
            if isinstance(record, ObjectMemoryRecord):
                matching = [
                    semantic
                    for semantic in assignments.semantics
                    if semantic.object_id == record.entity_id
                    and semantic.observation_id == record.oracle_observation_id
                    and semantic.source_video_id == record.source_video_id
                    and semantic.semantic_class == record.semantic_label
                ]
                if len(matching) == 1:
                    score = matching[0].semantic_confidence / serialized_byte_cost(
                        record
                    )
            candidates.append(ScoredMemoryCandidate(record=record, score=score))
    try:
        validate_variant_memory_candidates(
            manifest,
            assignments,
            candidates,
            context=MaterializeVariantMemoryContext(
                sealed_payload_resolver=resolver,
                identity_registry=identity_registry,
            ),
        )
    except TeacherOracleContractError as exc:
        raise TeacherMaterializationError(
            detail=f"strict oracle cache rows do not match compiled assignments: {exc}"
        ) from exc


def materialize_label_blind_oracle_memory(
    request: LabelBlindOracleMemoryMaterializationRequest,
) -> TeacherOracleOutputManifestV1:
    """Atomically materialize verified memory and its scientific completion manifest."""
    resolver = request.sealed_payload_resolver
    if not isinstance(resolver, SealedManifestPayloadResolver):
        raise TeacherMaterializationError(
            detail="oracle materialization requires a concrete sealed manifest resolver"
        )
    try:
        validate_request_manifest_fairness_trio(
            request.manifest,
            request.fairness_contract,
            request.cross_variant_manifests,
            authorized_registry=request.fairness_registry,
        )
    except TeacherOracleContractError as exc:
        raise TeacherMaterializationError(
            detail=f"oracle materialization rejected fairness authority: {exc}"
        ) from exc
    producer = validate_label_blind_payload(request.producer_input)
    validate_producer_input_manifest(producer, request.manifest)
    assignments = load_approved_compiled_assignments(
        request.manifest,
        request.approved_assignments,
        resolver,
    )
    summary = materialize_variant_memory(
        request.manifest,
        assignments,
        request.candidates,
        output_path=request.output,
        context=MaterializeVariantMemoryContext(
            sealed_payload_resolver=resolver,
            identity_registry=request.identity_registry,
        ),
    )
    typed_memory = DigestRefV1[str](
        uri=f"artifact://{request.output.name}",
        sha256=hashlib.sha256(request.output.read_bytes()).hexdigest(),
        schema="typed-memory-jsonl-v1",
    )
    output_manifest = build_output_manifest(
        request.manifest,
        typed_memory,
        summary,
        assignments=assignments,
        approved_assignments=request.approved_assignments,
    )
    _write_oracle_output_manifest_atomic(
        request.output.with_suffix(f"{request.output.suffix}.manifest.json"),
        output_manifest,
    )
    return output_manifest


def _write_oracle_output_manifest_atomic(
    path: Path, manifest: TeacherOracleOutputManifestV1
) -> None:
    payload = json.dumps(
        manifest.model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            _ = stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _ = temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_supervision(path: Path) -> tuple[TeacherSupervisionRow, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TeacherMaterializationError(
            detail=f"cannot read supervision {path}: {exc}",
        ) from exc

    rows: list[TeacherSupervisionRow] = []
    keys: set[tuple[str, str]] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = TeacherSupervisionRow.model_validate_json(line)
        except ValidationError as exc:
            raise TeacherMaterializationError(
                detail=f"{path}: line {line_number}: {exc}",
            ) from exc
        key = (row.observation_id, row.memory_id)
        if key in keys:
            raise TeacherMaterializationError(
                detail=f"{path}: line {line_number}: duplicate supervision key",
            )
        keys.add(key)
        rows.append(row)
    if not rows:
        raise TeacherMaterializationError(detail=f"{path}: no supervision rows")

    dimensions = {
        (len(row.features), len(row.teacher_embedding), len(row.geometry_target))
        for row in rows
    }
    if len(dimensions) != 1:
        raise TeacherMaterializationError(
            detail=f"{path}: inconsistent vector dimensions",
        )
    splits = {row.split for row in rows}
    if splits != {"train", "validation"}:
        raise TeacherMaterializationError(
            detail=f"{path}: train and validation supervision are required",
        )
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(row.group_id, set()).add(row.split)
    crossing = tuple(
        group_id
        for group_id, group_split in group_splits.items()
        if len(group_split) > 1
    )
    if crossing:
        raise TeacherMaterializationError(
            detail=f"{path}: group_id crosses splits: {min(crossing)!r}",
        )
    training_targets = {row.association_target for row in rows if row.split == "train"}
    if training_targets != set(range(max(training_targets) + 1)):
        raise TeacherMaterializationError(
            detail=f"{path}: train association targets must be contiguous from zero",
        )
    unseen = {
        row.association_target
        for row in rows
        if row.split == "validation" and row.association_target not in training_targets
    }
    if unseen:
        unseen_target = min(unseen)
        prefix = f"{path}: validation association target unseen in train"
        detail = f"{prefix}: {unseen_target}"
        raise TeacherMaterializationError(detail=detail)
    return tuple(rows)


def _encode_row(row: TeacherCacheRow) -> str:
    return json.dumps(
        {
            "sample_id": row.sample_id,
            "group_id": row.group_id,
            "split": row.split,
            "features": row.features,
            "teacher_embedding": row.teacher_embedding,
            "type_label": RECORD_TYPES[row.type_index],
            "geometry_target": row.geometry_target,
            "association_target": row.association_target,
            "uncertainty_target": row.uncertainty_target,
            "byte_cost": row.byte_cost,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _format_key(key: tuple[str, str]) -> str:
    return f"observation_id={key[0]!r}, memory_id={key[1]!r}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Materialize spatial student training rows from teacher cache.",
    )
    _ = parser.add_argument("--teacher-cache", type=Path, required=True)
    _ = parser.add_argument("--supervision", type=Path, required=True)
    _ = parser.add_argument("--out", type=Path, required=True)
    _ = parser.add_argument(
        "--production-attestation",
        required=True,
        help=(
            "canonical JSON SignedAttestationEnvelopeV1 for strict "
            "oracle materialization"
        ),
    )
    _ = parser.add_argument(
        "--authorized-registry",
        required=True,
        help=(
            "canonical JSON ImmutableAttestationKeyRegistryV1 for strict "
            "oracle materialization"
        ),
    )
    _ = parser.add_argument(
        "--approved-contract",
        required=True,
        help=(
            "canonical JSON CacheProductionContractV1 for strict oracle materialization"
        ),
    )
    _ = parser.add_argument("--oracle-manifest", required=True)
    _ = parser.add_argument("--approved-assignments", required=True)
    _ = parser.add_argument("--fairness-contract", required=True)
    _ = parser.add_argument("--fairness-registry", required=True)
    _ = parser.add_argument("--cross-variant-manifests", required=True)
    _ = parser.add_argument("--identity-registry")
    _ = parser.add_argument("--sealed-root", type=Path, required=True)
    _ = parser.add_argument("--sealed-root-manifest", type=Path, required=True)
    _ = parser.add_argument("--sealed-root-manifest-sha256", required=True)
    arguments = parser.parse_args(argv)
    teacher_cache_path = cast("Path", arguments.teacher_cache)
    supervision_path = cast("Path", arguments.supervision)
    out_path = cast("Path", arguments.out)
    provenance_arg = cast("str", arguments.production_attestation)
    registry_arg = cast("str", arguments.authorized_registry)
    contract_arg = cast("str", arguments.approved_contract)
    cross_variant_arg = cast("str", arguments.cross_variant_manifests)
    identity_registry_arg = cast("str | None", arguments.identity_registry)
    try:
        resolver = SealedManifestPayloadResolver(
            cast("Path", arguments.sealed_root),
            cast("Path", arguments.sealed_root_manifest).read_bytes(),
            cast("str", arguments.sealed_root_manifest_sha256),
        )
        try:
            rows = materialize_oracle_teacher_cache(
                OracleTeacherCacheMaterializationRequest(
                    teacher_cache=teacher_cache_path,
                    supervision=supervision_path,
                    output=out_path,
                    production_attestation=SignedAttestationEnvelopeV1.model_validate_json(
                        provenance_arg
                    ),
                    authorized_registry=ImmutableAttestationKeyRegistryV1.model_validate_json(
                        registry_arg
                    ),
                    approved_contract=CacheProductionContractV1.model_validate_json(
                        contract_arg
                    ),
                    oracle_manifest=TeacherOracleInputManifestV1.model_validate_json(
                        cast("str", arguments.oracle_manifest)
                    ),
                    approved_assignments=ApprovedCompiledAssignmentsArtifactV1.model_validate_json(
                        cast("str", arguments.approved_assignments)
                    ),
                    sealed_payload_resolver=resolver,
                    fairness_contract=CrossVariantFairnessContractV1.model_validate_json(
                        cast("str", arguments.fairness_contract)
                    ),
                    fairness_registry=ImmutableAttestationKeyRegistryV1.model_validate_json(
                        cast("str", arguments.fairness_registry)
                    ),
                    cross_variant_manifests=tuple(
                        TypeAdapter(list[TeacherOracleInputManifestV1]).validate_json(
                            cross_variant_arg
                        )
                    ),
                    identity_registry=(
                        ImmutableAttestationKeyRegistryV1.model_validate_json(
                            identity_registry_arg
                        )
                        if identity_registry_arg is not None
                        else None
                    ),
                )
            )
        finally:
            resolver.close()
    except (
        TeacherContractError,
        TeacherMaterializationError,
        TeacherOracleContractError,
        ValidationError,
    ) as exc:
        parser.error(str(exc))
    splits = {
        split: sum(row.split == split for row in rows)
        for split in ("train", "validation")
    }
    summary = {"out": str(out_path), "rows": len(rows), "splits": splits}
    _ = sys.stdout.write(f"{json.dumps(summary, sort_keys=True)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
