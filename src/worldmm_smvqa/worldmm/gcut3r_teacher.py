from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import sys
from argparse import ArgumentParser
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Self, cast, override, runtime_checkable

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, FiniteFloat, JsonValue, ValidationError, model_validator

from worldmm_smvqa.attestation import (
    AttestationError,
    ImmutableAttestationKeyRegistryV1,
    SignedAttestationEnvelopeV1,
)
from worldmm_smvqa.attestation import (
    verify_signed_attestation_envelope as _verify_shared_attestation_envelope,
)
from worldmm_smvqa.schema import FrozenModel
from worldmm_smvqa.worldmm.spatial_sensor import CameraIntrinsics as _CameraIntrinsics
from worldmm_smvqa.worldmm.typed_memory import (
    ObjectMemoryRecord,
    ObjectPresenceMemoryRecord,
    SourceCompactMemoryRecord,
    TypedMemoryRecord,
)

CameraIntrinsics = _CameraIntrinsics


type TeacherBackend = Literal["gcut3r_external", "cut3r_cache_fallback"]
type Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
type Quaternion = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]

EMPTY_PREFIX_SHA256 = hashlib.sha256(b"").hexdigest()
_POSE_UNIT_TOLERANCE = 1e-6
_POSE_COVARIANCE_TOLERANCE = 1e-9
_POSE_DIMENSION = 6
_POSE_DIAGONAL_STRIDE = 7
_CACHE_PROVENANCE_SIGNATURE_DOMAIN = b"worldmm-smvqa/cache-production-provenance/v1\x00"


class CacheProductionClaimsV1(FrozenModel):
    """Complete production cache lineage, carried as a signed envelope payload."""

    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runner_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mount_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resource_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_root_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_provider_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ontology_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    teacher_backend: Literal["gcut3r_external"] = "gcut3r_external"
    provider_id: str = Field(min_length=1)
    scope: str = Field(min_length=1)


def verify_signed_attestation_envelope(
    envelope: SignedAttestationEnvelopeV1,
    registry: ImmutableAttestationKeyRegistryV1,
    *,
    purpose: str,
) -> None:
    """Translate the shared protocol error at the teacher contract boundary."""
    try:
        _verify_shared_attestation_envelope(envelope, registry, purpose=purpose)
    except AttestationError as exc:
        raise TeacherContractError(detail=str(exc)) from exc


def validate_cache_production_attestation(
    records: Sequence[TeacherCacheRecord],
    attestation: SignedAttestationEnvelopeV1,
    *,
    authorized_registry: ImmutableAttestationKeyRegistryV1,
    approved_contract: CacheProductionContractV1,
) -> CacheProductionClaimsV1:
    """Validate the only production authority accepted by strict oracle paths."""
    validate_teacher_cache(records)
    verify_signed_attestation_envelope(
        attestation,
        authorized_registry,
        purpose="teacher_cache_production",
    )
    try:
        claims = CacheProductionClaimsV1.model_validate(attestation.payload)
    except ValidationError as exc:
        raise TeacherContractError(
            detail="cache production attestation payload does not use exact claims"
        ) from exc
    if claims.model_dump(mode="json") != attestation.payload:
        raise TeacherContractError(
            detail="cache production attestation payload has undeclared claims"
        )
    if claims.cache_content_sha256 != _cache_content_sha256(records):
        raise TeacherContractError(
            detail="cache production attestation does not bind ordered cache content"
        )
    contract_claims = (
        "oracle_manifest_sha256",
        "source_inventory_sha256",
        "frame_inventory_sha256",
        "provider_artifact_sha256",
        "semantic_provider_artifact_sha256",
        "ontology_sha256",
        "teacher_backend",
        "provider_id",
        "scope",
    )
    if any(
        getattr(claims, name) != getattr(approved_contract, name)
        for name in contract_claims
    ):
        raise TeacherContractError(
            detail="cache production attestation does not match approved contract"
        )
    for index, record in enumerate(records):
        _validate_response(record.request, record.response, strict=True)
        if (
            record.teacher_backend != claims.teacher_backend
            or record.provider_id != claims.provider_id
        ):
            raise TeacherContractError(
                detail=(
                    f"row {index}: cache backend/provider does not match attestation"
                )
            )
    return claims


@dataclass(frozen=True, slots=True)
class TeacherConfigurationError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"TeacherConfigurationError: {self.detail}"


@dataclass(frozen=True, slots=True)
class TeacherContractError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"TeacherContractError: {self.detail}"


class PoseGuidance(FrozenModel):
    source: Literal["imu", "vio", "slam", "ground_truth"]
    reference_frame_id: str = Field(min_length=1)
    translation_m: Vec3
    orientation_xyzw: Quaternion
    covariance_6x6: tuple[FiniteFloat, ...] = Field(min_length=36, max_length=36)

    @model_validator(mode="after")
    def _require_valid_pose(self) -> Self:
        norm_squared = sum(value * value for value in self.orientation_xyzw)
        if abs(norm_squared - 1.0) > _POSE_UNIT_TOLERANCE:
            msg = "orientation_xyzw must be unit length"
            raise ValueError(msg)
        covariance = self.covariance_6x6
        if any(
            abs(
                covariance[(row * _POSE_DIMENSION) + column]
                - covariance[(column * _POSE_DIMENSION) + row]
            )
            > _POSE_COVARIANCE_TOLERANCE
            for row in range(_POSE_DIMENSION)
            for column in range(row + 1, _POSE_DIMENSION)
        ):
            msg = "pose covariance must be symmetric"
            raise ValueError(msg)
        if any(
            covariance[index * _POSE_DIAGONAL_STRIDE] < 0.0
            for index in range(_POSE_DIMENSION)
        ):
            msg = "pose covariance diagonal must be non-negative"
            raise ValueError(msg)
        return self


class DepthGuidance(FrozenModel):
    depth_ref: str = Field(min_length=1)
    depth_scale_m: FiniteFloat = Field(gt=0.0)
    intrinsics: CameraIntrinsics


class TeacherObservation(FrozenModel):
    observation_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    timestamp: FiniteFloat = Field(ge=0.0)
    frame_ref: str = Field(min_length=1)
    local_frame_id: str = Field(min_length=1)
    camera_intrinsics: CameraIntrinsics | None = None
    pose_guidance: PoseGuidance | None = None
    depth_guidance: DepthGuidance | None = None

    @model_validator(mode="after")
    def _require_consistent_intrinsics(self) -> Self:
        if (
            self.camera_intrinsics is not None
            and self.depth_guidance is not None
            and self.camera_intrinsics != self.depth_guidance.intrinsics
        ):
            msg = "camera and depth intrinsics must match"
            raise ValueError(msg)
        return self


class TeacherRequest(TeacherObservation):
    sequence_index: int = Field(ge=0)
    previous_state_ref: str | None = None
    prefix_before_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TeacherResponse(FrozenModel):
    observation_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    timestamp: FiniteFloat = Field(ge=0.0)
    observed_through_time: FiniteFloat = Field(ge=0.0)
    state_ref: str = Field(min_length=1)
    records: tuple[TypedMemoryRecord, ...] = ()
    pointmap_ref: str | None = None
    confidence_ref: str | None = None
    frame_ref: str | None = None
    local_frame_id: str | None = None
    prefix_before_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def _require_causal_response(self) -> Self:
        if self.observed_through_time > self.timestamp:
            msg = "observed_through_time must not exceed response timestamp"
            raise ValueError(msg)
        if any(
            _record_end_time(record) > self.observed_through_time
            for record in self.records
        ):
            msg = "typed record validity must not exceed observed_through_time"
            raise ValueError(msg)
        return self


class LegacyNonProductionCacheSignerV1(FrozenModel):
    signer_id: str = Field(min_length=1)
    public_key_b64: str = Field(min_length=1)
    purposes: tuple[Literal["teacher_cache_production"], ...] = Field(min_length=1)
    not_before: FiniteFloat = Field(ge=0.0)
    not_after: FiniteFloat = Field(gt=0.0)
    revoked: bool = False
    allowed_scopes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_time_window(self) -> Self:
        if self.not_after <= self.not_before:
            msg = "signer not_after must be after not_before"
            raise ValueError(msg)
        return self


class LegacyNonProductionCacheSignerRegistryV1(FrozenModel):
    signers: tuple[LegacyNonProductionCacheSignerV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_signers(self) -> Self:
        if len({signer.signer_id for signer in self.signers}) != len(self.signers):
            msg = "cache signer registry signer_id must be unique"
            raise ValueError(msg)
        return self


CacheSignerRegistryV1 = ImmutableAttestationKeyRegistryV1


class LegacyNonProductionCacheProvenanceV1(FrozenModel):
    """Quarantined legacy Ed25519 provenance; never production authority."""

    signer_id: str = Field(min_length=1)
    signature_b64: str = Field(min_length=1)
    purpose: Literal["teacher_cache_production"] = "teacher_cache_production"
    issued_at: FiniteFloat = Field(ge=0.0)
    scope: str = Field(min_length=1)
    source_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_provider_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ontology_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CacheProductionContractV1(FrozenModel):
    """Exact local EXP-0005 cache approval; fallback is never production."""

    oracle_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_provider_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ontology_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    teacher_backend: Literal["gcut3r_external"] = "gcut3r_external"
    provider_id: str = Field(min_length=1)
    scope: str = Field(min_length=1)


class TeacherCacheRecord(FrozenModel):
    cache_version: Literal["teacher-cache-v1"] = "teacher-cache-v1"
    teacher_backend: TeacherBackend
    provider_id: str = Field(min_length=1)
    request: TeacherRequest
    response: TeacherResponse
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prefix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    legacy_non_production_provenance: LegacyNonProductionCacheProvenanceV1 | None = None


def validate_legacy_non_production_cache_provenance(
    records: Sequence[TeacherCacheRecord],
    expected: LegacyNonProductionCacheProvenanceV1,
    *,
    authorized_registry: LegacyNonProductionCacheSignerRegistryV1,
    approved_contract: CacheProductionContractV1 | None = None,
) -> None:
    """Validate quarantined legacy provenance.

    Never use this function as production authority.
    """
    if approved_contract is None:
        raise TeacherContractError(
            detail=(
                "legacy cache provenance validation is non-production; "
                "approved contract is required"
            )
        )
    validate_teacher_cache(records)
    if expected.cache_content_sha256 != _cache_content_sha256(records):
        raise TeacherContractError(
            detail="cache provenance does not bind the ordered cache content"
        )
    _verify_cache_provenance_signature(expected, authorized_registry)
    _validate_provenance_contract(expected, approved_contract)
    for index, record in enumerate(records):
        _validate_response(record.request, record.response, strict=True)
        if (
            record.teacher_backend != approved_contract.teacher_backend
            or record.provider_id != approved_contract.provider_id
        ):
            raise TeacherContractError(
                detail=(
                    f"row {index}: cache backend/provider does not "
                    "match approved contract"
                )
            )
        if record.legacy_non_production_provenance is None:
            raise TeacherContractError(
                detail=f"row {index}: legacy non-production provenance is required"
            )
        _verify_cache_provenance_signature(
            record.legacy_non_production_provenance,
            authorized_registry,
        )
        if record.legacy_non_production_provenance != expected:
            raise TeacherContractError(
                detail=(
                    f"row {index}: legacy non-production provenance does not "
                    "exactly match approved contract"
                )
            )


def _validate_provenance_contract(
    provenance: LegacyNonProductionCacheProvenanceV1,
    contract: CacheProductionContractV1,
) -> None:
    claims = (
        "oracle_manifest_sha256",
        "source_inventory_sha256",
        "frame_inventory_sha256",
        "provider_artifact_sha256",
        "semantic_provider_artifact_sha256",
        "ontology_sha256",
    )
    if any(getattr(provenance, claim) != getattr(contract, claim) for claim in claims):
        raise TeacherContractError(
            detail=(
                "cache provenance does not match approved "
                "manifest/inventory/provider contract"
            )
        )


def _verify_cache_provenance_signature(
    provenance: LegacyNonProductionCacheProvenanceV1,
    authorized_registry: LegacyNonProductionCacheSignerRegistryV1,
) -> None:
    signer = next(
        (
            item
            for item in authorized_registry.signers
            if item.signer_id == provenance.signer_id
        ),
        None,
    )
    if signer is None or signer.revoked:
        detail = "cache provenance signer is not authorized"
        raise TeacherContractError(detail=detail)
    if (
        provenance.purpose not in signer.purposes
        or provenance.scope not in signer.allowed_scopes
        or not signer.not_before <= provenance.issued_at <= signer.not_after
    ):
        detail = "cache provenance signer is not authorized for purpose, time, or scope"
        raise TeacherContractError(detail=detail)
    ed25519_public_key = Ed25519PublicKey
    invalid_signature = InvalidSignature
    try:
        signature = base64.b64decode(provenance.signature_b64, validate=True)
        public_key = base64.b64decode(signer.public_key_b64, validate=True)
        key = ed25519_public_key.from_public_bytes(public_key)
    except ValueError as exc:
        detail = "cache provenance Ed25519 verification is unavailable or malformed"
        raise TeacherContractError(detail=detail) from exc
    try:
        key.verify(
            signature,
            _CACHE_PROVENANCE_SIGNATURE_DOMAIN
            + _rfc8785_jcs(_cache_provenance_signing_payload(provenance)).encode(
                "utf-8"
            ),
        )
    except invalid_signature as exc:
        detail = "cache provenance Ed25519 signature verification failed"
        raise TeacherContractError(detail=detail) from exc


def _cache_content_sha256(records: Sequence[TeacherCacheRecord]) -> str:
    content = [
        {
            "teacher_backend": record.teacher_backend,
            "provider_id": record.provider_id,
            "request_sha256": record.request_sha256,
            "response_sha256": record.response_sha256,
        }
        for record in records
    ]
    return _digest(content)


def _cache_provenance_signing_payload(
    provenance: LegacyNonProductionCacheProvenanceV1,
) -> dict[str, object]:
    return {
        "signer_id": provenance.signer_id,
        "purpose": provenance.purpose,
        "issued_at": provenance.issued_at,
        "scope": provenance.scope,
        "source_inventory_sha256": provenance.source_inventory_sha256,
        "frame_inventory_sha256": provenance.frame_inventory_sha256,
        "provider_artifact_sha256": provenance.provider_artifact_sha256,
        "semantic_provider_artifact_sha256": (
            provenance.semantic_provider_artifact_sha256
        ),
        "ontology_sha256": provenance.ontology_sha256,
        "cache_content_sha256": provenance.cache_content_sha256,
        "oracle_manifest_sha256": provenance.oracle_manifest_sha256,
    }


class GCut3RPaths(FrozenModel):
    code_path: Path
    checkpoint_path: Path


class Cut3RPaths(FrozenModel):
    code_path: Path
    model_path: Path


@dataclass(frozen=True, slots=True)
class ProviderStep:
    response: TeacherResponse
    state: object | None


@runtime_checkable
class GCut3RProvider(Protocol):
    provider_id: str

    def infer(
        self,
        request: TeacherRequest,
        previous_state: object | None,
    ) -> ProviderStep:
        """Infer one observation using only the supplied causal prefix state."""
        ...


@runtime_checkable
class CacheProductionSigner(Protocol):
    """Capability that can sign a completed cache provenance payload."""

    signer_id: str
    issued_at: float
    scope: str

    def sign(self, payload: bytes) -> bytes:
        """Sign the domain-separated local canonical payload."""
        ...


class GCut3RTeacherAdapter:
    """Causal adapter for an external G-CUT3R-compatible provider.

    No G-CUT3R implementation is bundled or imported here. The provider receives
    one observation and the immediately preceding opaque state only.
    """

    def __init__(self, provider: object | None) -> None:
        """Bind an already-installed external provider without importing it."""
        if provider is None:
            raise TeacherConfigurationError(
                detail=(
                    "no G-CUT3R provider supplied; install an external provider, "
                    "set WORLDMM_GCUT3R_CODE_PATH and "
                    "WORLDMM_GCUT3R_CHECKPOINT_PATH, then pass a "
                    "GCut3RProvider implementation (this package never downloads "
                    "or imports G-CUT3R automatically)"
                ),
            )
        if not isinstance(provider, GCut3RProvider):
            raise TeacherConfigurationError(
                detail="provider does not implement the GCut3RProvider contract",
            )
        if not provider.provider_id:
            raise TeacherConfigurationError(detail="provider_id must not be empty")
        self._provider: GCut3RProvider = provider

    def run(
        self,
        observations: Iterable[TeacherObservation],
    ) -> tuple[TeacherCacheRecord, ...]:
        """Run the established causal cache path without production-only bindings."""
        return self._run(
            observations,
            legacy_non_production_provenance=None,
            strict_response=False,
        )

    def _run(
        self,
        observations: Iterable[TeacherObservation],
        *,
        legacy_non_production_provenance: LegacyNonProductionCacheProvenanceV1 | None,
        strict_response: bool,
    ) -> tuple[TeacherCacheRecord, ...]:
        rows: list[TeacherCacheRecord] = []
        previous_prefix = EMPTY_PREFIX_SHA256
        previous_state_ref: str | None = None
        previous_state: object | None = None
        video_id: str | None = None
        previous_timestamp: float | None = None
        observation_ids: set[str] = set()

        for sequence_index, observation in enumerate(observations):
            if video_id is None:
                video_id = observation.video_id
            elif observation.video_id != video_id:
                raise TeacherContractError(
                    detail="one adapter run must contain exactly one video_id",
                )
            if observation.observation_id in observation_ids:
                raise TeacherContractError(
                    detail=f"duplicate observation_id: {observation.observation_id}",
                )
            if (
                previous_timestamp is not None
                and observation.timestamp <= previous_timestamp
            ):
                raise TeacherContractError(
                    detail="observation timestamps must be strictly increasing",
                )

            request = TeacherRequest(
                observation_id=observation.observation_id,
                video_id=observation.video_id,
                timestamp=observation.timestamp,
                frame_ref=observation.frame_ref,
                local_frame_id=observation.local_frame_id,
                camera_intrinsics=observation.camera_intrinsics,
                pose_guidance=observation.pose_guidance,
                depth_guidance=observation.depth_guidance,
                sequence_index=sequence_index,
                previous_state_ref=previous_state_ref,
                prefix_before_sha256=previous_prefix,
            )
            step = self._provider.infer(request, previous_state)
            _validate_response(request, step.response, strict=strict_response)
            row = build_teacher_cache_record(
                teacher_backend="gcut3r_external",
                provider_id=self._provider.provider_id,
                request=request,
                response=step.response,
                legacy_non_production_provenance=legacy_non_production_provenance,
            )
            rows.append(row)
            previous_prefix = row.prefix_sha256
            previous_state_ref = step.response.state_ref
            previous_state = step.state
            previous_timestamp = observation.timestamp
            observation_ids.add(observation.observation_id)

        validate_teacher_cache(rows)
        return tuple(rows)


def resolve_gcut3r_paths(env: Mapping[str, str]) -> GCut3RPaths:
    return GCut3RPaths(
        code_path=_required_path(env, "WORLDMM_GCUT3R_CODE_PATH", directory=True),
        checkpoint_path=_required_path(
            env,
            "WORLDMM_GCUT3R_CHECKPOINT_PATH",
            directory=False,
        ),
    )


def resolve_cut3r_paths(env: Mapping[str, str]) -> Cut3RPaths:
    code_path = _required_path(env, "WORLDMM_CUT3R_CODE_PATH", directory=True)
    demo_path = code_path / "demo.py"
    if not demo_path.is_file():
        raise TeacherConfigurationError(
            detail=(
                f"WORLDMM_CUT3R_CODE_PATH must contain demo.py: {demo_path}; "
                "expected the official https://github.com/CUT3R/CUT3R checkout"
            ),
        )
    return Cut3RPaths(
        code_path=code_path,
        model_path=_required_path(env, "WORLDMM_CUT3R_MODEL_PATH", directory=False),
    )


def build_cut3r_demo_command(
    paths: Cut3RPaths,
    *,
    sequence_path: Path,
    output_dir: Path,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Build, but never execute, the official CUT3R demo command."""
    return (
        python_executable,
        str(paths.code_path / "demo.py"),
        "--model_path",
        str(paths.model_path),
        "--seq_path",
        str(sequence_path),
        "--output_dir",
        str(output_dir),
    )


def encode_teacher_request(request: TeacherRequest) -> str:
    """Encode one request line for an external G-CUT3R JSONL provider."""
    return _canonical_json(request.model_dump(mode="json"))


def decode_teacher_request(line: str) -> TeacherRequest:
    return TeacherRequest.model_validate_json(line)


def encode_teacher_response(response: TeacherResponse) -> str:
    """Encode one response line for an external G-CUT3R JSONL provider."""
    return _canonical_json(response.model_dump(mode="json"))


def decode_teacher_response(line: str) -> TeacherResponse:
    return TeacherResponse.model_validate_json(line)


def write_teacher_cache(
    path: Path,
    records: Sequence[TeacherCacheRecord],
) -> None:
    validate_teacher_cache(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _ = temporary.write_text(
            "".join(
                f"{_canonical_json(row.model_dump(mode='json'))}\n" for row in records
            ),
            encoding="utf-8",
        )
        _ = temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_teacher_cache(path: Path) -> tuple[TeacherCacheRecord, ...]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                msg = "teacher cache must be a regular file"
                raise OSError(msg)
            with os.fdopen(
                descriptor,
                "r",
                encoding="utf-8",
                closefd=False,
            ) as stream:
                lines = stream.read().splitlines()
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise TeacherContractError(
            detail=f"cannot read teacher cache {path}: {exc}",
        ) from exc
    try:
        records = tuple(
            TeacherCacheRecord.model_validate_json(line)
            for line in lines
            if line.strip()
        )
    except ValueError as exc:
        raise TeacherContractError(
            detail=f"invalid teacher cache row in {path}: {exc}",
        ) from exc
    validate_teacher_cache(records)
    return records


def validate_teacher_cache(records: Sequence[TeacherCacheRecord]) -> None:
    if not records:
        raise TeacherContractError(detail="teacher cache must contain at least one row")
    identities = {(row.teacher_backend, row.provider_id) for row in records}
    if len(identities) != 1:
        raise TeacherContractError(
            detail="teacher backend and provider_id must be stable across the cache",
        )
    streams: dict[str, list[TeacherCacheRecord]] = {}
    for row in records:
        streams.setdefault(row.request.video_id, []).append(row)
    for stream in streams.values():
        _validate_teacher_stream(stream)


def _validate_teacher_stream(records: Sequence[TeacherCacheRecord]) -> None:
    previous_prefix = EMPTY_PREFIX_SHA256
    previous_state_ref: str | None = None
    previous_timestamp: float | None = None
    observation_ids: set[str] = set()

    for sequence_index, row in enumerate(records):
        request = row.request
        if request.sequence_index != sequence_index:
            raise TeacherContractError(
                detail=(
                    f"row {sequence_index}: sequence_index must be {sequence_index}, "
                    f"received {request.sequence_index}"
                ),
            )
        if request.prefix_before_sha256 != previous_prefix:
            raise TeacherContractError(
                detail=f"row {sequence_index}: broken prefix hash chain",
            )
        if request.previous_state_ref != previous_state_ref:
            raise TeacherContractError(
                detail=(
                    f"row {sequence_index}: previous_state_ref does not match prefix"
                ),
            )
        if request.observation_id in observation_ids:
            raise TeacherContractError(
                detail=f"row {sequence_index}: duplicate observation_id",
            )
        if previous_timestamp is not None and request.timestamp <= previous_timestamp:
            raise TeacherContractError(
                detail=f"row {sequence_index}: timestamps must be strictly increasing",
            )

        _validate_response(request, row.response)
        expected = build_teacher_cache_record(
            teacher_backend=row.teacher_backend,
            provider_id=row.provider_id,
            request=request,
            response=row.response,
            legacy_non_production_provenance=row.legacy_non_production_provenance,
        )
        if (
            row.request_sha256 != expected.request_sha256
            or row.response_sha256 != expected.response_sha256
            or row.prefix_sha256 != expected.prefix_sha256
        ):
            raise TeacherContractError(
                detail=f"row {sequence_index}: cache digest mismatch",
            )

        previous_prefix = row.prefix_sha256
        previous_state_ref = row.response.state_ref
        previous_timestamp = request.timestamp
        observation_ids.add(request.observation_id)


def build_teacher_cache_record(
    *,
    teacher_backend: TeacherBackend,
    provider_id: str,
    request: TeacherRequest,
    response: TeacherResponse,
    legacy_non_production_provenance: LegacyNonProductionCacheProvenanceV1
    | None = None,
) -> TeacherCacheRecord:
    """Build one digest-bound row; caller supplies the honest backend identity."""
    request_sha256 = _digest(request.model_dump(mode="json"))
    response_sha256 = _digest(response.model_dump(mode="json"))
    digest_fields: dict[str, object] = {
        "prefix_before_sha256": request.prefix_before_sha256,
        "teacher_backend": teacher_backend,
        "provider_id": provider_id,
        "request_sha256": request_sha256,
        "response_sha256": response_sha256,
    }
    prefix_sha256 = _digest(digest_fields)
    return TeacherCacheRecord(
        teacher_backend=teacher_backend,
        provider_id=provider_id,
        request=request,
        response=response,
        request_sha256=request_sha256,
        response_sha256=response_sha256,
        prefix_sha256=prefix_sha256,
        legacy_non_production_provenance=legacy_non_production_provenance,
    )


def _record_end_time(record: TypedMemoryRecord) -> FiniteFloat:
    if isinstance(record, (ObjectPresenceMemoryRecord, SourceCompactMemoryRecord)):
        return record.timestamp
    return record.validity.end_time


def _validate_response(
    request: TeacherRequest,
    response: TeacherResponse,
    *,
    strict: bool = False,
) -> None:
    if (
        response.observation_id != request.observation_id
        or response.video_id != request.video_id
        or response.timestamp != request.timestamp
    ):
        raise TeacherContractError(
            detail=(
                f"{request.observation_id}: response does not match request identity"
            ),
        )
    if strict and response.frame_ref != request.frame_ref:
        raise TeacherContractError(
            detail=(
                f"{request.observation_id}: response frame_ref does not match request"
            ),
        )
    if strict and response.local_frame_id != request.local_frame_id:
        raise TeacherContractError(
            detail=(
                f"{request.observation_id}: response local_frame_id does not "
                "match request"
            ),
        )
    if strict and response.prefix_before_sha256 != request.prefix_before_sha256:
        raise TeacherContractError(
            detail=(
                f"{request.observation_id}: response prefix does not match "
                "causal request prefix"
            ),
        )
    if response.observed_through_time > request.timestamp:
        raise TeacherContractError(
            detail=f"{request.observation_id}: response observes future time",
        )
    for record in response.records:
        if strict and not isinstance(
            record,
            (ObjectMemoryRecord, ObjectPresenceMemoryRecord),
        ):
            raise TeacherContractError(
                detail=(
                    f"{request.observation_id}: only EXP-0005 object or "
                    "object-presence records are permitted in production caches"
                )
            )
        if record.source_video_id != request.video_id:
            raise TeacherContractError(
                detail=(
                    f"{request.observation_id}: record {record.memory_id} "
                    "references another source video"
                ),
            )
        if (
            strict
            and not isinstance(record, ObjectPresenceMemoryRecord)
            and record.local_frame_id != request.local_frame_id
        ):
            raise TeacherContractError(
                detail=(
                    f"{request.observation_id}: record {record.memory_id} "
                    "uses a different local frame"
                ),
            )
        if _record_end_time(record) > response.observed_through_time:
            raise TeacherContractError(
                detail=(
                    f"{request.observation_id}: record {record.memory_id} "
                    "has future validity"
                ),
            )


def _required_path(
    env: Mapping[str, str],
    name: str,
    *,
    directory: bool,
) -> Path:
    value = env.get(name)
    if not value:
        raise TeacherConfigurationError(
            detail=f"missing {name}; configure the external teacher artifact path",
        )
    path = Path(value).expanduser()
    expected = "directory" if directory else "file or model directory"
    if not path.exists() or (directory and not path.is_dir()):
        raise TeacherConfigurationError(
            detail=f"{name} must reference an existing {expected}: {path}",
        )
    return path


def _digest(value: object) -> str:
    return hashlib.sha256(_rfc8785_jcs(value).encode("utf-8")).hexdigest()


def _rfc8785_jcs(value: object) -> str:
    """Encode a closed cache-attestation payload as RFC 8785 canonical JSON."""
    try:
        payload = rfc8785.dumps(cast("JsonValue", value))
    except (rfc8785.CanonicalizationError, UnicodeEncodeError) as exc:
        raise TeacherContractError(
            detail="cache attestation payload is not RFC 8785 canonicalizable"
        ) from exc
    return payload.decode("utf-8")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Validate causal spatial teacher caches.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser(
        "validate-cache",
        help="validate hashes, state continuity, and prefix causality",
    )
    _ = validate.add_argument("--cache", type=Path, required=True)
    arguments = parser.parse_args(argv)
    cache_path = cast("Path", arguments.cache)
    records = read_teacher_cache(cache_path)
    payload = {
        "cache": str(cache_path),
        "record_count": len(records),
        "teacher_backend": records[0].teacher_backend if records else None,
        "provider_id": records[0].provider_id if records else None,
        "valid": True,
    }
    _ = sys.stdout.write(f"{_canonical_json(payload)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
