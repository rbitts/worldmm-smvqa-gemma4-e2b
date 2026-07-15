from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Generic, Literal, Protocol, Self, TypeIs, TypeVar, cast, override

from pydantic import (
    AliasChoices,
    Field,
    FiniteFloat,
    JsonValue,
    ValidationError,
    model_validator,
)

from worldmm_smvqa.attestation import (
    AttestationError,
    ImmutableAttestationKeyRegistryV1,
    SignedAttestationEnvelopeV1,
    verify_signed_attestation_envelope,
)
from worldmm_smvqa.openat2 import Openat2UnsupportedError, openat2_sealed
from worldmm_smvqa.schema import FrozenModel
from worldmm_smvqa.worldmm.typed_memory import (
    ObjectMemoryRecord,
    ObjectPresenceMemoryRecord,
    ScoredMemoryCandidate,
    TypedMemoryArtifactValidationOptions,
    TypedMemoryWriteOptions,
    TypedMemoryWriteSummary,
    serialized_byte_cost,
    validate_typed_memory_artifact,
    write_typed_memory_artifact,
)

type OracleVariant = Literal["E0", "T0", "T1"]
type GeometryDecision = Literal["selected", "abstained"]
type GeometryAbstentionReason = Literal[
    "no_valid_depth",
    "insufficient_valid_mask_points",
    "provider_uncertainty",
    "outside_approved_bounds",
]
type PlaceMode = Literal["stable_identity", "frame_bound"]
type ProviderGateDecision = Literal["go", "no_go", "not_measurable", "not_decidable"]
type TerminalState = Literal["completed", "failed", "blocked"]
type Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
SchemaId_co = TypeVar("SchemaId_co", bound=str, default=str, covariant=True)

WINDOW_MICROSECONDS = 30_000_000
RESULT_CLASS = "teacher_oracle"
CANDIDATE_SCORE_LITERAL = "score_per_actual_serialized_byte"
NORMALIZATION_LITERAL = "actual_serialized_bytes"
_SHA256 = r"^[0-9a-f]{64}$"
_COORDINATE_BOUNDS_ERROR = "coordinate contract min_xyz must not exceed max_xyz"
_IDENTITY_RETAINED_IDS_ERROR = "identity attestation retained_object_ids must be unique"
_SELECTED_POINT_COORDINATE_BOUNDS_ERROR = (
    "selected-point coordinate contract min must not exceed max"
)
_SELECTED_POINT_CONTRACT_BOUNDS_ERROR = (
    "selected-point bounds must lie within coordinate contract"
)
_XYZ_COORDINATE_COUNT = 3
_ORACLE_VARIANT_COUNT = 3
_SELECTED_POINT_SHAPE_ERROR = "selected-point shape must be (point_count, 3)"
_SELECTED_POINT_BOUNDS_ERROR = (
    "selected-point bounds_min_m must not exceed bounds_max_m"
)
_STABLE_IDENTITY_ERROR = "stable_identity place grounding requires identity_certificate"
_FRAME_BOUND_ERROR = "frame_bound place grounding must not carry identity_certificate"
_SELECTED_GEOMETRY_ERROR = "selected geometry requires points and no abstention_reason"
_ABSTAINED_GEOMETRY_ERROR = (
    "abstained geometry requires closed reason, evidence, and no points"
)
_TEACHER_PROVENANCE_ERROR = "teacher variants require provider_provenance"
_SEMANTIC_PROVENANCE_ERROR = "teacher variants require semantic provider and ontology"
_MAX_WINDOW_BYTES_ERROR = "max_window_bytes cannot exceed actual_bytes"
_PROVIDER_GATE_ERROR = "go requires completed operational_state"
_ASSIGNMENT_DIGEST_ERROR = "assignment_sha256 does not bind compiled assignments"
_PLACE_GROUNDING_T1_ERROR = "place grounding is T1-only"
_SEALED_URI_PREFIX = "sealed://"
_SEALED_MANIFEST_DOT_COMPONENT_ERROR = (
    "sealed manifest path must not contain dot components"
)

_DENIED_INPUT_KEYS = frozenset(
    {
        "answer",
        "answers",
        "choice",
        "choices",
        "correctchoice",
        "evidence",
        "evidenceannotation",
        "evidencelist",
        "evidences",
        "groundtruthanswer",
        "label",
        "labels",
        "qa",
        "question",
        "questionid",
        "target",
        "targets",
    },
)
_LABEL_BEARING_VALUE_TOKENS = frozenset(
    {
        "answer",
        "checkpoint",
        "choice",
        "correct",
        "evidence",
        "groundtruth",
        "inference",
        "label",
        "question",
        "student",
    },
)
_PRODUCER_INPUT_FIELDS = frozenset(
    {"source_inventory", "frame_inventory", "observations"}
)
_DIGEST_REF_FIELDS = frozenset({"uri", "sha256", "schema", "schema_id"})
_FRAME_INVENTORY_FIELDS = _DIGEST_REF_FIELDS | frozenset({"frames"})
_FRAME_FIELDS = frozenset(
    {"observation_id", "source_video_id", "frame_ref", "local_frame_id", "timestamp_us"}
)
_OPAQUE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]*$")
_ARTIFACT_URI_PATTERN = re.compile(r"^artifact://([A-Za-z0-9][A-Za-z0-9._~-]*)$")


class SealedManifestEntryV1(FrozenModel):
    """Pinned metadata for a payload opened only beneath the sealed root."""

    canonical_path: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._~-]*(?:/[A-Za-z0-9][A-Za-z0-9._~-]*)*$"
    )
    role: Literal[
        "mask",
        "points",
        "selected_indices",
        "pointmap",
        "identity",
        "compiled_assignments",
    ]
    object_id: str | None = None
    frame_ref: str | None = None
    size_bytes: int = Field(ge=0)
    mode: int = Field(ge=0)
    device: int = Field(ge=0)
    nlink: Literal[1] = 1
    sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _canonical_relative_path(self) -> Self:
        if any(
            component in {".", ".."} for component in self.canonical_path.split("/")
        ):
            raise ValueError(_SEALED_MANIFEST_DOT_COMPONENT_ERROR)
        return self


class SealedRootManifestBindingV1(FrozenModel):
    uri: str = Field(pattern=r"^sealed://[A-Za-z0-9][A-Za-z0-9._~/-]*$")
    entry: SealedManifestEntryV1


class SealedRootManifestV1(FrozenModel):
    """Canonical sealed-root authority for every payload descriptor."""

    entries: tuple[SealedRootManifestBindingV1, ...] = ()

    @model_validator(mode="after")
    def _uris_are_unique(self) -> Self:
        if len({binding.uri for binding in self.entries}) != len(self.entries):
            msg = "sealed root manifest URIs must be unique"
            raise ValueError(msg)
        return self


def sealed_root_manifest_bytes(manifest: SealedRootManifestV1) -> bytes:
    return _canonical_json(manifest.model_dump(mode="json")).encode("utf-8")


def sealed_root_manifest_sha256(manifest_bytes: bytes) -> str:
    return hashlib.sha256(manifest_bytes).hexdigest()


class SealedPayloadResolver:
    """Resolver boundary; arbitrary bytes-returning callbacks are not accepted."""

    def resolve(self, descriptor: SealedPayloadRefV1[str]) -> bytes:
        """Resolve one manifest-bound sealed payload descriptor."""
        del descriptor
        raise NotImplementedError


class _OracleRecordGeometryValidator(Protocol):
    def __call__(
        self,
        record: ObjectMemoryRecord,
        *,
        selected_points: SelectedPointPayloadRefV1,
        selected_points_bytes: bytes,
    ) -> None: ...


class SealedManifestPayloadResolver(SealedPayloadResolver):
    """Resolve allowlisted payloads beneath one non-symlinked sealed root."""

    def __init__(
        self,
        root: Path,
        manifest_bytes: bytes,
        manifest_sha256: str,
    ) -> None:
        """Open and pin one sealed payload root and its canonical manifest."""
        manifest = _parse_sealed_root_manifest(manifest_bytes, manifest_sha256)
        self._manifest_sha256: str = manifest_sha256
        self._manifest: dict[str, SealedManifestEntryV1] = {
            binding.uri: binding.entry for binding in manifest.entries
        }
        root_path = root.absolute()

        root_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        parent_fd = os.open(root_path.anchor, root_flags)
        try:
            self._root_fd: int = openat2_sealed(
                parent_fd,
                "/".join(root_path.parts[1:]),
                root_flags,
            )
        except Openat2UnsupportedError as exc:
            raise TeacherOracleContractError(
                detail=f"sealed payload resolution requires openat2: {exc}"
            ) from exc
        except OSError as exc:
            raise TeacherOracleContractError(
                detail=f"cannot open sealed payload root: {exc}"
            ) from exc
        finally:
            os.close(parent_fd)
        if not stat.S_ISDIR(os.fstat(self._root_fd).st_mode):
            os.close(self._root_fd)
            raise TeacherOracleContractError(
                detail="sealed payload root must be a directory"
            )

    def close(self) -> None:
        """Close the owned root directory descriptor."""
        os.close(self._root_fd)

    @override
    def resolve(self, descriptor: SealedPayloadRefV1[str]) -> bytes:
        if descriptor.sealed_root_sha256 != self._manifest_sha256:
            raise TeacherOracleContractError(
                detail=(
                    "sealed payload descriptor does not bind this sealed root manifest"
                )
            )
        entry = self._manifest.get(descriptor.uri)
        if entry is None:
            raise TeacherOracleContractError(
                detail="sealed payload descriptor is not authorized by sealed manifest"
            )
        relative_path = entry.canonical_path
        expected_role = (
            "mask"
            if isinstance(descriptor, MaskPayloadRefV1)
            else "points"
            if isinstance(descriptor, SelectedPointPayloadRefV1)
            else "pointmap"
            if isinstance(descriptor, SourcePointmapPayloadRefV1)
            else "selected_indices"
            if isinstance(descriptor, SelectedIndexPayloadRefV1)
            else "identity"
            if descriptor.schema_id == "stable-identity-certificate-v1"
            else "compiled_assignments"
            if descriptor.schema_id == "compiled-oracle-assignments-v1"
            else None
        )
        if expected_role is None or entry.role != expected_role:
            raise TeacherOracleContractError(
                detail="sealed manifest payload role does not match descriptor"
            )
        descriptor_frame = getattr(descriptor, "frame_ref", None)
        descriptor_object = getattr(descriptor, "object_id", None)
        if entry.frame_ref != descriptor_frame or entry.object_id != descriptor_object:
            raise TeacherOracleContractError(
                detail="sealed manifest payload object/frame does not match descriptor"
            )
        try:
            payload_fd = openat2_sealed(
                self._root_fd,
                relative_path,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
        except Openat2UnsupportedError as exc:
            raise TeacherOracleContractError(
                detail=f"sealed payload resolution requires openat2: {exc}"
            ) from exc
        except OSError as exc:
            raise TeacherOracleContractError(
                detail=f"cannot open sealed payload: {exc}"
            ) from exc
        try:
            metadata = os.fstat(payload_fd)
            _validate_sealed_payload_metadata(metadata, entry)
            with os.fdopen(payload_fd, "rb", closefd=False) as stream:
                payload = stream.read()
            if os.fstat(payload_fd) != metadata:
                raise TeacherOracleContractError(
                    detail="sealed manifest payload changed during read"
                )
        finally:
            os.close(payload_fd)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != descriptor.sha256 or digest != entry.sha256:
            raise TeacherOracleContractError(
                detail="sealed manifest payload bytes do not match descriptor digest"
            )
        return payload


def _parse_sealed_root_manifest(
    manifest_bytes: bytes, manifest_sha256: str
) -> SealedRootManifestV1:
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise TeacherOracleContractError(
            detail="sealed root manifest bytes do not match declared digest"
        )
    try:
        parsed = cast("JsonValue", json.loads(manifest_bytes))
        manifest = SealedRootManifestV1.model_validate(parsed)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise TeacherOracleContractError(
            detail="sealed root manifest bytes do not use the closed schema"
        ) from exc
    if sealed_root_manifest_bytes(manifest) != manifest_bytes:
        raise TeacherOracleContractError(
            detail="sealed root manifest bytes are not canonical JSON"
        )
    return manifest


class TeacherOracleContractError(ValueError):
    """Raised when an oracle input or artifact violates its closed contract."""

    def __init__(self, detail: str) -> None:
        """Initialize the contract error with its stable detail."""
        self.detail: str = detail
        super().__init__(detail)

    @override
    def __str__(self) -> str:
        return f"TeacherOracleContractError: {self.detail}"


def _validate_sealed_payload_metadata(
    metadata: os.stat_result, entry: SealedManifestEntryV1
) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise TeacherOracleContractError(
            detail="sealed manifest payload must be a regular file"
        )
    if stat.S_IMODE(metadata.st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise TeacherOracleContractError(
            detail="sealed manifest payload must be read-only"
        )
    if (
        metadata.st_size != entry.size_bytes
        or stat.S_IMODE(metadata.st_mode) != entry.mode
        or metadata.st_dev != entry.device
        or metadata.st_nlink != 1
    ):
        raise TeacherOracleContractError(
            detail="openat2 payload metadata does not match sealed manifest"
        )


class DigestRefV1(FrozenModel, Generic[SchemaId_co]):
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256)
    schema_id: SchemaId_co = Field(
        alias="schema",
        validation_alias=AliasChoices("schema", "schema_id"),
        min_length=1,
    )


class SealedPayloadRefV1(
    DigestRefV1[SchemaId_co],
    Generic[SchemaId_co],
):
    """Descriptor resolvable only through a manifest-bound sealed artifact root."""

    sealed_root_sha256: str = Field(pattern=_SHA256)
    manifest_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _sealed_uri(self) -> Self:
        if not self.uri.startswith(_SEALED_URI_PREFIX):
            msg = "payload uri must use sealed:// manifest-bound storage"
            raise ValueError(msg)
        return self


class ApprovedFrameRefV1(FrozenModel):
    frame_ref: str = Field(min_length=1)
    source_video_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    local_frame_id: str = Field(min_length=1)
    timestamp_us: int = Field(ge=0)

    @property
    def timestamp(self) -> float:
        """Return the exact microsecond timestamp as seconds."""
        return self.timestamp_us / 1_000_000


class FrameInventoryV1(DigestRefV1[Literal["frame-inventory-v1"]]):
    schema_id: Literal["frame-inventory-v1"] = Field(
        default="frame-inventory-v1",
        alias="schema",
        validation_alias=AliasChoices("schema", "schema_id"),
    )
    frames: tuple[ApprovedFrameRefV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_frames(self) -> Self:
        frame_refs = {frame.frame_ref for frame in self.frames}
        if len(frame_refs) != len(self.frames):
            msg = "frame inventory frame_ref must be globally unambiguous"
            raise ValueError(msg)
        return self


class MaskPayloadRefV1(SealedPayloadRefV1[Literal["mask-dense-u8-v1"]]):
    object_id: str = Field(min_length=1)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    dtype: Literal["bool", "uint8"]
    frame_ref: str = Field(min_length=1)


class CoordinateFrameContractV1(FrozenModel):
    """Signed experiment contract for every decoded geometry coordinate."""

    frame_id: str = Field(min_length=1)
    units: Literal["m"]
    min_xyz: Vec3
    max_xyz: Vec3

    @model_validator(mode="after")
    def _ordered_bounds(self) -> Self:
        if any(
            low > high for low, high in zip(self.min_xyz, self.max_xyz, strict=True)
        ):
            raise ValueError(_COORDINATE_BOUNDS_ERROR)
        return self


class SourcePointmapPayloadRefV1(
    SealedPayloadRefV1[Literal["source-pointmap-f32x3-v1"]]
):
    """Dense row-major source point map; pixel index N owns point row N."""

    schema_id: Literal["source-pointmap-f32x3-v1"] = Field(
        default="source-pointmap-f32x3-v1",
        alias="schema",
        validation_alias=AliasChoices("schema", "schema_id"),
    )
    object_id: str = Field(min_length=1)
    frame_ref: str = Field(min_length=1)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    dtype: Literal["float32"] = "float32"


class SelectedIndexPayloadRefV1(SealedPayloadRefV1[Literal["selected-indices-u32-v1"]]):
    """Canonical little-endian uint32 indices into the approved object mask."""

    schema_id: Literal["selected-indices-u32-v1"] = Field(
        default="selected-indices-u32-v1",
        alias="schema",
        validation_alias=AliasChoices("schema", "schema_id"),
    )
    dtype: Literal["uint32"] = "uint32"
    object_id: str = Field(min_length=1)
    frame_ref: str = Field(min_length=1)
    point_count: int = Field(gt=0)
    mask_width_px: int = Field(gt=0)
    mask_height_px: int = Field(gt=0)


class SelectedPointPayloadRefV1(
    SealedPayloadRefV1[Literal["selected-points-f32x3-v1"]]
):
    schema_id: Literal["selected-points-f32x3-v1"] = Field(
        default="selected-points-f32x3-v1",
        alias="schema",
        validation_alias=AliasChoices("schema", "schema_id"),
    )
    dtype: Literal["float32"] = "float32"
    object_id: str = Field(min_length=1)
    shape: tuple[int, int]
    point_count: int = Field(gt=0)
    frame_ref: str = Field(min_length=1)
    coordinate_frame_id: str = Field(min_length=1)
    coordinate_units: Literal["m"]
    coordinate_min_xyz: Vec3
    coordinate_max_xyz: Vec3
    bounds_min_m: Vec3
    bounds_max_m: Vec3

    @model_validator(mode="after")
    def _valid_points(self) -> Self:
        if self.shape != (self.point_count, _XYZ_COORDINATE_COUNT):
            msg = _SELECTED_POINT_SHAPE_ERROR
            raise ValueError(msg)
        if any(
            low > high
            for low, high in zip(
                self.bounds_min_m,
                self.bounds_max_m,
                strict=True,
            )
        ):
            msg = _SELECTED_POINT_BOUNDS_ERROR
            raise ValueError(msg)
        if any(
            low > high
            for low, high in zip(
                self.coordinate_min_xyz,
                self.coordinate_max_xyz,
                strict=True,
            )
        ):
            raise ValueError(_SELECTED_POINT_COORDINATE_BOUNDS_ERROR)
        if any(
            low < contract_low or high > contract_high
            for low, high, contract_low, contract_high in zip(
                self.bounds_min_m,
                self.bounds_max_m,
                self.coordinate_min_xyz,
                self.coordinate_max_xyz,
                strict=True,
            )
        ):
            raise ValueError(_SELECTED_POINT_CONTRACT_BOUNDS_ERROR)
        return self


class SharedObjectSemanticAssignmentV1(FrozenModel):
    object_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    source_video_id: str = Field(min_length=1)
    semantic_class: str = Field(min_length=1)
    semantic_confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    semantic_provider_id: str = Field(min_length=1)
    ontology_sha256: str = Field(pattern=_SHA256)
    mask: MaskPayloadRefV1


class IdentityIndexAttestationClaimsV1(FrozenModel):
    """Registry-authorized retained identity index used for stable place claims."""

    scope: str = Field(min_length=1)
    cutoff_us: int = Field(ge=0)
    retained_object_ids: tuple[str, ...] = Field(min_length=1)
    identity_index_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _unique_retained_ids(self) -> Self:
        if len(set(self.retained_object_ids)) != len(self.retained_object_ids):
            raise ValueError(_IDENTITY_RETAINED_IDS_ERROR)
        return self


class StableIdentityCertificateClaimsV1(FrozenModel):
    object_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    source_video_id: str = Field(min_length=1)
    frame_ref: str = Field(min_length=1)
    place_id: str = Field(min_length=1)
    identity_index_sha256: str = Field(pattern=_SHA256)
    identity_scope: str = Field(min_length=1)


class StableIdentityCertificateV1(StableIdentityCertificateClaimsV1):
    """Closed, manifest-bound certificate for a stable T1 place identity."""

    certificate: SealedPayloadRefV1[Literal["stable-identity-certificate-v1"]]
    identity_index_attestation: SignedAttestationEnvelopeV1


type SemanticObservationDecision = Literal["objects", "zero_object", "abstained"]


class SemanticObservationOutcomeV1(FrozenModel):
    observation_id: str = Field(min_length=1)
    source_video_id: str = Field(min_length=1)
    frame_ref: str = Field(min_length=1)
    decision: SemanticObservationDecision
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def _closed_decision(self) -> Self:
        if (self.decision == "abstained") != (self.abstention_reason is not None):
            message = (
                "semantic abstention requires a reason and other outcomes forbid one"
            )
            raise ValueError(message)
        return self


class PlaceGroundingAssignmentV1(FrozenModel):
    object_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    source_video_id: str = Field(min_length=1)
    place_id: str = Field(min_length=1)
    mode: PlaceMode
    frame_ref: str = Field(min_length=1)
    identity_certificate: StableIdentityCertificateV1 | None = None

    @model_validator(mode="after")
    def _certificate_matches_mode(self) -> Self:
        if self.mode == "stable_identity" and self.identity_certificate is None:
            msg = _STABLE_IDENTITY_ERROR
            raise ValueError(msg)
        if self.mode == "frame_bound" and self.identity_certificate is not None:
            msg = _FRAME_BOUND_ERROR
            raise ValueError(msg)
        return self


class GeometrySelectionOutcomeV1(FrozenModel):
    object_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    source_video_id: str = Field(min_length=1)
    decision: GeometryDecision
    selected_points: SelectedPointPayloadRefV1 | None = None
    selected_indices: SelectedIndexPayloadRefV1 | None = None
    source_pointmap: SourcePointmapPayloadRefV1 | None = None
    abstention_reason: GeometryAbstentionReason | None = None
    provider_status: (
        Literal["completed", "unavailable", "rejected", "invalid"] | None
    ) = None
    response_sha256: str | None = Field(default=None, pattern=_SHA256)
    evidence_count: int | None = Field(default=None, ge=0)
    valid_evidence_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _exclusive_outcome(self) -> Self:
        provider_status = self.provider_status
        response_sha256 = self.response_sha256
        evidence_count = self.evidence_count
        valid_evidence_count = self.valid_evidence_count
        evidence = (
            provider_status,
            response_sha256,
            evidence_count,
            valid_evidence_count,
        )
        if self.decision == "selected":
            if (
                self.selected_points is None
                or self.selected_indices is None
                or self.source_pointmap is None
                or self.abstention_reason is not None
            ):
                msg = _SELECTED_GEOMETRY_ERROR
                raise ValueError(msg)
            if any(item is not None for item in evidence):
                msg = "selected geometry must not carry abstention evidence"
                raise ValueError(msg)
            return self
        if (
            self.selected_points is not None
            or self.selected_indices is not None
            or self.source_pointmap is not None
            or self.abstention_reason is None
        ):
            msg = _ABSTAINED_GEOMETRY_ERROR
            raise ValueError(msg)
        if (
            provider_status is None
            or response_sha256 is None
            or evidence_count is None
            or valid_evidence_count is None
        ):
            msg = _ABSTAINED_GEOMETRY_ERROR
            raise ValueError(msg)
        if valid_evidence_count > evidence_count:
            msg = "abstention valid evidence cannot exceed evidence count"
            raise ValueError(msg)
        if provider_status != "completed":
            msg = _ABSTAINED_GEOMETRY_ERROR
            raise ValueError(msg)
        return self


class CompiledOracleAssignmentsV1(FrozenModel):
    """Digest-bound, exact assignment set accepted by oracle materialization."""

    manifest_sha256: str = Field(pattern=_SHA256)
    assignment_sha256: str = Field(pattern=_SHA256)
    semantics: tuple[SharedObjectSemanticAssignmentV1, ...] = ()
    semantic_outcomes: tuple[SemanticObservationOutcomeV1, ...] = ()
    outcomes: tuple[GeometrySelectionOutcomeV1, ...] = ()
    places: tuple[PlaceGroundingAssignmentV1, ...] = ()

    @model_validator(mode="after")
    def _verify_assignment_digest(self) -> Self:
        payload = {
            "semantics": [item.model_dump(mode="json") for item in self.semantics],
            "semantic_outcomes": [
                item.model_dump(mode="json") for item in self.semantic_outcomes
            ],
            "outcomes": [item.model_dump(mode="json") for item in self.outcomes],
            "places": [item.model_dump(mode="json") for item in self.places],
        }
        if self.assignment_sha256 != canonical_sha256(payload):
            msg = _ASSIGNMENT_DIGEST_ERROR
            raise ValueError(msg)
        return self


class ApprovedCompiledAssignmentsArtifactV1(FrozenModel):
    """Approval artifact whose sealed canonical bytes are the only assignment source."""

    manifest_sha256: str = Field(pattern=_SHA256)
    artifact: SealedPayloadRefV1[Literal["compiled-oracle-assignments-v1"]]

    @model_validator(mode="after")
    def _artifact_binds_manifest(self) -> Self:
        if self.artifact.manifest_sha256 != self.manifest_sha256:
            msg = "compiled-assignment artifact does not bind its oracle manifest"
            raise ValueError(msg)
        return self


def load_approved_compiled_assignments(
    manifest: TeacherOracleInputManifestV1,
    artifact: ApprovedCompiledAssignmentsArtifactV1,
    resolver: SealedManifestPayloadResolver,
) -> CompiledOracleAssignmentsV1:
    """Read assignment bytes only from the concrete sealed resolver."""
    manifest_sha256 = canonical_sha256(manifest.model_dump(mode="json"))
    if artifact.manifest_sha256 != manifest_sha256:
        raise TeacherOracleContractError(
            detail="approved compiled-assignment artifact does not bind oracle manifest"
        )
    _require_sealed_descriptor(artifact.artifact, manifest, "compiled assignments")
    payload = resolver.resolve(artifact.artifact)
    try:
        text = payload.decode("utf-8")
        parsed = cast("JsonValue", json.loads(text))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeacherOracleContractError(
            detail="approved compiled-assignment artifact is not canonical JSON"
        ) from exc
    if _canonical_json(parsed) != text:
        raise TeacherOracleContractError(
            detail="approved compiled-assignment artifact bytes are not canonical JSON"
        )
    try:
        assignments = CompiledOracleAssignmentsV1.model_validate(parsed)
    except ValidationError as exc:
        raise TeacherOracleContractError(
            detail=(
                "approved compiled-assignment artifact does not use the closed schema"
            )
        ) from exc
    if assignments.manifest_sha256 != manifest_sha256:
        raise TeacherOracleContractError(
            detail="approved compiled assignments do not bind oracle manifest"
        )
    return assignments


class ProducerObservationV1(FrozenModel):
    observation_id: str = Field(min_length=1)
    source_video_id: str = Field(min_length=1)
    frame_ref: str = Field(min_length=1)
    local_frame_id: str = Field(min_length=1)
    timestamp_us: int = Field(ge=0)

    @property
    def timestamp(self) -> float:
        """Return the exact microsecond timestamp as seconds."""
        return self.timestamp_us / 1_000_000


class TeacherProducerInputV1(FrozenModel):
    """Closed, label-blind producer contract accepted before teacher execution."""

    source_inventory: DigestRefV1[str]
    frame_inventory: FrameInventoryV1
    observations: tuple[ProducerObservationV1, ...] = Field(min_length=1)


class CrossVariantFairnessClaimsV1(FrozenModel):
    """Shared, signed fairness invariants for the E0/T0/T1 comparison."""

    sealed_root_manifest_sha256: str = Field(pattern=_SHA256)
    source_inventory_sha256: str = Field(pattern=_SHA256)
    frame_inventory_sha256: str = Field(pattern=_SHA256)
    qa_inventory_sha256: str = Field(pattern=_SHA256)
    split_sha256: str = Field(pattern=_SHA256)
    byte_budget_per_window: int = Field(gt=0)
    window_microseconds: Literal[30_000_000] = WINDOW_MICROSECONDS
    cadence_origin_us: int = Field(ge=0)
    variants: tuple[Literal["E0", "T0", "T1"], ...] = ("E0", "T0", "T1")

    @model_validator(mode="after")
    def _all_variants_once(self) -> Self:
        if self.variants != ("E0", "T0", "T1"):
            message = "fairness contract variants must be exactly E0, T0, T1"
            raise ValueError(message)
        return self


class CrossVariantFairnessContractV1(FrozenModel):
    """Shared-attestation envelope plus the exact immutable fairness claims."""

    claims: CrossVariantFairnessClaimsV1
    attestation: SignedAttestationEnvelopeV1

    @model_validator(mode="after")
    def _attestation_binds_claims(self) -> Self:
        if self.attestation.purpose != "teacher_cache_production":
            message = "fairness contract must use shared production attestation"
            raise ValueError(message)
        if self.attestation.payload != self.claims.model_dump(mode="json"):
            message = "fairness attestation payload must exactly equal fairness claims"
            raise ValueError(message)
        return self


def cross_variant_fairness_contract_sha256(
    contract: CrossVariantFairnessContractV1,
) -> str:
    """Return the digest carried by every variant and output manifest."""
    return canonical_sha256(contract.claims.model_dump(mode="json"))


def validate_cross_variant_fairness_contract(
    contract: CrossVariantFairnessContractV1,
    manifests: Sequence[TeacherOracleInputManifestV1],
    *,
    authorized_registry: ImmutableAttestationKeyRegistryV1,
) -> None:
    """Verify one signed contract and exact E0/T0/T1 manifest parity."""
    try:
        verify_signed_attestation_envelope(
            contract.attestation,
            authorized_registry,
            purpose="teacher_cache_production",
        )
    except AttestationError as exc:
        raise TeacherOracleContractError(
            detail=f"fairness contract attestation rejected: {exc}"
        ) from exc
    if len(manifests) != _ORACLE_VARIANT_COUNT or {
        manifest.variant for manifest in manifests
    } != {
        "E0",
        "T0",
        "T1",
    }:
        raise TeacherOracleContractError(
            detail="fairness contract requires exactly one E0, T0, and T1 manifest"
        )
    claims = contract.claims
    digest = cross_variant_fairness_contract_sha256(contract)
    for manifest in manifests:
        comparisons = (
            (
                "sealed root",
                manifest.sealed_root_manifest_sha256,
                claims.sealed_root_manifest_sha256,
            ),
            (
                "source inventory",
                manifest.source_inventory.sha256,
                claims.source_inventory_sha256,
            ),
            (
                "frame inventory",
                manifest.frame_inventory.sha256,
                claims.frame_inventory_sha256,
            ),
            (
                "byte cap",
                manifest.byte_budget_per_window,
                claims.byte_budget_per_window,
            ),
            ("window", manifest.window_microseconds, claims.window_microseconds),
            ("cadence origin", manifest.cadence_origin_us, claims.cadence_origin_us),
            ("fairness digest", manifest.fairness_contract_sha256, digest),
        )
        mismatch = next(
            (name for name, actual, expected in comparisons if actual != expected), None
        )
        if mismatch is not None:
            raise TeacherOracleContractError(
                detail=f"{manifest.variant} fairness contract {mismatch} mismatch"
            )


def validate_request_manifest_fairness_trio(
    manifest: TeacherOracleInputManifestV1,
    contract: CrossVariantFairnessContractV1,
    manifests: Sequence[TeacherOracleInputManifestV1],
    *,
    authorized_registry: ImmutableAttestationKeyRegistryV1,
) -> None:
    """Require this exact request manifest in the signed E0/T0/T1 comparison set."""
    validate_cross_variant_fairness_contract(
        contract,
        manifests,
        authorized_registry=authorized_registry,
    )
    same_variant = tuple(item for item in manifests if item.variant == manifest.variant)
    manifest_digest = canonical_sha256(manifest.model_dump(mode="json"))
    if (
        len(same_variant) != 1
        or canonical_sha256(same_variant[0].model_dump(mode="json")) != manifest_digest
    ):
        raise TeacherOracleContractError(
            detail=(
                "request oracle manifest must equal the unique same-variant "
                "manifest in the validated fairness trio"
            )
        )


class TeacherOracleInputManifestV1(FrozenModel):
    execution_profile: Literal["teacher-oracle"] = "teacher-oracle"
    variant: OracleVariant
    source_inventory: DigestRefV1[str]
    frame_inventory: FrameInventoryV1
    sealed_root_manifest_sha256: str = Field(pattern=_SHA256)
    fairness_contract_sha256: str = Field(pattern=_SHA256)
    provider_provenance: DigestRefV1[str] | None = None
    semantic_provider_provenance: DigestRefV1[str] | None = None
    ontology_sha256: str | None = Field(default=None, pattern=_SHA256)
    coordinate_contract: CoordinateFrameContractV1
    byte_budget_per_window: int = Field(gt=0)
    window_microseconds: Literal[30_000_000] = WINDOW_MICROSECONDS
    cadence_origin_us: int = Field(ge=0)

    @model_validator(mode="after")
    def _minimal_capabilities(self) -> Self:
        if self.variant == "E0" and (
            self.semantic_provider_provenance is None or self.ontology_sha256 is None
        ):
            msg = _SEMANTIC_PROVENANCE_ERROR
            raise ValueError(msg)
        if self.variant in {"T0", "T1"} and self.provider_provenance is None:
            msg = _TEACHER_PROVENANCE_ERROR
            raise ValueError(msg)
        if self.variant in {"T0", "T1"} and (
            self.semantic_provider_provenance is None or self.ontology_sha256 is None
        ):
            msg = _SEMANTIC_PROVENANCE_ERROR
            raise ValueError(msg)
        return self


class TeacherOracleOutputManifestV1(FrozenModel):
    execution_profile: Literal["teacher-oracle"] = "teacher-oracle"
    result_class: Literal["teacher_oracle"] = RESULT_CLASS
    variant: OracleVariant
    input_manifest_sha256: str = Field(pattern=_SHA256)
    sealed_root_manifest_sha256: str = Field(pattern=_SHA256)
    fairness_contract_sha256: str = Field(pattern=_SHA256)
    typed_memory: DigestRefV1[str]
    actual_bytes: int = Field(ge=0)
    max_window_bytes: int = Field(ge=0)
    window_microseconds: Literal[30_000_000] = WINDOW_MICROSECONDS
    candidate_score: Literal["score_per_actual_serialized_byte"] = (
        CANDIDATE_SCORE_LITERAL
    )
    normalization: Literal["actual_serialized_bytes"] = NORMALIZATION_LITERAL
    candidate_count: int = Field(default=0, ge=0)
    writable_candidate_count: int = Field(default=0, ge=0)
    selected_count: int = Field(default=0, ge=0)
    skipped_for_budget_count: int = Field(default=0, ge=0)
    approved_assignment_artifact: SealedPayloadRefV1[
        Literal["compiled-oracle-assignments-v1"]
    ]
    approved_outcome_artifact: SealedPayloadRefV1[
        Literal["compiled-oracle-assignments-v1"]
    ]
    semantic_denominator: int = Field(ge=0)
    abstention_counts: dict[GeometryAbstentionReason, int]
    place_completion_count: int = Field(ge=0)
    semantic_abstention_reasons: tuple[str, ...] = ()
    pre_budget_candidate_count: int = Field(ge=0)
    byte_drop_count: int = Field(ge=0)
    selected_memory_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _cap_is_honored(self) -> Self:
        if self.max_window_bytes > self.actual_bytes:
            msg = _MAX_WINDOW_BYTES_ERROR
            raise ValueError(msg)
        if (
            self.selected_count + self.skipped_for_budget_count
            > self.writable_candidate_count
        ):
            msg = "output completion counters exceed writable candidates"
            raise ValueError(msg)
        if self.selected_count != len(self.selected_memory_ids):
            msg = "output selected_count must equal selected_memory_ids length"
            raise ValueError(msg)
        return self


class EvaluatorContractV1(FrozenModel):
    result_class: Literal["teacher_oracle"] = RESULT_CLASS
    variant: OracleVariant
    input_manifest_sha256: str = Field(pattern=_SHA256)
    qa_inventory: DigestRefV1[str]
    metric_schema: DigestRefV1[str]
    label_blind_until_evaluator: Literal[True] = True


class ProviderGateResultV1(FrozenModel):
    decision: ProviderGateDecision
    operational_state: TerminalState
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _state_is_independent(self) -> Self:
        if self.decision == "go" and self.operational_state != "completed":
            msg = _PROVIDER_GATE_ERROR
            raise ValueError(msg)
        return self


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def validate_label_blind_payload(
    payload: TeacherProducerInputV1 | Mapping[str, object],
) -> TeacherProducerInputV1:
    """Validate the sole closed, provenance-bound producer input shape."""
    if isinstance(payload, TeacherProducerInputV1):
        candidate = cast(
            "Mapping[str, object]",
            payload.model_dump(mode="python", by_alias=True),
        )
    else:
        candidate = payload
    _validate_producer_input_shape(candidate, path="input")
    _validate_label_blind_value(candidate, path="input")
    try:
        return TeacherProducerInputV1.model_validate(candidate)
    except ValidationError as exc:
        detail = f"producer input is not allowlisted: {exc}"
        raise TeacherOracleContractError(detail=detail) from exc


def validate_producer_input_manifest(
    producer_input: TeacherProducerInputV1,
    manifest: TeacherOracleInputManifestV1,
) -> None:
    """Bind closed producer observations and inventories to one manifest."""
    producer_input = validate_label_blind_payload(producer_input)
    if producer_input.source_inventory != manifest.source_inventory:
        raise TeacherOracleContractError(
            detail="producer source inventory does not match oracle manifest",
        )
    if producer_input.frame_inventory != manifest.frame_inventory:
        raise TeacherOracleContractError(
            detail="producer frame inventory does not match oracle manifest",
        )
    expected = {
        (
            frame.observation_id,
            frame.source_video_id,
            frame.frame_ref,
            frame.local_frame_id,
            frame.timestamp_us,
        )
        for frame in manifest.frame_inventory.frames
    }
    observed = tuple(
        (
            item.observation_id,
            item.source_video_id,
            item.frame_ref,
            item.local_frame_id,
            item.timestamp_us,
        )
        for item in producer_input.observations
    )
    if len(set(observed)) != len(observed):
        raise TeacherOracleContractError(detail="producer observations must be unique")
    if set(observed) != expected or len(observed) != len(expected):
        raise TeacherOracleContractError(
            detail=(
                "producer observations do not exactly match approved frame inventory"
            ),
        )


def validate_oracle_assignments(
    manifest: TeacherOracleInputManifestV1,
    semantics: Sequence[SharedObjectSemanticAssignmentV1],
    semantic_outcomes: Sequence[SemanticObservationOutcomeV1],
    outcomes: Sequence[GeometrySelectionOutcomeV1],
    places: Sequence[PlaceGroundingAssignmentV1] = (),
) -> None:
    _validate_semantic_observation_outcomes(manifest, semantics, semantic_outcomes)
    _validate_assignment_key_sets(manifest, semantics, outcomes, places)
    approved = {frame.frame_ref: frame for frame in manifest.frame_inventory.frames}
    _validate_assignment_frames_and_ontology(
        manifest,
        semantics,
        outcomes,
        places,
        approved,
    )


def _validate_semantic_observation_outcomes(
    manifest: TeacherOracleInputManifestV1,
    semantics: Sequence[SharedObjectSemanticAssignmentV1],
    outcomes: Sequence[SemanticObservationOutcomeV1],
) -> None:
    approved = {
        (frame.observation_id, frame.source_video_id, frame.frame_ref)
        for frame in manifest.frame_inventory.frames
    }
    outcome_keys = [
        (item.observation_id, item.source_video_id, item.frame_ref) for item in outcomes
    ]
    if len(outcome_keys) != len(set(outcome_keys)) or set(outcome_keys) != approved:
        raise TeacherOracleContractError(
            detail=(
                "semantic outcomes must exactly and uniquely cover "
                "approved observations"
            )
        )
    semantic_frames = {
        (item.observation_id, item.source_video_id, item.mask.frame_ref)
        for item in semantics
    }
    for outcome in outcomes:
        key = (outcome.observation_id, outcome.source_video_id, outcome.frame_ref)
        has_objects = key in semantic_frames
        if (outcome.decision == "objects") != has_objects:
            raise TeacherOracleContractError(
                detail=(
                    "semantic outcome decision must exactly match semantic object joins"
                )
            )


def _validate_assignment_key_sets(
    manifest: TeacherOracleInputManifestV1,
    semantics: Sequence[SharedObjectSemanticAssignmentV1],
    outcomes: Sequence[GeometrySelectionOutcomeV1],
    places: Sequence[PlaceGroundingAssignmentV1],
) -> None:
    semantic_keys = [_assignment_key(item) for item in semantics]
    outcome_keys = [_assignment_key(item) for item in outcomes]
    if len(semantic_keys) != len(set(semantic_keys)):
        raise TeacherOracleContractError(
            detail="semantic composite object join must be unique",
        )
    if manifest.variant != "E0" and (
        len(outcome_keys) != len(set(outcome_keys))
        or set(outcome_keys) != set(semantic_keys)
    ):
        detail = "exactly one geometry outcome is required per semantic composite join"
        raise TeacherOracleContractError(detail=detail)
    _validate_assignment_variants(manifest, outcomes, places)
    place_keys = [_assignment_key(item) for item in places]
    if len(place_keys) != len(set(place_keys)) or not set(place_keys).issubset(
        set(semantic_keys)
    ):
        raise TeacherOracleContractError(
            detail="place grounding must uniquely reference a semantic composite join",
        )
    if manifest.variant == "T1" and set(place_keys) != set(semantic_keys):
        raise TeacherOracleContractError(
            detail=(
                "T1 requires exactly one validated place grounding per "
                "semantic composite join"
            ),
        )


def _validate_assignment_variants(
    manifest: TeacherOracleInputManifestV1,
    outcomes: Sequence[GeometrySelectionOutcomeV1],
    places: Sequence[PlaceGroundingAssignmentV1],
) -> None:
    if manifest.variant == "E0" and (outcomes or places):
        raise TeacherOracleContractError(detail="E0 forbids geometry/place assignments")
    if manifest.variant == "T0" and places:
        raise TeacherOracleContractError(
            detail="T0 must not contain place grounding",
        )
    if manifest.variant != "T1" and places:
        raise TeacherOracleContractError(detail=_PLACE_GROUNDING_T1_ERROR)


def _validate_assignment_frames_and_ontology(
    manifest: TeacherOracleInputManifestV1,
    semantics: Sequence[SharedObjectSemanticAssignmentV1],
    outcomes: Sequence[GeometrySelectionOutcomeV1],
    places: Sequence[PlaceGroundingAssignmentV1],
    approved: Mapping[str, ApprovedFrameRefV1],
) -> None:
    _validate_semantic_assignment_frames(manifest, semantics, approved)
    semantic_by_key = {_assignment_key(item): item for item in semantics}
    _validate_geometry_assignment_frames(manifest, outcomes, semantic_by_key, approved)
    _validate_place_assignment_frames(manifest, places, semantic_by_key, approved)


def _validate_semantic_assignment_frames(
    manifest: TeacherOracleInputManifestV1,
    semantics: Sequence[SharedObjectSemanticAssignmentV1],
    approved: Mapping[str, ApprovedFrameRefV1],
) -> None:
    for semantic in semantics:
        _require_approved_frame(approved, semantic.mask.frame_ref, semantic, "mask")
        if semantic.mask.object_id != semantic.object_id:
            raise TeacherOracleContractError(
                detail="mask payload object_id does not match semantic assignment"
            )
        if semantic.ontology_sha256 != manifest.ontology_sha256:
            raise TeacherOracleContractError(
                detail="semantic assignment ontology does not match manifest",
            )
        _require_sealed_descriptor(semantic.mask, manifest, "mask")


def _validate_geometry_assignment_frames(
    manifest: TeacherOracleInputManifestV1,
    outcomes: Sequence[GeometrySelectionOutcomeV1],
    semantic_by_key: Mapping[tuple[str, str, str], SharedObjectSemanticAssignmentV1],
    approved: Mapping[str, ApprovedFrameRefV1],
) -> None:
    for outcome in outcomes:
        if (
            outcome.selected_points is None
            or outcome.selected_indices is None
            or outcome.source_pointmap is None
        ):
            continue
        semantic = semantic_by_key[_assignment_key(outcome)]
        indices = outcome.selected_indices
        pointmap = outcome.source_pointmap
        if (
            outcome.selected_points.frame_ref != semantic.mask.frame_ref
            or indices.frame_ref != semantic.mask.frame_ref
            or pointmap.frame_ref != semantic.mask.frame_ref
        ):
            raise TeacherOracleContractError(
                detail=(
                    "mask, selected-index, and selected-point payloads must use "
                    "the identical frame"
                ),
            )
        if (
            outcome.selected_points.object_id != outcome.object_id
            or indices.object_id != outcome.object_id
            or pointmap.object_id != outcome.object_id
            or indices.point_count != outcome.selected_points.point_count
            or indices.mask_width_px != semantic.mask.width_px
            or indices.mask_height_px != semantic.mask.height_px
            or pointmap.width_px != semantic.mask.width_px
            or pointmap.height_px != semantic.mask.height_px
        ):
            raise TeacherOracleContractError(
                detail=(
                    "selected index descriptor does not bind mask dimensions and "
                    "selected points"
                ),
            )
        _validate_selected_point_coordinate_contract(manifest, outcome)
        _require_approved_frame(
            approved, outcome.selected_points.frame_ref, outcome, "selected points"
        )
        _require_sealed_descriptor(outcome.selected_points, manifest, "selected points")
        _require_sealed_descriptor(indices, manifest, "selected indices")
        _require_sealed_descriptor(pointmap, manifest, "source pointmap")


def _validate_selected_point_coordinate_contract(
    manifest: TeacherOracleInputManifestV1,
    outcome: GeometrySelectionOutcomeV1,
) -> None:
    selected_points = outcome.selected_points
    if selected_points is None:
        return
    contract = manifest.coordinate_contract
    if (
        selected_points.coordinate_frame_id != contract.frame_id
        or selected_points.coordinate_units != contract.units
        or selected_points.coordinate_min_xyz != contract.min_xyz
        or selected_points.coordinate_max_xyz != contract.max_xyz
    ):
        raise TeacherOracleContractError(
            detail=(
                "selected-point coordinate contract does not exactly "
                "match signed manifest"
            )
        )


def _validate_place_assignment_frames(
    manifest: TeacherOracleInputManifestV1,
    places: Sequence[PlaceGroundingAssignmentV1],
    semantic_by_key: Mapping[tuple[str, str, str], SharedObjectSemanticAssignmentV1],
    approved: Mapping[str, ApprovedFrameRefV1],
) -> None:
    for place in places:
        semantic = semantic_by_key[_assignment_key(place)]
        if place.frame_ref != semantic.mask.frame_ref:
            raise TeacherOracleContractError(
                detail="mask and place assignments must use the identical frame",
            )
        _require_approved_frame(approved, place.frame_ref, place, "place")
        _validate_stable_identity_assignment(manifest, place, semantic)


def _validate_stable_identity_assignment(
    manifest: TeacherOracleInputManifestV1,
    place: PlaceGroundingAssignmentV1,
    semantic: SharedObjectSemanticAssignmentV1,
) -> None:
    if place.mode != "stable_identity":
        return
    certificate = place.identity_certificate
    if certificate is None:
        raise TeacherOracleContractError(detail=_STABLE_IDENTITY_ERROR)
    if certificate.frame_ref != semantic.mask.frame_ref:
        raise TeacherOracleContractError(
            detail=(
                "mask, place, and identity certificate must use the identical frame"
            ),
        )
    _require_sealed_descriptor(
        certificate.certificate,
        manifest,
        "stable identity certificate",
    )
    if (
        certificate.object_id,
        certificate.observation_id,
        certificate.source_video_id,
        certificate.frame_ref,
        certificate.place_id,
    ) != (
        place.object_id,
        place.observation_id,
        place.source_video_id,
        place.frame_ref,
        place.place_id,
    ):
        raise TeacherOracleContractError(
            detail="stable identity certificate does not match place assignment"
        )


@dataclass(frozen=True, slots=True)
class MaterializeVariantMemoryContext:
    sealed_payload_resolver: SealedPayloadResolver | None = None
    identity_registry: ImmutableAttestationKeyRegistryV1 | None = None


def validate_variant_memory_candidates(
    manifest: TeacherOracleInputManifestV1,
    assignments: CompiledOracleAssignmentsV1,
    candidates: Sequence[ScoredMemoryCandidate],
    *,
    context: MaterializeVariantMemoryContext | None = None,
) -> None:
    """Validate candidates against the exact compiled assignment set before writing."""
    if context is None:
        context = MaterializeVariantMemoryContext()
    manifest_digest = canonical_sha256(manifest.model_dump(mode="json"))
    if assignments.manifest_sha256 != manifest_digest:
        raise TeacherOracleContractError(
            detail="compiled assignments do not bind the oracle manifest"
        )
    validate_oracle_assignments(
        manifest,
        assignments.semantics,
        assignments.semantic_outcomes,
        assignments.outcomes,
        assignments.places,
    )
    approved = {frame.frame_ref: frame for frame in manifest.frame_inventory.frames}
    if manifest.variant == "E0":
        _validate_e0_candidates(
            manifest,
            assignments,
            candidates,
            approved,
            context.sealed_payload_resolver,
        )
    else:
        _validate_teacher_candidates(
            _TeacherCandidateValidationContext(
                manifest=manifest,
                assignments=assignments,
                approved=approved,
                sealed_payload_resolver=context.sealed_payload_resolver,
                identity_registry=context.identity_registry,
            ),
            candidates,
        )
    if isinstance(context.sealed_payload_resolver, SealedManifestPayloadResolver):
        _validate_approval_bound_candidate_scores(assignments, candidates)


def materialize_variant_memory(
    manifest: TeacherOracleInputManifestV1,
    assignments: CompiledOracleAssignmentsV1,
    candidates: Sequence[ScoredMemoryCandidate],
    output_path: Path,
    *,
    context: MaterializeVariantMemoryContext | None = None,
) -> TypedMemoryWriteSummary:
    """Persist only digest-bound, fully linked records for one oracle variant."""
    if context is None:
        context = MaterializeVariantMemoryContext()
    validate_variant_memory_candidates(
        manifest,
        assignments,
        candidates,
        context=context,
    )
    summary = write_typed_memory_artifact(
        candidates,
        output=output_path,
        byte_budget=manifest.byte_budget_per_window,
        options=TypedMemoryWriteOptions(
            budget_scope="per_source_window",
            window_microseconds=manifest.window_microseconds,
            cadence_origin_us=manifest.cadence_origin_us,
            candidate_timestamps_us=_candidate_timestamps_us(
                manifest, assignments, candidates
            ),
        ),
    )
    artifact_summary = validate_typed_memory_artifact(
        output_path,
        TypedMemoryArtifactValidationOptions(
            byte_budget_per_window=manifest.byte_budget_per_window,
            window_microseconds=manifest.window_microseconds,
            cadence_origin_us=manifest.cadence_origin_us,
            candidate_timestamps_us=_candidate_timestamps_us(
                manifest, assignments, candidates
            ),
        ),
    )
    if artifact_summary.persisted_memory_ids != summary.selected_memory_ids:
        raise TeacherOracleContractError(
            detail="persisted typed-memory IDs do not match writer selection"
        )
    if (
        manifest.variant == "E0"
        and summary.selected_count == 0
        and assignments.semantics
    ):
        raise TeacherOracleContractError(
            detail=(
                "E0 object semantics require a nonempty frozen object-presence baseline"
            )
        )
    return summary


def _validate_approval_bound_candidate_scores(
    assignments: CompiledOracleAssignmentsV1,
    candidates: Sequence[ScoredMemoryCandidate],
) -> None:
    """Forbid caller-controlled priority at the materialization boundary."""
    semantics = {_assignment_key(item): item for item in assignments.semantics}
    for candidate in candidates:
        record = candidate.record
        if isinstance(record, ObjectPresenceMemoryRecord):
            key = (record.memory_id, record.observation_id, record.source_video_id)
        elif isinstance(record, ObjectMemoryRecord):
            observation_id = record.oracle_observation_id
            if observation_id is None:
                raise TeacherOracleContractError(
                    detail="oracle object candidate requires an observation ID"
                )
            key = (
                record.entity_id,
                observation_id,
                record.source_video_id,
            )
        else:
            continue
        semantic = semantics.get(key)
        if semantic is None:
            continue
        expected = semantic.semantic_confidence / serialized_byte_cost(record)
        if candidate.score != expected:
            raise TeacherOracleContractError(
                detail="candidate score does not derive from approved semantic priority"
            )


def _candidate_timestamps_us(
    manifest: TeacherOracleInputManifestV1,
    assignments: CompiledOracleAssignmentsV1,
    candidates: Sequence[ScoredMemoryCandidate],
) -> dict[str, int]:
    approved = {
        frame.frame_ref: frame.timestamp_us for frame in manifest.frame_inventory.frames
    }
    semantics = {_assignment_key(item): item for item in assignments.semantics}
    timestamps: dict[str, int] = {}
    for candidate in candidates:
        record = candidate.record
        if isinstance(record, ObjectPresenceMemoryRecord):
            timestamps[record.memory_id] = record.timestamp_us
            continue
        if not isinstance(record, ObjectMemoryRecord):
            raise TeacherOracleContractError(
                detail="candidate must be an object-presence or object memory record"
            )
        matches = [
            semantic
            for semantic in semantics.values()
            if semantic.object_id == record.entity_id
            and semantic.observation_id == record.oracle_observation_id
            and semantic.source_video_id == record.source_video_id
        ]
        if len(matches) != 1:
            raise TeacherOracleContractError(
                detail="candidate does not uniquely bind an approved integer timestamp"
            )
        timestamps[record.memory_id] = approved[matches[0].mask.frame_ref]
    return timestamps


def _validate_e0_candidates(
    manifest: TeacherOracleInputManifestV1,
    assignments: CompiledOracleAssignmentsV1,
    candidates: Sequence[ScoredMemoryCandidate],
    approved: Mapping[str, ApprovedFrameRefV1],
    sealed_payload_resolver: SealedPayloadResolver | None,
) -> None:
    if not candidates and assignments.semantics:
        raise TeacherOracleContractError(
            detail="E0 object semantics require object-presence baseline records"
        )
    if not candidates:
        return
    expected = {_assignment_key(item) for item in assignments.semantics}
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        record = candidate.record
        if not isinstance(record, ObjectPresenceMemoryRecord):
            raise TeacherOracleContractError(
                detail="E0 permits only object_presence_v1 records"
            )
        if record.source_inventory_sha256 != manifest.source_inventory.sha256:
            raise TeacherOracleContractError(
                detail="E0 object-presence record does not bind source inventory digest"
            )
        key = (record.memory_id, record.observation_id, record.source_video_id)
        if key not in expected or key in seen:
            raise TeacherOracleContractError(
                detail="E0 records must exactly and uniquely cover semantic objects"
            )
        semantic = next(
            item for item in assignments.semantics if _assignment_key(item) == key
        )
        frame = approved[semantic.mask.frame_ref]
        expected_record = {
            "source_video_id": semantic.source_video_id,
            "observation_id": semantic.observation_id,
            "frame_ref": semantic.mask.frame_ref,
            "timestamp_us": frame.timestamp_us,
            "semantic_class": semantic.semantic_class,
            "semantic_confidence": semantic.semantic_confidence,
            "semantic_provider_id": semantic.semantic_provider_id,
            "ontology_sha256": semantic.ontology_sha256,
            "mask_sha256": semantic.mask.sha256,
            "mask_schema_id": semantic.mask.schema_id,
            "mask_sealed_root_sha256": semantic.mask.sealed_root_sha256,
            "mask_manifest_sha256": semantic.mask.manifest_sha256,
            "mask_width_px": semantic.mask.width_px,
            "mask_height_px": semantic.mask.height_px,
            "mask_dtype": semantic.mask.dtype,
            "source_inventory_sha256": manifest.source_inventory.sha256,
        }
        actual_record = {name: getattr(record, name) for name in expected_record}
        if actual_record != expected_record:
            raise TeacherOracleContractError(
                detail=(
                    "E0 record must exactly preserve the shared "
                    "semantic/mask/frame assignment"
                )
            )
        if sealed_payload_resolver is not None:
            _ = sealed_payload_resolver.resolve(semantic.mask)
        seen.add(key)
    if seen != expected:
        raise TeacherOracleContractError(
            detail="E0 records must exactly cover every shared semantic object"
        )
    if not any(
        serialized_byte_cost(candidate.record) <= manifest.byte_budget_per_window
        for candidate in candidates
    ):
        raise TeacherOracleContractError(
            detail="E0 byte budget cannot materialize a nonempty baseline record"
        )


@dataclass(frozen=True, slots=True)
class _TeacherCandidateValidationContext:
    manifest: TeacherOracleInputManifestV1
    assignments: CompiledOracleAssignmentsV1
    approved: Mapping[str, ApprovedFrameRefV1]
    sealed_payload_resolver: SealedPayloadResolver | None
    identity_registry: ImmutableAttestationKeyRegistryV1 | None


def _validate_teacher_candidates(
    context: _TeacherCandidateValidationContext,
    candidates: Sequence[ScoredMemoryCandidate],
) -> None:
    manifest = context.manifest
    assignments = context.assignments
    approved = context.approved
    sealed_payload_resolver = context.sealed_payload_resolver
    semantic_by_key = {_assignment_key(item): item for item in assignments.semantics}
    outcome_by_key = {_assignment_key(item): item for item in assignments.outcomes}
    candidate_keys: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        record = candidate.record
        if not isinstance(record, ObjectMemoryRecord):
            raise TeacherOracleContractError(
                detail=f"{manifest.variant} permits only object records"
            )
        matching = [
            (key, semantic)
            for key, semantic in semantic_by_key.items()
            if semantic.object_id == record.entity_id
            and semantic.observation_id == record.oracle_observation_id
            and semantic.source_video_id == record.source_video_id
            and semantic.semantic_class == record.semantic_label
            and semantic.object_id == record.instance_id
        ]
        if len(matching) != 1:
            raise TeacherOracleContractError(
                detail=(
                    "object record does not uniquely link a compiled semantic "
                    "assignment"
                )
            )
        key, semantic = matching[0]
        if key in candidate_keys:
            raise TeacherOracleContractError(
                detail="object records must uniquely cover compiled semantic joins"
            )
        candidate_keys.add(key)
        frame = approved[semantic.mask.frame_ref]
        if record.local_frame_id != frame.local_frame_id:
            raise TeacherOracleContractError(
                detail="object record local_frame_id does not match compiled assignment"
            )
        if (
            record.first_seen_time != frame.timestamp
            or record.last_seen_time != frame.timestamp
            or record.validity.start_time != frame.timestamp
            or record.validity.end_time != frame.timestamp
        ):
            raise TeacherOracleContractError(
                detail=(
                    "object record time and validity must match "
                    "approved frame timestamp"
                )
            )
        outcome = outcome_by_key[key]
        if (
            outcome.decision != "selected"
            or outcome.selected_points is None
            or record.oracle_assignment_sha256 != assignments.assignment_sha256
            or record.selected_payload_sha256 != outcome.selected_points.sha256
        ):
            raise TeacherOracleContractError(
                detail=(
                    "object geometry must bind the compiled assignment and "
                    "sealed selected-point payload"
                )
            )
        _validate_record_place_label(context, record, key)
        if sealed_payload_resolver is None:
            raise TeacherOracleContractError(
                detail=(
                    "production teacher materialization requires "
                    "sealed payload verification"
                )
            )
        indices = outcome.selected_indices
        if indices is None or outcome.source_pointmap is None:
            raise TeacherOracleContractError(
                detail=(
                    "selected geometry requires sealed index and "
                    "source pointmap descriptors"
                )
            )
        proof = _SelectedGeometryProof(
            mask=semantic.mask,
            points=outcome.selected_points,
            indices=indices,
            pointmap=outcome.source_pointmap,
            mask_bytes=sealed_payload_resolver.resolve(semantic.mask),
            points_bytes=sealed_payload_resolver.resolve(outcome.selected_points),
            pointmap_bytes=sealed_payload_resolver.resolve(outcome.source_pointmap),
            index_bytes=sealed_payload_resolver.resolve(indices),
        )
        _validate_selected_geometry_proof(proof)
        validate_record_geometry = cast(
            "_OracleRecordGeometryValidator",
            vars(import_module("worldmm_smvqa.worldmm.spatial_teacher_targets"))[
                "validate_oracle_record_geometry"
            ],
        )

        validate_record_geometry(
            record,
            selected_points=outcome.selected_points,
            selected_points_bytes=proof.points_bytes,
        )
    selected_keys = {
        _assignment_key(outcome)
        for outcome in assignments.outcomes
        if outcome.decision == "selected"
    }
    if candidate_keys != selected_keys:
        raise TeacherOracleContractError(
            detail=(
                "selected geometry requires exactly one candidate and abstained "
                "geometry none"
            )
        )


@dataclass(frozen=True, slots=True)
class _SelectedGeometryProof:
    mask: MaskPayloadRefV1
    points: SelectedPointPayloadRefV1
    pointmap: SourcePointmapPayloadRefV1
    indices: SelectedIndexPayloadRefV1
    mask_bytes: bytes
    points_bytes: bytes
    pointmap_bytes: bytes
    index_bytes: bytes


def _validate_selected_geometry_proof(proof: _SelectedGeometryProof) -> None:
    """Bind canonical uint32 mask indices exactly to selected f32x3 point rows."""
    pixel_count = proof.mask.width_px * proof.mask.height_px
    if (
        len(proof.mask_bytes) != pixel_count
        or len(proof.points_bytes) != proof.points.point_count * 12
        or len(proof.pointmap_bytes) != pixel_count * 12
    ):
        raise TeacherOracleContractError(
            detail=(
                "sealed geometry payload dimensions do not match approved descriptors"
            )
        )
    if len(proof.index_bytes) != proof.indices.point_count * 4:
        raise TeacherOracleContractError(
            detail="selected-index payload size does not match selected point count"
        )
    values = cast(
        "tuple[int, ...]",
        struct.unpack(f"<{proof.indices.point_count}I", proof.index_bytes),
    )
    if len(set(values)) != len(values) or any(value >= pixel_count for value in values):
        raise TeacherOracleContractError(
            detail=(
                "selected-index payload must contain unique in-bounds integer indices"
            )
        )
    if any(proof.mask_bytes[value] == 0 for value in values):
        raise TeacherOracleContractError(
            detail="selected-index payload contains an index outside the approved mask"
        )
    selected = struct.unpack(f"<{proof.points.point_count * 3}f", proof.points_bytes)
    source = struct.unpack(f"<{pixel_count * 3}f", proof.pointmap_bytes)
    for row, index in enumerate(values):
        if selected[row * 3 : row * 3 + 3] != source[index * 3 : index * 3 + 3]:
            raise TeacherOracleContractError(
                detail=(
                    "selected point row must exactly derive from source pointmap index"
                )
            )


def _validate_record_place_label(
    context: _TeacherCandidateValidationContext,
    record: ObjectMemoryRecord,
    key: tuple[str, str, str],
) -> None:
    manifest = context.manifest
    assignments = context.assignments
    sealed_payload_resolver = context.sealed_payload_resolver
    identity_registry = context.identity_registry
    if manifest.variant == "T0" and record.place_label is not None:
        raise TeacherOracleContractError(
            detail="T0 object records must not include place_label"
        )
    if manifest.variant == "T1":
        place = next(
            item for item in assignments.places if _assignment_key(item) == key
        )
        if record.place_label != place.place_id:
            raise TeacherOracleContractError(
                detail="T1 object record place_label does not match compiled assignment"
            )
        if place.mode == "stable_identity":
            if sealed_payload_resolver is None:
                raise TeacherOracleContractError(
                    detail="stable identity requires sealed certificate verification"
                )
            certificate = place.identity_certificate
            if certificate is None:
                raise TeacherOracleContractError(detail=_STABLE_IDENTITY_ERROR)
            payload_bytes = sealed_payload_resolver.resolve(certificate.certificate)
            _validate_stable_identity_certificate(
                certificate,
                payload_bytes,
                observation_timestamp_us=context.approved[
                    certificate.frame_ref
                ].timestamp_us,
                identity_registry=identity_registry,
            )


def _validate_stable_identity_certificate(
    certificate: StableIdentityCertificateV1,
    payload_bytes: bytes,
    *,
    observation_timestamp_us: int,
    identity_registry: ImmutableAttestationKeyRegistryV1 | None,
) -> None:
    if identity_registry is None:
        raise TeacherOracleContractError(
            detail="stable identity requires an authorized identity index registry"
        )
    if hashlib.sha256(payload_bytes).hexdigest() != certificate.certificate.sha256:
        raise TeacherOracleContractError(
            detail="stable identity certificate bytes do not match sealed digest"
        )
    try:
        claims = StableIdentityCertificateClaimsV1.model_validate_json(payload_bytes)
    except ValidationError as exc:
        raise TeacherOracleContractError(
            detail="stable identity certificate bytes do not use the closed schema"
        ) from exc
    expected = claims.model_dump(mode="json")
    actual = certificate.model_dump(
        mode="json",
        exclude={"certificate", "identity_index_attestation"},
    )
    if expected != actual:
        raise TeacherOracleContractError(
            detail="stable identity certificate bytes do not match assignment claims"
        )
    try:
        verify_signed_attestation_envelope(
            certificate.identity_index_attestation,
            identity_registry,
            purpose="identity_index",
        )
        identity = IdentityIndexAttestationClaimsV1.model_validate(
            certificate.identity_index_attestation.payload
        )
    except (AttestationError, ValidationError) as exc:
        raise TeacherOracleContractError(
            detail="stable identity index attestation is unauthorized or malformed"
        ) from exc
    if (
        identity.model_dump(mode="json")
        != certificate.identity_index_attestation.payload
        or identity.identity_index_sha256 != certificate.identity_index_sha256
        or identity.scope != certificate.identity_scope
        or identity.cutoff_us != observation_timestamp_us
        or certificate.object_id not in identity.retained_object_ids
    ):
        raise TeacherOracleContractError(
            detail=(
                "stable identity certificate is not retained by its "
                "signed identity index"
            )
        )


def build_output_manifest(
    manifest: TeacherOracleInputManifestV1,
    typed_memory: DigestRefV1[str],
    summary: TypedMemoryWriteSummary,
    *,
    assignments: CompiledOracleAssignmentsV1,
    approved_assignments: ApprovedCompiledAssignmentsArtifactV1,
) -> TeacherOracleOutputManifestV1:
    artifact = _read_regular_nofollow(summary.output_path)
    artifact_summary = validate_typed_memory_artifact(
        summary.output_path,
        TypedMemoryArtifactValidationOptions(
            byte_budget_per_window=manifest.byte_budget_per_window,
            window_microseconds=manifest.window_microseconds,
            cadence_origin_us=manifest.cadence_origin_us,
            candidate_timestamps_us=summary.selected_candidate_timestamps_us,
        ),
    )
    verified_artifact = _read_regular_nofollow(summary.output_path)
    if artifact != verified_artifact:
        raise TeacherOracleContractError(
            detail="typed-memory artifact changed during validation",
        )
    digest = hashlib.sha256(verified_artifact).hexdigest()
    if typed_memory.sha256 != digest:
        raise TeacherOracleContractError(
            detail="typed_memory sha256 must match written artifact bytes"
        )
    if summary.actual_bytes != len(artifact):
        raise TeacherOracleContractError(
            detail="writer summary bytes do not match written artifact bytes"
        )
    if artifact_summary.persisted_memory_ids != summary.selected_memory_ids:
        raise TeacherOracleContractError(
            detail="writer selected IDs do not match persisted typed-memory artifact"
        )
    validate_oracle_assignments(
        manifest,
        assignments.semantics,
        assignments.semantic_outcomes,
        assignments.outcomes,
        assignments.places,
    )
    return TeacherOracleOutputManifestV1(
        variant=manifest.variant,
        input_manifest_sha256=canonical_sha256(manifest.model_dump(mode="json")),
        sealed_root_manifest_sha256=manifest.sealed_root_manifest_sha256,
        fairness_contract_sha256=manifest.fairness_contract_sha256,
        typed_memory=typed_memory,
        actual_bytes=artifact_summary.actual_bytes,
        max_window_bytes=artifact_summary.max_window_bytes,
        approved_assignment_artifact=approved_assignments.artifact,
        approved_outcome_artifact=approved_assignments.artifact,
        semantic_denominator=len(assignments.semantic_outcomes),
        abstention_counts={
            reason: sum(
                outcome.abstention_reason == reason for outcome in assignments.outcomes
            )
            for reason in (
                "no_valid_depth",
                "insufficient_valid_mask_points",
                "provider_uncertainty",
                "outside_approved_bounds",
            )
        },
        semantic_abstention_reasons=tuple(
            sorted(
                outcome.abstention_reason
                for outcome in assignments.semantic_outcomes
                if outcome.decision == "abstained"
                and outcome.abstention_reason is not None
            )
        ),
        place_completion_count=len(assignments.places),
        pre_budget_candidate_count=summary.candidate_count,
        byte_drop_count=summary.skipped_for_budget_count,
        candidate_count=summary.candidate_count,
        writable_candidate_count=summary.writable_candidate_count,
        selected_count=summary.selected_count,
        selected_memory_ids=artifact_summary.persisted_memory_ids,
        skipped_for_budget_count=summary.skipped_for_budget_count,
    )


def _assignment_key(
    item: SharedObjectSemanticAssignmentV1
    | GeometrySelectionOutcomeV1
    | PlaceGroundingAssignmentV1,
) -> tuple[str, str, str]:
    return item.object_id, item.observation_id, item.source_video_id


def _require_approved_frame(
    approved: Mapping[str, ApprovedFrameRefV1],
    frame_ref: str,
    assignment: SharedObjectSemanticAssignmentV1
    | GeometrySelectionOutcomeV1
    | PlaceGroundingAssignmentV1,
    purpose: str,
) -> None:
    frame = approved.get(frame_ref)
    if frame is None:
        detail = (
            f"{purpose} frame_ref is not in approved frame inventory: " + f"{frame_ref}"
        )
        raise TeacherOracleContractError(detail=detail)
    if (frame.observation_id, frame.source_video_id) != (
        assignment.observation_id,
        assignment.source_video_id,
    ):
        detail = (
            f"{purpose} frame_ref does not match "
            "object/observation/video composite join"
        )
        raise TeacherOracleContractError(detail=detail)


def _require_sealed_descriptor(
    descriptor: SealedPayloadRefV1[SchemaId_co],
    manifest: TeacherOracleInputManifestV1,
    purpose: str,
) -> None:
    if descriptor.manifest_sha256 != canonical_sha256(manifest.model_dump(mode="json")):
        raise TeacherOracleContractError(
            detail=f"{purpose} descriptor does not bind the oracle manifest"
        )
    if descriptor.sealed_root_sha256 != manifest.sealed_root_manifest_sha256:
        raise TeacherOracleContractError(
            detail=f"{purpose} descriptor does not bind the approved sealed root"
        )


def _validate_producer_input_shape(value: object, *, path: str) -> None:
    if not _is_object_mapping(value):
        raise TeacherOracleContractError(detail=f"{path} must be an object")
    _require_exact_fields(value, _PRODUCER_INPUT_FIELDS, path=path)
    _validate_digest_ref_shape(
        value["source_inventory"],
        path=f"{path}.source_inventory",
    )
    _validate_frame_inventory_shape(
        value["frame_inventory"],
        path=f"{path}.frame_inventory",
    )
    observations = value["observations"]
    if not _is_object_sequence(observations) or not observations:
        raise TeacherOracleContractError(
            detail=f"{path}.observations must be a non-empty declared sequence"
        )
    for index, observation in enumerate(observations):
        observation_path = f"{path}.observations[{index}]"
        if not _is_object_mapping(observation):
            raise TeacherOracleContractError(
                detail=f"{observation_path} must be an object"
            )
        _require_exact_fields(observation, _FRAME_FIELDS, path=observation_path)


def _validate_frame_inventory_shape(value: object, *, path: str) -> None:
    if not _is_object_mapping(value):
        raise TeacherOracleContractError(detail=f"{path} must be an object")
    _require_declared_fields(
        value,
        allowed=_FRAME_INVENTORY_FIELDS,
        required=frozenset({"uri", "sha256", "frames"}),
        path=path,
    )
    if ("schema" in value) == ("schema_id" in value):
        raise TeacherOracleContractError(
            detail=f"{path} must declare exactly one schema field"
        )
    frames = value["frames"]
    if not _is_object_sequence(frames) or not frames:
        raise TeacherOracleContractError(
            detail=f"{path}.frames must be a non-empty declared sequence"
        )
    for index, frame in enumerate(frames):
        frame_path = f"{path}.frames[{index}]"
        if not _is_object_mapping(frame):
            raise TeacherOracleContractError(detail=f"{frame_path} must be an object")
        _require_exact_fields(frame, _FRAME_FIELDS, path=frame_path)


def _validate_digest_ref_shape(value: object, *, path: str) -> None:
    if not _is_object_mapping(value):
        raise TeacherOracleContractError(detail=f"{path} must be an object")
    _require_declared_fields(
        value,
        allowed=_DIGEST_REF_FIELDS,
        required=frozenset({"uri", "sha256"}),
        path=path,
    )
    if ("schema" in value) == ("schema_id" in value):
        raise TeacherOracleContractError(
            detail=f"{path} must declare exactly one schema field"
        )


def _require_exact_fields(
    value: Mapping[object, object],
    allowed: frozenset[str],
    *,
    path: str,
) -> None:
    _require_declared_fields(
        value,
        allowed=allowed,
        required=allowed,
        path=path,
    )


def _require_declared_fields(
    value: Mapping[object, object],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    path: str,
) -> None:
    keys = set(value)
    unexpected = keys - allowed
    missing = required - keys
    if unexpected:
        rendered = ", ".join(sorted(repr(key) for key in unexpected))
        raise TeacherOracleContractError(
            detail=f"{path} contains undeclared input field(s): {rendered}"
        )
    if missing:
        rendered = ", ".join(sorted(missing))
        raise TeacherOracleContractError(
            detail=f"{path} is missing declared input field(s): {rendered}"
        )


def _validate_label_blind_value(value: object, *, path: str) -> None:
    if _is_object_mapping(value):
        for key, child in value.items():
            normalized = "".join(
                character for character in str(key).casefold() if character.isalnum()
            )
            if normalized in _DENIED_INPUT_KEYS:
                raise TeacherOracleContractError(
                    detail=f"label-blind input contains denied field: {path}.{key}"
                )
            child_path = f"{path}.{key}"
            if key == "uri":
                _validate_artifact_uri(child, path=child_path)
            else:
                _validate_label_blind_value(child, path=child_path)
    elif _is_object_sequence(value):
        for index, child in enumerate(value):
            _validate_label_blind_value(child, path=f"{path}[{index}]")
    elif isinstance(value, str):
        normalized = "".join(
            character for character in value.casefold() if character.isalnum()
        )
        if any(token in normalized for token in _LABEL_BEARING_VALUE_TOKENS):
            raise TeacherOracleContractError(
                detail=(
                    f"label-blind input contains prohibited label-bearing value: {path}"
                )
            )
        if (
            "/" in value
            or "\\" in value
            or "://" in value
            or value.startswith((".", "~"))
            or not _OPAQUE_IDENTIFIER_PATTERN.fullmatch(value)
        ):
            raise TeacherOracleContractError(
                detail=(
                    f"label-blind input contains unexpected path-bearing value: {path}"
                )
            )


def _validate_artifact_uri(value: object, *, path: str) -> None:
    if not isinstance(value, str) or not _ARTIFACT_URI_PATTERN.fullmatch(value):
        raise TeacherOracleContractError(
            detail=f"label-blind input contains unexpected provenance URI: {path}"
        )
    artifact_id = value.removeprefix("artifact://").casefold()
    if any(token in artifact_id for token in _LABEL_BEARING_VALUE_TOKENS):
        raise TeacherOracleContractError(
            detail=(
                "label-blind input contains prohibited label-bearing "
                f"provenance: {path}"
            )
        )


def _read_regular_nofollow(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise TeacherOracleContractError(
            detail=f"cannot open typed-memory artifact without following links: {exc}"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise TeacherOracleContractError(
                detail="typed-memory artifact must be a regular file"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _is_object_mapping(value: object) -> TypeIs[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_object_sequence(value: object) -> TypeIs[Sequence[object]]:
    return isinstance(value, (list, tuple))


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
