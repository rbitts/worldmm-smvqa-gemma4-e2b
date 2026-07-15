from __future__ import annotations

import base64
import hashlib
import json
import struct
from pathlib import Path
from typing import Literal, cast

import pytest
import rfc8785
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue, TypeAdapter, ValidationError

from worldmm_smvqa import openat2, teacher_oracle
from worldmm_smvqa.attestation import (
    ImmutableAttestationKeyRegistryV1,
    ImmutableAttestationKeyV1,
    SignedAttestationEnvelopeV1,
    key_id_from_public_key,
    signing_bytes,
    with_payload_sha256,
)
from worldmm_smvqa.teacher_oracle import (
    WINDOW_MICROSECONDS,
    ApprovedFrameRefV1,
    CompiledOracleAssignmentsV1,
    CoordinateFrameContractV1,
    CrossVariantFairnessClaimsV1,
    CrossVariantFairnessContractV1,
    DigestRefV1,
    FrameInventoryV1,
    GeometryAbstentionReason,
    GeometrySelectionOutcomeV1,
    MaskPayloadRefV1,
    MaterializeVariantMemoryContext,
    OracleVariant,
    PlaceGroundingAssignmentV1,
    ProducerObservationV1,
    SealedManifestEntryV1,
    SealedManifestPayloadResolver,
    SealedPayloadRefV1,
    SealedRootManifestBindingV1,
    SealedRootManifestV1,
    SelectedIndexPayloadRefV1,
    SelectedPointPayloadRefV1,
    SemanticObservationOutcomeV1,
    SharedObjectSemanticAssignmentV1,
    SourcePointmapPayloadRefV1,
    StableIdentityCertificateV1,
    TeacherOracleContractError,
    TeacherOracleInputManifestV1,
    TeacherOracleOutputManifestV1,
    TeacherProducerInputV1,
    canonical_sha256,
    cross_variant_fairness_contract_sha256,
    materialize_variant_memory,
    sealed_root_manifest_bytes,
    sealed_root_manifest_sha256,
    validate_cross_variant_fairness_contract,
    validate_label_blind_payload,
    validate_oracle_assignments,
    validate_producer_input_manifest,
    validate_request_manifest_fairness_trio,
)
from worldmm_smvqa.worldmm.gcut3r_teacher import (
    EMPTY_PREFIX_SHA256,
    CacheProductionClaimsV1,
    CacheProductionContractV1,
    LegacyNonProductionCacheProvenanceV1,
    LegacyNonProductionCacheSignerRegistryV1,
    LegacyNonProductionCacheSignerV1,
    TeacherCacheRecord,
    TeacherContractError,
    TeacherRequest,
    TeacherResponse,
    build_teacher_cache_record,
    validate_cache_production_attestation,
    validate_legacy_non_production_cache_provenance,
)
from worldmm_smvqa.worldmm.spatial_teacher_targets import (
    TeacherObjectTarget,
    compile_teacher_object_record,
)
from worldmm_smvqa.worldmm.typed_memory import (
    ObjectGeometry,
    ObjectMemoryRecord,
    ObjectPresenceMemoryRecord,
    ScoredMemoryCandidate,
    serialized_byte_cost,
)

DIGEST = hashlib.sha256(b"fixture").hexdigest()


def _ref(schema: str = "inventory-v1") -> DigestRefV1[str]:
    return DigestRefV1[str](uri="artifact://fixture", sha256=DIGEST, schema=schema)


def _frames() -> FrameInventoryV1:
    return FrameInventoryV1(
        uri="artifact://frames",
        sha256=DIGEST,
        schema="frame-inventory-v1",
        frames=(
            ApprovedFrameRefV1(
                frame_ref="frame-1",
                source_video_id="video-1",
                observation_id="obs-1",
                local_frame_id="local-1",
                timestamp_us=1_000_000,
            ),
        ),
    )


def _manifest(
    variant: OracleVariant = "T0", *, sealed_root_manifest_sha256: str = DIGEST
) -> TeacherOracleInputManifestV1:
    return TeacherOracleInputManifestV1(
        variant=variant,
        source_inventory=_ref(),
        frame_inventory=_frames(),
        sealed_root_manifest_sha256=sealed_root_manifest_sha256,
        fairness_contract_sha256=DIGEST,
        provider_provenance=_ref("provider-v1") if variant != "E0" else None,
        semantic_provider_provenance=_ref("semantic-provider-v1"),
        ontology_sha256=DIGEST,
        coordinate_contract=CoordinateFrameContractV1(
            frame_id="fixture-world",
            units="m",
            min_xyz=(-10.0, -10.0, -10.0),
            max_xyz=(10.0, 10.0, 10.0),
        ),
        byte_budget_per_window=4096,
        cadence_origin_us=0,
    )


def _compiled(
    manifest: TeacherOracleInputManifestV1,
    *,
    semantics: tuple[SharedObjectSemanticAssignmentV1, ...] = (),
    semantic_outcomes: tuple[SemanticObservationOutcomeV1, ...] = (),
    outcomes: tuple[GeometrySelectionOutcomeV1, ...] = (),
    places: tuple[PlaceGroundingAssignmentV1, ...] = (),
) -> CompiledOracleAssignmentsV1:
    effective_semantic_outcomes = semantic_outcomes or (
        _semantic_outcomes() if semantics else _semantic_outcomes("zero_object")
    )
    payload = {
        "semantics": [item.model_dump(mode="json") for item in semantics],
        "semantic_outcomes": [
            item.model_dump(mode="json") for item in effective_semantic_outcomes
        ],
        "outcomes": [item.model_dump(mode="json") for item in outcomes],
        "places": [item.model_dump(mode="json") for item in places],
    }
    return CompiledOracleAssignmentsV1(
        manifest_sha256=canonical_sha256(manifest.model_dump(mode="json")),
        assignment_sha256=canonical_sha256(payload),
        semantics=semantics,
        semantic_outcomes=effective_semantic_outcomes,
        outcomes=outcomes,
        places=places,
    )


def _object_presence(
    manifest: TeacherOracleInputManifestV1,
) -> ObjectPresenceMemoryRecord:
    return ObjectPresenceMemoryRecord(
        memory_id="object-1",
        source_video_id="video-1",
        observation_id="obs-1",
        frame_ref="frame-1",
        timestamp_us=1_000_000,
        semantic_class="mug",
        semantic_confidence=0.9,
        semantic_provider_id="semantic-v1",
        ontology_sha256=DIGEST,
        mask_sha256=_mask_digest(),
        mask_schema_id="mask-dense-u8-v1",
        mask_sealed_root_sha256=manifest.sealed_root_manifest_sha256,
        mask_manifest_sha256=canonical_sha256(manifest.model_dump(mode="json")),
        mask_width_px=640,
        mask_height_px=480,
        mask_dtype="bool",
        source_inventory_sha256=manifest.source_inventory.sha256,
    )


def _object_record(
    assignments: CompiledOracleAssignmentsV1,
) -> ObjectMemoryRecord:
    record = compile_teacher_object_record(
        TeacherObjectTarget(
            memory_id="t0-object-1",
            source_video_id="video-1",
            observation_id="obs-1",
            frame_ref="frame-1",
            local_frame_id="local-1",
            timestamp=1.0,
            observed_through_time=1.0,
            entity_id="object-1",
            instance_id="object-1",
            semantic_label="mug",
            points_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            confidence=1.0,
        )
    )
    return record.model_copy(
        update={
            "oracle_assignment_sha256": assignments.assignment_sha256,
            "selected_payload_sha256": _points_digest(),
        }
    )


def _cache_record(video_id: str) -> TeacherCacheRecord:
    request = TeacherRequest(
        observation_id=f"{video_id}:obs-1",
        video_id=video_id,
        timestamp=1.0,
        frame_ref=f"{video_id}:frame-1",
        local_frame_id="room-1",
        sequence_index=0,
        prefix_before_sha256=EMPTY_PREFIX_SHA256,
    )
    response = TeacherResponse(
        observation_id=request.observation_id,
        video_id=video_id,
        timestamp=1.0,
        observed_through_time=1.0,
        state_ref=f"{video_id}:state-1",
        frame_ref=request.frame_ref,
        local_frame_id=request.local_frame_id,
        prefix_before_sha256=EMPTY_PREFIX_SHA256,
    )
    return build_teacher_cache_record(
        teacher_backend="gcut3r_external",
        provider_id="fixture-provider",
        request=request,
        response=response,
    )


def _cache_content_sha256(records: tuple[TeacherCacheRecord, ...]) -> str:
    content = [
        {
            "teacher_backend": record.teacher_backend,
            "provider_id": record.provider_id,
            "request_sha256": record.request_sha256,
            "response_sha256": record.response_sha256,
        }
        for record in records
    ]
    return hashlib.sha256(
        json.dumps(
            content,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _signed_cache_provenance(
    records: tuple[TeacherCacheRecord, ...],
) -> tuple[
    LegacyNonProductionCacheProvenanceV1, LegacyNonProductionCacheSignerRegistryV1
]:
    private_key = Ed25519PrivateKey.generate()
    content_sha256 = _cache_content_sha256(records)
    payload = {
        "signer_id": "signer",
        "purpose": "teacher_cache_production",
        "issued_at": 1,
        "scope": "exp-0005",
        "source_inventory_sha256": DIGEST,
        "frame_inventory_sha256": DIGEST,
        "provider_artifact_sha256": DIGEST,
        "semantic_provider_artifact_sha256": DIGEST,
        "ontology_sha256": DIGEST,
        "cache_content_sha256": content_sha256,
        "oracle_manifest_sha256": DIGEST,
    }
    signature = private_key.sign(
        b"worldmm-smvqa/cache-production-provenance/v1\x00"
        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    provenance = LegacyNonProductionCacheProvenanceV1(
        signer_id="signer",
        signature_b64=base64.b64encode(signature).decode(),
        issued_at=1,
        scope="exp-0005",
        source_inventory_sha256=DIGEST,
        frame_inventory_sha256=DIGEST,
        provider_artifact_sha256=DIGEST,
        semantic_provider_artifact_sha256=DIGEST,
        ontology_sha256=DIGEST,
        oracle_manifest_sha256=DIGEST,
        cache_content_sha256=content_sha256,
    )
    registry = LegacyNonProductionCacheSignerRegistryV1(
        signers=(
            LegacyNonProductionCacheSignerV1(
                signer_id="signer",
                public_key_b64=base64.b64encode(
                    private_key.public_key().public_bytes(
                        serialization.Encoding.Raw,
                        serialization.PublicFormat.Raw,
                    )
                ).decode(),
                purposes=("teacher_cache_production",),
                not_before=0.0,
                not_after=2.0,
                allowed_scopes=("exp-0005",),
            ),
        )
    )
    return provenance, registry


def _cache_contract() -> CacheProductionContractV1:
    return CacheProductionContractV1(
        oracle_manifest_sha256=DIGEST,
        source_inventory_sha256=DIGEST,
        frame_inventory_sha256=DIGEST,
        provider_artifact_sha256=DIGEST,
        semantic_provider_artifact_sha256=DIGEST,
        ontology_sha256=DIGEST,
        provider_id="fixture-provider",
        scope="exp-0005",
    )


_IDENTITY_KEY = Ed25519PrivateKey.generate()


def _signed_envelope(
    private_key: Ed25519PrivateKey,
    *,
    purpose: str,
    payload: dict[str, object],
) -> SignedAttestationEnvelopeV1:
    normalized_payload = cast(
        "JsonValue",
        json.loads(rfc8785.dumps(cast("JsonValue", payload))),
    )
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    issued_at = 1.0
    unsigned = with_payload_sha256(
        {
            "version": "signed-attestation-envelope-v1",
            "key_id": key_id_from_public_key(public_key),
            "purpose": purpose,
            "payload": normalized_payload,
            "issued_at": issued_at,
        }
    )
    unsigned["signature_b64url"] = (
        base64.urlsafe_b64encode(private_key.sign(signing_bytes(unsigned, purpose)))
        .decode()
        .rstrip("=")
    )
    return SignedAttestationEnvelopeV1.model_validate(unsigned)


def _attestation_registry(
    private_key: Ed25519PrivateKey, *, purpose: str
) -> ImmutableAttestationKeyRegistryV1:
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return ImmutableAttestationKeyRegistryV1(
        keys=(
            ImmutableAttestationKeyV1(
                key_id=key_id_from_public_key(public_key),
                public_key_b64url=base64.urlsafe_b64encode(public_key)
                .decode()
                .rstrip("="),
                purposes=(purpose,),
                not_before=0.0,
                not_after=2.0,
            ),
        )
    )


def _identity_registry() -> ImmutableAttestationKeyRegistryV1:
    return _attestation_registry(_IDENTITY_KEY, purpose="identity_index")


def _semantic(
    manifest: TeacherOracleInputManifestV1,
) -> SharedObjectSemanticAssignmentV1:
    return SharedObjectSemanticAssignmentV1(
        object_id="object-1",
        observation_id="obs-1",
        source_video_id="video-1",
        semantic_class="mug",
        semantic_confidence=0.9,
        semantic_provider_id="semantic-v1",
        ontology_sha256=DIGEST,
        mask=MaskPayloadRefV1(
            uri="sealed://fixture/mask",
            sha256=_mask_digest(),
            schema="mask-dense-u8-v1",
            sealed_root_sha256=manifest.sealed_root_manifest_sha256,
            manifest_sha256=canonical_sha256(manifest.model_dump(mode="json")),
            width_px=640,
            height_px=480,
            dtype="bool",
            frame_ref="frame-1",
            object_id="object-1",
        ),
    )


def _mask_bytes() -> bytes:
    return bytes([1]) * (640 * 480)


def _mask_digest() -> str:
    return hashlib.sha256(_mask_bytes()).hexdigest()


def _points_bytes() -> bytes:
    return struct.pack("<6f", 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)


def _points_digest() -> str:
    return hashlib.sha256(_points_bytes()).hexdigest()


def _pointmap_bytes() -> bytes:
    return _points_bytes() + bytes((640 * 480 - 2) * 12)


def _pointmap_digest() -> str:
    return hashlib.sha256(_pointmap_bytes()).hexdigest()


def _indices_bytes() -> bytes:
    return struct.pack("<2I", 0, 1)


def _indices_digest() -> str:
    return hashlib.sha256(_indices_bytes()).hexdigest()


def _outcome(manifest: TeacherOracleInputManifestV1) -> GeometrySelectionOutcomeV1:
    return GeometrySelectionOutcomeV1(
        object_id="object-1",
        observation_id="obs-1",
        source_video_id="video-1",
        decision="selected",
        selected_points=SelectedPointPayloadRefV1(
            uri="sealed://fixture/points",
            sha256=_points_digest(),
            sealed_root_sha256=manifest.sealed_root_manifest_sha256,
            manifest_sha256=canonical_sha256(manifest.model_dump(mode="json")),
            shape=(2, 3),
            point_count=2,
            frame_ref="frame-1",
            object_id="object-1",
            coordinate_frame_id="fixture-world",
            coordinate_units="m",
            coordinate_min_xyz=(-10.0, -10.0, -10.0),
            coordinate_max_xyz=(10.0, 10.0, 10.0),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
        ),
        selected_indices=SelectedIndexPayloadRefV1(
            uri="sealed://fixture/indices",
            sha256=_indices_digest(),
            sealed_root_sha256=manifest.sealed_root_manifest_sha256,
            manifest_sha256=canonical_sha256(manifest.model_dump(mode="json")),
            point_count=2,
            mask_width_px=640,
            mask_height_px=480,
            frame_ref="frame-1",
            object_id="object-1",
        ),
        source_pointmap=SourcePointmapPayloadRefV1(
            uri="sealed://fixture/pointmap",
            sha256=_pointmap_digest(),
            sealed_root_sha256=manifest.sealed_root_manifest_sha256,
            manifest_sha256=canonical_sha256(manifest.model_dump(mode="json")),
            width_px=640,
            height_px=480,
            frame_ref="frame-1",
            object_id="object-1",
        ),
    )


def _semantic_outcomes(
    decision: Literal["objects", "zero_object", "abstained"] = "objects",
) -> tuple[SemanticObservationOutcomeV1, ...]:
    return (
        SemanticObservationOutcomeV1(
            observation_id="obs-1",
            source_video_id="video-1",
            frame_ref="frame-1",
            decision=decision,
            abstention_reason="provider unavailable"
            if decision == "abstained"
            else None,
        ),
    )


def test_schema_boundaries_and_exact_geometry_outcome() -> None:
    manifest = _manifest()
    semantic = _semantic(manifest)
    assert "place" not in semantic.model_dump()
    validate_oracle_assignments(
        manifest, (semantic,), _semantic_outcomes(), (_outcome(manifest),)
    )
    with pytest.raises(ValidationError, match="shape"):
        _ = SelectedPointPayloadRefV1(
            uri="sealed://fixture/invalid",
            sha256=DIGEST,
            sealed_root_sha256=DIGEST,
            manifest_sha256=DIGEST,
            shape=(3, 2),
            point_count=3,
            frame_ref="f",
            object_id="object-1",
            coordinate_frame_id="fixture-world",
            coordinate_units="m",
            coordinate_min_xyz=(-10.0, -10.0, -10.0),
            coordinate_max_xyz=(10.0, 10.0, 10.0),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
        )
    with pytest.raises(TeacherOracleContractError, match="exactly one"):
        validate_oracle_assignments(_manifest(), (semantic,), _semantic_outcomes(), ())
    abstained = GeometrySelectionOutcomeV1(
        object_id="object-1",
        observation_id="obs-1",
        source_video_id="video-1",
        decision="abstained",
        abstention_reason="insufficient_valid_mask_points",
        provider_status="completed",
        response_sha256=DIGEST,
        evidence_count=0,
        valid_evidence_count=0,
    )
    validate_oracle_assignments(
        manifest, (semantic,), _semantic_outcomes(), (abstained,)
    )


_ABSTENTION_CASES: tuple[
    tuple[
        GeometryAbstentionReason,
        Literal["completed", "unavailable", "rejected", "invalid"],
        int,
    ],
    ...,
] = (
    ("insufficient_valid_mask_points", "completed", 0),
    ("no_valid_depth", "completed", 0),
    ("provider_uncertainty", "completed", 1),
    ("outside_approved_bounds", "completed", 1),
)


@pytest.mark.parametrize(
    ("reason", "provider_status", "evidence_count"),
    _ABSTENTION_CASES,
)
def test_geometry_abstention_reasons_are_closed_and_evidence_bearing(
    reason: GeometryAbstentionReason,
    provider_status: Literal["completed", "unavailable", "rejected", "invalid"],
    evidence_count: int,
) -> None:
    outcome = GeometrySelectionOutcomeV1(
        object_id="object-1",
        observation_id="obs-1",
        source_video_id="video-1",
        decision="abstained",
        abstention_reason=reason,
        provider_status=provider_status,
        response_sha256=DIGEST,
        evidence_count=evidence_count,
        valid_evidence_count=0,
    )
    assert outcome.abstention_reason == reason
    with pytest.raises(ValidationError):
        _ = GeometrySelectionOutcomeV1.model_validate(
            {
                "object_id": "object-1",
                "observation_id": "obs-1",
                "source_video_id": "video-1",
                "decision": "abstained",
                "abstention_reason": "integrity_loss",
                "provider_status": "completed",
                "response_sha256": DIGEST,
                "evidence_count": 0,
                "valid_evidence_count": 0,
            }
        )


def test_e0_object_presence_forbids_geometry_identity_and_frame_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        _ = ObjectPresenceMemoryRecord.model_validate(
            {
                "record_type": "object_presence_v1",
                "memory_id": "e0-observation-1",
                "source_video_id": "video-1",
                "observation_id": "obs-1",
                "timestamp": 1.0,
                "source_inventory_sha256": DIGEST,
                "semantic_provenance_sha256": DIGEST,
                "mask_provenance_sha256": DIGEST,
                "geometry": {},
                "entity_id": "object-1",
                "frame_ref": "frame-1",
                "place_label": "kitchen",
            }
        )


def _identity_claims_bytes() -> bytes:
    return json.dumps(
        {
            "frame_ref": "frame-1",
            "identity_index_sha256": DIGEST,
            "identity_scope": "fixture-identity-scope",
            "object_id": "object-1",
            "observation_id": "obs-1",
            "place_id": "kitchen",
            "source_video_id": "video-1",
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _identity_certificate(
    manifest: TeacherOracleInputManifestV1,
) -> StableIdentityCertificateV1:
    return StableIdentityCertificateV1(
        certificate=SealedPayloadRefV1[Literal["stable-identity-certificate-v1"]](
            uri="sealed://fixture/identity",
            sha256=hashlib.sha256(_identity_claims_bytes()).hexdigest(),
            schema="stable-identity-certificate-v1",
            sealed_root_sha256=manifest.sealed_root_manifest_sha256,
            manifest_sha256=canonical_sha256(manifest.model_dump(mode="json")),
        ),
        object_id="object-1",
        observation_id="obs-1",
        source_video_id="video-1",
        frame_ref="frame-1",
        place_id="kitchen",
        identity_index_sha256=DIGEST,
        identity_scope="fixture-identity-scope",
        identity_index_attestation=_signed_envelope(
            _IDENTITY_KEY,
            purpose="identity_index",
            payload={
                "scope": "fixture-identity-scope",
                "cutoff_us": 1_000_000,
                "retained_object_ids": ("object-1",),
                "identity_index_sha256": DIGEST,
            },
        ),
    )


def _sealed_entry(
    path: Path,
    *,
    role: Literal[
        "mask",
        "points",
        "pointmap",
        "selected_indices",
        "identity",
        "compiled_assignments",
    ],
    frame_ref: str | None = "frame-1",
    object_id: str | None = "object-1",
) -> SealedManifestEntryV1:
    path.chmod(0o444)
    metadata = path.stat()
    nlink: Literal[1] = 1
    assert metadata.st_nlink == nlink
    return SealedManifestEntryV1(
        canonical_path=path.name,
        role=role,
        object_id=object_id,
        frame_ref=frame_ref,
        size_bytes=metadata.st_size,
        mode=metadata.st_mode & 0o7777,
        device=metadata.st_dev,
        nlink=nlink,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _sealed_root_manifest(
    entries: dict[str, SealedManifestEntryV1],
) -> tuple[bytes, str]:
    manifest = SealedRootManifestV1(
        entries=tuple(
            SealedRootManifestBindingV1(uri=uri, entry=entry)
            for uri, entry in entries.items()
        )
    )
    payload = sealed_root_manifest_bytes(manifest)
    return payload, sealed_root_manifest_sha256(payload)


def _fixture_sealed_root_manifest(tmp_path: Path) -> tuple[bytes, str]:
    points = tmp_path / "points"
    indices = tmp_path / "indices"
    identity = tmp_path / "identity"
    pointmap = tmp_path / "pointmap"
    mask = tmp_path / "mask"
    _ = points.write_bytes(_points_bytes())
    _ = indices.write_bytes(_indices_bytes())
    _ = identity.write_bytes(_identity_claims_bytes())
    _ = mask.write_bytes(_mask_bytes())
    _ = pointmap.write_bytes(_pointmap_bytes())
    entries = {
        "sealed://fixture/points": _sealed_entry(points, role="points"),
        "sealed://fixture/indices": _sealed_entry(indices, role="selected_indices"),
        "sealed://fixture/mask": _sealed_entry(mask, role="mask"),
        "sealed://fixture/identity": _sealed_entry(
            identity, role="identity", frame_ref=None, object_id=None
        ),
        "sealed://fixture/pointmap": _sealed_entry(pointmap, role="pointmap"),
    }
    return _sealed_root_manifest(entries)


def test_sealed_manifest_resolver_rejects_unlisted_payloads(tmp_path: Path) -> None:
    payload = _points_bytes()
    payload_path = tmp_path / "points"
    _ = payload_path.write_bytes(payload)
    descriptor = SelectedPointPayloadRefV1(
        uri="sealed://fixture/points",
        sha256=hashlib.sha256(payload).hexdigest(),
        sealed_root_sha256=DIGEST,
        manifest_sha256=DIGEST,
        shape=(2, 3),
        point_count=2,
        frame_ref="frame-1",
        object_id="object-1",
        coordinate_frame_id="fixture-world",
        coordinate_units="m",
        coordinate_min_xyz=(-10.0, -10.0, -10.0),
        coordinate_max_xyz=(10.0, 10.0, 10.0),
        bounds_min_m=(0.0, 0.0, 0.0),
        bounds_max_m=(1.0, 1.0, 1.0),
    )
    root_bytes, root_sha256 = _sealed_root_manifest(
        {descriptor.uri: _sealed_entry(payload_path, role="points")}
    )
    descriptor = descriptor.model_copy(update={"sealed_root_sha256": root_sha256})
    resolver = SealedManifestPayloadResolver(tmp_path, root_bytes, root_sha256)
    try:
        assert resolver.resolve(descriptor) == payload
        with pytest.raises(TeacherOracleContractError, match="not authorized"):
            _ = resolver.resolve(
                descriptor.model_copy(update={"uri": "sealed://other"})
            )
    finally:
        resolver.close()


def test_sealed_manifest_resolver_requires_openat2_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _points_bytes()
    payload_path = tmp_path / "points"
    _ = payload_path.write_bytes(payload)
    descriptor = SelectedPointPayloadRefV1(
        uri="sealed://fixture/points",
        sha256=hashlib.sha256(payload).hexdigest(),
        sealed_root_sha256=DIGEST,
        manifest_sha256=DIGEST,
        shape=(2, 3),
        point_count=2,
        frame_ref="frame-1",
        object_id="object-1",
        coordinate_frame_id="fixture-world",
        coordinate_units="m",
        coordinate_min_xyz=(-10.0, -10.0, -10.0),
        coordinate_max_xyz=(10.0, 10.0, 10.0),
        bounds_min_m=(0.0, 0.0, 0.0),
        bounds_max_m=(1.0, 1.0, 1.0),
    )

    def unavailable(*_args: object, **_kwargs: object) -> int:
        message = "kernel does not implement openat2"
        raise openat2.Openat2UnsupportedError(message)

    monkeypatch.setattr(teacher_oracle, "openat2_sealed", unavailable)
    with pytest.raises(TeacherOracleContractError, match="requires openat2"):
        _ = SealedManifestPayloadResolver(
            tmp_path,
            *_sealed_root_manifest(
                {descriptor.uri: _sealed_entry(payload_path, role="points")}
            ),
        )


def test_openat2_sealed_resolution_mask_is_exact() -> None:
    assert openat2.SEALED_RESOLVE == (
        openat2.RESOLVE_IN_ROOT
        | openat2.RESOLVE_NO_SYMLINKS
        | openat2.RESOLVE_NO_MAGICLINKS
        | openat2.RESOLVE_NO_XDEV
    )
    assert openat2.SEALED_RESOLVE & openat2.RESOLVE_BENEATH == 0


def test_sealed_manifest_resolver_rejects_ancestor_symlink(
    tmp_path: Path,
) -> None:
    backing = tmp_path / "backing"
    _ = backing.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(backing, target_is_directory=True)

    with pytest.raises(
        TeacherOracleContractError, match="cannot open sealed payload root"
    ):
        _ = SealedManifestPayloadResolver(alias, *_sealed_root_manifest({}))


def test_variant_and_place_boundaries_and_zero_object() -> None:
    e0_manifest = _manifest("E0")
    validate_oracle_assignments(
        e0_manifest, (_semantic(e0_manifest),), _semantic_outcomes(), ()
    )
    place = PlaceGroundingAssignmentV1(
        object_id="object-1",
        observation_id="obs-1",
        source_video_id="video-1",
        place_id="kitchen",
        mode="frame_bound",
        frame_ref="frame-1",
    )
    t0_manifest = _manifest("T0")
    with pytest.raises(TeacherOracleContractError, match="T0"):
        validate_oracle_assignments(
            t0_manifest,
            (_semantic(t0_manifest),),
            _semantic_outcomes(),
            (_outcome(t0_manifest),),
            (place,),
        )
    t1_manifest = _manifest("T1")
    stable = PlaceGroundingAssignmentV1(
        object_id="object-1",
        observation_id="obs-1",
        source_video_id="video-1",
        place_id="kitchen",
        mode="stable_identity",
        frame_ref="frame-1",
        identity_certificate=_identity_certificate(t1_manifest),
    )
    validate_oracle_assignments(
        t1_manifest,
        (_semantic(t1_manifest),),
        _semantic_outcomes(),
        (_outcome(t1_manifest),),
        (stable,),
    )
    with pytest.raises(ValidationError, match="stable_identity"):
        _ = PlaceGroundingAssignmentV1(
            object_id="object-1",
            observation_id="obs-1",
            source_video_id="video-1",
            place_id="kitchen",
            mode="stable_identity",
            frame_ref="frame-1",
        )


def test_fixed_window_and_closed_label_blind_input() -> None:
    assert WINDOW_MICROSECONDS == 30_000_000
    with pytest.raises(TeacherOracleContractError, match="declared"):
        _ = validate_label_blind_payload(
            {"observations": [{"QUESTION_ID": "q"}]},
        )
    with pytest.raises(TeacherOracleContractError, match="undeclared"):
        _ = validate_label_blind_payload(
            {"source_inventory": {}, "nested": [{"eViDeNcE-list": "x"}]},
        )
    payload: dict[str, object] = {
        "source_inventory": _ref().model_dump(mode="json"),
        "frame_inventory": _frames().model_dump(mode="json"),
        "observations": [
            {
                "observation_id": "obs-1",
                "source_video_id": "video-1",
                "frame_ref": "frame-1",
                "local_frame_id": "local-1",
                "timestamp_us": 1_000_000,
            },
        ],
    }
    source_inventory = cast("dict[str, object]", payload["source_inventory"])
    observations = cast("list[dict[str, object]]", payload["observations"])
    assert isinstance(
        validate_label_blind_payload(payload),
        TeacherProducerInputV1,
    )
    for field, value, error in (
        ("correct_choice", "A", "undeclared"),
        ("ground_truth_answer", "A", "undeclared"),
        ("student_checkpoint", "artifact://student-model", "undeclared"),
    ):
        injected = {
            **payload,
            "observations": [{**observations[0], field: value}],
        }
        with pytest.raises(TeacherOracleContractError, match=error):
            _ = validate_label_blind_payload(injected)
    for value, error in (
        ("/labels/root.jsonl", "provenance URI"),
        ("artifact://ground-truth-answer-key", "label-bearing provenance"),
        ("student-inference-output", "label-bearing value"),
    ):
        injected = {
            **payload,
            "source_inventory": (
                {**source_inventory, "uri": value}
                if value.startswith(("artifact://", "/"))
                else source_inventory
            ),
            "observations": [{**observations[0], "frame_ref": value}],
        }
        with pytest.raises(TeacherOracleContractError, match=error):
            _ = validate_label_blind_payload(injected)


def test_manifests_are_deterministic_and_byte_bound() -> None:
    manifest = _manifest()
    assert canonical_sha256(manifest.model_dump(mode="json")) == canonical_sha256(
        manifest.model_dump(mode="json")
    )
    artifact = SealedPayloadRefV1[Literal["compiled-oracle-assignments-v1"]](
        uri="sealed://assignments",
        sha256=DIGEST,
        schema="compiled-oracle-assignments-v1",
        sealed_root_sha256=DIGEST,
        manifest_sha256=DIGEST,
    )
    with pytest.raises(ValidationError, match="max_window_bytes"):
        _ = TeacherOracleOutputManifestV1(
            variant="T0",
            input_manifest_sha256=DIGEST,
            typed_memory=_ref("typed-memory-jsonl-v1"),
            sealed_root_manifest_sha256=DIGEST,
            fairness_contract_sha256=DIGEST,
            actual_bytes=10,
            max_window_bytes=11,
            approved_assignment_artifact=artifact,
            approved_outcome_artifact=artifact,
            semantic_denominator=0,
            abstention_counts={},
            place_completion_count=0,
            pre_budget_candidate_count=0,
            byte_drop_count=0,
            selected_memory_ids=(),
        )


def test_signed_cross_variant_fairness_contract_rejects_mismatched_variant() -> None:
    claims = CrossVariantFairnessClaimsV1(
        sealed_root_manifest_sha256=DIGEST,
        source_inventory_sha256=DIGEST,
        frame_inventory_sha256=DIGEST,
        qa_inventory_sha256=DIGEST,
        split_sha256=DIGEST,
        byte_budget_per_window=4096,
        cadence_origin_us=0,
    )
    contract = CrossVariantFairnessContractV1(
        claims=claims,
        attestation=_signed_envelope(
            _IDENTITY_KEY,
            purpose="teacher_cache_production",
            payload=claims.model_dump(mode="json"),
        ),
    )
    digest = cross_variant_fairness_contract_sha256(contract)
    manifests = tuple(
        _manifest(variant).model_copy(update={"fairness_contract_sha256": digest})
        for variant in ("E0", "T0", "T1")
    )
    registry = _attestation_registry(_IDENTITY_KEY, purpose="teacher_cache_production")
    validate_cross_variant_fairness_contract(
        contract, manifests, authorized_registry=registry
    )
    mismatched = manifests[1].model_copy(update={"byte_budget_per_window": 1})
    with pytest.raises(TeacherOracleContractError, match="byte cap mismatch"):
        validate_cross_variant_fairness_contract(
            contract,
            (manifests[0], mismatched, manifests[2]),
            authorized_registry=registry,
        )
    with pytest.raises(
        TeacherOracleContractError,
        match="request oracle manifest must equal the unique same-variant",
    ):
        validate_request_manifest_fairness_trio(
            manifests[1].model_copy(update={"cadence_origin_us": 1}),
            contract,
            manifests,
            authorized_registry=registry,
        )


def test_e0_materializes_a_nonempty_digest_bound_source_compact_record(
    tmp_path: Path,
) -> None:
    manifest = _manifest("E0")
    record = _object_presence(manifest)
    summary = materialize_variant_memory(
        manifest,
        _compiled(manifest, semantics=(_semantic(manifest),)),
        (ScoredMemoryCandidate(record=record, score=1.0),),
        tmp_path / "e0.jsonl",
    )
    assert summary.selected_memory_ids == (record.memory_id,)
    assert (tmp_path / "e0.jsonl").read_bytes()
    persisted = ObjectPresenceMemoryRecord.model_validate_json(
        (tmp_path / "e0.jsonl").read_bytes()
    )
    assert persisted.timestamp_us == 1_000_000
    assert persisted.semantic_class == "mug"
    assert persisted.mask_sha256 == _mask_digest()


def test_e0_rejects_record_timestamp_not_equal_to_approved_frame(
    tmp_path: Path,
) -> None:
    manifest = _manifest("E0")
    stale_record = _object_presence(manifest).model_copy(
        update={"timestamp_us": 1_000_001}
    )

    with pytest.raises(
        TeacherOracleContractError,
        match="exactly preserve the shared semantic/mask/frame assignment",
    ):
        _ = materialize_variant_memory(
            manifest,
            _compiled(manifest, semantics=(_semantic(manifest),)),
            (ScoredMemoryCandidate(record=stale_record, score=1.0),),
            tmp_path / "stale-e0.jsonl",
        )


@pytest.mark.parametrize("variant", ["T0", "T1"])
def test_teacher_records_bind_compiled_assignment_and_sealed_payload(
    variant: OracleVariant,
    tmp_path: Path,
) -> None:
    root_bytes, root_sha256 = _fixture_sealed_root_manifest(tmp_path)
    manifest = _manifest(variant, sealed_root_manifest_sha256=root_sha256)
    semantic = _semantic(manifest)
    outcome = _outcome(manifest)
    places: tuple[PlaceGroundingAssignmentV1, ...] = ()
    if variant == "T1":
        places = (
            PlaceGroundingAssignmentV1(
                object_id="object-1",
                observation_id="obs-1",
                source_video_id="video-1",
                place_id="kitchen",
                mode="stable_identity",
                frame_ref="frame-1",
                identity_certificate=_identity_certificate(manifest),
            ),
        )
    assignments = _compiled(
        manifest,
        semantics=(semantic,),
        outcomes=(outcome,),
        places=places,
    )
    assert assignments.assignment_sha256 != "0" * 64
    assert outcome.selected_points is not None
    record = _object_record(assignments)
    if variant == "T1":
        record = record.model_copy(update={"place_label": "kitchen"})

    output_path = tmp_path / f"{variant}.jsonl"
    summary = materialize_variant_memory(
        manifest,
        assignments,
        (
            ScoredMemoryCandidate(
                record=record,
                score=semantic.semantic_confidence / serialized_byte_cost(record),
            ),
        ),
        output_path,
        context=MaterializeVariantMemoryContext(
            sealed_payload_resolver=SealedManifestPayloadResolver(
                tmp_path, root_bytes, root_sha256
            ),
            identity_registry=_identity_registry() if variant == "T1" else None,
        ),
    )
    assert summary.selected_memory_ids == (record.memory_id,)
    persisted_record = TypeAdapter(dict[str, object]).validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert persisted_record["oracle_assignment_sha256"] == assignments.assignment_sha256
    assert persisted_record["selected_payload_sha256"] == outcome.selected_points.sha256

    for field in ("oracle_assignment_sha256", "selected_payload_sha256"):
        invalid_record = record.model_copy(update={field: "0" * 64})
        with pytest.raises(
            TeacherOracleContractError,
            match="must bind the compiled assignment and sealed selected-point payload",
        ):
            _ = materialize_variant_memory(
                manifest,
                assignments,
                (ScoredMemoryCandidate(record=invalid_record, score=1.0),),
                tmp_path / f"{variant}-{field}.jsonl",
            )


def test_compiled_assignment_digest_and_manifest_mismatch_fail_closed() -> None:
    manifest = _manifest()
    compiled = _compiled(
        manifest,
        semantics=(_semantic(manifest),),
        outcomes=(_outcome(manifest),),
    )
    assert compiled.assignment_sha256
    with pytest.raises(ValidationError, match="assignment_sha256"):
        _ = CompiledOracleAssignmentsV1(
            manifest_sha256=compiled.manifest_sha256,
            assignment_sha256=DIGEST,
            semantics=compiled.semantics,
            outcomes=compiled.outcomes,
        )


def test_cache_provenance_exact_join() -> None:

    # This test only fixes the exact-join contract; adapter input tests cover
    # causal rows.
    provenance = LegacyNonProductionCacheProvenanceV1(
        signer_id="signer",
        signature_b64="AA==",
        issued_at=1.0,
        scope="exp-0005",
        source_inventory_sha256=DIGEST,
        frame_inventory_sha256=DIGEST,
        provider_artifact_sha256=DIGEST,
        semantic_provider_artifact_sha256=DIGEST,
        ontology_sha256=DIGEST,
        oracle_manifest_sha256=DIGEST,
        cache_content_sha256="0" * 64,
    )
    registry = LegacyNonProductionCacheSignerRegistryV1(
        signers=(
            LegacyNonProductionCacheSignerV1(
                signer_id="signer",
                public_key_b64="AA==",
                purposes=("teacher_cache_production",),
                not_before=0.0,
                not_after=2.0,
                allowed_scopes=("exp-0005",),
            ),
        ),
    )
    with pytest.raises(TeacherContractError, match="at least one row"):
        validate_legacy_non_production_cache_provenance(
            (),
            provenance,
            authorized_registry=registry,
            approved_contract=_cache_contract(),
        )


def test_cache_provenance_binds_ordered_cache_content() -> None:
    unsigned_rows = (_cache_record("video-1"), _cache_record("video-2"))
    provenance, registry = _signed_cache_provenance(unsigned_rows)
    rows = tuple(
        build_teacher_cache_record(
            teacher_backend=row.teacher_backend,
            provider_id=row.provider_id,
            request=row.request,
            response=row.response,
            legacy_non_production_provenance=provenance,
        )
        for row in unsigned_rows
    )

    validate_legacy_non_production_cache_provenance(
        rows,
        provenance,
        authorized_registry=registry,
        approved_contract=_cache_contract(),
    )
    with pytest.raises(
        TeacherContractError,
        match="does not bind the ordered cache content",
    ):
        validate_legacy_non_production_cache_provenance(
            tuple(reversed(rows)),
            provenance,
            authorized_registry=registry,
            approved_contract=_cache_contract(),
        )


def test_teacher_materialization_rejects_geometry_tampering_with_valid_digest_stamps(
    tmp_path: Path,
) -> None:
    root_bytes, root_sha256 = _fixture_sealed_root_manifest(tmp_path)
    manifest = _manifest(sealed_root_manifest_sha256=root_sha256)
    semantic = _semantic(manifest)
    outcome = _outcome(manifest)
    assignments = _compiled(manifest, semantics=(semantic,), outcomes=(outcome,))
    record = _object_record(assignments).model_copy(
        update={
            "geometry": ObjectGeometry(
                centroid=(9.0, 9.0, 9.0),
                extent=(1.0, 1.0, 1.0),
            )
        }
    )
    with pytest.raises(TeacherOracleContractError, match="does not derive"):
        _ = materialize_variant_memory(
            manifest,
            assignments,
            (ScoredMemoryCandidate(record=record, score=1.0),),
            tmp_path / "tampered.jsonl",
            context=MaterializeVariantMemoryContext(
                sealed_payload_resolver=SealedManifestPayloadResolver(
                    tmp_path, root_bytes, root_sha256
                )
            ),
        )


def test_strict_cache_attestation_rejects_invalid_authority() -> None:
    records = (_cache_record("video-1"),)
    contract = _cache_contract()
    private_key = Ed25519PrivateKey.generate()
    claims = CacheProductionClaimsV1(
        bundle_sha256=DIGEST,
        runner_sha256=DIGEST,
        mount_policy_sha256=DIGEST,
        stage_spec_sha256=DIGEST,
        resource_spec_sha256=DIGEST,
        code_sha256=DIGEST,
        model_sha256=DIGEST,
        config_sha256=DIGEST,
        output_root_sha256=DIGEST,
        cache_content_sha256=_cache_content_sha256(records),
        oracle_manifest_sha256=contract.oracle_manifest_sha256,
        source_inventory_sha256=contract.source_inventory_sha256,
        frame_inventory_sha256=contract.frame_inventory_sha256,
        provider_artifact_sha256=contract.provider_artifact_sha256,
        semantic_provider_artifact_sha256=contract.semantic_provider_artifact_sha256,
        ontology_sha256=contract.ontology_sha256,
        provider_id=contract.provider_id,
        scope=contract.scope,
    )
    attestation = _signed_envelope(
        private_key,
        purpose="teacher_cache_production",
        payload=claims.model_dump(mode="json"),
    )
    registry = _attestation_registry(private_key, purpose="teacher_cache_production")
    assert (
        validate_cache_production_attestation(
            records,
            attestation,
            authorized_registry=registry,
            approved_contract=contract,
        )
        == claims
    )
    for invalid in (
        attestation.model_copy(update={"signature_b64url": "AA"}),
        attestation.model_copy(update={"purpose": "wrong-purpose"}),
    ):
        with pytest.raises(TeacherContractError):
            _ = validate_cache_production_attestation(
                records,
                invalid,
                authorized_registry=registry,
                approved_contract=contract,
            )
    for invalid_registry in (
        registry.model_copy(
            update={"keys": (registry.keys[0].model_copy(update={"revoked": True}),)}
        ),
        registry.model_copy(
            update={"keys": (registry.keys[0].model_copy(update={"not_after": 0.5}),)}
        ),
    ):
        with pytest.raises(TeacherContractError, match="not authorized"):
            _ = validate_cache_production_attestation(
                records,
                attestation,
                authorized_registry=invalid_registry,
                approved_contract=contract,
            )
    with pytest.raises(TeacherContractError, match="does not match approved"):
        _ = validate_cache_production_attestation(
            records,
            attestation,
            authorized_registry=registry,
            approved_contract=contract.model_copy(update={"scope": "wrong-scope"}),
        )


def test_producer_manifest_rejects_duplicate_observations() -> None:
    manifest = _manifest("E0")
    observation = ProducerObservationV1(
        observation_id="obs-1",
        source_video_id="video-1",
        frame_ref="frame-1",
        local_frame_id="local-1",
        timestamp_us=1_000_000,
    )
    producer = TeacherProducerInputV1(
        source_inventory=manifest.source_inventory,
        frame_inventory=manifest.frame_inventory,
        observations=(observation, observation),
    )
    with pytest.raises(TeacherOracleContractError, match="must be unique"):
        validate_producer_input_manifest(producer, manifest)


def test_frame_inventory_rejects_ambiguous_frame_ref() -> None:
    with pytest.raises(ValidationError, match="globally unambiguous"):
        _ = FrameInventoryV1(
            uri="artifact://frames",
            sha256=DIGEST,
            schema="frame-inventory-v1",
            frames=(
                ApprovedFrameRefV1(
                    frame_ref="frame-1",
                    source_video_id="video-1",
                    observation_id="obs-1",
                    local_frame_id="local-1",
                    timestamp_us=1_000_000,
                ),
                ApprovedFrameRefV1(
                    frame_ref="frame-1",
                    source_video_id="video-2",
                    observation_id="obs-2",
                    local_frame_id="local-2",
                    timestamp_us=2_000_000,
                ),
            ),
        )
