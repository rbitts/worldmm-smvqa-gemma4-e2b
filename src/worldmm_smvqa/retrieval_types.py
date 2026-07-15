from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Final, Literal, Self, cast

from pydantic import Field, model_validator

from worldmm_smvqa.schema import FrozenModel

type RetrievalStore = Literal["episodic", "semantic", "visual", "spatial"]
type RetrievalProtocol = Literal["smvqa-video-rag", "egobutler", "worldmm"]
type EvidenceLane = Literal["heuristic", "student", "teacher_oracle"]
type EvidenceProducer = Literal["heuristic-retrieval", "spatial-student"]
type OracleVariant = Literal["E0", "T0", "T1"]
RETRIEVAL_FRAME_REF_CAP: Final = 32
ORACLE_VARIANT_COUNT: Final = 3


class EvidenceLineage(FrozenModel):
    """Record immutable provenance for retrieval evidence."""

    lane: EvidenceLane
    producer: EvidenceProducer
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    typed_memory_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    inference_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sensor_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    data_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    memory_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    episodic_memory_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    semantic_memory_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    visual_memory_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def _require_student_inputs(self) -> Self:
        expected_producer: EvidenceProducer = (
            "spatial-student" if self.lane == "student" else "heuristic-retrieval"
        )
        if self.producer != expected_producer:
            msg = f"{self.lane} evidence requires producer {expected_producer}"
            raise ValueError(msg)
        if self.lane != "student":
            return self
        required = (
            "checkpoint_sha256",
            "typed_memory_sha256",
            "inference_manifest_sha256",
            "config_sha256",
            "sensor_sha256",
            "data_sha256",
            "memory_manifest_sha256",
            "episodic_memory_sha256",
            "semantic_memory_sha256",
            "visual_memory_sha256",
        )
        if missing := tuple(name for name in required if getattr(self, name) is None):
            msg = f"student evidence lineage missing: {', '.join(missing)}"
            raise ValueError(msg)
        return self


class SharedQALineage(FrozenModel):
    """Label-blind decoding contract that must be identical for E0, T0, and T1."""

    approved_salt: str = Field(min_length=1)
    world_size: int = Field(ge=1)
    question_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decoding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    python_inventory_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    torch_inventory_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    transformers_inventory_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    seed: int

    @property
    def sha256(self) -> str:
        """Return the canonical SHA-256 digest of this shared lineage."""
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()


class OracleQAInputLineage(FrozenModel):
    """Approved, pre-evaluation inputs for one teacher-oracle QA variant."""

    variant: OracleVariant
    memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pre_evaluation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OracleQAPreEvaluationLineage(FrozenModel):
    """The complete, output-free contract required to begin teacher-oracle QA."""

    lane: Literal["teacher_oracle"] = "teacher_oracle"
    producer: Literal["offline-teacher"]
    sensor_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    object_semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    geometry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    place_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    typed_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shared_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sensor_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_frame_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_frame_assets_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    window_microseconds: Literal[30_000_000] = 30_000_000
    qa_inputs: tuple[OracleQAInputLineage, ...]
    shared_qa_lineage: SharedQALineage

    @model_validator(mode="after")
    def _require_complete_input_set(self) -> Self:
        inputs = tuple(item.variant for item in self.qa_inputs)
        if set(inputs) != {"E0", "T0", "T1"} or len(inputs) != ORACLE_VARIANT_COUNT:
            msg = "oracle QA inputs require exactly E0, T0, and T1 variants"
            raise ValueError(msg)
        shared_digest = self.shared_qa_lineage.sha256
        if any(item.pre_evaluation_sha256 != shared_digest for item in self.qa_inputs):
            msg = "oracle QA inputs must bind the same pre-evaluation QA lineage"
            raise ValueError(msg)
        return self

    @property
    def sha256(self) -> str:
        """Return the canonical digest of the pre-evaluation QA contract."""
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()


class OracleQAResultLineage(FrozenModel):
    """Post-evaluation result artifacts, never required to start QA."""

    variant: OracleVariant
    input_lineage_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    finalization_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    finalization_receipt_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OracleVariantLineage(FrozenModel):
    """Bind a single oracle variant to its immutable output artifacts."""

    variant: OracleVariant
    memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pre_evaluation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    finalization_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    finalization_receipt_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OracleEvidenceLineage(FrozenModel):
    """Label-blind lineage for the offline teacher-oracle experiment.

    This intentionally has no checkpoint or inference-manifest fields: a teacher
    ceiling is not a student inference claim.
    """

    lane: Literal["teacher_oracle"] = "teacher_oracle"
    producer: Literal["offline-teacher"]
    sensor_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    object_semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    geometry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    place_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    typed_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shared_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sensor_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    selected_frame_inventory_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    selected_frame_assets_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    window_microseconds: Literal[30_000_000] = 30_000_000
    variants: tuple[OracleVariantLineage, ...]
    shared_qa_lineage: SharedQALineage

    @model_validator(mode="after")
    def _require_complete_variant_set(self) -> Self:
        variants = tuple(item.variant for item in self.variants)
        if set(variants) != {"E0", "T0", "T1"} or len(variants) != ORACLE_VARIANT_COUNT:
            msg = "oracle lineage requires exactly E0, T0, and T1 variants"
            raise ValueError(msg)
        return self

    @property
    def sha256(self) -> str:
        """Return the canonical digest of the complete teacher-oracle lineage."""
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()


type AnyEvidenceLineage = EvidenceLineage | OracleEvidenceLineage


class RetrievalMemoryRecord(FrozenModel):
    """Represent one candidate memory available to retrieval."""

    memory_id: str
    source_store: RetrievalStore
    video_id: str
    start_time: float = Field(allow_inf_nan=False)
    end_time: float = Field(allow_inf_nan=False)
    snippet: str
    frame_refs: tuple[str, ...]
    base_score: float = Field(default=0.0, allow_inf_nan=False)
    geometry: dict[str, float | str] | None = None


class EvidenceItem(FrozenModel):
    """Represent one selected evidence item."""

    memory_id: str
    video_id: str
    snippet: str
    frame_refs: tuple[str, ...]
    source_store: RetrievalStore
    start_time: float = Field(allow_inf_nan=False)
    end_time: float = Field(allow_inf_nan=False)
    retrieval_score: float = Field(allow_inf_nan=False)
    geometry: dict[str, float | str] | None = None


class RetrievalCandidateCount(FrozenModel):
    """Record candidate counts before and after causal filtering."""

    source_store: RetrievalStore
    before_causal_filter: int
    after_causal_filter: int

    @model_validator(mode="after")
    def _require_valid_counts(self) -> Self:
        if self.before_causal_filter < 0:
            msg = "before_causal_filter must be >= 0"
            raise ValueError(msg)
        if self.after_causal_filter < 0:
            msg = "after_causal_filter must be >= 0"
            raise ValueError(msg)
        if self.after_causal_filter > self.before_causal_filter:
            msg = "after_causal_filter must be <= before_causal_filter"
            raise ValueError(msg)
        return self


class RetrievalTrace(FrozenModel):
    """Record the retrieval protocol and bounded evidence-selection trace."""

    protocols: tuple[RetrievalProtocol, ...]
    eligible_shard_ids: tuple[str, ...]
    selected_clip_ids: tuple[str, ...]
    policy_route: str
    store_order: tuple[RetrievalStore, ...]
    candidate_counts: tuple[RetrievalCandidateCount, ...]
    causal_filtered_count: int
    frame_ref_count: int

    @model_validator(mode="after")
    def _require_valid_trace_counts(self) -> Self:
        if self.causal_filtered_count < 0:
            msg = "causal_filtered_count must be >= 0"
            raise ValueError(msg)
        if self.frame_ref_count < 0:
            msg = "frame_ref_count must be >= 0"
            raise ValueError(msg)
        if self.frame_ref_count > RETRIEVAL_FRAME_REF_CAP:
            msg = f"frame_ref_count must be <= {RETRIEVAL_FRAME_REF_CAP}"
            raise ValueError(msg)
        return self


def legacy_missing_retrieval_trace() -> RetrievalTrace:
    """Compatibility factory for legacy fixture JSON written before traces."""
    return RetrievalTrace(
        protocols=(),
        eligible_shard_ids=(),
        selected_clip_ids=(),
        policy_route="legacy-missing-trace",
        store_order=(),
        candidate_counts=(),
        causal_filtered_count=0,
        frame_ref_count=0,
    )


def _repair_legacy_evidence_item(item: object, video_id: str) -> object:
    if not isinstance(item, dict):
        return item
    mapping = cast("dict[str, object]", item)
    if "video_id" not in mapping:
        return {**mapping, "video_id": video_id}
    return mapping


class EvidencePack(FrozenModel):
    """Legacy-compatible retrieval payload used across existing QA surfaces."""

    question_id: str
    video_id: str
    requested_stores: tuple[RetrievalStore, ...]
    selected_stores: tuple[RetrievalStore, ...]
    evidence_budget: int
    evidence: tuple[EvidenceItem, ...]
    causal_filtered_count: int
    retrieval_trace: RetrievalTrace = Field(
        default_factory=legacy_missing_retrieval_trace,
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_legacy_evidence_video_ids(cls, value: object) -> object:
        """Repair only the explicitly legacy-shaped missing-video-id payload."""
        if not isinstance(value, dict):
            return value
        payload = cast("dict[str, object]", value)
        video_id = payload.get("video_id")
        evidence = payload.get("evidence")
        if not isinstance(video_id, str) or not isinstance(evidence, (list, tuple)):
            return payload
        evidence_items = cast("Sequence[object]", evidence)
        repaired: tuple[object, ...] = tuple(
            _repair_legacy_evidence_item(item, video_id) for item in evidence_items
        )
        return {**payload, "evidence": repaired}


class CanonicalOracleEvidencePack(FrozenModel):
    """Strict EXP-0005 oracle payload; never silently repairs legacy data."""

    variant: OracleVariant
    question_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    requested_stores: tuple[RetrievalStore, ...]
    selected_stores: tuple[RetrievalStore, ...]
    evidence_budget: int = Field(ge=0)
    evidence: tuple[EvidenceItem, ...]
    causal_filtered_count: int = Field(ge=0)
    retrieval_trace: RetrievalTrace

    @model_validator(mode="after")
    def _require_canonical_oracle_invariants(self) -> Self:  # noqa: PLR0912
        if not self.retrieval_trace.policy_route.strip():
            msg = (
                "canonical oracle evidence requires a nonblank "
                "retrieval_trace policy_route"
            )
            raise ValueError(msg)
        if self.retrieval_trace.policy_route == "legacy-missing-trace":
            msg = "canonical oracle evidence requires a documented retrieval_trace"
            raise ValueError(msg)
        if len(self.requested_stores) != len(set(self.requested_stores)):
            msg = "requested_stores must not contain duplicates"
            raise ValueError(msg)
        if len(self.selected_stores) != len(set(self.selected_stores)):
            msg = "selected_stores must not contain duplicates"
            raise ValueError(msg)
        if not set(self.selected_stores).issubset(self.requested_stores):
            msg = "selected_stores must be a subset of requested_stores"
            raise ValueError(msg)
        if len(self.evidence) > self.evidence_budget:
            msg = "evidence count must not exceed evidence_budget"
            raise ValueError(msg)
        if self.causal_filtered_count != self.retrieval_trace.causal_filtered_count:
            msg = "causal_filtered_count must match retrieval_trace"
            raise ValueError(msg)
        if self.retrieval_trace.frame_ref_count != sum(
            len(item.frame_refs) for item in self.evidence
        ):
            msg = "retrieval_trace.frame_ref_count must match evidence frame refs"
            raise ValueError(msg)
        evidence_stores = tuple(item.source_store for item in self.evidence)
        if not set(evidence_stores).issubset(self.selected_stores):
            msg = "evidence source stores must be selected"
            raise ValueError(msg)
        if any(item.video_id != self.video_id for item in self.evidence):
            msg = "evidence video_ids must match the evidence pack video_id"
            raise ValueError(msg)
        ids = tuple(item.memory_id for item in self.evidence)
        if len(ids) != len(set(ids)):
            msg = "evidence memory_ids must be unique"
            raise ValueError(msg)
        required_store = "semantic" if self.variant == "E0" else "spatial"
        if (
            self.requested_stores != (required_store,)
            or self.selected_stores != (required_store,)
            or not self.evidence
            or any(item.source_store != required_store for item in self.evidence)
        ):
            msg = (
                f"{self.variant} canonical oracle evidence requires exactly the "
                f"{required_store} store"
            )
            raise ValueError(msg)
        if self.variant == "E0":
            if any(item.geometry is not None for item in self.evidence):
                msg = "E0 canonical oracle evidence must be semantic presence only"
                raise ValueError(msg)
            return self
        variant = self.variant
        for item in self.evidence:
            if item.geometry is None:
                msg = f"{self.variant} canonical oracle evidence requires geometry"
                raise ValueError(msg)
            _validate_oracle_geometry(item.geometry, variant)
        return self


def canonical_oracle_to_evidence_pack(
    pack: CanonicalOracleEvidencePack,
) -> EvidencePack:
    """Project a validated canonical oracle pack onto the shared QA input contract."""
    return EvidencePack(
        question_id=pack.question_id,
        video_id=pack.video_id,
        requested_stores=pack.requested_stores,
        selected_stores=pack.selected_stores,
        evidence_budget=pack.evidence_budget,
        evidence=pack.evidence,
        causal_filtered_count=pack.causal_filtered_count,
        retrieval_trace=pack.retrieval_trace,
    )


def _validate_oracle_geometry(
    geometry: dict[str, float | str],
    variant: Literal["T0", "T1"],
) -> None:
    """Require the closed, discriminated geometry evidence contract."""
    common = frozenset(
        {
            "record_type",
            "coordinate_frame",
            "entity_id",
            "label",
            "relation",
            "distance_m",
            "x",
            "y",
            "z",
            "yaw_degrees",
            "uncertainty_degrees",
        }
    )
    place = frozenset({"place_id", "place_label"})
    allowed = common if variant == "T0" else common | place
    required = (
        frozenset({"record_type", "coordinate_frame", "entity_id", "x", "y", "z"})
        if variant == "T0"
        else frozenset(
            {
                "record_type",
                "coordinate_frame",
                "entity_id",
                "x",
                "y",
                "z",
                "place_id",
                "place_label",
            }
        )
    )
    if set(geometry).difference(allowed) or not required.issubset(geometry):
        msg = f"canonical {variant} oracle geometry violates its closed evidence shape"
        raise ValueError(msg)
    if geometry["record_type"] not in {"event", "object"}:
        msg = "canonical oracle geometry record_type must be event or object"
        raise ValueError(msg)
    if any(isinstance(geometry[axis], str) for axis in ("x", "y", "z")):
        msg = "canonical oracle geometry coordinates must be numeric"
        raise ValueError(msg)
    if variant == "T1" and (
        not isinstance(geometry["place_id"], str)
        or not geometry["place_id"].strip()
        or not isinstance(geometry["place_label"], str)
        or not geometry["place_label"].strip()
    ):
        msg = "canonical T1 oracle geometry requires nonblank place evidence"
        raise ValueError(msg)
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in geometry.values()
    ):
        msg = "canonical oracle geometry values must be finite"
        raise ValueError(msg)


def load_legacy_evidence_pack(value: object) -> EvidencePack:
    """Explicitly repair legacy-shaped retrieval payloads before parsing."""
    if not isinstance(value, dict):
        return EvidencePack.model_validate(value)
    payload = cast("dict[str, object]", value)
    video_id = payload.get("video_id")
    evidence = payload.get("evidence")
    if isinstance(video_id, str) and isinstance(evidence, (list, tuple)):
        evidence_items = cast("Sequence[object]", evidence)
        repaired: tuple[object, ...] = tuple(
            _repair_legacy_evidence_item(item, video_id) for item in evidence_items
        )
        payload = {**payload, "evidence": repaired}
    if "retrieval_trace" not in payload:
        payload = {**payload, "retrieval_trace": legacy_missing_retrieval_trace()}
    return EvidencePack.model_validate(payload)


def load_canonical_oracle_evidence_pack(
    value: object,
) -> CanonicalOracleEvidencePack:
    """Validate EXP-0005 evidence without legacy repair or implicit defaults."""
    if isinstance(value, str | bytes | bytearray):
        return CanonicalOracleEvidencePack.model_validate_json(value)
    return CanonicalOracleEvidencePack.model_validate(value)
