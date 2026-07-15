from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal, Self, cast, override

from pydantic import (
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from worldmm_smvqa.retrieval_types import (
    ORACLE_VARIANT_COUNT,
    OracleEvidenceLineage,
    OracleVariantLineage,
    SharedQALineage,
)
from worldmm_smvqa.schema import FrozenModel

_PAYLOAD_ADAPTER: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(dict[str, object])

REQUIRED_REPORT_SECTIONS: Final = (
    "Local code/config changed",
    "Remote command used",
    "Remote job ID or process reference",
    "Remote artifact path on company storage",
    "Key metrics or failure reason",
    "What was not copied locally",
)
SENSITIVE_MARKERS: Final = ("authorization", "bearer ", "password", "secret", "token")
ORACLE_METRIC_NAMES: Final = (
    "Ans-F1",
    "QA-Acc",
    "QA-MRR",
    "Coverage",
    "Fairness",
)
MARKDOWN_SPECIALS: Final = "\\*_{}[]<>()#+-.!|"
METRIC_PERCENT_MAX: Final = 100.0
_SHA256 = r"^[0-9a-f]{64}$"


@dataclass(frozen=True, slots=True)
class IncompleteRemoteManifestError(Exception):
    """Raised when a remote manifest cannot be loaded or validated."""

    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"IncompleteRemoteManifest: {self.path}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ReportManifestError(ValueError):
    """Raised when a report manifest violates its reporting contract."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


class ReportMetric(FrozenModel):
    """Describes a single reported experiment metric."""

    experiment: Literal["PROBE", "E0", "T0", "T1", "E1", "E2", "E3"]
    name: str = Field(min_length=1)
    value: float = Field(allow_inf_nan=False)


class _ManifestBase(FrozenModel):
    baseline_name: Literal["WorldMM-SMVQA"]
    remote_status: Literal["pending", "failed", "cancelled", "complete"]
    experiment_id: str = Field(min_length=1)
    lane: str = Field(min_length=1)
    split_id: str = Field(min_length=1)
    code_sha256: str | None = Field(default=None, pattern=_SHA256)
    local_changes: tuple[str, ...] = Field(min_length=1)
    remote_command: str = Field(min_length=1)
    remote_job_reference: str = Field(min_length=1)
    remote_artifact_path: str = Field(min_length=1)
    metrics: tuple[ReportMetric, ...] = ()
    failure_reason: str | None = None
    not_copied_locally: tuple[str, ...] = Field(min_length=1)

    @field_validator(
        "remote_command",
        "remote_job_reference",
        "remote_artifact_path",
        "failure_reason",
    )
    @classmethod
    def _strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("remote_command", "remote_job_reference", "remote_artifact_path")
    @classmethod
    def _require_nonblank_remote_text(cls, value: str) -> str:
        if not value:
            detail = "remote execution identity must not be blank"
            raise ReportManifestError(detail=detail)
        return value

    @field_validator("experiment_id", "lane", "split_id")
    @classmethod
    def _strip_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            detail = "result identity values must not be blank"
            raise ReportManifestError(detail=detail)
        return value

    @model_validator(mode="after")
    def _require_status_payload(self) -> Self:
        if self.remote_status != "complete" and self.metrics:
            detail = "pending or failed result must not contain metrics"
            raise ReportManifestError(detail=detail)
        if self.remote_status in {"failed", "cancelled"} and not self.failure_reason:
            detail = f"{self.remote_status} result requires failure_reason"
            raise ReportManifestError(detail=detail)
        if self.remote_status == "complete" and self.failure_reason is not None:
            detail = "complete result must not contain failure_reason"
            raise ReportManifestError(detail=detail)
        return self


class StudentRunManifest(_ManifestBase):
    """Student/probe reporting lane; it cannot represent a teacher-oracle run."""

    result_class: Literal["contract_probe", "mock", "heuristic", "student", "official"]
    execution_profile: Literal["probe", "full", "not-run"] = "not-run"
    checkpoint_sha256: str | None = Field(default=None, pattern=_SHA256)
    typed_memory_sha256: str | None = Field(default=None, pattern=_SHA256)
    inference_manifest_sha256: str | None = Field(default=None, pattern=_SHA256)
    evidence_sha256: str | None = Field(default=None, pattern=_SHA256)
    evidence_lineage_sha256: str | None = Field(default=None, pattern=_SHA256)
    model_sha256: str | None = Field(default=None, pattern=_SHA256)
    prompt_sha256: str | None = Field(default=None, pattern=_SHA256)
    predictions_sha256: str | None = Field(default=None, pattern=_SHA256)
    metrics_sha256: str | None = Field(default=None, pattern=_SHA256)
    qa_resume_manifest_sha256: str | None = Field(default=None, pattern=_SHA256)
    run_identity_sha256: str | None = Field(default=None, pattern=_SHA256)
    finalization_inputs_sha256: str | None = Field(default=None, pattern=_SHA256)

    @model_validator(mode="after")
    def _require_complete_student_identity(self) -> Self:
        if self.remote_status != "complete":
            return self
        if self.result_class == "official":
            raise ReportManifestError(
                detail=(
                    "official completion requires per-experiment immutable manifests "
                    "(not implemented)"
                )
            )
        if self.result_class in {"mock", "heuristic"}:
            raise ReportManifestError(
                detail=f"{self.result_class} result cannot claim remote completion"
            )
        if self.result_class not in {"student", "contract_probe"}:
            return self
        expected_profile = "probe" if self.result_class == "contract_probe" else "full"
        if self.execution_profile != expected_profile:
            raise ReportManifestError(
                detail=(
                    f"complete {self.result_class} result requires "
                    f"execution_profile={expected_profile}"
                )
            )
        digest_values = (
            ("code_sha256", self.code_sha256),
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("typed_memory_sha256", self.typed_memory_sha256),
            ("inference_manifest_sha256", self.inference_manifest_sha256),
            ("evidence_sha256", self.evidence_sha256),
            ("evidence_lineage_sha256", self.evidence_lineage_sha256),
            ("model_sha256", self.model_sha256),
            ("prompt_sha256", self.prompt_sha256),
            ("predictions_sha256", self.predictions_sha256),
            ("metrics_sha256", self.metrics_sha256),
            ("qa_resume_manifest_sha256", self.qa_resume_manifest_sha256),
            ("run_identity_sha256", self.run_identity_sha256),
            ("finalization_inputs_sha256", self.finalization_inputs_sha256),
        )
        missing = tuple(name for name, value in digest_values if value is None)
        if missing:
            raise ReportManifestError(
                detail=(
                    f"complete {self.result_class} result missing digests: "
                    f"{', '.join(missing)}"
                )
            )
        if self.lane != "student":
            raise ReportManifestError(
                detail=f"complete {self.result_class} result requires lane=student"
            )
        if re.fullmatch(_SHA256, self.split_id) is None:
            raise ReportManifestError(
                detail=(
                    f"complete {self.result_class} result requires a SHA-256 split_id"
                )
            )
        keys = tuple((metric.experiment, metric.name) for metric in self.metrics)
        if len(keys) != len(set(keys)):
            raise ReportManifestError(
                detail=f"complete {self.result_class} result contains duplicate metrics"
            )
        expected_experiment = "PROBE" if self.result_class == "contract_probe" else "E1"
        if any(experiment != expected_experiment for experiment, _ in keys):
            raise ReportManifestError(
                detail=(
                    f"complete {self.result_class} result may contain only "
                    f"{expected_experiment} metrics"
                )
            )
        core = {
            metric.name: metric.value
            for metric in self.metrics
            if metric.name in {"Ans-F1", "QA-Acc", "QA-MRR"}
        }
        if core.keys() != {"Ans-F1", "QA-Acc", "QA-MRR"}:
            raise ReportManifestError(
                detail=(
                    f"complete {self.result_class} result requires Ans-F1, QA-Acc, "
                    "and QA-MRR"
                )
            )
        if any(not 0.0 <= value <= METRIC_PERCENT_MAX for value in core.values()):
            raise ReportManifestError(
                detail=(
                    f"complete {self.result_class} core metrics must be between 0 "
                    "and 100"
                )
            )
        return self


class OracleRunManifest(_ManifestBase):
    """Teacher-oracle lane, deliberately disjoint from checkpoint/inference fields."""

    result_class: Literal["teacher_oracle"] = "teacher_oracle"
    execution_profile: Literal["teacher-oracle", "provider-audit"]
    manifest_schema: (
        Literal["provider_gate_terminal_v1", "oracle_variants_terminal_v1"] | None
    ) = None

    @model_validator(mode="before")
    @classmethod
    def _default_oracle_lane(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = _PAYLOAD_ADAPTER.validate_python(value)
        if "lane" not in payload:
            return {**payload, "lane": "teacher_oracle"}
        return payload

    scientific_decision: (
        Literal["go", "no_go", "not_measurable", "not_decidable"] | None
    ) = None
    operational_decision: Literal["pending", "accepted", "rejected"] | None = None
    scientific_thresholds_sha256: str | None = Field(default=None, pattern=_SHA256)
    scientific_evidence_sha256: str | None = Field(default=None, pattern=_SHA256)
    scientific_terminal_reason: str | None = None
    sensor_audit_sha256: str | None = Field(default=None, pattern=_SHA256)
    object_semantic_sha256: str | None = Field(default=None, pattern=_SHA256)
    geometry_sha256: str | None = Field(default=None, pattern=_SHA256)
    place_sha256: str | None = Field(default=None, pattern=_SHA256)
    typed_memory_sha256: str | None = Field(default=None, pattern=_SHA256)
    shared_input_sha256: str | None = Field(default=None, pattern=_SHA256)
    qa_model_sha256: str | None = Field(default=None, pattern=_SHA256)
    qa_prompt_sha256: str | None = Field(default=None, pattern=_SHA256)
    qa_seed: int | None = None
    qa_question_map_sha256: str | None = Field(default=None, pattern=_SHA256)
    qa_approved_salt: str | None = None
    qa_world_size: int | None = Field(default=None, ge=1)
    oracle_lineage_sha256: str | None = Field(default=None, pattern=_SHA256)
    qa_decoding_sha256: str | None = Field(default=None, pattern=_SHA256)
    qa_runtime_sha256: str | None = Field(default=None, pattern=_SHA256)
    qa_pre_evaluation_sha256: str | None = Field(default=None, pattern=_SHA256)
    qa_python_inventory_sha256: str | None = Field(default=None, pattern=_SHA256)
    qa_torch_inventory_sha256: str | None = Field(default=None, pattern=_SHA256)
    qa_transformers_inventory_sha256: str | None = Field(default=None, pattern=_SHA256)
    variants: tuple[OracleVariantLineage, ...] = ()
    provider_result_sha256: str | None = Field(default=None, pattern=_SHA256)
    provider_outcome: (
        Literal[
            "provider_result",
            "contract_ineligible",
            "producer_failure",
            "gate_failure",
            "cancelled_pre_continuation",
        ]
        | None
    ) = None
    provider_failure_sha256: str | None = Field(default=None, pattern=_SHA256)
    producer_status_sha256: str | None = Field(default=None, pattern=_SHA256)
    approval_sha256: str | None = Field(default=None, pattern=_SHA256)
    preflight_sha256: str | None = Field(default=None, pattern=_SHA256)
    payload_sha256: str | None = Field(default=None, pattern=_SHA256)
    outcome_sha256: str | None = Field(default=None, pattern=_SHA256)
    continuation_sha256: str | None = Field(default=None, pattern=_SHA256)
    continuation_consume_sha256: str | None = Field(default=None, pattern=_SHA256)
    operational_artifact_sha256: str | None = Field(default=None, pattern=_SHA256)
    scientific_artifact_sha256: str | None = Field(default=None, pattern=_SHA256)
    oracle_lineage: OracleEvidenceLineage | None = None

    @field_validator("scientific_terminal_reason")
    @classmethod
    def _strip_terminal_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ReportManifestError(
                detail="scientific_terminal_reason must not be blank"
            )
        return value

    @model_validator(mode="after")
    def _require_oracle_lane(self) -> Self:
        if self.lane != "teacher_oracle":
            raise ReportManifestError(
                detail="oracle result requires lane=teacher_oracle"
            )
        return self

    @model_validator(mode="after")
    def _require_oracle_completion(self) -> Self:
        self._require_profile_experiment_mapping()
        if self.execution_profile == "provider-audit":
            self._require_terminal_schema()
            self._require_provider_audit_terminal()
            return self
        self._require_exp0005_profile()
        return self

    def _require_terminal_schema(self) -> None:
        if self.execution_profile == "provider-audit":
            if self.manifest_schema != "provider_gate_terminal_v1":
                raise ReportManifestError(
                    detail=(
                        "provider-audit requires "
                        "manifest_schema=provider_gate_terminal_v1"
                    )
                )
            return
        if self.manifest_schema not in {
            "provider_gate_terminal_v1",
            "oracle_variants_terminal_v1",
        }:
            raise ReportManifestError(
                detail="oracle result requires a recognized terminal manifest schema"
            )

    def _require_profile_experiment_mapping(self) -> None:
        expected_experiment = (
            "EXP-0005" if self.execution_profile == "teacher-oracle" else "EXP-0004"
        )
        if self.experiment_id != expected_experiment:
            detail = (
                f"{self.execution_profile} requires experiment_id={expected_experiment}"
            )
            raise ReportManifestError(detail=detail)

    def _require_no_pending_decisions(self) -> None:
        if (
            self.scientific_decision is not None
            or self.operational_decision is not None
        ):
            detail = "oracle decisions require operational completion"
            raise ReportManifestError(detail=detail)

    def _require_provider_audit_terminal(self) -> None:
        if self.remote_status == "pending":
            self._require_no_pending_decisions()
            return
        excluded = {
            "scientific_decision",
            "scientific_thresholds_sha256",
            "scientific_evidence_sha256",
            "scientific_terminal_reason",
            "sensor_audit_sha256",
            "object_semantic_sha256",
            "geometry_sha256",
            "place_sha256",
            "typed_memory_sha256",
            "shared_input_sha256",
            "qa_model_sha256",
            "qa_prompt_sha256",
            "qa_seed",
            "qa_question_map_sha256",
            "qa_approved_salt",
            "qa_world_size",
            "qa_decoding_sha256",
            "qa_runtime_sha256",
            "qa_pre_evaluation_sha256",
            "qa_python_inventory_sha256",
            "qa_torch_inventory_sha256",
            "qa_transformers_inventory_sha256",
            "oracle_lineage_sha256",
            "variants",
            "metrics",
        }
        if self._has_scientific_evidence() or self.variants or self.metrics:
            detail = (
                "provider-audit terminal must not contain scientific decision or "
                "evidence"
            )
            raise ReportManifestError(detail=detail)
        expected_decisions = {
            "complete": "accepted",
            "failed": "rejected",
            "cancelled": "rejected",
        }
        if self.operational_decision != expected_decisions[self.remote_status]:
            raise ReportManifestError(
                detail="provider-audit decision violates the provider truth table"
            )
        if self.model_fields_set.intersection(excluded):
            detail = "provider_gate_terminal_v1 contains excluded oracle fields"
            raise ReportManifestError(detail=detail)
        permitted_outcomes = {
            "complete": {"provider_result"},
            "failed": {"producer_failure", "gate_failure"},
            "cancelled": {"cancelled_pre_continuation"},
        }
        if self.provider_outcome not in permitted_outcomes[self.remote_status]:
            raise ReportManifestError(
                detail="provider-audit outcome violates the provider truth table"
            )
        self._require_oracle_digest_set(full=False)

    def _has_scientific_evidence(self) -> bool:
        return any(
            value is not None
            for value in (
                self.scientific_decision,
                self.scientific_thresholds_sha256,
                self.scientific_evidence_sha256,
                self.scientific_terminal_reason,
                self.oracle_lineage_sha256,
            )
        )

    def _require_exp0005_profile(self) -> None:
        self._require_terminal_schema()
        if self.manifest_schema == "provider_gate_terminal_v1":
            self._require_early_exp0005_terminal()
            return
        self._require_full_exp0005_terminal()

    def _require_early_exp0005_terminal(self) -> None:
        if self.operational_decision != "rejected":
            raise ReportManifestError(
                detail=(
                    "early EXP-0005 terminal manifests require "
                    "operational_decision=rejected"
                )
            )
        if self.scientific_decision is None:
            raise ReportManifestError(
                detail="early EXP-0005 terminal requires a scientific decision"
            )
        permitted = {
            ("complete", "no_go"): {"provider_result"},
            ("complete", "not_measurable"): {"provider_result"},
            (
                "failed",
                "not_decidable",
            ): {"contract_ineligible", "producer_failure", "gate_failure"},
            ("cancelled", "not_decidable"): {"cancelled_pre_continuation"},
        }
        key = (self.remote_status, self.scientific_decision)
        if key not in permitted or self.provider_outcome not in permitted[key]:
            raise ReportManifestError(
                detail=(
                    "early EXP-0005 terminal outcome violates the "
                    "provider/gate truth table"
                )
            )
        if self.remote_status == "complete":
            if self.failure_reason is not None:
                raise ReportManifestError(
                    detail="complete early terminal must not contain failure_reason"
                )
        elif not self.failure_reason:
            raise ReportManifestError(
                detail="failed or cancelled early terminal requires failure_reason"
            )
        if self.metrics or self.variants:
            raise ReportManifestError(
                detail=(
                    "early EXP-0005 terminal manifests must not contain "
                    "Phase-B variants or metrics"
                )
            )
        self._require_oracle_digest_set(full=False)

    def _require_full_exp0005_terminal(self) -> None:
        if self.provider_outcome != "provider_result":
            raise ReportManifestError(
                detail="full EXP-0005 terminal requires a provider result"
            )
        if self.remote_status == "complete":
            if self.operational_decision != "accepted":
                raise ReportManifestError(
                    detail=(
                        "complete full EXP-0005 terminal manifests require "
                        "operational_decision=accepted"
                    )
                )
            self._require_oracle_digest_set(full=True)
            self._require_oracle_variants()
            self._require_oracle_identity()
            self._require_scientific_decision()
            return
        if (
            self.remote_status not in {"failed", "cancelled"}
            or self.operational_decision != "rejected"
            or self.scientific_decision != "not_decidable"
        ):
            raise ReportManifestError(
                detail=(
                    "full EXP-0005 terminal outcome violates the downstream truth table"
                )
            )
        if self.metrics or self.variants or not self.failure_reason:
            raise ReportManifestError(
                detail=(
                    "failed or cancelled full EXP-0005 terminal must contain "
                    "failure_reason "
                    "and no Phase-B variants or metrics"
                )
            )
        self._require_oracle_digest_set(full=True)

    def _require_oracle_digest_set(self, *, full: bool) -> None:
        required = (
            "producer_status_sha256",
            "approval_sha256",
            "preflight_sha256",
            "payload_sha256",
            "outcome_sha256",
            "operational_artifact_sha256",
            "scientific_artifact_sha256",
        )
        if full:
            required += ("continuation_sha256", "continuation_consume_sha256")
        self._require_provider_artifact()
        missing = tuple(name for name in required if getattr(self, name) is None)
        if missing:
            raise ReportManifestError(
                detail=(
                    "EXP-0005 terminal manifest missing mandatory digests: "
                    f"{', '.join(missing)}"
                )
            )
        if not full and (
            self.continuation_sha256 is not None
            or self.continuation_consume_sha256 is not None
        ):
            raise ReportManifestError(
                detail=(
                    "early EXP-0005 terminal manifests must not contain "
                    "continuation evidence"
                )
            )

    def _require_provider_artifact(self) -> None:
        result_outcomes = {"provider_result", "contract_ineligible"}
        failure_outcomes = {
            "producer_failure",
            "gate_failure",
            "cancelled_pre_continuation",
        }
        if self.provider_outcome is None:
            raise ReportManifestError(
                detail="EXP-0005 terminal requires a provider_outcome discriminator"
            )
        expected_field = (
            "provider_result_sha256"
            if self.provider_outcome in result_outcomes
            else "provider_failure_sha256"
            if self.provider_outcome in failure_outcomes
            else None
        )
        if expected_field is None:
            raise ReportManifestError(
                detail="provider_outcome is not a recognized provider artifact outcome"
            )
        provider_fields = (
            ("provider_result_sha256", self.provider_result_sha256),
            ("provider_failure_sha256", self.provider_failure_sha256),
        )
        present = tuple(name for name, value in provider_fields if value is not None)
        if present != (expected_field,):
            raise ReportManifestError(
                detail=(
                    "provider_outcome requires exactly its matching provider "
                    "result or failure digest"
                )
            )

    def _require_oracle_identity(self) -> None:
        if self.code_sha256 is None:
            raise ReportManifestError(
                detail="complete oracle result requires immutable code_sha256"
            )
        if re.fullmatch(_SHA256, self.split_id) is None:
            raise ReportManifestError(
                detail="complete oracle result requires a SHA-256 split_id"
            )
        required_digests = (
            ("sensor_audit_sha256", self.sensor_audit_sha256),
            ("object_semantic_sha256", self.object_semantic_sha256),
            ("geometry_sha256", self.geometry_sha256),
            ("place_sha256", self.place_sha256),
            ("typed_memory_sha256", self.typed_memory_sha256),
            ("shared_input_sha256", self.shared_input_sha256),
            ("qa_model_sha256", self.qa_model_sha256),
            ("qa_prompt_sha256", self.qa_prompt_sha256),
            ("qa_question_map_sha256", self.qa_question_map_sha256),
            ("qa_approved_salt", self.qa_approved_salt),
            ("qa_world_size", str(self.qa_world_size) if self.qa_world_size else None),
            ("qa_decoding_sha256", self.qa_decoding_sha256),
            ("qa_runtime_sha256", self.qa_runtime_sha256),
            ("qa_pre_evaluation_sha256", self.qa_pre_evaluation_sha256),
            (
                "qa_python_inventory_sha256",
                self.qa_python_inventory_sha256,
            ),
            ("qa_torch_inventory_sha256", self.qa_torch_inventory_sha256),
            (
                "qa_transformers_inventory_sha256",
                self.qa_transformers_inventory_sha256,
            ),
            ("oracle_lineage_sha256", self.oracle_lineage_sha256),
        )
        missing = tuple(name for name, value in required_digests if value is None)
        if self.qa_seed is None:
            missing += ("qa_seed",)
        if self.qa_world_size is None:
            missing += ("qa_world_size",)
        if missing:
            raise ReportManifestError(
                detail=(f"complete oracle result missing digests: {', '.join(missing)}")
            )
        shared_qa_lineage = SharedQALineage(
            approved_salt=cast("str", self.qa_approved_salt),
            world_size=cast("int", self.qa_world_size),
            question_map_sha256=cast("str", self.qa_question_map_sha256),
            model_sha256=cast("str", self.qa_model_sha256),
            prompt_sha256=cast("str", self.qa_prompt_sha256),
            decoding_sha256=cast("str", self.qa_decoding_sha256),
            runtime_sha256=cast("str", self.qa_runtime_sha256),
            python_inventory_sha256=cast("str", self.qa_python_inventory_sha256),
            torch_inventory_sha256=cast("str", self.qa_torch_inventory_sha256),
            transformers_inventory_sha256=cast(
                "str", self.qa_transformers_inventory_sha256
            ),
            seed=cast("int", self.qa_seed),
        )
        if shared_qa_lineage.sha256 != self.qa_pre_evaluation_sha256:
            raise ReportManifestError(
                detail=(
                    "qa_pre_evaluation_sha256 does not match canonical shared QA "
                    "lineage"
                )
            )
        if self.oracle_lineage is None or self.oracle_lineage_sha256 is None:
            raise ReportManifestError(
                detail="full EXP-0005 terminal requires canonical oracle lineage"
            )
        if self.oracle_lineage.sha256 != self.oracle_lineage_sha256:
            raise ReportManifestError(
                detail="oracle_lineage_sha256 does not match canonical oracle lineage"
            )
        if self.oracle_lineage.shared_qa_lineage != shared_qa_lineage:
            raise ReportManifestError(
                detail="canonical oracle lineage does not match shared QA lineage"
            )
        duplicated_digests = (
            ("sensor_audit_sha256", self.sensor_audit_sha256),
            ("object_semantic_sha256", self.object_semantic_sha256),
            ("geometry_sha256", self.geometry_sha256),
            ("place_sha256", self.place_sha256),
            ("typed_memory_sha256", self.typed_memory_sha256),
            ("shared_input_sha256", self.shared_input_sha256),
        )
        if any(
            getattr(self.oracle_lineage, name) != value
            for name, value in duplicated_digests
        ):
            raise ReportManifestError(
                detail=(
                    "canonical oracle lineage does not match top-level evidence digests"
                )
            )
        if self.oracle_lineage.variants != self.variants:
            raise ReportManifestError(
                detail=(
                    "canonical oracle lineage variants do not match manifest variants"
                )
            )

    def _require_oracle_variants(self) -> None:
        variants = tuple(item.variant for item in self.variants)
        if set(variants) != {"E0", "T0", "T1"} or len(variants) != ORACLE_VARIANT_COUNT:
            raise ReportManifestError(
                detail=(
                    "complete oracle result requires exactly E0, T0, and T1 variants"
                )
            )
        if any(
            item.pre_evaluation_sha256 != self.qa_pre_evaluation_sha256
            for item in self.variants
        ):
            raise ReportManifestError(
                detail="E0, T0, and T1 must bind the same pre-evaluation QA lineage"
            )

    def _require_scientific_decision(self) -> None:
        if self.scientific_decision is None:
            raise ReportManifestError(
                detail="complete oracle result requires scientific_decision"
            )
        if self.scientific_decision in {"not_measurable", "not_decidable"}:
            self._require_terminal_scientific_decision()
            return
        self._require_measurable_scientific_decision()

    def _require_terminal_scientific_decision(self) -> None:
        if not self.scientific_terminal_reason:
            raise ReportManifestError(
                detail=(
                    "non-measurable oracle decision requires scientific_terminal_reason"
                )
            )
        if self.metrics:
            raise ReportManifestError(
                detail="non-measurable oracle decision must not contain metrics"
            )

    def _require_measurable_scientific_decision(self) -> None:
        if self.scientific_terminal_reason is not None:
            raise ReportManifestError(
                detail=(
                    "measurable oracle decision must not contain "
                    "scientific_terminal_reason"
                )
            )
        if (
            self.scientific_thresholds_sha256 is None
            or self.scientific_evidence_sha256 is None
        ):
            raise ReportManifestError(
                detail=(
                    "measurable oracle decision requires frozen thresholds and evidence"
                )
            )
        self._require_oracle_metric_matrix()

    def _require_oracle_metric_matrix(self) -> None:
        keys = tuple((metric.experiment, metric.name) for metric in self.metrics)
        expected = {
            (variant, metric_name)
            for variant in ("E0", "T0", "T1")
            for metric_name in ORACLE_METRIC_NAMES
        }
        if set(keys) != expected or len(keys) != len(expected):
            raise ReportManifestError(
                detail=(
                    "complete oracle result requires the complete unique bounded "
                    "metric matrix"
                )
            )
        for metric in self.metrics:
            self._require_oracle_metric_range(metric)

    @staticmethod
    def _require_oracle_metric_range(metric: ReportMetric) -> None:
        upper = 1.0 if metric.name in {"Coverage", "Fairness"} else METRIC_PERCENT_MAX
        if not 0.0 <= metric.value <= upper:
            raise ReportManifestError(
                detail=(
                    "oracle metrics must satisfy bounded score, coverage, and "
                    "fairness ranges"
                )
            )


type RunManifest = Annotated[
    StudentRunManifest | OracleRunManifest, Field(discriminator="result_class")
]
_run_manifest_adapter: TypeAdapter[RunManifest] = TypeAdapter(RunManifest)
# Legacy API remains student-only so callers cannot silently parse an oracle manifest
# as student.
RemoteRunManifest = StudentRunManifest


def write_report(run_manifest: Path, output: Path) -> None:
    """Render a handoff report from a persisted run manifest."""
    manifest = read_run_manifest(run_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        _ = temporary.write_text(render_report(manifest), encoding="utf-8")
        _ = temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def read_run_manifest(path: Path) -> RunManifest:
    """Load and validate a discriminated run manifest."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IncompleteRemoteManifestError(path, str(exc)) from exc
    try:
        return _run_manifest_adapter.validate_json(raw)
    except ValidationError as exc:
        raise IncompleteRemoteManifestError(path, _validation_detail(exc)) from exc


def render_report(manifest: RunManifest) -> str:
    """Render a sanitized Markdown handoff report."""
    lines = [
        "# WorldMM-SMVQA Final Result Handoff",
        "",
        f"remote_status: {_plain(manifest.remote_status)}",
        f"baseline_name: {_plain(manifest.baseline_name)}",
        f"result_class: {_plain(manifest.result_class)}",
        f"experiment_id: {_plain(manifest.experiment_id)}",
        f"execution_profile: {_plain(manifest.execution_profile)}",
        f"lane: {_plain(manifest.lane)}",
        f"split_id: {_plain(manifest.split_id)}",
        f"code_sha256: {_plain(manifest.code_sha256 or 'not-recorded')}",
    ]
    for name, value in _digest_values(manifest):
        lines.append(f"{name}: {_plain(value or 'not-recorded')}")
    if isinstance(manifest, OracleRunManifest):
        scientific_decision = _plain(manifest.scientific_decision or "not-recorded")
        lines.append(f"scientific_decision: {scientific_decision}")
        operational_decision = _plain(manifest.operational_decision or "not-recorded")
        lines.append(f"operational_decision: {operational_decision}")
        lines.append(
            f"provider_outcome: {_plain(manifest.provider_outcome or 'not-recorded')}"
        )
        variants = ",".join(item.variant for item in manifest.variants)
        lines.append(f"oracle_variants: {_plain(variants or 'not-recorded')}")
        for variant in manifest.variants:
            lines.append(
                f"{variant.variant}_predictions_sha256: "
                f"{_plain(variant.predictions_sha256)}"
            )
            lines.append(
                f"{variant.variant}_finalization_receipt_sha256: "
                f"{_plain(variant.finalization_receipt_sha256)}"
            )
            lines.append(
                f"{variant.variant}_finalization_receipt_file_sha256: "
                f"{_plain(variant.finalization_receipt_file_sha256)}"
            )
    lines.extend(
        (
            "",
            "## Local code/config changed",
            *_bullets(manifest.local_changes),
            "",
            "## Remote command used",
            *_remote_command_lines(manifest),
            "",
            "## Remote job ID or process reference",
            _markdown_text(manifest.remote_job_reference),
            "",
            "## Remote artifact path on company storage",
            _markdown_text(manifest.remote_artifact_path),
            "",
            "## Key metrics or failure reason",
            *_metric_lines(manifest),
            "",
            "## What was not copied locally",
            *_bullets(manifest.not_copied_locally),
            "",
        )
    )
    return "\n".join(lines)


def _remote_command_lines(manifest: RunManifest) -> tuple[str, ...]:
    if manifest.remote_status == "pending":
        return (
            "Remote command not run.",
            "Planned remote command:",
            _code_block(_plain(manifest.remote_command)),
        )
    return (_code_block(_plain(manifest.remote_command)),)


def _digest_values(manifest: RunManifest) -> tuple[tuple[str, str | None], ...]:
    if isinstance(manifest, OracleRunManifest):
        return (
            ("provider_result_sha256", manifest.provider_result_sha256),
            ("provider_failure_sha256", manifest.provider_failure_sha256),
            ("producer_status_sha256", manifest.producer_status_sha256),
            ("approval_sha256", manifest.approval_sha256),
            ("preflight_sha256", manifest.preflight_sha256),
            ("payload_sha256", manifest.payload_sha256),
            ("outcome_sha256", manifest.outcome_sha256),
            ("continuation_sha256", manifest.continuation_sha256),
            ("continuation_consume_sha256", manifest.continuation_consume_sha256),
            ("operational_artifact_sha256", manifest.operational_artifact_sha256),
            ("scientific_artifact_sha256", manifest.scientific_artifact_sha256),
            ("scientific_thresholds_sha256", manifest.scientific_thresholds_sha256),
            ("scientific_evidence_sha256", manifest.scientific_evidence_sha256),
            ("sensor_audit_sha256", manifest.sensor_audit_sha256),
            ("object_semantic_sha256", manifest.object_semantic_sha256),
            ("geometry_sha256", manifest.geometry_sha256),
            ("place_sha256", manifest.place_sha256),
            ("typed_memory_sha256", manifest.typed_memory_sha256),
            ("shared_input_sha256", manifest.shared_input_sha256),
            ("qa_model_sha256", manifest.qa_model_sha256),
            ("qa_prompt_sha256", manifest.qa_prompt_sha256),
            ("qa_question_map_sha256", manifest.qa_question_map_sha256),
            ("qa_approved_salt", manifest.qa_approved_salt),
            (
                "qa_world_size",
                str(manifest.qa_world_size) if manifest.qa_world_size else None,
            ),
            ("qa_decoding_sha256", manifest.qa_decoding_sha256),
            ("qa_runtime_sha256", manifest.qa_runtime_sha256),
            ("qa_pre_evaluation_sha256", manifest.qa_pre_evaluation_sha256),
            ("oracle_lineage_sha256", manifest.oracle_lineage_sha256),
        )
    return (
        ("checkpoint_sha256", manifest.checkpoint_sha256),
        ("typed_memory_sha256", manifest.typed_memory_sha256),
        ("inference_manifest_sha256", manifest.inference_manifest_sha256),
        ("evidence_sha256", manifest.evidence_sha256),
        ("evidence_lineage_sha256", manifest.evidence_lineage_sha256),
        ("model_sha256", manifest.model_sha256),
        ("prompt_sha256", manifest.prompt_sha256),
        ("predictions_sha256", manifest.predictions_sha256),
        ("metrics_sha256", manifest.metrics_sha256),
        ("qa_resume_manifest_sha256", manifest.qa_resume_manifest_sha256),
        ("run_identity_sha256", manifest.run_identity_sha256),
        ("finalization_inputs_sha256", manifest.finalization_inputs_sha256),
    )


def _metric_lines(manifest: RunManifest) -> tuple[str, ...]:
    if manifest.metrics:
        return tuple(
            f"- [{metric.experiment}] {_markdown_text(metric.name)}: {metric.value:.6f}"
            for metric in manifest.metrics
        )
    if (
        isinstance(manifest, OracleRunManifest)
        and manifest.remote_status == "complete"
        and manifest.execution_profile == "provider-audit"
    ):
        return (
            "Provider audit completed: "
            f"{_markdown_text(manifest.operational_decision or 'not-recorded')}.",
        )
    if (
        isinstance(manifest, OracleRunManifest)
        and manifest.remote_status == "complete"
        and manifest.scientific_decision in {"not_measurable", "not_decidable"}
    ):
        terminal_reason = _markdown_text(
            manifest.scientific_terminal_reason or "Scientific decision unavailable."
        )
        return (f"No remote metrics reported. {terminal_reason}",)
    failure_reason = _markdown_text(
        manifest.failure_reason or "Remote benchmark not run."
    )
    return (f"No remote metrics reported. {failure_reason}",)


def _bullets(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"- {_markdown_text(value)}" for value in values)


def _code_block(value: str) -> str:
    return f"```bash\n{value}\n```"


def _validation_detail(exc: ValidationError) -> str:
    return " ".join(
        line.split("[type=", 1)[0].rstrip() if "input_value=" in line else line
        for line in str(exc).splitlines()
        if "For further information" not in line
    )


def _plain(value: str) -> str:
    text = " ".join(value.replace("`", "'").split())
    return (
        "[redacted sensitive manifest text]"
        if any(marker in text.lower() for marker in SENSITIVE_MARKERS)
        else text
    )


def _markdown_text(value: str) -> str:
    return "".join(
        f"\\{character}" if character in MARKDOWN_SPECIALS else character
        for character in _plain(value)
    )
