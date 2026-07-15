from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, ClassVar, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

PROTOCOL = "worldmm-model-load-v1"
PROVIDER_TIMEOUT_SECONDS = 900
MAX_REQUEST_BYTES = 32 * 1024
MAX_STDOUT_BYTES = 64 * 1024
MAX_STDERR_BYTES = 256 * 1024
MAX_DIAGNOSTIC_BYTES = 4096
ROLE_ORDER = (
    "gcut3r_teacher",
    "qwen_memory",
    "spatial_trainable",
    "gemma_qa",
)

type Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
type ModelRole = Literal[
    "gcut3r_teacher",
    "qwen_memory",
    "spatial_trainable",
    "gemma_qa",
    "spatial_inference",
]
type LoadMode = Literal[
    "fresh_constructor", "resume_checkpoint", "model_tree", "post_training_checkpoint"
]
type ProviderErrorCodeV1 = Literal[
    "invalid_request",
    "forbidden_environment",
    "network_attempt",
    "missing_artifact",
    "artifact_digest_mismatch",
    "loader_identity_mismatch",
    "loaded_class_mismatch",
    "revision_mismatch",
    "device_mismatch",
    "load_failure",
    "cleanup_failure",
    "timeout",
    "signal_death",
    "invalid_provider_output",
    "output_too_large",
    "exit_result_mismatch",
    "not_attempted_after_cleanup_failure",
]
type AuthorizationFailureCodeV1 = Literal[
    "submission_failed",
    "manifest_mismatch",
    "approval_missing",
    "approval_invalid",
    "release_failed",
    "release_uncertain",
    "worker_failed",
    "worker_missing",
    "provider_error",
    "receipt_invalid",
    "identity_mismatch",
    "physical_matrix_mismatch",
    "accounting_mismatch",
    "consensus_expired",
    "artifact_policy_violation",
    "downstream_release_failed",
    "downstream_cancel_failed",
    "terminal_incomplete",
]
type NormalizedSlurmStateV1 = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
    "timeout",
    "out_of_memory",
    "node_fail",
    "preempted",
    "boot_fail",
    "deadline",
    "revoked",
    "special_exit",
    "unknown",
]
type OperationalStateV1 = Literal["complete", "failed", "cancelled", "unknown"]
type ScientificStateV1 = Literal["complete", "not_decidable", "invalid"]
type RankCompletionStatusV1 = Literal["complete", "error", "partial"]
type StageControlObservedStateV1 = Literal[
    "held",
    "released",
    "cancel_requested",
    "cancelled",
    "running",
    "terminal",
    "unknown",
]
type ControlActionV1 = Literal["release", "cancel"]
type StageStateV1 = Literal["held", "released", "cancelled"]


class StrictModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class ExpectedDigestsV1(StrictModel):
    architecture_sha256: Sha256 | None = None
    config_sha256: Sha256 | None = None
    processor_sha256: Sha256 | None = None
    checkpoint_sha256: Sha256 | None = None
    checkpoint_tree_sha256: Sha256 | None = None
    model_tree_sha256: Sha256 | None = None
    executable_sha256: Sha256 | None = None
    origin_consensus_payload_sha256: Sha256 | None = None
    origin_consensus_file_sha256: Sha256 | None = None


class PhysicalDeviceV1(StrictModel):
    logical_device: Literal["cuda:0"]
    cuda_visible_device: str
    cuda_uuid: str
    pci_bus_id: str
    device_name: str
    total_memory_bytes: int = Field(ge=1)


class ActualDeviceV1(StrictModel):
    logical_device: Literal["cuda:0"]
    cuda_uuid: str
    pci_bus_id: str
    device_name: str
    total_memory_bytes: int = Field(ge=1)


class CleanupV1(StrictModel):
    model_released: bool
    processor_released: bool
    cuda_cache_cleared: bool


class ModelLoadRequestV1(StrictModel):
    schema_version: Literal["model_load_request_v1"] = "model_load_request_v1"
    protocol: Literal["worldmm-model-load-v1"] = PROTOCOL
    run_identity: dict[str, object]
    model_role: ModelRole
    load_mode: LoadMode
    model_path: str | None
    checkpoint_path: str | None
    config_path: str | None
    fixture_sha256: Sha256
    expected_loader_id: str
    expected_loaded_class: str
    expected_revision: str
    expected_digests: ExpectedDigestsV1
    logical_device: Literal["cuda:0"]
    physical_device: PhysicalDeviceV1
    timeout_seconds: Literal[900] = PROVIDER_TIMEOUT_SECONDS

    @model_validator(mode="after")
    def _role_mode_presence(self) -> ModelLoadRequestV1:
        d = self.expected_digests
        role_mode = (self.model_role, self.load_mode)
        allowed = {
            ("gcut3r_teacher", "model_tree"),
            ("qwen_memory", "model_tree"),
            ("gemma_qa", "model_tree"),
            ("spatial_trainable", "fresh_constructor"),
            ("spatial_trainable", "resume_checkpoint"),
            ("spatial_inference", "post_training_checkpoint"),
        }
        if role_mode not in allowed:
            msg = "model role/load mode combination is forbidden"
            raise ValueError(msg)
        if self.model_role in ("qwen_memory", "gemma_qa"):
            if (
                self.model_path is None
                or self.checkpoint_path is not None
                or self.config_path is not None
            ):
                msg = "transformer model_tree requires only model_path"
                raise ValueError(msg)
            _require_digests(
                d, "config_sha256", "processor_sha256", "model_tree_sha256"
            )
        elif self.model_role == "gcut3r_teacher":
            if self.model_path is None or self.checkpoint_path is None:
                msg = "G-CUT3R model_tree requires model and checkpoint paths"
                raise ValueError(msg)
            _require_digests(
                d, "checkpoint_sha256", "model_tree_sha256", "executable_sha256"
            )
        elif self.load_mode == "fresh_constructor":
            if (
                self.model_path is not None
                or self.checkpoint_path is not None
                or self.config_path is None
            ):
                msg = "fresh constructor requires only config_path"
                raise ValueError(msg)
            _require_digests(d, "architecture_sha256")
        elif self.load_mode == "resume_checkpoint":
            if (
                self.model_path is not None
                or self.checkpoint_path is None
                or self.config_path is None
            ):
                msg = "resume requires checkpoint_path and config_path"
                raise ValueError(msg)
            _require_digests(
                d,
                "architecture_sha256",
                "checkpoint_sha256",
                "origin_consensus_payload_sha256",
                "origin_consensus_file_sha256",
            )
        else:
            if self.model_path is not None or self.checkpoint_path is None:
                msg = "post-training inference requires checkpoint_path"
                raise ValueError(msg)
            _require_digests(d, "checkpoint_sha256", "executable_sha256")
        _validate_checkpoint_digest_union(d)
        return self


class ProviderResultV1(StrictModel):
    schema_version: Literal["model_load_provider_result_v1"] = (
        "model_load_provider_result_v1"
    )
    protocol: Literal["worldmm-model-load-v1"] = PROTOCOL
    status: Literal["ok", "error"]
    model_role: ModelRole
    request_sha256: Sha256
    loader_id: str
    expected_loaded_class: str
    actual_loaded_class: str | None
    revision: str | None
    config_sha256: Sha256 | None
    processor_sha256: Sha256 | None
    checkpoint_sha256: Sha256 | None
    checkpoint_tree_sha256: Sha256 | None
    model_tree_sha256: Sha256 | None
    executable_sha256: Sha256 | None
    actual_device: ActualDeviceV1 | None
    cleanup: CleanupV1
    load_ok: bool
    error_code: ProviderErrorCodeV1 | None
    diagnostic: str

    @model_validator(mode="after")
    def _coherent_result(self) -> ProviderResultV1:
        if len(self.diagnostic.encode("utf-8")) > MAX_DIAGNOSTIC_BYTES:
            msg = "diagnostic exceeds 4096 bytes"
            raise ValueError(msg)
        if self.status == "ok":
            if not self.load_ok or self.error_code is not None:
                msg = "successful result has inconsistent status"
                raise ValueError(msg)
            if (
                self.actual_loaded_class is None
                or self.revision is None
                or self.actual_device is None
            ):
                msg = "successful result lacks identity"
                raise ValueError(msg)
            if not all(self.cleanup.model_dump().values()):
                msg = "successful result lacks cleanup acknowledgement"
                raise ValueError(msg)
        elif self.load_ok or self.error_code is None:
            msg = "error result has inconsistent status"
            raise ValueError(msg)
        if (
            self.checkpoint_sha256 is not None
            and self.checkpoint_tree_sha256 is not None
        ):
            msg = "file and directory checkpoint digests are mutually exclusive"
            raise ValueError(msg)
        return self


class ProviderLockRoleV1(StrictModel):
    model_role: ModelRole
    loader_id: str
    loaded_class: str
    revision: str
    config_sha256: Sha256 | None
    processor_sha256: Sha256 | None
    checkpoint_sha256: Sha256 | None
    checkpoint_tree_sha256: Sha256 | None
    model_tree_sha256: Sha256 | None
    executable_sha256: Sha256 | None
    conformance_success_sha256: Sha256
    conformance_error_sha256: Sha256


class ProviderLockV1(StrictModel):
    schema_version: Literal["model_load_provider_lock_v1"] = (
        "model_load_provider_lock_v1"
    )
    lock_id: str
    status: Literal["pending", "accepted"]
    model_contract_sha256: Sha256
    roles: tuple[ProviderLockRoleV1, ...]
    owner: str | None
    reviewer: str | None
    reviewed_at_ms: int | None
    signature_algorithm: Literal["Ed25519"] | None
    signature_key_id: str | None
    signature: str | None

    @model_validator(mode="after")
    def _accepted_is_complete(self) -> ProviderLockV1:
        if len({role.model_role for role in self.roles}) != len(self.roles):
            msg = "provider lock model roles must be unique"
            raise ValueError(msg)
        accepted_fields = (
            self.owner,
            self.reviewer,
            self.reviewed_at_ms,
            self.signature_algorithm,
            self.signature_key_id,
            self.signature,
        )
        if self.status == "accepted" and any(
            value is None for value in accepted_fields
        ):
            msg = "accepted provider lock is incomplete"
            raise ValueError(msg)
        if self.status == "pending" and any(
            value is not None for value in accepted_fields
        ):
            msg = "pending provider lock cannot contain accepted identity"
            raise ValueError(msg)
        return self


class PreApprovalRunIdentityV1(StrictModel):
    run_id: str
    run_nonce: Sha256
    controller_id: str
    output_root: str
    code_snapshot_sha256: Sha256
    preflight_contract_sha256: Sha256
    preflight_inputs_sha256: Sha256
    preflight_inventory_sha256: Sha256
    model_contract_sha256: Sha256
    provider_lock_sha256: Sha256
    student_architecture_sha256: Sha256
    plan_profile: Literal["student"]
    execution_profile: Literal["full", "probe"]
    launch_attempt: int = Field(ge=1)
    expected_nodes: int = Field(ge=1)
    expected_gpus_per_node: int = Field(ge=1)
    expected_world_size: int = Field(ge=1)

    @model_validator(mode="after")
    def _matrix_size(self) -> PreApprovalRunIdentityV1:
        if (
            self.expected_nodes * self.expected_gpus_per_node
            != self.expected_world_size
        ):
            msg = "expected physical matrix does not equal world size"
            raise ValueError(msg)
        if self.execution_profile == "full" and (
            self.expected_nodes,
            self.expected_gpus_per_node,
            self.expected_world_size,
        ) != (10, 8, 80):
            msg = "full execution requires the 10x8 physical matrix"
            raise ValueError(msg)
        if self.execution_profile == "probe" and (
            self.expected_nodes,
            self.expected_gpus_per_node,
            self.expected_world_size,
        ) != (1, 1, 1):
            msg = "probe execution requires a pinned 1x1 matrix"
            raise ValueError(msg)
        return self


class RunIdentityV1(PreApprovalRunIdentityV1):
    submission_manifest_file_sha256: Sha256
    operator_approval_file_sha256: Sha256


class AccountingRowV1(StrictModel):
    stage_id: str
    job_id: str
    array_task_id: int | None
    state: NormalizedSlurmStateV1
    exit_code: int | None
    signal: int | None
    node_list: tuple[str, ...]
    start_time_ms: int | None
    end_time_ms: int | None


class MatrixRowV1(StrictModel):
    rank: int = Field(ge=0)
    node_rank: int = Field(ge=0)
    local_rank: int = Field(ge=0)
    hostname: str
    cuda_uuid: str
    pci_bus_id: str
    model_role: ModelRole
    row_payload_sha256: Sha256


class ModelInvariantV1(StrictModel):
    model_role: ModelRole
    loader_id: str
    loaded_class: str
    revision: str
    config_sha256: Sha256 | None
    processor_sha256: Sha256 | None
    checkpoint_sha256: Sha256 | None
    checkpoint_tree_sha256: Sha256 | None
    model_tree_sha256: Sha256 | None
    executable_sha256: Sha256 | None


class PhysicalGpuRowV1(StrictModel):
    rank: int = Field(ge=0)
    node_rank: int = Field(ge=0)
    local_rank: int = Field(ge=0)
    hostname: str
    cuda_uuid: str
    pci_bus_id: str
    device_name: str
    total_memory_bytes: int = Field(ge=1)


class ArtifactDigestRowV1(StrictModel):
    relative_path: str
    file_sha256: Sha256


class RankStartReceiptV1(StrictModel):
    attempt: int = Field(ge=1)
    created_at_ms: int = Field(ge=0)
    hostname: str
    local_rank: int = Field(ge=0)
    model_role: ModelRole
    protocol: Literal["worldmm-model-load-v1"] = PROTOCOL
    rank: int = Field(ge=0)
    request_file_sha256: Sha256
    run_identity: RunIdentityV1
    schema_version: Literal["model_load_rank_start_v1"] = "model_load_rank_start_v1"
    world_size: int = Field(ge=1)


class RankModelLoadReceiptV1(StrictModel):
    schema_version: Literal["model_load_rank_receipt_v1"] = "model_load_rank_receipt_v1"
    receipt_kind: Literal["model_load_role_v1"] = "model_load_role_v1"
    run_identity: RunIdentityV1
    attempt: int = Field(ge=1)
    model_role: ModelRole
    rank: int = Field(ge=0)
    world_size: int = Field(ge=1)
    node_rank: int = Field(ge=0)
    local_rank: int = Field(ge=0)
    hostname: str
    cuda_visible_devices: str
    logical_device: Literal["cuda:0"]
    cuda_uuid: str
    pci_bus_id: str
    device_name: str
    total_memory_bytes: int = Field(ge=1)
    request_sha256: Sha256
    loader_id: str
    expected_loaded_class: str
    actual_loaded_class: str | None
    revision: str | None
    config_sha256: Sha256 | None
    processor_sha256: Sha256 | None
    checkpoint_sha256: Sha256 | None
    checkpoint_tree_sha256: Sha256 | None
    model_tree_sha256: Sha256 | None
    executable_sha256: Sha256 | None
    python_version: str
    torch_version: str
    transformers_version: str
    cuda_runtime_version: str
    cuda_driver_version: str
    started_at_ms: int = Field(ge=0)
    finished_at_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    cleanup: CleanupV1
    load_ok: bool
    error_code: ProviderErrorCodeV1 | None
    diagnostic: str


class RankCompleteReceiptV1(StrictModel):
    attempt: int = Field(ge=1)
    completed_at_ms: int = Field(ge=0)
    rank: int = Field(ge=0)
    result_file_sha256s: tuple[Sha256, ...]
    run_identity: RunIdentityV1
    schema_version: Literal["model_load_rank_complete_v1"] = (
        "model_load_rank_complete_v1"
    )
    status: RankCompletionStatusV1


type FailureCodeV1 = ProviderErrorCodeV1 | AuthorizationFailureCodeV1


class ModelLoadConsensusReceiptV1(StrictModel):
    schema_version: Literal["model_load_consensus_v1"] = "model_load_consensus_v1"
    receipt_kind: Literal["model_load_consensus_v1"] = "model_load_consensus_v1"
    run_identity: RunIdentityV1
    submission_manifest_file_sha256: Sha256
    worker_accounting: tuple[AccountingRowV1, ...]
    expected_matrix: tuple[MatrixRowV1, ...]
    observed_matrix: tuple[MatrixRowV1, ...]
    model_invariants: tuple[ModelInvariantV1, ...]
    physical_gpu_matrix: tuple[PhysicalGpuRowV1, ...]
    row_payload_sha256s: tuple[Sha256, ...]
    started_rank_count: int = Field(ge=0)
    completed_rank_count: int = Field(ge=0)
    consensus_ok: bool
    failure_codes: tuple[FailureCodeV1, ...]
    created_at_ms: int = Field(ge=0)
    expires_at_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered_and_consistent(self) -> ModelLoadConsensusReceiptV1:
        if self.expires_at_ms != self.created_at_ms + 21_600_000:
            msg = "consensus expiry must be exactly six hours"
            raise ValueError(msg)
        if tuple(self.failure_codes) != tuple(sorted(set(self.failure_codes))):
            msg = "failure codes must be sorted and unique"
            raise ValueError(msg)
        if self.consensus_ok != (not self.failure_codes):
            msg = "consensus status and failure codes disagree"
            raise ValueError(msg)
        if self.observed_matrix != tuple(
            sorted(self.observed_matrix, key=lambda row: (row.rank, row.model_role))
        ):
            msg = "observed matrix is not canonical"
            raise ValueError(msg)
        if self.physical_gpu_matrix != tuple(
            sorted(self.physical_gpu_matrix, key=lambda row: row.rank)
        ):
            msg = "physical GPU matrix is not rank ordered"
            raise ValueError(msg)
        return self


class ModelLoadContinueV1(StrictModel):
    schema_version: Literal["model_load_continue_v1"] = "model_load_continue_v1"
    receipt_kind: Literal["model_load_continue_v1"] = "model_load_continue_v1"
    run_identity: RunIdentityV1
    submission_manifest_file_sha256: Sha256
    consensus_payload_sha256: Sha256
    consensus_file_sha256: Sha256
    gate_job_id: str
    created_at_ms: int = Field(ge=0)
    expires_at_ms: int = Field(ge=0)


class ModelLoadTerminalV1(StrictModel):
    schema_version: Literal["model_load_terminal_v1"] = "model_load_terminal_v1"
    receipt_kind: Literal["model_load_terminal_v1"] = "model_load_terminal_v1"
    run_identity: RunIdentityV1
    submission_manifest_file_sha256: Sha256
    gate_accounting: tuple[AccountingRowV1, ...]
    consensus_file_sha256: Sha256 | None
    continue_file_sha256: Sha256 | None
    operational_state: OperationalStateV1
    scientific_state: ScientificStateV1
    failure_codes: tuple[FailureCodeV1, ...]
    created_at_ms: int = Field(ge=0)


class PreManifestControlReceiptV1(StrictModel):
    schema_version: Literal["stage_control_receipt_v1"] = "stage_control_receipt_v1"
    receipt_kind: Literal["premanifest_control_v1"]
    controller_id: str
    controller_epoch: int = Field(ge=1)
    run_id: str
    run_nonce: Sha256
    stage_id: str
    job_id: str
    action: ControlActionV1
    reason: AuthorizationFailureCodeV1 | Literal["approved_go"]
    requested_at_ms: int = Field(ge=0)
    observed_state: StageControlObservedStateV1
    observed_at_ms: int = Field(ge=0)
    preapproval_identity_payload_sha256: Sha256


class PreApprovalControlReceiptV1(StrictModel):
    schema_version: Literal["stage_control_receipt_v1"] = "stage_control_receipt_v1"
    receipt_kind: Literal["preapproval_control_v1"]
    controller_id: str
    controller_epoch: int = Field(ge=1)
    run_id: str
    run_nonce: Sha256
    stage_id: str
    job_id: str
    action: ControlActionV1
    reason: AuthorizationFailureCodeV1 | Literal["approved_go"]
    requested_at_ms: int = Field(ge=0)
    observed_state: StageControlObservedStateV1
    observed_at_ms: int = Field(ge=0)
    preapproval_identity_payload_sha256: Sha256
    submission_manifest_file_sha256: Sha256


class PostApprovalControlReceiptV1(StrictModel):
    schema_version: Literal["stage_control_receipt_v1"] = "stage_control_receipt_v1"
    receipt_kind: Literal["postapproval_control_v1"]
    controller_id: str
    controller_epoch: int = Field(ge=1)
    run_id: str
    run_nonce: Sha256
    stage_id: str
    job_id: str
    action: ControlActionV1
    reason: AuthorizationFailureCodeV1 | Literal["approved_go"]
    requested_at_ms: int = Field(ge=0)
    observed_state: StageControlObservedStateV1
    observed_at_ms: int = Field(ge=0)
    submission_manifest_file_sha256: Sha256
    operator_approval_file_sha256: Sha256


StageControlReceiptV1 = Annotated[
    PreManifestControlReceiptV1
    | PreApprovalControlReceiptV1
    | PostApprovalControlReceiptV1,
    Field(discriminator="receipt_kind"),
]


class ProposedStageActionV1(StrictModel):
    schema_version: Literal["proposed_stage_action_v1"] = "proposed_stage_action_v1"
    controller_id: str
    controller_epoch: int = Field(ge=1)
    arbitration_record_file_sha256: Sha256
    run_id: str
    run_nonce: Sha256
    stage_id: str
    job_id: str
    action: ControlActionV1
    reason: AuthorizationFailureCodeV1 | Literal["approved_go"]
    proposed_at_ms: int = Field(ge=0)
    identity_variant: Literal["premanifest", "preapproval", "postapproval"]
    identity_digest: Sha256


class LeasePayloadV1(StrictModel):
    prior_controller_id: str | None
    prior_controller_epoch: int | None
    prior_accounting_sha256: Sha256 | None


class PreApprovalActionIntentPayloadV1(StrictModel):
    identity_variant: Literal["preapproval"] = "preapproval"
    run_id: str
    run_nonce: Sha256
    stage_id: str
    job_id: str
    stage_sequence: int = Field(ge=1)
    prior_executed_receipt_sha256: Sha256 | None
    from_state: StageStateV1
    action: ControlActionV1
    to_state: StageStateV1
    reason: AuthorizationFailureCodeV1 | Literal["approved_go"]
    proposal_file_sha256: Sha256
    submission_manifest_file_sha256: Sha256

    @model_validator(mode="after")
    def _legal_transition(self) -> PreApprovalActionIntentPayloadV1:
        _check_transition(self.from_state, self.action, self.to_state)
        return self


class PostApprovalActionIntentPayloadV1(StrictModel):
    identity_variant: Literal["postapproval"] = "postapproval"
    run_id: str
    run_nonce: Sha256
    stage_id: str
    job_id: str
    stage_sequence: int = Field(ge=1)
    prior_executed_receipt_sha256: Sha256 | None
    from_state: StageStateV1
    action: ControlActionV1
    to_state: StageStateV1
    reason: AuthorizationFailureCodeV1 | Literal["approved_go"]
    proposal_file_sha256: Sha256
    submission_manifest_file_sha256: Sha256
    operator_approval_file_sha256: Sha256

    @model_validator(mode="after")
    def _legal_transition(self) -> PostApprovalActionIntentPayloadV1:
        _check_transition(self.from_state, self.action, self.to_state)
        return self


ActionIntentPayloadV1 = Annotated[
    PreApprovalActionIntentPayloadV1 | PostApprovalActionIntentPayloadV1,
    Field(discriminator="identity_variant"),
]


class ControllerArbitrationRecordV1(StrictModel):
    schema_version: Literal["controller_arbitration_record_v1"] = (
        "controller_arbitration_record_v1"
    )
    sequence: int = Field(ge=1)
    prior_record_file_sha256: Sha256 | None
    record_kind: Literal[
        "acquire", "heartbeat", "relinquish", "takeover", "action_intent"
    ]
    controller_id: str
    controller_epoch: int = Field(ge=1)
    controller_job_id: str
    created_at_ms: int = Field(ge=0)
    expires_at_ms: int | None
    payload: LeasePayloadV1 | ActionIntentPayloadV1

    @model_validator(mode="after")
    def _genesis_and_expiry(self) -> ControllerArbitrationRecordV1:
        if self.sequence == 1:
            if (
                self.record_kind != "acquire"
                or self.prior_record_file_sha256 is not None
                or self.controller_epoch != 1
            ):
                msg = "sequence one must be the unique acquire genesis"
                raise ValueError(msg)
            if not isinstance(self.payload, LeasePayloadV1) or any(
                value is not None
                for value in (
                    self.payload.prior_controller_id,
                    self.payload.prior_controller_epoch,
                    self.payload.prior_accounting_sha256,
                )
            ):
                msg = "genesis acquire requires null prior controller fields"
                raise ValueError(msg)
        elif self.prior_record_file_sha256 is None:
            msg = "non-genesis record requires prior file digest"
            raise ValueError(msg)
        if self.record_kind in ("acquire", "heartbeat", "takeover"):
            if self.expires_at_ms is None or not (
                self.created_at_ms < self.expires_at_ms <= self.created_at_ms + 45_000
            ):
                msg = "lease-bearing record requires bounded expiry"
                raise ValueError(msg)
            if not isinstance(self.payload, LeasePayloadV1):
                msg = "lease record requires lease payload"
                raise ValueError(msg)
        elif self.record_kind == "relinquish":
            if self.expires_at_ms is not None or not isinstance(
                self.payload, LeasePayloadV1
            ):
                msg = "relinquish has no expiry and requires lease payload"
                raise ValueError(msg)
        elif self.expires_at_ms is not None or isinstance(self.payload, LeasePayloadV1):
            msg_0 = "action intent is irrevocable and non-expiring"
            raise ValueError(msg_0)
        return self


class ArbitrationLogError(ValueError):
    pass


def _require_digests(value: ExpectedDigestsV1, *names: str) -> None:
    missing = [name for name in names if getattr(value, name) is None]
    if missing:
        msg = f"required expected digests absent: {','.join(missing)}"
        raise ValueError(msg)


def _validate_checkpoint_digest_union(value: ExpectedDigestsV1) -> None:
    if value.checkpoint_sha256 is not None and value.checkpoint_tree_sha256 is not None:
        msg = "file and directory checkpoint digests are mutually exclusive"
        raise ValueError(msg)


def _check_transition(
    from_state: StageStateV1, action: ControlActionV1, to_state: StageStateV1
) -> None:
    legal = {
        ("held", "release", "released"),
        ("held", "cancel", "cancelled"),
        ("released", "cancel", "cancelled"),
    }
    if (from_state, action, to_state) not in legal:
        msg = "illegal stage action transition"
        raise ValueError(msg)


def encode_control_json(value: BaseModel | Mapping[str, object]) -> bytes:
    raw = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    _reject_non_control(raw)
    return json.dumps(
        raw, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()


def _reject_non_control(value: object) -> None:
    if value is None or isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        msg = "control JSON forbids floats"
        raise TypeError(msg)
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        for key, nested in mapping.items():
            if not isinstance(key, str):
                msg = "control JSON object keys must be strings"
                raise TypeError(msg)
            _reject_non_control(nested)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for nested in value:
            _reject_non_control(nested)
        return
    msg = f"unsupported control JSON value: {type(value).__name__}"
    raise ValueError(msg)


def payload_sha256(value: BaseModel | Mapping[str, object]) -> str:
    return hashlib.sha256(encode_control_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_control_file(path: Path, value: BaseModel | Mapping[str, object]) -> str:
    data = encode_control_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as stream:
            _ = stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            msg = f"control artifact already exists: {path}"
            raise FileExistsError(msg) from None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def load_control_file(path: Path, model_type: type[BaseModel]) -> BaseModel:
    data = path.read_bytes()
    if not data.endswith(b"\n") or len(data.splitlines()) != 1:
        msg = "control file must contain exactly one LF-terminated JSON object"
        raise ValueError(msg)
    parsed = cast("object", json.loads(data, object_pairs_hook=_unique_object))
    model = model_type.model_validate(parsed)
    if encode_control_json(model) + b"\n" != data:
        msg = "control file is not canonical"
        raise ValueError(msg)
    return model


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"duplicate JSON key: {key}"
            raise ValueError(msg)
        result[key] = value
    return result


def arbitration_path(root: Path, sequence: int) -> Path:
    return root / f"{sequence:020d}.json"


def validate_arbitration_log(root: Path) -> tuple[ControllerArbitrationRecordV1, ...]:
    if not root.exists():
        return ()
    files = sorted(path for path in root.iterdir() if path.suffix == ".json")
    records: list[ControllerArbitrationRecordV1] = []
    prior_digest: str | None = None
    for expected_sequence, path in enumerate(files, 1):
        if path.name != f"{expected_sequence:020d}.json":
            msg = "arbitration log is gapped or has an invalid filename"
            raise ArbitrationLogError(msg)
        record = ControllerArbitrationRecordV1.model_validate_json(path.read_bytes())
        if (
            record.sequence != expected_sequence
            or record.prior_record_file_sha256 != prior_digest
        ):
            msg = "arbitration record does not bind the contiguous head"
            raise ArbitrationLogError(msg)
        if encode_control_json(record) + b"\n" != path.read_bytes():
            msg = "arbitration record is not canonical"
            raise ArbitrationLogError(msg)
        prior_digest = file_sha256(path)
        records.append(record)
    return tuple(records)


def append_arbitration_record(root: Path, record: ControllerArbitrationRecordV1) -> str:
    records = validate_arbitration_log(root)
    expected_sequence = len(records) + 1
    prior = (
        file_sha256(arbitration_path(root, expected_sequence - 1)) if records else None
    )
    if record.sequence != expected_sequence or record.prior_record_file_sha256 != prior:
        msg = "arbitration append lost the compare-and-swap race"
        raise ArbitrationLogError(msg)
    try:
        return publish_control_file(arbitration_path(root, expected_sequence), record)
    except FileExistsError:
        msg = "arbitration append lost the compare-and-swap race"
        raise ArbitrationLogError(msg) from None


def parse_provider_output(stdout: bytes, exit_code: int) -> ProviderResultV1:
    if len(stdout) > MAX_STDOUT_BYTES:
        msg = "output_too_large"
        raise ValueError(msg)
    if not stdout.endswith(b"\n") or len(stdout.splitlines()) != 1:
        msg = "invalid_provider_output"
        raise ValueError(msg)
    try:
        raw = cast("object", json.loads(stdout, object_pairs_hook=_unique_object))
        result = ProviderResultV1.model_validate(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        msg = "invalid_provider_output"
        raise ValueError(msg) from exc
    if encode_control_json(result) + b"\n" != stdout:
        msg = "invalid_provider_output"
        raise ValueError(msg)
    if result.status == "error" and result.error_code is None:
        msg = "exit_result_mismatch"
        raise ValueError(msg)
    expected_exit = (
        0
        if result.status == "ok"
        else {
            "invalid_request": 10,
            "forbidden_environment": 10,
            "network_attempt": 10,
            "missing_artifact": 20,
            "artifact_digest_mismatch": 21,
            "loader_identity_mismatch": 21,
            "loaded_class_mismatch": 21,
            "revision_mismatch": 21,
            "device_mismatch": 21,
            "load_failure": 20,
            "cleanup_failure": 22,
            "timeout": 124,
            "signal_death": exit_code,
            "invalid_provider_output": exit_code,
            "output_too_large": exit_code,
            "exit_result_mismatch": exit_code,
            "not_attempted_after_cleanup_failure": 22,
        }[cast("ProviderErrorCodeV1", result.error_code)]
    )
    if exit_code != expected_exit:
        msg = "exit_result_mismatch"
        raise ValueError(msg)
    return result


def provider_environment(
    source: Mapping[str, str], cuda_uuid: str, local_rank: int
) -> dict[str, str]:
    allowed = (
        "PATH",
        "HOME",
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
    )
    projected = {key: source[key] for key in allowed if key in source}
    projected.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_uuid,
            "LOCAL_RANK": str(local_rank),
            "PYTHONNOUSERSITE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WORLDMM_MODEL_LOAD_PROTOCOL": PROTOCOL,
        }
    )
    return projected


def invoke_provider(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout_seconds: int = PROVIDER_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    if timeout_seconds != PROVIDER_TIMEOUT_SECONDS:
        msg = "provider timeout must be exactly 900 seconds"
        raise ValueError(msg)
    process = subprocess.Popen(  # noqa: S603 - argv is the approved provider ABI
        list(argv),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(
            argv, 124, stdout[: MAX_STDOUT_BYTES + 1], stderr[:MAX_STDERR_BYTES]
        )
    return subprocess.CompletedProcess(
        argv,
        process.returncode,
        stdout[: MAX_STDOUT_BYTES + 1],
        stderr[:MAX_STDERR_BYTES],
    )


_STAGE_CONTROL_ADAPTER: TypeAdapter[StageControlReceiptV1] = TypeAdapter(
    StageControlReceiptV1
)


def validate_stage_control_receipt(value: object) -> StageControlReceiptV1:
    return _STAGE_CONTROL_ADAPTER.validate_python(value)
