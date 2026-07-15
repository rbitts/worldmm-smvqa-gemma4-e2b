from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from worldmm_smvqa.model_load import (
    ArbitrationLogError,
    ControllerArbitrationRecordV1,
    ExpectedDigestsV1,
    LeasePayloadV1,
    ModelLoadRequestV1,
    PostApprovalActionIntentPayloadV1,
    ProviderLockV1,
    append_arbitration_record,
    encode_control_json,
    file_sha256,
    parse_provider_output,
    provider_environment,
    validate_arbitration_log,
)

SHA = "0" * 64


def _request(**changes: object) -> dict[str, object]:
    request: dict[str, object] = {
        "schema_version": "model_load_request_v1",
        "protocol": "worldmm-model-load-v1",
        "run_identity": {},
        "model_role": "qwen_memory",
        "load_mode": "model_tree",
        "model_path": "/approved/qwen",
        "checkpoint_path": None,
        "config_path": None,
        "fixture_sha256": SHA,
        "expected_loader_id": "loader",
        "expected_loaded_class": "package.Model",
        "expected_revision": f"tree-sha256:{SHA}",
        "expected_digests": {
            "architecture_sha256": None,
            "config_sha256": SHA,
            "processor_sha256": SHA,
            "checkpoint_sha256": None,
            "checkpoint_tree_sha256": None,
            "model_tree_sha256": SHA,
            "executable_sha256": None,
            "origin_consensus_payload_sha256": None,
            "origin_consensus_file_sha256": None,
        },
        "logical_device": "cuda:0",
        "physical_device": {
            "logical_device": "cuda:0",
            "cuda_visible_device": "GPU-test",
            "cuda_uuid": "GPU-test",
            "pci_bus_id": "0000:00:00.0",
            "device_name": "NVIDIA H100 80GB HBM3",
            "total_memory_bytes": 80_000_000_000,
        },
        "timeout_seconds": 900,
    }
    request.update(changes)
    return request


def test_role_mode_matrix_rejects_forbidden_path_and_digest_combinations() -> None:
    assert ModelLoadRequestV1.model_validate(_request()).model_role == "qwen_memory"
    with pytest.raises(ValidationError, match="requires only model_path"):
        _ = ModelLoadRequestV1.model_validate(_request(checkpoint_path="/forbidden"))
    with pytest.raises(ValidationError, match="required expected digests absent"):
        _ = ModelLoadRequestV1.model_validate(
            _request(expected_digests=ExpectedDigestsV1())
        )


def test_pending_and_fake_accepted_provider_locks_are_disjoint() -> None:
    root = Path(__file__).resolve().parents[1]
    pending = ProviderLockV1.model_validate_json(
        (
            root / "configs/spatial/model_load_provider_lock_v1.template.json"
        ).read_bytes()
    )
    accepted = ProviderLockV1.model_validate_json(
        (
            root / "tests/fixtures/model_load_v1/provider_lock.accepted.fake.json"
        ).read_bytes()
    )
    assert pending.status == "pending"
    assert pending.roles == ()
    assert accepted.status == "accepted"
    assert tuple(role.model_role for role in accepted.roles) == (
        "gcut3r_teacher",
        "qwen_memory",
        "spatial_trainable",
        "gemma_qa",
    )


def test_provider_environment_drops_network_credentials_and_unrelated_worldmm() -> None:
    result = provider_environment(
        {
            "PATH": "/bin",
            "HOME": "/home/test",
            "HTTPS_PROXY": "forbidden",
            "HF_TOKEN": "forbidden",
            "WORLDMM_OUTPUT_ROOT": "forbidden",
        },
        "GPU-test",
        7,
    )
    assert result["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert result["LOCAL_RANK"] == "7"
    assert result["TRANSFORMERS_OFFLINE"] == "1"
    assert "HTTPS_PROXY" not in result
    assert "HF_TOKEN" not in result
    assert "WORLDMM_OUTPUT_ROOT" not in result


def test_provider_parser_rejects_second_line_and_exit_mismatch() -> None:
    with pytest.raises(ValueError, match="invalid_provider_output"):
        _ = parse_provider_output(b"{}\n{}\n", 0)
    payload = {
        "actual_device": None,
        "actual_loaded_class": None,
        "checkpoint_sha256": None,
        "checkpoint_tree_sha256": None,
        "cleanup": {
            "cuda_cache_cleared": False,
            "model_released": False,
            "processor_released": False,
        },
        "config_sha256": None,
        "diagnostic": "load failed",
        "error_code": "load_failure",
        "executable_sha256": None,
        "expected_loaded_class": "package.Model",
        "load_ok": False,
        "loader_id": "loader",
        "model_role": "qwen_memory",
        "model_tree_sha256": None,
        "processor_sha256": None,
        "protocol": "worldmm-model-load-v1",
        "request_sha256": SHA,
        "revision": None,
        "schema_version": "model_load_provider_result_v1",
        "status": "error",
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    with pytest.raises(ValueError, match="exit_result_mismatch"):
        _ = parse_provider_output(encoded, 0)


def _genesis() -> ControllerArbitrationRecordV1:
    return ControllerArbitrationRecordV1(
        sequence=1,
        prior_record_file_sha256=None,
        record_kind="acquire",
        controller_id="controller-a",
        controller_epoch=1,
        controller_job_id="job-a",
        created_at_ms=1000,
        expires_at_ms=46_000,
        payload=LeasePayloadV1(
            prior_controller_id=None,
            prior_controller_epoch=None,
            prior_accounting_sha256=None,
        ),
    )


def test_arbitration_genesis_is_unique_and_chain_is_contiguous(tmp_path: Path) -> None:
    root = tmp_path / "arbitration"
    _ = append_arbitration_record(root, _genesis())
    prior = file_sha256(root / "00000000000000000001.json")
    heartbeat = ControllerArbitrationRecordV1(
        sequence=2,
        prior_record_file_sha256=prior,
        record_kind="heartbeat",
        controller_id="controller-a",
        controller_epoch=1,
        controller_job_id="job-a",
        created_at_ms=2000,
        expires_at_ms=47_000,
        payload=LeasePayloadV1(
            prior_controller_id="controller-a",
            prior_controller_epoch=1,
            prior_accounting_sha256=SHA,
        ),
    )
    _ = append_arbitration_record(root, heartbeat)
    assert [row.sequence for row in validate_arbitration_log(root)] == [1, 2]
    with pytest.raises(ArbitrationLogError, match="compare-and-swap"):
        _ = append_arbitration_record(root, heartbeat)


def test_action_intent_is_non_expiring_and_transition_checked() -> None:
    payload = PostApprovalActionIntentPayloadV1(
        run_id="run",
        run_nonce=SHA,
        stage_id="train",
        job_id="job",
        stage_sequence=1,
        prior_executed_receipt_sha256=None,
        from_state="held",
        action="release",
        to_state="released",
        reason="approved_go",
        proposal_file_sha256=SHA,
        submission_manifest_file_sha256=SHA,
        operator_approval_file_sha256=SHA,
    )
    record = ControllerArbitrationRecordV1(
        sequence=2,
        prior_record_file_sha256=SHA,
        record_kind="action_intent",
        controller_id="controller-a",
        controller_epoch=1,
        controller_job_id="job-a",
        created_at_ms=2000,
        expires_at_ms=None,
        payload=payload,
    )
    assert b'"expires_at_ms":null' in encode_control_json(record)
    with pytest.raises(ValidationError, match="illegal stage action transition"):
        _ = PostApprovalActionIntentPayloadV1.model_validate(
            {**payload.model_dump(), "from_state": "cancelled"}
        )
