from __future__ import annotations

import copy
from pathlib import Path

import pytest

from worldmm_smvqa.memory_contract_v2 import (
    CONTRACT_ID,
    ENVELOPE_SCHEMA_ID,
    MemoryContractV2Error,
    load_model_boundary_contract_v2,
    validate_memory_store_envelope_v2,
    validate_model_boundary_contract_v2,
)


def test_reviewed_v2_contract_loads_with_stable_digest() -> None:
    contract, digest = load_model_boundary_contract_v2(
        Path("configs/spatial/model_boundary_contract_v2.json")
    )

    assert contract["contract_id"] == CONTRACT_ID
    assert digest == "6d7ad8546e63bd5d25260c8a6b5fe04e9bb23a424e726a09a44946435016fe77"


def test_v2_contract_rejects_order_mapping_and_unknown_fields() -> None:
    contract, _ = load_model_boundary_contract_v2(
        Path("configs/spatial/model_boundary_contract_v2.json")
    )
    reordered = copy.deepcopy(contract)
    reordered["record_contracts"][0], reordered["record_contracts"][1] = (
        reordered["record_contracts"][1],
        reordered["record_contracts"][0],
    )
    with pytest.raises(MemoryContractV2Error, match="order or mapping"):
        validate_model_boundary_contract_v2(reordered)

    extra = copy.deepcopy(contract)
    extra["unexpected"] = True
    with pytest.raises(MemoryContractV2Error, match="fields mismatch"):
        validate_model_boundary_contract_v2(extra)


def test_semantic_rebuild_envelope_keeps_semantic_row_contract() -> None:
    envelope = {
        "schema_version": ENVELOPE_SCHEMA_ID,
        "contract_id": CONTRACT_ID,
        "store_kind": "semantic_rebuild",
        "record_contract_id": "memory-semantic-record-contract-v2",
        "records": [
            {
                "record_type": "semantic_triple",
                "memory_id": "semantic-1",
                "store": "semantic",
                "video_id": "video-1",
                "subject": "person",
                "predicate": "holds",
                "object": "cup",
                "text": "person holds cup",
                "support_memory_ids": ["episodic-1", "episodic-2"],
                "support_event_count": 2,
                "start_time": 0.0,
                "end_time": 1.0,
                "confidence": 1.0,
                "text_embedding_id": "embedding-1",
            }
        ],
    }

    records = validate_memory_store_envelope_v2(envelope)
    assert len(records) == 1

    invalid = copy.deepcopy(envelope)
    invalid_records = invalid["records"]
    assert isinstance(invalid_records, list)
    invalid_record = invalid_records[0]
    assert isinstance(invalid_record, dict)
    invalid_record["store"] = "semantic_rebuild"
    with pytest.raises(MemoryContractV2Error, match="invalid record"):
        _ = validate_memory_store_envelope_v2(invalid)
