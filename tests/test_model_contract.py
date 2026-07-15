# pyright: reportAny=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnusedCallResult=false
from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldmm_smvqa.model_contract import (
    DIGEST_DOMAIN_BY_SCHEMA_ID,
    EXPECTED_BOUNDARIES,
    FORBIDDEN_EDGES,
    ModelContractError,
    encode_control_json,
    load_model_boundary_contract,
    load_student_architecture,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/spatial/model_boundary_contract_v1.json"
ARCHITECTURE = ROOT / "configs/spatial/student_architecture_v1.json"
PINNED_CONTRACT_SHA256 = (
    "d0f46c28549e36ebcc9f5b9a93c9b90cf974d5ebed9e260842cacffcbfe53247"
)


def test_canonical_contract_bytes_topology_and_domains_are_frozen() -> None:
    contract, digest = load_model_boundary_contract(CONTRACT)

    assert digest == PINNED_CONTRACT_SHA256
    assert tuple(contract) == (
        "schema_version",
        "contract_id",
        "topology",
        "symbols",
        "boundaries",
        "transaction",
        "negative_cases",
    )
    assert (
        tuple(item["schema_id"] for item in contract["boundaries"])
        == EXPECTED_BOUNDARIES
    )
    assert tuple(contract["topology"]["forbidden_edges"]) == FORBIDDEN_EDGES
    assert all(
        item["digest_domain"] == DIGEST_DOMAIN_BY_SCHEMA_ID[item["schema_id"]]
        for item in contract["boundaries"]
    )


def test_student_architecture_binds_exact_contract_digest() -> None:
    architecture, digest = load_student_architecture(
        ARCHITECTURE,
        expected_model_contract_sha256=PINNED_CONTRACT_SHA256,
    )

    assert architecture.input_dim == 3
    assert architecture.record_types[-1] == "no_write"
    assert len(digest) == 64
    with pytest.raises(ModelContractError, match="contract digest mismatch"):
        _ = load_student_architecture(
            ARCHITECTURE,
            expected_model_contract_sha256="0" * 64,
        )


def test_declared_contract_rejects_duplicate_keys_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(ModelContractError, match="duplicate key"):
        _ = load_model_boundary_contract(duplicate)

    pretty = tmp_path / "pretty.json"
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    pretty.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ModelContractError, match="not canonical"):
        _ = load_model_boundary_contract(pretty)


def test_control_encoder_sorts_keys_and_rejects_floats() -> None:
    assert encode_control_json({"z": 1, "a": [2, True]}) == b'{"a":[2,true],"z":1}'
    with pytest.raises(ModelContractError, match="numeric value"):
        _ = encode_control_json({"float": 1.0})


def test_boundary_mutation_fails_before_consumer(tmp_path: Path) -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    value["transaction"]["observations"][1]["timestamp"] = 0.25
    malformed = tmp_path / "malformed.json"
    malformed.write_text(
        json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    called = False

    def consume() -> None:
        nonlocal called
        _ = load_model_boundary_contract(malformed)
        called = True

    with pytest.raises(ModelContractError, match="strictly increasing"):
        consume()
    assert not called
