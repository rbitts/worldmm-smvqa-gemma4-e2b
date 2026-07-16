from __future__ import annotations

# ruff: noqa: EM101, EM102, TRY003
# pyright: reportAny=false
# pyright: reportExplicitAny=false
# pyright: reportImplicitOverride=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import ValidationError

from worldmm_smvqa.worldmm.episodic_types import (
    EpisodicEdgeRecord,
    EpisodicNodeRecord,
)
from worldmm_smvqa.worldmm.semantic import SemanticTripleRecord
from worldmm_smvqa.worldmm.visual import VisualMemoryRecord

CONTRACT_SCHEMA_VERSION: Final = "model-boundary-contract-v2"
CONTRACT_ID: Final = "worldmm-smvqa-memory-v2"
REQUEST_SCHEMA_ID: Final = "memory-request-v2"
RESPONSE_SCHEMA_ID: Final = "memory-response-v2"
ENVELOPE_SCHEMA_ID: Final = "memory-store-envelope-v2"
VISUAL_RECORD_CONTRACT_ID: Final = "memory-visual-record-contract-v2"
EPISODIC_RECORD_CONTRACT_ID: Final = "memory-episodic-record-contract-v2"
SEMANTIC_RECORD_CONTRACT_ID: Final = "memory-semantic-record-contract-v2"
StoreKind = Literal["visual", "episodic", "semantic", "semantic_rebuild"]

_RECORD_CONTRACT_BY_STORE: Final[dict[str, str]] = {
    "visual": VISUAL_RECORD_CONTRACT_ID,
    "episodic": EPISODIC_RECORD_CONTRACT_ID,
    "semantic": SEMANTIC_RECORD_CONTRACT_ID,
    "semantic_rebuild": SEMANTIC_RECORD_CONTRACT_ID,
}
_EXPECTED_CONTRACTS: Final = (
    {
        "store_kind": "visual",
        "record_contract_id": VISUAL_RECORD_CONTRACT_ID,
        "record_types": ["visual"],
        "row_store": "visual",
    },
    {
        "store_kind": "episodic",
        "record_contract_id": EPISODIC_RECORD_CONTRACT_ID,
        "record_types": ["node", "edge"],
        "row_store": None,
    },
    {
        "store_kind": "semantic",
        "record_contract_id": SEMANTIC_RECORD_CONTRACT_ID,
        "record_types": ["semantic_triple"],
        "row_store": "semantic",
    },
    {
        "store_kind": "semantic_rebuild",
        "record_contract_id": SEMANTIC_RECORD_CONTRACT_ID,
        "record_types": ["semantic_triple"],
        "row_store": "semantic",
    },
)


@dataclass(frozen=True, slots=True)
class MemoryContractV2Error(Exception):
    detail: str

    def __str__(self) -> str:
        return f"MemoryContractV2Error: {self.detail}"


def load_model_boundary_contract_v2(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MemoryContractV2Error(f"cannot load contract: {exc}") from exc
    validate_model_boundary_contract_v2(value)
    return cast("dict[str, Any]", value), hashlib.sha256(raw).hexdigest()


def validate_model_boundary_contract_v2(value: object) -> None:
    root = _mapping(value, "contract root")
    _keys(
        root,
        (
            "schema_version",
            "contract_id",
            "request_schema_id",
            "response_schema_id",
            "envelope_schema_id",
            "record_contracts",
            "dataflow",
        ),
        "contract root",
    )
    expected_scalars = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "request_schema_id": REQUEST_SCHEMA_ID,
        "response_schema_id": RESPONSE_SCHEMA_ID,
        "envelope_schema_id": ENVELOPE_SCHEMA_ID,
    }
    for field, expected in expected_scalars.items():
        if root[field] != expected:
            raise MemoryContractV2Error(f"{field} must equal {expected}")
    contracts = _sequence(root["record_contracts"], "record_contracts")
    if contracts != list(_EXPECTED_CONTRACTS):
        raise MemoryContractV2Error("record contract order or mapping mismatch")
    if root["dataflow"] != ["visual", "episodic", "semantic", "semantic_rebuild"]:
        raise MemoryContractV2Error("dataflow order mismatch")


def validate_memory_store_envelope_v2(value: object) -> tuple[object, ...]:
    root = _mapping(value, "memory envelope")
    _keys(
        root,
        (
            "schema_version",
            "contract_id",
            "store_kind",
            "record_contract_id",
            "records",
        ),
        "memory envelope",
    )
    if root["schema_version"] != ENVELOPE_SCHEMA_ID:
        raise MemoryContractV2Error("unsupported envelope schema_version")
    if root["contract_id"] != CONTRACT_ID:
        raise MemoryContractV2Error("envelope contract_id mismatch")
    store = root["store_kind"]
    if not isinstance(store, str) or store not in _RECORD_CONTRACT_BY_STORE:
        raise MemoryContractV2Error("unsupported envelope store_kind")
    expected_contract = _RECORD_CONTRACT_BY_STORE[store]
    if root["record_contract_id"] != expected_contract:
        raise MemoryContractV2Error("envelope record_contract_id mismatch")
    records = _sequence(root["records"], "memory envelope records")
    return tuple(_validate_row(store, row, index) for index, row in enumerate(records))


def _validate_row(store: str, value: object, index: int) -> object:
    row = _mapping(value, f"records[{index}]")
    try:
        if store == "visual":
            return VisualMemoryRecord.model_validate(row)
        if store in {"semantic", "semantic_rebuild"}:
            return SemanticTripleRecord.model_validate(row)
        record_type = row.get("record_type")
        if record_type == "node":
            return EpisodicNodeRecord.model_validate(row)
        if record_type == "edge":
            return EpisodicEdgeRecord.model_validate(row)
        raise MemoryContractV2Error(f"records[{index}]: invalid episodic record_type")
    except ValidationError as exc:
        raise MemoryContractV2Error(f"records[{index}]: invalid record: {exc}") from exc


def _mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MemoryContractV2Error(f"{where} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise MemoryContractV2Error(f"{where} keys must be strings")
    return cast("dict[str, Any]", dict(value))


def _sequence(value: object, where: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise MemoryContractV2Error(f"{where} must be an array")
    return list(value)


def _keys(value: Mapping[str, object], expected: tuple[str, ...], where: str) -> None:
    if set(value) != set(expected):
        raise MemoryContractV2Error(f"{where} fields mismatch")
