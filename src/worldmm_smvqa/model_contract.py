# ruff: noqa: EM101, EM102, PLR2004, TRY003
# pyright: reportAny=false
# pyright: reportExplicitAny=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportPrivateUsage=false
# pyright: reportImplicitOverride=false
# pyright: reportUnusedCallResult=false
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, Literal, cast

from worldmm_smvqa.spatial_train import RECORD_TYPES
from worldmm_smvqa.worldmm.gcut3r_teacher import _digest

SHA256_HEX_LENGTH: Final = 64
CONTRACT_SCHEMA_IDS: Final = (
    "model-boundary-contract-v1",
    "gcut3r-request-v1",
    "gcut3r-response-v1",
    "qwen-memory-request-v1",
    "qwen-memory-response-v1",
    "qwen-store-envelope-v1",
    "spatial-train-batch-v1",
    "spatial-train-output-v1",
    "spatial-checkpoint-v2",
    "spatial-typed-memory-v1",
    "retrieval-fan-in-v1",
    "gemma-prompt-v1",
    "gemma-generated-output-v1",
    "normalized-prediction-v1",
    "student-architecture-config-v1",
    "local-mock-authorization-v1",
    "teacher-cache-contract-v1",
)
DIGEST_DOMAIN_BY_SCHEMA_ID: Final = {
    "model-boundary-contract-v1": "declared_json_v1",
    "gcut3r-request-v1": "existing_teacher_jcs_v1",
    "gcut3r-response-v1": "existing_teacher_jcs_v1",
    "qwen-memory-request-v1": "declared_json_v1",
    "qwen-memory-response-v1": "declared_json_v1",
    "qwen-store-envelope-v1": "declared_json_v1",
    "spatial-train-batch-v1": "declared_json_v1",
    "spatial-train-output-v1": "declared_json_v1",
    "spatial-checkpoint-v2": "raw_file_v1",
    "spatial-typed-memory-v1": "existing_jsonl_v1",
    "retrieval-fan-in-v1": "declared_json_v1",
    "gemma-prompt-v1": "declared_json_v1",
    "gemma-generated-output-v1": "declared_json_v1",
    "normalized-prediction-v1": "declared_json_v1",
    "student-architecture-config-v1": "declared_json_v1",
    "local-mock-authorization-v1": "declared_json_v1",
    "teacher-cache-contract-v1": "declared_json_v1",
}


@dataclass(frozen=True, slots=True)
class ModelContractError(Exception):
    detail: str

    def __str__(self) -> str:
        return f"ModelContractError: {self.detail}"


@dataclass(frozen=True, slots=True)
class StudentArchitectureConfigV1:
    schema_version: Literal["student-architecture-config-v1"]
    architecture_id: str
    input_dim: int
    teacher_dim: int
    geometry_dim: int
    association_classes: int
    hidden_dim: int
    learning_rate: float
    rate_normalizer_bytes: float
    record_types: tuple[str, ...]
    model_contract_sha256: str

    def __post_init__(self) -> None:
        if not self.architecture_id:
            raise ModelContractError("architecture_id must be non-empty")
        dimensions = (
            self.input_dim,
            self.teacher_dim,
            self.geometry_dim,
            self.association_classes,
            self.hidden_dim,
        )
        if any(
            isinstance(value, bool) or not 1 <= value <= 65536 for value in dimensions
        ):
            raise ModelContractError(
                "architecture dimensions must be integers in [1,65536]"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ModelContractError("learning_rate must be finite and positive")
        if (
            not math.isfinite(self.rate_normalizer_bytes)
            or self.rate_normalizer_bytes <= 0
        ):
            raise ModelContractError(
                "rate_normalizer_bytes must be finite and positive"
            )
        if self.record_types != RECORD_TYPES:
            raise ModelContractError("record_types must equal the repository order")
        _require_sha256(self.model_contract_sha256, "model_contract_sha256")


ROOT_KEYS: Final = (
    "schema_version",
    "contract_id",
    "topology",
    "symbols",
    "boundaries",
    "transaction",
    "negative_cases",
)
TOPOLOGY_KEYS: Final = ("components", "edges", "forbidden_edges")
SYMBOL_KEYS: Final = ("O", "B", "I", "T", "G", "H", "A", "R", "C", "E")
BOUNDARY_KEYS: Final = (
    "schema_id",
    "version",
    "owner",
    "producer",
    "consumer",
    "request",
    "response",
    "tensors",
    "provenance",
    "digest_domain",
    "loader",
)
EXPECTED_BOUNDARIES: Final = CONTRACT_SCHEMA_IDS[1:14]
FORBIDDEN_EDGES: Final = (
    "gcut3r_teacher->qwen_memory",
    "qwen_memory->spatial_student",
    "spatial_student->gemma_qa",
    "gcut3r_teacher->qwen_memory->spatial_student->gemma_qa",
)


def raw_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encode_declared_json(model: object, schema_id: str) -> bytes:
    domain = DIGEST_DOMAIN_BY_SCHEMA_ID.get(schema_id)
    if domain != "declared_json_v1":
        raise ModelContractError(f"{schema_id}: not a declared-json schema")
    value = _plain_value(model)
    _validate_json_value(value, floats=True)
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode()


def encode_control_json(model: object) -> bytes:
    value = _plain_value(model)
    _validate_json_value(value, floats=False)
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def existing_teacher_digest(value: object) -> str:
    return _digest(_plain_value(value))


def load_model_boundary_contract(path: Path) -> tuple[dict[str, Any], str]:
    raw, value = _load_declared_file(path)
    validate_model_boundary_contract(value)
    return value, raw_sha256(raw)


def validate_model_boundary_contract(value: object) -> None:  # noqa: PLR0912, PLR0915
    root = _mapping(value, "contract root")
    _require_keys(root, ROOT_KEYS, "contract root")
    if root["schema_version"] != "model-boundary-contract-v1":
        raise ModelContractError("unsupported contract schema_version")
    if not isinstance(root["contract_id"], str) or not root["contract_id"]:
        raise ModelContractError("contract_id must be non-empty")
    topology = _mapping(root["topology"], "topology")
    _require_keys(topology, TOPOLOGY_KEYS, "topology")
    symbols = _mapping(root["symbols"], "symbols")
    _require_keys(symbols, SYMBOL_KEYS, "symbols")
    expected_symbols = {
        "O": 2,
        "B": 2,
        "I": 3,
        "T": 2,
        "G": 4,
        "H": 8,
        "A": 2,
        "R": 7,
        "C": 4,
        "E": 4,
    }
    if symbols != expected_symbols:
        raise ModelContractError(
            "fixture symbols do not match the frozen tiny transaction"
        )
    boundaries = _sequence(root["boundaries"], "boundaries")
    ids: list[str] = []
    for index, item in enumerate(boundaries):
        boundary = _mapping(item, f"boundaries[{index}]")
        _require_keys(boundary, BOUNDARY_KEYS, f"boundaries[{index}]")
        schema_id = boundary["schema_id"]
        if not isinstance(schema_id, str):
            raise ModelContractError(f"boundaries[{index}].schema_id must be a string")
        ids.append(schema_id)
        if boundary["digest_domain"] != DIGEST_DOMAIN_BY_SCHEMA_ID.get(schema_id):
            raise ModelContractError(f"{schema_id}: incorrect digest domain")
    if tuple(ids) != EXPECTED_BOUNDARIES:
        raise ModelContractError("boundary order or coverage mismatch")
    forbidden = _sequence(topology["forbidden_edges"], "topology.forbidden_edges")
    if tuple(forbidden) != FORBIDDEN_EDGES:
        raise ModelContractError("forbidden topology edges mismatch")
    transaction = _mapping(root["transaction"], "transaction")
    _require_keys(
        transaction, ("video", "observations", "question", "label"), "transaction"
    )
    observations = _sequence(transaction["observations"], "transaction.observations")
    if not 2 <= len(observations) <= 16:
        raise ModelContractError("transaction must contain 2..16 observations")
    times: list[float] = []
    video_id = _mapping(transaction["video"], "transaction.video").get("video_id")
    for index, item in enumerate(observations):
        observation = _mapping(item, f"observation[{index}]")
        if (
            observation.get("sequence_index") != index
            or observation.get("video_id") != video_id
        ):
            raise ModelContractError("observation sequence/video mismatch")
        timestamp = observation.get("timestamp")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int | float)
            or not math.isfinite(timestamp)
        ):
            raise ModelContractError("observation timestamp must be finite")
        times.append(float(timestamp))
    if any(right <= left for left, right in pairwise(times)):
        raise ModelContractError("observation timestamps must be strictly increasing")
    question = _mapping(transaction["question"], "transaction.question")
    choices = _sequence(question.get("answer_choices"), "question.answer_choices")
    choice_ids = tuple(
        _mapping(choice, "answer choice").get("choice_id") for choice in choices
    )
    if len(choice_ids) != 4 or len(set(choice_ids)) != 4:
        raise ModelContractError("question must contain four unique choices")
    label = _mapping(transaction["label"], "transaction.label")
    if label.get("answer") not in choice_ids:
        raise ModelContractError("label answer must name a choice")
    negative_cases = _sequence(root["negative_cases"], "negative_cases")
    covered = {
        case.get("boundary") for case in negative_cases if isinstance(case, dict)
    }
    if not set(EXPECTED_BOUNDARIES).issubset(covered):
        raise ModelContractError("negative cases must cover every boundary")


def load_student_architecture(
    path: Path,
    *,
    expected_model_contract_sha256: str | None = None,
) -> tuple[StudentArchitectureConfigV1, str]:
    raw, value = _load_declared_file(path)
    mapping = _mapping(value, "student architecture")
    keys = (
        "schema_version",
        "architecture_id",
        "input_dim",
        "teacher_dim",
        "geometry_dim",
        "association_classes",
        "hidden_dim",
        "learning_rate",
        "rate_normalizer_bytes",
        "record_types",
        "model_contract_sha256",
    )
    _require_keys(mapping, keys, "student architecture")
    try:
        architecture = StudentArchitectureConfigV1(
            schema_version=cast(
                "Literal['student-architecture-config-v1']", mapping["schema_version"]
            ),
            architecture_id=cast("str", mapping["architecture_id"]),
            input_dim=cast("int", mapping["input_dim"]),
            teacher_dim=cast("int", mapping["teacher_dim"]),
            geometry_dim=cast("int", mapping["geometry_dim"]),
            association_classes=cast("int", mapping["association_classes"]),
            hidden_dim=cast("int", mapping["hidden_dim"]),
            learning_rate=cast("float", mapping["learning_rate"]),
            rate_normalizer_bytes=cast("float", mapping["rate_normalizer_bytes"]),
            record_types=tuple(cast("Sequence[str]", mapping["record_types"])),
            model_contract_sha256=cast("str", mapping["model_contract_sha256"]),
        )
    except (TypeError, ValueError) as exc:
        raise ModelContractError(f"invalid student architecture: {exc}") from exc
    if architecture.schema_version != "student-architecture-config-v1":
        raise ModelContractError("unsupported student architecture schema")
    if (
        expected_model_contract_sha256 is not None
        and architecture.model_contract_sha256 != expected_model_contract_sha256
    ):
        raise ModelContractError("student architecture contract digest mismatch")
    return architecture, raw_sha256(raw)


def _load_declared_file(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ModelContractError(f"cannot read {path}: {exc}") from exc
    if (
        not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
        or raw.startswith(b"\xef\xbb\xbf")
    ):
        raise ModelContractError(
            f"{path}: declared JSON must have exactly one trailing LF and no BOM"
        )
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ModelContractError) as exc:
        raise ModelContractError(f"invalid declared JSON {path}: {exc}") from exc
    mapping = _mapping(value, str(path))
    encoded = (
        json.dumps(
            mapping, ensure_ascii=True, allow_nan=False, separators=(",", ":")
        ).encode()
        + b"\n"
    )
    if encoded != raw:
        raise ModelContractError(f"{path}: bytes are not canonical declared JSON")
    return raw, dict(mapping)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelContractError(f"duplicate key: {key}")
        result[key] = value
    return result


def _plain_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain_value(item) for key, item in value.items()}
    return value


def _validate_json_value(value: object, *, floats: bool) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not floats or not math.isfinite(value):
            raise ModelContractError("numeric value is not valid in this digest domain")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, floats=floats)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json_value(item, floats=floats)
        return
    raise ModelContractError(f"unsupported JSON value: {type(value).__name__}")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ModelContractError(f"{name} must be an object")
    return cast("dict[str, Any]", value)


def _sequence(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ModelContractError(f"{name} must be an array")
    return cast("list[Any]", value)


def _require_keys(value: Mapping[str, object], keys: Sequence[str], name: str) -> None:
    if tuple(value) != tuple(keys):
        raise ModelContractError(f"{name} field order/coverage mismatch")


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ModelContractError(f"{name} must be a lowercase SHA-256")
    return value


if set(DIGEST_DOMAIN_BY_SCHEMA_ID) != set(CONTRACT_SCHEMA_IDS):
    detail = "DIGEST_DOMAIN_BY_SCHEMA_ID must cover every contract schema"
    raise RuntimeError(detail)
