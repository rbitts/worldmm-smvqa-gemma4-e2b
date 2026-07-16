from __future__ import annotations

# ruff: noqa: PLR0912, PLR2004
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnnecessaryComparison=false
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, override

from worldmm_smvqa.memory_contract_v2 import (
    CONTRACT_ID,
    MemoryContractV2Error,
    load_model_boundary_contract_v2,
)

_SCHEMA: Final = "memory-alignment-config-v1"
_MODEL_PATH_LITERAL: Final = "${WORLDMM_MEMORY_MODEL_PATH}"
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_ROOT_KEYS: Final = frozenset({"schema_version", "runtime", "memory_alignment"})
_MEMORY_KEYS: Final = frozenset(
    {
        "backend",
        "artifact_role",
        "model_family",
        "model_variant",
        "model_path",
        "contract_id",
        "contract_path",
        "contract_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class MemoryAlignmentConfigError(Exception):
    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"MemoryAlignmentConfigError: {self.path}: {self.detail}"


@dataclass(frozen=True, slots=True)
class MemoryAlignmentConfig:
    path: Path
    repository_root: Path
    contract_path: Path
    contract_relative_path: str
    contract_sha256: str
    model_env_name: str = "WORLDMM_MEMORY_MODEL_PATH"
    schema_version: str = _SCHEMA
    backend: str = "memory"
    artifact_role: str = "memory_builder"
    model_family: str = "gemma"
    model_variant: str = "Gemma-4-E2B-IT"
    contract_id: str = CONTRACT_ID


def load_memory_alignment_config(
    path: Path, repository_root: Path
) -> MemoryAlignmentConfig:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise MemoryAlignmentConfigError(path, f"cannot read config: {exc}") from exc
    parsed = _parse_reviewed_yaml(raw, path)
    if set(parsed) != _ROOT_KEYS:
        raise MemoryAlignmentConfigError(path, "root fields mismatch")
    if parsed["schema_version"] != _SCHEMA:
        raise MemoryAlignmentConfigError(path, "unsupported schema_version")
    runtime = parsed["runtime"]
    memory = parsed["memory_alignment"]
    if not isinstance(runtime, dict) or set(runtime) != {"location"}:
        raise MemoryAlignmentConfigError(path, "runtime fields mismatch")
    if runtime["location"] != "remote":
        raise MemoryAlignmentConfigError(path, "runtime.location must be remote")
    if not isinstance(memory, dict) or set(memory) != _MEMORY_KEYS:
        raise MemoryAlignmentConfigError(path, "memory_alignment fields mismatch")
    expected = {
        "backend": "memory",
        "artifact_role": "memory_builder",
        "model_family": "gemma",
        "model_variant": "Gemma-4-E2B-IT",
        "model_path": _MODEL_PATH_LITERAL,
        "contract_id": CONTRACT_ID,
    }
    for key, value in expected.items():
        if memory[key] != value:
            raise MemoryAlignmentConfigError(path, f"memory_alignment.{key} mismatch")
    digest = memory["contract_sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise MemoryAlignmentConfigError(
            path, "contract_sha256 must be lowercase SHA-256"
        )
    if len(set(digest)) == 1:
        raise MemoryAlignmentConfigError(path, "contract_sha256 must be reviewed")
    relative = memory["contract_path"]
    if not isinstance(relative, str):
        raise MemoryAlignmentConfigError(path, "contract_path must be a string")
    contract_path = _safe_repository_file(repository_root, relative, path)
    try:
        _, actual_digest = load_model_boundary_contract_v2(contract_path)
    except MemoryContractV2Error as exc:
        raise MemoryAlignmentConfigError(path, str(exc)) from exc
    if actual_digest != digest:
        raise MemoryAlignmentConfigError(
            path, "contract_sha256 does not match contract bytes"
        )
    return MemoryAlignmentConfig(
        path=path,
        repository_root=repository_root,
        contract_path=contract_path,
        contract_relative_path=relative,
        contract_sha256=digest,
    )


def _parse_reviewed_yaml(raw: str, path: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    section: dict[str, str] | None = None
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if "\t" in raw_line:
            raise MemoryAlignmentConfigError(
                path, f"line {line_number}: tabs forbidden"
            )
        indent = len(line) - len(line.lstrip(" "))
        if indent not in {0, 2}:
            raise MemoryAlignmentConfigError(
                path, f"line {line_number}: invalid indentation"
            )
        key, separator, value = line.strip().partition(":")
        if not separator or not key:
            raise MemoryAlignmentConfigError(
                path, f"line {line_number}: expected key: value"
            )
        if indent == 0:
            if key in result:
                raise MemoryAlignmentConfigError(
                    path, f"line {line_number}: duplicate key"
                )
            if value.strip():
                result[key] = _scalar(value.strip(), path, line_number)
                section = None
            else:
                section = {}
                result[key] = section
            continue
        if section is None or not value.strip() or key in section:
            raise MemoryAlignmentConfigError(
                path, f"line {line_number}: invalid section field"
            )
        section[key] = _scalar(value.strip(), path, line_number)
    return result


def _scalar(value: str, path: Path, line_number: int) -> str:
    if value.startswith(("'", '"')) or value.endswith(("'", '"')):
        if len(value) < 2 or value[0] != value[-1]:
            raise MemoryAlignmentConfigError(
                path, f"line {line_number}: invalid quoting"
            )
        value = value[1:-1]
    if not value:
        raise MemoryAlignmentConfigError(path, f"line {line_number}: empty value")
    return value


def _safe_repository_file(root: Path, relative: str, config_path: Path) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or "\\" in relative
        or "\x00" in relative
        or any(part in {"", ".", ".."} for part in pure.parts)
        or str(pure) != relative
    ):
        raise MemoryAlignmentConfigError(config_path, "contract_path is unsafe")
    root = root.absolute()
    candidate = root.joinpath(*pure.parts)
    current = root
    try:
        if not root.is_dir() or root.is_symlink():
            raise MemoryAlignmentConfigError(config_path, "repository_root is unsafe")
        for part in pure.parts:
            current = current / part
            if current.is_symlink():
                raise MemoryAlignmentConfigError(
                    config_path, "contract_path contains a link"
                )
        if not candidate.is_file():
            raise MemoryAlignmentConfigError(
                config_path, "contract_path is not a regular file"
            )
    except OSError as exc:
        raise MemoryAlignmentConfigError(
            config_path, f"cannot inspect contract_path: {exc}"
        ) from exc
    return candidate


def contract_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
