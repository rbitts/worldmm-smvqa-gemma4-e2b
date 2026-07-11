from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast, override

from pydantic import Field, FiniteFloat, ValidationError

from worldmm_smvqa.schema import FrozenModel
from worldmm_smvqa.spatial_train import RECORD_TYPES, TeacherCacheRow
from worldmm_smvqa.worldmm.gcut3r_teacher import (
    TeacherContractError,
    read_teacher_cache,
)
from worldmm_smvqa.worldmm.typed_memory import (
    TypedMemoryRecord,
    serialized_byte_cost,
)

type FiniteVector = Annotated[tuple[FiniteFloat, ...], Field(min_length=1)]
type NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


@dataclass(frozen=True, slots=True)
class TeacherMaterializationError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"TeacherMaterializationError: {self.detail}"


class TeacherSupervisionRow(FrozenModel):
    observation_id: str = Field(min_length=1)
    memory_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    split: Literal["train", "validation"]
    features: FiniteVector
    teacher_embedding: FiniteVector
    geometry_target: FiniteVector
    association_target: NonNegativeInt


def materialize_teacher_rows(
    teacher_cache: Path,
    supervision: Path,
) -> tuple[TeacherCacheRow, ...]:
    """Join supervision to a digest-validated causal teacher cache."""
    cache = read_teacher_cache(teacher_cache)
    labels = _read_supervision(supervision)

    cache_records: dict[tuple[str, str], TypedMemoryRecord] = {}
    ordered_keys: list[tuple[str, str]] = []
    for cache_row in cache:
        observation_id = cache_row.request.observation_id
        for record in cache_row.response.records:
            key = (observation_id, record.memory_id)
            if key in cache_records:
                raise TeacherMaterializationError(
                    detail=f"duplicate teacher record: {_format_key(key)}",
                )
            cache_records[key] = record
            ordered_keys.append(key)
    if not cache_records:
        raise TeacherMaterializationError(detail="teacher cache has no typed records")

    supervision_by_key = {(row.observation_id, row.memory_id): row for row in labels}
    unknown = supervision_by_key.keys() - cache_records.keys()
    if unknown:
        unknown_key = _format_key(min(unknown))
        raise TeacherMaterializationError(
            detail=f"supervision references missing teacher record: {unknown_key}",
        )
    missing = cache_records.keys() - supervision_by_key.keys()
    if missing:
        missing_key = _format_key(min(missing))
        raise TeacherMaterializationError(
            detail=f"missing supervision for teacher record: {missing_key}",
        )

    rows: list[TeacherCacheRow] = []
    sample_ids: set[str] = set()
    for key in ordered_keys:
        label = supervision_by_key[key]
        record = cache_records[key]
        record_type = record.record_type
        type_index = RECORD_TYPES.index(record_type)
        uncertainty = float(record.geometry_uncertainty.standard_deviation_m)
        if uncertainty <= 0.0:
            raise TeacherMaterializationError(
                detail=f"{_format_key(key)}: uncertainty_target must be positive",
            )
        byte_cost = serialized_byte_cost(record)
        if (record_type == "no_write") != (byte_cost == 0):
            raise TeacherMaterializationError(
                detail=f"{_format_key(key)}: only no_write may have byte_cost 0",
            )
        sample_id = f"{key[0]}:{key[1]}"
        if sample_id in sample_ids:
            raise TeacherMaterializationError(
                detail=f"sample_id collision: {sample_id}",
            )
        sample_ids.add(sample_id)
        rows.append(
            TeacherCacheRow(
                sample_id=sample_id,
                group_id=label.group_id,
                split=label.split,
                features=tuple(float(value) for value in label.features),
                teacher_embedding=tuple(
                    float(value) for value in label.teacher_embedding
                ),
                type_index=type_index,
                geometry_target=tuple(float(value) for value in label.geometry_target),
                association_target=label.association_target,
                uncertainty_target=uncertainty,
                byte_cost=float(byte_cost),
            ),
        )
    return tuple(rows)


def write_teacher_rows(path: Path, rows: Sequence[TeacherCacheRow]) -> None:
    if not rows:
        raise TeacherMaterializationError(detail="no materialized rows")
    payload = "".join(f"{_encode_row(row)}\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _ = temporary.write_text(payload, encoding="utf-8")
        _ = temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_teacher_cache(
    teacher_cache: Path,
    supervision: Path,
    out: Path,
) -> tuple[TeacherCacheRow, ...]:
    rows = materialize_teacher_rows(teacher_cache, supervision)
    write_teacher_rows(out, rows)
    return rows


def _read_supervision(path: Path) -> tuple[TeacherSupervisionRow, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TeacherMaterializationError(
            detail=f"cannot read supervision {path}: {exc}",
        ) from exc

    rows: list[TeacherSupervisionRow] = []
    keys: set[tuple[str, str]] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = TeacherSupervisionRow.model_validate_json(line)
        except ValidationError as exc:
            raise TeacherMaterializationError(
                detail=f"{path}: line {line_number}: {exc}",
            ) from exc
        key = (row.observation_id, row.memory_id)
        if key in keys:
            raise TeacherMaterializationError(
                detail=f"{path}: line {line_number}: duplicate supervision key",
            )
        keys.add(key)
        rows.append(row)
    if not rows:
        raise TeacherMaterializationError(detail=f"{path}: no supervision rows")

    dimensions = {
        (len(row.features), len(row.teacher_embedding), len(row.geometry_target))
        for row in rows
    }
    if len(dimensions) != 1:
        raise TeacherMaterializationError(
            detail=f"{path}: inconsistent vector dimensions",
        )
    splits = {row.split for row in rows}
    if splits != {"train", "validation"}:
        raise TeacherMaterializationError(
            detail=f"{path}: train and validation supervision are required",
        )
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(row.group_id, set()).add(row.split)
    crossing = tuple(
        group_id
        for group_id, group_split in group_splits.items()
        if len(group_split) > 1
    )
    if crossing:
        raise TeacherMaterializationError(
            detail=f"{path}: group_id crosses splits: {min(crossing)!r}",
        )
    training_targets = {row.association_target for row in rows if row.split == "train"}
    if training_targets != set(range(max(training_targets) + 1)):
        raise TeacherMaterializationError(
            detail=f"{path}: train association targets must be contiguous from zero",
        )
    unseen = {
        row.association_target
        for row in rows
        if row.split == "validation" and row.association_target not in training_targets
    }
    if unseen:
        unseen_target = min(unseen)
        prefix = f"{path}: validation association target unseen in train"
        detail = f"{prefix}: {unseen_target}"
        raise TeacherMaterializationError(detail=detail)
    return tuple(rows)


def _encode_row(row: TeacherCacheRow) -> str:
    return json.dumps(
        {
            "sample_id": row.sample_id,
            "group_id": row.group_id,
            "split": row.split,
            "features": row.features,
            "teacher_embedding": row.teacher_embedding,
            "type_label": RECORD_TYPES[row.type_index],
            "geometry_target": row.geometry_target,
            "association_target": row.association_target,
            "uncertainty_target": row.uncertainty_target,
            "byte_cost": row.byte_cost,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _format_key(key: tuple[str, str]) -> str:
    return f"observation_id={key[0]!r}, memory_id={key[1]!r}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Materialize spatial student training rows from teacher cache.",
    )
    _ = parser.add_argument("--teacher-cache", type=Path, required=True)
    _ = parser.add_argument("--supervision", type=Path, required=True)
    _ = parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    teacher_cache_path = cast("Path", arguments.teacher_cache)
    supervision_path = cast("Path", arguments.supervision)
    out_path = cast("Path", arguments.out)
    try:
        rows = materialize_teacher_cache(
            teacher_cache_path,
            supervision_path,
            out_path,
        )
    except (TeacherContractError, TeacherMaterializationError) as exc:
        parser.error(str(exc))
    splits = {
        split: sum(row.split == split for row in rows)
        for split in ("train", "validation")
    }
    summary = {"out": str(out_path), "rows": len(rows), "splits": splits}
    _ = sys.stdout.write(f"{json.dumps(summary, sort_keys=True)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
