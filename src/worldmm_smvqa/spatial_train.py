# PyTorch stays optional and is imported dynamically on remote-only paths.
# pyright: reportAny=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportUnannotatedClassAttribute=false

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Final, Literal, TypedDict, cast, override

from worldmm_smvqa.config import (
    ConfigNotFoundError,
    MalformedConfigError,
    RemoteOnlyError,
    load_config,
    require_remote,
)

RECORD_TYPES: Final = (
    "object",
    "plane",
    "portal",
    "free_space",
    "landmark",
    "event",
    "no_write",
)
CHECKPOINT_VERSION: Final = 1
SHA256_HEX_LENGTH: Final = 64


@dataclass(frozen=True, slots=True)
class SpatialTrainingError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"SpatialTrainingError: {self.detail}"


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main(self) -> bool:
        """Return whether this process owns shared outputs."""
        return self.rank == 0


@dataclass(frozen=True, slots=True)
class TeacherCacheRow:
    sample_id: str
    group_id: str
    split: Literal["train", "validation"]
    features: tuple[float, ...]
    teacher_embedding: tuple[float, ...]
    type_index: int
    geometry_target: tuple[float, ...]
    association_target: int
    uncertainty_target: float
    byte_cost: float


@dataclass(frozen=True, slots=True)
class StudentConfig:
    input_dim: int
    teacher_dim: int
    geometry_dim: int
    association_classes: int
    hidden_dim: int = 32
    learning_rate: float = 1e-3
    rate_normalizer_bytes: float = 4096.0
    teacher_cache_sha256: str = ""

    def __post_init__(self) -> None:
        integer_fields = {
            "input_dim": self.input_dim,
            "teacher_dim": self.teacher_dim,
            "geometry_dim": self.geometry_dim,
            "association_classes": self.association_classes,
            "hidden_dim": self.hidden_dim,
        }
        invalid = tuple(name for name, value in integer_fields.items() if value < 1)
        if invalid:
            raise SpatialTrainingError(
                detail=f"positive dimensions required: {', '.join(invalid)}",
            )
        if self.learning_rate <= 0.0 or self.rate_normalizer_bytes <= 0.0:
            raise SpatialTrainingError(
                detail="learning rate and rate normalizer must be positive",
            )
        if self.teacher_cache_sha256 and (
            len(self.teacher_cache_sha256) != SHA256_HEX_LENGTH
            or any(char not in "0123456789abcdef" for char in self.teacher_cache_sha256)
        ):
            raise SpatialTrainingError(detail="invalid teacher-cache digest")


@dataclass(frozen=True, slots=True)
class LossWeights:
    record_type: float = 1.0
    geometry: float = 1.0
    association: float = 1.0
    uncertainty: float = 0.1
    rate: float = 0.01
    distillation: float = 1.0


DEFAULT_LOSS_WEIGHTS: Final = LossWeights()
LOSS_MEAN_COMPONENTS: Final = (
    "record_type",
    "geometry",
    "association",
    "uncertainty",
    "rate_bce",
    "distillation",
)
EXPECTED_BYTES_INDEX: Final = len(LOSS_MEAN_COMPONENTS)


class DryRunSummary(TypedDict):
    rows: int
    split_counts: dict[str, int]
    record_types: list[str]
    losses: dict[str, float]


class TeacherCacheDataset:
    """Validated adapter over offline teacher-cache JSONL."""

    def __init__(
        self,
        path: Path,
        *,
        split: Literal["train", "validation"] | None = None,
    ) -> None:
        """Load and validate a complete teacher cache."""
        self.path = path
        all_rows, cache_sha256 = _read_teacher_cache(path)
        training_rows = tuple(row for row in all_rows if row.split == "train")
        if not training_rows:
            raise SpatialTrainingError(detail=f"{path}: train split is empty")
        self.config = _config_from_rows(training_rows, cache_sha256)
        _validate_association_targets(path, all_rows, self.config)
        self.rows = tuple(
            row for row in all_rows if split is None or row.split == split
        )
        if not self.rows:
            raise SpatialTrainingError(detail=f"{path}: split {split!r} is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch = _require_torch()
        row = self.rows[index]
        return {
            "sample_id": row.sample_id,
            "features": torch.tensor(row.features, dtype=torch.float32),
            "teacher_embedding": torch.tensor(
                row.teacher_embedding,
                dtype=torch.float32,
            ),
            "type_target": torch.tensor(row.type_index, dtype=torch.long),
            "geometry_target": torch.tensor(
                row.geometry_target,
                dtype=torch.float32,
            ),
            "association_target": torch.tensor(
                row.association_target,
                dtype=torch.long,
            ),
            "uncertainty_target": torch.tensor(
                row.uncertainty_target,
                dtype=torch.float32,
            ),
            "byte_cost": torch.tensor(row.byte_cost, dtype=torch.float32),
        }


def distributed_context(env: Mapping[str, str]) -> DistributedContext:
    rank = _env_int(env, "RANK", 0)
    world_size = _env_int(env, "WORLD_SIZE", 1)
    local_rank = _env_int(env, "LOCAL_RANK", rank)
    if world_size < 1:
        raise SpatialTrainingError(detail="WORLD_SIZE must be positive")
    if rank < 0 or rank >= world_size:
        raise SpatialTrainingError(detail="RANK must be in [0, WORLD_SIZE)")
    if local_rank < 0:
        raise SpatialTrainingError(detail="LOCAL_RANK must be non-negative")
    return DistributedContext(rank=rank, world_size=world_size, local_rank=local_rank)


def build_student(config: StudentConfig) -> object:
    """Build a feature-level candidate head, not a raw RGB/IMU device model."""
    torch = _require_torch()

    class TypedCandidateHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(config.input_dim, config.hidden_dim),
                torch.nn.ReLU(),
            )
            self.type_head = torch.nn.Linear(config.hidden_dim, len(RECORD_TYPES))
            self.typed_heads = torch.nn.ModuleDict(
                {
                    record_type: torch.nn.Linear(
                        config.hidden_dim,
                        config.geometry_dim,
                    )
                    for record_type in RECORD_TYPES
                },
            )
            self.association_head = torch.nn.Linear(
                config.hidden_dim,
                config.association_classes,
            )
            self.uncertainty_head = torch.nn.Linear(config.hidden_dim, 1)
            self.rate_head = torch.nn.Linear(config.hidden_dim, 1)
            self.distillation_head = torch.nn.Linear(
                config.hidden_dim,
                config.teacher_dim,
            )

        def forward(self, features: object) -> dict[str, object]:
            hidden = self.encoder(features)
            return {
                "type_logits": self.type_head(hidden),
                "typed_geometry": torch.stack(
                    tuple(self.typed_heads[name](hidden) for name in RECORD_TYPES),
                    dim=1,
                ),
                "association_logits": self.association_head(hidden),
                "uncertainty": torch.nn.functional.softplus(
                    self.uncertainty_head(hidden).squeeze(-1),
                )
                + 1e-6,
                "rate_logit": self.rate_head(hidden).squeeze(-1),
                "distillation": self.distillation_head(hidden),
            }

    return TypedCandidateHead()


def compute_losses(
    outputs: Mapping[str, object],
    batch: Mapping[str, object],
    config: StudentConfig,
    *,
    weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
    aggregate_distributed: bool = True,
) -> dict[str, object]:
    torch = _require_torch()
    numerators, denominators = _local_loss_totals(outputs, batch, torch)
    if aggregate_distributed:
        numerators, denominators = _distributed_loss_totals(
            numerators,
            denominators,
            torch,
        )
    return _compose_losses(numerators, denominators, config, weights)


def _local_loss_totals(
    outputs: Mapping[str, object],
    batch: Mapping[str, object],
    torch: ModuleType,
) -> tuple[object, object]:
    functional = torch.nn.functional
    type_target = batch["type_target"]
    typed_geometry = outputs["typed_geometry"]
    batch_indices = torch.arange(type_target.shape[0], device=type_target.device)
    predicted_geometry = typed_geometry[batch_indices, type_target]
    no_write_index = RECORD_TYPES.index("no_write")
    write_target = (type_target != no_write_index).to(dtype=torch.float32)
    row_count = write_target.new_tensor(float(type_target.shape[0]))
    write_count = write_target.sum()
    record_type_sum = functional.cross_entropy(
        outputs["type_logits"],
        type_target,
        reduction="sum",
    )
    geometry_error = functional.mse_loss(
        predicted_geometry,
        batch["geometry_target"],
        reduction="none",
    ).mean(dim=1)
    geometry_sum = (geometry_error * write_target).sum()
    association_error = functional.cross_entropy(
        outputs["association_logits"],
        batch["association_target"],
        reduction="none",
    )
    association_sum = (association_error * write_target).sum()
    uncertainty_error = functional.smooth_l1_loss(
        outputs["uncertainty"],
        batch["uncertainty_target"],
        reduction="none",
    )
    uncertainty_sum = (uncertainty_error * write_target).sum()
    rate_logit = outputs["rate_logit"]
    expected_bytes = torch.sigmoid(rate_logit) * batch["byte_cost"]
    expected_byte_sum = expected_bytes.sum()
    rate_bce_sum = functional.binary_cross_entropy_with_logits(
        rate_logit,
        write_target,
        reduction="sum",
    )
    distillation_error = functional.mse_loss(
        outputs["distillation"],
        batch["teacher_embedding"],
        reduction="none",
    )
    distillation_sum = (
        distillation_error.reshape(type_target.shape[0], -1)
        .mean(
            dim=1,
        )
        .sum()
    )
    numerators = torch.stack(
        (
            record_type_sum,
            geometry_sum,
            association_sum,
            uncertainty_sum,
            rate_bce_sum,
            distillation_sum,
            expected_byte_sum,
        ),
    )
    denominators = torch.stack(
        (
            row_count,
            write_count,
            write_count,
            write_count,
            row_count,
            row_count,
        ),
    )
    return numerators, denominators


def _compose_losses(
    numerators: object,
    denominators: object,
    config: StudentConfig,
    weights: LossWeights,
) -> dict[str, object]:
    means = numerators[:EXPECTED_BYTES_INDEX] / denominators.clamp_min(1.0)
    record_type_loss = means[0]
    geometry_loss = means[1]
    association_loss = means[2]
    uncertainty_loss = means[3]
    # ponytail: hard cap stays in the serialized-memory writer; this is a
    # differentiable actual-byte regularizer, not a pretend minibatch cap.
    expected_byte_sum = numerators[EXPECTED_BYTES_INDEX]
    rate_loss = means[4] + expected_byte_sum / config.rate_normalizer_bytes
    distillation_loss = means[5]
    total = (
        weights.record_type * record_type_loss
        + weights.geometry * geometry_loss
        + weights.association * association_loss
        + weights.uncertainty * uncertainty_loss
        + weights.rate * rate_loss
        + weights.distillation * distillation_loss
    )
    return {
        "total": total,
        "record_type": record_type_loss,
        "geometry": geometry_loss,
        "association": association_loss,
        "uncertainty": uncertainty_loss,
        "rate": rate_loss,
        "distillation": distillation_loss,
        "expected_bytes": expected_byte_sum,
    }


def dry_run(
    teacher_cache: Path,
    *,
    batch_size: int = 2,
    hidden_dim: int = 32,
) -> DryRunSummary:
    torch = _require_torch()
    dataset = TeacherCacheDataset(teacher_cache)
    split_counts = {
        split: sum(row.split == split for row in dataset.rows)
        for split in ("train", "validation")
    }
    if not all(split_counts.values()):
        raise SpatialTrainingError(
            detail="teacher cache must contain train and validation rows",
        )
    config = _replace_config(dataset.config, hidden_dim=hidden_dim)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
    )
    model = build_student(config)
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        outputs = model(batch["features"])
        losses = compute_losses(outputs, batch, config)
    return {
        "rows": len(dataset),
        "split_counts": split_counts,
        "record_types": list(RECORD_TYPES),
        "losses": {
            name: float(value.detach().cpu().item()) for name, value in losses.items()
        },
    }


def save_checkpoint_atomic(  # noqa: PLR0913
    path: Path,
    *,
    model: object,
    optimizer: object,
    config: StudentConfig,
    next_epoch: int,
    global_step: int,
) -> None:
    torch = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    module = model.module if hasattr(model, "module") else model
    payload = {
        "version": CHECKPOINT_VERSION,
        "config": asdict(config),
        "next_epoch": next_epoch,
        "global_step": global_step,
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    try:
        torch.save(payload, temporary)
        _ = temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    *,
    model: object,
    optimizer: object,
    device: object,
    expected_config: StudentConfig | None = None,
) -> tuple[int, int]:
    torch = _require_torch()
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SpatialTrainingError(detail=f"cannot load {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != CHECKPOINT_VERSION:
        raise SpatialTrainingError(detail=f"unsupported checkpoint: {path}")
    if expected_config is not None and payload.get("config") != asdict(expected_config):
        raise SpatialTrainingError(detail=f"checkpoint config mismatch: {path}")
    try:
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        next_epoch = int(payload["next_epoch"])
        global_step = int(payload["global_step"])
    except (KeyError, TypeError, RuntimeError, ValueError) as exc:
        raise SpatialTrainingError(detail=f"invalid checkpoint {path}: {exc}") from exc
    if next_epoch < 0 or global_step < 0:
        raise SpatialTrainingError(detail=f"invalid counters in {path}")
    return next_epoch, global_step


def train(  # noqa: PLR0913, PLR0915
    teacher_cache: Path,
    checkpoint: Path,
    *,
    resume: Path | None,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    env: Mapping[str, str],
) -> dict[str, object]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise SpatialTrainingError(
            detail="epochs, batch-size, and learning-rate must be positive",
        )
    torch = _require_torch()
    context = distributed_context(env)
    distributed = context.world_size > 1
    device = _training_device(torch, context)
    if distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            rank=context.rank,
            world_size=context.world_size,
        )
    try:
        full_dataset = TeacherCacheDataset(teacher_cache)
        splits = {row.split for row in full_dataset.rows}
        if splits != {"train", "validation"}:
            raise SpatialTrainingError(
                detail="teacher cache must contain train and validation rows",
            )
        training_indices = tuple(
            index for index, row in enumerate(full_dataset.rows) if row.split == "train"
        )
        validation_indices = tuple(
            index
            for index, row in enumerate(full_dataset.rows)
            if row.split == "validation"
        )
        training_dataset = torch.utils.data.Subset(full_dataset, training_indices)
        validation_dataset = torch.utils.data.Subset(full_dataset, validation_indices)
        config = _replace_config(
            full_dataset.config,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
        )
        model = build_student(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        start_epoch = 0
        global_step = 0
        if resume is not None:
            start_epoch, global_step = load_checkpoint(
                resume,
                model=model,
                optimizer=optimizer,
                device=device,
                expected_config=config,
            )
            if start_epoch >= epochs:
                raise SpatialTrainingError(
                    detail="checkpoint already reached requested epochs",
                )
        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if device.type == "cuda" else None,
            )
        sampler = torch.utils.data.distributed.DistributedSampler(
            training_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
        )
        loader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
        )
        validation_rank_indices = tuple(
            range(context.rank, len(validation_dataset), context.world_size)
        )
        validation_rank_dataset = torch.utils.data.Subset(
            validation_dataset,
            validation_rank_indices,
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_rank_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        last_loss = math.nan
        validation_loss = math.nan
        for epoch in range(start_epoch, epochs):
            sampler.set_epoch(epoch)
            model.train()
            for raw_batch in loader:
                batch = _move_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                losses = compute_losses(model(batch["features"]), batch, config)
                losses["total"].backward()
                optimizer.step()
                global_step += 1
                last_loss = float(losses["total"].detach().item())
            validation_loss = _validation_loss(
                model.module if hasattr(model, "module") else model,
                validation_loader,
                config,
                device,
                torch,
                distributed=distributed,
            )
            if context.is_main:
                save_checkpoint_atomic(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    next_epoch=epoch + 1,
                    global_step=global_step,
                )
        if distributed:
            summary = torch.tensor(
                [last_loss, float(global_step)],
                dtype=torch.float64,
                device=device,
            )
            torch.distributed.all_reduce(summary)
            summary /= context.world_size
            last_loss = float(summary[0].item())
        return {
            "checkpoint": str(checkpoint),
            "epochs": epochs,
            "global_step": global_step,
            "last_loss": last_loss,
            "validation_loss": validation_loss,
            "rank": context.rank,
            "world_size": context.world_size,
        }
    finally:
        if distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _validation_loss(  # noqa: PLR0913
    model: object,
    loader: object,
    config: StudentConfig,
    device: object,
    torch: ModuleType,
    *,
    distributed: bool,
) -> float:
    model.eval()
    numerator_count = EXPECTED_BYTES_INDEX + 1
    denominator_count = len(LOSS_MEAN_COMPONENTS)
    summary = torch.zeros(
        numerator_count + denominator_count,
        dtype=torch.float64,
        device=device,
    )
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            numerators, denominators = _local_loss_totals(
                model(batch["features"]),
                batch,
                torch,
            )
            summary[:numerator_count] += numerators.detach().to(dtype=torch.float64)
            summary[numerator_count:] += denominators.detach().to(
                dtype=torch.float64,
            )
    if distributed:
        torch.distributed.all_reduce(summary)
    numerators = summary[:numerator_count]
    denominators = summary[numerator_count:]
    if denominators[0].item() == 0:
        raise SpatialTrainingError(detail="validation split is empty")
    losses = _compose_losses(
        numerators,
        denominators,
        config,
        DEFAULT_LOSS_WEIGHTS,
    )
    return float(losses["total"].item())


def _distributed_loss_totals(
    local_numerators: object,
    local_denominators: object,
    torch: ModuleType,
) -> tuple[object, object]:
    if not torch.distributed.is_initialized():
        return local_numerators, local_denominators
    global_numerators = local_numerators.detach().clone()
    global_denominators = local_denominators.detach().clone()
    torch.distributed.all_reduce(global_numerators)
    torch.distributed.all_reduce(global_denominators)
    world_size = torch.distributed.get_world_size()
    return (
        _global_sum_with_local_gradient(
            local_numerators,
            global_numerators,
            world_size,
        ),
        global_denominators,
    )


def _global_sum_with_local_gradient(
    local_value: object,
    global_value: object,
    world_size: int,
) -> object:
    """Expose a global forward value and DDP-correct local gradient."""
    # DDP averages parameter gradients. Scaling each local derivative by the
    # world size makes that average equal the derivative of the global sum.
    return local_value * world_size + global_value - local_value.detach() * world_size


def _read_teacher_cache(path: Path) -> tuple[tuple[TeacherCacheRow, ...], str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpatialTrainingError(detail=f"cannot read {path}: {exc}") from exc
    lines = text.splitlines()
    rows: list[TeacherCacheRow] = []
    sample_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            row = _parse_teacher_row(raw)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise SpatialTrainingError(
                detail=f"{path}: line {line_number}: {exc}",
            ) from exc
        if row.sample_id in sample_ids:
            raise SpatialTrainingError(
                detail=f"{path}: line {line_number}: duplicate sample_id",
            )
        sample_ids.add(row.sample_id)
        rows.append(row)
    if not rows:
        raise SpatialTrainingError(detail=f"{path}: no teacher-cache rows")
    _validate_dimensions(path, rows)
    return tuple(rows), hashlib.sha256(text.encode()).hexdigest()


def _parse_teacher_row(raw: object) -> TeacherCacheRow:
    if not isinstance(raw, dict):
        detail = "row must be a JSON object"
        raise TypeError(detail)
    sample_id = raw["sample_id"]
    group_id = raw["group_id"]
    split = raw["split"]
    type_label = raw["type_label"]
    association_target = raw["association_target"]
    if not isinstance(sample_id, str) or not sample_id.strip():
        detail = "sample_id must be a non-empty string"
        raise ValueError(detail)
    if not isinstance(group_id, str) or not group_id.strip():
        detail = "group_id must be a non-empty string"
        raise ValueError(detail)
    if split not in {"train", "validation"}:
        detail = "split must be 'train' or 'validation'"
        raise ValueError(detail)
    if not isinstance(type_label, str) or type_label not in RECORD_TYPES:
        detail = f"type_label must be one of {RECORD_TYPES}"
        raise ValueError(detail)
    if isinstance(association_target, bool) or not isinstance(
        association_target,
        int,
    ):
        detail = "association_target must be an integer"
        raise TypeError(detail)
    if association_target < 0:
        detail = "association_target must be non-negative"
        raise ValueError(detail)
    uncertainty_target = _finite_number(
        raw["uncertainty_target"],
        "uncertainty_target",
    )
    byte_cost = _finite_number(raw["byte_cost"], "byte_cost")
    if uncertainty_target <= 0.0:
        detail = "uncertainty_target must be positive"
        raise ValueError(detail)
    if byte_cost < 0.0:
        detail = "byte_cost must be non-negative"
        raise ValueError(detail)
    return TeacherCacheRow(
        sample_id=sample_id,
        group_id=group_id,
        split=cast("Literal['train', 'validation']", split),
        features=_finite_vector(raw["features"], "features"),
        teacher_embedding=_finite_vector(
            raw["teacher_embedding"],
            "teacher_embedding",
        ),
        type_index=RECORD_TYPES.index(type_label),
        geometry_target=_finite_vector(
            raw["geometry_target"],
            "geometry_target",
        ),
        association_target=association_target,
        uncertainty_target=uncertainty_target,
        byte_cost=byte_cost,
    )


def _finite_vector(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        detail = f"{name} must be a non-empty JSON array"
        raise TypeError(detail)
    return tuple(_finite_number(item, name) for item in value)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        detail = f"{name} must contain finite numbers"
        raise TypeError(detail)
    result = float(value)
    if not math.isfinite(result):
        detail = f"{name} must contain finite numbers"
        raise ValueError(detail)
    return result


def _validate_dimensions(path: Path, rows: Sequence[TeacherCacheRow]) -> None:
    dimensions = {
        (len(row.features), len(row.teacher_embedding), len(row.geometry_target))
        for row in rows
    }
    if len(dimensions) != 1:
        raise SpatialTrainingError(detail=f"{path}: inconsistent vector dimensions")
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(row.group_id, set()).add(row.split)
    leaking = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking:
        raise SpatialTrainingError(
            detail=f"{path}: group crosses splits: {leaking[0]}",
        )


def _validate_association_targets(
    path: Path,
    rows: Sequence[TeacherCacheRow],
    config: StudentConfig,
) -> None:
    training_targets = {row.association_target for row in rows if row.split == "train"}
    if training_targets != set(range(config.association_classes)):
        raise SpatialTrainingError(
            detail=f"{path}: train association targets must be contiguous from zero",
        )
    unseen = sorted(
        {
            row.association_target
            for row in rows
            if row.split == "validation"
            and row.association_target not in training_targets
        },
    )
    if unseen:
        raise SpatialTrainingError(
            detail=(
                f"{path}: validation association target unseen in train: {unseen[0]}"
            ),
        )


def _config_from_rows(
    rows: Sequence[TeacherCacheRow],
    cache_sha256: str,
) -> StudentConfig:
    first = rows[0]
    return StudentConfig(
        input_dim=len(first.features),
        teacher_dim=len(first.teacher_embedding),
        geometry_dim=len(first.geometry_target),
        association_classes=max(row.association_target for row in rows) + 1,
        teacher_cache_sha256=cache_sha256,
    )


def _replace_config(
    config: StudentConfig,
    *,
    hidden_dim: int,
    learning_rate: float | None = None,
) -> StudentConfig:
    return replace(
        config,
        hidden_dim=hidden_dim,
        learning_rate=(
            config.learning_rate if learning_rate is None else learning_rate
        ),
    )


def _require_torch() -> ModuleType:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise SpatialTrainingError(
            detail="PyTorch is required; install the project remote extra",
        ) from exc


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SpatialTrainingError(detail=f"{name} must be an integer") from exc


def _training_device(torch: ModuleType, context: DistributedContext) -> object:
    if not torch.cuda.is_available():
        raise SpatialTrainingError(
            detail="CUDA is required for training; use dry-run for CPU validation",
        )
    if context.local_rank >= torch.cuda.device_count():
        raise SpatialTrainingError(detail="LOCAL_RANK exceeds visible CUDA devices")
    torch.cuda.set_device(context.local_rank)
    return torch.device("cuda", context.local_rank)


def _move_batch(batch: Mapping[str, object], device: object) -> dict[str, object]:
    return {
        key: value if key == "sample_id" else value.to(device)
        for key, value in batch.items()
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the typed spatial student from an offline teacher cache.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    dry = commands.add_parser("dry-run")
    _add_data_args(dry)

    train_parser = commands.add_parser("train")
    _add_data_args(train_parser)
    _ = train_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/remote.example.yaml"),
    )
    _ = train_parser.add_argument("--checkpoint", type=Path, required=True)
    _ = train_parser.add_argument("--resume", type=Path)
    _ = train_parser.add_argument("--epochs", type=int, default=1)
    _ = train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("--teacher-cache", type=Path, required=True)
    _ = parser.add_argument("--batch-size", type=int, default=8)
    _ = parser.add_argument("--hidden-dim", type=int, default=32)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = cast("str", args.command)
    try:
        if command == "dry-run":
            payload = dry_run(
                cast("Path", args.teacher_cache),
                batch_size=cast("int", args.batch_size),
                hidden_dim=cast("int", args.hidden_dim),
            )
        else:
            require_remote(
                load_config(cast("Path", args.config)),
                "train typed spatial student",
                os.environ,
            )
            payload = train(
                cast("Path", args.teacher_cache),
                cast("Path", args.checkpoint),
                resume=cast("Path | None", args.resume),
                epochs=cast("int", args.epochs),
                batch_size=cast("int", args.batch_size),
                hidden_dim=cast("int", args.hidden_dim),
                learning_rate=cast("float", args.learning_rate),
                env=os.environ,
            )
    except (
        ConfigNotFoundError,
        MalformedConfigError,
        RemoteOnlyError,
        SpatialTrainingError,
    ) as exc:
        _ = sys.stderr.write(f"{exc}\n")
        return 2
    if command == "train" and payload["rank"] != 0:
        return 0
    _ = sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
