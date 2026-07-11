# PyTorch is an optional remote dependency loaded dynamically by these checks.
# pyright: reportAny=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportPrivateUsage=false

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

from worldmm_smvqa.spatial_train import (
    DEFAULT_LOSS_WEIGHTS,
    RECORD_TYPES,
    SpatialTrainingError,
    StudentConfig,
    TeacherCacheDataset,
    _compose_losses,
    _global_sum_with_local_gradient,
    build_student,
    compute_losses,
    distributed_context,
    dry_run,
    load_checkpoint,
    save_checkpoint_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _row(
    sample_id: str,
    split: str,
    type_label: str,
    association_target: int,
    *,
    group_id: str | None = None,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "group_id": sample_id if group_id is None else group_id,
        "split": split,
        "features": [0.1, 0.2, 0.3],
        "teacher_embedding": [0.4, 0.5],
        "type_label": type_label,
        "geometry_target": [1.0, 2.0, 3.0, 0.5],
        "association_target": association_target,
        "uncertainty_target": 0.2,
        "byte_cost": 96.0,
    }


def _write_cache(path: Path, *, include_validation: bool = True) -> None:
    rows = [_row("train-1", "train", "object", 0)]
    if include_validation:
        rows.append(_row("validation-1", "validation", "plane", 0))
    _ = path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def test_module_import_does_not_eagerly_import_torch() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "python",
            "-c",
            (
                "import sys; import worldmm_smvqa.spatial_train; "
                "print('torch' in sys.modules)"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_teacher_cache_requires_explicit_consistent_split(tmp_path: Path) -> None:
    cache = tmp_path / "teacher.jsonl"
    row = _row("sample-1", "train", "object", 0)
    del row["split"]
    _ = cache.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(SpatialTrainingError, match="split"):
        _ = TeacherCacheDataset(cache)


def test_split_dataset_never_crosses_train_validation(tmp_path: Path) -> None:
    cache = tmp_path / "teacher.jsonl"
    _write_cache(cache)

    training = TeacherCacheDataset(cache, split="train")
    validation = TeacherCacheDataset(cache, split="validation")

    assert [row.sample_id for row in training.rows] == ["train-1"]
    assert [row.sample_id for row in validation.rows] == ["validation-1"]
    assert training.config == validation.config


def test_cache_rejects_group_leakage_and_unseen_validation_target(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "teacher.jsonl"
    leaking = (
        _row("train-1", "train", "object", 0, group_id="participant-1"),
        _row(
            "validation-1",
            "validation",
            "plane",
            0,
            group_id="participant-1",
        ),
    )
    _ = cache.write_text(
        "".join(f"{json.dumps(row)}\n" for row in leaking),
        encoding="utf-8",
    )
    with pytest.raises(SpatialTrainingError, match="group crosses splits"):
        _ = TeacherCacheDataset(cache)

    unseen = (
        _row("train-1", "train", "object", 0),
        _row("validation-1", "validation", "plane", 4),
    )
    _ = cache.write_text(
        "".join(f"{json.dumps(row)}\n" for row in unseen),
        encoding="utf-8",
    )
    with pytest.raises(SpatialTrainingError, match="unseen in train"):
        _ = TeacherCacheDataset(cache)


def test_distributed_context_validates_rank_environment() -> None:
    context = distributed_context(
        {"RANK": "3", "WORLD_SIZE": "8", "LOCAL_RANK": "3"},
    )

    assert (context.rank, context.world_size, context.local_rank) == (3, 8, 3)
    with pytest.raises(SpatialTrainingError, match="RANK"):
        _ = distributed_context({"RANK": "8", "WORLD_SIZE": "8"})


@pytest.mark.skipif(not HAS_TORCH, reason="remote extra is not installed")
def test_cpu_dry_run_covers_all_typed_heads_and_losses(tmp_path: Path) -> None:
    cache = tmp_path / "teacher.jsonl"
    _write_cache(cache)

    result = dry_run(cache, batch_size=2, hidden_dim=8)

    assert result["rows"] == 2
    assert result["record_types"] == list(RECORD_TYPES)
    assert result["split_counts"] == {"train": 1, "validation": 1}
    losses = result["losses"]
    assert set(losses) == {
        "total",
        "record_type",
        "geometry",
        "association",
        "uncertainty",
        "rate",
        "distillation",
        "expected_bytes",
    }
    assert all(value >= 0.0 for value in losses.values())


@pytest.mark.skipif(not HAS_TORCH, reason="remote extra is not installed")
def test_rate_loss_uses_serialized_byte_sum_not_mean() -> None:
    torch = importlib.import_module("torch")
    config = StudentConfig(
        input_dim=3,
        teacher_dim=2,
        geometry_dim=4,
        association_classes=1,
    )
    outputs = {
        "type_logits": torch.zeros((2, len(RECORD_TYPES))),
        "typed_geometry": torch.zeros((2, len(RECORD_TYPES), 4)),
        "association_logits": torch.zeros((2, 1)),
        "uncertainty": torch.ones(2),
        "rate_logit": torch.zeros(2),
        "distillation": torch.zeros((2, 2)),
    }
    batch = {
        "type_target": torch.tensor([0, RECORD_TYPES.index("no_write")]),
        "geometry_target": torch.zeros((2, 4)),
        "association_target": torch.zeros(2, dtype=torch.long),
        "uncertainty_target": torch.ones(2),
        "byte_cost": torch.tensor([100.0, 200.0]),
        "teacher_embedding": torch.zeros((2, 2)),
    }

    losses = compute_losses(outputs, batch, config)

    assert losses["expected_bytes"].item() == 150.0


@pytest.mark.skipif(not HAS_TORCH, reason="remote extra is not installed")
def test_global_masked_mean_has_ddp_correct_forward_and_gradient() -> None:
    torch = importlib.import_module("torch")
    rank_0_parameter = torch.tensor(1.0, requires_grad=True)
    rank_1_parameter = torch.tensor(1.0, requires_grad=True)
    rank_0_numerator = rank_0_parameter * 2.0
    rank_1_numerator = rank_1_parameter * 12.0
    global_numerator = rank_0_numerator.detach() + rank_1_numerator.detach()
    global_denominator = torch.tensor(4.0)

    rank_0_loss = (
        _global_sum_with_local_gradient(
            rank_0_numerator,
            global_numerator,
            2,
        )
        / global_denominator
    )
    rank_1_loss = (
        _global_sum_with_local_gradient(
            rank_1_numerator,
            global_numerator,
            2,
        )
        / global_denominator
    )
    rank_0_loss.backward()
    rank_1_loss.backward()

    assert rank_0_loss.item() == pytest.approx(3.5)
    assert rank_1_loss.item() == pytest.approx(3.5)
    ddp_averaged_gradient = (rank_0_parameter.grad + rank_1_parameter.grad) / 2.0
    assert ddp_averaged_gradient.item() == pytest.approx(3.5)
    rank_local_mean_average = ((2.0 / 1.0) + (12.0 / 3.0)) / 2.0
    assert rank_local_mean_average != pytest.approx(3.5)


@pytest.mark.skipif(not HAS_TORCH, reason="remote extra is not installed")
def test_loss_composition_uses_global_component_denominators() -> None:
    torch = importlib.import_module("torch")
    config = StudentConfig(
        input_dim=3,
        teacher_dim=2,
        geometry_dim=4,
        association_classes=1,
        rate_normalizer_bytes=100.0,
    )
    numerators = torch.tensor([8.0, 14.0, 10.0, 6.0, 4.0, 2.0, 50.0])
    denominators = torch.tensor([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

    losses = _compose_losses(
        numerators,
        denominators,
        config,
        DEFAULT_LOSS_WEIGHTS,
    )

    assert losses["record_type"].item() == pytest.approx(2.0)
    assert losses["geometry"].item() == pytest.approx(3.5)
    assert losses["association"].item() == pytest.approx(2.5)
    assert losses["uncertainty"].item() == pytest.approx(1.5)
    assert losses["rate"].item() == pytest.approx(1.5)
    assert losses["distillation"].item() == pytest.approx(0.5)


@pytest.mark.skipif(not HAS_TORCH, reason="remote extra is not installed")
def test_checkpoint_is_atomic_and_resumable(tmp_path: Path) -> None:
    torch = importlib.import_module("torch")

    checkpoint = tmp_path / "student.pt"
    config = StudentConfig(
        input_dim=3,
        teacher_dim=2,
        geometry_dim=4,
        association_classes=2,
        hidden_dim=8,
    )
    model = build_student(config)
    optimizer = torch.optim.AdamW(model.parameters())
    saved_parameter = next(model.parameters()).detach().clone()
    save_checkpoint_atomic(
        checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        next_epoch=4,
        global_step=17,
    )
    with torch.no_grad():
        next(model.parameters()).zero_()

    counters = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        expected_config=config,
    )

    assert counters == (4, 17)
    assert torch.equal(next(model.parameters()), saved_parameter)
    assert not tuple(tmp_path.glob(".*.tmp"))
    mismatched = StudentConfig(
        input_dim=3,
        teacher_dim=2,
        geometry_dim=4,
        association_classes=2,
        hidden_dim=8,
        learning_rate=0.02,
    )
    with pytest.raises(SpatialTrainingError, match="config mismatch"):
        _ = load_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            expected_config=mismatched,
        )


@pytest.mark.skipif(not HAS_TORCH, reason="remote extra is not installed")
def test_dry_run_rejects_cache_without_validation_split(tmp_path: Path) -> None:
    cache = tmp_path / "teacher.jsonl"
    _write_cache(cache, include_validation=False)

    with pytest.raises(SpatialTrainingError, match="train and validation"):
        _ = dry_run(cache)


def test_train_cli_remains_remote_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "student.pt"
    env = os.environ.copy()
    _ = env.pop("WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST", None)
    result = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "python",
            "-m",
            "worldmm_smvqa.spatial_train",
            "train",
            "--teacher-cache",
            str(tmp_path / "missing.jsonl"),
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "remote-only" in result.stderr
    assert not checkpoint.exists()
