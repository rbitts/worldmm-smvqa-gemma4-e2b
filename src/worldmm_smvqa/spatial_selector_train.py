from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast, override

from pydantic import ValidationError

from worldmm_smvqa.chunking import read_source_streams
from worldmm_smvqa.config import (
    ConfigNotFoundError,
    MalformedConfigError,
    RemoteOnlyError,
    load_config,
    require_remote,
)
from worldmm_smvqa.fixtures import FixtureValidationError
from worldmm_smvqa.schema import FrozenModel, QALabelExample
from worldmm_smvqa.worldmm.spatial_compression import (
    DEFAULT_SELECTOR,
    FEATURE_NAMES,
    SpatialCompressionError,
    SpatialExperimentConfig,
    SpatialSelectorModel,
    SpatialTokenCandidate,
    build_spatial_token_candidates,
    load_spatial_experiment_config,
    resolve_spatial_memory_model,
)
from worldmm_smvqa.worldmm.spatial_diagnostics import (
    InvalidEvidenceSpanError,
    parse_evidence_span,
)

GEOMETRY_QUESTION_TERMS: Final = frozenset(
    {
        "above",
        "behind",
        "below",
        "distance",
        "front",
        "left",
        "near",
        "right",
        "spatial",
        "where",
        "zone",
    },
)
CLASSIFICATION_THRESHOLD: Final = 0.5


@dataclass(frozen=True, slots=True)
class SelectorTrainingError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"SelectorTrainingError: {self.detail}"


class SelectorTrainingRow(FrozenModel):
    question_id: str
    candidate_id: str
    features: dict[str, float]
    label: Literal[0, 1]


@dataclass(frozen=True, slots=True)
class SelectorTrainingResult:
    model: SpatialSelectorModel
    rows: int
    positives: int
    loss: float
    accuracy: float


def build_selector_training_rows(
    fixture_dir: Path,
    *,
    negative_ratio: int = 4,
    env: Mapping[str, str] | None = None,
    experiment_config: SpatialExperimentConfig | None = None,
) -> tuple[SelectorTrainingRow, ...]:
    if negative_ratio < 0:
        raise SelectorTrainingError(detail="negative_ratio must be >= 0")
    sources = read_source_streams(fixture_dir)
    labels = _read_labels(fixture_dir / "labels.jsonl")
    runtime_env = os.environ if env is None else env
    experiment = resolve_spatial_memory_model(
        runtime_env,
        config=experiment_config,
    )
    candidates = build_spatial_token_candidates(
        sources,
        experiment=experiment,
    ).candidates
    rows: list[SelectorTrainingRow] = []
    for label in labels:
        if (
            not label.is_answerable
            or not label.evidence_list
            or not _is_geometry_question(label)
        ):
            continue
        spans = tuple(parse_evidence_span(raw) for raw in label.evidence_list)
        video_ids = set(label.video_ids or (label.video_id,))
        scoped = tuple(
            candidate
            for candidate in candidates
            if candidate.record.video_id in video_ids
            and candidate.record.end_time <= label.question_time
        )
        positives = tuple(
            candidate
            for candidate in scoped
            if any(
                candidate.record.video_id == span.video_id
                and _overlaps(
                    candidate.record.start_time,
                    candidate.record.end_time,
                    span.start,
                    span.end,
                )
                for span in spans
            )
        )
        positive_ids = {candidate.record.memory_id for candidate in positives}
        negatives = tuple(
            sorted(
                (
                    candidate
                    for candidate in scoped
                    if candidate.record.memory_id not in positive_ids
                ),
                key=lambda candidate: (
                    -DEFAULT_SELECTOR.score(candidate.features),
                    candidate.record.memory_id,
                ),
            )[: max(1, len(positives)) * negative_ratio]
        )
        rows.extend(
            _training_row(label.question_id, candidate, 1)
            for candidate in positives
        )
        rows.extend(
            _training_row(label.question_id, candidate, 0)
            for candidate in negatives
        )
    if not rows:
        raise SelectorTrainingError(
            detail="no answerable geometry-QA training rows were produced",
        )
    return tuple(rows)


def train_selector_model(
    rows: Sequence[SelectorTrainingRow],
    *,
    epochs: int = 200,
    learning_rate: float = 0.1,
    l2: float = 1e-4,
) -> SelectorTrainingResult:
    if epochs <= 0:
        raise SelectorTrainingError(detail="epochs must be positive")
    if learning_rate <= 0.0:
        raise SelectorTrainingError(detail="learning_rate must be positive")
    if l2 < 0.0:
        raise SelectorTrainingError(detail="l2 must be >= 0")
    training_rows = tuple(rows)
    if not training_rows:
        raise SelectorTrainingError(detail="training rows are empty")
    if any(not row.features for row in training_rows):
        raise SelectorTrainingError(detail="training features are empty")
    positives = sum(row.label for row in training_rows)
    negatives = len(training_rows) - positives
    if positives == 0 or negatives == 0:
        raise SelectorTrainingError(
            detail="training requires positive and negative rows",
        )

    feature_names = (
        *FEATURE_NAMES,
        *tuple(
            sorted(
                {
                    name
                    for row in training_rows
                    for name in row.features
                    if name not in FEATURE_NAMES
                },
            ),
        ),
    )
    default_weights = dict(
        zip(
            DEFAULT_SELECTOR.feature_names,
            DEFAULT_SELECTOR.weights,
            strict=True,
        ),
    )
    weights = [default_weights.get(name, 0.0) for name in feature_names]
    bias = DEFAULT_SELECTOR.bias
    positive_weight = negatives / positives
    for _epoch in range(epochs):
        gradient = [0.0] * len(weights)
        bias_gradient = 0.0
        total_weight = 0.0
        for row in training_rows:
            sample_weight = positive_weight if row.label == 1 else 1.0
            probability = _sigmoid(
                bias
                + sum(
                    weight * row.features.get(name, 0.0)
                    for weight, name in zip(
                        weights,
                        feature_names,
                        strict=True,
                    )
                ),
            )
            error = (probability - row.label) * sample_weight
            for index, name in enumerate(feature_names):
                gradient[index] += error * row.features.get(name, 0.0)
            bias_gradient += error
            total_weight += sample_weight
        for index, weight in enumerate(weights):
            weights[index] -= learning_rate * (
                gradient[index] / total_weight + (l2 * weight)
            )
        bias -= learning_rate * bias_gradient / total_weight

    model = SpatialSelectorModel(
        feature_names=feature_names,
        weights=tuple(weights),
        bias=bias,
    )
    loss, accuracy = _evaluate(model, training_rows, positive_weight)
    return SelectorTrainingResult(
        model=model,
        rows=len(training_rows),
        positives=positives,
        loss=loss,
        accuracy=accuracy,
    )


def write_training_rows(
    rows: Sequence[SelectorTrainingRow],
    output: Path,
) -> None:
    _write_atomic(
        output,
        "".join(f"{row.model_dump_json()}\n" for row in rows),
    )


def write_selector_model(
    result: SelectorTrainingResult,
    output: Path,
) -> None:
    _write_atomic(output, result.model.model_dump_json(indent=2) + "\n")


def _read_training_rows(path: Path) -> tuple[SelectorTrainingRow, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SelectorTrainingError(detail=str(exc)) from exc
    try:
        rows = tuple(
            SelectorTrainingRow.model_validate_json(line)
            for line in lines
            if line.strip()
        )
    except ValidationError as exc:
        raise SelectorTrainingError(detail=str(exc)) from exc
    if not rows:
        raise SelectorTrainingError(detail=f"{path}: no training rows")
    return rows


def _read_labels(path: Path) -> tuple[QALabelExample, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FixtureValidationError(path=path, detail=str(exc)) from exc
    try:
        return tuple(
            QALabelExample.model_validate_json(line)
            for line in lines
            if line.strip()
        )
    except ValidationError as exc:
        raise FixtureValidationError(path=path, detail=str(exc)) from exc


def _training_row(
    question_id: str,
    candidate: SpatialTokenCandidate,
    label: Literal[0, 1],
) -> SelectorTrainingRow:
    return SelectorTrainingRow(
        question_id=question_id,
        candidate_id=candidate.record.memory_id,
        features=dict(candidate.features),
        label=label,
    )


def _is_geometry_question(label: QALabelExample) -> bool:
    terms = _tokens(
        " ".join(
            (
                label.question,
                *(choice.text for choice in label.answer_choices),
            ),
        ),
    )
    return bool(terms & GEOMETRY_QUESTION_TERMS)


def _tokens(text: str) -> frozenset[str]:
    cleaned = "".join(char if char.isalnum() else " " for char in text.lower())
    return frozenset(cleaned.split())


def _overlaps(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> bool:
    return left_start <= right_end and right_start <= left_end


def _evaluate(
    model: SpatialSelectorModel,
    rows: Sequence[SelectorTrainingRow],
    positive_weight: float,
) -> tuple[float, float]:
    loss = 0.0
    correct = 0
    total_weight = 0.0
    for row in rows:
        probability = min(1.0 - 1e-12, max(1e-12, model.score(row.features)))
        sample_weight = positive_weight if row.label == 1 else 1.0
        loss -= sample_weight * (
            row.label * math.log(probability)
            + (1 - row.label) * math.log(1.0 - probability)
        )
        correct += int(
            (probability >= CLASSIFICATION_THRESHOLD) == bool(row.label),
        )
        total_weight += sample_weight
    return loss / total_weight, correct / len(rows)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _ = temporary.write_text(text, encoding="utf-8")
        _ = temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and train the compact spatial-token selector.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    _ = prepare.add_argument("--fixture", type=Path, required=True)
    _ = prepare.add_argument("--out", type=Path, required=True)
    _ = prepare.add_argument("--negative-ratio", type=int, default=4)
    _ = prepare.add_argument("--experiment", type=Path)

    train = commands.add_parser("train")
    _ = train.add_argument(
        "--config",
        type=Path,
        default=Path("configs/remote.example.yaml"),
    )
    _ = train.add_argument("--input", type=Path, required=True)
    _ = train.add_argument("--out", type=Path, required=True)
    _ = train.add_argument("--epochs", type=int, default=200)
    _ = train.add_argument("--learning-rate", type=float, default=0.1)
    _ = train.add_argument("--l2", type=float, default=1e-4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = cast("str", args.command)
    try:
        if command == "prepare":
            fixture = cast("Path", args.fixture)
            output = cast("Path", args.out)
            negative_ratio = cast("int", args.negative_ratio)
            experiment_path = cast("Path | None", args.experiment)
            rows = build_selector_training_rows(
                fixture,
                negative_ratio=negative_ratio,
                experiment_config=(
                    load_spatial_experiment_config(experiment_path)
                    if experiment_path is not None
                    else None
                ),
            )
            write_training_rows(rows, output)
            payload = {
                "output": str(output),
                "rows": len(rows),
                "positives": sum(row.label for row in rows),
            }
        else:
            config = cast("Path", args.config)
            input_path = cast("Path", args.input)
            output = cast("Path", args.out)
            epochs = cast("int", args.epochs)
            learning_rate = cast("float", args.learning_rate)
            l2 = cast("float", args.l2)
            require_remote(
                load_config(config),
                "train spatial selector",
                os.environ,
            )
            result = train_selector_model(
                _read_training_rows(input_path),
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
            )
            write_selector_model(result, output)
            payload = {
                "output": str(output),
                "rows": result.rows,
                "positives": result.positives,
                "loss": round(result.loss, 6),
                "accuracy": round(result.accuracy, 6),
            }
    except (
        ConfigNotFoundError,
        FixtureValidationError,
        InvalidEvidenceSpanError,
        MalformedConfigError,
        RemoteOnlyError,
        SelectorTrainingError,
        SpatialCompressionError,
    ) as exc:
        _ = sys.stderr.write(f"{exc}\n")
        return 2
    _ = sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
