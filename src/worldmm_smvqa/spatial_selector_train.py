from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Self, cast, override

from pydantic import Field, ValidationError, model_validator

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
SHA256_HEX_LENGTH: Final = 64
DEFAULT_UTILITY_CACHE: Final = "selector_utility.jsonl"
DEFAULT_SPLIT_MANIFEST: Final = "selector_split_manifest.json"
type SelectorSplit = Literal["train", "validation", "test"]
type SelectorSupervisionMode = Literal[
    "counterfactual",
    "legacy-evidence-overlap",
]


@dataclass(frozen=True, slots=True)
class SelectorTrainingError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"SelectorTrainingError: {self.detail}"


class SelectorCounterfactualUtility(FrozenModel):
    baseline_qa_loss: float = Field(ge=0.0)
    deleted_qa_loss: float = Field(ge=0.0)
    baseline_qa_score: float = Field(ge=0.0, le=1.0)
    deleted_qa_score: float = Field(ge=0.0, le=1.0)
    geometry_coverage_gain: float = Field(ge=0.0)
    uncertainty_reduction: float = Field(ge=0.0)
    pose_information_gain: float = Field(ge=0.0)
    surprise: float = Field(ge=0.0)
    redundancy: float = Field(ge=0.0)
    actual_serialized_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def _require_finite_metrics(self) -> Self:
        values = (
            self.baseline_qa_loss,
            self.deleted_qa_loss,
            self.baseline_qa_score,
            self.deleted_qa_score,
            self.geometry_coverage_gain,
            self.uncertainty_reduction,
            self.pose_information_gain,
            self.surprise,
            self.redundancy,
        )
        if not all(math.isfinite(value) for value in values):
            msg = "counterfactual utility metrics must be finite"
            raise ValueError(msg)
        return self

    @property
    def qa_loss_delta(self) -> float:
        """Return QA loss increase caused by deleting this candidate."""
        return self.deleted_qa_loss - self.baseline_qa_loss

    @property
    def qa_score_delta(self) -> float:
        """Return QA score decrease caused by deleting this candidate."""
        return self.baseline_qa_score - self.deleted_qa_score

    @property
    def gross_utility(self) -> float:
        """Combine cached QA and geometry benefits before byte normalization."""
        return (
            self.qa_loss_delta
            + self.qa_score_delta
            + self.geometry_coverage_gain
            + self.uncertainty_reduction
            + self.pose_information_gain
            + self.surprise
            - self.redundancy
        )

    @property
    def value_per_kib(self) -> float:
        """Return the signed candidate utility per actual serialized KiB."""
        return self.gross_utility / (self.actual_serialized_bytes / 1024.0)

    @property
    def target_probability(self) -> float:
        """Map signed value per KiB to the linear selector's probability range."""
        return _sigmoid(self.value_per_kib)


class SelectorUtilityCacheRecord(SelectorCounterfactualUtility):
    question_id: str
    candidate_id: str


class SelectorSplitAssignment(FrozenModel):
    question_id: str
    participant_id: str
    session_id: str
    split: SelectorSplit


class SelectorSplitManifest(FrozenModel):
    assignments: tuple[SelectorSplitAssignment, ...]

    @model_validator(mode="after")
    def _require_disjoint_assignments(self) -> Self:
        if not self.assignments:
            msg = "split assignments are empty"
            raise ValueError(msg)
        question_splits: dict[str, SelectorSplit] = {}
        participant_splits: dict[str, SelectorSplit] = {}
        session_splits: dict[tuple[str, str], SelectorSplit] = {}
        for assignment in self.assignments:
            if not (
                assignment.question_id
                and assignment.participant_id
                and assignment.session_id
            ):
                msg = "split assignment identifiers must be non-empty"
                raise ValueError(msg)
            if assignment.question_id in question_splits:
                msg = f"duplicate question split assignment: {assignment.question_id}"
                raise ValueError(msg)
            _assign_split(
                question_splits,
                assignment.question_id,
                assignment.split,
                "question",
            )
            _assign_split(
                participant_splits,
                assignment.participant_id,
                assignment.split,
                "participant",
            )
            _assign_split(
                session_splits,
                (assignment.participant_id, assignment.session_id),
                assignment.split,
                "session",
            )
        present = {assignment.split for assignment in self.assignments}
        if not {"train", "validation"} <= present:
            msg = "split manifest requires train and validation assignments"
            raise ValueError(msg)
        return self


class SelectorTrainingRow(FrozenModel):
    question_id: str
    candidate_id: str
    video_id: str
    participant_id: str
    session_id: str
    split: SelectorSplit
    supervision_mode: SelectorSupervisionMode
    utility_cache_sha256: str | None
    split_manifest_sha256: str
    features: dict[str, float]
    label: Literal[0, 1]
    utility: SelectorCounterfactualUtility | None = None

    @model_validator(mode="after")
    def _require_matching_supervision(self) -> Self:
        if not all(
            (
                self.question_id,
                self.candidate_id,
                self.video_id,
                self.participant_id,
                self.session_id,
                self.split_manifest_sha256,
            ),
        ):
            msg = "training row identifiers and split hash must be non-empty"
            raise ValueError(msg)
        if not _is_sha256(self.split_manifest_sha256):
            msg = "split_manifest_sha256 must be a SHA-256 hex digest"
            raise ValueError(msg)
        if self.supervision_mode == "counterfactual":
            if self.utility is None or self.utility_cache_sha256 is None:
                msg = "counterfactual rows require utility and cache hash"
                raise ValueError(msg)
            if not _is_sha256(self.utility_cache_sha256):
                msg = "utility_cache_sha256 must be a SHA-256 hex digest"
                raise ValueError(msg)
            expected = int(self.utility.gross_utility > 0.0)
            if self.label != expected:
                msg = "counterfactual label must match utility sign"
                raise ValueError(msg)
        elif self.utility is not None or self.utility_cache_sha256 is not None:
            msg = "legacy rows cannot contain counterfactual utility"
            raise ValueError(msg)
        return self


@dataclass(frozen=True, slots=True)
class SelectorTrainingResult:
    model: SpatialSelectorModel
    rows: int
    positives: int
    loss: float
    accuracy: float
    training_rows: int
    validation_rows: int
    validation_loss: float
    validation_accuracy: float
    supervision_mode: SelectorSupervisionMode
    utility_cache_sha256: str | None
    split_manifest_sha256: str


class SelectorModelManifest(FrozenModel):
    supervision_mode: SelectorSupervisionMode
    utility_cache_sha256: str | None
    split_manifest_sha256: str
    model_sha256: str
    rows: int
    positives: int
    training_rows: int
    validation_rows: int
    training_loss: float
    training_accuracy: float
    validation_loss: float
    validation_accuracy: float


def build_selector_training_rows(  # noqa: PLR0913
    fixture_dir: Path,
    *,
    negative_ratio: int = 4,
    env: Mapping[str, str] | None = None,
    experiment_config: SpatialExperimentConfig | None = None,
    utility_cache: Path | None = None,
    split_manifest: Path | None = None,
    supervision_mode: SelectorSupervisionMode = "counterfactual",
) -> tuple[SelectorTrainingRow, ...]:
    if negative_ratio < 0:
        raise SelectorTrainingError(detail="negative_ratio must be >= 0")
    split_path = split_manifest or fixture_dir / DEFAULT_SPLIT_MANIFEST
    splits = _read_split_manifest(split_path)
    split_sha256 = _sha256(split_path)
    utilities: tuple[SelectorUtilityCacheRecord, ...] = ()
    utility_sha256 = ""
    if supervision_mode == "counterfactual":
        utility_path = utility_cache or fixture_dir / DEFAULT_UTILITY_CACHE
        utilities = _read_utility_cache(utility_path)
        utility_sha256 = _sha256(utility_path)
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
    if supervision_mode == "legacy-evidence-overlap":
        return _build_legacy_rows(
            labels,
            candidates,
            splits,
            split_sha256=split_sha256,
            negative_ratio=negative_ratio,
        )
    return _build_counterfactual_rows(
        labels,
        candidates,
        utilities,
        splits,
        utility_sha256=utility_sha256,
        split_sha256=split_sha256,
        negative_ratio=negative_ratio,
    )


def _build_counterfactual_rows(  # noqa: PLR0913
    labels: Sequence[QALabelExample],
    candidates: Sequence[SpatialTokenCandidate],
    utilities: Sequence[SelectorUtilityCacheRecord],
    splits: SelectorSplitManifest,
    *,
    utility_sha256: str,
    split_sha256: str,
    negative_ratio: int,
) -> tuple[SelectorTrainingRow, ...]:
    label_by_id = _unique_by_id(labels, "question_id", "QA label")
    candidate_by_id = _unique_candidates(candidates)
    assignment_by_question = {
        assignment.question_id: assignment for assignment in splits.assignments
    }
    seen: set[tuple[str, str]] = set()
    baselines: dict[str, tuple[float, float]] = {}
    rows: list[SelectorTrainingRow] = []
    for utility in utilities:
        cache_key = (utility.question_id, utility.candidate_id)
        if cache_key in seen:
            raise SelectorTrainingError(
                detail=(
                    "duplicate counterfactual utility row: "
                    f"{utility.question_id}/{utility.candidate_id}"
                ),
            )
        seen.add(cache_key)
        baseline = (utility.baseline_qa_loss, utility.baseline_qa_score)
        previous_baseline = baselines.setdefault(utility.question_id, baseline)
        if previous_baseline != baseline:
            raise SelectorTrainingError(
                detail=(
                    f"{utility.question_id}: inconsistent counterfactual QA baseline"
                ),
            )
        try:
            label = label_by_id[utility.question_id]
            candidate = candidate_by_id[utility.candidate_id]
            assignment = assignment_by_question[utility.question_id]
        except KeyError as exc:
            raise SelectorTrainingError(
                detail=f"counterfactual cache references unknown ID: {exc.args[0]}",
            ) from exc
        _require_causal_candidate(label, candidate)
        actual_bytes = _serialized_bytes(candidate)
        if utility.actual_serialized_bytes != actual_bytes:
            raise SelectorTrainingError(
                detail=(
                    f"{utility.candidate_id}: cached actual_serialized_bytes "
                    f"{utility.actual_serialized_bytes} != {actual_bytes}"
                ),
            )
        rows.append(
            _training_row(
                label.question_id,
                candidate,
                1 if utility.gross_utility > 0.0 else 0,
                assignment=assignment,
                supervision_mode="counterfactual",
                utility_cache_sha256=utility_sha256,
                split_manifest_sha256=split_sha256,
                utility=SelectorCounterfactualUtility.model_validate(
                    utility.model_dump(
                        exclude={"question_id", "candidate_id"},
                    ),
                ),
            ),
        )
    return _sample_rows(rows, negative_ratio=negative_ratio)


def _build_legacy_rows(
    labels: Sequence[QALabelExample],
    candidates: Sequence[SpatialTokenCandidate],
    splits: SelectorSplitManifest,
    *,
    split_sha256: str,
    negative_ratio: int,
) -> tuple[SelectorTrainingRow, ...]:
    assignment_by_question = {
        assignment.question_id: assignment for assignment in splits.assignments
    }
    rows: list[SelectorTrainingRow] = []
    for label in labels:
        if (
            not label.is_answerable
            or not label.evidence_list
            or not _is_geometry_question(label)
            or label.question_id not in assignment_by_question
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
            _training_row(
                label.question_id,
                candidate,
                1,
                assignment=assignment_by_question[label.question_id],
                supervision_mode="legacy-evidence-overlap",
                utility_cache_sha256=None,
                split_manifest_sha256=split_sha256,
                utility=None,
            )
            for candidate in positives
        )
        rows.extend(
            _training_row(
                label.question_id,
                candidate,
                0,
                assignment=assignment_by_question[label.question_id],
                supervision_mode="legacy-evidence-overlap",
                utility_cache_sha256=None,
                split_manifest_sha256=split_sha256,
                utility=None,
            )
            for candidate in negatives
        )
    return _require_rows(rows)


def _sample_rows(
    rows: Sequence[SelectorTrainingRow],
    *,
    negative_ratio: int,
) -> tuple[SelectorTrainingRow, ...]:
    sampled: list[SelectorTrainingRow] = []
    for question_id in sorted({row.question_id for row in rows}):
        question_rows = tuple(row for row in rows if row.question_id == question_id)
        positives = tuple(row for row in question_rows if row.label == 1)
        negatives = tuple(
            sorted(
                (row for row in question_rows if row.label == 0),
                key=lambda row: (
                    -row.utility.value_per_kib if row.utility is not None else 0.0,
                    row.candidate_id,
                ),
            )[: max(1, len(positives)) * negative_ratio]
        )
        sampled.extend((*positives, *negatives))
    return _require_rows(sampled)


def _require_rows(
    rows: Sequence[SelectorTrainingRow],
) -> tuple[SelectorTrainingRow, ...]:
    if not rows:
        raise SelectorTrainingError(
            detail="no selector training rows were produced",
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
    supervision_mode, utility_sha256, split_sha256 = _validate_row_contract(
        training_rows,
    )
    fit_rows = tuple(row for row in training_rows if row.split == "train")
    validation_rows = tuple(
        row for row in training_rows if row.split == "validation"
    )
    if not fit_rows or not validation_rows:
        raise SelectorTrainingError(
            detail="training requires non-empty train and validation splits",
        )
    fit_positives = sum(row.label for row in fit_rows)
    fit_negatives = len(fit_rows) - fit_positives
    if fit_positives == 0 or fit_negatives == 0:
        raise SelectorTrainingError(
            detail="train split requires positive and negative rows",
        )

    feature_names = (
        *FEATURE_NAMES,
        *tuple(
            sorted(
                {
                    name
                    for row in fit_rows
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
    positive_weight = fit_negatives / fit_positives
    for _epoch in range(epochs):
        gradient = [0.0] * len(weights)
        bias_gradient = 0.0
        total_weight = 0.0
        for row in fit_rows:
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
            error = (probability - _target_probability(row)) * sample_weight
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
    loss, accuracy = _evaluate(model, fit_rows, positive_weight)
    validation_loss, validation_accuracy = _evaluate(
        model,
        validation_rows,
        positive_weight,
    )
    return SelectorTrainingResult(
        model=model,
        rows=len(training_rows),
        positives=sum(row.label for row in training_rows),
        loss=loss,
        accuracy=accuracy,
        training_rows=len(fit_rows),
        validation_rows=len(validation_rows),
        validation_loss=validation_loss,
        validation_accuracy=validation_accuracy,
        supervision_mode=supervision_mode,
        utility_cache_sha256=utility_sha256,
        split_manifest_sha256=split_sha256,
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
    model_text = result.model.model_dump_json(indent=2) + "\n"
    _write_atomic(output, model_text)
    manifest = SelectorModelManifest(
        supervision_mode=result.supervision_mode,
        utility_cache_sha256=result.utility_cache_sha256,
        split_manifest_sha256=result.split_manifest_sha256,
        model_sha256=hashlib.sha256(model_text.encode("utf-8")).hexdigest(),
        rows=result.rows,
        positives=result.positives,
        training_rows=result.training_rows,
        validation_rows=result.validation_rows,
        training_loss=result.loss,
        training_accuracy=result.accuracy,
        validation_loss=result.validation_loss,
        validation_accuracy=result.validation_accuracy,
    )
    _write_atomic(
        selector_model_manifest_path(output),
        manifest.model_dump_json(indent=2) + "\n",
    )


def selector_model_manifest_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.manifest.json")


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


def _read_utility_cache(path: Path) -> tuple[SelectorUtilityCacheRecord, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SelectorTrainingError(
            detail=f"counterfactual utility cache required: {path}: {exc}",
        ) from exc
    try:
        records = tuple(
            SelectorUtilityCacheRecord.model_validate_json(line)
            for line in lines
            if line.strip()
        )
    except ValidationError as exc:
        raise SelectorTrainingError(
            detail=f"invalid utility cache {path}: {exc}",
        ) from exc
    if not records:
        raise SelectorTrainingError(detail=f"{path}: utility cache is empty")
    return records


def _read_split_manifest(path: Path) -> SelectorSplitManifest:
    try:
        return SelectorSplitManifest.model_validate_json(
            path.read_text(encoding="utf-8"),
        )
    except (OSError, ValidationError) as exc:
        raise SelectorTrainingError(
            detail=f"explicit split manifest required: {path}: {exc}",
        ) from exc


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise SelectorTrainingError(detail=f"cannot hash {path}: {exc}") from exc


def _is_sha256(value: str) -> bool:
    return len(value) == SHA256_HEX_LENGTH and all(
        char in "0123456789abcdef" for char in value
    )


def _training_row(  # noqa: PLR0913
    question_id: str,
    candidate: SpatialTokenCandidate,
    label: Literal[0, 1],
    *,
    assignment: SelectorSplitAssignment,
    supervision_mode: SelectorSupervisionMode,
    utility_cache_sha256: str | None,
    split_manifest_sha256: str,
    utility: SelectorCounterfactualUtility | None,
) -> SelectorTrainingRow:
    return SelectorTrainingRow(
        question_id=question_id,
        candidate_id=candidate.record.memory_id,
        video_id=candidate.record.video_id,
        participant_id=assignment.participant_id,
        session_id=assignment.session_id,
        split=assignment.split,
        supervision_mode=supervision_mode,
        utility_cache_sha256=utility_cache_sha256,
        split_manifest_sha256=split_manifest_sha256,
        features=dict(candidate.features),
        label=label,
        utility=utility,
    )


def _serialized_bytes(candidate: SpatialTokenCandidate) -> int:
    return len(candidate.record.model_dump_json().encode("utf-8")) + 1


def _require_causal_candidate(
    label: QALabelExample,
    candidate: SpatialTokenCandidate,
) -> None:
    video_ids = set(label.video_ids or (label.video_id,))
    if candidate.record.video_id not in video_ids:
        raise SelectorTrainingError(
            detail=(
                f"{label.question_id}/{candidate.record.memory_id}: "
                "candidate is outside question video scope"
            ),
        )
    if candidate.record.end_time > label.question_time:
        raise SelectorTrainingError(
            detail=(
                f"{label.question_id}/{candidate.record.memory_id}: "
                "candidate is available after question_time"
            ),
        )


def _unique_candidates(
    candidates: Sequence[SpatialTokenCandidate],
) -> dict[str, SpatialTokenCandidate]:
    indexed: dict[str, SpatialTokenCandidate] = {}
    for candidate in candidates:
        memory_id = candidate.record.memory_id
        if memory_id in indexed:
            raise SelectorTrainingError(
                detail=f"duplicate candidate memory ID: {memory_id}",
            )
        indexed[memory_id] = candidate
    return indexed


def _unique_by_id[T](
    values: Sequence[T],
    attribute: str,
    kind: str,
) -> dict[str, T]:
    indexed: dict[str, T] = {}
    for value in values:
        identifier = getattr(value, attribute, None)
        if not isinstance(identifier, str) or not identifier:
            raise SelectorTrainingError(detail=f"{kind} has invalid {attribute}")
        if identifier in indexed:
            raise SelectorTrainingError(
                detail=f"duplicate {kind} {attribute}: {identifier}",
            )
        indexed[identifier] = value
    return indexed


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


def _assign_split[T](
    assigned: dict[T, SelectorSplit],
    key: T,
    split: SelectorSplit,
    kind: str,
) -> None:
    previous = assigned.setdefault(key, split)
    if previous != split:
        msg = f"{kind} {key!r} crosses {previous}/{split} splits"
        raise ValueError(msg)


def _validate_row_contract(
    rows: Sequence[SelectorTrainingRow],
) -> tuple[SelectorSupervisionMode, str | None, str]:
    modes = {row.supervision_mode for row in rows}
    utility_hashes = {row.utility_cache_sha256 for row in rows}
    split_hashes = {row.split_manifest_sha256 for row in rows}
    if len(modes) != 1 or len(utility_hashes) != 1 or len(split_hashes) != 1:
        raise SelectorTrainingError(
            detail="training rows mix supervision modes or source hashes",
        )
    question_splits: dict[str, SelectorSplit] = {}
    participant_splits: dict[str, SelectorSplit] = {}
    session_splits: dict[tuple[str, str], SelectorSplit] = {}
    video_splits: dict[str, SelectorSplit] = {}
    try:
        for row in rows:
            _assign_split(
                question_splits,
                row.question_id,
                row.split,
                "question",
            )
            _assign_split(
                participant_splits,
                row.participant_id,
                row.split,
                "participant",
            )
            _assign_split(
                session_splits,
                (row.participant_id, row.session_id),
                row.split,
                "session",
            )
            _assign_split(video_splits, row.video_id, row.split, "video")
    except ValueError as exc:
        raise SelectorTrainingError(detail=str(exc)) from exc
    return (
        cast("SelectorSupervisionMode", modes.pop()),
        utility_hashes.pop(),
        split_hashes.pop(),
    )


def _target_probability(row: SelectorTrainingRow) -> float:
    if row.utility is None:
        return float(row.label)
    return row.utility.target_probability


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
        target = _target_probability(row)
        sample_weight = positive_weight if row.label == 1 else 1.0
        loss -= sample_weight * (
            target * math.log(probability)
            + (1.0 - target) * math.log(1.0 - probability)
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
    _ = prepare.add_argument("--utility-cache", type=Path)
    _ = prepare.add_argument("--split-manifest", type=Path)
    _ = prepare.add_argument(
        "--supervision-mode",
        choices=("counterfactual", "legacy-evidence-overlap"),
        default="counterfactual",
    )

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
            utility_cache = cast("Path | None", args.utility_cache)
            split_manifest = cast("Path | None", args.split_manifest)
            supervision_mode = cast(
                "SelectorSupervisionMode",
                args.supervision_mode,
            )
            rows = build_selector_training_rows(
                fixture,
                negative_ratio=negative_ratio,
                experiment_config=(
                    load_spatial_experiment_config(experiment_path)
                    if experiment_path is not None
                    else None
                ),
                utility_cache=utility_cache,
                split_manifest=split_manifest,
                supervision_mode=supervision_mode,
            )
            write_training_rows(rows, output)
            payload = {
                "output": str(output),
                "rows": len(rows),
                "positives": sum(row.label for row in rows),
                "supervision_mode": rows[0].supervision_mode,
                "utility_cache_sha256": rows[0].utility_cache_sha256,
                "split_manifest_sha256": rows[0].split_manifest_sha256,
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
                "manifest": str(selector_model_manifest_path(output)),
                "rows": result.rows,
                "positives": result.positives,
                "loss": round(result.loss, 6),
                "accuracy": round(result.accuracy, 6),
                "training_loss": round(result.loss, 6),
                "training_accuracy": round(result.accuracy, 6),
                "validation_loss": round(result.validation_loss, 6),
                "validation_accuracy": round(result.validation_accuracy, 6),
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
