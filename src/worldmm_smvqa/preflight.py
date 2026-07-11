from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, cast

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from worldmm_smvqa.schema import (
    ANSWER_CHOICE_COUNT,
    LocalTimedModel,
    QALabelExample,
    QuestionRequest,
    SourceStreamExample,
    is_unanswerable_choice,
)
from worldmm_smvqa.sensor_frames import (
    SensorFrameManifestError,
    build_sensor_frame_manifest,
)
from worldmm_smvqa.worldmm.spatial_diagnostics import STORES

PREFLIGHT_VERSION: Final = "smvqa-preflight-v1"
FRAME_EXTENSIONS: Final = (".jpg", ".jpeg", ".png", ".webp")
STORE_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_-]*$")
TIMESTAMP_EPSILON: Final = 1e-9
EPOCH_SCALE: Final = 100_000_000.0
PREVIEW_LIMIT: Final = 3
EVIDENCE_SPAN_PARTS: Final = 4
_JSON_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


class PreflightIssue(BaseModel):
    code: str
    message: str
    record_id: str | None = None


class PreflightReport(BaseModel):
    version: Literal["smvqa-preflight-v1"] = PREFLIGHT_VERSION
    status: Literal["pass", "warn", "fail"]
    input_dir: str
    frame_root: str | None
    counts: dict[str, int]
    coverage: dict[str, float | int | None]
    distributions: dict[str, dict[str, int]]
    errors: tuple[PreflightIssue, ...]
    warnings: tuple[PreflightIssue, ...]


def inspect_prepared_dataset(
    input_dir: Path,
    *,
    frame_root: Path | None = None,
) -> PreflightReport:
    errors: list[PreflightIssue] = []
    warnings: list[PreflightIssue] = []
    source_rows = _read_jsonl(
        input_dir / "sources.jsonl",
        SourceStreamExample,
        errors,
    )
    question_rows = _read_jsonl(
        input_dir / "questions.jsonl",
        QuestionRequest,
        errors,
    )
    label_rows = _read_jsonl(
        input_dir / "labels.jsonl",
        QALabelExample,
        errors,
    )

    sources = tuple(model for _, model in source_rows)
    questions = tuple(model for _, model in question_rows)
    labels = tuple(model for _, model in label_rows)
    counts: Counter[str] = Counter(
        sources_valid=len(sources),
        questions_valid=len(questions),
        labels_valid=len(labels),
    )
    timestamps: list[float] = []

    sources_by_video: dict[str, SourceStreamExample] = {}
    for source in sources:
        if source.video_id in sources_by_video:
            _issue(
                errors,
                "duplicate_source_id",
                f"duplicate source video_id: {source.video_id}",
                source.video_id,
            )
            continue
        sources_by_video[source.video_id] = source
        _inspect_source(
            source,
            counts=counts,
            timestamps=timestamps,
            errors=errors,
        )

    resolved_frame_root = _resolved_frame_root(input_dir, frame_root)
    _inspect_frame_files(
        sources,
        resolved_frame_root,
        counts=counts,
        errors=errors,
        warnings=warnings,
    )

    questions_by_id = _unique_questions(questions, errors)
    labels_by_id = _unique_labels(labels, errors)
    _inspect_question_label_sets(questions_by_id, labels_by_id, errors)

    task_distribution: Counter[str] = Counter()
    choice_type_distribution: Counter[str] = Counter()
    for raw, question in question_rows:
        task_distribution[_task_name(raw)] += 1
        choice_type_distribution.update(
            choice.choice_ltype for choice in question.answer_choices
        )
        _inspect_question(
            question,
            sources_by_video,
            timestamps=timestamps,
            errors=errors,
        )

    evidence_store_distribution: Counter[str] = Counter()
    for _, label in label_rows:
        _inspect_label_evidence(
            label,
            sources_by_video,
            evidence_store_distribution,
            timestamps=timestamps,
            errors=errors,
            warnings=warnings,
        )

    _add_coverage_warnings(counts, task_distribution, warnings)
    if {_timestamp_scale(value) for value in timestamps} == {"epoch", "relative"}:
        _issue(
            errors,
            "common_timebase_risk",
            "epoch-scale and relative timestamps are mixed",
        )

    status: Literal["pass", "warn", "fail"] = (
        "fail" if errors else "warn" if warnings else "pass"
    )
    return PreflightReport(
        status=status,
        input_dir=str(input_dir),
        frame_root=None if resolved_frame_root is None else str(resolved_frame_root),
        counts=dict(sorted(counts.items())),
        coverage=_coverage(counts),
        distributions={
            "task": dict(sorted(task_distribution.items())),
            "answer_choice_type": dict(sorted(choice_type_distribution.items())),
            "evidence_store": dict(sorted(evidence_store_distribution.items())),
        },
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def write_preflight_report(
    input_dir: Path,
    output: Path,
    *,
    frame_root: Path | None = None,
) -> PreflightReport:
    report = inspect_prepared_dataset(input_dir, frame_root=frame_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def _read_jsonl[ModelT: BaseModel](
    path: Path,
    model: type[ModelT],
    errors: list[PreflightIssue],
) -> tuple[tuple[dict[str, object], ModelT], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _issue(errors, "file_unreadable", f"{path}: {exc}")
        return ()
    rows: list[tuple[dict[str, object], ModelT]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw_object = _JSON_ADAPTER.validate_json(line)
        except ValidationError as exc:
            _issue(
                errors,
                "invalid_jsonl_row",
                f"{path.name}:{line_number}: {exc}",
            )
            continue
        if not isinstance(raw_object, dict):
            _issue(
                errors,
                "invalid_jsonl_row",
                f"{path.name}:{line_number}: JSONL row must be an object",
            )
            continue
        raw = cast("dict[str, object]", raw_object)
        try:
            rows.append((raw, model.model_validate(raw)))
        except ValidationError as exc:
            _issue(
                errors,
                "invalid_jsonl_row",
                f"{path.name}:{line_number}: {exc}",
            )
    if not rows:
        _issue(errors, "empty_file", f"{path} has no valid records")
    return tuple(rows)


def _inspect_source(
    source: SourceStreamExample,
    *,
    counts: Counter[str],
    timestamps: list[float],
    errors: list[PreflightIssue],
) -> None:
    timestamps.extend((source.start_time, source.end_time))
    _check_finite(source.start_time, "source.start_time", source.video_id, errors)
    _check_finite(source.end_time, "source.end_time", source.video_id, errors)
    for field_name, spans in (
        ("transcript_spans", source.transcript_spans),
        ("ocr_entries", source.ocr_entries),
        ("object_detections", source.object_detections),
    ):
        _inspect_intervals(source, field_name, spans, timestamps, errors)
    for field_name, samples in (
        ("pose_samples", source.pose_samples),
        ("gaze_samples", source.gaze_samples),
        ("frame_metadata", source.frame_metadata),
    ):
        _inspect_timestamps(
            source,
            field_name,
            tuple(sample.timestamp for sample in samples),
            timestamps,
            errors,
        )

    detections = source.object_detections
    counts["object_detections"] += len(detections)
    counts["object_detections_with_xyz"] += sum(
        detection.x is not None and detection.y is not None and detection.z is not None
        for detection in detections
    )
    counts["object_detections_with_instance_id"] += sum(
        bool(getattr(detection, "instance_id", None)) for detection in detections
    )
    counts["sources_with_pose"] += bool(source.pose_samples)
    counts["sources_with_gaze"] += bool(source.gaze_samples)
    counts["pose_samples"] += len(source.pose_samples)
    counts["gaze_samples"] += len(source.gaze_samples)
    counts["source_frames"] += len(source.frame_metadata)

    metadata_refs = tuple(frame.frame_ref for frame in source.frame_metadata)
    if source.frame_refs and set(source.frame_refs) != set(metadata_refs):
        _issue(
            errors,
            "frame_inventory_mismatch",
            "frame_refs and frame_metadata frame_ref values differ",
            source.video_id,
        )
    metadata_ref_set = set(metadata_refs)
    for entry in source.ocr_entries:
        if entry.frame_ref not in metadata_ref_set:
            _issue(
                errors,
                "ocr_frame_ref_missing",
                f"OCR frame_ref absent from frame_metadata: {entry.frame_ref}",
                source.video_id,
            )
    try:
        manifest = build_sensor_frame_manifest((source,))[0]
    except SensorFrameManifestError as exc:
        _issue(errors, "invalid_sensor_frames", exc.detail, source.video_id)
    else:
        counts["selected_1hz_frames"] += len(manifest.selected_frames)


def _inspect_intervals(
    source: SourceStreamExample,
    field_name: str,
    spans: Sequence[LocalTimedModel],
    timestamps: list[float],
    errors: list[PreflightIssue],
) -> None:
    previous_start: float | None = None
    for index, span in enumerate(spans):
        timestamps.extend((span.start_time, span.end_time))
        if not math.isfinite(span.start_time) or not math.isfinite(span.end_time):
            _issue(
                errors,
                "nonfinite_nested_interval",
                f"{field_name}[{index}] has non-finite time",
                source.video_id,
            )
        if (
            span.start_time < source.start_time - TIMESTAMP_EPSILON
            or span.end_time > source.end_time + TIMESTAMP_EPSILON
        ):
            _issue(
                errors,
                "nested_interval_out_of_bounds",
                f"{field_name}[{index}] is outside source interval",
                source.video_id,
            )
        if previous_start is not None and span.start_time < previous_start:
            _issue(
                errors,
                "nested_interval_out_of_order",
                f"{field_name} must be sorted by start_time",
                source.video_id,
            )
            break
        previous_start = span.start_time


def _inspect_timestamps(
    source: SourceStreamExample,
    field_name: str,
    values: Sequence[float],
    timestamps: list[float],
    errors: list[PreflightIssue],
) -> None:
    timestamps.extend(values)
    for index, value in enumerate(values):
        if not math.isfinite(value):
            _issue(
                errors,
                "nonfinite_nested_timestamp",
                f"{field_name}[{index}] has non-finite timestamp",
                source.video_id,
            )
        if (
            value < source.start_time - TIMESTAMP_EPSILON
            or value > source.end_time + TIMESTAMP_EPSILON
        ):
            _issue(
                errors,
                "nested_timestamp_out_of_bounds",
                f"{field_name}[{index}] is outside source interval",
                source.video_id,
            )
    if tuple(values) != tuple(sorted(values)):
        _issue(
            errors,
            "nested_timestamp_out_of_order",
            f"{field_name} must be sorted by timestamp",
            source.video_id,
        )


def _inspect_frame_files(
    sources: Sequence[SourceStreamExample],
    frame_root: Path | None,
    *,
    counts: Counter[str],
    errors: list[PreflightIssue],
    warnings: list[PreflightIssue],
) -> None:
    if counts["source_frames"] and frame_root is None:
        _issue(
            warnings,
            "frame_files_unchecked",
            "no frame root found; frame file existence was not checked",
        )
        return
    if frame_root is None:
        return
    missing: list[str] = []
    for source in sources:
        for frame in source.frame_metadata:
            counts["frame_files_checked"] += 1
            if _frame_exists(frame_root, source.video_id, frame.frame_ref):
                counts["frame_files_found"] += 1
            else:
                missing.append(f"{source.video_id}/{frame.frame_ref}")
    if missing:
        preview = ", ".join(missing[:PREVIEW_LIMIT])
        suffix = (
            ""
            if len(missing) <= PREVIEW_LIMIT
            else f" (+{len(missing) - PREVIEW_LIMIT} more)"
        )
        _issue(
            errors,
            "frame_file_missing",
            f"missing {len(missing)} frame file(s): {preview}{suffix}",
        )


def _inspect_question(
    question: QuestionRequest,
    sources_by_video: dict[str, SourceStreamExample],
    *,
    timestamps: list[float],
    errors: list[PreflightIssue],
) -> None:
    timestamps.append(question.question_time)
    _check_finite(
        question.question_time,
        "question_time",
        question.question_id,
        errors,
    )
    choice_ids = tuple(choice.choice_id for choice in question.answer_choices)
    if len(choice_ids) != ANSWER_CHOICE_COUNT or len(set(choice_ids)) != len(
        choice_ids,
    ):
        _issue(
            errors,
            "invalid_answer_choices",
            "question requires exactly four unique answer choices",
            question.question_id,
        )
    unanswerable_count = sum(
        is_unanswerable_choice(choice) for choice in question.answer_choices
    )
    if unanswerable_count != 1:
        _issue(
            errors,
            "invalid_unanswerable_choice",
            "question requires exactly one unanswerable choice",
            question.question_id,
        )
    scope = question.video_ids or (question.video_id,)
    if question.video_id not in scope:
        _issue(
            errors,
            "primary_video_out_of_scope",
            "primary video_id is absent from video_ids",
            question.question_id,
        )
    missing = sorted(set(scope) - sources_by_video.keys())
    if missing:
        _issue(
            errors,
            "question_unknown_video",
            f"unknown scoped video_id(s): {', '.join(missing)}",
            question.question_id,
        )
        return
    scoped_sources = tuple(sources_by_video[video_id] for video_id in scope)
    if not any(
        source.start_time - TIMESTAMP_EPSILON
        <= question.question_time
        <= source.end_time + TIMESTAMP_EPSILON
        for source in scoped_sources
    ):
        ranges = ", ".join(
            f"{source.video_id}:[{source.start_time}, {source.end_time}]"
            for source in scoped_sources
        )
        _issue(
            errors,
            "question_time_out_of_bounds",
            (
                f"question_time {question.question_time} is outside every scoped "
                f"source interval: {ranges}"
            ),
            question.question_id,
        )


def _inspect_label_evidence(  # noqa: PLR0913
    label: QALabelExample,
    sources_by_video: dict[str, SourceStreamExample],
    stores: Counter[str],
    *,
    timestamps: list[float],
    errors: list[PreflightIssue],
    warnings: list[PreflightIssue],
) -> None:
    choice_ids = tuple(choice.choice_id for choice in label.answer_choices)
    unanswerable_ids = tuple(
        choice.choice_id
        for choice in label.answer_choices
        if is_unanswerable_choice(choice)
    )
    if label.answer not in choice_ids:
        _issue(
            errors,
            "invalid_gold_choice",
            "label answer is not one of its choice IDs",
            label.question_id,
        )
    elif len(unanswerable_ids) == 1 and label.is_answerable == (
        label.answer == unanswerable_ids[0]
    ):
        _issue(
            errors,
            "label_answerability_mismatch",
            "label answerability disagrees with its gold choice",
            label.question_id,
        )
    scope = set(label.video_ids or (label.video_id,))
    for raw_span in label.evidence_list:
        parsed = _parse_evidence_span(raw_span)
        if parsed is None:
            _issue(
                errors,
                "invalid_evidence_grammar",
                f"expected video:start:end:store: {raw_span}",
                label.question_id,
            )
            continue
        video_id, start, end, store = parsed
        stores[store] += 1
        timestamps.extend((start, end))
        if not STORE_PATTERN.fullmatch(store):
            _issue(
                errors,
                "invalid_evidence_store",
                f"invalid evidence store name: {store}",
                label.question_id,
            )
        elif store not in STORES:
            _issue(
                warnings,
                "unsupported_evidence_store",
                f"evidence store is not supported by diagnostics: {store}",
                label.question_id,
            )
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            _issue(
                errors,
                "invalid_evidence_interval",
                f"evidence must have finite start < end: {raw_span}",
                label.question_id,
            )
            continue
        if video_id not in scope:
            _issue(
                errors,
                "evidence_video_out_of_scope",
                f"evidence video_id is outside question scope: {video_id}",
                label.question_id,
            )
        source = sources_by_video.get(video_id)
        if source is None:
            _issue(
                errors,
                "evidence_unknown_video",
                f"evidence references unknown video_id: {video_id}",
                label.question_id,
            )
        elif (
            start < source.start_time - TIMESTAMP_EPSILON
            or end > source.end_time + TIMESTAMP_EPSILON
        ):
            _issue(
                errors,
                "evidence_out_of_bounds",
                f"evidence interval is outside source bounds: {raw_span}",
                label.question_id,
            )
        if end > label.question_time + TIMESTAMP_EPSILON:
            _issue(
                errors,
                "future_evidence",
                f"evidence ends after question_time: {raw_span}",
                label.question_id,
            )


def _unique_questions(
    questions: Sequence[QuestionRequest],
    errors: list[PreflightIssue],
) -> dict[str, QuestionRequest]:
    result: dict[str, QuestionRequest] = {}
    for question in questions:
        if question.question_id in result:
            _issue(
                errors,
                "duplicate_question_id",
                f"duplicate question_id: {question.question_id}",
                question.question_id,
            )
        else:
            result[question.question_id] = question
    return result


def _unique_labels(
    labels: Sequence[QALabelExample],
    errors: list[PreflightIssue],
) -> dict[str, QALabelExample]:
    result: dict[str, QALabelExample] = {}
    for label in labels:
        if label.question_id in result:
            _issue(
                errors,
                "duplicate_label_id",
                f"duplicate label question_id: {label.question_id}",
                label.question_id,
            )
        else:
            result[label.question_id] = label
    return result


def _inspect_question_label_sets(
    questions: dict[str, QuestionRequest],
    labels: dict[str, QALabelExample],
    errors: list[PreflightIssue],
) -> None:
    missing = sorted(labels.keys() - questions.keys())
    extra = sorted(questions.keys() - labels.keys())
    if missing or extra:
        _issue(
            errors,
            "question_label_id_mismatch",
            f"question/label IDs differ; missing={missing} extra={extra}",
        )
    for question_id in sorted(questions.keys() & labels.keys()):
        question = questions[question_id]
        label = labels[question_id]
        mismatched_fields = tuple(
            field_name
            for field_name in (
                "video_id",
                "video_ids",
                "question",
                "question_time",
                "answer_choices",
                "task",
                "skill",
            )
            if getattr(question, field_name) != getattr(label, field_name)
        )
        if mismatched_fields:
            _issue(
                errors,
                "question_label_semantic_mismatch",
                "question/label fields differ: " + ", ".join(mismatched_fields),
                question_id,
            )


def _add_coverage_warnings(
    counts: Counter[str],
    task_distribution: Counter[str],
    warnings: list[PreflightIssue],
) -> None:
    detections = counts["object_detections"]
    if detections and counts["object_detections_with_xyz"] < detections:
        _issue(
            warnings,
            "object_xyz_coverage",
            "some object detections lack XYZ geometry",
        )
    if detections and counts["object_detections_with_instance_id"] < detections:
        _issue(
            warnings,
            "object_instance_coverage",
            "some object detections lack instance_id",
        )
    sources = counts["sources_valid"]
    if sources and counts["sources_with_pose"] < sources:
        _issue(warnings, "pose_coverage", "some sources have no pose samples")
    if sources and counts["sources_with_gaze"] < sources:
        _issue(warnings, "gaze_coverage", "some sources have no gaze samples")
    if task_distribution.get("unspecified"):
        _issue(
            warnings,
            "task_metadata_missing",
            "some questions have no task/category metadata",
        )


def _coverage(counts: Counter[str]) -> dict[str, float | int | None]:
    return {
        "object_xyz_percent": _percent(
            counts["object_detections_with_xyz"],
            counts["object_detections"],
        ),
        "object_instance_id_percent": _percent(
            counts["object_detections_with_instance_id"],
            counts["object_detections"],
        ),
        "source_pose_percent": _percent(
            counts["sources_with_pose"],
            counts["sources_valid"],
        ),
        "source_gaze_percent": _percent(
            counts["sources_with_gaze"],
            counts["sources_valid"],
        ),
        "selected_1hz_percent": _percent(
            counts["selected_1hz_frames"],
            counts["source_frames"],
        ),
        "frame_file_percent": _percent(
            counts["frame_files_found"],
            counts["frame_files_checked"],
        ),
    }


def _task_name(raw: dict[str, object]) -> str:
    metadata = raw.get("metadata")
    typed_metadata = (
        cast("dict[str, object]", metadata) if isinstance(metadata, dict) else {}
    )
    candidates = (
        raw.get("task"),
        raw.get("task_type"),
        raw.get("question_type"),
        raw.get("skill"),
        raw.get("category"),
        typed_metadata.get("task"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "unspecified"


def _parse_evidence_span(raw_span: str) -> tuple[str, float, float, str] | None:
    parts = raw_span.rsplit(":", 3)
    if len(parts) != EVIDENCE_SPAN_PARTS:
        return None
    video_id, raw_start, raw_end, store = parts
    if not video_id or not store:
        return None
    try:
        return video_id, float(raw_start), float(raw_end), store
    except ValueError:
        return None


def _resolved_frame_root(input_dir: Path, frame_root: Path | None) -> Path | None:
    if frame_root is not None:
        return frame_root
    local = input_dir / "frames"
    return local if local.is_dir() else None


def _frame_exists(frame_root: Path, video_id: str, frame_ref: str) -> bool:
    for base in (frame_root / video_id / frame_ref, frame_root / frame_ref):
        if base.is_file():
            return True
        if any(base.with_suffix(suffix).is_file() for suffix in FRAME_EXTENSIONS):
            return True
    return False


def _check_finite(
    value: float,
    field_name: str,
    record_id: str,
    errors: list[PreflightIssue],
) -> None:
    if not math.isfinite(value):
        _issue(
            errors,
            "nonfinite_timestamp",
            f"{field_name} must be finite",
            record_id,
        )


def _timestamp_scale(value: float) -> str:
    return "epoch" if abs(value) >= EPOCH_SCALE else "relative"


def _percent(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return round(100.0 * numerator / denominator, 2)


def _issue(
    destination: list[PreflightIssue],
    code: str,
    message: str,
    record_id: str | None = None,
) -> None:
    destination.append(
        PreflightIssue(code=code, message=message, record_id=record_id),
    )
