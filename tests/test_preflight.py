from __future__ import annotations

from pathlib import Path

from worldmm_smvqa.preflight import (
    PreflightReport,
    inspect_prepared_dataset,
    write_preflight_report,
)
from worldmm_smvqa.schema import (
    AnswerChoice,
    FrameMetadata,
    GazeSample,
    ObjectMetadata,
    PoseSample,
    QALabelExample,
    QuestionRequest,
    SourceStreamExample,
)


def test_preflight_reports_coverage_and_1hz_selection(tmp_path: Path) -> None:
    fixture = tmp_path / "prepared"
    frame_root = fixture / "frames"
    source = SourceStreamExample(
        video_id="video-1",
        start_time=0.0,
        end_time=3.0,
        object_detections=(
            ObjectMetadata(
                start_time=0.1,
                end_time=0.2,
                label="mug",
                confidence=0.9,
                instance_id="mug-1",
                x=1.0,
                y=2.0,
                z=0.8,
            ),
        ),
        pose_samples=(PoseSample(timestamp=0.1, x=0.0, y=0.0, z=1.5),),
        gaze_samples=(GazeSample(timestamp=0.1, x=1.0, y=2.0, z=0.8),),
        frame_refs=("frame-01", "frame-05", "frame-11"),
        frame_metadata=(
            FrameMetadata(frame_ref="frame-01", timestamp=0.1, description="a"),
            FrameMetadata(frame_ref="frame-05", timestamp=0.5, description="b"),
            FrameMetadata(frame_ref="frame-11", timestamp=1.1, description="c"),
        ),
    )
    question = QuestionRequest(
        question_id="q1",
        video_id="video-1",
        question="Where is the mug?",
        question_time=2.0,
        answer_choices=_choices(),
        task="geometry",
    )
    label = QALabelExample(
        question_id=question.question_id,
        video_id=question.video_id,
        video_ids=question.video_ids,
        question=question.question,
        question_time=question.question_time,
        answer_choices=question.answer_choices,
        task=question.task,
        skill=question.skill,
        answer="A",
        is_answerable=True,
        evidence_list=("video-1:0.1:0.2:spatial",),
        verification_score=1.0,
    )
    _write_fixture(fixture, (source,), (question,), (label,))
    for frame_ref in source.frame_refs:
        path = frame_root / source.video_id / f"{frame_ref}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(b"frame")

    report = inspect_prepared_dataset(fixture, frame_root=frame_root)

    assert report.status == "pass"
    assert report.counts["selected_1hz_frames"] == 2
    assert report.coverage["object_xyz_percent"] == 100.0
    assert report.coverage["object_instance_id_percent"] == 100.0
    assert report.distributions["task"] == {"geometry": 1}
    assert report.distributions["evidence_store"] == {"spatial": 1}


def test_preflight_writes_errors_and_warnings(tmp_path: Path) -> None:
    fixture = tmp_path / "prepared"
    source = SourceStreamExample(
        video_id="video-1",
        start_time=0.0,
        end_time=10.0,
        object_detections=(
            ObjectMetadata(
                start_time=11.0,
                end_time=12.0,
                label="mug",
                confidence=0.5,
            ),
        ),
    )
    question = QuestionRequest(
        question_id="q1",
        video_id="video-1",
        question="Where is it?",
        question_time=1_700_000_000.0,
        answer_choices=_choices(),
    )
    label = QALabelExample(
        question_id=question.question_id,
        video_id=question.video_id,
        video_ids=question.video_ids,
        question=question.question,
        question_time=question.question_time,
        answer_choices=question.answer_choices,
        task=question.task,
        skill=question.skill,
        answer="A",
        is_answerable=True,
        evidence_list=("video-2:1:2:supermemory",),
        verification_score=1.0,
    )
    _write_fixture(fixture, (source, source), (question,), (label,))
    output = tmp_path / "preflight.json"

    written = write_preflight_report(fixture, output)
    report = PreflightReport.model_validate_json(output.read_text(encoding="utf-8"))
    error_codes = {issue.code for issue in report.errors}
    warning_codes = {issue.code for issue in report.warnings}

    assert written == report
    assert report.status == "fail"
    assert "duplicate_source_id" in error_codes
    assert "nested_interval_out_of_bounds" in error_codes
    assert "evidence_video_out_of_scope" in error_codes
    assert "question_time_out_of_bounds" in error_codes
    assert "common_timebase_risk" in error_codes
    assert "unsupported_evidence_store" not in warning_codes
    assert "task_metadata_missing" in warning_codes


def test_preflight_rejects_question_label_semantic_mismatch(tmp_path: Path) -> None:
    fixture = tmp_path / "prepared"
    sources = (
        SourceStreamExample(video_id="video-1", start_time=0.0, end_time=10.0),
        SourceStreamExample(video_id="video-2", start_time=0.0, end_time=10.0),
    )
    question = QuestionRequest(
        question_id="q1",
        video_id="video-1",
        video_ids=("video-1",),
        question="Where is the mug?",
        question_time=5.0,
        answer_choices=_choices(),
        task="geometry",
        skill="localization",
    )
    label = QALabelExample(
        question_id=question.question_id,
        video_id="video-2",
        video_ids=("video-2",),
        question="Where is the notebook?",
        question_time=6.0,
        answer_choices=tuple(
            AnswerChoice(choice_id=choice, text=f"other-{choice}", choice_ltype="place")
            for choice in ("A", "B", "C", "D")
        ),
        task="temporal",
        skill="tracking",
        answer="A",
        is_answerable=True,
        evidence_list=(),
        verification_score=1.0,
    )
    _write_fixture(fixture, sources, (question,), (label,))

    report = inspect_prepared_dataset(fixture)
    issues = [
        issue
        for issue in report.errors
        if issue.code == "question_label_semantic_mismatch"
    ]

    assert len(issues) == 1
    assert issues[0].record_id == "q1"
    assert issues[0].message.endswith(
        "video_id, video_ids, question, question_time, answer_choices, task, skill"
    )


def test_preflight_rejects_question_time_in_gap_between_scoped_sources(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "prepared"
    sources = (
        SourceStreamExample(video_id="video-1", start_time=0.0, end_time=10.0),
        SourceStreamExample(video_id="video-2", start_time=20.0, end_time=30.0),
    )
    question = QuestionRequest(
        question_id="q1",
        video_id="video-1",
        video_ids=("video-1", "video-2"),
        question="Where is the mug?",
        question_time=15.0,
        answer_choices=_choices(),
    )
    label = _label_for(question)
    _write_fixture(fixture, sources, (question,), (label,))

    report = inspect_prepared_dataset(fixture)

    assert "question_time_out_of_bounds" in {issue.code for issue in report.errors}


def test_preflight_accepts_question_time_in_any_scoped_relative_source(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "prepared"
    sources = (
        SourceStreamExample(video_id="video-1", start_time=0.0, end_time=10.0),
        SourceStreamExample(video_id="video-2", start_time=20.0, end_time=30.0),
    )
    question = QuestionRequest(
        question_id="q1",
        video_id="video-1",
        video_ids=("video-1", "video-2"),
        question="Where is the mug?",
        question_time=25.0,
        answer_choices=_choices(),
        task="geometry",
    )
    label = _label_for(question)
    _write_fixture(fixture, sources, (question,), (label,))

    report = inspect_prepared_dataset(fixture)
    error_codes = {issue.code for issue in report.errors}

    assert "question_time_out_of_bounds" not in error_codes
    assert "common_timebase_risk" not in error_codes


def test_preflight_rejects_mixed_epoch_and_relative_source_timebases(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "prepared"
    epoch = 1_700_000_000.0
    sources = (
        SourceStreamExample(video_id="video-1", start_time=0.0, end_time=10.0),
        SourceStreamExample(
            video_id="video-2",
            start_time=epoch,
            end_time=epoch + 10.0,
        ),
    )
    question = QuestionRequest(
        question_id="q1",
        video_id="video-1",
        question="Where is the mug?",
        question_time=5.0,
        answer_choices=_choices(),
    )
    label = _label_for(question)
    _write_fixture(fixture, sources, (question,), (label,))

    report = inspect_prepared_dataset(fixture)

    assert "common_timebase_risk" in {issue.code for issue in report.errors}


def _choices() -> tuple[AnswerChoice, ...]:
    return (
        AnswerChoice(choice_id="A", text="A", choice_ltype="place"),
        AnswerChoice(choice_id="B", text="B", choice_ltype="place"),
        AnswerChoice(choice_id="C", text="C", choice_ltype="place"),
        AnswerChoice(
            choice_id="D",
            text="This question cannot be answered.",
            choice_ltype="unanswerable",
        ),
    )


def _label_for(question: QuestionRequest) -> QALabelExample:
    return QALabelExample(
        question_id=question.question_id,
        video_id=question.video_id,
        video_ids=question.video_ids,
        question=question.question,
        question_time=question.question_time,
        answer_choices=question.answer_choices,
        task=question.task,
        skill=question.skill,
        answer="A",
        is_answerable=True,
        evidence_list=(),
        verification_score=1.0,
    )


def _write_fixture(
    fixture: Path,
    sources: tuple[SourceStreamExample, ...],
    questions: tuple[QuestionRequest, ...],
    labels: tuple[QALabelExample, ...],
) -> None:
    fixture.mkdir(parents=True, exist_ok=True)
    for name, records in (
        ("sources.jsonl", sources),
        ("questions.jsonl", questions),
        ("labels.jsonl", labels),
    ):
        _ = (fixture / name).write_text(
            "".join(f"{record.model_dump_json()}\n" for record in records),
            encoding="utf-8",
        )
