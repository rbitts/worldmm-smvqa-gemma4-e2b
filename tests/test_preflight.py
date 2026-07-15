from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import rfc8785
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue, TypeAdapter

from worldmm_smvqa import preflight
from worldmm_smvqa.attestation import signing_bytes, with_payload_sha256
from worldmm_smvqa.preflight import (
    PreflightReport,
    inspect_prepared_dataset,
    validate_teacher_oracle_inputs,
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

if TYPE_CHECKING:
    import pytest

_JSON_OBJECT = dict[str, JsonValue]
_JSON_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def _json_object(value: object) -> _JSON_OBJECT:
    assert isinstance(value, dict)
    return cast("_JSON_OBJECT", value)


def _json_list(value: object) -> list[JsonValue]:
    assert isinstance(value, list)
    return cast("list[JsonValue]", value)


def _read_json_object(path: Path) -> _JSON_OBJECT:
    raw: object = _JSON_ADAPTER.validate_json(path.read_text(encoding="utf-8"))
    return _json_object(raw)


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


def test_teacher_oracle_preflight_ready_and_fail_closed(tmp_path: Path) -> None:
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    result = validate_teacher_oracle_inputs(audit_path, config_path)
    assert result.status == "pass"

    audit = _read_json_object(audit_path)
    audit["operational_state"] = "blocked"
    _ = audit_path.write_text(json.dumps(audit), encoding="utf-8")
    config = _read_json_object(config_path)
    config.update(
        {
            "sensor_audit_digest": "wrong",
            "profile": "probe",
            "semantic": {"digest": "TBD"},
            "signing": {},
            "accounting": {},
            "paths": {"data": "/outside/data"},
            "qa": {
                "production_roots": [str(tmp_path / "company" / "labels")],
                "label_roots": [str(tmp_path / "company" / "labels")],
            },
        }
    )
    capabilities = _json_object(config["capabilities"])
    _json_object(capabilities["provider"])["digest"] = ""
    _json_object(capabilities["semantic"])["digest"] = "TBD"
    capabilities["signing"] = {}
    capabilities["accounting"] = {}
    _ = config_path.write_text(json.dumps(config), encoding="utf-8")

    codes = {
        blocker.code
        for blocker in validate_teacher_oracle_inputs(
            audit_path,
            config_path,
        ).blockers
    }
    assert {
        "sensor_audit_not_ready",
        "sensor_audit_digest_mismatch",
        "profile_mismatch",
        "unresolved_placeholder",
        "missing_capability_contract",
        "path_outside_company_root",
        "qa_label_root_exposure",
    } <= codes


def test_teacher_oracle_preflight_rejects_sensor_inventory_digest_mismatch(
    tmp_path: Path,
) -> None:
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    config = _read_json_object(config_path)
    config["sensor_inventory_digest"] = "0" * 64
    _ = config_path.write_text(json.dumps(config), encoding="utf-8")

    result = validate_teacher_oracle_inputs(audit_path, config_path)

    assert result.status == "fail"
    assert {blocker.code for blocker in result.blockers} == {
        "sensor_inventory_digest_mismatch"
    }


def test_teacher_oracle_preflight_rejects_symlink_and_incomplete_accounting(
    tmp_path: Path,
) -> None:
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    link = tmp_path / "company" / "linked-data"
    link.symlink_to(tmp_path / "company" / "data", target_is_directory=True)
    config = _read_json_object(config_path)
    _json_object(config["paths"])["data"] = str(link)
    accounting = _json_object(config["accounting"])
    accounting["command"] = "sacct"
    _ = config_path.write_text(json.dumps(config), encoding="utf-8")

    codes = {
        blocker.code
        for blocker in validate_teacher_oracle_inputs(
            audit_path,
            config_path,
        ).blockers
    }
    assert {"path_symlink", "accounting_command_incomplete"} <= codes


def test_teacher_oracle_preflight_rejects_duplicate_and_nonfinite_json(
    tmp_path: Path,
) -> None:
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    _ = config_path.write_text(
        '{"profile":"teacher-oracle","profile":"teacher-oracle"}',
        encoding="utf-8",
    )
    duplicate_codes = {
        blocker.code
        for blocker in validate_teacher_oracle_inputs(audit_path, config_path).blockers
    }
    assert "invalid_json" in duplicate_codes

    audit_path, config_path = _write_oracle_inputs(tmp_path)
    _ = audit_path.write_text(
        audit_path.read_text(encoding="utf-8").replace("100.0", "NaN", 1),
        encoding="utf-8",
    )
    nonfinite_codes = {
        blocker.code
        for blocker in validate_teacher_oracle_inputs(audit_path, config_path).blockers
    }
    assert "invalid_json" in nonfinite_codes
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    _ = config_path.write_text('{"overflow":1e9999}', encoding="utf-8")
    overflow_codes = {
        blocker.code
        for blocker in validate_teacher_oracle_inputs(audit_path, config_path).blockers
    }
    assert "invalid_json" in overflow_codes


def test_teacher_oracle_preflight_rejects_invented_capability_digest(
    tmp_path: Path,
) -> None:
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    config = _read_json_object(config_path)
    _json_object(_json_object(config["capabilities"])["code"])["sha256"] = "0" * 64
    _ = config_path.write_text(json.dumps(config), encoding="utf-8")

    codes = {
        blocker.code
        for blocker in validate_teacher_oracle_inputs(audit_path, config_path).blockers
    }

    assert "capability_digest_mismatch" in codes


def test_teacher_oracle_preflight_rejects_incomplete_phase_a_graph(
    tmp_path: Path,
) -> None:
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    config = _read_json_object(config_path)
    stages = _json_list(config["stage_specs"])
    _ = stages.pop()
    _ = config_path.write_text(json.dumps(config), encoding="utf-8")

    codes = {
        blocker.code
        for blocker in validate_teacher_oracle_inputs(audit_path, config_path).blockers
    }

    assert "stage_graph_incomplete" in codes


def test_teacher_oracle_preflight_binds_receipts_and_stable_producers(
    tmp_path: Path,
) -> None:
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    config = _read_json_object(config_path)
    checks = _json_object(config["self_checks"])
    runner = _json_object(checks["capability_runner"])
    artifact = Path(cast("str", runner["artifact"]))
    _ = artifact.write_text("pass", encoding="utf-8")
    runner["sha256"] = hashlib.sha256(b"pass").hexdigest()
    config["t1_location_mode"] = "stable_last_location"
    _ = config_path.write_text(json.dumps(config), encoding="utf-8")

    codes = {
        blocker.code
        for blocker in validate_teacher_oracle_inputs(audit_path, config_path).blockers
    }

    assert {"self_check_receipt_invalid", "producer_configured_mismatch"} <= codes


def test_teacher_oracle_preflight_rejects_tampered_signed_receipt(
    tmp_path: Path,
) -> None:
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    config = _read_json_object(config_path)
    runner = _json_object(_json_object(config["self_checks"])["capability_runner"])
    artifact = Path(cast("str", runner["artifact"]))
    receipt = _read_json_object(artifact)
    _json_object(_json_object(receipt["payload"])["outcomes"])["capability_sha256"] = {}
    payload = json.dumps(receipt).encode()
    _ = artifact.write_bytes(payload)
    runner["sha256"] = hashlib.sha256(payload).hexdigest()
    _ = config_path.write_text(json.dumps(config), encoding="utf-8")

    codes = {
        blocker.code
        for blocker in validate_teacher_oracle_inputs(audit_path, config_path).blockers
    }

    assert "self_check_receipt_invalid" in codes


_WRONG_HOST_UNAME: Final[os.uname_result] = os.uname_result(
    ("Linux", "test-host", "wrong-host", "0", "x86_64")
)


def _wrong_host_uname() -> os.uname_result:
    return _WRONG_HOST_UNAME


def test_teacher_oracle_preflight_rejects_wrong_host_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    monkeypatch.setattr(os, "uname", _wrong_host_uname)

    codes = {
        blocker.code
        for blocker in validate_teacher_oracle_inputs(audit_path, config_path).blockers
    }

    assert "self_check_receipt_invalid" in codes


def test_teacher_oracle_preflight_rejects_forged_resolver_vector(
    tmp_path: Path,
) -> None:
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    config = _read_json_object(config_path)
    resolver = _json_object(_json_object(config["self_checks"])["resolver"])
    artifact = Path(cast("str", resolver["artifact"]))
    receipt = _read_json_object(artifact)
    _json_object(_json_object(receipt["payload"])["outcomes"])["mount_crossing"] = (
        "opened"
    )
    payload = json.dumps(receipt).encode()
    _ = artifact.write_bytes(payload)
    resolver["sha256"] = hashlib.sha256(payload).hexdigest()
    _ = config_path.write_text(json.dumps(config), encoding="utf-8")

    codes = {
        blocker.code
        for blocker in validate_teacher_oracle_inputs(audit_path, config_path).blockers
    }

    assert "self_check_receipt_invalid" in codes


def test_teacher_oracle_preflight_requires_stable_identity_stage(
    tmp_path: Path,
) -> None:
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    config = _read_json_object(config_path)
    config["t1_location_mode"] = "stable_last_location"
    producer = _json_object(config["producer"])
    producer["configured"] = ["geometry", "semantic", "place", "identity"]
    _ = config_path.write_text(json.dumps(config), encoding="utf-8")

    codes = {
        blocker.code
        for blocker in validate_teacher_oracle_inputs(audit_path, config_path).blockers
    }

    assert "stage_graph_incomplete" in codes


def test_teacher_oracle_preflight_rejects_external_and_symlinked_sensor_roots(
    tmp_path: Path,
) -> None:
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    external_root = tmp_path / "external-sensors"
    external_root.mkdir()
    config = _read_json_object(config_path)
    _json_object(config["sensor_policy"])["approved_roots"] = [str(external_root)]
    _ = config_path.write_text(json.dumps(config), encoding="utf-8")

    external_codes = {
        blocker.code
        for blocker in validate_teacher_oracle_inputs(audit_path, config_path).blockers
    }
    assert "sensor_root_outside_company_root" in external_codes
    assert "sensor_root_outside_data_root" in external_codes

    audit_path, config_path = _write_oracle_inputs(tmp_path / "symlinked")
    alias = tmp_path / "symlinked" / "company" / "sensor-alias"
    alias.symlink_to(
        tmp_path / "symlinked" / "company" / "data", target_is_directory=True
    )
    config = _read_json_object(config_path)
    _json_object(config["sensor_policy"])["approved_roots"] = [str(alias)]
    _ = config_path.write_text(json.dumps(config), encoding="utf-8")

    symlink_codes = {
        blocker.code
        for blocker in validate_teacher_oracle_inputs(audit_path, config_path).blockers
    }
    assert "sensor_root_symlink" in symlink_codes


def test_teacher_oracle_preflight_rejects_oversized_oracle_inputs(
    tmp_path: Path,
) -> None:
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    _ = audit_path.write_bytes(b"x" * (preflight.ORACLE_AUDIT_MAX_BYTES + 1))
    _ = config_path.write_bytes(b"x" * (preflight.ORACLE_CONFIG_MAX_BYTES + 1))

    blockers = validate_teacher_oracle_inputs(audit_path, config_path).blockers

    assert {
        blocker.field for blocker in blockers if blocker.code == "input_too_large"
    } == {"sensor_audit", "experiment_config"}


def test_teacher_oracle_preflight_rejects_symlinked_oracle_inputs(
    tmp_path: Path,
) -> None:
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    audit_link = tmp_path / "sensor-audit-link.json"
    config_link = tmp_path / "experiment-link.json"
    audit_link.symlink_to(audit_path)
    config_link.symlink_to(config_path)

    blockers = validate_teacher_oracle_inputs(audit_link, config_link).blockers

    assert {
        blocker.field for blocker in blockers if blocker.code == "input_symlink"
    } == {"sensor_audit", "experiment_config"}


def test_teacher_oracle_preflight_requires_nofollow_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, config_path = _write_oracle_inputs(tmp_path)
    monkeypatch.delattr(
        preflight.os,  # pyright: ignore[reportPrivateLocalImportUsage]
        "O_NOFOLLOW",
    )

    blockers = validate_teacher_oracle_inputs(audit_path, config_path).blockers

    assert {
        blocker.field
        for blocker in blockers
        if blocker.code == "input_nofollow_unavailable"
    } == {"sensor_audit", "experiment_config"}


def _oracle_stage_specs() -> list[_JSON_OBJECT]:
    resources: _JSON_OBJECT = {
        "partition": "gpu",
        "nodes": 1,
        "gpus_per_node": 1,
        "cpus": 4,
        "memory": "16G",
        "time": "01:00:00",
    }

    def stage(
        name: str,
        role: str,
        variant: str | None,
        dependencies: _JSON_OBJECT,
    ) -> _JSON_OBJECT:
        return {
            "name": name,
            "role": role,
            "variant": variant,
            "dependencies": dependencies,
            "retries": 0,
            "resources": resources.copy(),
        }

    stages = [
        stage("preflight", "preflight", None, {}),
        *[
            stage(
                role,
                role,
                None,
                {"kind": "afterok", "stages": ["preflight"]},
            )
            for role in ("geometry", "semantic", "place")
        ],
        stage(
            "gate",
            "gate",
            None,
            {"kind": "afterany", "stages": ["geometry", "semantic", "place"]},
        ),
        stage("terminal", "terminal", None, {"kind": "afterany", "stages": ["gate"]}),
    ]
    qa_stages: list[str] = []
    for variant in ("E0", "T0", "T1"):
        prefix = variant.lower()
        materialize = f"{prefix}_materialize"
        retrieve = f"{prefix}_retrieve"
        qa = f"{prefix}_qa"
        qa_stages.append(qa)
        stages.extend(
            [
                stage(
                    materialize,
                    "materialize",
                    variant,
                    {"kind": "afterok", "stages": ["gate"]},
                ),
                stage(
                    retrieve,
                    "retrieve",
                    variant,
                    {"kind": "afterok", "stages": [materialize]},
                ),
                stage(qa, "qa", variant, {"kind": "afterok", "stages": [retrieve]}),
            ]
        )
    stages.extend(
        [
            stage(
                "evaluator",
                "evaluator",
                None,
                cast("_JSON_OBJECT", {"kind": "afterok", "stages": qa_stages}),
            ),
            stage(
                "finalizer",
                "finalizer",
                None,
                {"kind": "afterany", "stages": ["evaluator", "terminal"]},
            ),
        ]
    )
    return stages


def _write_oracle_inputs(tmp_path: Path) -> tuple[Path, Path]:
    company = tmp_path / "company"
    for name in ("code", "env", "data", "model", "labels"):
        (company / name).mkdir(parents=True, exist_ok=True)
    audit_path = tmp_path / "sensor-audit.json"
    audit: _JSON_OBJECT = {
        "version": "sensor-audit-v1",
        "operational_state": "ready",
        "provider_gate_decision": "go",
        "window_us": 30_000_000,
        "input_digest": "a" * 64,
        "config_digest": "b" * 64,
        "manifest_digest": "c" * 64,
        "observations_digest": "d" * 64,
        "frame_root_digest": hashlib.sha256(
            (
                f"{(company / 'data').stat().st_dev}:{(company / 'data').stat().st_ino}"
            ).encode()
        ).hexdigest(),
        "counts": {
            "selected_frames": 1,
            "observations": 1,
            "joined_observations": 1,
            "rgb_verified": 1,
            "intrinsics_available": 1,
            "trusted_pose_available": 0,
            "depth_available": 0,
            "gaze_available": 0,
        },
        "coverage": {
            "rgb_percent": 100.0,
            "intrinsics_percent": 100.0,
            "trusted_pose_percent": 0.0,
            "depth_percent": 0.0,
            "gaze_percent": 0.0,
        },
        "issues": [],
    }
    _ = audit_path.write_text(json.dumps(audit), encoding="utf-8")
    private_key = Ed25519PrivateKey.generate()
    public_key = (
        base64.urlsafe_b64encode(
            private_key.public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
        )
        .decode()
        .rstrip("=")
    )
    registry: _JSON_OBJECT = {
        "registry_version": "immutable-attestation-key-registry-v1",
        "keys": [
            {
                "key_id": public_key,
                "public_key_b64url": public_key,
                "purposes": [
                    f"oracle-preflight-self-check:{name}"
                    for name in (
                        "capability_runner",
                        "signer_vectors",
                        "resolver",
                        "quality",
                    )
                ],
                "not_before": 0,
                "not_after": 4_102_444_800,
                "revoked": False,
            }
        ],
    }
    capabilities: _JSON_OBJECT = {}
    for name in (
        "code",
        "environment",
        "data",
        "model",
        "provider",
        "semantic",
        "ontology",
        "signing",
        "accounting",
    ):
        artifact = company / f"{name}.contract"
        payload = (
            json.dumps(registry, separators=(",", ":")).encode()
            if name == "signing"
            else name.encode()
        )
        _ = artifact.write_bytes(payload)
        capabilities[name] = {
            "artifact": str(artifact),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    _json_object(capabilities["provider"])["policy"] = "pinned"
    accounting: _JSON_OBJECT = {
        "cluster": "teacher",
        "command": (
            "/opt/slurm/bin/sacct -D -X -n -P --clusters=teacher "
            "--jobs=job-1,job-2 "
            "--format=JobIDRaw,Cluster,State%64,ExitCode,Restarts,"
            "SLUID,OriginalSLUID"
        ),
        "fields": [
            "JobIDRaw",
            "Cluster",
            "State%64",
            "ExitCode",
            "Restarts",
            "SLUID",
            "OriginalSLUID",
        ],
        "version": "23.11",
        "job_id_injection": {"flag": "--jobs", "placeholder": "job-1,job-2"},
        "settle_policy": {
            "max_wait_seconds": 300,
            "poll_interval_seconds": 5,
        },
    }
    config: _JSON_OBJECT = {
        "profile": "teacher-oracle",
        "experiment_id": "EXP-0005",
        "result_class": "teacher_oracle",
        "lane": "teacher_oracle",
        "variants": ["E0", "T0", "T1"],
        "window_us": 30_000_000,
        "byte_budget": 4096,
        "sensor_audit_digest": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "sensor_inventory_digest": "c" * 64,
        "allowed_company_roots": [str(company)],
        "paths": {
            name: str(company / name) for name in ("code", "env", "data", "model")
        },
        "capabilities": capabilities,
        "accounting": accounting,
        "producer": {
            "requeue": False,
            "input_manifest": ["sources.jsonl", "sensor_frames.jsonl"],
            "configured": ["geometry", "semantic", "place"],
        },
        "sensor_policy": {
            "modalities": ["rgb", "intrinsics"],
            "approved_roots": [str(company / "data")],
        },
        "stage_specs": cast("list[JsonValue]", _oracle_stage_specs()),
        "stage_topology": {
            "producer_to_gate": "afterany",
            "gate_to_terminal": "afterany",
        },
        "t1_location_mode": "frame_bound_place",
        "signer_registry": {
            "policy": "registry",
            "registry_digest": _json_object(capabilities["signing"])["sha256"],
            "require_verified_signature": True,
        },
        "quality": {
            "utility_rule": "frozen-utility",
            "confidence_interval_rule": "frozen-confidence-interval",
            "selective_risk_rule": "frozen-selective-risk",
        },
        "resolver": {
            "provider_config": "provider",
            "semantic_provider": "semantic",
            "place_provider": "place",
            "openat2_resolve_mask": [
                "RESOLVE_IN_ROOT",
                "RESOLVE_NO_SYMLINKS",
                "RESOLVE_NO_MAGICLINKS",
                "RESOLVE_NO_XDEV",
            ],
            "forbidden_resolve_flags": ["RESOLVE_BENEATH"],
        },
        "qa": {
            "production_roots": [str(company / "data")],
            "label_roots": [str(company / "labels")],
            "label_root_owner": "evaluator",
        },
    }
    receipts: dict[str, _JSON_OBJECT] = {}
    for name in ("capability_runner", "signer_vectors", "resolver", "quality"):
        receipt_payload = preflight.expected_self_check_payload(name, config)
        canonical_payload = _JSON_ADAPTER.validate_python(receipt_payload)
        payload_sha256 = hashlib.sha256(rfc8785.dumps(canonical_payload)).hexdigest()
        issued_at = 1.0
        purpose = f"oracle-preflight-self-check:{name}"
        unsigned_envelope: _JSON_OBJECT = {
            "version": "signed-attestation-envelope-v1",
            "key_id": public_key,
            "purpose": purpose,
            "payload": canonical_payload,
            "payload_sha256": payload_sha256,
            "issued_at": issued_at,
        }
        signed_envelope = with_payload_sha256(unsigned_envelope)
        signature = private_key.sign(signing_bytes(signed_envelope, purpose))
        receipts[name] = {
            **signed_envelope,
            "signature_b64url": base64.urlsafe_b64encode(signature)
            .decode()
            .rstrip("="),
        }
    self_checks: _JSON_OBJECT = {}
    for name, receipt in receipts.items():
        artifact = company / f"{name}.receipt.json"
        payload = json.dumps(receipt).encode()
        _ = artifact.write_bytes(payload)
        self_checks[name] = {
            "artifact": str(artifact),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    config["self_checks"] = self_checks
    config_path = tmp_path / "experiment.json"
    _ = config_path.write_text(json.dumps(config), encoding="utf-8")
    return audit_path, config_path
