from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from worldmm_smvqa.worldmm.gcut3r_teacher import (
    EMPTY_PREFIX_SHA256,
    CameraIntrinsics,
    Cut3RPaths,
    DepthGuidance,
    GCut3RTeacherAdapter,
    PoseGuidance,
    ProviderStep,
    TeacherCacheRecord,
    TeacherConfigurationError,
    TeacherContractError,
    TeacherObservation,
    TeacherRequest,
    TeacherResponse,
    build_cut3r_demo_command,
    build_teacher_cache_record,
    decode_teacher_request,
    decode_teacher_response,
    encode_teacher_request,
    encode_teacher_response,
    main,
    read_teacher_cache,
    resolve_cut3r_paths,
    resolve_gcut3r_paths,
    validate_teacher_cache,
    write_teacher_cache,
)
from worldmm_smvqa.worldmm.typed_memory import (
    NoWriteMemoryRecord,
    SpatialUncertainty,
    ValidityInterval,
)


def _uncertainty() -> SpatialUncertainty:
    return SpatialUncertainty(
        covariance_xyz=((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
        standard_deviation_m=0.1,
    )


def _no_write(request: TeacherRequest) -> NoWriteMemoryRecord:
    timestamp = request.timestamp
    return NoWriteMemoryRecord(
        memory_id=f"candidate:{request.observation_id}",
        source_video_id=request.video_id,
        entity_id=f"entity:{request.observation_id}",
        instance_id=f"candidate-instance:{request.observation_id}",
        local_frame_id=request.local_frame_id,
        geometry_uncertainty=_uncertainty(),
        validity=ValidityInterval(start_time=timestamp, end_time=timestamp),
        first_seen_time=timestamp,
        last_seen_time=timestamp,
        observation_count=1,
        confidence=0.25,
        provenance="model_inferred",
        evidence_refs=(request.frame_ref,),
        candidate_type="landmark",
        reason="below write threshold",
    )


class _MockProvider:
    provider_id: str = "mock-gcut3r-v1"

    def __init__(self) -> None:
        self.requests: list[TeacherRequest] = []

    def infer(
        self,
        request: TeacherRequest,
        previous_state: object | None,
    ) -> ProviderStep:
        assert previous_state == (
            None if request.sequence_index == 0 else request.sequence_index - 1
        )
        self.requests.append(request)
        response = TeacherResponse(
            observation_id=request.observation_id,
            video_id=request.video_id,
            timestamp=request.timestamp,
            observed_through_time=request.timestamp,
            state_ref=f"state:{request.sequence_index}",
            records=(_no_write(request),),
            pointmap_ref=f"pointmap:{request.observation_id}",
            confidence_ref=f"confidence:{request.observation_id}",
        )
        return ProviderStep(response=response, state=request.sequence_index)


def _observations() -> tuple[TeacherObservation, ...]:
    covariance = (1.0,) * 36
    intrinsics = CameraIntrinsics(
        width_px=640,
        height_px=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
    )
    return (
        TeacherObservation(
            observation_id="obs-1",
            video_id="video-1",
            timestamp=1.0,
            frame_ref="frame-1.jpg",
            local_frame_id="room-1",
            pose_guidance=PoseGuidance(
                source="vio",
                reference_frame_id="room-1",
                translation_m=(0.0, 0.0, 1.5),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                covariance_6x6=covariance,
            ),
        ),
        TeacherObservation(
            observation_id="obs-2",
            video_id="video-1",
            timestamp=2.0,
            frame_ref="frame-2.jpg",
            local_frame_id="room-1",
            depth_guidance=DepthGuidance(
                depth_ref="depth-2.npy",
                depth_scale_m=0.001,
                intrinsics=intrinsics,
            ),
        ),
    )


def test_mock_provider_cache_roundtrip_is_prefix_causal(tmp_path: Path) -> None:
    provider = _MockProvider()
    rows = GCut3RTeacherAdapter(provider).run(_observations())
    cache = tmp_path / "teacher-cache.jsonl"

    write_teacher_cache(cache, rows)
    loaded = read_teacher_cache(cache)

    assert loaded == rows
    assert provider.requests[0].previous_state_ref is None
    assert provider.requests[1].previous_state_ref == "state:0"
    assert provider.requests[1].prefix_before_sha256 == rows[0].prefix_sha256
    assert (
        decode_teacher_request(encode_teacher_request(rows[0].request))
        == rows[0].request
    )
    assert (
        decode_teacher_response(encode_teacher_response(rows[0].response))
        == rows[0].response
    )


def test_cache_digest_tampering_is_rejected() -> None:
    rows = GCut3RTeacherAdapter(_MockProvider()).run(_observations())
    tampered = rows[1].model_copy(update={"prefix_sha256": "0" * 64})

    with pytest.raises(TeacherContractError, match="digest mismatch"):
        validate_teacher_cache((rows[0], tampered))


def test_cache_supports_independent_video_prefixes() -> None:
    first = GCut3RTeacherAdapter(_MockProvider()).run(_observations())
    second_observations = tuple(
        observation.model_copy(
            update={
                "video_id": "video-2",
                "observation_id": f"video-2:{observation.observation_id}",
            },
        )
        for observation in _observations()
    )
    second = GCut3RTeacherAdapter(_MockProvider()).run(second_observations)

    validate_teacher_cache((*first, *second))

    assert first[0].request.sequence_index == second[0].request.sequence_index == 0
    assert first[0].request.prefix_before_sha256 == EMPTY_PREFIX_SHA256
    assert second[0].request.prefix_before_sha256 == EMPTY_PREFIX_SHA256


def test_future_teacher_record_is_rejected() -> None:
    request = TeacherRequest(
        observation_id="obs-1",
        video_id="video-1",
        timestamp=1.0,
        frame_ref="frame-1.jpg",
        local_frame_id="room-1",
        pose_guidance=_observations()[0].pose_guidance,
        depth_guidance=None,
        sequence_index=0,
        prefix_before_sha256=EMPTY_PREFIX_SHA256,
    )
    future = _no_write(request).model_copy(
        update={
            "validity": ValidityInterval(start_time=1.0, end_time=2.0),
            "last_seen_time": 2.0,
        },
    )

    with pytest.raises(ValidationError, match="validity must not exceed"):
        _ = TeacherResponse(
            observation_id=request.observation_id,
            video_id=request.video_id,
            timestamp=1.0,
            observed_through_time=1.0,
            state_ref="state:0",
            records=(future,),
        )


def test_provider_configuration_and_cut3r_fallback_are_explicit(
    tmp_path: Path,
) -> None:
    with pytest.raises(TeacherConfigurationError, match="never downloads"):
        _ = GCut3RTeacherAdapter(None)

    code = tmp_path / "gcut3r"
    code.mkdir()
    checkpoint = tmp_path / "gcut3r.ckpt"
    checkpoint.touch()
    paths = resolve_gcut3r_paths(
        {
            "WORLDMM_GCUT3R_CODE_PATH": str(code),
            "WORLDMM_GCUT3R_CHECKPOINT_PATH": str(checkpoint),
        },
    )
    assert paths.code_path == code
    assert paths.checkpoint_path == checkpoint

    cut3r_code = tmp_path / "CUT3R"
    cut3r_code.mkdir()
    (cut3r_code / "demo.py").touch()
    cut3r_model = tmp_path / "cut3r-model"
    cut3r_model.mkdir()
    cut3r_paths = resolve_cut3r_paths(
        {
            "WORLDMM_CUT3R_CODE_PATH": str(cut3r_code),
            "WORLDMM_CUT3R_MODEL_PATH": str(cut3r_model),
        },
    )
    command = build_cut3r_demo_command(
        cut3r_paths,
        sequence_path=tmp_path / "sequence",
        output_dir=tmp_path / "output",
        python_executable="python",
    )
    assert command == (
        "python",
        str(cut3r_code / "demo.py"),
        "--model_path",
        str(cut3r_model),
        "--seq_path",
        str(tmp_path / "sequence"),
        "--output_dir",
        str(tmp_path / "output"),
    )

    row = GCut3RTeacherAdapter(_MockProvider()).run(_observations()[:1])[0]
    unguided_request = row.request.model_copy(
        update={"pose_guidance": None, "depth_guidance": None},
    )
    fallback = build_teacher_cache_record(
        teacher_backend="cut3r_cache_fallback",
        provider_id="official-cut3r-cache-test",
        request=unguided_request,
        response=row.response,
    )
    validate_teacher_cache((fallback,))
    assert fallback.teacher_backend == "cut3r_cache_fallback"
    assert fallback.request.pose_guidance is None


def test_validate_cache_cli_reports_backend(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache = tmp_path / "cache.jsonl"
    rows = GCut3RTeacherAdapter(_MockProvider()).run(_observations())
    write_teacher_cache(cache, rows)

    assert main(("validate-cache", "--cache", str(cache))) == 0
    payload = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    assert payload == {
        "cache": str(cache),
        "provider_id": "mock-gcut3r-v1",
        "record_count": 2,
        "teacher_backend": "gcut3r_external",
        "valid": True,
    }


def test_build_cut3r_command_is_pure() -> None:
    paths = Cut3RPaths(code_path=Path("/opt/CUT3R"), model_path=Path("/models/cut3r"))

    command = build_cut3r_demo_command(
        paths,
        sequence_path=Path("/data/sequence"),
        output_dir=Path("/output"),
    )

    assert command[1:] == (
        "/opt/CUT3R/demo.py",
        "--model_path",
        "/models/cut3r",
        "--seq_path",
        "/data/sequence",
        "--output_dir",
        "/output",
    )


def test_cache_record_schema_rejects_unknown_backend() -> None:
    rows = GCut3RTeacherAdapter(_MockProvider()).run(_observations())
    payload = rows[0].model_dump(mode="json")
    payload["teacher_backend"] = "gcut3r_official"

    with pytest.raises(ValidationError):
        _ = TeacherCacheRecord.model_validate(payload)
