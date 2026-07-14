from __future__ import annotations

import pytest
from pydantic import ValidationError

from worldmm_smvqa.schema import PoseSample
from worldmm_smvqa.worldmm.spatial_sensor import (
    CameraIntrinsics,
    CausalSensorObservation,
    DepthObservation,
    GazeRay,
    is_trusted_causal_pose,
)


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width_px=640,
        height_px=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
    )


def _pose(**updates: object) -> PoseSample:
    payload: dict[str, object] = {
        "timestamp": 1.0,
        "x": 0.0,
        "y": 0.0,
        "z": 1.5,
        "yaw_degrees": 0.0,
        "source": "vio",
        "processing_mode": "online_causal",
        "observed_through_time": 1.0,
        "coordinate_frame": "room-1",
        "pose_covariance_xyz_m_rpy_deg": (0.0,) * 36,
    }
    payload.update(updates)
    return PoseSample.model_validate(payload)


def test_causal_sensor_observation_keeps_calibration_without_depth() -> None:
    observation = CausalSensorObservation(
        observation_id="obs-1",
        video_id="video-1",
        timestamp=1.0,
        frame_ref="frame-1.jpg",
        local_frame_id="room-1",
        intrinsics=_intrinsics(),
        pose=_pose(),
        gaze=GazeRay(origin_m=(0.0, 0.0, 1.5), direction=(0.0, 1.0, 0.0)),
    )

    assert observation.intrinsics.fx == 500.0
    assert observation.depth is None
    assert is_trusted_causal_pose(
        _pose(),
        cutoff_time=1.0,
        coordinate_frame="room-1",
    )


def test_sensor_observation_accepts_optional_metric_depth() -> None:
    observation = CausalSensorObservation(
        observation_id="obs-1",
        video_id="video-1",
        timestamp=1.0,
        frame_ref="frame-1.jpg",
        local_frame_id="room-1",
        intrinsics=_intrinsics(),
        depth=DepthObservation(depth_ref="depth-1.npy", depth_scale_m=0.001),
    )

    assert observation.depth is not None
    assert observation.depth.depth_scale_m == pytest.approx(0.001)


@pytest.mark.parametrize(
    "pose",
    [
        _pose(source="slam", processing_mode="offline"),
        _pose(observed_through_time=2.0),
        _pose(coordinate_frame="other-room"),
    ],
)
def test_sensor_observation_rejects_untrusted_pose(pose: PoseSample) -> None:
    with pytest.raises(ValidationError, match="raw IMU or online-causal VIO"):
        _ = CausalSensorObservation(
            observation_id="obs-1",
            video_id="video-1",
            timestamp=1.0,
            frame_ref="frame-1.jpg",
            local_frame_id="room-1",
            intrinsics=_intrinsics(),
            pose=pose,
        )


@pytest.mark.parametrize("direction", [(0.0, 0.0, 0.0), (0.0, 2.0, 0.0)])
def test_gaze_ray_requires_unit_direction(
    direction: tuple[float, float, float],
) -> None:
    with pytest.raises(ValidationError, match="gaze direction must be a unit vector"):
        _ = GazeRay(origin_m=(0.0, 0.0, 0.0), direction=direction)
