from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from math import isclose, isfinite, sqrt
from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, model_validator

from worldmm_smvqa.schema import FrozenModel, PoseSample

type Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
type PositiveFiniteFloat = Annotated[FiniteFloat, Field(gt=0.0)]
_CANONICAL_TIMESTAMP_SCALE = Decimal(1_000_000)
_CANONICAL_TIMESTAMP_HALF_MICROSECOND = Decimal("0.5")


def canonical_timestamp_us(value: Decimal | float | str) -> int:
    """Map a finite non-negative timestamp to canonical integer microseconds."""
    try:
        timestamp = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        msg = "timestamp must be a finite decimal"
        raise ValueError(msg) from exc
    if not timestamp.is_finite() or timestamp < 0:
        msg = "timestamp must be finite and non-negative"
        raise ValueError(msg)
    microseconds = timestamp * _CANONICAL_TIMESTAMP_SCALE
    try:
        rounded = microseconds.to_integral_value(rounding=ROUND_HALF_EVEN)
    except InvalidOperation as exc:
        msg = "timestamp cannot be represented in microseconds"
        raise ValueError(msg) from exc
    if abs(microseconds - rounded) > _CANONICAL_TIMESTAMP_HALF_MICROSECOND:
        msg = "timestamp cannot be represented within 0.5 microseconds"
        raise ValueError(msg)
    return int(rounded)


class CameraIntrinsics(FrozenModel):
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    fx: PositiveFiniteFloat
    fy: PositiveFiniteFloat
    cx: FiniteFloat
    cy: FiniteFloat


class DepthObservation(FrozenModel):
    """A native metric-depth asset bound to one RGB observation."""

    depth_ref: str = Field(min_length=1)
    depth_scale_m: PositiveFiniteFloat
    depth_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    shape: tuple[int, int]
    format: Literal["npy"]
    provenance: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def _require_image_shaped_depth(self) -> Self:
        if self.shape != (self.height_px, self.width_px):
            msg = "depth shape must be (height_px, width_px)"
            raise ValueError(msg)
        return self


class GazeRay(FrozenModel):
    origin_m: Vec3
    direction: Vec3

    @model_validator(mode="after")
    def _require_nonzero_direction(self) -> Self:
        norm = sqrt(sum(float(value) ** 2 for value in self.direction))
        if not isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            msg = "gaze direction must be a unit vector"
            raise ValueError(msg)
        return self


class CausalSensorObservation(FrozenModel):
    """One on-device RGB observation plus optional calibrated native signals."""

    observation_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    timestamp: FiniteFloat = Field(ge=0.0)
    frame_ref: str = Field(min_length=1)
    local_frame_id: str = Field(min_length=1)
    intrinsics: CameraIntrinsics
    pose: PoseSample | None = None
    depth: DepthObservation | None = None
    gaze: GazeRay | None = None

    @model_validator(mode="after")
    def _require_causal_native_pose(self) -> Self:
        if self.pose is not None and not is_trusted_causal_pose(
            self.pose,
            cutoff_time=float(self.timestamp),
            coordinate_frame=self.local_frame_id,
        ):
            msg = (
                "pose must be raw IMU or online-causal VIO with yaw, covariance, "
                "matching frame, and a non-future causal certificate"
            )
            raise ValueError(msg)
        return self


def is_trusted_causal_pose(
    sample: PoseSample,
    *,
    cutoff_time: float,
    coordinate_frame: str,
) -> bool:
    """Return whether a pose may ground an on-device direction proof."""
    if not isfinite(cutoff_time):
        return False
    return (
        0.0 <= sample.timestamp <= cutoff_time
        and sample.observed_through_time is not None
        and sample.timestamp <= sample.observed_through_time <= cutoff_time
        and sample.yaw_degrees is not None
        and sample.pose_covariance_xyz_m_rpy_deg is not None
        and (
            (sample.source == "imu" and sample.processing_mode == "raw")
            or (sample.source == "vio" and sample.processing_mode == "online_causal")
        )
        and (sample.coordinate_frame or "source_world") == coordinate_frame
    )
