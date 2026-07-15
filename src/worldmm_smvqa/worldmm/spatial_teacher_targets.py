from __future__ import annotations

import hashlib
import struct
from math import isfinite, sqrt
from typing import Annotated, Self, cast

from pydantic import Field, FiniteFloat, model_validator

from worldmm_smvqa.schema import FrozenModel
from worldmm_smvqa.teacher_oracle import (
    FrameInventoryV1,
    SelectedPointPayloadRefV1,
    TeacherOracleContractError,
)
from worldmm_smvqa.worldmm.typed_memory import (
    ObjectGeometry,
    ObjectMemoryRecord,
    SpatialUncertainty,
    ValidityInterval,
)

type Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
type NonEmptyStr = Annotated[str, Field(min_length=1)]

_SINGLE_FRAME_TARGET_CUTOFF_ERROR = (
    "single-frame teacher target observed_through_time must equal timestamp"
)
_SINGLE_FRAME_GEOMETRY_CUTOFF_ERROR = (
    "single-frame geometry target observed_through_time must equal timestamp"
)
_MIN_EXTENT_ERROR = "min_extent_m must be positive and finite"
_UNCERTAINTY_FLOOR_ERROR = "uncertainty_floor_m must be non-negative and finite"


class TeacherObjectTarget(FrozenModel):
    """Evidence-bound object points emitted by an offline geometry teacher."""

    memory_id: NonEmptyStr
    observation_id: NonEmptyStr
    source_video_id: NonEmptyStr
    frame_ref: NonEmptyStr
    local_frame_id: NonEmptyStr
    timestamp: FiniteFloat = Field(ge=0.0)
    observed_through_time: FiniteFloat = Field(ge=0.0)
    entity_id: NonEmptyStr
    instance_id: NonEmptyStr
    semantic_label: NonEmptyStr
    place_label: NonEmptyStr | None = None
    points_m: tuple[Vec3, ...] = Field(min_length=1)
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _require_single_frame_cutoff(self) -> Self:
        if self.observed_through_time != self.timestamp:
            msg = _SINGLE_FRAME_TARGET_CUTOFF_ERROR
            raise ValueError(msg)
        return self


class TeacherGeometryTargetV1(FrozenModel):
    """T0 geometry-only target; semantics and T1 place grounding stay external."""

    memory_id: NonEmptyStr
    observation_id: NonEmptyStr
    source_video_id: NonEmptyStr
    frame_ref: NonEmptyStr
    local_frame_id: NonEmptyStr
    timestamp: FiniteFloat = Field(ge=0.0)
    observed_through_time: FiniteFloat = Field(ge=0.0)
    entity_id: NonEmptyStr
    instance_id: NonEmptyStr
    points_m: tuple[Vec3, ...] = Field(min_length=1)
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _require_single_frame_cutoff(self) -> Self:
        if self.observed_through_time != self.timestamp:
            raise ValueError(_SINGLE_FRAME_GEOMETRY_CUTOFF_ERROR)
        return self


class OracleGeometryBinding(FrozenModel):
    """Verified selected-point bytes, not caller-supplied geometry lineage."""

    assignment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_points: SelectedPointPayloadRefV1
    selected_points_bytes: bytes = Field(min_length=12)


def compile_geometry_target_record(
    target: TeacherGeometryTargetV1,
    *,
    semantic_class: str,
    min_extent_m: float = 0.01,
    uncertainty_floor_m: float = 0.02,
) -> ObjectMemoryRecord:
    """Compile T0 geometry with a separate semantic assignment and no place."""
    return compile_teacher_object_record(
        TeacherObjectTarget(
            memory_id=target.memory_id,
            observation_id=target.observation_id,
            source_video_id=target.source_video_id,
            frame_ref=target.frame_ref,
            local_frame_id=target.local_frame_id,
            timestamp=target.timestamp,
            observed_through_time=target.observed_through_time,
            entity_id=target.entity_id,
            instance_id=target.instance_id,
            semantic_label=semantic_class,
            points_m=target.points_m,
            confidence=target.confidence,
        ),
        min_extent_m=min_extent_m,
        uncertainty_floor_m=uncertainty_floor_m,
    )


def compile_oracle_geometry_target_record(
    target: TeacherGeometryTargetV1,
    *,
    semantic_class: str,
    frame_inventory: FrameInventoryV1,
    binding: OracleGeometryBinding,
) -> ObjectMemoryRecord:
    _validate_target_frame(
        TeacherObjectTarget(
            memory_id=target.memory_id,
            observation_id=target.observation_id,
            source_video_id=target.source_video_id,
            frame_ref=target.frame_ref,
            local_frame_id=target.local_frame_id,
            timestamp=target.timestamp,
            observed_through_time=target.observed_through_time,
            entity_id=target.entity_id,
            instance_id=target.instance_id,
            semantic_label=semantic_class,
            points_m=target.points_m,
            confidence=target.confidence,
        ),
        frame_inventory,
    )
    _verify_target_points(target, binding)
    record = compile_geometry_target_record(
        target,
        semantic_class=semantic_class,
    )
    return _bind_oracle_geometry(record, binding)


def compile_teacher_object_record(
    target: TeacherObjectTarget,
    *,
    min_extent_m: float = 0.01,
    uncertainty_floor_m: float = 0.02,
) -> ObjectMemoryRecord:
    """Compile selected teacher points; semantic extraction stays external."""
    if not isfinite(min_extent_m) or min_extent_m <= 0.0:
        msg = _MIN_EXTENT_ERROR
        raise ValueError(msg)
    if not isfinite(uncertainty_floor_m) or uncertainty_floor_m < 0.0:
        msg = _UNCERTAINTY_FLOOR_ERROR
        raise ValueError(msg)

    point_count = len(target.points_m)
    centroid = cast(
        "Vec3",
        tuple(
            sum(float(point[axis]) for point in target.points_m) / point_count
            for axis in range(3)
        ),
    )
    extent = cast(
        "tuple[float, float, float]",
        tuple(
            max(
                max(float(point[axis]) for point in target.points_m)
                - min(float(point[axis]) for point in target.points_m),
                min_extent_m,
            )
            for axis in range(3)
        ),
    )
    floor_variance = uncertainty_floor_m**2
    covariance = cast(
        "tuple[Vec3, Vec3, Vec3]",
        tuple(
            tuple(
                sum(
                    (float(point[row]) - centroid[row])
                    * (float(point[column]) - centroid[column])
                    for point in target.points_m
                )
                / point_count
                + (floor_variance if row == column else 0.0)
                for column in range(3)
            )
            for row in range(3)
        ),
    )
    standard_deviation_m = sqrt(max(covariance[index][index] for index in range(3)))

    return ObjectMemoryRecord(
        memory_id=target.memory_id,
        source_video_id=target.source_video_id,
        entity_id=target.entity_id,
        instance_id=target.instance_id,
        local_frame_id=target.local_frame_id,
        geometry_uncertainty=SpatialUncertainty(
            covariance_xyz=covariance,
            standard_deviation_m=standard_deviation_m,
        ),
        validity=ValidityInterval(
            start_time=target.timestamp,
            end_time=target.timestamp,
        ),
        first_seen_time=target.timestamp,
        last_seen_time=target.timestamp,
        observation_count=1,
        confidence=target.confidence,
        provenance="model_inferred",
        evidence_refs=(target.frame_ref,),
        geometry=ObjectGeometry(centroid=centroid, extent=extent),
        semantic_label=target.semantic_label,
        place_label=target.place_label,
        oracle_observation_id=target.observation_id,
    )


def compile_oracle_teacher_object_record(
    target: TeacherObjectTarget,
    *,
    frame_inventory: FrameInventoryV1,
    binding: OracleGeometryBinding,
    min_extent_m: float = 0.01,
    uncertainty_floor_m: float = 0.02,
) -> ObjectMemoryRecord:
    """Compile strict oracle targets after binding evidence to approved frames."""
    _validate_target_frame(target, frame_inventory)
    _verify_target_points(target, binding)
    record = compile_teacher_object_record(
        target,
        min_extent_m=min_extent_m,
        uncertainty_floor_m=uncertainty_floor_m,
    )
    return _bind_oracle_geometry(record, binding)


def _bind_oracle_geometry(
    record: ObjectMemoryRecord,
    binding: OracleGeometryBinding,
) -> ObjectMemoryRecord:
    payload = record.model_dump(mode="json")
    payload["oracle_assignment_sha256"] = binding.assignment_sha256
    payload["selected_payload_sha256"] = binding.selected_points.sha256
    return ObjectMemoryRecord.model_validate(payload)


def validate_oracle_record_geometry(
    record: ObjectMemoryRecord,
    *,
    selected_points: SelectedPointPayloadRefV1,
    selected_points_bytes: bytes,
) -> None:
    """Re-derive every geometry and temporal field from sealed selected-point bytes."""
    target = TeacherObjectTarget(
        memory_id=record.memory_id,
        observation_id=record.oracle_observation_id or "",
        source_video_id=record.source_video_id,
        frame_ref=selected_points.frame_ref,
        local_frame_id=record.local_frame_id,
        timestamp=record.first_seen_time,
        observed_through_time=record.last_seen_time,
        entity_id=record.entity_id,
        instance_id=record.instance_id,
        semantic_label=record.semantic_label or "",
        place_label=record.place_label,
        points_m=_decode_selected_points(selected_points, selected_points_bytes),
        confidence=record.confidence,
    )
    expected = compile_teacher_object_record(target)
    fields = (
        "geometry_uncertainty",
        "validity",
        "first_seen_time",
        "last_seen_time",
        "observation_count",
        "evidence_refs",
        "geometry",
    )
    if any(getattr(record, field) != getattr(expected, field) for field in fields):
        raise TeacherOracleContractError(
            detail=(
                "object geometry/time/evidence does not derive from "
                "sealed selected-point bytes"
            )
        )


def _verify_target_points(
    target: TeacherGeometryTargetV1 | TeacherObjectTarget,
    binding: OracleGeometryBinding,
) -> None:
    if binding.selected_points.frame_ref != target.frame_ref:
        raise TeacherOracleContractError(
            detail="selected-point payload frame_ref does not match target frame"
        )
    decoded = _decode_selected_points(
        binding.selected_points, binding.selected_points_bytes
    )
    if tuple(target.points_m) != decoded:
        raise TeacherOracleContractError(
            detail=(
                "target geometry does not exactly match "
                "sealed selected-point payload bytes"
            )
        )


def _decode_selected_points(
    descriptor: SelectedPointPayloadRefV1, payload: bytes
) -> tuple[Vec3, ...]:
    if hashlib.sha256(payload).hexdigest() != descriptor.sha256:
        raise TeacherOracleContractError(
            detail="selected-point payload bytes do not match sealed payload digest"
        )
    if len(payload) != descriptor.point_count * 12:
        raise TeacherOracleContractError(
            detail="selected-point payload bytes do not match declared shape"
        )
    values = struct.unpack(f"<{descriptor.point_count * 3}f", payload)
    points = cast(
        "tuple[Vec3, ...]",
        tuple(
            (values[index], values[index + 1], values[index + 2])
            for index in range(0, len(values), 3)
        ),
    )
    if any(
        not isfinite(value)
        or value < descriptor.coordinate_min_xyz[axis]
        or value > descriptor.coordinate_max_xyz[axis]
        or (value == 0.0 and struct.pack("<f", value) != b"\x00\x00\x00\x00")
        for point in points
        for axis, value in enumerate(point)
    ):
        raise TeacherOracleContractError(
            detail=(
                "selected-point payload contains non-canonical or "
                "coordinate-contract-out-of-bounds coordinates"
            )
        )
    minima = tuple(min(point[axis] for point in points) for axis in range(3))
    maxima = tuple(max(point[axis] for point in points) for axis in range(3))
    if minima != descriptor.bounds_min_m or maxima != descriptor.bounds_max_m:
        raise TeacherOracleContractError(
            detail=(
                "selected-point declared bounds must equal exact component-wise extrema"
            )
        )
    return points


def _validate_target_frame(
    target: TeacherObjectTarget,
    frame_inventory: FrameInventoryV1,
) -> None:
    frame = next(
        (item for item in frame_inventory.frames if item.frame_ref == target.frame_ref),
        None,
    )
    if frame is None:
        raise TeacherOracleContractError(
            detail=(
                "target frame_ref is not in approved frame inventory: "
                f"{target.frame_ref}"
            ),
        )
    if (
        frame.observation_id != target.observation_id
        or frame.source_video_id != target.source_video_id
        or frame.local_frame_id != target.local_frame_id
        or frame.timestamp != target.timestamp
    ):
        raise TeacherOracleContractError(
            detail=(
                "target frame_ref does not match "
                "object/observation/video/local-frame/timestamp join"
            ),
        )
