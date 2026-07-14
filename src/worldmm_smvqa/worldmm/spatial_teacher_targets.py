from __future__ import annotations

from math import isfinite, sqrt
from typing import Annotated, Self, cast

from pydantic import Field, FiniteFloat, model_validator

from worldmm_smvqa.schema import FrozenModel
from worldmm_smvqa.worldmm.typed_memory import (
    ObjectGeometry,
    ObjectMemoryRecord,
    SpatialUncertainty,
    ValidityInterval,
)

type Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
type NonEmptyStr = Annotated[str, Field(min_length=1)]


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
            msg = (
                "single-frame teacher target observed_through_time must equal timestamp"
            )
            raise ValueError(msg)
        return self


def compile_teacher_object_record(
    target: TeacherObjectTarget,
    *,
    min_extent_m: float = 0.01,
    uncertainty_floor_m: float = 0.02,
) -> ObjectMemoryRecord:
    """Compile selected teacher points; semantic extraction stays external."""
    if not isfinite(min_extent_m) or min_extent_m <= 0.0:
        msg = "min_extent_m must be positive and finite"
        raise ValueError(msg)
    if not isfinite(uncertainty_floor_m) or uncertainty_floor_m < 0.0:
        msg = "uncertainty_floor_m must be non-negative and finite"
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
    )
