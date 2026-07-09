# ruff: noqa: E501

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol, override, runtime_checkable

from worldmm_smvqa.schema import FrozenModel

if TYPE_CHECKING:
    from worldmm_smvqa.worldmm.spatial_types import SpatialAnchorRecord

type GeometryPrimitiveSource = Literal["slam_pose", "gaze"]

_FRAME_TIME_RE: Final[re.Pattern[str]] = re.compile(r"(\d+)(?!.*\d)")


@dataclass(frozen=True, slots=True)
class EmptyGeometryPrimitivesError(Exception):
    anchor_memory_id: str

    @override
    def __str__(self) -> str:
        return f"EmptyGeometryPrimitivesError: {self.anchor_memory_id}"


class GeometryPrimitive(FrozenModel):
    frame_ref: str
    x: float
    y: float
    z: float
    source: GeometryPrimitiveSource


class BoundAnchor(FrozenModel):
    anchor_memory_id: str
    embedding_ref: str | None
    primitive: GeometryPrimitive


@runtime_checkable
class SemanticGeometryBinder(Protocol):
    def bind(
        self,
        anchor: SpatialAnchorRecord,
        primitives: Sequence[GeometryPrimitive],
    ) -> BoundAnchor:
        """Bind one spatial anchor to one geometry primitive."""
        ...


# ponytail: interface only - real CLIP/SigLIP/DINO x SLAM/CUT3R binding is the v3 upgrade path, gated on remote GPU work.
class NoopBinder:
    def bind(
        self,
        anchor: SpatialAnchorRecord,
        primitives: Sequence[GeometryPrimitive],
    ) -> BoundAnchor:
        """Bind with no semantic embedding backend."""
        if not primitives:
            raise EmptyGeometryPrimitivesError(anchor_memory_id=anchor.memory_id)
        primitive = sorted(
            primitives,
            key=lambda item: (
                abs(_frame_time(item.frame_ref) - _anchor_midpoint(anchor)),
                item.frame_ref,
                item.source,
                item.x,
                item.y,
                item.z,
            ),
        )[0]
        return BoundAnchor(
            anchor_memory_id=anchor.memory_id,
            embedding_ref=None,
            primitive=primitive,
        )


def _anchor_midpoint(anchor: SpatialAnchorRecord) -> float:
    return (anchor.start_time + anchor.end_time) / 2.0


def _frame_time(frame_ref: str) -> float:
    match = _FRAME_TIME_RE.search(frame_ref)
    if match is None:
        return math.inf
    return float(match.group(1))
