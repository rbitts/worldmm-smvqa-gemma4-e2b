from __future__ import annotations

import hashlib
import json
import os
import sys
from argparse import ArgumentParser
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Self, cast, override, runtime_checkable

from pydantic import Field, FiniteFloat, model_validator

from worldmm_smvqa.schema import FrozenModel
from worldmm_smvqa.worldmm.typed_memory import TypedMemoryRecord  # noqa: TC001

type TeacherBackend = Literal["gcut3r_external", "cut3r_cache_fallback"]
type Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
type Quaternion = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]

EMPTY_PREFIX_SHA256 = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True, slots=True)
class TeacherConfigurationError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"TeacherConfigurationError: {self.detail}"


@dataclass(frozen=True, slots=True)
class TeacherContractError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"TeacherContractError: {self.detail}"


class PoseGuidance(FrozenModel):
    source: Literal["imu", "vio", "slam", "ground_truth"]
    reference_frame_id: str = Field(min_length=1)
    translation_m: Vec3
    orientation_xyzw: Quaternion
    covariance_6x6: tuple[FiniteFloat, ...] = Field(min_length=36, max_length=36)

    @model_validator(mode="after")
    def _require_valid_pose(self) -> Self:
        if self.orientation_xyzw == (0.0, 0.0, 0.0, 0.0):
            msg = "orientation_xyzw must be non-zero"
            raise ValueError(msg)
        if any(self.covariance_6x6[index] < 0.0 for index in (0, 7, 14, 21, 28, 35)):
            msg = "pose covariance diagonal must be non-negative"
            raise ValueError(msg)
        return self


class CameraIntrinsics(FrozenModel):
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    fx: FiniteFloat = Field(gt=0.0)
    fy: FiniteFloat = Field(gt=0.0)
    cx: FiniteFloat
    cy: FiniteFloat


class DepthGuidance(FrozenModel):
    depth_ref: str = Field(min_length=1)
    depth_scale_m: FiniteFloat = Field(gt=0.0)
    intrinsics: CameraIntrinsics


class TeacherObservation(FrozenModel):
    observation_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    timestamp: FiniteFloat = Field(ge=0.0)
    frame_ref: str = Field(min_length=1)
    local_frame_id: str = Field(min_length=1)
    pose_guidance: PoseGuidance | None = None
    depth_guidance: DepthGuidance | None = None


class TeacherRequest(TeacherObservation):
    sequence_index: int = Field(ge=0)
    previous_state_ref: str | None = None
    prefix_before_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TeacherResponse(FrozenModel):
    observation_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    timestamp: FiniteFloat = Field(ge=0.0)
    observed_through_time: FiniteFloat = Field(ge=0.0)
    state_ref: str = Field(min_length=1)
    records: tuple[TypedMemoryRecord, ...] = ()
    pointmap_ref: str | None = None
    confidence_ref: str | None = None

    @model_validator(mode="after")
    def _require_causal_response(self) -> Self:
        if self.observed_through_time > self.timestamp:
            msg = "observed_through_time must not exceed response timestamp"
            raise ValueError(msg)
        if any(
            record.validity.end_time > self.observed_through_time
            for record in self.records
        ):
            msg = "typed record validity must not exceed observed_through_time"
            raise ValueError(msg)
        return self


class TeacherCacheRecord(FrozenModel):
    cache_version: Literal["teacher-cache-v1"] = "teacher-cache-v1"
    teacher_backend: TeacherBackend
    provider_id: str = Field(min_length=1)
    request: TeacherRequest
    response: TeacherResponse
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prefix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GCut3RPaths(FrozenModel):
    code_path: Path
    checkpoint_path: Path


class Cut3RPaths(FrozenModel):
    code_path: Path
    model_path: Path


@dataclass(frozen=True, slots=True)
class ProviderStep:
    response: TeacherResponse
    state: object | None


@runtime_checkable
class GCut3RProvider(Protocol):
    provider_id: str

    def infer(
        self,
        request: TeacherRequest,
        previous_state: object | None,
    ) -> ProviderStep:
        """Infer one observation using only the supplied causal prefix state."""
        ...


class GCut3RTeacherAdapter:
    """Causal adapter for an external G-CUT3R-compatible provider.

    No G-CUT3R implementation is bundled or imported here. The provider receives
    one observation and the immediately preceding opaque state only.
    """

    def __init__(self, provider: object | None) -> None:
        """Bind an already-installed external provider without importing it."""
        if provider is None:
            raise TeacherConfigurationError(
                detail=(
                    "no G-CUT3R provider supplied; install an external provider, "
                    "set WORLDMM_GCUT3R_CODE_PATH and "
                    "WORLDMM_GCUT3R_CHECKPOINT_PATH, then pass a "
                    "GCut3RProvider implementation (this package never downloads "
                    "or imports G-CUT3R automatically)"
                ),
            )
        if not isinstance(provider, GCut3RProvider):
            raise TeacherConfigurationError(
                detail="provider does not implement the GCut3RProvider contract",
            )
        if not provider.provider_id:
            raise TeacherConfigurationError(detail="provider_id must not be empty")
        self._provider: GCut3RProvider = provider

    def run(
        self,
        observations: Iterable[TeacherObservation],
    ) -> tuple[TeacherCacheRecord, ...]:
        """Run one ordered video prefix and return validated cache rows."""
        rows: list[TeacherCacheRecord] = []
        previous_prefix = EMPTY_PREFIX_SHA256
        previous_state_ref: str | None = None
        previous_state: object | None = None
        video_id: str | None = None
        previous_timestamp: float | None = None
        observation_ids: set[str] = set()

        for sequence_index, observation in enumerate(observations):
            if video_id is None:
                video_id = observation.video_id
            elif observation.video_id != video_id:
                raise TeacherContractError(
                    detail="one adapter run must contain exactly one video_id",
                )
            if observation.observation_id in observation_ids:
                raise TeacherContractError(
                    detail=f"duplicate observation_id: {observation.observation_id}",
                )
            if (
                previous_timestamp is not None
                and observation.timestamp <= previous_timestamp
            ):
                raise TeacherContractError(
                    detail="observation timestamps must be strictly increasing",
                )

            request = TeacherRequest(
                observation_id=observation.observation_id,
                video_id=observation.video_id,
                timestamp=observation.timestamp,
                frame_ref=observation.frame_ref,
                local_frame_id=observation.local_frame_id,
                pose_guidance=observation.pose_guidance,
                depth_guidance=observation.depth_guidance,
                sequence_index=sequence_index,
                previous_state_ref=previous_state_ref,
                prefix_before_sha256=previous_prefix,
            )
            step = self._provider.infer(request, previous_state)
            _validate_response(request, step.response)
            row = build_teacher_cache_record(
                teacher_backend="gcut3r_external",
                provider_id=self._provider.provider_id,
                request=request,
                response=step.response,
            )
            rows.append(row)
            previous_prefix = row.prefix_sha256
            previous_state_ref = step.response.state_ref
            previous_state = step.state
            previous_timestamp = observation.timestamp
            observation_ids.add(observation.observation_id)

        validate_teacher_cache(rows)
        return tuple(rows)


def resolve_gcut3r_paths(env: Mapping[str, str]) -> GCut3RPaths:
    return GCut3RPaths(
        code_path=_required_path(env, "WORLDMM_GCUT3R_CODE_PATH", directory=True),
        checkpoint_path=_required_path(
            env,
            "WORLDMM_GCUT3R_CHECKPOINT_PATH",
            directory=False,
        ),
    )


def resolve_cut3r_paths(env: Mapping[str, str]) -> Cut3RPaths:
    code_path = _required_path(env, "WORLDMM_CUT3R_CODE_PATH", directory=True)
    demo_path = code_path / "demo.py"
    if not demo_path.is_file():
        raise TeacherConfigurationError(
            detail=(
                f"WORLDMM_CUT3R_CODE_PATH must contain demo.py: {demo_path}; "
                "expected the official https://github.com/CUT3R/CUT3R checkout"
            ),
        )
    return Cut3RPaths(
        code_path=code_path,
        model_path=_required_path(env, "WORLDMM_CUT3R_MODEL_PATH", directory=False),
    )


def build_cut3r_demo_command(
    paths: Cut3RPaths,
    *,
    sequence_path: Path,
    output_dir: Path,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Build, but never execute, the official CUT3R demo command."""
    return (
        python_executable,
        str(paths.code_path / "demo.py"),
        "--model_path",
        str(paths.model_path),
        "--seq_path",
        str(sequence_path),
        "--output_dir",
        str(output_dir),
    )


def encode_teacher_request(request: TeacherRequest) -> str:
    """Encode one request line for an external G-CUT3R JSONL provider."""
    return _canonical_json(request.model_dump(mode="json"))


def decode_teacher_request(line: str) -> TeacherRequest:
    return TeacherRequest.model_validate_json(line)


def encode_teacher_response(response: TeacherResponse) -> str:
    """Encode one response line for an external G-CUT3R JSONL provider."""
    return _canonical_json(response.model_dump(mode="json"))


def decode_teacher_response(line: str) -> TeacherResponse:
    return TeacherResponse.model_validate_json(line)


def write_teacher_cache(
    path: Path,
    records: Sequence[TeacherCacheRecord],
) -> None:
    validate_teacher_cache(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _ = temporary.write_text(
            "".join(
                f"{_canonical_json(row.model_dump(mode='json'))}\n" for row in records
            ),
            encoding="utf-8",
        )
        _ = temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_teacher_cache(path: Path) -> tuple[TeacherCacheRecord, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TeacherContractError(
            detail=f"cannot read teacher cache {path}: {exc}",
        ) from exc
    try:
        records = tuple(
            TeacherCacheRecord.model_validate_json(line)
            for line in lines
            if line.strip()
        )
    except ValueError as exc:
        raise TeacherContractError(
            detail=f"invalid teacher cache row in {path}: {exc}",
        ) from exc
    validate_teacher_cache(records)
    return records


def validate_teacher_cache(records: Sequence[TeacherCacheRecord]) -> None:
    if not records:
        raise TeacherContractError(detail="teacher cache must contain at least one row")
    identities = {(row.teacher_backend, row.provider_id) for row in records}
    if len(identities) != 1:
        raise TeacherContractError(
            detail="teacher backend and provider_id must be stable across the cache",
        )
    streams: dict[str, list[TeacherCacheRecord]] = {}
    for row in records:
        streams.setdefault(row.request.video_id, []).append(row)
    for stream in streams.values():
        _validate_teacher_stream(stream)


def _validate_teacher_stream(records: Sequence[TeacherCacheRecord]) -> None:
    previous_prefix = EMPTY_PREFIX_SHA256
    previous_state_ref: str | None = None
    previous_timestamp: float | None = None
    observation_ids: set[str] = set()

    for sequence_index, row in enumerate(records):
        request = row.request
        if request.sequence_index != sequence_index:
            raise TeacherContractError(
                detail=(
                    f"row {sequence_index}: sequence_index must be {sequence_index}, "
                    f"received {request.sequence_index}"
                ),
            )
        if request.prefix_before_sha256 != previous_prefix:
            raise TeacherContractError(
                detail=f"row {sequence_index}: broken prefix hash chain",
            )
        if request.previous_state_ref != previous_state_ref:
            raise TeacherContractError(
                detail=(
                    f"row {sequence_index}: previous_state_ref does not match prefix"
                ),
            )
        if request.observation_id in observation_ids:
            raise TeacherContractError(
                detail=f"row {sequence_index}: duplicate observation_id",
            )
        if previous_timestamp is not None and request.timestamp <= previous_timestamp:
            raise TeacherContractError(
                detail=f"row {sequence_index}: timestamps must be strictly increasing",
            )

        _validate_response(request, row.response)
        expected = build_teacher_cache_record(
            teacher_backend=row.teacher_backend,
            provider_id=row.provider_id,
            request=request,
            response=row.response,
        )
        if (
            row.request_sha256 != expected.request_sha256
            or row.response_sha256 != expected.response_sha256
            or row.prefix_sha256 != expected.prefix_sha256
        ):
            raise TeacherContractError(
                detail=f"row {sequence_index}: cache digest mismatch",
            )

        previous_prefix = row.prefix_sha256
        previous_state_ref = row.response.state_ref
        previous_timestamp = request.timestamp
        observation_ids.add(request.observation_id)


def build_teacher_cache_record(
    *,
    teacher_backend: TeacherBackend,
    provider_id: str,
    request: TeacherRequest,
    response: TeacherResponse,
) -> TeacherCacheRecord:
    """Build one digest-bound row; caller supplies the honest backend identity."""
    request_sha256 = _digest(request.model_dump(mode="json"))
    response_sha256 = _digest(response.model_dump(mode="json"))
    prefix_sha256 = _digest(
        {
            "prefix_before_sha256": request.prefix_before_sha256,
            "teacher_backend": teacher_backend,
            "provider_id": provider_id,
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
        },
    )
    return TeacherCacheRecord(
        teacher_backend=teacher_backend,
        provider_id=provider_id,
        request=request,
        response=response,
        request_sha256=request_sha256,
        response_sha256=response_sha256,
        prefix_sha256=prefix_sha256,
    )


def _validate_response(request: TeacherRequest, response: TeacherResponse) -> None:
    if (
        response.observation_id != request.observation_id
        or response.video_id != request.video_id
        or response.timestamp != request.timestamp
    ):
        raise TeacherContractError(
            detail=(
                f"{request.observation_id}: response does not match request identity"
            ),
        )
    if response.observed_through_time > request.timestamp:
        raise TeacherContractError(
            detail=f"{request.observation_id}: response observes future time",
        )
    for record in response.records:
        if record.source_video_id != request.video_id:
            raise TeacherContractError(
                detail=(
                    f"{request.observation_id}: record {record.memory_id} "
                    "references another source video"
                ),
            )
        if record.validity.end_time > response.observed_through_time:
            raise TeacherContractError(
                detail=(
                    f"{request.observation_id}: record {record.memory_id} "
                    "has future validity"
                ),
            )


def _required_path(
    env: Mapping[str, str],
    name: str,
    *,
    directory: bool,
) -> Path:
    value = env.get(name)
    if not value:
        raise TeacherConfigurationError(
            detail=f"missing {name}; configure the external teacher artifact path",
        )
    path = Path(value).expanduser()
    expected = "directory" if directory else "file or model directory"
    if not path.exists() or (directory and not path.is_dir()):
        raise TeacherConfigurationError(
            detail=f"{name} must reference an existing {expected}: {path}",
        )
    return path


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Validate causal spatial teacher caches.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser(
        "validate-cache",
        help="validate hashes, state continuity, and prefix causality",
    )
    _ = validate.add_argument("--cache", type=Path, required=True)
    arguments = parser.parse_args(argv)
    cache_path = cast("Path", arguments.cache)
    records = read_teacher_cache(cache_path)
    payload = {
        "cache": str(cache_path),
        "record_count": len(records),
        "teacher_backend": records[0].teacher_backend if records else None,
        "provider_id": records[0].provider_id if records else None,
        "valid": True,
    }
    _ = sys.stdout.write(f"{_canonical_json(payload)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
