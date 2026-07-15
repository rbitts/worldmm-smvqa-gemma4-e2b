from __future__ import annotations

import ast
import errno
import hashlib
import json
import math
import os
import stat
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final, Literal, Self, cast, override

from PIL import Image, UnidentifiedImageError
from pydantic import Field, TypeAdapter, ValidationError, model_validator

from worldmm_smvqa.openat2 import Openat2UnsupportedError, openat2_sealed
from worldmm_smvqa.schema import FrozenModel
from worldmm_smvqa.sensor_frames import SensorFrameManifestRecord
from worldmm_smvqa.worldmm.spatial_sensor import (
    CausalSensorObservation,
    DepthObservation,
    canonical_timestamp_us,
)

SENSOR_AUDIT_VERSION: Final = "sensor-audit-v1"
SENSOR_AUDIT_WINDOW_US: Final = 30_000_000
_TIMESTAMP_JOIN_TOLERANCE_SECONDS: Final = 0.5e-6
_LEAKAGE_TERMS: Final = frozenset(
    {"answer", "choice", "evidence", "label", "qa", "question", "target"},
)
_PPM_MAGIC: Final = b"P6"
_PPM_PART_COUNT: Final = 5
_PPM_MAX_VALUE: Final = b"255"
_PPM_PAYLOAD_INDEX: Final = 4
_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
_PNG_HEADER_SIZE: Final = 8
_PNG_CHUNK_OVERHEAD: Final = 12
_PNG_LENGTH_SIZE: Final = 4
_PNG_TYPE_START: Final = 4
_PNG_TYPE_END: Final = 8
_PNG_DATA_START: Final = 8
_PNG_IHDR: Final = b"IHDR"
_PNG_IDAT: Final = b"IDAT"
_PNG_IEND: Final = b"IEND"
_PNG_IHDR_LENGTH: Final = 13
_PNG_DIMENSION_SIZE: Final = 4
_PNG_BIT_DEPTH: Final = 8
_PNG_RGB_COLOR_TYPE: Final = 2
_PNG_RGBA_COLOR_TYPE: Final = 6
_PNG_RGB_CHANNELS: Final = 3
_PNG_RGBA_CHANNELS: Final = 4
_PNG_FILTER_BYTES_PER_ROW: Final = 1
_PNG_RGB_COLOR_TYPES: Final = frozenset({_PNG_RGB_COLOR_TYPE, _PNG_RGBA_COLOR_TYPE})
_PNG_ALPHA_COLOR_TYPE: Final = _PNG_RGBA_COLOR_TYPE
_MAX_RGB_BYTES: Final = 64 * 1024 * 1024
_MAX_RGB_PIXELS: Final = 16_777_216
_MAX_DEPTH_BYTES: Final = 64 * 1024 * 1024
_MAX_DEPTH_PIXELS: Final = 16_777_216
_NPY_MAGIC: Final = b"\x93NUMPY"
_NPY_PREFIX_SIZE: Final = 8
_NPY_V1_HEADER_LENGTH_SIZE: Final = 2
_NPY_V2_HEADER_LENGTH_SIZE: Final = 4
_NPY_ALLOWED_DESCRIPTORS: Final = frozenset({"<f4", "<f8", "<u2", "<u4", "|u1"})
_NPY_ITEM_SIZES: Final = {
    "<f4": 4,
    "<f8": 8,
    "<u2": 2,
    "<u4": 4,
    "|u1": 1,
}
_JPEG_SOI: Final = b"\xff\xd8"
_JPEG_SOF_MARKERS: Final = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)
_JPEG_MARKER_PREFIX: Final = 0xFF
_JPEG_SOI_MARKER: Final = 0xD8
_JPEG_EOI_MARKER: Final = 0xD9
_JPEG_SOS_MARKER: Final = 0xDA
_JPEG_RESTART_MARKER_START: Final = 0xD0
_JPEG_RESTART_MARKER_END: Final = 0xD7
_JPEG_LENGTH_SIZE: Final = 2
_JPEG_MIN_MARKER_SEGMENT_SIZE: Final = 4
_JPEG_SOF_MIN_LENGTH: Final = 8
_JPEG_SOF_HEIGHT_START: Final = 3
_JPEG_SOF_HEIGHT_END: Final = 5
_JPEG_SOF_WIDTH_START: Final = 5
_JPEG_SOF_WIDTH_END: Final = 7
_WEBP_RIFF_END: Final = 4
_WEBP_SIGNATURE_START: Final = 8
_WEBP_SIGNATURE_END: Final = 12
_WEBP_CHUNK_TYPE_START: Final = 12
_WEBP_CHUNK_TYPE_END: Final = 16
_WEBP_CHUNK_START: Final = 20
_WEBP_VP8_MIN_SIZE: Final = 30
_WEBP_VP8_FRAME_HEADER_START: Final = 23
_WEBP_VP8_FRAME_HEADER_END: Final = 26
_WEBP_VP8_WIDTH_START: Final = 26
_WEBP_VP8_WIDTH_END: Final = 28
_WEBP_VP8_HEIGHT_START: Final = 28
_WEBP_VP8_HEIGHT_END: Final = 30
_WEBP_VP8_DIMENSION_MASK: Final = 0x3FFF
_WEBP_VP8L_MIN_SIZE: Final = 25
_WEBP_VP8L_SIGNATURE_OFFSET: Final = 20
_WEBP_VP8L_SIGNATURE: Final = 0x2F
_WEBP_VP8L_DIMENSIONS_START: Final = 21
_WEBP_VP8L_DIMENSIONS_END: Final = 25
_WEBP_VP8L_HEIGHT_SHIFT: Final = 14
_NPY_MAJOR_VERSION_OFFSET: Final = 6
_NPY_V1_MAJOR_VERSION: Final = 1
_NPY_V2_MAJOR_VERSION: Final = 2
_NPY_V3_MAJOR_VERSION: Final = 3
_NPY_MAX_HEADER_LENGTH: Final = 65_536
_NPY_DIMENSION_COUNT: Final = 2
_WEBP_RIFF: Final = b"RIFF"
_WEBP_SIGNATURE: Final = b"WEBP"
_WEBP_VP8X: Final = b"VP8X"
_WEBP_VP8: Final = b"VP8 "
_WEBP_VP8L: Final = b"VP8L"
_JSON_VALUE_ADAPTER: Final = TypeAdapter(object)
_WEBP_VP8_FRAME_HEADER: Final = b"\x9d\x01\x2a"
_JSON_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])
_JSON_LIST_ADAPTER: Final = TypeAdapter(list[object])


@dataclass(frozen=True, slots=True)
class SensorAuditContractError(ValueError):
    detail: str

    @override
    def __str__(self) -> str:
        return f"SensorAuditContractError: {self.detail}"


class SensorAuditObservation(CausalSensorObservation):
    rgb_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SensorAuditManifest(FrozenModel):
    records: tuple[SensorFrameManifestRecord, ...]

    @model_validator(mode="after")
    def _require_unique_videos(self) -> Self:
        video_ids = tuple(record.video_id for record in self.records)
        if len(video_ids) != len(set(video_ids)):
            detail = "sensor manifest has duplicate video_id values"
            raise SensorAuditContractError(detail=detail)
        return self


class SensorModalityPolicy(FrozenModel):
    min_pose_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    min_depth_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    min_gaze_percent: float = Field(default=0.0, ge=0.0, le=100.0)


class SensorAuditIssue(FrozenModel):
    code: str
    detail: str
    observation_id: str | None = None


class SensorAuditReport(FrozenModel):
    version: Literal["sensor-audit-v1"] = SENSOR_AUDIT_VERSION
    operational_state: Literal["ready", "blocked"]
    provider_gate_decision: Literal["go", "no_go", "not_measurable", "not_decidable"]
    window_us: Literal[30_000_000] = SENSOR_AUDIT_WINDOW_US
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    observations_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_root_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    counts: dict[str, int]
    coverage: dict[str, float | None]
    issues: tuple[SensorAuditIssue, ...]


def audit_sensors(
    manifest_path: Path,
    observations_path: Path,
    frame_root: Path,
    modality_policy: SensorModalityPolicy | None = None,
    *,
    depth_root: Path | None = None,
) -> SensorAuditReport:
    """Audit selected RGB inventory without manufacturing unavailable modalities."""
    issues: list[SensorAuditIssue] = []
    policy = modality_policy or SensorModalityPolicy()
    try:
        frame_root_descriptor = _open_directory_nofollow(frame_root)
    except Openat2UnsupportedError as exc:
        raise SensorAuditContractError(
            detail=f"sensor audit requires openat2 sealed-root resolution: {exc}"
        ) from exc
    frame_root_digest = _directory_identity_digest(frame_root_descriptor)
    if frame_root_descriptor is None:
        _issue(issues, "frame_root_invalid", str(frame_root))
    manifest_bytes = _read_bytes(manifest_path, "manifest", issues)
    observations_bytes = _read_bytes(observations_path, "observations", issues)
    manifest = _read_manifest(manifest_path, manifest_bytes, issues)
    observations = _read_observations(observations_bytes, issues)
    expected = _selected_inventory(manifest, issues)
    actual = _observations_by_key(observations, issues)
    _compare_inventory(expected, actual, issues)

    raw_pose_count = 0
    trusted_pose_count = 0
    raw_depth_count = 0
    verified_depth_count = 0
    raw_gaze_count = 0
    intrinsics_count = 0
    rgb_valid_count = 0
    requires_depth = policy.min_depth_percent > 0.0
    if requires_depth and depth_root is None:
        _issue(
            issues,
            "depth_root_required",
            "depth policy requires an approved depth root",
        )
    try:
        for key in sorted(set(expected) & set(actual)):
            observation = actual[key][0]
            intrinsics_count += 1
            raw_pose_count += observation.pose is not None
            trusted_pose_count += observation.pose is not None
            raw_gaze_count += observation.gaze is not None
            raw_depth_count += observation.depth is not None
            if observation.depth is not None and depth_root is not None:
                verified_depth_count += _check_depth_asset(
                    depth_root, observation, issues
                )
            if _check_rgb_asset(frame_root, frame_root_descriptor, observation, issues):
                rgb_valid_count += 1
    finally:
        _verify_directory_identity(
            frame_root, frame_root_descriptor, "frame_root", issues
        )
        if frame_root_descriptor is not None:
            os.close(frame_root_descriptor)

    selected_count = len(expected)
    counts = {
        "selected_frames": selected_count,
        "observations": len(observations),
        "joined_observations": len(set(expected) & set(actual)),
        "rgb_verified": rgb_valid_count,
        "intrinsics_available": intrinsics_count,
        "raw_pose_available": raw_pose_count,
        "trusted_pose_available": trusted_pose_count,
        "raw_depth_available": raw_depth_count,
        "verified_depth_available": verified_depth_count,
        "raw_gaze_available": raw_gaze_count,
        "depth_available": verified_depth_count,
        "gaze_available": raw_gaze_count,
    }
    for name, minimum in (
        ("trusted_pose", policy.min_pose_percent),
        ("depth", policy.min_depth_percent),
        ("gaze", policy.min_gaze_percent),
    ):
        actual = _percent(
            {
                "trusted_pose": trusted_pose_count,
                "depth": verified_depth_count,
                "gaze": raw_gaze_count,
            }[name],
            selected_count,
        )
        if actual is None or actual < minimum:
            _issue(issues, "modality_coverage_below_policy", name)
    digest = _sha256(manifest_bytes + b"\0" + observations_bytes)
    config_digest = _canonical_digest(
        {
            "version": SENSOR_AUDIT_VERSION,
            "window_us": SENSOR_AUDIT_WINDOW_US,
            "modality_policy": policy.model_dump(mode="json"),
            "depth_root_configured": depth_root is not None,
            "frame_root_digest": frame_root_digest,
        }
    )
    return SensorAuditReport(
        operational_state="blocked" if issues else "ready",
        provider_gate_decision=_provider_gate(issues, selected_count, rgb_valid_count),
        input_digest=digest,
        config_digest=config_digest,
        manifest_digest=_sha256(manifest_bytes),
        observations_digest=_sha256(observations_bytes),
        frame_root_digest=frame_root_digest,
        counts=counts,
        coverage={
            "rgb_percent": _percent(rgb_valid_count, selected_count),
            "intrinsics_percent": _percent(intrinsics_count, selected_count),
            "trusted_pose_percent": _percent(trusted_pose_count, selected_count),
            "depth_percent": _percent(verified_depth_count, selected_count),
            "gaze_percent": _percent(raw_gaze_count, selected_count),
        },
        issues=tuple(
            sorted(
                issues,
                key=lambda issue: (
                    issue.code,
                    issue.observation_id or "",
                    issue.detail,
                ),
            )
        ),
    )


def write_sensor_audit_report(
    manifest_path: Path,
    observations_path: Path,
    frame_root: Path,
    output: Path,
    *,
    depth_root: Path | None = None,
) -> SensorAuditReport:
    report = audit_sensors(
        manifest_path, observations_path, frame_root, depth_root=depth_root
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            _ = stream.write(report.model_dump_json(indent=2) + "\n")
        _ = temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report


def _read_manifest(
    path: Path,
    payload: bytes,
    issues: list[SensorAuditIssue],
) -> SensorAuditManifest:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        _issue(issues, "invalid_manifest", f"{path}: {exc}")
        return SensorAuditManifest(records=())
    records: list[SensorFrameManifestRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = _load_json(line)
            _check_leakage(raw, f"line {line_number}", issues)
            records.append(SensorFrameManifestRecord.model_validate(raw))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            _issue(
                issues,
                "invalid_manifest",
                f"{path}: line {line_number}: {exc}",
            )
            return SensorAuditManifest(records=())
    if not records:
        _issue(issues, "invalid_manifest", f"{path}: manifest has no records")
        return SensorAuditManifest(records=())
    try:
        return SensorAuditManifest(records=tuple(records))
    except (SensorAuditContractError, ValidationError) as exc:
        _issue(issues, "invalid_manifest", str(exc))
        return SensorAuditManifest(records=())


def _read_observations(
    payload: bytes,
    issues: list[SensorAuditIssue],
) -> tuple[SensorAuditObservation, ...]:
    rows: list[SensorAuditObservation] = []
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        _issue(issues, "observations_unreadable", str(exc))
        return ()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = _load_json(line)
        except (json.JSONDecodeError, ValueError) as exc:
            _issue(issues, "invalid_observation", f"line {line_number}: {exc}")
            continue
        row = _json_object(raw)
        if row is None:
            _issue(
                issues,
                "invalid_observation",
                f"line {line_number}: row must be an object",
            )
            continue
        _check_leakage(row, f"line {line_number}", issues)
        try:
            rows.append(SensorAuditObservation.model_validate(row))
        except ValidationError as exc:
            _issue(issues, "invalid_observation", f"line {line_number}: {exc}")
    if not rows:
        _issue(issues, "empty_observations", "observations has no valid records")
    return tuple(rows)


def _selected_inventory(
    manifest: SensorAuditManifest,
    issues: list[SensorAuditIssue],
) -> dict[tuple[str, str, int], float]:
    selected: dict[tuple[str, str, int], float] = {}
    timestamps: dict[tuple[str, str, int], float] = {}
    for record in manifest.records:
        for frame in record.selected_frames:
            key = _canonical_key(
                record.video_id, frame.frame_ref, frame.timestamp, issues
            )
            if key is None:
                continue
            previous_timestamp = timestamps.get(key)
            if previous_timestamp is not None and previous_timestamp != frame.timestamp:
                _issue(issues, "canonical_key_collision", repr(key))
            elif key in selected:
                _issue(issues, "duplicate_selected_frame", repr(key))
            selected[key] = frame.timestamp
            timestamps[key] = frame.timestamp
    if not selected:
        _issue(issues, "empty_selected_inventory", "manifest has no selected frames")
    return selected


def _observations_by_key(
    observations: Iterable[SensorAuditObservation],
    issues: list[SensorAuditIssue],
) -> dict[tuple[str, str, int], tuple[SensorAuditObservation, float]]:
    result: dict[tuple[str, str, int], tuple[SensorAuditObservation, float]] = {}
    timestamps: dict[tuple[str, str, int], float] = {}
    for observation in observations:
        key = _canonical_key(
            observation.video_id,
            observation.frame_ref,
            observation.timestamp,
            issues,
            observation.observation_id,
        )
        if key is None:
            continue
        previous_timestamp = timestamps.get(key)
        if (
            previous_timestamp is not None
            and previous_timestamp != observation.timestamp
        ):
            _issue(
                issues,
                "canonical_key_collision",
                repr(key),
                observation.observation_id,
            )
            continue
        if key in result:
            _issue(
                issues, "duplicate_observation", repr(key), observation.observation_id
            )
            continue
        result[key] = (observation, observation.timestamp)
        timestamps[key] = observation.timestamp
    return result


def _canonical_key(
    video_id: str,
    frame_ref: str,
    timestamp: float,
    issues: list[SensorAuditIssue],
    observation_id: str | None = None,
) -> tuple[str, str, int] | None:
    try:
        return (video_id, frame_ref, canonical_timestamp_us(timestamp))
    except ValueError as exc:
        _issue(issues, "invalid_canonical_timestamp", str(exc), observation_id)
        return None


def _compare_inventory(
    expected: dict[tuple[str, str, int], float],
    actual: dict[tuple[str, str, int], tuple[SensorAuditObservation, float]],
    issues: list[SensorAuditIssue],
) -> None:
    for key in sorted(expected.keys() - actual.keys()):
        _issue(issues, "missing_observation", repr(key))
    for key in sorted(actual.keys() - expected.keys()):
        _issue(issues, "extra_observation", repr(key), actual[key][0].observation_id)
    for key in sorted(expected.keys() & actual.keys()):
        observation, observation_timestamp = actual[key]
        if (
            abs(expected[key] - observation_timestamp)
            > _TIMESTAMP_JOIN_TOLERANCE_SECONDS
        ):
            _issue(
                issues,
                "timestamp_join_ambiguous",
                f"{key}: {expected[key]!r} != {observation_timestamp!r}",
                observation.observation_id,
            )


def _check_rgb_asset(  # noqa: PLR0911
    frame_root: Path,
    frame_root_descriptor: int | None,
    observation: SensorAuditObservation,
    issues: list[SensorAuditIssue],
) -> bool:
    path = frame_root / observation.video_id / observation.frame_ref
    if not _rgb_path_is_safe(frame_root, path, observation.observation_id, issues):
        return False
    if frame_root_descriptor is None:
        _issue(issues, "rgb_unreadable", str(path), observation.observation_id)
        return False
    payload = _read_nofollow_bytes_at(
        frame_root_descriptor,
        path.relative_to(frame_root),
        path,
        "rgb",
        issues,
        observation_id=observation.observation_id,
        limit=_MAX_RGB_BYTES,
    )
    if payload is None:
        return False
    if _sha256(payload) != observation.rgb_sha256:
        _issue(issues, "rgb_hash_mismatch", str(path), observation.observation_id)
        return False
    dimensions = _rgb_dimensions(payload)
    if dimensions is None:
        _issue(issues, "rgb_not_readable", str(path), observation.observation_id)
        return False
    if dimensions != (
        observation.intrinsics.width_px,
        observation.intrinsics.height_px,
    ):
        _issue(
            issues,
            "rgb_dimensions_mismatch",
            str(path),
            observation.observation_id,
        )
        return False
    return True


def _check_depth_asset(
    depth_root: Path,
    observation: SensorAuditObservation,
    issues: list[SensorAuditIssue],
) -> bool:
    depth = observation.depth
    if depth is None:
        return False
    path = depth_root / observation.video_id / depth.depth_ref
    descriptor = _open_directory_nofollow(depth_root)
    if descriptor is None:
        _issue(issues, "depth_unreadable", str(depth_root), observation.observation_id)
        return False
    try:
        if not _asset_path_is_safe(
            depth_root, path, "depth", observation.observation_id, issues
        ):
            return False
        payload = _read_nofollow_bytes_at(
            descriptor,
            path.relative_to(depth_root),
            path,
            "depth",
            issues,
            observation_id=observation.observation_id,
            limit=_MAX_DEPTH_BYTES,
        )
        if payload is None:
            return False
        return _depth_payload_matches(payload, path, observation, depth, issues)
    finally:
        _verify_directory_identity(
            depth_root, descriptor, "depth_root", issues, observation.observation_id
        )
        os.close(descriptor)


def _depth_payload_matches(
    payload: bytes,
    path: Path,
    observation: SensorAuditObservation,
    depth: DepthObservation,
    issues: list[SensorAuditIssue],
) -> bool:
    if _sha256(payload) != depth.depth_sha256:
        _issue(issues, "depth_hash_mismatch", str(path), observation.observation_id)
        return False
    dimensions = _depth_dimensions(payload, depth.format)
    if dimensions is None:
        _issue(issues, "depth_not_readable", str(path), observation.observation_id)
        return False
    if dimensions != (depth.width_px, depth.height_px) or dimensions != (
        observation.intrinsics.width_px,
        observation.intrinsics.height_px,
    ):
        _issue(
            issues,
            "depth_dimensions_mismatch",
            str(path),
            observation.observation_id,
        )
        return False
    return True


def _asset_path_is_safe(
    root: Path,
    path: Path,
    kind: str,
    observation_id: str,
    issues: list[SensorAuditIssue],
) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        _issue(issues, f"{kind}_path_outside_root", str(path), observation_id)
        return False
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        _issue(issues, f"{kind}_path_outside_root", str(path), observation_id)
        return False
    return True


def _rgb_path_is_safe(
    frame_root: Path,
    path: Path,
    observation_id: str,
    issues: list[SensorAuditIssue],
) -> bool:
    return _asset_path_is_safe(frame_root, path, "rgb", observation_id, issues)


def _rgb_dimensions(payload: bytes) -> tuple[int, int] | None:
    try:
        with Image.open(BytesIO(payload)) as image:
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP", "PPM"}:
                return None
            if image.mode not in {"RGB", "RGBA"}:
                return None
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > _MAX_RGB_PIXELS:
                return None
            _ = image.load()
            strict_dimensions = {
                "JPEG": _jpeg_dimensions,
                "PNG": _png_dimensions,
                "WEBP": _webp_dimensions,
                "PPM": _ppm_dimensions,
            }[image.format](payload)
            if strict_dimensions != (width, height):
                return None
    except (
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        UnidentifiedImageError,
        ValueError,
    ):
        return None
    return width, height


def _jpeg_dimensions(  # noqa: PLR0911
    payload: bytes,
) -> tuple[int, int] | None:
    position = len(_JPEG_SOI)
    dimensions: tuple[int, int] | None = None
    while position + _JPEG_MIN_MARKER_SEGMENT_SIZE <= len(payload):
        marker, position = _jpeg_marker(payload, position)
        if marker is None:
            return None
        if marker == _JPEG_EOI_MARKER:
            return None
        if _jpeg_marker_has_no_segment(marker):
            continue
        segment = _jpeg_segment(payload, position)
        if segment is None:
            return None
        length, data_start = segment
        if marker in _JPEG_SOF_MARKERS:
            dimensions = _jpeg_sof_dimensions(payload, data_start, length)
            if dimensions is None:
                return None
        if marker == _JPEG_SOS_MARKER:
            scan_start = position + length
            if dimensions is None or scan_start >= len(payload) - 2:
                return None
            return dimensions if payload.endswith(b"\xff\xd9") else None
        position += length
    return None


def _jpeg_marker(payload: bytes, position: int) -> tuple[int | None, int]:
    if payload[position] != _JPEG_MARKER_PREFIX:
        return None, position
    while position < len(payload) and payload[position] == _JPEG_MARKER_PREFIX:
        position += 1
    if position >= len(payload):
        return None, position
    return payload[position], position + 1


def _jpeg_marker_has_no_segment(marker: int) -> bool:
    return marker in {_JPEG_SOI_MARKER, _JPEG_EOI_MARKER} or (
        _JPEG_RESTART_MARKER_START <= marker <= _JPEG_RESTART_MARKER_END
    )


def _jpeg_segment(payload: bytes, position: int) -> tuple[int, int] | None:
    if position + _JPEG_LENGTH_SIZE > len(payload):
        return None
    length = int.from_bytes(
        payload[position : position + _JPEG_LENGTH_SIZE],
        "big",
    )
    if length < _JPEG_LENGTH_SIZE or position + length > len(payload):
        return None
    return length, position


def _jpeg_sof_dimensions(
    payload: bytes,
    data_start: int,
    length: int,
) -> tuple[int, int] | None:
    if length < _JPEG_SOF_MIN_LENGTH:
        return None
    height = int.from_bytes(
        payload[
            data_start + _JPEG_SOF_HEIGHT_START : data_start + _JPEG_SOF_HEIGHT_END
        ],
        "big",
    )
    width = int.from_bytes(
        payload[data_start + _JPEG_SOF_WIDTH_START : data_start + _JPEG_SOF_WIDTH_END],
        "big",
    )
    if not width or not height or width * height > _MAX_RGB_PIXELS:
        return None
    return width, height


def _webp_dimensions(payload: bytes) -> tuple[int, int] | None:
    if (
        len(payload) < _WEBP_CHUNK_START
        or payload[:_WEBP_RIFF_END] != _WEBP_RIFF
        or payload[_WEBP_SIGNATURE_START:_WEBP_SIGNATURE_END] != _WEBP_SIGNATURE
    ):
        return None
    chunk_type = payload[_WEBP_CHUNK_TYPE_START:_WEBP_CHUNK_TYPE_END]
    dimensions = _webp_chunk_dimensions(payload, chunk_type)
    if dimensions is None:
        return None
    width, height = dimensions
    if width * height > _MAX_RGB_PIXELS:
        return None
    return width, height


def _webp_chunk_dimensions(
    payload: bytes,
    chunk_type: bytes,
) -> tuple[int, int] | None:
    if chunk_type == _WEBP_VP8X:
        return None
    if (
        chunk_type == _WEBP_VP8
        and len(payload) >= _WEBP_VP8_MIN_SIZE
        and payload[_WEBP_VP8_FRAME_HEADER_START:_WEBP_VP8_FRAME_HEADER_END]
        == _WEBP_VP8_FRAME_HEADER
    ):
        return (
            int.from_bytes(
                payload[_WEBP_VP8_WIDTH_START:_WEBP_VP8_WIDTH_END],
                "little",
            )
            & _WEBP_VP8_DIMENSION_MASK,
            int.from_bytes(
                payload[_WEBP_VP8_HEIGHT_START:_WEBP_VP8_HEIGHT_END],
                "little",
            )
            & _WEBP_VP8_DIMENSION_MASK,
        )
    if (
        chunk_type == _WEBP_VP8L
        and len(payload) >= _WEBP_VP8L_MIN_SIZE
        and payload[_WEBP_VP8L_SIGNATURE_OFFSET] == _WEBP_VP8L_SIGNATURE
    ):
        bits = int.from_bytes(
            payload[_WEBP_VP8L_DIMENSIONS_START:_WEBP_VP8L_DIMENSIONS_END],
            "little",
        )
        return (
            (bits & _WEBP_VP8_DIMENSION_MASK) + 1,
            ((bits >> _WEBP_VP8L_HEIGHT_SHIFT) & _WEBP_VP8_DIMENSION_MASK) + 1,
        )
    return None


def _depth_dimensions(payload: bytes, expected_format: str) -> tuple[int, int] | None:
    if expected_format != "npy" or not payload.startswith(_NPY_MAGIC):
        return None
    header_data = _npy_header(payload)
    if header_data is None:
        return None
    header, header_end = header_data
    dimensions = _npy_array_dimensions(header)
    if dimensions is None:
        return None
    width, height, item_size = dimensions
    if len(payload) != header_end + width * height * item_size:
        return None
    return width, height


def _npy_header(payload: bytes) -> tuple[dict[object, object], int] | None:
    if len(payload) < _NPY_PREFIX_SIZE + _NPY_V1_HEADER_LENGTH_SIZE:
        return None
    header_length_size = _npy_header_length_size(payload[_NPY_MAJOR_VERSION_OFFSET])
    if header_length_size is None:
        return None
    header_start = _NPY_PREFIX_SIZE + header_length_size
    if len(payload) < header_start:
        return None
    header_length = int.from_bytes(
        payload[_NPY_PREFIX_SIZE:header_start],
        "little",
    )
    header_end = header_start + header_length
    if header_length > _NPY_MAX_HEADER_LENGTH or header_end > len(payload):
        return None
    try:
        parsed_header = cast(
            "object",
            ast.literal_eval(payload[header_start:header_end].decode("ascii")),
        )
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return None
    return (
        (cast("dict[object, object]", parsed_header), header_end)
        if isinstance(parsed_header, dict)
        else None
    )


def _npy_header_length_size(major: int) -> int | None:
    if major == _NPY_V1_MAJOR_VERSION:
        return _NPY_V1_HEADER_LENGTH_SIZE
    if major in {_NPY_V2_MAJOR_VERSION, _NPY_V3_MAJOR_VERSION}:
        return _NPY_V2_HEADER_LENGTH_SIZE
    return None


def _npy_array_dimensions(
    header: dict[object, object],
) -> tuple[int, int, int] | None:
    descriptor_value: object = header.get("descr")
    fortran_order: object = header.get("fortran_order")
    shape_value: object = header.get("shape")
    if (
        not isinstance(descriptor_value, str)
        or descriptor_value not in _NPY_ALLOWED_DESCRIPTORS
        or fortran_order is not False
        or not isinstance(shape_value, tuple)
    ):
        return None
    shape = cast("tuple[object, ...]", shape_value)
    if len(shape) != _NPY_DIMENSION_COUNT:
        return None
    dimensions: list[int] = []
    for value in shape:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return None
        dimensions.append(value)
    height, width = dimensions
    if width * height > _MAX_DEPTH_PIXELS:
        return None
    return width, height, _NPY_ITEM_SIZES[descriptor_value]


def _ppm_dimensions(payload: bytes) -> tuple[int, int] | None:
    parts = payload.split(maxsplit=_PPM_PART_COUNT - 1)
    if (
        len(parts) != _PPM_PART_COUNT
        or parts[0] != _PPM_MAGIC
        or parts[3] != _PPM_MAX_VALUE
    ):
        return None
    try:
        width, height = int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if (
        width <= 0
        or height <= 0
        or width * height > _MAX_RGB_PIXELS
        or len(parts[_PPM_PAYLOAD_INDEX]) != width * height * _PNG_RGB_CHANNELS
    ):
        return None
    return width, height


def _png_dimensions(  # noqa: PLR0911, PLR0912
    payload: bytes,
) -> tuple[int, int] | None:
    if not payload.startswith(_PNG_SIGNATURE):
        return None
    position = _PNG_HEADER_SIZE
    width: int | None = None
    height: int | None = None
    color_type: int | None = None
    compressed = bytearray()
    saw_iend = False
    while position + _PNG_CHUNK_OVERHEAD <= len(payload):
        length = int.from_bytes(
            payload[position : position + _PNG_LENGTH_SIZE],
            "big",
        )
        chunk_type = payload[position + _PNG_TYPE_START : position + _PNG_TYPE_END]
        data_end = position + _PNG_DATA_START + length
        end = data_end + _PNG_LENGTH_SIZE
        if end > len(payload):
            return None
        chunk = payload[position + _PNG_DATA_START : data_end]
        checksum = int.from_bytes(payload[data_end:end], "big")
        if zlib.crc32(chunk_type + chunk) != checksum:
            return None
        if width is None and chunk_type != _PNG_IHDR:
            return None
        if chunk_type == _PNG_IHDR:
            if width is not None or length != _PNG_IHDR_LENGTH:
                return None
            width = int.from_bytes(chunk[:_PNG_DIMENSION_SIZE], "big")
            height = int.from_bytes(
                chunk[_PNG_DIMENSION_SIZE : _PNG_DIMENSION_SIZE * 2],
                "big",
            )
            bit_depth, color_type = chunk[_PNG_TYPE_END], chunk[_PNG_TYPE_END + 1]
            if (
                not width
                or not height
                or width * height > _MAX_RGB_PIXELS
                or bit_depth != _PNG_BIT_DEPTH
                or color_type not in _PNG_RGB_COLOR_TYPES
            ):
                return None
        elif chunk_type == _PNG_IDAT:
            if saw_iend:
                return None
            compressed.extend(chunk)
        elif chunk_type == _PNG_IEND:
            if length != 0 or not compressed or end != len(payload):
                return None
            saw_iend = True
            break
        position = end
    if (
        not saw_iend
        or width is None
        or height is None
        or color_type is None
        or not compressed
    ):
        return None
    try:
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(bytes(compressed), _MAX_RGB_BYTES)
        decoded += decoder.flush(_MAX_RGB_BYTES - len(decoded))
    except (ValueError, zlib.error):
        return None
    channels = (
        _PNG_RGBA_CHANNELS if color_type == _PNG_ALPHA_COLOR_TYPE else _PNG_RGB_CHANNELS
    )
    if (
        not decoder.eof
        or decoder.unused_data
        or len(decoded) != height * (_PNG_FILTER_BYTES_PER_ROW + width * channels)
    ):
        return None
    return width, height


def _load_json(line: str) -> object:
    return _JSON_VALUE_ADAPTER.validate_python(
        json.loads(
            line,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_constant,
            parse_float=_parse_finite_json_float,
        )
    )


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            detail = f"duplicate JSON key {key!r}"
            raise ValueError(detail)
        result[key] = value
    return result


def _reject_non_finite_json_constant(value: str) -> object:
    detail = f"non-finite JSON value {value}"
    raise ValueError(detail)


def _parse_finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        detail = f"non-finite JSON value {value}"
        raise ValueError(detail)
    return result


def _json_object(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return _JSON_OBJECT_ADAPTER.validate_python(value)


def _json_list(value: object) -> list[object] | None:
    if not isinstance(value, list):
        return None
    return _JSON_LIST_ADAPTER.validate_python(value)


def _check_leakage(
    value: object, location: str, issues: list[SensorAuditIssue]
) -> None:
    mapping = _json_object(value)
    if mapping is not None:
        for key, nested_value in mapping.items():
            if _contains_leakage_term(key):
                _issue(issues, "leakage_field", f"{location}.{key}")
            _check_leakage(nested_value, f"{location}.{key}", issues)
        return
    sequence = _json_list(value)
    if sequence is not None:
        for index, nested_value in enumerate(sequence):
            _check_leakage(nested_value, f"{location}[{index}]", issues)
        return
    if (
        isinstance(value, str)
        and ("/" in value or "\\" in value)
        and any(
            _contains_leakage_term(part) for part in value.replace("\\", "/").split("/")
        )
    ):
        _issue(issues, "leakage_path", f"{location}={value}")


def _contains_leakage_term(value: str) -> bool:
    lowered = value.lower()
    return any(term in lowered for term in _LEAKAGE_TERMS)


def _read_bytes(path: Path, kind: str, issues: list[SensorAuditIssue]) -> bytes:
    payload = _read_nofollow_bytes(path, kind, issues, limit=_MAX_RGB_BYTES)
    return payload if payload is not None else b""


def _open_directory_nofollow(path: Path) -> int | None:
    """Pin an absolute directory without crossing symlinks.

    Host-path acquisition intentionally does not use ``RESOLVE_NO_XDEV``:
    approved company roots may themselves be mount points.  Assets below the
    pinned descriptor are resolved by ``openat2_sealed`` and therefore cannot
    traverse a mount, symlink, or magic link.
    """
    absolute_path = Path(os.path.abspath(path))  # noqa: PTH100
    descriptor = os.open(
        absolute_path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        for component in absolute_path.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError:
        os.close(descriptor)
        return None
    return descriptor


def _read_nofollow_bytes(
    path: Path,
    kind: str,
    issues: list[SensorAuditIssue],
    *,
    limit: int,
    observation_id: str | None = None,
) -> bytes | None:
    absolute_path = Path(os.path.abspath(path))  # noqa: PTH100
    descriptor = _open_directory_nofollow(absolute_path.parent)
    if descriptor is None:
        _issue(issues, f"{kind}_unreadable", str(path), observation_id)
        return None
    try:
        return _read_nofollow_bytes_at(
            descriptor,
            Path(absolute_path.name),
            path,
            kind,
            issues,
            limit=limit,
            observation_id=observation_id,
        )
    finally:
        os.close(descriptor)


def _read_nofollow_bytes_at(  # noqa: PLR0913
    root_descriptor: int,
    relative: Path,
    path: Path,
    kind: str,
    issues: list[SensorAuditIssue],
    *,
    limit: int,
    observation_id: str | None = None,
) -> bytes | None:
    try:
        descriptor = openat2_sealed(root_descriptor, relative, os.O_RDONLY)
    except Openat2UnsupportedError as exc:
        _issue(
            issues,
            f"{kind}_unreadable",
            f"{path}: openat2 sealed-root resolution unavailable: {exc}",
            observation_id,
        )
        return None
    except OSError as exc:
        issue_kind = (
            f"{kind}_symlink" if exc.errno == errno.ELOOP else f"{kind}_unreadable"
        )
        _issue(issues, issue_kind, f"{path}: {exc}", observation_id)
        return None
    try:
        return _read_open_descriptor(
            descriptor, path, kind, issues, limit, observation_id
        )
    finally:
        os.close(descriptor)


def _read_open_descriptor(  # noqa: PLR0913
    descriptor: int,
    path: Path,
    kind: str,
    issues: list[SensorAuditIssue],
    limit: int,
    observation_id: str | None,
) -> bytes | None:
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            _issue(issues, f"{kind}_not_regular_file", str(path), observation_id)
            return None
        if metadata.st_size > limit:
            _issue(issues, f"{kind}_too_large", str(path), observation_id)
            return None
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > limit:
            _issue(issues, f"{kind}_too_large", str(path), observation_id)
            return None
    except OSError as exc:
        _issue(issues, f"{kind}_unreadable", f"{path}: {exc}", observation_id)
        return None
    else:
        return payload


def _directory_identity_digest(descriptor: int | None) -> str:
    if descriptor is None:
        return _sha256(b"invalid-root")
    metadata = os.fstat(descriptor)
    return _sha256(f"{metadata.st_dev}:{metadata.st_ino}".encode())


def _verify_directory_identity(
    path: Path,
    descriptor: int | None,
    kind: str,
    issues: list[SensorAuditIssue],
    observation_id: str | None = None,
) -> None:
    if descriptor is None:
        return
    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        _issue(issues, f"{kind}_changed", f"{path}: {exc}", observation_id)
        return
    current_descriptor = _open_directory_nofollow(path)
    if current_descriptor is None:
        _issue(
            issues,
            f"{kind}_changed",
            f"{path}: cannot reopen sealed directory root",
            observation_id,
        )
        return
    try:
        current = os.fstat(current_descriptor)
    except OSError as exc:
        _issue(issues, f"{kind}_changed", f"{path}: {exc}", observation_id)
        return
    finally:
        os.close(current_descriptor)
    if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
        _issue(issues, f"{kind}_changed", str(path), observation_id)


def _provider_gate(
    issues: list[SensorAuditIssue],
    selected_count: int,
    rgb_valid_count: int,
) -> Literal["go", "no_go", "not_measurable", "not_decidable"]:
    integrity_prefixes = (
        "canonical_",
        "duplicate_",
        "extra_",
        "invalid_",
        "leakage_",
        "missing_",
        "rgb_",
        "manifest_",
        "observations_",
    )
    if any(
        issue.code.startswith(integrity_prefixes)
        or (issue.code.startswith("depth_") and issue.code != "depth_root_required")
        or issue.code
        in {
            "empty_observations",
            "empty_selected_inventory",
            "frame_root_changed",
            "timestamp_join_ambiguous",
        }
        for issue in issues
    ):
        return "not_decidable"
    if issues:
        return "no_go"
    if not selected_count or not rgb_valid_count:
        return "not_measurable"
    return "go"


def _canonical_digest(value: object) -> str:
    return _sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _percent(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(100.0 * numerator / denominator, 2)


def _issue(
    issues: list[SensorAuditIssue],
    code: str,
    detail: str,
    observation_id: str | None = None,
) -> None:
    issues.append(
        SensorAuditIssue(code=code, detail=detail, observation_id=observation_id)
    )
