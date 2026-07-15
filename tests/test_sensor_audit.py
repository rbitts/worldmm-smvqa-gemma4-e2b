from __future__ import annotations

import errno
import hashlib
import json
import os
import zlib
from pathlib import Path

import pytest

from worldmm_smvqa import sensor_audit
from worldmm_smvqa.openat2 import Openat2UnsupportedError, openat2_sealed
from worldmm_smvqa.sensor_audit import (
    SensorAuditReport,
    SensorModalityPolicy,
    audit_sensors,
    write_sensor_audit_report,
)
from worldmm_smvqa.worldmm.spatial_sensor import canonical_timestamp_us

_RGB = b"P6\n1 1\n255\n\x00\x00\x00"


def _write_manifest(
    path: Path,
    frame_ref: str = "frame.ppm",
    timestamp: float = 0.0,
) -> None:
    _ = path.write_text(
        json.dumps(
            {
                "video_id": "video-a",
                "cadence_origin": 0.0,
                "source_frame_count": 1,
                "source_frame_sha256": "0" * 64,
                "selected_frames": [
                    {"sample_index": 0, "frame_ref": frame_ref, "timestamp": timestamp}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _observation(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "observation_id": "observation-1",
        "video_id": "video-a",
        "timestamp": 0.0,
        "frame_ref": "frame.ppm",
        "local_frame_id": "device",
        "intrinsics": {
            "width_px": 1,
            "height_px": 1,
            "fx": 1.0,
            "fy": 1.0,
            "cx": 0.0,
            "cy": 0.0,
        },
        "rgb_sha256": hashlib.sha256(_RGB).hexdigest(),
    }
    _ = result.update(updates)
    return result


def _setup(
    tmp_path: Path,
    *,
    frame_ref: str = "frame.ppm",
    rgb: bytes = _RGB,
    timestamp: float = 0.0,
) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "manifest.jsonl"
    observations = tmp_path / "observations.jsonl"
    frame_root = tmp_path / "frames"
    _ = (frame_root / "video-a").mkdir(parents=True)
    _ = (frame_root / "video-a" / frame_ref).write_bytes(rgb)
    _write_manifest(manifest, frame_ref, timestamp)
    payload_digest = hashlib.sha256(rgb).hexdigest()
    serialized = json.dumps(
        _observation(
            frame_ref=frame_ref, timestamp=timestamp, rgb_sha256=payload_digest
        )
    )
    _ = observations.write_text(serialized + "\n", encoding="utf-8")
    return manifest, observations, frame_root


def _codes(report: SensorAuditReport) -> set[str]:
    return {issue.code for issue in report.issues}


def _npy_depth(width: int = 1, height: int = 1) -> bytes:
    shape = f"({height}, {width})"
    header_text = "{'descr': '<f4', 'fortran_order': False, "
    header_text += f"'shape': {shape}, }}"
    header = header_text.encode()
    padding = (16 - ((10 + len(header) + 1) % 16)) % 16
    header += b" " * padding + b"\n"
    return (
        b"\x93NUMPY\x01\x00"
        + len(header).to_bytes(2, "little")
        + header
        + b"\x00" * (width * height * 4)
    )


def _depth_observation(payload: bytes, **updates: object) -> dict[str, object]:
    depth: dict[str, object] = {
        "depth_ref": "depth.npy",
        "depth_scale_m": 0.001,
        "depth_sha256": hashlib.sha256(payload).hexdigest(),
        "width_px": 1,
        "height_px": 1,
        "shape": [1, 1],
        "format": "npy",
        "provenance": "native_depth_sensor",
    }
    depth.update(updates)
    return depth


def test_exact_join_with_verified_rgb_and_intrinsics(tmp_path: Path) -> None:
    manifest, observations, frame_root = _setup(tmp_path)

    report = audit_sensors(manifest, observations, frame_root)

    assert report.operational_state == "ready"
    assert report.counts["joined_observations"] == 1
    assert report.coverage["rgb_percent"] == 100.0
    assert report.coverage["intrinsics_percent"] == 100.0


def test_canonical_timestamp_identity_uses_decimal_half_even_rounding(
    tmp_path: Path,
) -> None:
    assert canonical_timestamp_us("0.0000005") == 0
    assert canonical_timestamp_us("0.0000015") == 2

    manifest, observations, frame_root = _setup(tmp_path, timestamp=0.0000005)
    _ = observations.write_text(
        json.dumps(_observation(timestamp="0.0000005")) + "\n", encoding="utf-8"
    )

    report = audit_sensors(manifest, observations, frame_root)

    assert report.operational_state == "ready"
    assert report.counts["joined_observations"] == 1
    _ = observations.write_text(
        json.dumps(_observation(timestamp=0.00000051)) + "\n", encoding="utf-8"
    )
    beyond_half_microsecond = audit_sensors(manifest, observations, frame_root)

    assert {"missing_observation", "extra_observation"} <= _codes(
        beyond_half_microsecond
    )
    assert beyond_half_microsecond.provider_gate_decision == "not_decidable"
    manifest, observations, frame_root = _setup(
        tmp_path / "same-bin",
        timestamp=0.00000051,
    )
    _ = observations.write_text(
        json.dumps(_observation(timestamp=0.00000149)) + "\n", encoding="utf-8"
    )
    same_bin_ambiguous = audit_sensors(manifest, observations, frame_root)

    assert "timestamp_join_ambiguous" in _codes(same_bin_ambiguous)
    assert same_bin_ambiguous.provider_gate_decision == "not_decidable"


def test_rejects_overflowed_json_float(tmp_path: Path) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    _ = observations.write_text(
        json.dumps(_observation()).replace("0.0", "1e9999", 1) + "\n",
        encoding="utf-8",
    )

    report = audit_sensors(manifest, observations, frame_root)

    assert "invalid_observation" in _codes(report)


def test_canonical_timestamp_collision_and_duplicate_manifest_block(
    tmp_path: Path,
) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    _ = observations.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _observation(observation_id="first", timestamp=0.0000004),
                _observation(observation_id="second", timestamp=0.00000049),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    collision = audit_sensors(manifest, observations, frame_root)

    assert "canonical_key_collision" in _codes(collision)
    assert collision.operational_state == "blocked"
    assert collision.provider_gate_decision == "not_decidable"

    duplicated = manifest.read_text(encoding="utf-8")
    _ = manifest.write_text(duplicated + duplicated, encoding="utf-8")
    duplicate_manifest = audit_sensors(manifest, observations, frame_root)

    assert "invalid_manifest" in _codes(duplicate_manifest)
    assert duplicate_manifest.provider_gate_decision == "not_decidable"


def test_rejects_missing_extra_and_duplicate_observations(tmp_path: Path) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    duplicate = _observation(observation_id="observation-2")
    extra = _observation(observation_id="observation-3", timestamp=1.0)
    _ = observations.write_text(
        "\n".join(json.dumps(row) for row in (_observation(), duplicate, extra)) + "\n",
        encoding="utf-8",
    )

    report = audit_sensors(manifest, observations, frame_root)

    assert {"duplicate_observation", "extra_observation"} <= _codes(report)


def test_rejects_ancestor_symlinked_frame_root(tmp_path: Path) -> None:
    manifest, observations, frame_root = _setup(tmp_path / "backing")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "backing", target_is_directory=True)

    report = audit_sensors(manifest, observations, alias / frame_root.name)

    assert "frame_root_invalid" in _codes(report)


def test_pins_mounted_root_before_sealed_relative_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, observations, frame_root = _setup(tmp_path / "mounted-company-root")
    resolved: list[str] = []
    real_openat2 = openat2_sealed

    def record_relative_resolution(
        dir_fd: int, relative_path: str | os.PathLike[str], flags: int
    ) -> int:
        resolved.append(str(relative_path))
        return real_openat2(dir_fd, relative_path, flags)

    monkeypatch.setattr(sensor_audit, "openat2_sealed", record_relative_resolution)

    report = audit_sensors(manifest, observations, frame_root)

    assert not report.issues
    assert resolved
    assert all(not Path(path).is_absolute() for path in resolved)
    assert all("mounted-company-root" not in path for path in resolved)


def test_openat2_reports_magic_link_and_mount_crossing_asset_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    magic_ref = "proc/self/fd/7"
    _write_manifest(manifest, magic_ref)
    _ = observations.write_text(
        json.dumps(_observation(frame_ref=magic_ref)) + "\n",
        encoding="utf-8",
    )
    real_openat2 = openat2_sealed

    def reject_magic_link(
        dir_fd: int,
        relative_path: str | os.PathLike[str],
        flags: int,
    ) -> int:
        if str(relative_path) == f"video-a/{magic_ref}":
            raise OSError(errno.ELOOP, "too many symbolic links")
        return real_openat2(dir_fd, relative_path, flags)

    monkeypatch.setattr(sensor_audit, "openat2_sealed", reject_magic_link)
    assert "rgb_symlink" in _codes(audit_sensors(manifest, observations, frame_root))

    _write_manifest(manifest)
    _ = observations.write_text(json.dumps(_observation()) + "\n", encoding="utf-8")

    def reject_mount_crossing(
        dir_fd: int,
        relative_path: str | os.PathLike[str],
        flags: int,
    ) -> int:
        if str(relative_path) == "video-a/frame.ppm":
            raise OSError(errno.EXDEV, "cross-device link")
        return real_openat2(dir_fd, relative_path, flags)

    monkeypatch.setattr(sensor_audit, "openat2_sealed", reject_mount_crossing)
    assert "rgb_unreadable" in _codes(audit_sensors(manifest, observations, frame_root))


def test_sensor_audit_requires_openat2_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, observations, frame_root = _setup(tmp_path)

    def unavailable(*_args: object, **_kwargs: object) -> int:
        message = "kernel does not implement openat2"
        raise Openat2UnsupportedError(message)

    monkeypatch.setattr(
        "worldmm_smvqa.sensor_audit.openat2_sealed",
        unavailable,
    )
    report = audit_sensors(manifest, observations, frame_root)
    assert report.operational_state == "blocked"
    assert report.provider_gate_decision == "not_decidable"
    assert {"manifest_unreadable", "observations_unreadable"} <= _codes(report)


def test_rejects_frame_root_swap_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    replacement = tmp_path / "replacement"
    _ = replacement.mkdir()

    def swap_then_check(
        root: Path,
        frame_root_descriptor: int | None,
        observation: sensor_audit.SensorAuditObservation,
        issues: list[sensor_audit.SensorAuditIssue],
    ) -> bool:
        _ = root, frame_root_descriptor, observation, issues
        _ = frame_root.rename(tmp_path / "old-frames")
        _ = replacement.rename(frame_root)
        return True

    monkeypatch.setattr(sensor_audit, "_check_rgb_asset", swap_then_check)

    report = audit_sensors(manifest, observations, frame_root)

    assert "frame_root_changed" in _codes(report)


def test_rejects_missing_observation(tmp_path: Path) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    _ = observations.write_text("", encoding="utf-8")

    assert "missing_observation" in _codes(
        audit_sensors(manifest, observations, frame_root)
    )


def test_rejects_unreadable_hash_mismatch_and_symlink_rgb(tmp_path: Path) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    asset = frame_root / "video-a" / "frame.ppm"
    invalid = b"not an RGB image"
    _ = asset.write_bytes(invalid)
    _ = observations.write_text(
        json.dumps(_observation(rgb_sha256=hashlib.sha256(invalid).hexdigest())) + "\n",
        encoding="utf-8",
    )
    report = audit_sensors(manifest, observations, frame_root)
    assert "rgb_not_readable" in _codes(report)

    _ = asset.write_bytes(_RGB)
    _ = observations.write_text(
        json.dumps(_observation(rgb_sha256=hashlib.sha256(b"wrong").hexdigest()))
        + "\n",
        encoding="utf-8",
    )
    assert "rgb_hash_mismatch" in _codes(
        audit_sensors(manifest, observations, frame_root)
    )

    _ = asset.unlink()
    _ = asset.symlink_to(tmp_path / "asset.ppm")
    _ = (tmp_path / "asset.ppm").write_bytes(_RGB)
    _ = observations.write_text(json.dumps(_observation()) + "\n", encoding="utf-8")
    assert "rgb_symlink" in _codes(audit_sensors(manifest, observations, frame_root))


def test_reports_absent_pose_without_synthesis_and_rejects_untrusted_pose(
    tmp_path: Path,
) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    report = audit_sensors(manifest, observations, frame_root)
    assert report.coverage["trusted_pose_percent"] == 0.0
    assert report.counts["depth_available"] == report.counts["gaze_available"] == 0

    _ = observations.write_text(
        json.dumps(
            _observation(
                pose={"timestamp": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "source": "slam"}
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert "invalid_observation" in _codes(
        audit_sensors(manifest, observations, frame_root)
    )


def test_requires_intrinsics_and_rejects_leakage(tmp_path: Path) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    observation = _observation()
    del observation["intrinsics"]
    _ = observations.write_text(json.dumps(observation) + "\n", encoding="utf-8")
    assert "invalid_observation" in _codes(
        audit_sensors(manifest, observations, frame_root)
    )

    _ = observations.write_text(
        json.dumps(_observation(qa_label="leak")) + "\n", encoding="utf-8"
    )
    assert "leakage_field" in _codes(audit_sensors(manifest, observations, frame_root))


@pytest.mark.parametrize(
    ("replacement", "diagnostic"),
    [
        ('"timestamp": 0.0, "timestamp": 0.0', "duplicate JSON key"),
        ('"timestamp": NaN', "non-finite JSON value NaN"),
        ('"timestamp": Infinity', "non-finite JSON value Infinity"),
    ],
)
def test_manifest_rejects_ambiguous_json_before_validation(
    tmp_path: Path,
    replacement: str,
    diagnostic: str,
) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    payload = manifest.read_text(encoding="utf-8").replace(
        '"timestamp": 0.0', replacement
    )
    _ = manifest.write_text(payload, encoding="utf-8")

    report = audit_sensors(manifest, observations, frame_root)

    assert report.operational_state == "blocked"
    assert any(
        issue.code == "invalid_manifest"
        and "line 1" in issue.detail
        and diagnostic in issue.detail
        for issue in report.issues
    )


@pytest.mark.parametrize(
    ("replacement", "diagnostic"),
    [
        ('"timestamp": 0.0, "timestamp": 0.0', "duplicate JSON key"),
        ('"timestamp": NaN', "non-finite JSON value NaN"),
        ('"timestamp": Infinity', "non-finite JSON value Infinity"),
    ],
)
def test_observations_reject_ambiguous_json_before_validation(
    tmp_path: Path,
    replacement: str,
    diagnostic: str,
) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    payload = observations.read_text(encoding="utf-8").replace(
        '"timestamp": 0.0', replacement
    )
    _ = observations.write_text(payload, encoding="utf-8")

    report = audit_sensors(manifest, observations, frame_root)

    assert report.operational_state == "blocked"
    assert any(
        issue.code == "invalid_observation"
        and "line 1" in issue.detail
        and diagnostic in issue.detail
        for issue in report.issues
    )


def test_report_digest_and_serialization_are_deterministic(tmp_path: Path) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    report_one = write_sensor_audit_report(manifest, observations, frame_root, first)
    report_two = write_sensor_audit_report(manifest, observations, frame_root, second)

    assert report_one == report_two
    assert first.read_bytes() == second.read_bytes()


def test_blocks_corrupt_and_dimension_mismatched_rgb(tmp_path: Path) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    asset = frame_root / "video-a" / "frame.ppm"
    mismatched = b"P6\n2 1\n255\n\x00\x00\x00\x00\x00\x00"
    _ = asset.write_bytes(mismatched)
    _ = observations.write_text(
        json.dumps(_observation(rgb_sha256=hashlib.sha256(mismatched).hexdigest()))
        + "\n",
        encoding="utf-8",
    )
    assert "rgb_dimensions_mismatch" in _codes(
        audit_sensors(manifest, observations, frame_root)
    )

    corrupt = b"P6\n1 1\n255\n\x00"
    _ = asset.write_bytes(corrupt)
    _ = observations.write_text(
        json.dumps(_observation(rgb_sha256=hashlib.sha256(corrupt).hexdigest())) + "\n",
        encoding="utf-8",
    )
    assert "rgb_not_readable" in _codes(
        audit_sensors(manifest, observations, frame_root)
    )


def test_input_digest_and_modality_policy_fail_closed(tmp_path: Path) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    baseline = audit_sensors(manifest, observations, frame_root)
    _ = observations.write_text(
        json.dumps(_observation(local_frame_id="other-device")) + "\n",
        encoding="utf-8",
    )
    changed = audit_sensors(
        manifest,
        observations,
        frame_root,
        SensorModalityPolicy(min_depth_percent=100.0),
    )
    assert baseline.input_digest != changed.input_digest
    assert changed.operational_state == "blocked"
    assert changed.provider_gate_decision == "no_go"
    assert "modality_coverage_below_policy" in _codes(changed)


def test_depth_policy_counts_only_verified_assets_and_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    depth_root = tmp_path / "depth"
    _ = (depth_root / "video-a").mkdir(parents=True)
    depth = _npy_depth()
    asset = depth_root / "video-a" / "depth.npy"
    _ = asset.write_bytes(depth)
    policy = SensorModalityPolicy(min_depth_percent=100.0)
    _ = observations.write_text(
        json.dumps(_observation(depth=_depth_observation(depth))) + "\n",
        encoding="utf-8",
    )

    verified = audit_sensors(
        manifest, observations, frame_root, policy, depth_root=depth_root
    )
    assert verified.operational_state == "ready"
    assert verified.counts["depth_available"] == 1

    _ = asset.unlink()
    assert "depth_unreadable" in _codes(
        audit_sensors(manifest, observations, frame_root, policy, depth_root=depth_root)
    )
    _ = asset.write_bytes(depth)

    _ = asset.write_bytes(_npy_depth(width=2))
    assert "depth_hash_mismatch" in _codes(
        audit_sensors(manifest, observations, frame_root, policy, depth_root=depth_root)
    )

    mismatched = _npy_depth(width=2)
    _ = asset.write_bytes(mismatched)
    _ = observations.write_text(
        json.dumps(_observation(depth=_depth_observation(mismatched))) + "\n",
        encoding="utf-8",
    )
    assert "depth_dimensions_mismatch" in _codes(
        audit_sensors(manifest, observations, frame_root, policy, depth_root=depth_root)
    )

    _ = asset.unlink()
    _ = asset.symlink_to(tmp_path / "depth.npy")
    _ = (tmp_path / "depth.npy").write_bytes(depth)
    _ = observations.write_text(
        json.dumps(_observation(depth=_depth_observation(depth))) + "\n",
        encoding="utf-8",
    )
    assert "depth_symlink" in _codes(
        audit_sensors(manifest, observations, frame_root, policy, depth_root=depth_root)
    )


def test_depth_policy_rejects_missing_root_and_invalid_depth_contract(
    tmp_path: Path,
) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    depth = _npy_depth()
    policy = SensorModalityPolicy(min_depth_percent=100.0)
    _ = observations.write_text(
        json.dumps(_observation(depth=_depth_observation(depth))) + "\n",
        encoding="utf-8",
    )

    missing_root = audit_sensors(manifest, observations, frame_root, policy)
    assert {"depth_root_required", "modality_coverage_below_policy"} <= _codes(
        missing_root
    )

    _ = observations.write_text(
        json.dumps(
            _observation(depth=_depth_observation(depth, depth_sha256="malformed"))
        )
        + "\n",
        encoding="utf-8",
    )
    assert "invalid_observation" in _codes(
        audit_sensors(manifest, observations, frame_root, policy)
    )

    _ = observations.write_text(
        json.dumps(_observation(depth=_depth_observation(depth, provenance=""))) + "\n",
        encoding="utf-8",
    )
    assert "invalid_observation" in _codes(
        audit_sensors(manifest, observations, frame_root, policy)
    )


def test_only_decodable_rgb_assets_are_verified(tmp_path: Path) -> None:
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            len(data).to_bytes(4, "big")
            + chunk_type
            + data
            + zlib.crc32(chunk_type + data).to_bytes(4, "big")
        )

    png = b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"),
            chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")),
            chunk(b"IEND", b""),
        )
    )
    for frame_ref, payload in (("frame", _RGB), ("frame.png", png)):
        manifest, observations, frame_root = _setup(
            tmp_path / hashlib.sha256(frame_ref.encode()).hexdigest(),
            frame_ref=frame_ref,
            rgb=payload,
        )
        report = audit_sensors(manifest, observations, frame_root)
        assert report.operational_state == "ready"

    manifest, observations, frame_root = _setup(tmp_path / "corrupt", rgb=png[:-1])
    corrupt = audit_sensors(manifest, observations, frame_root)

    assert "rgb_not_readable" in _codes(corrupt)
    assert corrupt.provider_gate_decision == "not_decidable"
    crc_invalid = bytearray(png)
    crc_invalid[-5] ^= 1
    manifest, observations, frame_root = _setup(
        tmp_path / "crc-invalid", rgb=bytes(crc_invalid)
    )
    assert "rgb_not_readable" in _codes(
        audit_sensors(manifest, observations, frame_root)
    )

    header_only = b"\xff\xd8\xff\xc0\x00\x08\x08\x00\x01\x00\x01\x03\xff\xd9"
    manifest, observations, frame_root = _setup(
        tmp_path / "header-only", rgb=header_only
    )
    assert "rgb_not_readable" in _codes(
        audit_sensors(manifest, observations, frame_root)
    )


def test_rejects_symlinked_manifest_observations_and_frame_root(
    tmp_path: Path,
) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    manifest_link = tmp_path / "manifest-link.jsonl"
    observations_link = tmp_path / "observations-link.jsonl"
    frame_root_link = tmp_path / "frames-link"
    manifest_link.symlink_to(manifest)
    observations_link.symlink_to(observations)
    frame_root_link.symlink_to(frame_root, target_is_directory=True)

    assert "manifest_symlink" in _codes(
        audit_sensors(manifest_link, observations, frame_root)
    )
    assert "observations_symlink" in _codes(
        audit_sensors(manifest, observations_link, frame_root)
    )
    assert "frame_root_invalid" in _codes(
        audit_sensors(manifest, observations, frame_root_link)
    )


def test_input_files_are_read_from_opened_snapshot_after_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    original_manifest = manifest.read_bytes()
    original_observations = observations.read_bytes()
    replacements = {
        manifest.stat().st_ino: manifest,
        observations.stat().st_ino: observations,
    }
    replaced: set[Path] = set()
    real_fstat = sensor_audit.os.fstat  # pyright: ignore[reportPrivateLocalImportUsage]

    def replace_after_validation(descriptor: int) -> os.stat_result:
        metadata = real_fstat(descriptor)
        target = replacements.get(metadata.st_ino)
        if target is not None and target not in replaced:
            replacement = target.with_suffix(".replacement")
            _ = replacement.write_bytes(b"")
            _ = replacement.replace(target)
            replaced.add(target)
        return metadata

    monkeypatch.setattr(
        sensor_audit.os,  # pyright: ignore[reportPrivateLocalImportUsage]
        "fstat",
        replace_after_validation,
    )
    report = audit_sensors(manifest, observations, frame_root)

    assert replaced == {manifest, observations}
    assert manifest.read_bytes() == observations.read_bytes() == b""
    assert report.operational_state == "ready"
    assert report.manifest_digest == hashlib.sha256(original_manifest).hexdigest()
    assert (
        report.observations_digest == hashlib.sha256(original_observations).hexdigest()
    )


def test_frame_root_is_pinned_before_path_validation_and_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, observations, frame_root = _setup(tmp_path)
    original_path_is_safe = sensor_audit._asset_path_is_safe  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    moved_root = tmp_path / "frames-approved"
    replaced = False

    def replace_root_after_validation(
        root: Path,
        path: Path,
        kind: str,
        observation_id: str,
        issues: list[sensor_audit.SensorAuditIssue],
    ) -> bool:
        nonlocal replaced
        safe = original_path_is_safe(root, path, kind, observation_id, issues)
        if safe and not replaced:
            _ = frame_root.replace(moved_root)
            _ = (frame_root / "video-a").mkdir(parents=True)
            _ = (frame_root / "video-a" / "frame.ppm").write_bytes(b"truncated")
            replaced = True
        return safe

    monkeypatch.setattr(
        sensor_audit, "_asset_path_is_safe", replace_root_after_validation
    )
    report = audit_sensors(manifest, observations, frame_root)

    assert replaced
    assert report.operational_state == "blocked"
    assert "frame_root_changed" in _codes(report)
    assert (frame_root / "video-a" / "frame.ppm").read_bytes() == b"truncated"
