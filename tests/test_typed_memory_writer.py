from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldmm_smvqa.schema import FrameMetadata, SourceStreamExample
from worldmm_smvqa.sensor_frames import build_sensor_frame_manifest
from worldmm_smvqa.worldmm.typed_memory import (
    NoWriteMemoryRecord,
    ObjectGeometry,
    ObjectMemoryRecord,
    ScoredMemoryCandidate,
    SpatialUncertainty,
    TypedMemoryWriterError,
    ValidityInterval,
    canonical_jsonl_bytes,
    serialized_byte_cost,
    validate_typed_memory_artifact,
    write_typed_memory_artifact,
)


def _object(memory_id: str, *, label: str = "mug") -> ObjectMemoryRecord:
    return ObjectMemoryRecord(
        memory_id=memory_id,
        source_video_id="video-1",
        entity_id=f"entity-{memory_id}",
        instance_id=f"instance-{memory_id}",
        local_frame_id="room-1",
        geometry_uncertainty=SpatialUncertainty(
            covariance_xyz=(
                (0.01, 0.0, 0.0),
                (0.0, 0.01, 0.0),
                (0.0, 0.0, 0.01),
            ),
            standard_deviation_m=0.1,
        ),
        validity=ValidityInterval(start_time=1.0, end_time=2.0),
        first_seen_time=1.0,
        last_seen_time=2.0,
        observation_count=1,
        confidence=0.9,
        provenance="observed",
        geometry=ObjectGeometry(centroid=(1, 2, 3), extent=(1, 1, 1)),
        semantic_label=label,
    )


def _no_write(memory_id: str) -> NoWriteMemoryRecord:
    source = _object(memory_id)
    common = source.model_dump(exclude={"geometry", "semantic_label", "record_type"})
    return NoWriteMemoryRecord.model_validate(
        {
            **common,
            "candidate_type": "object",
            "reason": "duplicate observation",
        }
    )


def _scored(
    record: ObjectMemoryRecord | NoWriteMemoryRecord,
    score: float,
) -> ScoredMemoryCandidate:
    return ScoredMemoryCandidate(record=record, score=score)


def test_writer_uses_exact_utf8_jsonl_bytes_and_summary(tmp_path: Path) -> None:
    record = _object("memory-한글", label="머그컵")
    output = tmp_path / "nested" / "memory.jsonl"
    expected = canonical_jsonl_bytes(record)

    summary = write_typed_memory_artifact(
        (_scored(record, 1.0),),
        output=output,
        byte_budget=len(expected),
    )

    assert output.read_bytes() == expected
    assert output.stat().st_size == len(expected)
    assert summary.actual_bytes == len(expected)
    assert summary.selected_memory_ids == ("memory-한글",)
    assert json.loads(output.read_text().strip())["semantic_label"] == "머그컵"


def test_writer_skips_oversize_candidate_and_keeps_later_fit(tmp_path: Path) -> None:
    large = _object("large", label="x" * 200)
    small = _object("small")
    budget = serialized_byte_cost(small)

    summary = write_typed_memory_artifact(
        (_scored(large, 100.0), _scored(small, 1.0)),
        output=tmp_path / "memory.jsonl",
        byte_budget=budget,
    )

    assert summary.selected_memory_ids == ("small",)
    assert summary.skipped_for_budget_count == 1


def test_equal_value_per_byte_preserves_decoder_order(tmp_path: Path) -> None:
    first = _object("first")
    second = _object("other")
    first_bytes = serialized_byte_cost(first)
    second_bytes = serialized_byte_cost(second)

    summary = write_typed_memory_artifact(
        (
            _scored(first, float(first_bytes)),
            _scored(second, float(second_bytes)),
        ),
        output=tmp_path / "memory.jsonl",
        byte_budget=first_bytes + second_bytes,
    )

    assert summary.selected_memory_ids == ("first", "other")


def test_no_write_is_counted_but_never_persisted(tmp_path: Path) -> None:
    no_write = _no_write("candidate-no-write")
    kept = _object("kept")
    output = tmp_path / "memory.jsonl"

    summary = write_typed_memory_artifact(
        (_scored(no_write, 1000.0), _scored(kept, 1.0)),
        output=output,
        byte_budget=serialized_byte_cost(kept),
    )

    assert summary.no_write_count == 1
    assert summary.writable_candidate_count == 1
    assert summary.selected_memory_ids == ("kept",)
    assert b"no_write" not in output.read_bytes()


def test_total_actual_bytes_never_exceed_budget(tmp_path: Path) -> None:
    first = _object("first")
    second = _object("second")
    budget = serialized_byte_cost(first)

    summary = write_typed_memory_artifact(
        (_scored(first, 2.0), _scored(second, 1.0)),
        output=tmp_path / "memory.jsonl",
        byte_budget=budget,
    )

    assert summary.actual_bytes <= budget
    assert (tmp_path / "memory.jsonl").stat().st_size == summary.actual_bytes
    assert summary.selected_count == 1


def test_artifact_validator_recounts_actual_window_bytes(tmp_path: Path) -> None:
    first = _object("first")
    second = _object("second").model_copy(
        update={
            "validity": ValidityInterval(start_time=31.0, end_time=32.0),
            "first_seen_time": 31.0,
            "last_seen_time": 32.0,
        },
    )
    output = tmp_path / "memory.jsonl"
    payload = canonical_jsonl_bytes(first) + canonical_jsonl_bytes(second)
    _ = output.write_bytes(payload)

    summary = validate_typed_memory_artifact(
        output,
        byte_budget_per_window=max(map(serialized_byte_cost, (first, second))),
    )

    assert summary.record_count == 2
    assert summary.actual_bytes == len(payload)
    assert summary.window_count == 2
    assert summary.max_window_bytes == max(map(serialized_byte_cost, (first, second)))


def test_artifact_validator_rejects_actual_window_budget_overflow(
    tmp_path: Path,
) -> None:
    first = _object("first")
    second = _object("second")
    output = tmp_path / "memory.jsonl"
    _ = output.write_bytes(
        canonical_jsonl_bytes(first) + canonical_jsonl_bytes(second),
    )

    with pytest.raises(TypedMemoryWriterError, match="window exceeds byte budget"):
        _ = validate_typed_memory_artifact(
            output,
            byte_budget_per_window=serialized_byte_cost(first),
        )


def test_artifact_validator_rejects_oversized_record_without_loading_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "oversized.jsonl"
    _ = output.write_bytes(b"{" + (b"x" * (1024 * 1024)) + b"}\n")

    with pytest.raises(TypedMemoryWriterError, match="row exceeds 1 MiB"):
        _ = validate_typed_memory_artifact(output)


@pytest.mark.parametrize("bad_artifact", ["no_write", "duplicate", "noncanonical"])
def test_artifact_validator_rejects_non_writer_output(
    tmp_path: Path,
    bad_artifact: str,
) -> None:
    record = _object("memory")
    payloads = {
        "no_write": canonical_jsonl_bytes(_no_write("memory")),
        "duplicate": canonical_jsonl_bytes(record) * 2,
        "noncanonical": f"{record.model_dump_json()}\n".encode(),
    }
    output = tmp_path / "memory.jsonl"
    _ = output.write_bytes(payloads[bad_artifact])

    with pytest.raises(TypedMemoryWriterError):
        _ = validate_typed_memory_artifact(output, byte_budget_per_window=10_000)


def test_artifact_validator_accepts_contextually_grounded_record(
    tmp_path: Path,
) -> None:
    sources = _grounding_sources()
    sensors = build_sensor_frame_manifest(sources)
    record = _object("grounded").model_copy(
        update={
            "last_seen_time": 1.0,
            "evidence_refs": ("video-1-frame-1",),
        },
    )
    output = tmp_path / "memory.jsonl"
    _ = output.write_bytes(canonical_jsonl_bytes(record))

    summary = validate_typed_memory_artifact(
        output,
        byte_budget_per_window=10_000,
        sources=sources,
        sensor_records=sensors,
    )

    assert summary.record_count == 1


@pytest.mark.parametrize(
    ("record", "error"),
    [
        (
            _object("unknown").model_copy(
                update={
                    "source_video_id": "unknown-video",
                    "evidence_refs": ("video-1-frame-1",),
                },
            ),
            "unknown source_video_id",
        ),
        (
            _object("out-of-bounds").model_copy(
                update={
                    "validity": ValidityInterval(start_time=11.0, end_time=12.0),
                    "first_seen_time": 11.0,
                    "last_seen_time": 12.0,
                    "evidence_refs": ("video-1-frame-1",),
                },
            ),
            "outside source bounds",
        ),
        (
            _object("missing-evidence"),
            "requires evidence_refs",
        ),
        (
            _object("fake-evidence").model_copy(
                update={"evidence_refs": ("not-a-selected-frame",)},
            ),
            "not selected sensor frames",
        ),
        (
            _object("other-video-evidence").model_copy(
                update={"evidence_refs": ("video-2-frame-1",)},
            ),
            "not selected sensor frames",
        ),
        (
            _object("backdated-evidence").model_copy(
                update={"evidence_refs": ("video-1-frame-before",)},
            ),
            "outside record observation interval",
        ),
        (
            _object("future-evidence").model_copy(
                update={"evidence_refs": ("video-1-frame-after",)},
            ),
            "outside record observation interval",
        ),
        (
            _object("unsupported-last-seen").model_copy(
                update={"evidence_refs": ("video-1-frame-1",)},
            ),
            "observation interval/count do not match evidence_refs",
        ),
    ],
    ids=(
        "unknown-video",
        "out-of-bounds-time",
        "missing-evidence",
        "fake-evidence",
        "other-video-evidence",
        "backdated-evidence",
        "future-evidence",
        "unsupported-last-seen",
    ),
)
def test_artifact_validator_rejects_contextually_ungrounded_record(
    tmp_path: Path,
    record: ObjectMemoryRecord,
    error: str,
) -> None:
    output = tmp_path / "memory.jsonl"
    _ = output.write_bytes(canonical_jsonl_bytes(record))

    with pytest.raises(TypedMemoryWriterError, match=error):
        _ = validate_typed_memory_artifact(
            output,
            byte_budget_per_window=10_000,
            sources=_grounding_sources(),
            sensor_records=build_sensor_frame_manifest(_grounding_sources()),
        )


@pytest.mark.parametrize("byte_budget", [0, -1])
def test_writer_rejects_nonpositive_budget(
    tmp_path: Path,
    byte_budget: int,
) -> None:
    with pytest.raises(TypedMemoryWriterError, match="byte_budget must be positive"):
        _ = write_typed_memory_artifact(
            (_scored(_object("memory"), 1.0),),
            output=tmp_path / "memory.jsonl",
            byte_budget=byte_budget,
        )


def test_writer_rejects_duplicates_before_replacing_output(tmp_path: Path) -> None:
    output = tmp_path / "memory.jsonl"
    _ = output.write_bytes(b"existing\n")
    record = _object("duplicate")

    with pytest.raises(TypedMemoryWriterError, match="duplicate memory_id"):
        _ = write_typed_memory_artifact(
            (_scored(record, 1.0), _scored(record, 2.0)),
            output=output,
            byte_budget=10_000,
        )

    assert output.read_bytes() == b"existing\n"


def test_writer_rejects_nonfinite_score_even_if_validation_was_bypassed(
    tmp_path: Path,
) -> None:
    candidate = ScoredMemoryCandidate.model_construct(
        record=_object("memory"),
        score=float("nan"),
    )

    with pytest.raises(TypedMemoryWriterError, match="score must be finite"):
        _ = write_typed_memory_artifact(
            (candidate,),
            output=tmp_path / "memory.jsonl",
            byte_budget=10_000,
        )


def _grounding_sources() -> tuple[SourceStreamExample, ...]:
    return (
        SourceStreamExample(
            video_id="video-1",
            start_time=0.0,
            end_time=10.0,
            frame_metadata=(
                FrameMetadata(
                    frame_ref="video-1-frame-before",
                    timestamp=0.5,
                    description="selected before interval",
                ),
                FrameMetadata(
                    frame_ref="video-1-frame-1",
                    timestamp=1.0,
                    description="selected",
                ),
                FrameMetadata(
                    frame_ref="video-1-frame-after",
                    timestamp=3.0,
                    description="selected after interval",
                ),
            ),
        ),
        SourceStreamExample(
            video_id="video-2",
            start_time=0.0,
            end_time=10.0,
            frame_metadata=(
                FrameMetadata(
                    frame_ref="video-2-frame-1",
                    timestamp=1.0,
                    description="selected",
                ),
            ),
        ),
    )
