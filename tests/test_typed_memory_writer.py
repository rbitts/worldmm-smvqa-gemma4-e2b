from __future__ import annotations

import json
from pathlib import Path

import pytest

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
            covariance_xyz=((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
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
