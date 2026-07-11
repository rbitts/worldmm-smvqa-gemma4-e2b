from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from worldmm_smvqa.retrieval_types import (
    EvidenceItem,
    EvidencePack,
    RetrievalStore,
    RetrievalTrace,
)
from worldmm_smvqa.schema import AnswerChoice, QALabelExample, SourceStreamExample
from worldmm_smvqa.worldmm.spatial import build_object_anchors, derive_relations
from worldmm_smvqa.worldmm.spatial_diagnostics import (
    ExpectedSpatialRelation,
    InvalidEvidenceSpanError,
    RelationTuple,
    memory_recall_at_k,
    parse_evidence_span,
    relation_accuracy,
    relation_metric_accuracy,
    write_spatial_retrieval_diagnostics,
)
from worldmm_smvqa.worldmm.spatial_types import SpatialRelationRecord

FIXTURE_RELATIONS = Path("tests/fixtures/tiny_smvqa/expected_relations.jsonl")


def _label(
    question_id: str,
    *,
    evidence_list: tuple[str, ...],
) -> QALabelExample:
    return QALabelExample(
        question_id=question_id,
        video_id="fake_video_001",
        question="Where was the object?",
        question_time=50.0,
        answer_choices=(
            AnswerChoice(
                choice_id="A",
                text="beside the notebook",
                choice_ltype="place",
            ),
        ),
        answer="A",
        is_answerable=True,
        evidence_list=evidence_list,
        verification_score=1.0,
    )


def _pack(question_id: str, evidence: tuple[EvidenceItem, ...]) -> EvidencePack:
    return EvidencePack(
        question_id=question_id,
        video_id="fake_video_001",
        requested_stores=("episodic", "semantic", "visual", "spatial"),
        selected_stores=tuple(dict.fromkeys(item.source_store for item in evidence)),
        evidence_budget=6,
        evidence=evidence,
        causal_filtered_count=0,
        retrieval_trace=RetrievalTrace(
            protocols=("smvqa-video-rag", "egobutler", "worldmm"),
            eligible_shard_ids=("fake_video_001:0:1800",),
            selected_clip_ids=("fake_video_001:0:30",),
            policy_route="spatial-first",
            store_order=("spatial", "episodic", "semantic", "visual"),
            candidate_counts=(),
            causal_filtered_count=0,
            frame_ref_count=0,
        ),
    )


def _item(
    memory_id: str,
    *,
    store: RetrievalStore,
    start_time: float,
    end_time: float,
    video_id: str = "fake_video_001",
) -> EvidenceItem:
    return EvidenceItem(
        memory_id=memory_id,
        video_id=video_id,
        snippet="diagnostic evidence",
        frame_refs=(),
        source_store=store,
        start_time=start_time,
        end_time=end_time,
        retrieval_score=1.0,
    )


def test_parse_evidence_span_returns_typed_tuple() -> None:
    # Given: a fixture evidence span.
    raw_span = "fake_video_001:5:12:spatial"

    # When: the diagnostics parser reads it.
    span = parse_evidence_span(raw_span)

    # Then: it returns the typed tuple shape used by overlap matching.
    assert span == ("fake_video_001", 5.0, 12.0, "spatial")


def test_parse_evidence_span_accepts_fixture_label_stores() -> None:
    # Given: current tiny fixture evidence span stores.
    transcript_span = "fake_video_001:5:12:transcript"
    ocr_span = "fake_video_002:22:23:ocr"

    # When: the diagnostics parser reads them.
    transcript = parse_evidence_span(transcript_span)
    ocr = parse_evidence_span(ocr_span)

    # Then: raw label stores stay visible for per-store recall grouping.
    assert transcript == ("fake_video_001", 5.0, 12.0, "transcript")
    assert ocr == ("fake_video_002", 22.0, 23.0, "ocr")


def test_parse_evidence_span_rejects_malformed_span() -> None:
    # Given: an evidence span without the four required fields.
    raw_span = "fake_video_001:5:spatial"

    # When / Then: diagnostics raises a typed parse error.
    with pytest.raises(
        InvalidEvidenceSpanError,
        match="expected video:start:end:store",
    ):
        _ = parse_evidence_span(raw_span)


def test_relation_accuracy_uses_exact_relation_tuples() -> None:
    # Given: one exact relation hit plus one extra predicted relation.
    expected = (
        ExpectedSpatialRelation(
            video_id="spatial_video",
            subject="mug",
            relation="near",
            object="notebook",
            zone_id="zone_spatial_video_0_0",
        ),
    )
    predicted = (
        SpatialRelationRecord(
            memory_id="spatial_relation:spatial_video:mug:near:notebook:2.2",
            video_id="spatial_video",
            subject="mug",
            relation="near",
            object="notebook",
            zone_id="zone_spatial_video_0_0",
            start_time=2.2,
            end_time=3.0,
        ),
        SpatialRelationRecord(
            memory_id="spatial_relation:spatial_video:lamp:near:mug:4",
            video_id="spatial_video",
            subject="lamp",
            relation="near",
            object="mug",
            zone_id="zone_spatial_video_0_0",
            start_time=4.0,
            end_time=5.0,
        ),
    )

    # When: relation accuracy is computed.
    result = relation_accuracy(predicted, expected)

    # Then: exact tuple precision/recall/F1 are zero-division safe.
    assert result.true_positive == 1
    assert result.predicted == 2
    assert result.expected == 1
    assert result.precision == 0.5
    assert result.recall == 1.0
    assert result.f1 == pytest.approx(2 / 3)


def test_relation_metric_accuracy_checks_distance_and_delta_tolerance() -> None:
    # Given: exact relation labels plus metric geometry fields.
    expected = (
        ExpectedSpatialRelation(
            video_id="spatial_video",
            subject="mug",
            relation="left_of",
            object="notebook",
            zone_id="zone_spatial_video_0_0",
            distance_m=1.5,
            delta_x=1.0,
            delta_y=-1.0,
            delta_z=-0.5,
        ),
    )
    predicted = (
        SpatialRelationRecord(
            memory_id="spatial_relation:spatial_video:mug:left_of:notebook:2.2",
            video_id="spatial_video",
            subject="mug",
            relation="left_of",
            object="notebook",
            zone_id="zone_spatial_video_0_0",
            start_time=2.2,
            end_time=3.0,
            distance_m=1.55,
            delta_x=1.02,
            delta_y=-1.01,
            delta_z=-0.52,
        ),
    )

    # When: metric relation accuracy is computed with a realistic tolerance.
    result = relation_metric_accuracy(predicted, expected, distance_tolerance_m=0.1)

    # Then: metric agreement is counted, not just label equality.
    assert result.true_positive == 1
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_expected_relations_fixture_matches_todo_3_geometry() -> None:
    # Given: the current tiny source streams and expected relation fixture.
    sources = tuple(
        SourceStreamExample.model_validate_json(line)
        for line in Path("tests/fixtures/tiny_smvqa/sources.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    anchors = tuple(
        anchor for source in sources for anchor in build_object_anchors(source)
    )
    rows = tuple(
        ExpectedSpatialRelation.model_validate_json(line)
        for line in FIXTURE_RELATIONS.read_text(encoding="utf-8").splitlines()
    )

    # When: relations are derived from the source fixture.
    predicted = derive_relations(anchors)
    result = relation_accuracy(predicted, rows)

    # Then: checked-in expectations match actual tiny spatial output.
    assert result.predicted == len(predicted)
    assert result.expected == len(rows)
    if rows:
        assert result.true_positive > 0
        assert result.f1 > 0.0
    else:
        assert result.true_positive == 0
        assert result.f1 == 0.0


def test_relation_accuracy_empty_expected_is_zero_division_safe() -> None:
    # Given: no hand labels yet for a fixture.
    predicted = (
        RelationTuple(
            subject="mug",
            relation="near",
            object="notebook",
            zone_id="zone_spatial_video_0_0",
        ),
    )

    # When: diagnostics are computed.
    result = relation_accuracy(predicted, ())

    # Then: recall and F1 are defined, not NaN.
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_memory_recall_at_k_groups_hits_by_store_and_protocol() -> None:
    # Given: labels with dataset-level and memory-level evidence span stores.
    labels = (
        _label("q1", evidence_list=("fake_video_001:5:12:transcript",)),
        _label("q2", evidence_list=("fake_video_001:20:30:ocr",)),
        _label("q3", evidence_list=("fake_video_001:40:50:spatial",)),
        _label("q4", evidence_list=("fake_video_001:60:70:semantic",)),
    )
    packs = (
        _pack(
            "q1",
            (
                _item(
                    "episodic:fake_video_001:0:30",
                    store="episodic",
                    start_time=10.0,
                    end_time=14.0,
                ),
            ),
        ),
        _pack(
            "q2",
            (
                _item(
                    "visual:fake_video_001:20",
                    store="visual",
                    start_time=22.0,
                    end_time=23.0,
                ),
            ),
        ),
        _pack(
            "q3",
            (
                _item(
                    "spatial_anchor:fake_video_001:mug:40",
                    store="spatial",
                    start_time=40.0,
                    end_time=42.0,
                ),
            ),
        ),
        _pack(
            "q4",
            (
                _item(
                    "semantic:fake_video_001:1",
                    store="semantic",
                    start_time=70.0,
                    end_time=80.0,
                ),
            ),
        ),
    )

    # When: top-1 memory recall is computed.
    result = memory_recall_at_k(packs, labels, 1)

    # Then: interval overlap counts as a store/protocol hit.
    assert result.k == 1
    assert result.recall_at_k["transcript"] == 1.0
    assert result.recall_at_k["ocr"] == 1.0
    assert result.recall_at_k["spatial"] == 1.0
    assert result.recall_at_k["semantic"] == 0.0
    assert result.recall_at_k["episodic"] == 0.0
    assert result.recall_at_k["visual"] == 0.0
    assert result.protocol_recall_at_k["smvqa-video-rag"] == 0.75
    assert result.protocol_recall_at_k["egobutler"] == 0.75
    assert result.protocol_recall_at_k["worldmm"] == 0.75


def test_memory_recall_uses_item_video_and_accepts_frame_points() -> None:
    # Given: primary-video pack with evidence from another allowed video.
    labels = (
        _label("q_cross_video", evidence_list=("fake_video_002:20:30:visual",)),
    )
    packs = (
        _pack(
            "q_cross_video",
            (
                _item(
                    "visual:fake_video_002:frame_22",
                    store="visual",
                    video_id="fake_video_002",
                    start_time=22.0,
                    end_time=22.0,
                ),
            ),
        ),
    )

    # When: diagnostics score retrieval support.
    result = memory_recall_at_k(packs, labels, 1)

    # Then: cross-video point evidence is counted.
    assert result.recall_at_k["visual"] == pytest.approx(1.0)


def test_supermemory_evidence_span_accepts_any_retrieval_store() -> None:
    labels = (
        _label(
            "q_supermemory",
            evidence_list=("fake_video_001:5:12:supermemory",),
        ),
    )
    packs = (
        _pack(
            "q_supermemory",
            (
                _item(
                    "spatial:fake_video_001:mug",
                    store="spatial",
                    start_time=6.0,
                    end_time=8.0,
                ),
            ),
        ),
    )

    result = memory_recall_at_k(packs, labels, 1)

    assert result.recall_at_k["supermemory"] == pytest.approx(1.0)


def test_write_spatial_retrieval_diagnostics_emits_run_artifact(
    tmp_path: Path,
) -> None:
    # Given: one causal geometry-backed spatial retrieval.
    label = _label(
        "q_diag",
        evidence_list=("fake_video_001:40:50:spatial",),
    )
    pack = _pack(
        "q_diag",
        (
            _item(
                "spatial:anchor",
                store="spatial",
                start_time=42.0,
                end_time=44.0,
            ).model_copy(update={"geometry": {"x": 1.0}}),
        ),
    )
    evidence_path = tmp_path / "evidence.jsonl"
    label_path = tmp_path / "labels.jsonl"
    output = tmp_path / "diagnostics.json"
    _ = evidence_path.write_text(f"{pack.model_dump_json()}\n", encoding="utf-8")
    _ = label_path.write_text(f"{label.model_dump_json()}\n", encoding="utf-8")

    # When: production diagnostics artifact is written.
    write_spatial_retrieval_diagnostics(evidence_path, label_path, output)

    # Then: usefulness and causal counters are concrete.
    payload = cast(
        "dict[str, object]",
        json.loads(output.read_text(encoding="utf-8")),
    )
    memory_recall = cast(
        "dict[str, dict[str, dict[str, float]]]",
        payload["memory_recall"],
    )
    assert payload["spatial_selected_packs"] == 1
    assert payload["geometry_evidence_items"] == 1
    assert payload["causal_violation_count"] == 0
    assert memory_recall["1"]["recall_at_k"]["spatial"] == 1.0
