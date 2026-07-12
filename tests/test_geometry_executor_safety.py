from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from worldmm_smvqa.fixtures import read_fixture_questions
from worldmm_smvqa.retrieval_types import EvidencePack
from worldmm_smvqa.schema import QuestionRequest
from worldmm_smvqa.worldmm.geometry_executor import (
    GeometryEntityFact,
    GeometryQuery,
    execute_geometry,
    geometry_proofs_for_question,
    plan_geometry_query,
)
from worldmm_smvqa.worldmm.typed_memory import (
    EventGeometry,
    EventMemoryRecord,
    ObjectGeometry,
    ObjectMemoryRecord,
    SpatialUncertainty,
    ValidityInterval,
)


def _entity(entity_id: str, x: float) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "label": "mug",
        "x": x,
        "y": 0.0,
        "z": 0.0,
        "coordinate_frame": "room:1",
        "uncertainty_m": 0.1,
        "provenance": "observed",
        "evidence_refs": [f"memory:{entity_id}"],
    }


def _typed_object(name: str, label: str, x: float) -> ObjectMemoryRecord:
    return ObjectMemoryRecord(
        memory_id=f"memory:{name}",
        source_video_id="video:1",
        entity_id=f"entity:{name}",
        instance_id=f"instance:{name}",
        local_frame_id="room:1",
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
        evidence_refs=(f"frame:{name}",),
        geometry=ObjectGeometry(
            centroid=(x, 0.0, 0.0),
            extent=(0.2, 0.2, 0.2),
        ),
        semantic_label=label,
    )


def _typed_event(
    kind: str,
    *,
    end_time: float = 3.0,
) -> EventMemoryRecord:
    geometry_by_kind = {
        "appeared": EventGeometry(after_position=(0.0, 0.0, 0.0)),
        "disappeared": EventGeometry(before_position=(0.0, 0.0, 0.0)),
        "moved": EventGeometry(
            before_position=(0.0, 0.0, 0.0),
            after_position=(1.0, 0.0, 0.0),
        ),
        "opened": EventGeometry(),
    }
    return EventMemoryRecord.model_validate(
        {
            "memory_id": f"memory:event:{kind}",
            "source_video_id": "video:1",
            "entity_id": f"event:{kind}",
            "instance_id": f"event-instance:{kind}",
            "local_frame_id": "room:1",
            "geometry_uncertainty": SpatialUncertainty(
                covariance_xyz=(
                    (0.01, 0.0, 0.0),
                    (0.0, 0.01, 0.0),
                    (0.0, 0.0, 0.01),
                ),
                standard_deviation_m=0.1,
            ),
            "validity": ValidityInterval(
                start_time=end_time,
                end_time=end_time,
            ),
            "first_seen_time": end_time,
            "last_seen_time": end_time,
            "observation_count": 1,
            "confidence": 0.9,
            "provenance": "observed",
            "evidence_refs": (f"frame:event:{kind}",),
            "geometry": geometry_by_kind[kind],
            "event_kind": kind,
            "involved_entity_ids": ("entity:mug",),
        },
    )


def test_count_requires_explicit_complete_entity_index() -> None:
    top_k_records = (_entity("mug:1", 0.0), _entity("mug:2", 1.0))
    incomplete_query = GeometryQuery(
        operation="count",
        coordinate_frame="room:1",
        entity_label="mug",
    )

    incomplete = execute_geometry(top_k_records, incomplete_query)
    complete = execute_geometry(
        top_k_records,
        incomplete_query.model_copy(update={"entity_index_complete": True}),
    )

    assert not incomplete.answerable
    assert incomplete.value is None
    assert incomplete.reason == "count requires a complete entity index"
    assert complete.answerable
    assert complete.value == 2


def test_last_seen_requires_explicit_complete_entity_index() -> None:
    records = ({**_entity("mug:1", 0.0), "last_seen_time": 2.0},)
    incomplete_query = GeometryQuery(
        operation="last_seen",
        coordinate_frame="video-time",
        subject="mug:1",
    )

    incomplete = execute_geometry(records, incomplete_query)
    complete = execute_geometry(
        records,
        incomplete_query.model_copy(update={"entity_index_complete": True}),
    )

    assert not incomplete.answerable
    assert incomplete.reason == "last-seen requires a complete entity index"
    assert complete.answerable
    assert complete.value == 2.0


def test_production_planner_never_certifies_count_completeness() -> None:
    query = plan_geometry_query(
        "How many mugs were visible?",
        (_entity("mug:1", 0.0),),
        coordinate_frame="room:1",
    )

    assert query is not None
    assert not query.entity_index_complete


def test_proof_hash_covers_every_behavior_option() -> None:
    records = (_entity("mug:1", 0.0), _entity("mug:2", 1.0))
    baseline = GeometryQuery(
        operation="relative_direction",
        coordinate_frame="room:1",
        subject="mug:1",
        object="mug:2",
        wearer_yaw_degrees=0.0,
    )
    queries = (
        baseline,
        baseline.model_copy(update={"wearer_yaw_degrees": 1.0}),
        baseline.model_copy(update={"wearer_yaw_uncertainty_degrees": 1.0}),
        baseline.model_copy(update={"near_threshold_m": 2.0}),
        baseline.model_copy(update={"max_uncertainty_m": 0.6}),
        baseline.model_copy(update={"entity_index_complete": True}),
    )

    proof_ids = {execute_geometry(records, query).proof_id for query in queries}

    assert len(proof_ids) == len(queries)


def test_direct_nested_typed_objects_are_normalized() -> None:
    proof = execute_geometry(
        (
            _typed_object("mug", "mug", 0.0),
            _typed_object("notebook", "notebook", 3.0),
        ),
        GeometryQuery(
            operation="distance",
            coordinate_frame="room:1",
            subject="entity:mug",
            object="entity:notebook",
        ),
    )

    assert proof.answerable
    assert proof.value == pytest.approx(3.0)
    assert proof.uncertainty == pytest.approx(2**0.5 * 0.1)
    assert proof.evidence_refs == (
        "frame:mug",
        "frame:notebook",
        "memory:mug",
        "memory:notebook",
    )


def test_missing_coordinate_frame_blocks_metric_ops_but_not_count() -> None:
    missing_frame = _entity("mug:1", 0.0)
    del missing_frame["coordinate_frame"]
    records = (missing_frame, _entity("desk:1", 1.0))

    distance = execute_geometry(
        records,
        GeometryQuery(
            operation="distance",
            coordinate_frame="room:1",
            subject="mug:1",
            object="desk:1",
        ),
    )
    count = execute_geometry(
        (missing_frame,),
        GeometryQuery(
            operation="count",
            coordinate_frame="room:1",
            entity_label="mug",
            entity_index_complete=True,
        ),
    )

    assert not distance.answerable
    assert distance.reason == "coordinate frame mismatch: mug:1"
    assert count.answerable
    assert count.value == 1


@pytest.mark.parametrize(
    "conflicting_update",
    [{"x": 2.0}, {"coordinate_frame": "room:2"}],
)
def test_conflicting_latest_entity_records_abstain_order_independently(
    conflicting_update: dict[str, object],
) -> None:
    first = {**_entity("mug:1", 0.0), "last_seen_time": 5.0}
    conflicting = {**first, **conflicting_update}
    desk = {**_entity("desk:1", 4.0), "label": "desk"}
    query = GeometryQuery(
        operation="distance",
        coordinate_frame="room:1",
        subject="mug:1",
        object="desk:1",
    )

    forward = execute_geometry((first, conflicting, desk), query)
    reversed_ = execute_geometry((conflicting, first, desk), query)

    assert not forward.answerable
    assert forward.reason == "conflicting latest records: mug:1"
    assert forward.subject_entity_id == "mug:1"
    assert forward.object_entity_id == "desk:1"
    assert reversed_.reason == forward.reason
    assert reversed_.proof_id == forward.proof_id


def test_time_uncertainty_must_be_finite() -> None:
    with pytest.raises(ValidationError, match="geometry values must be finite"):
        _ = GeometryEntityFact(
            entity_id="mug:1",
            label="mug",
            provenance="observed",
            evidence_refs=(),
            time_uncertainty_s=float("inf"),
        )


def test_derived_metric_overflow_abstains() -> None:
    proof = execute_geometry(
        (_entity("left", 1e308), _entity("right", -1e308)),
        GeometryQuery(
            operation="distance",
            coordinate_frame="room:1",
            subject="left",
            object="right",
        ),
    )

    assert not proof.answerable
    assert proof.reason == "derived metric geometry is not finite"


def test_same_local_entity_id_in_different_videos_never_collapses() -> None:
    first = GeometryEntityFact.model_validate(
        {**_entity("cup-1", 0.0), "source_video_id": "video-1"},
    )
    second = GeometryEntityFact.model_validate(
        {**_entity("cup-1", 1.0), "source_video_id": "video-2"},
    )
    proof = execute_geometry(
        (first, second),
        GeometryQuery(
            operation="count",
            coordinate_frame="room:1",
            entity_label="mug",
            entity_index_complete=True,
        ),
    )

    assert proof.answerable
    assert proof.value == 2


def test_count_does_not_require_metric_frame_compatibility() -> None:
    first = _entity("cup-1", 0.0)
    second = {**_entity("cup-2", 1.0), "coordinate_frame": "room:2"}
    proof = execute_geometry(
        (first, second),
        GeometryQuery(
            operation="count",
            coordinate_frame="non-metric-count-scope",
            entity_label="mug",
            entity_index_complete=True,
        ),
    )

    assert proof.answerable
    assert proof.value == 2


def test_count_conflicting_latest_state_abstains_without_type_error() -> None:
    first = {**_entity("cup-1", 0.0), "last_seen_time": 5.0}
    conflicting = {**first, "x": 1.0}
    proof = execute_geometry(
        (first, conflicting),
        GeometryQuery(
            operation="count",
            coordinate_frame="room:1",
            entity_label="mug",
            entity_index_complete=True,
        ),
    )

    assert not proof.answerable
    assert proof.reason == "conflicting latest records: cup-1"


def test_recognized_geometry_intent_emits_unanswerable_planning_proof() -> None:
    question = read_fixture_questions(Path("tests/fixtures/tiny_smvqa"))[4]
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("spatial",),
        selected_stores=(),
        evidence_budget=1,
        evidence=(),
        causal_filtered_count=0,
    )

    proofs = geometry_proofs_for_question(question, pack)

    assert len(proofs) == 1
    assert not proofs[0].answerable
    assert proofs[0].reason == "geometry query planning failed or selector is ambiguous"


def test_last_seen_where_never_cites_timestamp_as_location() -> None:
    question = read_fixture_questions(Path("tests/fixtures/tiny_smvqa"))[0].model_copy(
        update={"question": "Where was the mug last seen?"},
    )
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("spatial",),
        selected_stores=(),
        evidence_budget=1,
        evidence=(),
        causal_filtered_count=0,
    )

    proof = geometry_proofs_for_question(
        question,
        pack,
        spatial_records=(),
    )[0]

    assert not proof.answerable
    assert proof.reason == "last-seen location proof is not implemented"


@pytest.mark.parametrize(
    "question_text",
    [
        "Which room was the mug last seen in?",
        "What location last saw the mug?",
        "Where and when was the mug last seen?",
    ],
)
def test_last_seen_location_variants_never_return_timestamp(
    question_text: str,
) -> None:
    question = QuestionRequest(
        question_id="q-last-seen-location",
        video_id="video:1",
        question=question_text,
        question_time=3.0,
        answer_choices=(),
    )
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("spatial",),
        selected_stores=(),
        evidence_budget=1,
        evidence=(),
        causal_filtered_count=0,
    )

    proof = geometry_proofs_for_question(
        question,
        pack,
        spatial_records=(_typed_object("mug", "mug", 0.0),),
    )[0]

    assert not proof.answerable
    assert proof.value is None
    assert proof.reason == "last-seen location proof is not implemented"


def test_budgeted_spatial_records_do_not_certify_count_completeness() -> None:
    question = QuestionRequest(
        question_id="q-budgeted-count",
        video_id="video:1",
        question="How many mugs were visible?",
        question_time=3.0,
        answer_choices=(),
    )
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("spatial",),
        selected_stores=(),
        evidence_budget=1,
        evidence=(),
        causal_filtered_count=0,
    )

    # A second matching object may have been dropped by the byte-budgeted writer.
    proof = geometry_proofs_for_question(
        question,
        pack,
        spatial_records=(_typed_object("mug", "mug", 0.0),),
    )[0]

    assert not proof.answerable
    assert proof.reason == "count requires a complete entity index"


def test_budgeted_spatial_records_do_not_certify_last_seen_completeness() -> None:
    question = QuestionRequest(
        question_id="q-budgeted-last-seen",
        video_id="video:1",
        question="When was the mug last seen?",
        question_time=4.0,
        answer_choices=(),
    )
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("spatial",),
        selected_stores=(),
        evidence_budget=1,
        evidence=(),
        causal_filtered_count=0,
    )

    # A newer state or change event may have been dropped by the writer.
    proof = geometry_proofs_for_question(
        question,
        pack,
        spatial_records=(_typed_object("mug", "mug", 0.0),),
    )[0]

    assert not proof.answerable
    assert proof.reason == "last-seen requires a complete entity index"


def test_budgeted_spatial_records_support_explicit_id_pair_geometry() -> None:
    question = QuestionRequest(
        question_id="q-budgeted-distance",
        video_id="video:1",
        question="How far was entity:mug from entity:notebook?",
        question_time=3.0,
        answer_choices=(),
    )
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("spatial",),
        selected_stores=(),
        evidence_budget=1,
        evidence=(),
        causal_filtered_count=0,
    )

    proof = geometry_proofs_for_question(
        question,
        pack,
        spatial_records=(
            _typed_object("mug", "mug", 0.0),
            _typed_object("notebook", "notebook", 3.0),
        ),
    )[0]

    assert proof.answerable
    assert proof.value == pytest.approx(3.0)


def test_top_k_label_pair_abstains_when_duplicate_may_have_been_dropped() -> None:
    question = QuestionRequest(
        question_id="q-budgeted-label-distance",
        video_id="video:1",
        question="How far was the mug from the notebook?",
        question_time=3.0,
        answer_choices=(),
    )
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("spatial",),
        selected_stores=(),
        evidence_budget=1,
        evidence=(),
        causal_filtered_count=0,
    )

    # A second `mug` may be absent from this bounded persisted/retrieved set.
    proof = geometry_proofs_for_question(
        question,
        pack,
        spatial_records=(
            _typed_object("mug", "mug", 0.0),
            _typed_object("notebook", "notebook", 3.0),
        ),
    )[0]

    assert not proof.answerable
    assert proof.reason == "geometry query planning failed or selector is ambiguous"


@pytest.mark.parametrize("event_kind", ["appeared", "disappeared", "moved"])
def test_count_abstains_when_typed_change_event_requires_consolidation(
    event_kind: str,
) -> None:
    question = QuestionRequest(
        question_id=f"q-count-{event_kind}",
        video_id="video:1",
        question="How many mugs were visible?",
        question_time=4.0,
        answer_choices=(),
    )
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("spatial",),
        selected_stores=(),
        evidence_budget=1,
        evidence=(),
        causal_filtered_count=0,
    )

    proof = geometry_proofs_for_question(
        question,
        pack,
        spatial_records=(
            _typed_object("mug", "mug", 0.0),
            _typed_event(event_kind),
        ),
        entity_index_complete=True,
    )[0]

    assert not proof.answerable
    assert proof.reason == "dynamic count requires consolidated object state"
    assert proof.entity_ids == ("entity:mug",)


def test_non_cardinality_event_does_not_pollute_object_count() -> None:
    question = QuestionRequest(
        question_id="q-count-opened",
        video_id="video:1",
        question="How many mugs were visible?",
        question_time=4.0,
        answer_choices=(),
    )
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("spatial",),
        selected_stores=(),
        evidence_budget=1,
        evidence=(),
        causal_filtered_count=0,
    )

    proof = geometry_proofs_for_question(
        question,
        pack,
        spatial_records=(
            _typed_object("mug", "mug", 0.0),
            _typed_event("opened"),
        ),
        entity_index_complete=True,
    )[0]

    assert proof.answerable
    assert proof.value == 1
    assert proof.entity_ids == ("entity:mug",)


def test_last_seen_abstains_when_newer_change_event_outdates_object_state() -> None:
    question = QuestionRequest(
        question_id="q-last-seen-stale",
        video_id="video:1",
        question="When was the mug last seen?",
        question_time=4.0,
        answer_choices=(),
    )
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("spatial",),
        selected_stores=(),
        evidence_budget=1,
        evidence=(),
        causal_filtered_count=0,
    )

    proof = geometry_proofs_for_question(
        question,
        pack,
        spatial_records=(
            _typed_object("mug", "mug", 0.0),
            _typed_event("moved", end_time=3.0),
        ),
        entity_index_complete=True,
    )[0]

    assert not proof.answerable
    assert proof.reason == (
        "last-seen state is stale relative to a typed change event"
    )
    assert proof.subject_entity_id == "entity:mug"


def test_last_seen_accepts_object_state_consolidated_at_event_time() -> None:
    question = QuestionRequest(
        question_id="q-last-seen-consolidated",
        video_id="video:1",
        question="When was the mug last seen?",
        question_time=3.0,
        answer_choices=(),
    )
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("spatial",),
        selected_stores=(),
        evidence_budget=1,
        evidence=(),
        causal_filtered_count=0,
    )
    object_state = _typed_object("mug", "mug", 0.0).model_copy(
        update={
            "validity": ValidityInterval(start_time=1.0, end_time=3.0),
            "last_seen_time": 3.0,
        },
    )

    proof = geometry_proofs_for_question(
        question,
        pack,
        spatial_records=(
            object_state,
            _typed_event("moved", end_time=3.0),
        ),
        entity_index_complete=True,
    )[0]

    assert proof.answerable
    assert proof.value == 3.0
