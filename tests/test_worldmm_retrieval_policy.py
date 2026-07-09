from __future__ import annotations

from worldmm_smvqa.retrieval_protocols import WorldMMRetrievalPolicy
from worldmm_smvqa.schema import AnswerChoice, QuestionRequest


def _question(question: str, choices: tuple[str, ...] = ()) -> QuestionRequest:
    answer_choices = tuple(
        AnswerChoice(choice_id=chr(65 + index), text=choice, choice_ltype="text")
        for index, choice in enumerate(choices)
    )
    return QuestionRequest(
        question_id="q_test",
        video_id="fake_video_001",
        question=question,
        question_time=1850.0,
        answer_choices=answer_choices,
    )


def test_routes_spatial_first_when_q_fake_005_asks_where() -> None:
    # Given: the spatial fixture question relies on raw "Where" surviving routing.
    question = _question(
        "Where did the spatial trace focus during the mug placement?",
        ("beside the notebook", "CEREAL-FAKE", "lamp", "blue"),
    )

    # When: the WorldMM policy chooses a route from all stores.
    route = WorldMMRetrievalPolicy().route(
        question,
        available_stores=("episodic", "semantic", "visual", "spatial"),
    )

    # Then: spatial evidence is requested before other memories.
    assert route.store_order == ("spatial", "episodic", "semantic", "visual")
    assert route.hierarchy_depth == "record"
    assert route.reason == "location"


def test_routes_visual_first_when_ocr_or_frame_terms_appear() -> None:
    # Given: OCR terms can live in either question text or choices.
    question = _question(
        "Which note was visible on the counter?",
        ("OCR NOTE-7", "printed receipt", "plain label", "small sign"),
    )

    # When: the route is selected.
    route = WorldMMRetrievalPolicy().route(
        question,
        available_stores=("episodic", "semantic", "visual", "spatial"),
    )

    # Then: visual/OCR memory is searched first.
    assert route.store_order == ("visual", "episodic", "semantic", "spatial")
    assert route.reason == "visual"


def test_routes_episodic_first_when_event_terms_appear() -> None:
    # Given: event/time wording asks what happened rather than where.
    question = _question(
        "What happened after the cereal moved?",
        ("mug was placed", "drawer opened", "cereal poured", "cup picked up"),
    )

    # When: the route is selected.
    route = WorldMMRetrievalPolicy().route(
        question,
        available_stores=("episodic", "semantic", "visual", "spatial"),
    )

    # Then: episodic memory is searched first.
    assert route.store_order == ("episodic", "semantic", "visual", "spatial")
    assert route.reason == "event_time"


def test_omits_disabled_stores_from_policy_route() -> None:
    # Given: spatial wording but spatial store disabled by an ablation/CLI choice.
    question = _question("Where was the fake mug last seen?")

    # When: the policy filters to available stores.
    route = WorldMMRetrievalPolicy().route(
        question,
        available_stores=("episodic", "semantic"),
    )

    # Then: disabled stores never appear in the route.
    assert route.store_order == ("episodic", "semantic")
    assert route.reason == "location"
