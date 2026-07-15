from __future__ import annotations

from pathlib import Path

from worldmm_smvqa.retrieval_types import EvidencePack, RetrievalTrace
from worldmm_smvqa.schema import (
    AnswerChoice,
    FrameMetadata,
    QuestionRequest,
    SourceStreamExample,
)
from worldmm_smvqa.video_frames import sample_video_frames


def test_samples_32_uniform_frames_from_selected_shard_without_future_refs() -> None:
    # Given: one selected clip inside a pre-question 30m shard with many frames.
    frames = tuple(
        FrameMetadata(
            frame_ref=f"video_uniform_frame_{index:04d}",
            timestamp=float(index),
            description=f"frame {index}",
        )
        for index in range(64)
    )
    source = SourceStreamExample(
        video_id="video_uniform",
        start_time=0.0,
        end_time=1900.0,
        frame_refs=tuple(frame.frame_ref for frame in frames),
        frame_metadata=(
            *frames,
            FrameMetadata(
                frame_ref="video_uniform_frame_1900",
                timestamp=1900.0,
                description="future frame",
            ),
        ),
    )
    question = _question("video_uniform", question_time=1801.0)
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("episodic",),
        selected_stores=("episodic",),
        evidence_budget=0,
        evidence=(),
        causal_filtered_count=0,
        retrieval_trace=RetrievalTrace(
            protocols=("smvqa-video-rag", "egobutler", "worldmm"),
            eligible_shard_ids=("video_uniform:0:1800:shard_30m",),
            selected_clip_ids=("video_uniform:30:60:clip_30s",),
            policy_route="balanced",
            store_order=("episodic",),
            candidate_counts=(),
            causal_filtered_count=0,
            frame_ref_count=0,
        ),
    )

    # When: official-parity frames are sampled.
    sampled = sample_video_frames(
        (source,),
        question,
        pack,
        frame_root=Path("/frames"),
        max_frames=32,
    )

    # Then: exactly 32 refs come uniformly from the selected shard, not future time.
    assert len(sampled) == 32
    assert sampled[0].frame_ref == "video_uniform_frame_0000"
    assert sampled[-1].frame_ref == "video_uniform_frame_0063"
    assert all(frame.timestamp < question.question_time for frame in sampled)
    assert sampled[0].path == Path("/frames/video_uniform/video_uniform_frame_0000.jpg")


def test_samples_all_available_frames_when_less_than_cap() -> None:
    # Given: sparse frame metadata inside one eligible shard.
    source = SourceStreamExample(
        video_id="video_sparse",
        start_time=0.0,
        end_time=90.0,
        frame_metadata=(
            FrameMetadata(frame_ref="f0", timestamp=0.0, description="zero"),
            FrameMetadata(frame_ref="f30", timestamp=30.0, description="thirty"),
        ),
    )
    question = _question("video_sparse", question_time=100.0)
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("visual",),
        selected_stores=("visual",),
        evidence_budget=0,
        evidence=(),
        causal_filtered_count=0,
        retrieval_trace=RetrievalTrace(
            protocols=("smvqa-video-rag",),
            eligible_shard_ids=("video_sparse:0:90:shard_30m",),
            selected_clip_ids=(),
            policy_route="visual",
            store_order=("visual",),
            candidate_counts=(),
            causal_filtered_count=0,
            frame_ref_count=0,
        ),
    )

    # When: sampling below the cap.
    sampled = sample_video_frames(
        (source,),
        question,
        pack,
        frame_root=None,
        max_frames=32,
    )

    # Then: no synthetic frames are invented.
    assert tuple(frame.frame_ref for frame in sampled) == ("f0", "f30")


def test_rejects_trace_shard_outside_question_video_scope() -> None:
    # Given: a stale or malformed pack points frame sampling at an unallowed video.
    source = SourceStreamExample(
        video_id="outside",
        start_time=0.0,
        end_time=10.0,
        frame_metadata=(
            FrameMetadata(frame_ref="f", timestamp=1.0, description="outside"),
        ),
    )
    question = QuestionRequest(
        question_id="q",
        video_id="primary",
        video_ids=("primary", "allowed"),
        question="Where is it?",
        question_time=5.0,
        answer_choices=(),
    )
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("visual",),
        selected_stores=("visual",),
        evidence_budget=0,
        evidence=(),
        causal_filtered_count=0,
        retrieval_trace=RetrievalTrace(
            protocols=("smvqa-video-rag",),
            eligible_shard_ids=("outside:0:10:shard_30m",),
            selected_clip_ids=(),
            policy_route="visual",
            store_order=("visual",),
            candidate_counts=(),
            causal_filtered_count=0,
            frame_ref_count=0,
        ),
    )

    # When / Then: frame sampling does not cross the question's allowed video scope.
    assert (
        sample_video_frames((source,), question, pack, frame_root=Path("/frames")) == ()
    )


def test_samples_frames_across_selected_videos() -> None:
    # Given: selected clips map to one eligible shard in each allowed video.
    sources = (
        SourceStreamExample(
            video_id="primary",
            start_time=0.0,
            end_time=30.0,
            frame_metadata=tuple(
                FrameMetadata(
                    frame_ref=f"primary-{index}",
                    timestamp=float(index),
                    description="primary",
                )
                for index in range(6)
            ),
        ),
        SourceStreamExample(
            video_id="support",
            start_time=0.0,
            end_time=30.0,
            frame_metadata=(
                FrameMetadata(
                    frame_ref="support-0",
                    timestamp=0.0,
                    description="support",
                ),
            ),
        ),
    )
    question = QuestionRequest(
        question_id="q-multi",
        video_id="primary",
        video_ids=("primary", "support"),
        question="Where is it?",
        question_time=31.0,
        answer_choices=(),
    )
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("visual",),
        selected_stores=("visual",),
        evidence_budget=0,
        evidence=(),
        causal_filtered_count=0,
        retrieval_trace=RetrievalTrace(
            protocols=("smvqa-video-rag",),
            eligible_shard_ids=(
                "primary:0:30:shard_30m",
                "support:0:30:shard_30m",
            ),
            selected_clip_ids=(
                "primary:0:30:clip_30s",
                "support:0:30:clip_30s",
            ),
            policy_route="visual",
            store_order=("visual",),
            candidate_counts=(),
            causal_filtered_count=0,
            frame_ref_count=0,
        ),
    )

    # When: the frame cap is smaller than the combined candidate set.
    sampled = sample_video_frames(
        sources,
        question,
        pack,
        frame_root=Path("/frames"),
        max_frames=4,
    )

    # Then: sparse support video remains represented.
    assert len(sampled) == 4
    assert "support-0" in {frame.frame_ref for frame in sampled}
    support = next(frame for frame in sampled if frame.frame_ref == "support-0")
    assert support.path == Path("/frames/support/support-0.jpg")


def test_frame_ref_cannot_escape_frame_root() -> None:
    source = SourceStreamExample(
        video_id="video",
        start_time=0.0,
        end_time=30.0,
        frame_metadata=(
            FrameMetadata(
                frame_ref="../../outside",
                timestamp=1.0,
                description="unsafe",
            ),
        ),
    )
    question = _question("video", question_time=31.0)
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("visual",),
        selected_stores=("visual",),
        evidence_budget=0,
        evidence=(),
        causal_filtered_count=0,
        retrieval_trace=RetrievalTrace(
            protocols=("smvqa-video-rag",),
            eligible_shard_ids=("video:0:30:shard_30m",),
            selected_clip_ids=(),
            policy_route="visual",
            store_order=("visual",),
            candidate_counts=(),
            causal_filtered_count=0,
            frame_ref_count=0,
        ),
    )

    sampled = sample_video_frames(
        (source,),
        question,
        pack,
        frame_root=Path("/frames"),
    )

    assert len(sampled) == 1
    assert sampled[0].path is None


def _question(video_id: str, *, question_time: float) -> QuestionRequest:
    return QuestionRequest(
        question_id="q",
        video_id=video_id,
        question="Where is it?",
        question_time=question_time,
        answer_choices=(
            AnswerChoice(choice_id="A", text="a", choice_ltype="place"),
            AnswerChoice(choice_id="B", text="b", choice_ltype="place"),
            AnswerChoice(choice_id="C", text="c", choice_ltype="place"),
            AnswerChoice(choice_id="D", text="d", choice_ltype="place"),
        ),
    )
