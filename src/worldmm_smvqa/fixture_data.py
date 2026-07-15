from __future__ import annotations

from dataclasses import dataclass

from worldmm_smvqa.schema import (
    AnswerChoice,
    FrameMetadata,
    GazeSample,
    ObjectMetadata,
    OCRMetadata,
    PoseSample,
    QALabelExample,
    SourceStreamExample,
    TranscriptSpan,
)


@dataclass(frozen=True, slots=True)
class LabelSeed:
    question_id: str
    video_id: str
    question: str
    question_time: float
    answer: str
    evidence: str | None
    is_answerable: bool = True


def tiny_fixture_examples() -> tuple[
    tuple[SourceStreamExample, ...],
    tuple[QALabelExample, ...],
]:
    sources = (
        SourceStreamExample(
            video_id="fake_video_001",
            start_time=0.0,
            end_time=1900.0,
            transcript="A staged desk routine with a mug, notebook, and lamp.",
            transcript_spans=(
                TranscriptSpan(
                    start_time=5.0,
                    end_time=12.0,
                    text="The fake mug is placed beside the notebook.",
                ),
                TranscriptSpan(
                    start_time=70.0,
                    end_time=79.0,
                    text="The lamp is switched on for the staged scene.",
                ),
            ),
            captions=(
                "Synthetic desk scene with mug and notebook.",
                "Synthetic close-up of a lamp on a table.",
            ),
            ocr=("NOTE-42", "TODO-SAMPLE"),
            ocr_entries=(
                OCRMetadata(
                    start_time=8.0,
                    end_time=9.0,
                    text="NOTE-42",
                    frame_ref="fake_video_001_frame_0008",
                ),
            ),
            objects=("mug", "notebook", "lamp"),
            object_detections=(
                ObjectMetadata(
                    start_time=6.0,
                    end_time=11.0,
                    label="mug",
                    confidence=0.91,
                    instance_id="mug-1",
                    x=0.5,
                    y=1.5,
                    z=1.0,
                ),
                ObjectMetadata(
                    start_time=6.0,
                    end_time=11.0,
                    label="notebook",
                    confidence=0.9,
                    instance_id="notebook-1",
                    x=1.0,
                    y=1.5,
                    z=1.0,
                ),
                ObjectMetadata(
                    start_time=72.0,
                    end_time=78.0,
                    label="lamp",
                    confidence=0.88,
                ),
            ),
            pose_samples=(
                PoseSample(timestamp=6.0, x=0.2, y=1.1, z=1.5, yaw_degrees=25.0),
                PoseSample(timestamp=12.0, x=0.3, y=1.2, z=1.5, yaw_degrees=27.0),
                PoseSample(timestamp=1800.0, x=1.0, y=2.0, z=1.5, yaw_degrees=90.0),
                PoseSample(timestamp=1900.0, x=1.1, y=2.1, z=1.5, yaw_degrees=91.0),
            ),
            gaze_samples=(
                GazeSample(timestamp=7.0, x=0.4, y=1.4, z=1.0),
                GazeSample(timestamp=12.0, x=0.5, y=1.5, z=1.0),
                GazeSample(timestamp=1850.0, x=1.4, y=2.4, z=1.0),
                GazeSample(timestamp=1900.0, x=1.5, y=2.5, z=1.0),
            ),
            frame_refs=(
                "fake_video_001_frame_0008",
                "fake_video_001_frame_0072",
                "fake_video_001_frame_1800",
                "fake_video_001_frame_1900",
            ),
            frame_metadata=(
                FrameMetadata(
                    frame_ref="fake_video_001_frame_0008",
                    timestamp=8.0,
                    description="Fake desk frame with mug and notebook.",
                ),
                FrameMetadata(
                    frame_ref="fake_video_001_frame_0072",
                    timestamp=72.0,
                    description="Fake desk frame with lamp switched on.",
                ),
                FrameMetadata(
                    frame_ref="fake_video_001_frame_1800",
                    timestamp=1800.0,
                    description="Fake long-horizon desk frame before shard boundary.",
                ),
                FrameMetadata(
                    frame_ref="fake_video_001_frame_1900",
                    timestamp=1900.0,
                    description="Fake long-horizon desk frame after shard boundary.",
                ),
            ),
        ),
        SourceStreamExample(
            video_id="fake_video_002",
            start_time=0.0,
            end_time=210.0,
            transcript="A staged kitchen routine with a box, bowl, and magnet.",
            transcript_spans=(
                TranscriptSpan(
                    start_time=20.0,
                    end_time=28.0,
                    text="A fake cereal box is moved near the bowl.",
                ),
                TranscriptSpan(
                    start_time=130.0,
                    end_time=138.0,
                    text="A synthetic blue magnet is visible on the fridge.",
                ),
            ),
            captions=(
                "Synthetic kitchen counter with box and bowl.",
                "Synthetic fridge view with blue magnet.",
            ),
            ocr=("CEREAL-FAKE", "MAGNET-BLUE"),
            ocr_entries=(
                OCRMetadata(
                    start_time=22.0,
                    end_time=23.0,
                    text="CEREAL-FAKE",
                    frame_ref="fake_video_002_frame_0022",
                ),
            ),
            objects=("box", "bowl", "magnet"),
            object_detections=(
                ObjectMetadata(
                    start_time=21.0,
                    end_time=27.0,
                    label="box",
                    confidence=0.9,
                ),
                ObjectMetadata(
                    start_time=132.0,
                    end_time=137.0,
                    label="magnet",
                    confidence=0.86,
                ),
            ),
            pose_samples=(
                PoseSample(timestamp=21.0, x=0.1, y=0.7, z=1.5, yaw_degrees=10.0),
                PoseSample(timestamp=27.0, x=0.2, y=0.8, z=1.5, yaw_degrees=12.0),
                PoseSample(timestamp=132.0, x=2.0, y=1.0, z=1.5, yaw_degrees=55.0),
                PoseSample(timestamp=137.0, x=2.1, y=1.1, z=1.5, yaw_degrees=56.0),
            ),
            gaze_samples=(
                GazeSample(timestamp=22.0, x=0.3, y=0.9, z=1.0),
                GazeSample(timestamp=134.0, x=2.3, y=1.3, z=1.0),
            ),
            frame_refs=("fake_video_002_frame_0022", "fake_video_002_frame_0132"),
            frame_metadata=(
                FrameMetadata(
                    frame_ref="fake_video_002_frame_0022",
                    timestamp=22.0,
                    description="Fake counter frame with cereal box.",
                ),
                FrameMetadata(
                    frame_ref="fake_video_002_frame_0132",
                    timestamp=132.0,
                    description="Fake fridge frame with blue magnet.",
                ),
            ),
        ),
    )
    labels = (
        _label(
            LabelSeed(
                question_id="q_fake_001",
                video_id="fake_video_001",
                question="Where is the fake mug placed?",
                question_time=45.0,
                answer="A",
                evidence="fake_video_001:5:12:transcript",
            )
        ),
        _label(
            LabelSeed(
                question_id="q_fake_002",
                video_id="fake_video_001",
                question="Which object is switched on in the staged scene?",
                question_time=100.0,
                answer="C",
                evidence="fake_video_001:70:79:transcript",
            )
        ),
        _label(
            LabelSeed(
                question_id="q_fake_003",
                video_id="fake_video_002",
                question="What fake label appears on the cereal box?",
                question_time=60.0,
                answer="B",
                evidence="fake_video_002:22:23:ocr",
            )
        ),
        _label(
            LabelSeed(
                question_id="q_fake_004",
                video_id="fake_video_002",
                question="What color is the synthetic fridge magnet?",
                question_time=160.0,
                answer="D",
                evidence="fake_video_002:130:138:transcript",
            )
        ),
        _label(
            LabelSeed(
                question_id="q_fake_005",
                video_id="fake_video_001",
                question="How far was mug-1 from notebook-1 during placement?",
                question_time=1850.0,
                answer="B",
                evidence="fake_video_001:5:12:spatial",
            )
        ),
        _label(
            LabelSeed(
                question_id="q_fake_006",
                video_id="fake_video_002",
                question="What color is the magnet in the fridge zone?",
                question_time=15.0,
                answer="D",
                evidence=None,
                is_answerable=False,
            )
        ),
    )
    return sources, labels


def _label(seed: LabelSeed) -> QALabelExample:
    answer_choices = (
        (
            AnswerChoice(choice_id="A", text="2.0 meters", choice_ltype="wrong"),
            AnswerChoice(choice_id="B", text="0.5 meters", choice_ltype="correct"),
            AnswerChoice(choice_id="C", text="1.0 meter", choice_ltype="vague"),
            AnswerChoice(
                choice_id="D",
                text="This question cannot be answered.",
                choice_ltype="unanswerable",
            ),
        )
        if seed.question_id == "q_fake_005"
        else (
            AnswerChoice(
                choice_id="A",
                text="beside the notebook",
                choice_ltype="place",
            ),
            AnswerChoice(choice_id="B", text="CEREAL-FAKE", choice_ltype="text"),
            AnswerChoice(
                choice_id="C",
                text=(
                    "This question cannot be answered."
                    if seed.question_id == "q_fake_004"
                    else "lamp"
                ),
                choice_ltype=(
                    "unanswerable" if seed.question_id == "q_fake_004" else "object"
                ),
            ),
            AnswerChoice(
                choice_id="D",
                text=(
                    "blue"
                    if seed.question_id == "q_fake_004"
                    else "This question cannot be answered."
                ),
                choice_ltype=(
                    "attribute" if seed.question_id == "q_fake_004" else "unanswerable"
                ),
            ),
        )
    )
    return QALabelExample(
        question_id=seed.question_id,
        video_id=seed.video_id,
        question=seed.question,
        question_time=seed.question_time,
        answer_choices=answer_choices,
        answer=seed.answer,
        is_answerable=seed.is_answerable,
        evidence_list=() if seed.evidence is None else (seed.evidence,),
        verification_score=1.0,
    )
