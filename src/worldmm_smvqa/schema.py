from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, Self, override

from pydantic import BaseModel, ConfigDict, model_validator

PROHIBITED_MEMORY_FIELDS: Final = (
    "answer",
    "answer_choices.choice_ltype",
    "is_answerable",
    "evidence_list",
    "verification_score",
)

type ChunkGranularity = Literal["clip_30s", "shard_30m"]


@dataclass(frozen=True, slots=True)
class LeakageError(Exception):
    received_type: str

    @override
    def __str__(self) -> str:
        return f"LeakageError: memory builders reject {self.received_type}"


class FrozenModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


class TimedModel(FrozenModel):
    video_id: str
    start_time: float
    end_time: float

    @model_validator(mode="after")
    def _require_forward_time(self) -> Self:
        if self.end_time <= self.start_time:
            msg = "end_time must be greater than start_time"
            raise ValueError(msg)
        return self


class LocalTimedModel(FrozenModel):
    start_time: float
    end_time: float

    @model_validator(mode="after")
    def _require_forward_time(self) -> Self:
        if self.end_time <= self.start_time:
            msg = "end_time must be greater than start_time"
            raise ValueError(msg)
        return self


class TranscriptSpan(LocalTimedModel):
    text: str


class OCRMetadata(LocalTimedModel):
    text: str
    frame_ref: str


class ObjectMetadata(LocalTimedModel):
    label: str
    confidence: float


class PoseSample(FrozenModel):
    """Pose in meters: x/y are horizontal plane coordinates; z is vertical."""

    timestamp: float
    x: float
    y: float
    z: float
    yaw: float | None = None


class GazeSample(FrozenModel):
    timestamp: float
    x: float
    y: float
    z: float


class FrameMetadata(FrozenModel):
    frame_ref: str
    timestamp: float
    description: str


class SourceStreamExample(TimedModel):
    transcript: str | None = None
    transcript_spans: tuple[TranscriptSpan, ...] = ()
    captions: tuple[str, ...] = ()
    ocr: tuple[str, ...] = ()
    ocr_entries: tuple[OCRMetadata, ...] = ()
    objects: tuple[str, ...] = ()
    object_detections: tuple[ObjectMetadata, ...] = ()
    pose_samples: tuple[PoseSample, ...] = ()
    gaze_samples: tuple[GazeSample, ...] = ()
    frame_refs: tuple[str, ...] = ()
    frame_metadata: tuple[FrameMetadata, ...] = ()


class StreamChunk(SourceStreamExample):
    chunk_id: str
    granularity: ChunkGranularity


class MemoryRecord(TimedModel):
    memory_id: str
    store: str
    text: str
    source_chunk_id: str | None = None
    frame_refs: tuple[str, ...] = ()


class AnswerChoice(FrozenModel):
    choice_id: str
    text: str
    choice_ltype: str


class QuestionRequest(FrozenModel):
    question_id: str
    video_id: str
    question: str
    question_time: float
    answer_choices: tuple[AnswerChoice, ...]


class QALabelExample(QuestionRequest):
    answer: str
    is_answerable: bool
    evidence_list: tuple[str, ...]
    verification_score: float


class PredictionRecord(FrozenModel):
    question_id: str
    answerable: bool
    ranked_choices: tuple[str, ...]
    answer: str | None
    confidence: float
    supporting_memory_ids: tuple[str, ...]
    prompt_token_count: int
    raw_model_output_path: str | None


class MetricRecord(FrozenModel):
    metric_name: str
    value: float
    count: int | None = None


type MemoryBuilderInput = SourceStreamExample | StreamChunk
type MemoryBuilderCandidate = SourceStreamExample | StreamChunk | QALabelExample


def ensure_memory_builder_input(
    value: MemoryBuilderCandidate,
) -> MemoryBuilderInput:
    match value:
        case QALabelExample():
            raise LeakageError(received_type=type(value).__name__)
        case StreamChunk() | SourceStreamExample():
            return value
