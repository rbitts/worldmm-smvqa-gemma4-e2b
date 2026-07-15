from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar, Final, Literal, Self, override

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)

PROHIBITED_MEMORY_FIELDS: Final = (
    "answer",
    "answer_choices.choice_ltype",
    "is_answerable",
    "evidence_list",
    "verification_score",
)
ANSWER_CHOICE_COUNT: Final = 4
POSE_COVARIANCE_VALUE_COUNT: Final = 36

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
    start_time: float = Field(allow_inf_nan=False)
    end_time: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _require_forward_time(self) -> Self:
        if self.end_time <= self.start_time:
            msg = "end_time must be greater than start_time"
            raise ValueError(msg)
        return self


class LocalTimedModel(FrozenModel):
    start_time: float = Field(allow_inf_nan=False)
    end_time: float = Field(allow_inf_nan=False)

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
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    instance_id: str | None = None
    x: float | None = Field(default=None, allow_inf_nan=False)
    y: float | None = Field(default=None, allow_inf_nan=False)
    z: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def _require_complete_position(self) -> Self:
        coordinates = (self.x, self.y, self.z)
        if any(value is not None for value in coordinates) and not all(
            value is not None for value in coordinates
        ):
            msg = "object geometry requires x, y, and z"
            raise ValueError(msg)
        return self


class PoseSample(FrozenModel):
    """Pose in meters/degrees in the declared coordinate frame.

    x/y are horizontal-plane coordinates and z is vertical.

    Canonical JSON uses ``*_degrees`` and
    ``pose_covariance_xyz_m_rpy_deg``. The covariance is row-major for
    ``[x_m, y_m, z_m, roll_deg, pitch_deg, yaw_deg]``; entry 35 is therefore
    yaw variance in degrees squared. ``processing_mode`` and
    ``observed_through_time`` certify online causality when a pose is used by a
    direction proof. Legacy input names remain read-only validation aliases so
    existing prepared fixtures can be migrated safely.
    """

    timestamp: float = Field(allow_inf_nan=False)
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    z: float = Field(allow_inf_nan=False)
    roll_degrees: float | None = Field(
        default=None,
        allow_inf_nan=False,
        validation_alias=AliasChoices("roll_degrees", "roll"),
    )
    pitch_degrees: float | None = Field(
        default=None,
        allow_inf_nan=False,
        validation_alias=AliasChoices("pitch_degrees", "pitch"),
    )
    yaw_degrees: float | None = Field(
        default=None,
        allow_inf_nan=False,
        validation_alias=AliasChoices("yaw_degrees", "yaw"),
    )
    source: Literal["imu", "vio", "slam", "ground_truth", "model_only"] | None = None
    processing_mode: Literal["raw", "online_causal", "offline"] | None = None
    observed_through_time: float | None = Field(default=None, allow_inf_nan=False)
    coordinate_frame: str | None = None
    pose_covariance_xyz_m_rpy_deg: tuple[float, ...] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "pose_covariance_xyz_m_rpy_deg",
            "pose_covariance",
        ),
    )

    @model_validator(mode="after")
    def _require_valid_optional_pose(self) -> Self:
        if (self.roll_degrees is None) != (self.pitch_degrees is None):
            msg = "roll_degrees and pitch_degrees must be provided together"
            raise ValueError(msg)
        if self.roll_degrees is not None and self.yaw_degrees is None:
            msg = "full roll/pitch orientation requires yaw_degrees"
            raise ValueError(msg)
        covariance = self.pose_covariance_xyz_m_rpy_deg
        if covariance is not None and (
            len(covariance) != POSE_COVARIANCE_VALUE_COUNT
            or not all(isfinite(value) for value in covariance)
        ):
            msg = "pose_covariance_xyz_m_rpy_deg must contain 36 finite values"
            raise ValueError(msg)
        if covariance is not None and any(
            covariance[index] < 0 for index in (0, 7, 14, 21, 28, 35)
        ):
            msg = "pose covariance diagonal variances must be non-negative"
            raise ValueError(msg)
        if (
            self.observed_through_time is not None
            and self.observed_through_time < self.timestamp
        ):
            msg = "observed_through_time must not precede pose timestamp"
            raise ValueError(msg)
        return self


class GazeSample(FrozenModel):
    timestamp: float = Field(allow_inf_nan=False)
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    z: float = Field(allow_inf_nan=False)


class FrameMetadata(FrozenModel):
    frame_ref: str
    timestamp: float = Field(allow_inf_nan=False)
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


def is_unanswerable_choice(choice: AnswerChoice) -> bool:
    normalized = " ".join(choice.text.casefold().replace("'", "").split())
    return choice.choice_ltype.casefold() == "unanswerable" or any(
        phrase in normalized
        for phrase in (
            "cannot be answered",
            "can not be answered",
            "not enough information",
            "unanswerable",
        )
    )


class QuestionRequest(FrozenModel):
    question_id: str
    video_id: str
    video_ids: tuple[str, ...] = ()
    question: str
    question_time: float = Field(allow_inf_nan=False)
    answer_choices: tuple[AnswerChoice, ...]
    task: str | None = None
    skill: str | None = None


class QALabelExample(QuestionRequest):
    answer: str
    is_answerable: bool
    evidence_list: tuple[str, ...]
    verification_score: float = Field(allow_inf_nan=False)


class SupportingEvidence(FrozenModel):
    memory_id: str
    store: str
    video_id: str
    start_time: float
    end_time: float

    @model_validator(mode="after")
    def _require_forward_or_point_time(self) -> Self:
        if (
            not isfinite(self.start_time)
            or not isfinite(self.end_time)
            or self.end_time < self.start_time
        ):
            msg = "times must be finite and end_time must be >= start_time"
            raise ValueError(msg)
        return self


class PredictionRecord(FrozenModel):
    question_id: str
    answerable: bool
    ranked_choices: tuple[str, ...]
    answer: str | None
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    supporting_memory_ids: tuple[str, ...]
    geometry_proof_ids: tuple[str, ...] = ()
    geometry_proofs: tuple[dict[str, JsonValue], ...] = ()
    supporting_evidence: tuple[SupportingEvidence, ...] = ()
    retrieved_evidence: tuple[SupportingEvidence, ...] = ()
    input_frame_refs: tuple[str, ...] = ()
    prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prompt_token_count: int
    raw_model_output_path: str | None

    @model_validator(mode="after")
    def _require_consistent_answer(self) -> Self:
        if not self.ranked_choices:
            msg = "ranked_choices must not be empty"
            raise ValueError(msg)
        if len(self.ranked_choices) != len(set(self.ranked_choices)):
            msg = "ranked_choices contains duplicate choice IDs"
            raise ValueError(msg)
        if self.answerable:
            if self.answer is None:
                msg = "answerable prediction requires answer"
                raise ValueError(msg)
            if self.answer != self.ranked_choices[0]:
                msg = "answer must match the top-ranked choice"
                raise ValueError(msg)
        elif self.answer is not None:
            msg = "unanswerable prediction must use null answer"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _require_matching_geometry_proofs(self) -> Self:
        if len(self.geometry_proof_ids) != len(set(self.geometry_proof_ids)):
            msg = "geometry_proof_ids contains duplicate proof IDs"
            raise ValueError(msg)
        if self.geometry_proofs:
            proof_ids = tuple(proof.get("proof_id") for proof in self.geometry_proofs)
            if not all(
                isinstance(proof_id, str) and proof_id for proof_id in proof_ids
            ):
                msg = "every geometry proof requires a non-empty proof_id"
                raise ValueError(msg)
            if proof_ids != self.geometry_proof_ids:
                msg = "geometry_proofs must match geometry_proof_ids in order"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _require_matching_supporting_evidence(self) -> Self:
        if (
            self.supporting_evidence
            and tuple(item.memory_id for item in self.supporting_evidence)
            != self.supporting_memory_ids
        ):
            msg = "supporting_evidence must match supporting_memory_ids in order"
            raise ValueError(msg)
        retrieved_ids = tuple(item.memory_id for item in self.retrieved_evidence)
        if len(retrieved_ids) != len(set(retrieved_ids)):
            msg = "retrieved_evidence contains duplicate memory IDs"
            raise ValueError(msg)
        if retrieved_ids and not set(self.supporting_memory_ids) <= set(retrieved_ids):
            msg = "supporting_memory_ids must be a subset of retrieved_evidence"
            raise ValueError(msg)
        return self


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
