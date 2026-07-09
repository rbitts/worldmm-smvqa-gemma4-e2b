from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, override

from pydantic import ValidationError

from worldmm_smvqa.chunking import build_chunks, read_source_streams
from worldmm_smvqa.fixtures import read_fixture_questions
from worldmm_smvqa.qa_prompt import build_qa_prompt
from worldmm_smvqa.retrieval import (
    RetrievalOptions,
    build_fixture_retrieval_stores,
    retrieve_evidence,
)
from worldmm_smvqa.schema import FrozenModel, PredictionRecord, QuestionRequest
from worldmm_smvqa.video_frames import QAVideoFrame, sample_video_frames

if TYPE_CHECKING:
    from worldmm_smvqa.retrieval_types import EvidencePack, RetrievalStore
    from worldmm_smvqa.schema import StreamChunk

DEFAULT_MODEL_ID: Final = "google/gemma-4-E2B-it"
PARSE_ATTEMPT_LIMIT: Final = 2
MIN_TOKEN_LENGTH: Final = 2
DEFAULT_QA_STORES: Final[frozenset[RetrievalStore]] = frozenset(
    {"episodic", "semantic", "visual", "spatial"},
)


@dataclass(frozen=True, slots=True)
class QAParseError(Exception):
    question_id: str
    attempt: int
    detail: str

    @override
    def __str__(self) -> str:
        return (
            f"QAParseError: {self.question_id}: attempt {self.attempt}: "
            f"{self.detail}"
        )


@dataclass(frozen=True, slots=True)
class QABackendUnavailableError(Exception):
    backend: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"QABackendUnavailable: {self.backend}: {self.detail}"


@dataclass(frozen=True, slots=True)
class QARetrievalOptions:
    enabled_stores: frozenset[RetrievalStore] = DEFAULT_QA_STORES
    chunks: Sequence[StreamChunk] | None = None
    use_chunk_protocol: bool = True
    max_frame_refs: int = 32
    frame_root: Path | None = None


DEFAULT_QA_RETRIEVAL_OPTIONS: Final = QARetrievalOptions()


class QABackend(Protocol):
    def raw_outputs(
        self,
        prompt: str,
        question: QuestionRequest,
        evidence_pack: EvidencePack,
        video_frames: Sequence[QAVideoFrame] = (),
    ) -> tuple[str, ...]:
        """Return bounded raw model JSON attempts for one QA prompt."""
        ...


class ModelQAOutput(FrozenModel):
    answerable: bool
    ranked_choices: tuple[str, ...]
    answer: str | None
    confidence: float
    supporting_memory_ids: tuple[str, ...]


class MockQABackend:
    def raw_outputs(
        self,
        prompt: str,
        question: QuestionRequest,
        evidence_pack: EvidencePack,
        video_frames: Sequence[QAVideoFrame] = (),
    ) -> tuple[str, ...]:
        """Return deterministic fixture-only JSON without model inference."""
        _ = (prompt, video_frames)
        ranked_choices = _rank_choices(question, evidence_pack)
        answerable = bool(evidence_pack.evidence)
        payload = ModelQAOutput(
            answerable=answerable,
            ranked_choices=ranked_choices,
            answer=ranked_choices[0] if answerable else None,
            confidence=0.75 if answerable else 0.0,
            supporting_memory_ids=tuple(
                item.memory_id for item in evidence_pack.evidence[:3]
            ),
        )
        return (payload.model_dump_json(),)


@dataclass(frozen=True, slots=True)
class Gemma4QABackend:
    model_path: str | None = None

    def raw_outputs(
        self,
        prompt: str,
        question: QuestionRequest,
        evidence_pack: EvidencePack,
        video_frames: Sequence[QAVideoFrame] = (),
    ) -> tuple[str, ...]:
        """Represent the remote Gemma 4 E2B Transformers backend."""
        from worldmm_smvqa.transformers_backend import (  # noqa: PLC0415
            TransformersGenerationError,
            generate_transformers_multimodal,
            generate_transformers_text,
        )

        _ = (question, evidence_pack)
        model_ref = self.model_path or DEFAULT_MODEL_ID
        try:
            if video_frames:
                return (
                    generate_transformers_multimodal(prompt, model_ref, video_frames),
                )
            return (generate_transformers_text(prompt, model_ref),)
        except TransformersGenerationError as exc:
            raise QABackendUnavailableError(
                backend="gemma4",
                detail=str(exc),
            ) from exc


def parse_qa_output(
    *,
    question: QuestionRequest,
    raw_outputs: Sequence[str],
    prompt_token_count: int,
    raw_model_output_path: str | None,
) -> PredictionRecord:
    attempts = raw_outputs[:PARSE_ATTEMPT_LIMIT]
    last_detail = "no model output"
    for raw_output in attempts:
        try:
            model_output = ModelQAOutput.model_validate_json(raw_output)
            return _prediction_from_model_output(
                question,
                model_output,
                prompt_token_count,
                raw_model_output_path,
            )
        except (ValidationError, QAParseError) as exc:
            last_detail = str(exc)
    raise QAParseError(
        question_id=question.question_id,
        attempt=len(attempts),
        detail=last_detail,
    )


def run_qa(
    fixture_dir: Path,
    backend: QABackend,
    *,
    retrieval_options: QARetrievalOptions = DEFAULT_QA_RETRIEVAL_OPTIONS,
) -> tuple[PredictionRecord, ...]:
    sources = read_source_streams(fixture_dir)
    memories = build_fixture_retrieval_stores(fixture_dir)
    retrieval_chunks = retrieval_options.chunks
    if retrieval_chunks is None and retrieval_options.use_chunk_protocol:
        retrieval_chunks = build_chunks(sources)
    predictions: list[PredictionRecord] = []
    for question in read_fixture_questions(fixture_dir):
        pack = retrieve_evidence(
            question,
            memories,
            enabled_stores=retrieval_options.enabled_stores,
            options=RetrievalOptions(
                chunks=retrieval_chunks,
                max_frame_refs=retrieval_options.max_frame_refs,
            ),
        )
        video_frames = sample_video_frames(
            sources,
            question,
            pack,
            frame_root=retrieval_options.frame_root,
            max_frames=retrieval_options.max_frame_refs,
        )
        prompt = build_qa_prompt(question, pack, video_frames)
        raw_outputs = backend.raw_outputs(prompt, question, pack, video_frames)
        predictions.append(
            parse_qa_output(
                question=question,
                raw_outputs=raw_outputs,
                prompt_token_count=_token_count(prompt),
                raw_model_output_path=None,
            ),
        )
    return tuple(predictions)


def write_predictions_jsonl(
    predictions: Iterable[PredictionRecord],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        "".join(f"{prediction.model_dump_json()}\n" for prediction in predictions),
        encoding="utf-8",
    )


def _prediction_from_model_output(
    question: QuestionRequest,
    model_output: ModelQAOutput,
    prompt_token_count: int,
    raw_model_output_path: str | None,
) -> PredictionRecord:
    valid_choice_ids = tuple(choice.choice_id for choice in question.answer_choices)
    if tuple(dict.fromkeys(model_output.ranked_choices)) != model_output.ranked_choices:
        raise QAParseError(
            question_id=question.question_id,
            attempt=1,
            detail="ranked_choices contains duplicate choice IDs",
        )
    if set(model_output.ranked_choices) != set(valid_choice_ids):
        raise QAParseError(
            question_id=question.question_id,
            attempt=1,
            detail="ranked_choices must contain every prompt choice ID exactly once",
        )
    if model_output.answer is not None and model_output.answer not in valid_choice_ids:
        raise QAParseError(
            question_id=question.question_id,
            attempt=1,
            detail="answer must be null or one prompt choice ID",
        )
    return PredictionRecord(
        question_id=question.question_id,
        answerable=model_output.answerable,
        ranked_choices=model_output.ranked_choices,
        answer=model_output.answer,
        confidence=model_output.confidence,
        supporting_memory_ids=model_output.supporting_memory_ids,
        prompt_token_count=prompt_token_count,
        raw_model_output_path=raw_model_output_path,
    )


def _rank_choices(
    question: QuestionRequest,
    evidence_pack: EvidencePack,
) -> tuple[str, ...]:
    evidence_text = " ".join(item.snippet for item in evidence_pack.evidence)
    context_terms = _terms(
        f"{evidence_text} {question.question}",
    )
    scored = sorted(
        (
            (-len(_terms(choice.text) & context_terms), index, choice.choice_id)
            for index, choice in enumerate(question.answer_choices)
        ),
    )
    return tuple(choice_id for _score, _index, choice_id in scored)


def _terms(text: str) -> frozenset[str]:
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return frozenset(part for part in cleaned.split() if len(part) > MIN_TOKEN_LENGTH)


def _token_count(prompt: str) -> int:
    return len(prompt.split())
