from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
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
from worldmm_smvqa.schema import (
    FrozenModel,
    PredictionRecord,
    QuestionRequest,
    SupportingEvidence,
    is_unanswerable_choice,
)
from worldmm_smvqa.video_frames import QAVideoFrame, sample_video_frames
from worldmm_smvqa.worldmm.geometry_executor import (
    GeometryProof,
    geometry_proofs_for_question,
)

if TYPE_CHECKING:
    from worldmm_smvqa.retrieval_types import EvidenceItem, EvidencePack, RetrievalStore
    from worldmm_smvqa.schema import StreamChunk

DEFAULT_MODEL_ID: Final = "google/gemma-4-E2B-it"
PARSE_ATTEMPT_LIMIT: Final = 2
MIN_TOKEN_LENGTH: Final = 2
SECONDS_PER_MINUTE: Final = 60
SECONDS_PER_HOUR: Final = 3_600
CLOCK_WITHOUT_HOURS_PARTS: Final = 2
CLOCK_WITH_HOURS_PARTS: Final = 3
NUMBER_PATTERN: Final = re.compile(r"[-+]?\d+(?:\.\d+)?")
WORD_PATTERN: Final = re.compile(r"[a-z]+")
DISTANCE_CHOICE_PATTERN: Final = re.compile(
    r"(?<![\w.])(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>kilometers?|kilometres?|km|centimeters?|centimetres?|cm|millimeters?|millimetres?|mm|meters?|metres?|m|feet|foot|ft|inches|inch|in)(?!\w)",
    re.IGNORECASE,
)
TIME_CHOICE_PATTERN: Final = re.compile(
    r"(?<![\d:])\d+:\d{2}(?::\d{2})?(?![\d:])",
)
TIME_UNIT_CHOICE_PATTERN: Final = re.compile(
    r"(?<![\w.])(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>seconds?|secs?|s|minutes?|mins?|hours?|hrs?|h)(?!\w)",
    re.IGNORECASE,
)
DISTANCE_TO_METERS: Final = {
    "km": 1_000.0,
    "kilometer": 1_000.0,
    "kilometers": 1_000.0,
    "kilometre": 1_000.0,
    "kilometres": 1_000.0,
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "centimetre": 0.01,
    "centimetres": 0.01,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "millimetre": 0.001,
    "millimetres": 0.001,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
}
TIME_TO_SECONDS: Final = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3_600.0,
    "hr": 3_600.0,
    "hrs": 3_600.0,
    "hour": 3_600.0,
    "hours": 3_600.0,
}
SMALL_COUNTS: Final = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
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
            f"QAParseError: {self.question_id}: attempt {self.attempt}: {self.detail}"
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
    spatial_env: Mapping[str, str] | None = None


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
    geometry_proof_ids: tuple[str, ...] = ()


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
        geometry_proofs = geometry_proofs_for_question(
            question,
            evidence_pack,
            coordinate_frame="source_world",
        )
        ranked_choices = _rank_choices(question, evidence_pack, geometry_proofs)
        geometry_requires_abstention = bool(geometry_proofs) and not any(
            proof.answerable for proof in geometry_proofs
        )
        answerable = bool(evidence_pack.evidence) and not geometry_requires_abstention
        if not answerable:
            unanswerable = tuple(
                choice.choice_id
                for choice in question.answer_choices
                if is_unanswerable_choice(choice)
            )
            if len(unanswerable) == 1:
                ranked_choices = (
                    unanswerable[0],
                    *(choice for choice in ranked_choices if choice != unanswerable[0]),
                )
        payload = ModelQAOutput(
            answerable=answerable,
            ranked_choices=ranked_choices,
            answer=ranked_choices[0] if answerable else None,
            confidence=0.75 if answerable else 0.0,
            supporting_memory_ids=(
                tuple(item.memory_id for item in evidence_pack.evidence[:3])
                if answerable
                else ()
            ),
            geometry_proof_ids=tuple(
                proof.proof_id for proof in geometry_proofs if proof.answerable
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


def parse_qa_output(  # noqa: PLR0913
    *,
    question: QuestionRequest,
    raw_outputs: Sequence[str],
    prompt_token_count: int,
    raw_model_output_path: str | None,
    evidence_pack: EvidencePack | None = None,
    geometry_proofs: Sequence[GeometryProof] = (),
    input_frame_refs: Sequence[str] = (),
    prompt_sha256: str | None = None,
) -> PredictionRecord:
    if evidence_pack is not None and (
        detail := evidence_pack_validation_error(question, evidence_pack)
    ):
        raise QAParseError(
            question_id=question.question_id,
            attempt=0,
            detail=detail,
        )
    attempts = raw_outputs[:PARSE_ATTEMPT_LIMIT]
    last_detail = "no model output"
    for raw_output in attempts:
        try:
            model_output = ModelQAOutput.model_validate_json(
                _strip_json_fence(raw_output),
            )
            return _prediction_from_model_output(
                question,
                model_output,
                prompt_token_count,
                raw_model_output_path,
                evidence_pack,
                geometry_proofs,
                input_frame_refs,
                prompt_sha256,
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
    memories = build_fixture_retrieval_stores(
        fixture_dir,
        env=retrieval_options.spatial_env,
    )
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
        geometry_proofs = geometry_proofs_for_question(
            question,
            pack,
            coordinate_frame="source_world",
        )
        prompt = build_qa_prompt(
            question,
            pack,
            video_frames,
            geometry_proofs,
        )
        raw_outputs = backend.raw_outputs(prompt, question, pack, video_frames)
        predictions.append(
            parse_qa_output(
                question=question,
                raw_outputs=raw_outputs,
                prompt_token_count=_token_count(prompt),
                raw_model_output_path=None,
                evidence_pack=pack,
                geometry_proofs=geometry_proofs,
                input_frame_refs=tuple(
                    f"{frame.video_id}/{frame.frame_ref}" for frame in video_frames
                ),
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            ),
        )
    return tuple(predictions)


def write_predictions_jsonl(
    predictions: Iterable[PredictionRecord],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            for prediction in predictions:
                _ = output.write(f"{prediction.model_dump_json()}\n")
        _ = temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _prediction_from_model_output(  # noqa: PLR0913
    question: QuestionRequest,
    model_output: ModelQAOutput,
    prompt_token_count: int,
    raw_model_output_path: str | None,
    evidence_pack: EvidencePack | None,
    geometry_proofs: Sequence[GeometryProof],
    input_frame_refs: Sequence[str],
    prompt_sha256: str | None,
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
    if tuple(dict.fromkeys(model_output.supporting_memory_ids)) != (
        model_output.supporting_memory_ids
    ):
        raise QAParseError(
            question_id=question.question_id,
            attempt=1,
            detail="supporting_memory_ids contains duplicate memory IDs",
        )
    if tuple(dict.fromkeys(model_output.geometry_proof_ids)) != (
        model_output.geometry_proof_ids
    ):
        raise QAParseError(
            question_id=question.question_id,
            attempt=1,
            detail="geometry_proof_ids contains duplicate proof IDs",
        )
    proof_by_id = {proof.proof_id: proof for proof in geometry_proofs}
    invalid_proof_id = next(
        (
            proof_id
            for proof_id in model_output.geometry_proof_ids
            if proof_id not in proof_by_id or not proof_by_id[proof_id].answerable
        ),
        None,
    )
    if invalid_proof_id is not None:
        raise QAParseError(
            question_id=question.question_id,
            attempt=1,
            detail=f"unknown or unanswerable geometry proof ID: {invalid_proof_id}",
        )
    if any(proof.answerable for proof in geometry_proofs) and not (
        model_output.geometry_proof_ids
    ):
        raise QAParseError(
            question_id=question.question_id,
            attempt=1,
            detail="answerable geometry question requires a geometry proof ID",
        )
    if (
        geometry_proofs
        and not any(proof.answerable for proof in geometry_proofs)
        and (model_output.answerable or model_output.answer is not None)
    ):
        raise QAParseError(
            question_id=question.question_id,
            attempt=1,
            detail="unanswerable geometry proof requires model abstention",
        )
    selected_proofs = tuple(
        proof_by_id[proof_id] for proof_id in model_output.geometry_proof_ids
    )
    _require_geometry_choice_matches(question, model_output, selected_proofs)
    supporting_evidence = _supporting_evidence(
        question,
        model_output.supporting_memory_ids,
        evidence_pack,
    )
    retrieved_evidence = _retrieved_evidence(question, evidence_pack)
    return PredictionRecord(
        question_id=question.question_id,
        answerable=model_output.answerable,
        ranked_choices=model_output.ranked_choices,
        answer=model_output.answer,
        confidence=model_output.confidence,
        supporting_memory_ids=model_output.supporting_memory_ids,
        geometry_proof_ids=model_output.geometry_proof_ids,
        geometry_proofs=tuple(
            proof.model_dump(mode="json") for proof in selected_proofs
        ),
        supporting_evidence=supporting_evidence,
        retrieved_evidence=retrieved_evidence,
        input_frame_refs=tuple(input_frame_refs),
        prompt_sha256=prompt_sha256,
        prompt_token_count=prompt_token_count,
        raw_model_output_path=raw_model_output_path,
    )


def _require_geometry_choice_matches(
    question: QuestionRequest,
    model_output: ModelQAOutput,
    proofs: Sequence[GeometryProof],
) -> None:
    if not model_output.answerable or model_output.answer is None or not proofs:
        return
    answerable_choices = tuple(
        choice
        for choice in question.answer_choices
        if not is_unanswerable_choice(choice)
    )
    choice_text = {choice.choice_id: choice.text for choice in answerable_choices}
    if model_output.answer not in choice_text:
        raise QAParseError(
            question_id=question.question_id,
            attempt=1,
            detail="answerable geometry proof cannot select an unanswerable choice",
        )
    chosen = choice_text[model_output.answer]
    all_choice_texts = tuple(choice_text.values())
    for proof in proofs:
        checks = tuple(
            _proof_supports_choice(proof, text, all_choice_texts)
            for text in all_choice_texts
        )
        if any(check is None for check in checks):
            raise QAParseError(
                question_id=question.question_id,
                attempt=1,
                detail=(
                    "cited geometry proof requires explicit, parseable choice units"
                ),
            )
        matches = tuple(
            text for text, check in zip(all_choice_texts, checks, strict=True) if check
        )
        if len(matches) != 1:
            raise QAParseError(
                question_id=question.question_id,
                attempt=1,
                detail="cited geometry proof must match exactly one answer choice",
            )
        if chosen != matches[0]:
            raise QAParseError(
                question_id=question.question_id,
                attempt=1,
                detail="selected answer choice contradicts cited geometry proof",
            )


def _proof_supports_choice(  # noqa: PLR0911
    proof: GeometryProof,
    chosen_text: str,
    all_choice_texts: Iterable[str],
) -> bool | None:
    if proof.operation in {"distance", "count", "last_seen"} and isinstance(
        proof.value,
        int | float,
    ):
        expected_unit = {
            "distance": "meters",
            "count": "count",
            "last_seen": "seconds",
        }[proof.operation]
        if proof.uncertainty_unit != expected_unit:
            return None
        chosen_value = _choice_number(chosen_text, proof.operation)
        if chosen_value is None:
            return None
        tolerance = max(0.05, proof.uncertainty or 0.0)
        return abs(chosen_value - float(proof.value)) <= tolerance
    normalized = " ".join(chosen_text.casefold().replace("_", " ").split())
    all_normalized = tuple(
        " ".join(text.casefold().replace("_", " ").split()) for text in all_choice_texts
    )
    if proof.operation == "last_location" and isinstance(proof.value, str):
        expected = " ".join(proof.value.casefold().replace("_", " ").split())
        if not expected:
            return None
        pattern = re.compile(rf"(?<!\w){re.escape(expected)}(?!\w)")
        if not any(pattern.search(text) for text in all_normalized):
            return None
        return pattern.search(normalized) is not None
    if proof.operation == "relative_direction" and isinstance(proof.value, str):
        aliases = {
            "left": ("left",),
            "right": ("right",),
            "front": ("front", "ahead"),
            "behind": ("behind", "back"),
        }.get(proof.value)
        if aliases is None or not any(
            any(alias in text for alias in aliases) for text in all_normalized
        ):
            return None
        return any(alias in normalized for alias in aliases)
    if proof.operation == "near" and isinstance(proof.value, bool):
        positive = ("yes", "near", "close")
        negative = ("no", "far", "not near")
        aliases = positive if proof.value else negative
        if not any(
            any(alias in text for alias in (*positive, *negative))
            for text in all_normalized
        ):
            return None
        return any(alias in normalized for alias in aliases)
    return None


def _choice_number(text: str, operation: str) -> float | None:  # noqa: PLR0911
    normalized = text.casefold().replace(",", "")
    if operation == "distance":
        matches = tuple(DISTANCE_CHOICE_PATTERN.finditer(normalized))
        if len(matches) != 1 or len(NUMBER_PATTERN.findall(normalized)) != 1:
            return None
        match = matches[0]
        return float(match["value"]) * DISTANCE_TO_METERS[match["unit"]]
    if operation == "last_seen":
        return _choice_seconds(normalized)
    if operation != "count":
        return None
    number_matches = tuple(
        match.group() for match in NUMBER_PATTERN.finditer(normalized)
    )
    word_matches = tuple(
        SMALL_COUNTS[match.group()]
        for match in WORD_PATTERN.finditer(normalized)
        if match.group() in SMALL_COUNTS
    )
    if len(number_matches) == 1 and not word_matches:
        value = float(number_matches[0])
        return value if value >= 0.0 and value.is_integer() else None
    if not number_matches and len(word_matches) == 1:
        return float(word_matches[0])
    return None


def _choice_seconds(text: str) -> float | None:  # noqa: PLR0911
    clock_matches = tuple(TIME_CHOICE_PATTERN.finditer(text))
    unit_matches = tuple(TIME_UNIT_CHOICE_PATTERN.finditer(text))
    if len(clock_matches) == 1 and not unit_matches:
        match = clock_matches[0]
        if NUMBER_PATTERN.search(f"{text[: match.start()]}{text[match.end() :]}"):
            return None
        parts = tuple(int(part) for part in match.group().split(":"))
        if parts[-1] >= SECONDS_PER_MINUTE or (
            len(parts) == CLOCK_WITH_HOURS_PARTS and parts[-2] >= SECONDS_PER_MINUTE
        ):
            return None
        if len(parts) == CLOCK_WITHOUT_HOURS_PARTS:
            return float((parts[0] * SECONDS_PER_MINUTE) + parts[1])
        return float(
            (parts[0] * SECONDS_PER_HOUR) + (parts[1] * SECONDS_PER_MINUTE) + parts[2],
        )
    if clock_matches or len(unit_matches) != 1:
        return None
    match = unit_matches[0]
    if len(NUMBER_PATTERN.findall(text)) != 1:
        return None
    return float(match["value"]) * TIME_TO_SECONDS[match["unit"]]


def _supporting_evidence(
    question: QuestionRequest,
    memory_ids: tuple[str, ...],
    evidence_pack: EvidencePack | None,
) -> tuple[SupportingEvidence, ...]:
    if evidence_pack is None:
        if memory_ids:
            raise QAParseError(
                question_id=question.question_id,
                attempt=1,
                detail="supporting memory IDs require a trusted evidence pack",
            )
        return ()
    if evidence_pack.question_id != question.question_id:
        raise QAParseError(
            question_id=question.question_id,
            attempt=1,
            detail="evidence pack question_id does not match question",
        )
    valid_video_ids = (question.video_id, *question.video_ids)
    if evidence_pack.video_id not in valid_video_ids:
        raise QAParseError(
            question_id=question.question_id,
            attempt=1,
            detail="evidence pack video_id does not match question",
        )
    evidence_by_id = {item.memory_id: item for item in evidence_pack.evidence}
    unknown = tuple(
        memory_id for memory_id in memory_ids if memory_id not in evidence_by_id
    )
    if unknown:
        raise QAParseError(
            question_id=question.question_id,
            attempt=1,
            detail=f"unknown supporting memory ID: {unknown[0]}",
        )
    invalid_video = next(
        (
            memory_id
            for memory_id in memory_ids
            if evidence_by_id[memory_id].video_id not in valid_video_ids
        ),
        None,
    )
    if invalid_video is not None:
        raise QAParseError(
            question_id=question.question_id,
            attempt=1,
            detail=(
                f"supporting memory video_id is outside question scope: {invalid_video}"
            ),
        )
    return tuple(
        _evidence_metadata(evidence_by_id[memory_id]) for memory_id in memory_ids
    )


def evidence_pack_validation_error(  # noqa: PLR0911
    question: QuestionRequest,
    evidence_pack: EvidencePack,
) -> str | None:
    """Return first trust-boundary error in a question-scoped evidence pack."""
    if evidence_pack.question_id != question.question_id:
        return "evidence pack question_id does not match question"
    valid_video_ids = frozenset((question.video_id, *question.video_ids))
    if evidence_pack.video_id not in valid_video_ids:
        return "evidence pack video_id is outside question scope"
    seen_memory_ids: set[str] = set()
    for item in evidence_pack.evidence:
        if item.memory_id in seen_memory_ids:
            return f"duplicate evidence memory ID: {item.memory_id}"
        seen_memory_ids.add(item.memory_id)
        if item.video_id not in valid_video_ids:
            return f"evidence video_id is outside question scope: {item.memory_id}"
        if not isfinite(item.start_time) or not isfinite(item.end_time):
            return f"evidence time must be finite: {item.memory_id}"
        if item.start_time > item.end_time:
            return f"evidence start_time exceeds end_time: {item.memory_id}"
        if item.end_time > question.question_time:
            return f"evidence ends after question_time: {item.memory_id}"
    return None


def _retrieved_evidence(
    question: QuestionRequest,
    evidence_pack: EvidencePack | None,
) -> tuple[SupportingEvidence, ...]:
    if evidence_pack is None:
        return ()
    if evidence_pack.question_id != question.question_id:
        raise QAParseError(
            question_id=question.question_id,
            attempt=1,
            detail="evidence pack question_id does not match question",
        )
    return tuple(_evidence_metadata(item) for item in evidence_pack.evidence)


def _evidence_metadata(item: EvidenceItem) -> SupportingEvidence:
    return SupportingEvidence(
        memory_id=item.memory_id,
        store=item.source_store,
        video_id=item.video_id,
        start_time=item.start_time,
        end_time=item.end_time,
    )


def _rank_choices(
    question: QuestionRequest,
    evidence_pack: EvidencePack,
    geometry_proofs: Sequence[GeometryProof] = (),
) -> tuple[str, ...]:
    evidence_text = " ".join(item.snippet for item in evidence_pack.evidence)
    proof_text = " ".join(
        str(proof.value) for proof in geometry_proofs if proof.answerable
    )
    context_terms = _terms(f"{evidence_text} {proof_text}")
    scored = sorted(
        (
            (-len(_terms(choice.text) & context_terms), index, choice.choice_id)
            for index, choice in enumerate(question.answer_choices)
        ),
    )
    return tuple(choice_id for _score, _index, choice_id in scored)


def _terms(text: str) -> frozenset[str]:
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return frozenset(
        part
        for part in cleaned.split()
        if len(part) > MIN_TOKEN_LENGTH or part.isdigit()
    )


def _token_count(prompt: str) -> int:
    return len(prompt.split())


def _strip_json_fence(raw_output: str) -> str:
    text = raw_output.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines[1:]).strip()
