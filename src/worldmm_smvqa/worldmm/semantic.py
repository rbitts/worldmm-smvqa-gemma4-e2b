from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from worldmm_smvqa.chunking import read_source_streams
from worldmm_smvqa.schema import (
    FrozenModel,
    MemoryBuilderCandidate,
    SourceStreamExample,
    ensure_memory_builder_input,
)

SEMANTIC_STORE: Final = "semantic"
MIN_SUPPORT_EVENTS: Final = 2
STOP_WORDS: Final = frozenset(
    {
        "a",
        "and",
        "is",
        "on",
        "the",
        "with",
    },
)


@dataclass(frozen=True, slots=True)
class SemanticBuildSummary:
    path: Path
    records: int


@dataclass(frozen=True, slots=True)
class TextEvent:
    event_id: str
    text: str
    start_time: float
    end_time: float


class SemanticTripleRecord(FrozenModel):
    record_type: Literal["semantic_triple"] = "semantic_triple"
    memory_id: str
    store: Literal["semantic"] = "semantic"
    video_id: str
    subject: str
    predicate: str
    object: str
    text: str
    support_memory_ids: tuple[str, ...]
    support_event_count: int
    start_time: float
    end_time: float
    confidence: float
    text_embedding_id: str


def build_semantic_memory(
    candidates: Sequence[MemoryBuilderCandidate],
) -> tuple[SemanticTripleRecord, ...]:
    sources = tuple(_source(candidate) for candidate in candidates)
    return tuple(
        triple
        for source in sorted(sources, key=lambda item: item.video_id)
        for triple in _semantic_triples(source)
    )


def write_fixture_semantic_memory(
    fixture_dir: Path,
    output: Path,
) -> SemanticBuildSummary:
    records = build_semantic_memory(read_source_streams(fixture_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in records),
        encoding="utf-8",
    )
    return SemanticBuildSummary(path=output, records=len(records))


def _source(candidate: MemoryBuilderCandidate) -> SourceStreamExample:
    value = ensure_memory_builder_input(candidate)
    return SourceStreamExample(
        video_id=value.video_id,
        start_time=value.start_time,
        end_time=value.end_time,
        transcript=value.transcript,
        transcript_spans=value.transcript_spans,
        captions=value.captions,
        ocr=value.ocr,
        ocr_entries=value.ocr_entries,
        objects=value.objects,
        object_detections=value.object_detections,
        frame_refs=value.frame_refs,
        frame_metadata=value.frame_metadata,
    )


def _semantic_triples(source: SourceStreamExample) -> tuple[SemanticTripleRecord, ...]:
    events = _text_events(source)
    terms = tuple(
        sorted(
            {
                token
                for event in events
                for token in _tokens(event.text)
                if _support_count(token, events) >= MIN_SUPPORT_EVENTS
            },
        ),
    )
    return tuple(_habit_triple(source, term, events) for term in terms)


def _habit_triple(
    source: SourceStreamExample,
    term: str,
    events: Sequence[TextEvent],
) -> SemanticTripleRecord:
    support_events = tuple(event for event in events if term in _tokens(event.text))
    memory_id = f"semantic:{source.video_id}:habitually_mentions:{term}"
    return SemanticTripleRecord(
        memory_id=memory_id,
        video_id=source.video_id,
        subject=source.video_id,
        predicate="habitually_mentions",
        object=term,
        text=f"{source.video_id} habitually_mentions {term}",
        support_memory_ids=tuple(event.event_id for event in support_events),
        support_event_count=len(support_events),
        start_time=min(event.start_time for event in support_events),
        end_time=max(event.end_time for event in support_events),
        confidence=min(1.0, len(support_events) / 4.0),
        text_embedding_id=f"embedding:{memory_id}:text",
    )


def _text_events(source: SourceStreamExample) -> tuple[TextEvent, ...]:
    events: list[TextEvent] = []
    events.extend(
        TextEvent(
            event_id=f"{source.video_id}:caption:{index}",
            text=caption,
            start_time=source.start_time,
            end_time=source.end_time,
        )
        for index, caption in enumerate(source.captions)
    )
    events.extend(
        TextEvent(
            event_id=f"{source.video_id}:transcript:{index}",
            text=span.text,
            start_time=span.start_time,
            end_time=span.end_time,
        )
        for index, span in enumerate(source.transcript_spans)
    )
    events.extend(
        TextEvent(
            event_id=f"{source.video_id}:frame:{index}",
            text=frame.description,
            start_time=frame.timestamp,
            end_time=frame.timestamp,
        )
        for index, frame in enumerate(source.frame_metadata)
    )
    return tuple(events)


def _tokens(text: str) -> frozenset[str]:
    cleaned = "".join(char if char.isalnum() else " " for char in text.lower())
    return frozenset(token for token in cleaned.split() if token not in STOP_WORDS)


def _support_count(token: str, events: Sequence[TextEvent]) -> int:
    return sum(1 for event in events if token in _tokens(event.text))
