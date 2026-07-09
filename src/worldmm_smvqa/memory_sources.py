from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

from worldmm_smvqa.chunking import build_chunks, read_source_streams
from worldmm_smvqa.schema import (
    MemoryBuilderCandidate,
    MemoryBuilderInput,
    MemoryRecord,
    SourceStreamExample,
    StreamChunk,
    ensure_memory_builder_input,
)

type SourceMemoryStore = Literal["caption", "transcript", "ocr", "object", "frame"]


@dataclass(frozen=True, slots=True)
class MemorySourceBuildSummary:
    path: Path
    records: int
    stores: tuple[SourceMemoryStore, ...]


@dataclass(frozen=True, slots=True)
class InvalidSourceMemoryStoreError(Exception):
    store: str

    @override
    def __str__(self) -> str:
        return f"InvalidSourceMemoryStoreError: {self.store}"


def build_source_memories(
    candidates: Sequence[MemoryBuilderCandidate],
) -> tuple[MemoryRecord, ...]:
    memories: list[MemoryRecord] = []
    caption_videos: set[str] = set()
    for candidate in candidates:
        source = ensure_memory_builder_input(candidate)
        memories.extend(_caption_memories(source, caption_videos))
        memories.extend(_transcript_memories(source))
        memories.extend(_ocr_memories(source))
        memories.extend(_object_memories(source))
        memories.extend(_frame_memories(source))
    return tuple(memories)


def write_fixture_source_memories(
    fixture_dir: Path,
    output: Path,
) -> MemorySourceBuildSummary:
    sources = read_source_streams(fixture_dir)
    chunks = tuple(
        chunk for chunk in build_chunks(sources) if chunk.granularity == "clip_30s"
    )
    memories = build_source_memories(chunks)
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(
        "".join(f"{memory.model_dump_json()}\n" for memory in memories),
        encoding="utf-8",
    )
    return MemorySourceBuildSummary(
        path=output,
        records=len(memories),
        stores=_stores(memories),
    )


def _caption_memories(
    source: MemoryBuilderInput,
    caption_videos: set[str],
) -> tuple[MemoryRecord, ...]:
    if source.video_id in caption_videos:
        return ()
    caption_videos.add(source.video_id)
    base_id = _base_id(source)
    return tuple(
        MemoryRecord(
            memory_id=f"{base_id}:caption:{index}",
            store="caption",
            text=caption,
            start_time=source.start_time,
            end_time=source.end_time,
            video_id=source.video_id,
            source_chunk_id=_source_chunk_id(source),
            frame_refs=source.frame_refs,
        )
        for index, caption in enumerate(source.captions)
    )


def _transcript_memories(source: MemoryBuilderInput) -> tuple[MemoryRecord, ...]:
    base_id = _base_id(source)
    source_chunk_id = _source_chunk_id(source)
    return tuple(
        MemoryRecord(
            memory_id=f"{base_id}:transcript:{index}",
            store="transcript",
            text=span.text,
            start_time=span.start_time,
            end_time=span.end_time,
            video_id=source.video_id,
            source_chunk_id=source_chunk_id,
        )
        for index, span in enumerate(source.transcript_spans)
    )


def _ocr_memories(source: MemoryBuilderInput) -> tuple[MemoryRecord, ...]:
    base_id = _base_id(source)
    source_chunk_id = _source_chunk_id(source)
    return tuple(
        MemoryRecord(
            memory_id=f"{base_id}:ocr:{index}",
            store="ocr",
            text=f"OCR {entry.text}",
            start_time=entry.start_time,
            end_time=entry.end_time,
            video_id=source.video_id,
            source_chunk_id=source_chunk_id,
            frame_refs=(entry.frame_ref,),
        )
        for index, entry in enumerate(source.ocr_entries)
    )


def _object_memories(source: MemoryBuilderInput) -> tuple[MemoryRecord, ...]:
    base_id = _base_id(source)
    source_chunk_id = _source_chunk_id(source)
    return tuple(
        MemoryRecord(
            memory_id=f"{base_id}:object:{index}",
            store="object",
            text=f"object {item.label} confidence={item.confidence:.3f}",
            start_time=item.start_time,
            end_time=item.end_time,
            video_id=source.video_id,
            source_chunk_id=source_chunk_id,
        )
        for index, item in enumerate(source.object_detections)
    )


def _frame_memories(source: MemoryBuilderInput) -> tuple[MemoryRecord, ...]:
    base_id = _base_id(source)
    source_chunk_id = _source_chunk_id(source)
    return tuple(
        MemoryRecord(
            memory_id=f"{base_id}:frame:{index}",
            store="frame",
            text=frame.description,
            start_time=source.start_time,
            end_time=source.end_time,
            video_id=source.video_id,
            source_chunk_id=source_chunk_id,
            frame_refs=(frame.frame_ref,),
        )
        for index, frame in enumerate(source.frame_metadata)
    )


def _base_id(source: MemoryBuilderInput) -> str:
    source_chunk_id = _source_chunk_id(source)
    if source_chunk_id is not None:
        return source_chunk_id
    start_time = _format_seconds(source.start_time)
    end_time = _format_seconds(source.end_time)
    return f"{source.video_id}:{start_time}:{end_time}"


def _source_chunk_id(source: MemoryBuilderInput) -> str | None:
    match source:
        case StreamChunk(chunk_id=chunk_id):
            return chunk_id
        case SourceStreamExample():
            return None


def _format_seconds(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _stores(memories: Sequence[MemoryRecord]) -> tuple[SourceMemoryStore, ...]:
    result: list[SourceMemoryStore] = []
    for memory in memories:
        store = _source_store(memory.store)
        if store not in result:
            result.append(store)
    return tuple(result)


def _source_store(value: str) -> SourceMemoryStore:
    match value:
        case "caption" | "transcript" | "ocr" | "object" | "frame":
            return value
        case other:
            raise InvalidSourceMemoryStoreError(store=other)
