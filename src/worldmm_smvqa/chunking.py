from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, override

from pydantic import ValidationError

from worldmm_smvqa.fixtures import FixtureValidationError
from worldmm_smvqa.schema import (
    ChunkGranularity,
    SourceStreamExample,
    StreamChunk,
    TranscriptSpan,
)

CLIP_SECONDS: Final = 30.0
SHARD_SECONDS: Final = 1800.0


@dataclass(frozen=True, slots=True)
class TemporalOrderError(Exception):
    video_id: str
    field_name: str
    previous_timestamp: float
    current_timestamp: float

    @override
    def __str__(self) -> str:
        return (
            "TemporalOrderError: "
            f"{self.video_id} {self.field_name} timestamp "
            f"{self.current_timestamp} follows {self.previous_timestamp}"
        )


@dataclass(frozen=True, slots=True)
class ChunkBuildSummary:
    path: Path
    chunks: int
    granularities: tuple[ChunkGranularity, ...]


@dataclass(frozen=True, slots=True)
class ChunkWindow:
    start_time: float
    end_time: float
    granularity: ChunkGranularity


def build_chunks(sources: Sequence[SourceStreamExample]) -> tuple[StreamChunk, ...]:
    chunks: list[StreamChunk] = []
    for source in sources:
        _require_temporal_order(source)
        chunks.extend(_chunk_source(source, CLIP_SECONDS, "clip_30s"))
        chunks.extend(_chunk_source(source, SHARD_SECONDS, "shard_30m"))
    return tuple(chunks)


def write_fixture_chunks(fixture_dir: Path, output: Path) -> ChunkBuildSummary:
    sources = read_source_streams(fixture_dir)
    chunks = build_chunks(sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(
        "".join(f"{chunk.model_dump_json()}\n" for chunk in chunks),
        encoding="utf-8",
    )
    return ChunkBuildSummary(
        path=output,
        chunks=len(chunks),
        granularities=_granularities(chunks),
    )


def read_source_streams(fixture_dir: Path) -> tuple[SourceStreamExample, ...]:
    path = fixture_dir / "sources.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FixtureValidationError(path=path, detail=str(exc)) from exc

    records: list[SourceStreamExample] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(SourceStreamExample.model_validate_json(line))
        except ValidationError as exc:
            detail = f"line {line_number}: {exc}"
            raise FixtureValidationError(path=path, detail=detail) from exc
    return tuple(records)


def _granularities(chunks: Sequence[StreamChunk]) -> tuple[ChunkGranularity, ...]:
    result: list[ChunkGranularity] = []
    for chunk in chunks:
        if chunk.granularity not in result:
            result.append(chunk.granularity)
    return tuple(result)


def _chunk_source(
    source: SourceStreamExample,
    seconds: float,
    granularity: ChunkGranularity,
) -> tuple[StreamChunk, ...]:
    chunks: list[StreamChunk] = []
    start_time = source.start_time
    while start_time < source.end_time:
        end_time = min(start_time + seconds, source.end_time)
        chunks.append(
            _build_chunk(
                source,
                ChunkWindow(
                    start_time=start_time,
                    end_time=end_time,
                    granularity=granularity,
                ),
            ),
        )
        start_time = end_time
    return tuple(chunks)


def _build_chunk(source: SourceStreamExample, window: ChunkWindow) -> StreamChunk:
    transcript_spans = tuple(
        span for span in source.transcript_spans if _inside(span.start_time, window)
    )
    ocr_entries = tuple(
        entry for entry in source.ocr_entries if _inside(entry.start_time, window)
    )
    object_detections = tuple(
        item for item in source.object_detections if _inside(item.start_time, window)
    )
    frame_metadata = tuple(
        frame for frame in source.frame_metadata if _inside(frame.timestamp, window)
    )
    return StreamChunk(
        chunk_id=_chunk_id(source.video_id, window),
        video_id=source.video_id,
        start_time=window.start_time,
        end_time=window.end_time,
        granularity=window.granularity,
        transcript=_chunk_transcript(transcript_spans),
        transcript_spans=transcript_spans,
        captions=source.captions,
        ocr=source.ocr,
        ocr_entries=ocr_entries,
        objects=source.objects,
        object_detections=object_detections,
        frame_refs=tuple(dict.fromkeys(frame.frame_ref for frame in frame_metadata)),
        frame_metadata=frame_metadata,
    )


def _chunk_transcript(spans: tuple[TranscriptSpan, ...]) -> str | None:
    if not spans:
        return None
    return "\n".join(span.text for span in spans)


def _chunk_id(video_id: str, window: ChunkWindow) -> str:
    start = _format_seconds(window.start_time)
    end = _format_seconds(window.end_time)
    return f"{video_id}:{start}:{end}:{window.granularity}"


def _format_seconds(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _inside(timestamp: float, window: ChunkWindow) -> bool:
    return window.start_time <= timestamp < window.end_time


def _require_temporal_order(source: SourceStreamExample) -> None:
    _require_sorted(
        source.video_id,
        "transcript_spans",
        tuple(span.start_time for span in source.transcript_spans),
    )
    _require_sorted(
        source.video_id,
        "ocr_entries",
        tuple(entry.start_time for entry in source.ocr_entries),
    )
    _require_sorted(
        source.video_id,
        "object_detections",
        tuple(item.start_time for item in source.object_detections),
    )
    _require_sorted(
        source.video_id,
        "frame_metadata",
        tuple(frame.timestamp for frame in source.frame_metadata),
    )


def _require_sorted(
    video_id: str,
    field_name: str,
    timestamps: Sequence[float],
) -> None:
    previous: float | None = None
    for timestamp in timestamps:
        if previous is not None and timestamp < previous:
            raise TemporalOrderError(
                video_id=video_id,
                field_name=field_name,
                previous_timestamp=previous,
                current_timestamp=timestamp,
            )
        previous = timestamp
