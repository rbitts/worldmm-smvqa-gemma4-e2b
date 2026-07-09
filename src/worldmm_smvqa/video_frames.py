from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from worldmm_smvqa.retrieval_types import RETRIEVAL_FRAME_REF_CAP

if TYPE_CHECKING:
    from worldmm_smvqa.retrieval_types import EvidencePack
    from worldmm_smvqa.schema import (
        FrameMetadata,
        QuestionRequest,
        SourceStreamExample,
    )

FRAME_EXTENSIONS: Final = (".jpg", ".jpeg", ".png", ".webp")
CHUNK_ID_PARTS: Final = 4


@dataclass(frozen=True, slots=True)
class QAVideoFrame:
    frame_ref: str
    timestamp: float
    path: Path | None


@dataclass(frozen=True, slots=True)
class _ChunkSpan:
    chunk_id: str
    video_id: str
    start_time: float
    end_time: float


def sample_video_frames(
    sources: Sequence[SourceStreamExample],
    question: QuestionRequest,
    pack: EvidencePack,
    *,
    frame_root: Path | None,
    max_frames: int = RETRIEVAL_FRAME_REF_CAP,
) -> tuple[QAVideoFrame, ...]:
    """Sample official-style QA frames: one relevant pre-question shard, uniform cap."""
    if max_frames <= 0:
        return ()
    shard = _selected_shard(pack)
    if shard is None:
        return ()
    candidates = tuple(
        frame
        for source in sources
        if source.video_id == question.video_id
        for frame in source.frame_metadata
        if shard.start_time <= frame.timestamp < shard.end_time
        and frame.timestamp < question.question_time
    )
    return tuple(
        QAVideoFrame(
            frame_ref=frame.frame_ref,
            timestamp=frame.timestamp,
            path=_frame_path(frame_root, question.video_id, frame.frame_ref),
        )
        for frame in _uniform(candidates, max_frames)
    )


def _selected_shard(pack: EvidencePack) -> _ChunkSpan | None:
    shards: list[_ChunkSpan] = []
    for chunk_id in pack.retrieval_trace.eligible_shard_ids:
        shard = _parse_chunk_id(chunk_id)
        if shard is not None:
            shards.append(shard)
    if not shards:
        return None
    clips: list[_ChunkSpan] = []
    for chunk_id in pack.retrieval_trace.selected_clip_ids:
        clip = _parse_chunk_id(chunk_id)
        if clip is not None:
            clips.append(clip)
    for clip in clips:
        for shard in shards:
            if (
                shard.video_id == clip.video_id
                and shard.start_time <= clip.start_time
                and clip.end_time <= shard.end_time
            ):
                return shard
    return shards[0]


def _uniform(
    frames: Sequence[FrameMetadata],
    max_frames: int,
) -> tuple[FrameMetadata, ...]:
    ordered = tuple(
        sorted(frames, key=lambda frame: (frame.timestamp, frame.frame_ref)),
    )
    if len(ordered) <= max_frames:
        return ordered
    if max_frames == 1:
        return (ordered[0],)
    last = len(ordered) - 1
    return tuple(
        ordered[round(index * last / (max_frames - 1))]
        for index in range(max_frames)
    )


def _frame_path(frame_root: Path | None, video_id: str, frame_ref: str) -> Path | None:
    if frame_root is None:
        return None
    base = frame_root / video_id / frame_ref
    for suffix in FRAME_EXTENSIONS:
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return base.with_suffix(FRAME_EXTENSIONS[0])


def _parse_chunk_id(chunk_id: str) -> _ChunkSpan | None:
    parts = chunk_id.rsplit(":", 3)
    if len(parts) != CHUNK_ID_PARTS:
        return None
    video_id, raw_start, raw_end, _granularity = parts
    try:
        return _ChunkSpan(
            chunk_id=chunk_id,
            video_id=video_id,
            start_time=float(raw_start),
            end_time=float(raw_end),
        )
    except ValueError:
        return None
