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
    video_id: str
    frame_ref: str
    timestamp: float
    path: Path | None


@dataclass(frozen=True, slots=True)
class _ChunkSpan:
    chunk_id: str
    video_id: str
    start_time: float
    end_time: float


@dataclass(frozen=True, slots=True)
class _FrameCandidate:
    video_id: str
    frame: FrameMetadata


def sample_video_frames(
    sources: Sequence[SourceStreamExample],
    question: QuestionRequest,
    pack: EvidencePack,
    *,
    frame_root: Path | None,
    max_frames: int = RETRIEVAL_FRAME_REF_CAP,
) -> tuple[QAVideoFrame, ...]:
    """Sample official-style QA frames across selected pre-question shards."""
    if max_frames <= 0:
        return ()
    allowed_video_ids = frozenset(_question_video_ids(question))
    shards = tuple(
        shard
        for shard in _selected_shards(pack)
        if shard.video_id in allowed_video_ids
    )
    if not shards:
        return ()
    groups = tuple(
        tuple(
            _FrameCandidate(video_id=source.video_id, frame=frame)
            for source in sources
            if source.video_id == shard.video_id
            for frame in source.frame_metadata
            if shard.start_time <= frame.timestamp < shard.end_time
            and frame.timestamp < question.question_time
        )
        for shard in shards
    )
    return tuple(
        QAVideoFrame(
            video_id=candidate.video_id,
            frame_ref=candidate.frame.frame_ref,
            timestamp=candidate.frame.timestamp,
            path=_frame_path(
                frame_root,
                candidate.video_id,
                candidate.frame.frame_ref,
            ),
        )
        for candidate in _balanced_sample(groups, max_frames)
    )


def _question_video_ids(question: QuestionRequest) -> tuple[str, ...]:
    return question.video_ids or (question.video_id,)


def _selected_shards(pack: EvidencePack) -> tuple[_ChunkSpan, ...]:
    shards = tuple(
        shard
        for chunk_id in pack.retrieval_trace.eligible_shard_ids
        if (shard := _parse_chunk_id(chunk_id)) is not None
    )
    clips = tuple(
        clip
        for chunk_id in pack.retrieval_trace.selected_clip_ids
        if (clip := _parse_chunk_id(chunk_id)) is not None
    )
    if not clips:
        return shards
    selected = tuple(
        shard
        for shard in shards
        if any(
            shard.video_id == clip.video_id
            and shard.start_time <= clip.start_time
            and clip.end_time <= shard.end_time
            for clip in clips
        )
    )
    return tuple({shard.chunk_id: shard for shard in selected}.values())


def _balanced_sample(
    groups: Sequence[Sequence[_FrameCandidate]],
    max_frames: int,
) -> tuple[_FrameCandidate, ...]:
    populated = tuple(_uniform(group, len(group)) for group in groups if group)
    if sum(map(len, populated)) <= max_frames:
        return tuple(candidate for group in populated for candidate in group)
    quotas = [0] * len(populated)
    remaining = max_frames
    while remaining:
        progressed = False
        for index, group in enumerate(populated):
            if quotas[index] >= len(group):
                continue
            quotas[index] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return tuple(
        candidate
        for group, quota in zip(populated, quotas, strict=True)
        for candidate in _uniform(group, quota)
    )


def _uniform(
    frames: Sequence[_FrameCandidate],
    max_frames: int,
) -> tuple[_FrameCandidate, ...]:
    ordered = tuple(
        sorted(
            frames,
            key=lambda item: (item.frame.timestamp, item.frame.frame_ref),
        ),
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
    root = frame_root.resolve()
    base = (root / video_id / frame_ref).resolve()
    try:
        _ = base.relative_to(root)
    except ValueError:
        return None
    for suffix in FRAME_EXTENSIONS:
        candidate = base.with_suffix(suffix)
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        try:
            _ = resolved.relative_to(root)
        except ValueError:
            return None
        if resolved.is_file():
            return resolved
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
