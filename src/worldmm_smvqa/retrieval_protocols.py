from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol, override

from worldmm_smvqa.retrieval_types import RETRIEVAL_FRAME_REF_CAP

if TYPE_CHECKING:
    from worldmm_smvqa.retrieval_types import RetrievalMemoryRecord
    from worldmm_smvqa.schema import QuestionRequest, StreamChunk

type WorldMMPolicyStore = Literal["spatial", "episodic", "semantic", "visual"]
type WorldMMHierarchyDepth = Literal["shard", "clip", "record"]
type EgobutlerSelectionMode = Literal["coarse-to-fine", "flat-causal"]
type WorldMMRouteReason = Literal[
    "location",
    "event_time",
    "semantic",
    "visual",
    "balanced",
]
type AvailableWorldMMStores = (
    Sequence[WorldMMPolicyStore] | frozenset[WorldMMPolicyStore]
)

LOCATION_TERMS: Final = (
    "where",
    "how far",
    "distance",
    "last seen",
    "near",
    "zone",
    "left",
    "right",
)
EVENT_TIME_TERMS: Final = (
    "after",
    "before",
    "during",
    "event",
    "happen",
    "happened",
    "when",
    "then",
    "time",
)
SEMANTIC_TERMS: Final = (
    "category",
    "class",
    "kind",
    "object",
    "relation",
    "related",
    "type",
)
VISUAL_TERMS: Final = (
    "appearance",
    "color",
    "colour",
    "frame",
    "look",
    "ocr",
    "read",
    "text",
    "visible",
)
LOCATION_ORDER: Final[tuple[WorldMMPolicyStore, ...]] = (
    "spatial",
    "episodic",
    "semantic",
    "visual",
)
EVENT_TIME_ORDER: Final[tuple[WorldMMPolicyStore, ...]] = (
    "episodic",
    "semantic",
    "visual",
    "spatial",
)
SEMANTIC_ORDER: Final[tuple[WorldMMPolicyStore, ...]] = (
    "semantic",
    "episodic",
    "spatial",
    "visual",
)
VISUAL_ORDER: Final[tuple[WorldMMPolicyStore, ...]] = (
    "visual",
    "episodic",
    "semantic",
    "spatial",
)
DEFAULT_STORE_ORDER: Final[tuple[WorldMMPolicyStore, ...]] = (
    "episodic",
    "semantic",
    "visual",
    "spatial",
)


class _ShardScopedRecord(Protocol):
    video_id: str
    start_time: float
    end_time: float


@dataclass(frozen=True, slots=True)
class WorldMMPolicyRoute:
    store_order: tuple[WorldMMPolicyStore, ...]
    hierarchy_depth: WorldMMHierarchyDepth
    reason: WorldMMRouteReason


@dataclass(frozen=True, slots=True)
class WorldMMRetrievalPolicy:
    def route(
        self,
        question: QuestionRequest,
        available_stores: AvailableWorldMMStores,
    ) -> WorldMMPolicyRoute:
        """Return a deterministic WorldMM store route for one question."""
        haystack = _question_text(question)
        reason, store_order = _ranked_order(haystack)
        return WorldMMPolicyRoute(
            store_order=_filter_available(store_order, available_stores),
            hierarchy_depth="record",
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class InvalidFrameRefCapError(Exception):
    max_frame_refs: int

    @override
    def __str__(self) -> str:
        return f"InvalidFrameRefCapError: max_frame_refs={self.max_frame_refs}"


@dataclass(frozen=True, slots=True)
class EgobutlerClipNode:
    clip_id: str
    video_id: str
    start_time: float
    end_time: float
    memory_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EgobutlerShardNode:
    shard_id: str
    video_id: str
    start_time: float
    end_time: float
    clip_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EgobutlerHierarchy:
    shards: tuple[EgobutlerShardNode, ...]
    clips: tuple[EgobutlerClipNode, ...]


@dataclass(frozen=True, slots=True)
class EgobutlerCandidateSelection:
    records: tuple[RetrievalMemoryRecord, ...]
    eligible_shard_ids: tuple[str, ...]
    selected_clip_ids: tuple[str, ...]
    selection_mode: EgobutlerSelectionMode
    causal_filtered_count: int


def eligible_video_rag_shards(
    question: QuestionRequest,
    chunks: Sequence[StreamChunk],
) -> tuple[StreamChunk, ...]:
    video_ids = _question_video_ids(question)
    return tuple(
        chunk
        for chunk in chunks
        if chunk.video_id in video_ids
        and chunk.granularity == "shard_30m"
        and chunk.start_time < question.question_time
    )


def filter_records_to_shards[ShardScopedRecordT: _ShardScopedRecord](
    records: Sequence[ShardScopedRecordT],
    eligible_shards: Sequence[StreamChunk],
) -> tuple[ShardScopedRecordT, ...]:
    return tuple(
        record
        for record in records
        if _record_is_inside_shards(record, eligible_shards)
    )


def cap_frame_refs(
    frame_refs: Sequence[str],
    max_frame_refs: int = RETRIEVAL_FRAME_REF_CAP,
) -> tuple[str, ...]:
    if max_frame_refs < 0:
        raise InvalidFrameRefCapError(max_frame_refs=max_frame_refs)
    return tuple(frame_refs[:max_frame_refs])


def build_egobutler_hierarchy(
    chunks: Sequence[StreamChunk],
    records: Sequence[RetrievalMemoryRecord],
) -> EgobutlerHierarchy:
    clips = tuple(
        _clip_node(clip, records)
        for clip in sorted(_clips(chunks), key=_chunk_sort_key)
    )
    shards = tuple(
        _shard_node(shard, clips)
        for shard in sorted(_shards(chunks), key=_chunk_sort_key)
    )
    return EgobutlerHierarchy(shards=shards, clips=clips)


def coarse_to_fine_candidates(
    question: QuestionRequest,
    hierarchy: EgobutlerHierarchy,
    records: Sequence[RetrievalMemoryRecord],
    *,
    use_coarse_to_fine: bool = True,
    max_clips: int = 1,
) -> EgobutlerCandidateSelection:
    video_ids = _question_video_ids(question)
    same_video = tuple(record for record in records if record.video_id in video_ids)
    causal = tuple(
        record for record in same_video if record.end_time <= question.question_time
    )
    eligible_shards = tuple(
        shard
        for shard in hierarchy.shards
        if shard.video_id in video_ids and shard.start_time < question.question_time
    )
    eligible_records = _records_inside_shard_nodes(causal, eligible_shards)
    if not use_coarse_to_fine:
        return EgobutlerCandidateSelection(
            records=tuple(sorted(eligible_records, key=_record_sort_key)),
            eligible_shard_ids=tuple(shard.shard_id for shard in eligible_shards),
            selected_clip_ids=(),
            selection_mode="flat-causal",
            causal_filtered_count=len(same_video) - len(causal),
        )

    selected_clip_ids = _select_clip_ids(
        question,
        eligible_shards,
        hierarchy.clips,
        eligible_records,
        max_clips,
    )
    selected_memory_ids = {
        memory_id
        for clip in hierarchy.clips
        if clip.clip_id in selected_clip_ids
        for memory_id in clip.memory_ids
    }
    return EgobutlerCandidateSelection(
        records=tuple(
            sorted(
                (
                    record
                    for record in eligible_records
                    if record.memory_id in selected_memory_ids
                ),
                key=_record_sort_key,
            ),
        ),
        eligible_shard_ids=tuple(shard.shard_id for shard in eligible_shards),
        selected_clip_ids=selected_clip_ids,
        selection_mode="coarse-to-fine",
        causal_filtered_count=len(same_video) - len(causal),
    )


def _question_video_ids(question: QuestionRequest) -> tuple[str, ...]:
    return question.video_ids or (question.video_id,)


def _record_is_inside_shards(
    record: _ShardScopedRecord,
    eligible_shards: Sequence[StreamChunk],
) -> bool:
    return any(
        shard.video_id == record.video_id
        and _inside_window(record, shard.start_time, shard.end_time)
        for shard in eligible_shards
    )


def _clip_node(
    clip: StreamChunk,
    records: Sequence[RetrievalMemoryRecord],
) -> EgobutlerClipNode:
    return EgobutlerClipNode(
        clip_id=clip.chunk_id,
        video_id=clip.video_id,
        start_time=clip.start_time,
        end_time=clip.end_time,
        memory_ids=tuple(
            record.memory_id
            for record in sorted(records, key=_record_sort_key)
            if record.video_id == clip.video_id
            and _inside_window(record, clip.start_time, clip.end_time)
        ),
    )


def _shard_node(
    shard: StreamChunk,
    clips: Sequence[EgobutlerClipNode],
) -> EgobutlerShardNode:
    return EgobutlerShardNode(
        shard_id=shard.chunk_id,
        video_id=shard.video_id,
        start_time=shard.start_time,
        end_time=shard.end_time,
        clip_ids=tuple(
            clip.clip_id
            for clip in clips
            if clip.video_id == shard.video_id
            and shard.start_time <= clip.start_time
            and clip.end_time <= shard.end_time
        ),
    )


def _select_clip_ids(
    question: QuestionRequest,
    eligible_shards: Sequence[EgobutlerShardNode],
    clips: Sequence[EgobutlerClipNode],
    records: Sequence[RetrievalMemoryRecord],
    max_clips: int,
) -> tuple[str, ...]:
    selected: list[str] = []
    for video_id in _question_video_ids(question):
        video_shards = tuple(
            shard for shard in eligible_shards if shard.video_id == video_id
        )
        eligible_clip_ids = {
            clip_id
            for shard in _rank_shards(question, video_shards, clips, records)
            for clip_id in shard.clip_ids
        }
        ranked_clips = sorted(
            (
                clip
                for clip in clips
                if clip.video_id == video_id and clip.clip_id in eligible_clip_ids
            ),
            key=lambda clip: _clip_score_key(question, clip, records),
        )
        selected.extend(clip.clip_id for clip in ranked_clips[:max_clips])
    return tuple(selected)


def _rank_shards(
    question: QuestionRequest,
    shards: Sequence[EgobutlerShardNode],
    clips: Sequence[EgobutlerClipNode],
    records: Sequence[RetrievalMemoryRecord],
) -> tuple[EgobutlerShardNode, ...]:
    return tuple(
        sorted(
            shards,
            key=lambda shard: _shard_score_key(question, shard, clips, records),
        ),
    )


def _clip_score_key(
    question: QuestionRequest,
    clip: EgobutlerClipNode,
    records: Sequence[RetrievalMemoryRecord],
) -> tuple[float, float, str]:
    score = _snippet_score(question, _memory_snippet(clip.memory_ids, records))
    return (-score, -clip.start_time, clip.clip_id)


def _shard_score_key(
    question: QuestionRequest,
    shard: EgobutlerShardNode,
    clips: Sequence[EgobutlerClipNode],
    records: Sequence[RetrievalMemoryRecord],
) -> tuple[float, float, str]:
    memory_ids = tuple(
        memory_id
        for clip in clips
        if clip.clip_id in shard.clip_ids
        for memory_id in clip.memory_ids
    )
    score = _snippet_score(question, _memory_snippet(memory_ids, records))
    return (-score, -shard.start_time, shard.shard_id)


def _memory_snippet(
    memory_ids: Sequence[str],
    records: Sequence[RetrievalMemoryRecord],
) -> str:
    wanted = frozenset(memory_ids)
    return " ".join(
        record.snippet
        for record in sorted(records, key=_record_sort_key)
        if record.memory_id in wanted
    )


def _snippet_score(question: QuestionRequest, snippet: str) -> float:
    query_terms = _tokens(
        " ".join(
            (
                question.question,
                *(choice.text for choice in question.answer_choices),
            ),
        ),
    )
    if not query_terms:
        return 0.0
    return len(query_terms & _tokens(snippet)) / len(query_terms)


def _tokens(text: str) -> frozenset[str]:
    cleaned = "".join(char if char.isalnum() else " " for char in text.lower())
    return frozenset(token for token in cleaned.split() if token not in _STOP_WORDS)


def _clips(chunks: Sequence[StreamChunk]) -> tuple[StreamChunk, ...]:
    return tuple(chunk for chunk in chunks if chunk.granularity == "clip_30s")


def _shards(chunks: Sequence[StreamChunk]) -> tuple[StreamChunk, ...]:
    return tuple(chunk for chunk in chunks if chunk.granularity == "shard_30m")


def _records_inside_shard_nodes(
    records: Sequence[RetrievalMemoryRecord],
    eligible_shards: Sequence[EgobutlerShardNode],
) -> tuple[RetrievalMemoryRecord, ...]:
    return tuple(
        record
        for record in records
        if any(
            shard.video_id == record.video_id
            and _inside_window(record, shard.start_time, shard.end_time)
            for shard in eligible_shards
        )
    )


def _inside_window(
    record: _ShardScopedRecord,
    start_time: float,
    end_time: float,
) -> bool:
    if record.start_time == record.end_time:
        return start_time <= record.start_time < end_time
    return record.start_time < end_time and start_time < record.end_time


def _chunk_sort_key(chunk: StreamChunk) -> tuple[str, float, float, str]:
    return (chunk.video_id, chunk.start_time, chunk.end_time, chunk.chunk_id)


def _record_sort_key(record: RetrievalMemoryRecord) -> tuple[float, str, str]:
    return (record.start_time, record.source_store, record.memory_id)


def _question_text(question: QuestionRequest) -> str:
    choices = " ".join(choice.text for choice in question.answer_choices)
    return f"{question.question} {choices}".casefold()


def _ranked_order(
    haystack: str,
) -> tuple[WorldMMRouteReason, tuple[WorldMMPolicyStore, ...]]:
    if any(term in haystack for term in LOCATION_TERMS):
        return "location", LOCATION_ORDER
    if any(term in haystack for term in EVENT_TIME_TERMS):
        return "event_time", EVENT_TIME_ORDER
    if any(term in haystack for term in SEMANTIC_TERMS):
        return "semantic", SEMANTIC_ORDER
    if any(term in haystack for term in VISUAL_TERMS):
        return "visual", VISUAL_ORDER
    return "balanced", _default_store_order()


def _default_store_order() -> tuple[WorldMMPolicyStore, ...]:
    return DEFAULT_STORE_ORDER


def _filter_available(
    store_order: tuple[WorldMMPolicyStore, ...],
    available_stores: AvailableWorldMMStores,
) -> tuple[WorldMMPolicyStore, ...]:
    available = frozenset(available_stores)
    return tuple(store for store in store_order if store in available)


_STOP_WORDS: Final = frozenset(
    {
        "a",
        "and",
        "is",
        "last",
        "seen",
        "the",
        "was",
        "what",
        "which",
        "with",
    },
)

# ponytail: add hour/day summary nodes when ingest produces real summaries.
