from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from pydantic import ValidationError

from worldmm_smvqa.schema import FrozenModel
from worldmm_smvqa.worldmm.episodic import build_episodic_graph
from worldmm_smvqa.worldmm.episodic_types import EpisodicNodeRecord, contains_node
from worldmm_smvqa.worldmm.llm_errors import LLMMemoryError
from worldmm_smvqa.worldmm.semantic import SemanticTripleRecord
from worldmm_smvqa.worldmm.visual import VisualMemoryRecord, build_visual_memory

if TYPE_CHECKING:
    from pydantic import BaseModel

    from worldmm_smvqa.schema import MemoryRecord, SourceStreamExample, StreamChunk
    from worldmm_smvqa.worldmm.episodic_types import EpisodicRecord

type TextGenerator = Callable[[str], str]
type FrameCaptioner = Callable[[Path], str]

CLIP_PROMPT: Final = (
    "You are building episodic memory for a long egocentric video.\n"
    "Summarize the clip observations below into one factual event sentence.\n"
    'Return strict JSON only: {{"summary": "..."}}\n\n'
    "Clip window [{start},{end}] seconds.\n"
    "Observations:\n{text}\n"
)
SHARD_PROMPT: Final = (
    "You are building episodic memory for a long egocentric video.\n"
    "Combine these clip summaries into one factual shard-level summary.\n"
    'Return strict JSON only: {{"summary": "..."}}\n\n'
    "Shard window [{start},{end}] seconds.\n"
    "Time-ordered clip summaries:\n{text}\n"
)
TRIPLET_PROMPT: Final = (
    "You are building semantic memory for a long egocentric video.\n"
    "Extract factual and semantic triplets from this event summary.\n"
    'Return strict JSON only: {{"triplets": '
    '[{{"subject": "...", "predicate": "...", "object": "..."}}]}}\n\n'
    "Event summary: {summary}\n"
)
CONSOLIDATION_PROMPT: Final = (
    "You are consolidating semantic memory for a long egocentric video.\n"
    "An existing triplet conflicts with a newer observation.\n"
    'Return strict JSON only: {{"action": "keep"}} to keep the existing object '
    'or {{"action": "replace"}} to adopt the new object.\n\n'
    "existing triplet: {subject} {predicate} {old_object}\n"
    "new observation: {subject} {predicate} {new_object}\n"
)


class ClipEventSummary(FrozenModel):
    summary: str


class TripletProposal(FrozenModel):
    subject: str
    predicate: str
    object: str


class TripletProposals(FrozenModel):
    triplets: tuple[TripletProposal, ...]


class ConsolidationDecision(FrozenModel):
    action: Literal["keep", "replace"]


@dataclass(slots=True)
class _TripleDraft:
    """Accumulator for one consolidated triple; mutation is the purpose."""

    subject: str
    predicate: str
    object: str
    support_ids: list[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0


def build_llm_episodic_graph(
    chunks: Sequence[StreamChunk],
    source_memories: Sequence[MemoryRecord],
    generate: TextGenerator,
) -> tuple[EpisodicRecord, ...]:
    """WorldMM-style episodic build: LLM event summaries on the chunk graph."""
    records = build_episodic_graph(chunks, source_memories)
    texts = _texts_by_chunk(source_memories)
    nodes = [item for item in records if isinstance(item, EpisodicNodeRecord)]
    summarized: dict[str, EpisodicNodeRecord] = {}
    clips: list[EpisodicNodeRecord] = []
    for node in nodes:
        if node.granularity != "clip_30s":
            continue
        prompt = CLIP_PROMPT.format(
            start=node.start_time,
            end=node.end_time,
            text="\n".join(texts.get(node.source_chunk_id, ("no observations",))),
        )
        summary = _parse(ClipEventSummary, generate(prompt), "episodic-clip").summary
        updated = node.model_copy(update={"summary": summary})
        summarized[node.node_id] = updated
        clips.append(updated)
    for node in nodes:
        if node.granularity != "shard_30m":
            continue
        children = sorted(
            (clip for clip in clips if contains_node(node, clip)),
            key=lambda clip: clip.start_time,
        )
        prompt = SHARD_PROMPT.format(
            start=node.start_time,
            end=node.end_time,
            text="\n".join(clip.summary for clip in children) or "no observations",
        )
        summary = _parse(ClipEventSummary, generate(prompt), "episodic-shard").summary
        summarized[node.node_id] = node.model_copy(update={"summary": summary})
    return tuple(
        summarized.get(item.node_id, item)
        if isinstance(item, EpisodicNodeRecord)
        else item
        for item in records
    )


def build_llm_semantic_memory(
    episodic_records: Sequence[EpisodicRecord],
    generate: TextGenerator,
) -> tuple[SemanticTripleRecord, ...]:
    """WorldMM-style semantic build: triplet extraction + LLM consolidation."""
    clips = sorted(
        (
            item
            for item in episodic_records
            if isinstance(item, EpisodicNodeRecord)
            and item.granularity == "clip_30s"
            and item.summary
        ),
        key=lambda node: (node.video_id, node.start_time, node.node_id),
    )
    state: dict[tuple[str, str, str], _TripleDraft] = {}
    for clip in clips:
        prompt = TRIPLET_PROMPT.format(summary=clip.summary)
        proposals = _parse(TripletProposals, generate(prompt), "semantic-extract")
        for proposal in proposals.triplets:
            _consolidate(state, clip, proposal, generate)
    return tuple(
        _triple_record(video_id, draft)
        for (video_id, _subject, _predicate), draft in sorted(state.items())
    )


def build_llm_visual_memory(
    sources: Sequence[SourceStreamExample],
    *,
    frame_root: Path,
    caption: FrameCaptioner,
) -> tuple[VisualMemoryRecord, ...]:
    """WorldMM-style visual build: VLM captions with frame/time grounding."""
    return tuple(
        _visual_with_llm(record, frame_root, caption)
        for record in build_visual_memory(sources)
    )


def _visual_with_llm(
    record: VisualMemoryRecord,
    frame_root: Path,
    caption: FrameCaptioner,
) -> VisualMemoryRecord:
    from worldmm_smvqa.worldmm.llm_frame_paths import frame_file  # noqa: PLC0415

    path = frame_file(frame_root, record.video_id, record.frame_ref)
    return record.model_copy(
        update={
            "source_frame_description": caption(path),
            "embedding_ref": f"vlm-caption:{record.frame_ref}",
        },
    )


def _consolidate(
    state: dict[tuple[str, str, str], _TripleDraft],
    clip: EpisodicNodeRecord,
    proposal: TripletProposal,
    generate: TextGenerator,
) -> None:
    key = (clip.video_id, _slug(proposal.subject), _slug(proposal.predicate))
    existing = state.get(key)
    if existing is None:
        state[key] = _TripleDraft(
            subject=proposal.subject,
            predicate=proposal.predicate,
            object=proposal.object,
            support_ids=[clip.node_id],
            start_time=clip.start_time,
            end_time=clip.end_time,
        )
        return
    if existing.object != proposal.object:
        # ponytail: lexical subject/predicate overlap + keep/replace only;
        # paper uses embedding-similarity overlap and remove/revise/add.
        prompt = CONSOLIDATION_PROMPT.format(
            subject=proposal.subject,
            predicate=proposal.predicate,
            old_object=existing.object,
            new_object=proposal.object,
        )
        decision = _parse(ConsolidationDecision, generate(prompt), "semantic-merge")
        if decision.action == "replace":
            existing.object = proposal.object
    if clip.node_id not in existing.support_ids:
        existing.support_ids.append(clip.node_id)
    existing.start_time = min(existing.start_time, clip.start_time)
    existing.end_time = max(existing.end_time, clip.end_time)


def _triple_record(video_id: str, draft: _TripleDraft) -> SemanticTripleRecord:
    memory_id = f"semantic:{video_id}:{_slug(draft.subject)}:{_slug(draft.predicate)}"
    return SemanticTripleRecord(
        memory_id=memory_id,
        video_id=video_id,
        subject=draft.subject,
        predicate=draft.predicate,
        object=draft.object,
        text=f"{draft.subject} {draft.predicate} {draft.object}",
        support_memory_ids=tuple(draft.support_ids),
        support_event_count=len(draft.support_ids),
        start_time=draft.start_time,
        end_time=draft.end_time,
        confidence=min(1.0, len(draft.support_ids) / 4.0),
        text_embedding_id=f"embedding:{memory_id}:text",
    )


def _texts_by_chunk(
    source_memories: Sequence[MemoryRecord],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for memory in source_memories:
        if memory.source_chunk_id is not None:
            grouped.setdefault(memory.source_chunk_id, []).append(memory.text)
    return {chunk_id: tuple(texts) for chunk_id, texts in grouped.items()}


def _parse[ModelT: BaseModel](
    model: type[ModelT],
    raw_output: str,
    stage: str,
) -> ModelT:
    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        return model.model_validate_json(text)
    except ValidationError as exc:
        raise LLMMemoryError(stage=stage, detail=str(exc)) from exc


def _slug(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    return "_".join(part for part in cleaned.split("_") if part)
