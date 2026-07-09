from __future__ import annotations

from worldmm_smvqa.retrieval_protocols import (
    build_egobutler_hierarchy,
    coarse_to_fine_candidates,
)
from worldmm_smvqa.retrieval_types import RetrievalMemoryRecord
from worldmm_smvqa.schema import (
    AnswerChoice,
    ChunkGranularity,
    QuestionRequest,
    StreamChunk,
)


def test_selected_evidence_memory_ids_belong_to_selected_clip_ids() -> None:
    # Given: a shard with two clips and lexical evidence only in the first clip.
    chunks = (
        _chunk("v1:0:1800:shard_30m", "v1", 0.0, 1800.0, "shard_30m"),
        _chunk("v1:0:30:clip_30s", "v1", 0.0, 30.0, "clip_30s"),
        _chunk("v1:30:60:clip_30s", "v1", 30.0, 60.0, "clip_30s"),
    )
    records = (
        _record("mem-mug", "v1", 5.0, 10.0, "mug was beside notebook"),
        _record("mem-plant", "v1", 35.0, 40.0, "plant near window"),
    )
    hierarchy = build_egobutler_hierarchy(chunks, records)

    # When: coarse-to-fine selection expands chosen clips to memory records.
    result = coarse_to_fine_candidates(_question(), hierarchy, records)

    # Then: every selected memory id belongs to one selected clip.
    selected_clip_memory_ids = {
        memory_id
        for clip in hierarchy.clips
        if clip.clip_id in result.selected_clip_ids
        for memory_id in clip.memory_ids
    }
    assert tuple(record.memory_id for record in result.records) == ("mem-mug",)
    assert {record.memory_id for record in result.records} <= selected_clip_memory_ids
    assert result.selection_mode == "coarse-to-fine"


def test_flat_selection_mode_keeps_same_causal_cutoff() -> None:
    # Given: one future same-video record that must stay causally filtered.
    chunks = (
        _chunk("v1:0:1800:shard_30m", "v1", 0.0, 1800.0, "shard_30m"),
        _chunk("v1:0:30:clip_30s", "v1", 0.0, 30.0, "clip_30s"),
        _chunk("v1:1800:1830:clip_30s", "v1", 1800.0, 1830.0, "clip_30s"),
    )
    records = (
        _record("mem-past", "v1", 5.0, 10.0, "mug beside notebook"),
        _record("mem-future", "v1", 1815.0, 1820.0, "mug future answer"),
    )
    hierarchy = build_egobutler_hierarchy(chunks, records)
    question = _question(question_time=1810.0)

    # When: the test flag disables coarse-to-fine selection.
    coarse = coarse_to_fine_candidates(question, hierarchy, records)
    flat = coarse_to_fine_candidates(
        question,
        hierarchy,
        records,
        use_coarse_to_fine=False,
    )

    # Then: trace mode changes, but causal cutoff is identical.
    assert coarse.selection_mode == "coarse-to-fine"
    assert flat.selection_mode == "flat-causal"
    assert flat.causal_filtered_count == coarse.causal_filtered_count == 1
    assert all(record.end_time <= question.question_time for record in flat.records)


def test_hierarchy_has_no_cross_video_edges() -> None:
    # Given: same-time chunks and an injected memory from a different video.
    chunks = (
        _chunk("v1:0:1800:shard_30m", "v1", 0.0, 1800.0, "shard_30m"),
        _chunk("v1:0:30:clip_30s", "v1", 0.0, 30.0, "clip_30s"),
        _chunk("v2:0:1800:shard_30m", "v2", 0.0, 1800.0, "shard_30m"),
        _chunk("v2:0:30:clip_30s", "v2", 0.0, 30.0, "clip_30s"),
    )
    records = (
        _record("mem-v1", "v1", 5.0, 10.0, "mug beside notebook"),
        _record("mem-v2-injected", "v2", 5.0, 10.0, "mug beside notebook"),
    )

    # When: parent links are built.
    hierarchy = build_egobutler_hierarchy(chunks, records)

    # Then: v1 shard links only to v1 clips and v1 memory ids.
    v1_shard = next(
        shard for shard in hierarchy.shards if shard.shard_id == "v1:0:1800:shard_30m"
    )
    v1_clips = tuple(
        clip for clip in hierarchy.clips if clip.clip_id in v1_shard.clip_ids
    )
    assert tuple(clip.video_id for clip in v1_clips) == ("v1",)
    assert tuple(memory_id for clip in v1_clips for memory_id in clip.memory_ids) == (
        "mem-v1",
    )


def _chunk(
    chunk_id: str,
    video_id: str,
    start_time: float,
    end_time: float,
    granularity: ChunkGranularity,
) -> StreamChunk:
    return StreamChunk(
        chunk_id=chunk_id,
        video_id=video_id,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
    )


def _record(
    memory_id: str,
    video_id: str,
    start_time: float,
    end_time: float,
    snippet: str,
) -> RetrievalMemoryRecord:
    return RetrievalMemoryRecord(
        memory_id=memory_id,
        source_store="semantic",
        video_id=video_id,
        start_time=start_time,
        end_time=end_time,
        snippet=snippet,
        frame_refs=(),
    )


def _question(question_time: float = 1805.0) -> QuestionRequest:
    return QuestionRequest(
        question_id="q1",
        video_id="v1",
        question="Where was the mug last seen?",
        question_time=question_time,
        answer_choices=(
            AnswerChoice(
                choice_id="a",
                text="beside the notebook",
                choice_ltype="place",
            ),
        ),
    )
