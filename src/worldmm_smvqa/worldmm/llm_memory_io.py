from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, Protocol

from pydantic import BaseModel, ConfigDict

from worldmm_smvqa.chunking import build_chunks, read_source_streams
from worldmm_smvqa.memory_sources import build_source_memories
from worldmm_smvqa.video_frames import QAVideoFrame
from worldmm_smvqa.worldmm.episodic_types import (
    EpisodicBuildSummary,
    EpisodicNodeRecord,
)
from worldmm_smvqa.worldmm.llm_errors import LLMMemoryError
from worldmm_smvqa.worldmm.llm_memory import (
    FrameCaptioner,
    TextGenerator,
    build_llm_episodic_graph,
    build_llm_semantic_memory,
    build_llm_visual_memory,
)
from worldmm_smvqa.worldmm.semantic import SemanticBuildSummary
from worldmm_smvqa.worldmm.visual import VisualBuildSummary

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class _RecordHeader(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    record_type: str
    edge_type: str | None = None


class _VideoScoped(Protocol):
    video_id: str


DEFAULT_MEMORY_MODEL_ID: Final = "Qwen/Qwen3-VL-8B-Instruct"
CAPTION_PROMPT: Final = (
    "Describe this egocentric video frame in one factual sentence covering "
    "visible objects, text, and their positions."
)


@dataclass(frozen=True, slots=True)
class LLMMemoryBindings:
    generate: TextGenerator
    caption: FrameCaptioner


def qwen_bindings(env: Mapping[str, str]) -> LLMMemoryBindings:
    """Bind memory construction to the paper's open-weight models.

    WorldMM constructs episodic/semantic memory with GPT-5-mini; this repo
    uses the paper's own open-weight backbone Qwen3-VL-8B-Instruct. Visual
    retrieval is caption-based until a matching text/image query encoder exists.
    """
    model_ref = _required_env(env, "WORLDMM_MEMORY_MODEL_PATH")
    from worldmm_smvqa.transformers_backend import (  # noqa: PLC0415
        generate_transformers_multimodal,
    )

    def generate(prompt: str) -> str:
        return generate_transformers_multimodal(prompt, model_ref, ())

    def caption(path: Path) -> str:
        frame = QAVideoFrame(
            video_id=path.parent.name,
            frame_ref=path.stem,
            timestamp=0.0,
            path=path,
        )
        return generate_transformers_multimodal(CAPTION_PROMPT, model_ref, (frame,))

    return LLMMemoryBindings(generate=generate, caption=caption)


def _required_env(env: Mapping[str, str], name: str) -> str:
    value = env.get(name)
    if value:
        return value
    raise LLMMemoryError(stage="backend", detail=f"missing {name}")


def write_llm_episodic_memory(
    fixture_dir: Path,
    output: Path,
    generate: TextGenerator,
) -> EpisodicBuildSummary:
    completed = _completed_rank_artifact(output)
    if completed is not None:
        if completed != output:
            _merge_rank_artifacts(output)
        return _episodic_summary(completed)
    sources = partition_by_video(read_source_streams(fixture_dir))
    chunks = build_chunks(sources)
    clip_chunks = tuple(chunk for chunk in chunks if chunk.granularity == "clip_30s")
    memories = build_source_memories(clip_chunks)
    records = build_llm_episodic_graph(chunks, memories, generate)
    written = write_distributed_jsonl(records, output)
    return _episodic_summary(written)


def write_llm_semantic_memory(
    episodic_path: Path,
    output: Path,
    generate: TextGenerator,
) -> SemanticBuildSummary:
    completed = _completed_rank_artifact(output)
    if completed is not None:
        if completed != output:
            _merge_rank_artifacts(output)
        return SemanticBuildSummary(path=output, records=_line_count(completed))
    nodes = partition_by_video(_read_episodic_nodes(episodic_path))
    records = build_llm_semantic_memory(nodes, generate)
    written = write_distributed_jsonl(records, output)
    return SemanticBuildSummary(path=output, records=_line_count(written))


def write_llm_visual_memory(
    fixture_dir: Path,
    output: Path,
    *,
    frame_root: Path,
    caption: FrameCaptioner,
) -> VisualBuildSummary:
    completed = _completed_rank_artifact(output)
    if completed is not None:
        if completed != output:
            _merge_rank_artifacts(output)
        return VisualBuildSummary(path=output, records=_line_count(completed))
    records = build_llm_visual_memory(
        partition_by_video(read_source_streams(fixture_dir)),
        frame_root=frame_root,
        caption=caption,
    )
    written = write_distributed_jsonl(records, output)
    return VisualBuildSummary(path=output, records=_line_count(written))


def _read_episodic_nodes(path: Path) -> tuple[EpisodicNodeRecord, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LLMMemoryError(stage="semantic", detail=str(exc)) from exc
    return tuple(
        EpisodicNodeRecord.model_validate_json(line)
        for line in lines
        if line.strip()
        and _RecordHeader.model_validate_json(line).record_type == "node"
    )


def write_distributed_jsonl(
    records: Sequence[BaseModel],
    output: Path,
) -> Path:
    rank, world_size = distributed_env()
    rank_output = _rank_output_path(output, rank, world_size)
    _write_jsonl_atomic(records, rank_output)
    _merge_rank_artifacts(output)
    return output if rank == 0 else rank_output


def _write_jsonl_atomic(records: Sequence[BaseModel], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        _ = temporary.write_text(
            "".join(f"{record.model_dump_json()}\n" for record in records),
            encoding="utf-8",
        )
        _ = temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _merge_rank_artifacts(output: Path) -> None:
    rank, world_size = distributed_env()
    if world_size == 1 or rank != 0:
        return
    rank_outputs = tuple(
        _rank_output_path(output, rank_index, world_size)
        for rank_index in range(world_size)
    )
    timeout = _env_int("WORLDMM_MEMORY_SHARD_TIMEOUT_SECONDS", 86400)
    deadline = time.monotonic() + timeout
    while missing := tuple(path for path in rank_outputs if not path.exists()):
        if time.monotonic() >= deadline:
            names = ", ".join(str(path) for path in missing)
            raise LLMMemoryError(
                stage="distributed-merge",
                detail=f"missing memory rank shard(s): {names}",
            )
        time.sleep(0.1)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.merge.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as merged:
            for rank_output in rank_outputs:
                _ = merged.write(rank_output.read_text(encoding="utf-8"))
        _ = temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _completed_rank_artifact(output: Path) -> Path | None:
    if output.exists():
        return output
    rank, world_size = distributed_env()
    rank_output = _rank_output_path(output, rank, world_size)
    return rank_output if rank_output.exists() else None


def _rank_output_path(output: Path, rank: int, world_size: int) -> Path:
    if world_size == 1:
        return output
    return output.with_name(
        f"{output.stem}.rank{rank:05d}-of{world_size:05d}{output.suffix}",
    )


def distributed_env() -> tuple[int, int]:
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise LLMMemoryError(
            stage="distributed",
            detail=f"invalid RANK/WORLD_SIZE: {rank}/{world_size}",
        )
    return rank, world_size


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise LLMMemoryError(
            stage="distributed",
            detail=f"{name} must be an integer",
        ) from exc


def partition_by_video[RecordT: _VideoScoped](
    records: Sequence[RecordT],
) -> tuple[RecordT, ...]:
    rank, world_size = distributed_env()
    video_ids = sorted({record.video_id for record in records})
    selected = {
        video_id
        for index, video_id in enumerate(video_ids)
        if index % world_size == rank
    }
    return tuple(
        record
        for record in records
        if record.video_id in selected
    )


def _episodic_summary(path: Path) -> EpisodicBuildSummary:
    records = tuple(
        _RecordHeader.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    edges = tuple(record for record in records if record.record_type == "edge")
    return EpisodicBuildSummary(
        path=path,
        nodes=sum(1 for record in records if record.record_type == "node"),
        edges=len(edges),
        contains_edges=sum(
            1 for record in edges if record.edge_type == "contains"
        ),
    )


def _line_count(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
