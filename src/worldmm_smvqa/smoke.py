from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, override

from worldmm_smvqa.chunking import build_chunks, read_source_streams
from worldmm_smvqa.fixtures import read_fixture_questions, validate_fixture
from worldmm_smvqa.memory_sources import build_source_memories
from worldmm_smvqa.metrics import evaluate_prediction_files, write_metrics
from worldmm_smvqa.qa import (
    MockQABackend,
    build_qa_prompt,
    parse_qa_output,
)
from worldmm_smvqa.retrieval import (
    build_retrieval_records,
    retrieve_evidence,
)
from worldmm_smvqa.schema import FrozenModel
from worldmm_smvqa.worldmm.episodic import build_episodic_graph
from worldmm_smvqa.worldmm.semantic import build_semantic_memory
from worldmm_smvqa.worldmm.visual import build_visual_memory

if TYPE_CHECKING:
    from worldmm_smvqa.retrieval_types import EvidencePack, RetrievalMemoryRecord
    from worldmm_smvqa.schema import MemoryRecord, PredictionRecord, QuestionRequest

MOCK_DISABLE_ENV = "WORLDMM_SMVQA_DISABLE_MOCK"


class _JsonLineModel(Protocol):
    def model_dump_json(self) -> str: ...


@dataclass(frozen=True, slots=True)
class NoLocalModelBackendError(Exception):
    reason: str

    @override
    def __str__(self) -> str:
        return f"NoLocalModelBackend: {self.reason}"


class SmokeArtifacts(FrozenModel):
    metrics: str
    predictions: str
    evidence_packs: str
    memory_manifest: str


class SourceStoreCounts(FrozenModel):
    caption: int
    transcript: int
    ocr: int
    object: int
    frame: int


class WorldMMStoreCounts(FrozenModel):
    episodic: int
    semantic: int
    visual: int


class RetrievalStoreCounts(FrozenModel):
    episodic: int
    semantic: int
    visual: int


class SmokeCountsByStore(FrozenModel):
    source: SourceStoreCounts
    worldmm: WorldMMStoreCounts
    retrieval: RetrievalStoreCounts


class SmokeMemoryManifest(FrozenModel):
    fixture: str
    source_examples: int
    questions: int
    chunks: int
    source_memories: int
    counts_by_store: SmokeCountsByStore
    artifacts: SmokeArtifacts


@dataclass(frozen=True, slots=True)
class SmokeResult:
    manifest: SmokeMemoryManifest
    predictions: int
    evidence_packs: int


def run_smoke_pipeline(
    fixture_dir: Path,
    out_dir: Path,
    env: Mapping[str, str],
) -> SmokeResult:
    if env.get(MOCK_DISABLE_ENV) == "1":
        raise NoLocalModelBackendError(
            reason=f"{MOCK_DISABLE_ENV}=1 disables MockQABackend",
        )

    fixture_counts = validate_fixture(fixture_dir)
    sources = read_source_streams(fixture_dir)
    questions = read_fixture_questions(fixture_dir)
    chunks = build_chunks(sources)
    clip_chunks = tuple(chunk for chunk in chunks if chunk.granularity == "clip_30s")
    source_memories = build_source_memories(clip_chunks)
    episodic = build_episodic_graph(chunks, source_memories)
    semantic = build_semantic_memory(sources)
    visual = build_visual_memory(sources)
    retrieval_memories = build_retrieval_records(episodic, semantic, visual)
    backend = MockQABackend()
    packs, predictions = _run_retrieval_qa(questions, retrieval_memories, backend)

    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / "predictions.jsonl"
    evidence_path = out_dir / "evidence_packs.jsonl"
    metrics_path = out_dir / "metrics.json"
    manifest_path = out_dir / "memory_manifest.json"
    _write_jsonl(predictions, prediction_path)
    _write_jsonl(packs, evidence_path)
    metrics = evaluate_prediction_files(prediction_path, fixture_dir / "labels.jsonl")
    write_metrics(metrics, metrics_path)
    manifest = SmokeMemoryManifest(
        fixture=str(fixture_dir),
        source_examples=fixture_counts.source_examples,
        questions=len(questions),
        chunks=len(chunks),
        source_memories=len(source_memories),
        counts_by_store=SmokeCountsByStore(
            source=_source_counts(source_memories),
            worldmm=WorldMMStoreCounts(
                episodic=len(episodic),
                semantic=len(semantic),
                visual=len(visual),
            ),
            retrieval=_retrieval_counts(retrieval_memories),
        ),
        artifacts=SmokeArtifacts(
            metrics=str(metrics_path),
            predictions=str(prediction_path),
            evidence_packs=str(evidence_path),
            memory_manifest=str(manifest_path),
        ),
    )
    _ = manifest_path.write_text(
        f"{manifest.model_dump_json()}\n",
        encoding="utf-8",
    )
    return SmokeResult(
        manifest=manifest,
        predictions=len(predictions),
        evidence_packs=len(packs),
    )


def smoke_stdout(out_dir: Path, result: SmokeResult) -> str:
    stores = result.manifest.counts_by_store.worldmm
    return (
        f"wrote {out_dir}\n"
        f"predictions={result.predictions} evidence_packs={result.evidence_packs}\n"
        f"worldmm_stores=episodic:{stores.episodic},"
        f"semantic:{stores.semantic},visual:{stores.visual}\n"
    )


def _run_retrieval_qa(
    questions: Sequence[QuestionRequest],
    memories: Sequence[RetrievalMemoryRecord],
    backend: MockQABackend,
) -> tuple[tuple[EvidencePack, ...], tuple[PredictionRecord, ...]]:
    packs: list[EvidencePack] = []
    predictions: list[PredictionRecord] = []
    for question in questions:
        pack = retrieve_evidence(
            question,
            memories,
            enabled_stores=frozenset({"episodic", "semantic", "visual"}),
        )
        prompt = build_qa_prompt(question, pack)
        packs.append(pack)
        predictions.append(
            parse_qa_output(
                question=question,
                raw_outputs=backend.raw_outputs(prompt, question, pack),
                prompt_token_count=len(prompt.split()),
                raw_model_output_path=None,
            ),
        )
    return tuple(packs), tuple(predictions)


def _write_jsonl(records: Iterable[_JsonLineModel], path: Path) -> None:
    _ = path.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in records),
        encoding="utf-8",
    )


def _source_counts(memories: Sequence[MemoryRecord]) -> SourceStoreCounts:
    counts = Counter(memory.store for memory in memories)
    return SourceStoreCounts(
        caption=counts["caption"],
        transcript=counts["transcript"],
        ocr=counts["ocr"],
        object=counts["object"],
        frame=counts["frame"],
    )


def _retrieval_counts(
    memories: Sequence[RetrievalMemoryRecord],
) -> RetrievalStoreCounts:
    counts = Counter(memory.source_store for memory in memories)
    return RetrievalStoreCounts(
        episodic=counts["episodic"],
        semantic=counts["semantic"],
        visual=counts["visual"],
    )
