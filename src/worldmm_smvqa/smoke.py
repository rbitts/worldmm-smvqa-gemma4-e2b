from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, override

from worldmm_smvqa.chunking import build_chunks, read_source_streams
from worldmm_smvqa.fixtures import read_fixture_questions, validate_fixture
from worldmm_smvqa.memory_sources import build_source_memories
from worldmm_smvqa.metrics import (
    EvaluationMetrics,
    evaluate_prediction_files,
    evaluate_predictions,
    write_metrics,
)
from worldmm_smvqa.qa import (
    MockQABackend,
    parse_qa_output,
)
from worldmm_smvqa.qa_prompt import build_qa_prompt
from worldmm_smvqa.retrieval import (
    RetrievalOptions,
    build_retrieval_records,
    parse_retrieval_stores,
    retrieve_evidence,
)
from worldmm_smvqa.schema import (
    FrozenModel,
    QALabelExample,
    SourceStreamExample,
    StreamChunk,
)
from worldmm_smvqa.worldmm.episodic import build_episodic_graph
from worldmm_smvqa.worldmm.semantic import build_semantic_memory
from worldmm_smvqa.worldmm.spatial import (
    build_object_anchors,
    build_object_state_snapshots,
    build_trajectory_summaries,
    build_zones,
    derive_relations,
)
from worldmm_smvqa.worldmm.spatial_diagnostics import (
    ExpectedSpatialRelation,
    memory_recall_at_k,
    relation_accuracy,
)
from worldmm_smvqa.worldmm.spatial_types import SpatialRelationRecord, ZoneRecord
from worldmm_smvqa.worldmm.visual import build_visual_memory

if TYPE_CHECKING:
    from worldmm_smvqa.retrieval import SpatialRetrievalRecord
    from worldmm_smvqa.retrieval_types import (
        EvidencePack,
        RetrievalMemoryRecord,
        RetrievalStore,
    )
    from worldmm_smvqa.schema import (
        MemoryRecord,
        PredictionRecord,
        QuestionRequest,
    )

MOCK_DISABLE_ENV = "WORLDMM_SMVQA_DISABLE_MOCK"
BASELINE_PROTOCOL: Final = "worldmm-smvqa"
BASELINE_STORES: Final[tuple[RetrievalStore, ...]] = (
    "episodic",
    "semantic",
    "visual",
    "spatial",
)


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
    spatial_memory: str
    spatial_diagnostics: str
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
    spatial: int


class RetrievalStoreCounts(FrozenModel):
    episodic: int
    semantic: int
    visual: int
    spatial: int


class SmokeCountsByStore(FrozenModel):
    source: SourceStoreCounts
    worldmm: WorldMMStoreCounts
    retrieval: RetrievalStoreCounts


class SmokeMemoryArtifact(FrozenModel):
    path: str
    count: int


class SmokeMemoryManifest(FrozenModel):
    fixture: str
    source_examples: int
    questions: int
    chunks: int
    source_memories: int
    spatial_memory: SmokeMemoryArtifact
    counts_by_store: SmokeCountsByStore
    artifacts: SmokeArtifacts


class SmokeAblationConfig(FrozenModel):
    stores: tuple[str, ...]
    protocol: str


class SmokeAblationTraceSummary(FrozenModel):
    pack_count: int
    evidence_count: int
    selected_stores: tuple[str, ...]
    protocols: tuple[str, ...]
    causal_filtered_count: int
    frame_ref_count: int


class SmokeAblationRun(FrozenModel):
    config: SmokeAblationConfig
    metrics: EvaluationMetrics
    trace_summary: SmokeAblationTraceSummary


class SmokeAblationReport(FrozenModel):
    baseline: SmokeAblationRun
    ablation: SmokeAblationRun
    delta: dict[str, float]


@dataclass(frozen=True, slots=True)
class SmokeResult:
    manifest: SmokeMemoryManifest
    predictions: int
    evidence_packs: int


def run_smoke_pipeline(
    fixture_dir: Path,
    out_dir: Path,
    env: Mapping[str, str],
    *,
    ablation_stores: str | None = None,
    ablation_protocol: str | None = None,
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
    spatial = _build_spatial_memory(sources, clip_chunks)
    retrieval_memories = build_retrieval_records(episodic, semantic, visual, spatial)
    backend = MockQABackend()
    packs, predictions = _run_retrieval_qa(
        questions,
        retrieval_memories,
        backend,
        enabled_stores=frozenset(BASELINE_STORES),
        chunks=chunks,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / "predictions.jsonl"
    evidence_path = out_dir / "evidence_packs.jsonl"
    metrics_path = out_dir / "metrics.json"
    spatial_memory_path = out_dir / "spatial_memory.jsonl"
    spatial_diagnostics_path = out_dir / "spatial_diagnostics.json"
    manifest_path = out_dir / "memory_manifest.json"
    _write_jsonl(predictions, prediction_path)
    _write_jsonl(packs, evidence_path)
    _write_jsonl(spatial, spatial_memory_path)
    metrics = evaluate_prediction_files(prediction_path, fixture_dir / "labels.jsonl")
    write_metrics(metrics, metrics_path)
    _write_spatial_diagnostics(
        packs,
        fixture_dir,
        spatial,
        spatial_diagnostics_path,
    )
    if ablation_stores is not None or ablation_protocol is not None:
        labels = _read_labels(fixture_dir / "labels.jsonl")
        ablation_store_set = (
            parse_retrieval_stores(ablation_stores)
            if ablation_stores is not None
            else frozenset(BASELINE_STORES)
        )
        ablation_protocol_name = ablation_protocol or BASELINE_PROTOCOL
        ablation_packs, ablation_predictions = _run_retrieval_qa(
            questions,
            retrieval_memories,
            backend,
            enabled_stores=ablation_store_set,
            chunks=_chunks_for_protocol(chunks, ablation_protocol_name),
        )
        _write_ablation_report(
            out_dir / "ablation.json",
            baseline=SmokeAblationRun(
                config=SmokeAblationConfig(
                    stores=BASELINE_STORES,
                    protocol=BASELINE_PROTOCOL,
                ),
                metrics=metrics,
                trace_summary=_trace_summary(packs),
            ),
            ablation=SmokeAblationRun(
                config=SmokeAblationConfig(
                    stores=_ordered_stores(ablation_store_set),
                    protocol=ablation_protocol_name,
                ),
                metrics=evaluate_predictions(labels, ablation_predictions),
                trace_summary=_trace_summary(ablation_packs),
            ),
        )
    manifest = SmokeMemoryManifest(
        fixture=str(fixture_dir),
        source_examples=fixture_counts.source_examples,
        questions=len(questions),
        chunks=len(chunks),
        source_memories=len(source_memories),
        spatial_memory=SmokeMemoryArtifact(
            path=str(spatial_memory_path),
            count=len(spatial),
        ),
        counts_by_store=SmokeCountsByStore(
            source=_source_counts(source_memories),
            worldmm=WorldMMStoreCounts(
                episodic=len(episodic),
                semantic=len(semantic),
                visual=len(visual),
                spatial=len(spatial),
            ),
            retrieval=_retrieval_counts(retrieval_memories),
        ),
        artifacts=SmokeArtifacts(
            metrics=str(metrics_path),
            predictions=str(prediction_path),
            evidence_packs=str(evidence_path),
            spatial_memory=str(spatial_memory_path),
            spatial_diagnostics=str(spatial_diagnostics_path),
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
        f"semantic:{stores.semantic},visual:{stores.visual},spatial:{stores.spatial}\n"
    )


def _run_retrieval_qa(
    questions: Sequence[QuestionRequest],
    memories: Sequence[RetrievalMemoryRecord],
    backend: MockQABackend,
    *,
    enabled_stores: frozenset[RetrievalStore],
    chunks: Sequence[StreamChunk] | None,
) -> tuple[tuple[EvidencePack, ...], tuple[PredictionRecord, ...]]:
    packs: list[EvidencePack] = []
    predictions: list[PredictionRecord] = []
    for question in questions:
        pack = retrieve_evidence(
            question,
            memories,
            enabled_stores=enabled_stores,
            options=RetrievalOptions(chunks=chunks, max_frame_refs=32),
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


def _chunks_for_protocol(
    chunks: Sequence[StreamChunk],
    protocol: str,
) -> Sequence[StreamChunk] | None:
    if protocol == "legacy-round-robin":
        return None
    return chunks


def _write_ablation_report(
    path: Path,
    *,
    baseline: SmokeAblationRun,
    ablation: SmokeAblationRun,
) -> None:
    report = SmokeAblationReport(
        baseline=baseline,
        ablation=ablation,
        delta=_metric_delta(baseline.metrics, ablation.metrics),
    )
    _ = path.write_text(
        report.model_dump_json(by_alias=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _metric_delta(
    baseline: EvaluationMetrics,
    ablation: EvaluationMetrics,
) -> dict[str, float]:
    return {
        "Ans-F1": round(ablation.ans_f1 - baseline.ans_f1, 6),
        "QA-Acc": round(ablation.qa_acc - baseline.qa_acc, 6),
        "QA-MRR": round(ablation.qa_mrr - baseline.qa_mrr, 6),
        "Memory-Recall@1": round(
            ablation.memory_recall_at_1 - baseline.memory_recall_at_1,
            6,
        ),
        "Memory-Recall@3": round(
            ablation.memory_recall_at_3 - baseline.memory_recall_at_3,
            6,
        ),
        "Memory-Recall@5": round(
            ablation.memory_recall_at_5 - baseline.memory_recall_at_5,
            6,
        ),
    }


def _trace_summary(packs: Sequence[EvidencePack]) -> SmokeAblationTraceSummary:
    return SmokeAblationTraceSummary(
        pack_count=len(packs),
        evidence_count=sum(len(pack.evidence) for pack in packs),
        selected_stores=_ordered_stores(
            frozenset(
                store for pack in packs for store in pack.selected_stores
            ),
        ),
        protocols=tuple(
            dict.fromkeys(
                protocol
                for pack in packs
                for protocol in pack.retrieval_trace.protocols
            ),
        ),
        causal_filtered_count=sum(pack.causal_filtered_count for pack in packs),
        frame_ref_count=sum(
            pack.retrieval_trace.frame_ref_count for pack in packs
        ),
    )


def _ordered_stores(stores: frozenset[RetrievalStore]) -> tuple[RetrievalStore, ...]:
    return tuple(store for store in BASELINE_STORES if store in stores)


def _build_spatial_memory(
    sources: Sequence[SourceStreamExample],
    clip_chunks: Sequence[StreamChunk],
) -> tuple[SpatialRetrievalRecord, ...]:
    zones = tuple(record for source in sources for record in build_zones(source))
    anchors = tuple(
        record for source in sources for record in build_object_anchors(source)
    )
    trajectory_chunks = tuple(
        chunk for chunk in clip_chunks if _has_zone_overlap(chunk, zones)
    )
    return (
        *zones,
        *anchors,
        *tuple(derive_relations(anchors)),
        *build_object_state_snapshots(clip_chunks, anchors),
        *build_trajectory_summaries(trajectory_chunks, zones),
    )


def _write_spatial_diagnostics(
    packs: Sequence[EvidencePack],
    fixture_dir: Path,
    spatial: Sequence[SpatialRetrievalRecord],
    path: Path,
) -> None:
    labels = _read_labels(fixture_dir / "labels.jsonl")
    expected = _read_expected_relations(fixture_dir / "expected_relations.jsonl")
    predicted = tuple(
        record for record in spatial if isinstance(record, SpatialRelationRecord)
    )
    relation = relation_accuracy(predicted, expected)
    recall = memory_recall_at_k(packs, labels, 6)
    payload = {
        "relation_accuracy": relation.model_dump(),
        "recall_at_k": recall.recall_at_k,
        "protocol_recall_at_k": recall.protocol_recall_at_k,
        "k": recall.k,
    }
    _ = path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_labels(path: Path) -> tuple[QALabelExample, ...]:
    return tuple(
        QALabelExample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _read_expected_relations(path: Path) -> tuple[ExpectedSpatialRelation, ...]:
    if not path.exists():
        return ()
    return tuple(
        ExpectedSpatialRelation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _has_zone_overlap(
    chunk: StreamChunk,
    zones: Sequence[ZoneRecord],
) -> bool:
    return any(
        zone.video_id == chunk.video_id
        and any(
            start < chunk.end_time and chunk.start_time < end
            for start, end in zone.visit_intervals
        )
        for zone in zones
    )


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
        spatial=counts["spatial"],
    )
