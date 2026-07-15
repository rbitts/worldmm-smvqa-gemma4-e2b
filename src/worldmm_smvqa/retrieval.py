from __future__ import annotations

# allow: SIZE_OK - retrieval policy module predates this change; split loaders,
# protocol selection, and scoring when retrieval behavior changes next.
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, override

from pydantic import BaseModel, ConfigDict

from worldmm_smvqa.chunking import build_chunks, read_source_streams
from worldmm_smvqa.memory_sources import build_source_memories
from worldmm_smvqa.retrieval_protocols import (
    WorldMMRetrievalPolicy,
    build_egobutler_hierarchy,
    cap_frame_refs,
    coarse_to_fine_candidates,
    eligible_video_rag_shards,
    filter_records_to_shards,
)
from worldmm_smvqa.retrieval_types import (
    RETRIEVAL_FRAME_REF_CAP,
    EvidenceItem,
    EvidencePack,
    RetrievalCandidateCount,
    RetrievalMemoryRecord,
    RetrievalStore,
    RetrievalTrace,
)
from worldmm_smvqa.worldmm.episodic import build_episodic_graph
from worldmm_smvqa.worldmm.episodic_types import (
    EpisodicEdgeRecord,
    EpisodicNodeRecord,
    EpisodicRecord,
)
from worldmm_smvqa.worldmm.semantic import (
    SemanticTripleRecord,
    build_semantic_memory,
)
from worldmm_smvqa.worldmm.spatial_compression import (
    build_compressed_spatial_memory,
    load_spatial_experiment_config,
    load_spatial_plugins,
    spatial_token_geometry,
    spatial_token_snippet,
)
from worldmm_smvqa.worldmm.spatial_types import (
    ObjectStateSnapshotRecord,
    SpatialAnchorRecord,
    SpatialRelationRecord,
    SpatialTokenRecord,
    WearerTrajectorySummaryRecord,
    ZoneRecord,
)
from worldmm_smvqa.worldmm.typed_memory import (
    EventMemoryRecord,
    FreeSpaceMemoryRecord,
    LandmarkMemoryRecord,
    NoWriteMemoryRecord,
    ObjectMemoryRecord,
    ObjectPresenceMemoryRecord,
    PlaneMemoryRecord,
    PortalMemoryRecord,
)
from worldmm_smvqa.worldmm.visual import VisualMemoryRecord, build_visual_memory

if TYPE_CHECKING:
    from worldmm_smvqa.schema import QuestionRequest, StreamChunk

STORE_ORDER: Final[tuple[RetrievalStore, ...]] = (
    "episodic",
    "semantic",
    "visual",
    "spatial",
)
DEFAULT_EVIDENCE_BUDGET: Final = 6
CLIP_SECONDS: Final = 30.0
SHARD_SECONDS: Final = 1800.0
RELATION_ENDPOINT_KEYS: Final = ("subject_instance_id", "object_instance_id")
STOP_WORDS: Final = frozenset(
    {"a", "and", "is", "on", "the", "what", "which", "with"},
)
GEOMETRY_TERMS: Final = frozenset(
    {
        "above",
        "behind",
        "below",
        "count",
        "distance",
        "far",
        "front",
        "in_front_of",
        "left",
        "left_of",
        "last",
        "many",
        "near",
        "right",
        "right_of",
        "where",
    },
)

type SpatialRetrievalRecord = (
    ZoneRecord
    | SpatialAnchorRecord
    | SpatialRelationRecord
    | SpatialTokenRecord
    | ObjectStateSnapshotRecord
    | WearerTrajectorySummaryRecord
    | ObjectMemoryRecord
    | ObjectPresenceMemoryRecord
    | PlaneMemoryRecord
    | PortalMemoryRecord
    | FreeSpaceMemoryRecord
    | LandmarkMemoryRecord
    | EventMemoryRecord
    | NoWriteMemoryRecord
)
type TypedWritableRetrievalRecord = (
    ObjectMemoryRecord
    | PlaneMemoryRecord
    | PortalMemoryRecord
    | FreeSpaceMemoryRecord
    | LandmarkMemoryRecord
    | EventMemoryRecord
)


class _SpatialMemoryArtifact(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    path: str


class _MemoryManifest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    episodic_memory: str
    semantic_memory: str
    visual_memory: str
    spatial_memory: _SpatialMemoryArtifact
    spatial_experiment: str | None = None


class _RecordHeader(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    record_type: str


class _RecordVideoScope(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    video_id: str | None = None
    source_video_id: str | None = None


@dataclass(frozen=True, slots=True)
class InvalidRetrievalStoreError(Exception):
    store: str

    @override
    def __str__(self) -> str:
        return f"InvalidRetrievalStoreError: {self.store}"


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    record: RetrievalMemoryRecord
    score: float


@dataclass(frozen=True, slots=True)
class ProtocolSelection:
    records: tuple[RetrievalMemoryRecord, ...]
    eligible_shard_ids: tuple[str, ...]
    selected_clip_ids: tuple[str, ...]
    causal_filtered_count: int
    candidate_counts: tuple[RetrievalCandidateCount, ...]


@dataclass(frozen=True, slots=True)
class EvidenceSelection:
    items: tuple[EvidenceItem, ...]
    frame_ref_count: int


@dataclass(frozen=True, slots=True)
class RetrievalOptions:
    evidence_budget: int = DEFAULT_EVIDENCE_BUDGET
    chunks: Sequence[StreamChunk] | None = None
    max_frame_refs: int = RETRIEVAL_FRAME_REF_CAP


DEFAULT_RETRIEVAL_OPTIONS: Final = RetrievalOptions()


def retrieve_evidence(
    question: QuestionRequest,
    memory_records: Sequence[RetrievalMemoryRecord],
    *,
    enabled_stores: frozenset[RetrievalStore],
    options: RetrievalOptions = DEFAULT_RETRIEVAL_OPTIONS,
) -> EvidencePack:
    requested_stores = _ordered_stores(enabled_stores)
    video_ids = _question_video_ids(question)
    scoped = tuple(
        record
        for record in memory_records
        if record.video_id in video_ids and record.source_store in enabled_stores
    )
    route = WorldMMRetrievalPolicy().route(
        question,
        available_stores=requested_stores,
    )
    selected = _protocol_records(
        question,
        scoped,
        requested_stores,
        chunks=options.chunks,
    )
    scored = tuple(
        sorted(
            (_score_candidate(question, record) for record in selected.records),
            key=_score_sort_key,
        ),
    )
    evidence = _policy_evidence(
        scored,
        route.store_order,
        options.evidence_budget,
        max_frame_refs=options.max_frame_refs,
        geometry_query=bool(_query_terms(question) & GEOMETRY_TERMS),
    )
    return EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=requested_stores,
        selected_stores=tuple(
            dict.fromkeys(item.source_store for item in evidence.items),
        ),
        evidence_budget=options.evidence_budget,
        evidence=evidence.items,
        causal_filtered_count=selected.causal_filtered_count,
        retrieval_trace=RetrievalTrace(
            protocols=("smvqa-video-rag", "egobutler", "worldmm"),
            eligible_shard_ids=selected.eligible_shard_ids,
            selected_clip_ids=selected.selected_clip_ids,
            policy_route=route.reason,
            store_order=tuple(route.store_order),
            candidate_counts=selected.candidate_counts,
            causal_filtered_count=selected.causal_filtered_count,
            frame_ref_count=evidence.frame_ref_count,
        ),
    )


def _question_video_ids(question: QuestionRequest) -> tuple[str, ...]:
    return question.video_ids or (question.video_id,)


def build_fixture_retrieval_stores(
    fixture_dir: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[RetrievalMemoryRecord, ...]:
    sources = read_source_streams(fixture_dir)
    chunks = build_chunks(sources)
    clip_chunks = tuple(chunk for chunk in chunks if chunk.granularity == "clip_30s")
    source_memories = build_source_memories(clip_chunks)
    spatial = build_compressed_spatial_memory(sources, clip_chunks, env=env)
    return build_retrieval_records(
        build_episodic_graph(chunks, source_memories),
        build_semantic_memory(sources),
        build_visual_memory(sources),
        spatial.records,
    )


def read_retrieval_memory_artifacts(
    manifest_path: Path,
    *,
    video_ids: Collection[str] | None = None,
    memory_ids_by_store: Mapping[RetrievalStore, Collection[str]] | None = None,
) -> tuple[RetrievalMemoryRecord, ...]:
    """Stream canonical stores into retrieval projections.

    ``video_ids`` keeps distributed QA ranks from parsing unrelated videos.
    Per-store IDs further avoid materializing unused records from long videos.
    Omitted stores remain unfiltered. The manifest remains the source of paths.
    """
    manifest = _MemoryManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8"),
    )
    if manifest.spatial_experiment is not None:
        spatial_config = load_spatial_experiment_config(
            Path(manifest.spatial_experiment),
        )
        load_spatial_plugins(spatial_config)
    selected_video_ids = None if video_ids is None else frozenset(video_ids)
    return (
        *_read_projected_jsonl_records(
            Path(manifest.episodic_memory),
            _episodic_artifact_record,
            _episodic_retrieval_candidate,
            selected_video_ids,
            _selected_memory_ids(memory_ids_by_store, "episodic"),
        ),
        *_read_projected_jsonl_records(
            Path(manifest.semantic_memory),
            _semantic_artifact_record,
            _semantic_candidate,
            selected_video_ids,
            _selected_memory_ids(memory_ids_by_store, "semantic"),
        ),
        *_read_projected_jsonl_records(
            Path(manifest.visual_memory),
            _visual_artifact_record,
            _visual_candidate,
            selected_video_ids,
            _selected_memory_ids(memory_ids_by_store, "visual"),
        ),
        *_read_projected_jsonl_records(
            Path(manifest.spatial_memory.path),
            _spatial_artifact_record,
            _spatial_candidate,
            selected_video_ids,
            _selected_memory_ids(memory_ids_by_store, "spatial"),
        ),
    )


def _selected_memory_ids(
    selected: Mapping[RetrievalStore, Collection[str]] | None,
    store: RetrievalStore,
) -> frozenset[str] | None:
    if selected is None or store not in selected:
        return None
    return frozenset(selected[store])


def read_typed_spatial_retrieval_records(
    path: Path,
    *,
    video_ids: Collection[str] | None = None,
) -> tuple[RetrievalMemoryRecord, ...]:
    """Stream typed memory into exact spatial records, optionally video-scoped."""
    selected_video_ids = None if video_ids is None else frozenset(video_ids)
    candidates: list[RetrievalMemoryRecord] = []
    with path.open(encoding="utf-8") as rows:
        for line in rows:
            if not line.strip():
                continue
            candidate = _spatial_candidate(_spatial_artifact_record(line))
            if candidate is not None and (
                selected_video_ids is None or candidate.video_id in selected_video_ids
            ):
                candidates.append(candidate)
    return tuple(candidates)


def build_retrieval_records(
    episodic: Sequence[EpisodicRecord],
    semantic: Sequence[SemanticTripleRecord],
    visual: Sequence[VisualMemoryRecord],
    spatial: Sequence[SpatialRetrievalRecord] = (),
) -> tuple[RetrievalMemoryRecord, ...]:
    records: list[RetrievalMemoryRecord] = []
    records.extend(
        _episodic_candidate(record)
        for record in episodic
        if isinstance(record, EpisodicNodeRecord)
    )
    records.extend(_semantic_candidate(record) for record in semantic)
    records.extend(_visual_candidate(record) for record in visual)
    for record in spatial:
        candidate = _spatial_candidate(record)
        if candidate is not None:
            records.append(candidate)
    return tuple(records)


def _read_projected_jsonl_records[RecordT](
    path: Path,
    parse: Callable[[str], RecordT],
    project: Callable[[RecordT], RetrievalMemoryRecord | None],
    video_ids: frozenset[str] | None,
    memory_ids: frozenset[str] | None,
) -> tuple[RetrievalMemoryRecord, ...]:
    records: list[RetrievalMemoryRecord] = []
    with path.open(encoding="utf-8") as rows:
        for line in rows:
            if not line.strip():
                continue
            if video_ids is not None:
                line_video_id = _raw_record_video_id(line)
                if line_video_id is not None and line_video_id not in video_ids:
                    continue
            candidate = project(parse(line))
            if (
                candidate is not None
                and (video_ids is None or candidate.video_id in video_ids)
                and (memory_ids is None or candidate.memory_id in memory_ids)
            ):
                records.append(candidate)
    return tuple(records)


def _raw_record_video_id(line: str) -> str | None:
    scope = _RecordVideoScope.model_validate_json(line)
    return (
        scope.source_video_id if scope.source_video_id is not None else scope.video_id
    )


def _episodic_artifact_record(line: str) -> EpisodicRecord:
    header = _RecordHeader.model_validate_json(line)
    match header.record_type:
        case "node":
            return EpisodicNodeRecord.model_validate_json(line)
        case "edge":
            return EpisodicEdgeRecord.model_validate_json(line)
        case other:
            raise InvalidRetrievalStoreError(store=other)


def _episodic_retrieval_candidate(
    record: EpisodicRecord,
) -> RetrievalMemoryRecord | None:
    if not isinstance(record, EpisodicNodeRecord):
        return None
    return _episodic_candidate(record)


def _semantic_artifact_record(line: str) -> SemanticTripleRecord:
    return SemanticTripleRecord.model_validate_json(line)


def _visual_artifact_record(line: str) -> VisualMemoryRecord:
    return VisualMemoryRecord.model_validate_json(line)


def _spatial_artifact_record(  # noqa: PLR0911, PLR0912
    line: str,
) -> SpatialRetrievalRecord:
    header = _RecordHeader.model_validate_json(line)
    match header.record_type:
        case "zone":
            return ZoneRecord.model_validate_json(line)
        case "spatial_anchor":
            return SpatialAnchorRecord.model_validate_json(line)
        case "spatial_relation":
            return SpatialRelationRecord.model_validate_json(line)
        case "spatial_token":
            return SpatialTokenRecord.model_validate_json(line)
        case "object_state_snapshot":
            return ObjectStateSnapshotRecord.model_validate_json(line)
        case "wearer_trajectory_summary":
            return WearerTrajectorySummaryRecord.model_validate_json(line)
        case "object":
            return ObjectMemoryRecord.model_validate_json(line)
        case "object_presence_v1":
            return ObjectPresenceMemoryRecord.model_validate_json(line)
        case "plane":
            return PlaneMemoryRecord.model_validate_json(line)
        case "portal":
            return PortalMemoryRecord.model_validate_json(line)
        case "free_space":
            return FreeSpaceMemoryRecord.model_validate_json(line)
        case "landmark":
            return LandmarkMemoryRecord.model_validate_json(line)
        case "event":
            return EventMemoryRecord.model_validate_json(line)
        case "no_write":
            return NoWriteMemoryRecord.model_validate_json(line)
        case other:
            raise InvalidRetrievalStoreError(store=other)


def parse_retrieval_stores(value: str) -> frozenset[RetrievalStore]:
    stores: list[RetrievalStore] = []
    for part in value.split(","):
        store = part.strip()
        if store:
            stores.append(_parse_retrieval_store(store))
    return frozenset(stores)


def injected_future_memory(question: QuestionRequest) -> RetrievalMemoryRecord:
    return RetrievalMemoryRecord(
        memory_id="injected-post-question-high-score",
        source_store="semantic",
        video_id=question.video_id,
        start_time=question.question_time + 1.0,
        end_time=question.question_time + 2.0,
        snippet=f"{question.question} future perfect high score",
        frame_refs=(),
        base_score=100.0,
    )


def _episodic_candidate(record: EpisodicNodeRecord) -> RetrievalMemoryRecord:
    return RetrievalMemoryRecord(
        memory_id=record.node_id,
        source_store="episodic",
        video_id=record.video_id,
        start_time=record.start_time,
        end_time=record.end_time,
        snippet=_episodic_snippet(record),
        frame_refs=record.frame_refs,
        base_score=record.confidence,
    )


def _semantic_candidate(record: SemanticTripleRecord) -> RetrievalMemoryRecord:
    return RetrievalMemoryRecord(
        memory_id=record.memory_id,
        source_store="semantic",
        video_id=record.video_id,
        start_time=record.start_time,
        end_time=record.end_time,
        snippet=record.text,
        frame_refs=(),
        base_score=record.confidence,
    )


def _visual_candidate(record: VisualMemoryRecord) -> RetrievalMemoryRecord:
    return RetrievalMemoryRecord(
        memory_id=record.memory_id,
        source_store="visual",
        video_id=record.video_id,
        start_time=record.timestamp,
        end_time=record.timestamp,
        snippet=" ".join(
            (
                record.source_frame_description,
                *record.ocr_refs,
                *record.object_refs,
            ),
        ),
        frame_refs=(record.frame_ref,),
        base_score=1.0,
    )


def _spatial_candidate(  # noqa: PLR0911, PLR0912
    record: SpatialRetrievalRecord,
) -> RetrievalMemoryRecord | None:
    match record:
        case ZoneRecord():
            start_time, end_time = _zone_time_span(record)
            return RetrievalMemoryRecord(
                memory_id=f"spatial_zone:{record.zone_id}",
                source_store="spatial",
                video_id=record.video_id,
                start_time=start_time,
                end_time=end_time,
                snippet=(
                    f"zone {record.zone_id} centered near "
                    f"({_format_float(record.centroid_x)},"
                    f"{_format_float(record.centroid_y)})"
                ),
                frame_refs=(),
            )
        case SpatialAnchorRecord():
            return RetrievalMemoryRecord(
                memory_id=record.memory_id,
                source_store="spatial",
                video_id=record.video_id,
                start_time=record.start_time,
                end_time=record.end_time,
                snippet=(
                    f"{record.object_label} anchored in {record.zone_id} "
                    f"during [{_format_float(record.start_time)},"
                    f"{_format_float(record.end_time)}] near "
                    f"({_format_float(record.x)},{_format_float(record.y)},"
                    f"{_format_float(record.z)}) "
                    f"provenance={record.provenance}"
                    f"{_anchor_geometry_text(record)}"
                ),
                frame_refs=record.frame_refs,
                base_score=record.confidence,
                geometry=_anchor_geometry(record),
            )
        case SpatialRelationRecord():
            return RetrievalMemoryRecord(
                memory_id=record.memory_id,
                source_store="spatial",
                video_id=record.video_id,
                start_time=record.start_time,
                end_time=record.end_time,
                snippet=(
                    f"{record.subject} {record.relation} {record.object} "
                    f"in {record.zone_id} during "
                    f"[{_format_float(record.start_time)},"
                    f"{_format_float(record.end_time)}]"
                    f"{_relation_geometry_text(record)}"
                ),
                frame_refs=(),
                base_score=1.0,
                geometry=_relation_geometry(record),
            )
        case SpatialTokenRecord():
            return RetrievalMemoryRecord(
                memory_id=record.memory_id,
                source_store="spatial",
                video_id=record.video_id,
                start_time=record.start_time,
                end_time=record.end_time,
                snippet=spatial_token_snippet(record),
                frame_refs=record.frame_refs,
                base_score=record.importance,
                geometry=spatial_token_geometry(record),
            )
        case ObjectStateSnapshotRecord() | WearerTrajectorySummaryRecord():
            return RetrievalMemoryRecord(
                memory_id=record.memory_id,
                source_store="spatial",
                video_id=record.video_id,
                start_time=record.start_time,
                end_time=record.end_time,
                snippet=record.snippet,
                frame_refs=(),
                base_score=record.base_score,
            )
        case ObjectPresenceMemoryRecord():
            return RetrievalMemoryRecord(
                memory_id=record.memory_id,
                source_store="semantic",
                video_id=record.source_video_id,
                start_time=record.timestamp,
                end_time=record.timestamp,
                snippet=(
                    f"{record.semantic_class} observed "
                    f"observation_id={record.observation_id} "
                    f"semantic_confidence={_format_float(record.semantic_confidence)} "
                    f"semantic_provider_id={record.semantic_provider_id} "
                    f"ontology_sha256={record.ontology_sha256} "
                    f"mask_sha256={record.mask_sha256} "
                    f"mask_schema_id={record.mask_schema_id} "
                    f"mask_sealed_root_sha256={record.mask_sealed_root_sha256} "
                    f"mask_manifest_sha256={record.mask_manifest_sha256} "
                    f"mask_width_px={record.mask_width_px} "
                    f"mask_height_px={record.mask_height_px} "
                    f"mask_dtype={record.mask_dtype} "
                    f"source_inventory_sha256={record.source_inventory_sha256} "
                    f"timestamp_us={record.timestamp_us}"
                ),
                frame_refs=(record.frame_ref,),
                base_score=record.semantic_confidence,
            )
        case ObjectMemoryRecord():
            x, y, z = record.geometry.centroid
            extent_x, extent_y, extent_z = record.geometry.extent
            geometry = _typed_geometry(record, record.semantic_label)
            if record.place_label is not None:
                geometry["place_label"] = record.place_label
            geometry.update(
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "extent_x": extent_x,
                    "extent_y": extent_y,
                    "extent_z": extent_z,
                },
            )
            return _typed_candidate(
                record,
                snippet=(
                    f"{record.semantic_label} {record.instance_id} in "
                    f"{record.place_label or record.local_frame_id} near "
                    f"({_format_float(x)},{_format_float(y)},{_format_float(z)})"
                ),
                geometry=geometry,
            )
        case PlaneMemoryRecord():
            nx, ny, nz = record.geometry.normal
            geometry = _typed_geometry(record, "plane")
            geometry.update(
                {
                    "normal_x": nx,
                    "normal_y": ny,
                    "normal_z": nz,
                    "offset_m": record.geometry.offset_m,
                    "boundary_vertex_count": float(len(record.geometry.boundary)),
                },
            )
            return _typed_candidate(
                record,
                snippet=(
                    f"plane {record.instance_id} in {record.local_frame_id} "
                    f"normal=({_format_float(nx)},{_format_float(ny)},"
                    f"{_format_float(nz)}) "
                    f"offset_m={_format_float(record.geometry.offset_m)}"
                ),
                geometry=geometry,
            )
        case PortalMemoryRecord():
            x, y, z = record.geometry.centroid
            nx, ny, nz = record.geometry.normal
            geometry = _typed_geometry(record, "portal")
            geometry.update(
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "normal_x": nx,
                    "normal_y": ny,
                    "normal_z": nz,
                    "width_m": record.geometry.width_m,
                    "height_m": record.geometry.height_m,
                    "connects_frame_a": record.connects_frame_ids[0],
                    "connects_frame_b": record.connects_frame_ids[1],
                },
            )
            return _typed_candidate(
                record,
                snippet=(
                    f"portal {record.instance_id} connects "
                    f"{record.connects_frame_ids[0]} and "
                    f"{record.connects_frame_ids[1]} near "
                    f"({_format_float(x)},{_format_float(y)},{_format_float(z)})"
                ),
                geometry=geometry,
            )
        case FreeSpaceMemoryRecord():
            geometry = _typed_geometry(record, "free_space")
            geometry.update(
                {
                    "height_m": record.geometry.height_m,
                    "floor_vertex_count": float(
                        len(record.geometry.floor_polygon),
                    ),
                },
            )
            return _typed_candidate(
                record,
                snippet=(
                    f"free space {record.instance_id} in {record.local_frame_id} "
                    f"height_m={_format_float(record.geometry.height_m)}"
                ),
                geometry=geometry,
            )
        case LandmarkMemoryRecord():
            x, y, z = record.geometry.position
            ray_x, ray_y, ray_z = record.geometry.ray_direction
            geometry = _typed_geometry(record, "landmark")
            geometry.update(
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "ray_x": ray_x,
                    "ray_y": ray_y,
                    "ray_z": ray_z,
                    "view_cone_degrees": record.geometry.view_cone_degrees,
                },
            )
            if record.descriptor_ref is not None:
                geometry["descriptor_ref"] = record.descriptor_ref
            return _typed_candidate(
                record,
                snippet=(
                    f"landmark {record.instance_id} in {record.local_frame_id} "
                    f"near ({_format_float(x)},{_format_float(y)},"
                    f"{_format_float(z)})"
                ),
                geometry=geometry,
            )
        case EventMemoryRecord():
            geometry = _typed_geometry(record, "event")
            geometry["event_kind"] = record.event_kind
            involved_entity_ids = json.dumps(
                record.involved_entity_ids,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            geometry["involved_entity_ids"] = involved_entity_ids
            if record.geometry.before_position is not None:
                before_x, before_y, before_z = record.geometry.before_position
                geometry.update(
                    {
                        "before_x": before_x,
                        "before_y": before_y,
                        "before_z": before_z,
                    },
                )
            if record.geometry.after_position is not None:
                after_x, after_y, after_z = record.geometry.after_position
                geometry.update(
                    {
                        "after_x": after_x,
                        "after_y": after_y,
                        "after_z": after_z,
                    },
                )
            return _typed_candidate(
                record,
                snippet=(
                    f"{record.event_kind} event {record.instance_id} involves "
                    f"{involved_entity_ids} in "
                    f"{record.local_frame_id}"
                ),
                geometry=geometry,
            )
        case NoWriteMemoryRecord():
            return None


def _typed_candidate(
    record: TypedWritableRetrievalRecord,
    *,
    snippet: str,
    geometry: dict[str, float | str],
) -> RetrievalMemoryRecord:
    return RetrievalMemoryRecord(
        memory_id=record.memory_id,
        source_store="spatial",
        video_id=record.source_video_id,
        start_time=record.validity.start_time,
        end_time=record.validity.end_time,
        snippet=(
            f"{snippet} during "
            f"[{_format_float(record.validity.start_time)},"
            f"{_format_float(record.validity.end_time)}] "
            f"provenance={record.provenance}"
        ),
        frame_refs=record.evidence_refs,
        base_score=record.confidence,
        geometry=geometry,
    )


def _typed_geometry(
    record: TypedWritableRetrievalRecord,
    label: str,
) -> dict[str, float | str]:
    geometry: dict[str, float | str] = {
        "record_type": record.record_type,
        "entity_id": record.entity_id,
        "instance_id": record.instance_id,
        "label": label,
        "coordinate_frame": record.local_frame_id,
        "uncertainty_m": record.geometry_uncertainty.standard_deviation_m,
        "last_seen_time": record.last_seen_time,
        "confidence": record.confidence,
        "provenance": record.provenance,
    }
    if record.evidence_refs:
        geometry["evidence_refs"] = "\n".join(record.evidence_refs)
    return geometry


def _parse_retrieval_store(store: str) -> RetrievalStore:
    match store:
        case "episodic":
            return "episodic"
        case "semantic":
            return "semantic"
        case "visual":
            return "visual"
        case "spatial":
            return "spatial"
        case other:
            raise InvalidRetrievalStoreError(store=other)


def _relation_geometry_text(record: SpatialRelationRecord) -> str:
    if record.distance_m is None:
        return ""
    return (
        f" distance_m={_format_float(record.distance_m)}"
        f" delta=({_format_float(record.delta_x or 0.0)},"
        f"{_format_float(record.delta_y or 0.0)},"
        f"{_format_float(record.delta_z or 0.0)})"
    )


def _anchor_geometry(record: SpatialAnchorRecord) -> dict[str, float | str]:
    geometry: dict[str, float | str] = {
        "entity_id": record.instance_id or record.memory_id,
        "instance_id": record.instance_id or record.memory_id,
        "label": record.object_label,
        "x": record.x,
        "y": record.y,
        "z": record.z,
        "coordinate_frame": record.coordinate_frame,
        "last_seen_time": (
            record.end_time if record.last_seen_time is None else record.last_seen_time
        ),
        "provenance": record.provenance,
        "evidence_refs": "\n".join((record.memory_id, *record.frame_refs)),
    }
    if record.uncertainty_m is not None:
        geometry["uncertainty_m"] = record.uncertainty_m
    if record.geometry_frame_ref is not None:
        geometry["geometry_frame_ref"] = record.geometry_frame_ref
    if record.geometry_source is not None:
        geometry["geometry_source"] = record.geometry_source
    if record.geometry_distance_m is not None:
        geometry["geometry_distance_m"] = record.geometry_distance_m
    if record.geometry_embedding_ref is not None:
        geometry["geometry_embedding_ref"] = record.geometry_embedding_ref
    return geometry


def _relation_geometry(record: SpatialRelationRecord) -> dict[str, float | str] | None:
    if record.distance_m is None:
        return None
    geometry: dict[str, float | str] = {
        "relation": record.relation,
        "distance_m": record.distance_m,
        "coordinate_frame": record.coordinate_frame,
    }
    if record.subject_instance_id is not None:
        geometry["subject_instance_id"] = record.subject_instance_id
    if record.object_instance_id is not None:
        geometry["object_instance_id"] = record.object_instance_id
    if record.valid_from is not None:
        geometry["valid_from"] = record.valid_from
    if record.valid_to is not None:
        geometry["valid_to"] = record.valid_to
    if record.delta_x is not None:
        geometry["delta_x"] = record.delta_x
    if record.delta_y is not None:
        geometry["delta_y"] = record.delta_y
    if record.delta_z is not None:
        geometry["delta_z"] = record.delta_z
    return geometry


def _anchor_geometry_text(record: SpatialAnchorRecord) -> str:
    if record.geometry_frame_ref is None:
        return ""
    distance = (
        ""
        if record.geometry_distance_m is None
        else f" geometry_distance_m={_format_float(record.geometry_distance_m)}"
    )
    return (
        f" geometry_frame_ref={record.geometry_frame_ref}"
        f" geometry_source={record.geometry_source}"
        f"{distance}"
    )


def _score_candidate(
    question: QuestionRequest,
    record: RetrievalMemoryRecord,
) -> ScoredCandidate:
    query_terms = _query_terms(question)
    snippet_terms = _tokens(record.snippet)
    overlap = len(query_terms & snippet_terms)
    normalized_overlap = overlap / max(len(query_terms), 1)
    geometry_boost = _geometry_score_boost(query_terms, record)
    return ScoredCandidate(
        record=record,
        score=round(
            normalized_overlap + geometry_boost + (record.base_score * 0.01),
            6,
        ),
    )


def _geometry_score_boost(
    query_terms: frozenset[str],
    record: RetrievalMemoryRecord,
) -> float:
    if record.source_store != "spatial" or not query_terms & GEOMETRY_TERMS:
        return 0.0
    snippet_terms = _tokens(record.snippet)
    if query_terms & snippet_terms & GEOMETRY_TERMS:
        return 0.25
    if "geometry_source" in record.snippet or "distance_m" in record.snippet:
        return 0.1
    return 0.0


def _protocol_records(
    question: QuestionRequest,
    scoped: Sequence[RetrievalMemoryRecord],
    requested_stores: Sequence[RetrievalStore],
    *,
    chunks: Sequence[StreamChunk] | None,
) -> ProtocolSelection:
    if chunks is None:
        return _legacy_protocol_records(question, scoped, requested_stores)

    causal = tuple(
        record for record in scoped if record.end_time <= question.question_time
    )
    eligible_shards = eligible_video_rag_shards(question, chunks)
    shard_scoped = filter_records_to_shards(causal, eligible_shards)
    hierarchy = build_egobutler_hierarchy(chunks, scoped)
    coarse_selection = coarse_to_fine_candidates(question, hierarchy, scoped)
    selected_memory_ids = {record.memory_id for record in coarse_selection.records}
    selected_records = tuple(
        record for record in shard_scoped if record.memory_id in selected_memory_ids
    )
    return ProtocolSelection(
        records=selected_records,
        eligible_shard_ids=tuple(shard.chunk_id for shard in eligible_shards),
        selected_clip_ids=coarse_selection.selected_clip_ids,
        causal_filtered_count=len(scoped) - len(causal),
        candidate_counts=_candidate_counts(scoped, causal, requested_stores),
    )


def _legacy_protocol_records(
    question: QuestionRequest,
    scoped: Sequence[RetrievalMemoryRecord],
    requested_stores: Sequence[RetrievalStore],
) -> ProtocolSelection:
    causal = tuple(
        record for record in scoped if record.end_time <= question.question_time
    )
    shard_ids = _eligible_shard_ids(question, causal)
    shard_scoped = tuple(
        record for record in causal if _record_shard_id(record) in shard_ids
    )
    selected_clip_ids = _selected_clip_ids(question, shard_scoped)
    selected_records = tuple(
        record
        for record in shard_scoped
        if not selected_clip_ids or _record_clip_id(record) in selected_clip_ids
    )
    return ProtocolSelection(
        records=selected_records,
        eligible_shard_ids=shard_ids,
        selected_clip_ids=selected_clip_ids,
        causal_filtered_count=len(scoped) - len(causal),
        candidate_counts=_candidate_counts(scoped, causal, requested_stores),
    )


def _policy_evidence(
    scored: Sequence[ScoredCandidate],
    stores: Sequence[RetrievalStore],
    evidence_budget: int,
    *,
    max_frame_refs: int,
    geometry_query: bool = False,
) -> EvidenceSelection:
    if evidence_budget <= 0:
        return EvidenceSelection(items=(), frame_ref_count=0)

    selected: list[EvidenceItem] = []
    used_ids: set[str] = set()
    frame_refs: list[str] = []
    if geometry_query:
        for candidate in _geometry_bundle(scored):
            if len(selected) >= evidence_budget:
                break
            item, item_frame_refs = _evidence_item(
                candidate,
                remaining_frame_refs=max_frame_refs - len(frame_refs),
            )
            selected.append(item)
            frame_refs.extend(item_frame_refs)
            used_ids.add(candidate.record.memory_id)
    candidates_by_store = {
        store: iter(_store_candidates(scored, store)) for store in stores
    }
    while len(selected) < evidence_budget:
        added = False
        for store in stores:
            candidate = next(
                (
                    item
                    for item in candidates_by_store[store]
                    if item.record.memory_id not in used_ids
                ),
                None,
            )
            if candidate is None:
                continue
            added = True
            item, item_frame_refs = _evidence_item(
                candidate,
                remaining_frame_refs=max_frame_refs - len(frame_refs),
            )
            selected.append(item)
            frame_refs.extend(item_frame_refs)
            used_ids.add(candidate.record.memory_id)
            if len(selected) >= evidence_budget:
                break
        if not added:
            break
    return EvidenceSelection(
        items=tuple(selected),
        frame_ref_count=len(frame_refs),
    )


def _geometry_bundle(
    scored: Sequence[ScoredCandidate],
) -> tuple[ScoredCandidate, ...]:
    """Return one relation plus both endpoint objects needed by the executor."""
    relation = next(
        (
            candidate
            for candidate in scored
            if candidate.record.source_store == "spatial"
            and candidate.record.geometry is not None
            and "relation" in candidate.record.geometry
        ),
        None,
    )
    if relation is not None:
        if relation.record.geometry is None:
            return ()
        endpoint_ids = tuple(
            value
            for key in RELATION_ENDPOINT_KEYS
            if isinstance((value := relation.record.geometry.get(key)), str) and value
        )
        if len(endpoint_ids) != len(RELATION_ENDPOINT_KEYS):
            return ()
        endpoints = tuple(
            next(
                (
                    candidate
                    for candidate in scored
                    if candidate.record.source_store == "spatial"
                    and candidate.record.video_id == relation.record.video_id
                    and candidate.record.start_time <= relation.record.end_time
                    and candidate.record.end_time >= relation.record.start_time
                    and candidate.record.geometry is not None
                    and candidate.record.geometry.get("coordinate_frame")
                    == relation.record.geometry.get("coordinate_frame")
                    and entity_id
                    in (
                        candidate.record.geometry.get("entity_id"),
                        candidate.record.geometry.get("instance_id"),
                    )
                ),
                None,
            )
            for entity_id in endpoint_ids
        )
        if any(endpoint is None for endpoint in endpoints):
            return ()
        return (
            relation,
            *(endpoint for endpoint in endpoints if endpoint is not None),
        )

    typed_objects = tuple(
        candidate
        for candidate in scored
        if candidate.record.source_store == "spatial"
        and candidate.record.geometry is not None
        and candidate.record.geometry.get("record_type") == "object"
        and isinstance(candidate.record.geometry.get("entity_id"), str)
    )
    endpoint_count = len(RELATION_ENDPOINT_KEYS)
    for first in typed_objects:
        compatible = tuple(
            candidate
            for candidate in typed_objects
            if candidate.record.video_id == first.record.video_id
            and candidate.record.geometry is not None
            and first.record.geometry is not None
            and candidate.record.geometry.get("coordinate_frame")
            == first.record.geometry.get("coordinate_frame")
        )
        if len(compatible) >= endpoint_count:
            return compatible[:endpoint_count]
    return ()


def _store_candidates(
    scored: Sequence[ScoredCandidate],
    store: RetrievalStore,
) -> tuple[ScoredCandidate, ...]:
    return tuple(
        candidate for candidate in scored if candidate.record.source_store == store
    )


def _evidence_item(
    candidate: ScoredCandidate,
    *,
    remaining_frame_refs: int,
) -> tuple[EvidenceItem, tuple[str, ...]]:
    record = candidate.record
    frame_refs = cap_frame_refs(record.frame_refs, remaining_frame_refs)
    return (
        EvidenceItem(
            memory_id=record.memory_id,
            video_id=record.video_id,
            snippet=record.snippet,
            frame_refs=frame_refs,
            source_store=record.source_store,
            start_time=record.start_time,
            end_time=record.end_time,
            retrieval_score=candidate.score,
            geometry=record.geometry,
        ),
        frame_refs,
    )


def _eligible_shard_ids(
    question: QuestionRequest,
    causal_records: Sequence[RetrievalMemoryRecord],
) -> tuple[str, ...]:
    shard_ids = tuple(
        dict.fromkeys(
            _record_shard_id(record)
            for record in sorted(causal_records, key=_record_time_key)
            if _record_shard_end(record) <= question.question_time
        ),
    )
    if shard_ids or not causal_records:
        return shard_ids
    first_record = min(causal_records, key=_record_time_key)
    return (_record_shard_id(first_record),)


def _selected_clip_ids(
    question: QuestionRequest,
    records: Sequence[RetrievalMemoryRecord],
) -> tuple[str, ...]:
    selected: list[str] = []
    for video_id in _question_video_ids(question):
        clip_ids = tuple(
            dict.fromkeys(
                _record_clip_id(record)
                for record in sorted(records, key=_record_time_key)
                if record.video_id == video_id
            ),
        )
        if clip_ids:
            selected.append(
                min(
                    clip_ids,
                    key=lambda clip_id: _clip_sort_key(question, clip_id, records),
                ),
            )
    return tuple(selected)


def _candidate_counts(
    scoped: Sequence[RetrievalMemoryRecord],
    causal: Sequence[RetrievalMemoryRecord],
    stores: Sequence[RetrievalStore],
) -> tuple[RetrievalCandidateCount, ...]:
    return tuple(
        RetrievalCandidateCount(
            source_store=store,
            before_causal_filter=sum(
                1 for record in scoped if record.source_store == store
            ),
            after_causal_filter=sum(
                1 for record in causal if record.source_store == store
            ),
        )
        for store in stores
    )


def _clip_sort_key(
    question: QuestionRequest,
    clip_id: str,
    records: Sequence[RetrievalMemoryRecord],
) -> tuple[float, float, str]:
    snippet = " ".join(
        record.snippet for record in records if _record_clip_id(record) == clip_id
    )
    query_terms = _query_terms(question)
    score = 0.0
    if query_terms:
        score = len(query_terms & _tokens(snippet)) / len(query_terms)
    return (-score, _window_start_from_id(clip_id), clip_id)


def _record_shard_id(record: RetrievalMemoryRecord) -> str:
    start = floor(record.start_time / SHARD_SECONDS) * SHARD_SECONDS
    return _window_id(
        record.video_id,
        start,
        start + SHARD_SECONDS,
        "shard_30m",
    )


def _record_shard_end(record: RetrievalMemoryRecord) -> float:
    start = floor(record.start_time / SHARD_SECONDS) * SHARD_SECONDS
    return start + SHARD_SECONDS


def _record_clip_id(record: RetrievalMemoryRecord) -> str:
    start = floor(record.start_time / CLIP_SECONDS) * CLIP_SECONDS
    return _window_id(
        record.video_id,
        start,
        start + CLIP_SECONDS,
        "clip_30s",
    )


def _window_id(
    video_id: str,
    start_time: float,
    end_time: float,
    granularity: str,
) -> str:
    return (
        f"{video_id}:{_format_float(start_time)}:"
        f"{_format_float(end_time)}:{granularity}"
    )


def _window_start_from_id(window_id: str) -> float:
    return float(window_id.split(":")[1])


def _record_time_key(record: RetrievalMemoryRecord) -> tuple[float, float, str]:
    return (record.start_time, record.end_time, record.memory_id)


def _zone_time_span(record: ZoneRecord) -> tuple[float, float]:
    starts = tuple(start for start, _end in record.visit_intervals)
    ends = tuple(end for _start, end in record.visit_intervals)
    return min(starts), max(ends)


def _format_float(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _query_terms(question: QuestionRequest) -> frozenset[str]:
    return _tokens(
        " ".join(
            (
                question.question,
                *(choice.text for choice in question.answer_choices),
            ),
        ),
    )


def _tokens(text: str) -> frozenset[str]:
    cleaned = "".join(char if char.isalnum() else " " for char in text.lower())
    return frozenset(token for token in cleaned.split() if token not in STOP_WORDS)


def _score_sort_key(candidate: ScoredCandidate) -> tuple[float, str, float, str]:
    record = candidate.record
    return (-candidate.score, record.source_store, -record.end_time, record.memory_id)


def _ordered_stores(stores: frozenset[RetrievalStore]) -> tuple[RetrievalStore, ...]:
    return tuple(store for store in STORE_ORDER if store in stores)


def _episodic_snippet(record: EpisodicNodeRecord) -> str:
    if record.summary:
        return f"{record.summary} {record.granularity}"
    return " ".join(
        (
            record.granularity,
            *record.source_modalities,
            *record.source_modality_refs,
            *record.frame_refs,
        ),
    )
