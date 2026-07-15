from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import sys
import sysconfig
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Final,
    Literal,
    NoReturn,
    Protocol,
    cast,
    override,
)

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from worldmm_smvqa.chunking import read_source_streams
from worldmm_smvqa.config import REMOTE_ENV_FLAG, RemoteOnlyError
from worldmm_smvqa.fixtures import read_fixture_questions
from worldmm_smvqa.qa import evidence_pack_validation_error
from worldmm_smvqa.qa_shards import (
    DistributedEnv,
    QAShardError,
    QAShardLineage,
    QuestionShardMap,
    checkpoint_rank,
    complete_rank,
    distributed_env,
    load_rank_progress,
    merge_sealed_shards,
    merge_shards,
    packs_for_rank,
    packs_for_rank_from_map,
    partial_output_path,
    rank_output_path,
    sealed_checkpoint_rank,
    sealed_complete_rank,
    sealed_load_rank_progress,
    wait_for_shards,
)
from worldmm_smvqa.retrieval import read_retrieval_memory_artifacts
from worldmm_smvqa.retrieval_protocols import cap_frame_refs
from worldmm_smvqa.retrieval_types import (
    RETRIEVAL_FRAME_REF_CAP,
    EvidenceLane,
    EvidenceLineage,
    EvidencePack,
    OracleQAPreEvaluationLineage,
    OracleVariant,
    SharedQALineage,
    canonical_oracle_to_evidence_pack,
    load_canonical_oracle_evidence_pack,
)
from worldmm_smvqa.sensor_frames import (
    SensorFrameManifestError,
    apply_sensor_frame_manifest,
    read_sensor_frame_manifest,
)
from worldmm_smvqa.transformers_backend import TransformersGenerationError
from worldmm_smvqa.video_frames import sample_video_frames
from worldmm_smvqa.worldmm.spatial_sensor import is_trusted_causal_pose
from worldmm_smvqa.worldmm.typed_memory import (
    DEFAULT_TYPED_MEMORY_WINDOW_SECONDS,
    TypedMemoryWriterError,
    validate_typed_memory_artifact,
)

if TYPE_CHECKING:
    from worldmm_smvqa.qa import QABackend
    from worldmm_smvqa.retrieval_types import RetrievalMemoryRecord, RetrievalStore
    from worldmm_smvqa.schema import (
        PredictionRecord,
        QuestionRequest,
        SourceStreamExample,
    )
    from worldmm_smvqa.sensor_frames import SensorFrameManifestRecord

type TransformersBackendName = Literal["gemma4", "real", "mock"]

QA_RESUME_MANIFEST_VERSION: Final = "qa-resume-manifest-v6"
# Bump whenever resume-bound inputs or prompt/prediction serialization changes.
QA_PROMPT_SCHEMA_VERSION: Final = "qa-prompt-prediction-schema-v4"
SHA256_HEX_LENGTH: Final = 64
MEMORY_MANIFEST_MAX_BYTES: Final = 64 * 1024
TYPED_VALIDATION_SEAL_VERSION: Final = 1
QA_DISTRIBUTED_TIMEOUT_MAX_SECONDS: Final = 24 * 60 * 60


class _DistributedObjectBroadcaster(Protocol):
    def is_available(self) -> bool: ...

    def is_initialized(self) -> bool: ...

    def get_rank(self) -> int: ...

    def get_world_size(self) -> int: ...

    def barrier(self) -> None: ...

    def all_gather_object(
        self,
        object_list: list[object],
        obj: object,
    ) -> None: ...

    def broadcast_object_list(
        self,
        object_list: list[object],
        src: int = 0,
    ) -> None: ...


class _DistributedProcessGroup(_DistributedObjectBroadcaster, Protocol):
    def init_process_group(
        self,
        backend: str,
        *,
        init_method: str,
        timeout: timedelta,
    ) -> None: ...

    def destroy_process_group(self) -> None: ...


class TypedMemoryValidationSeal(BaseModel):
    """Rank-0 result broadcast after full typed-memory validation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    schema_version: Literal[1] = TYPED_VALIDATION_SEAL_VERSION
    lineage: EvidenceLineage
    typed_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inference_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(gt=0)
    actual_bytes: int = Field(ge=0)
    window_count: int = Field(gt=0)
    max_window_bytes: int = Field(ge=0)
    window_seconds: float = Field(gt=0.0)


class _DistributedValidationSuccess(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    schema_version: Literal[1] = TYPED_VALIDATION_SEAL_VERSION
    status: Literal["ok"] = "ok"
    seal: TypedMemoryValidationSeal


class _DistributedValidationFailure(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    schema_version: Literal[1] = TYPED_VALIDATION_SEAL_VERSION
    status: Literal["error"] = "error"
    detail: str = Field(min_length=1, max_length=4096)


class _DistributedRankReadSuccess(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    schema_version: Literal[1] = TYPED_VALIDATION_SEAL_VERSION
    status: Literal["ok"] = "ok"
    rank: int = Field(ge=0)


class _DistributedRankReadFailure(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    schema_version: Literal[1] = TYPED_VALIDATION_SEAL_VERSION
    status: Literal["error"] = "error"
    rank: int = Field(ge=0)
    detail: str = Field(min_length=1, max_length=4096)


@dataclass(frozen=True, slots=True)
class TransformersCliArgs:
    model: str
    fixture: Path
    evidence: Path
    evidence_lane: EvidenceLane
    evidence_lineage: Path | None
    checkpoint: Path | None
    typed_memory: Path | None
    inference_manifest: Path | None
    require_frames: bool
    out: Path
    backend: TransformersBackendName
    model_fingerprint: Path | None = None
    frame_assets_manifest: Path | None = None
    lineage_config: Path | None = None
    sensor_frame_manifest: Path | None = None
    sensor_audit: Path | None = None
    memory_manifest: Path | None = None
    inference_sources: Path | None = None
    inference_producer: Path | None = None
    approved_shard_map: Path | None = None
    qa_shard_lineage: Path | None = None
    expected_variant: OracleVariant | None = None


@dataclass(frozen=True, slots=True)
class TransformersCliResult:
    written: Path
    predictions: int


@dataclass(slots=True)
class TransformersCliUsageError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"UsageError: {self.detail}"


def run_transformers_cli(  # noqa: PLR0912,PLR0915
    args: TransformersCliArgs,
    env: Mapping[str, str],
    *,
    validation_process_group: _DistributedProcessGroup | None = None,
) -> TransformersCliResult:
    from worldmm_smvqa.qa import (  # noqa: PLC0415
        Gemma4QABackend,
        MockQABackend,
        QABackendUnavailableError,
        QAParseError,
        parse_qa_output,
    )
    from worldmm_smvqa.qa_prompt import build_qa_prompt  # noqa: PLC0415
    from worldmm_smvqa.worldmm.geometry_executor import (  # noqa: PLC0415
        geometry_proofs_for_question,
    )

    if args.evidence_lane == "teacher_oracle" and args.backend == "mock":
        raise TransformersCliUsageError(detail="teacher_oracle forbids MockQABackend")
    questions = _questions_by_id(args.fixture)
    sensor_manifest_path = args.sensor_frame_manifest
    if sensor_manifest_path is None:
        sensor_records = None
        sources = read_source_streams(args.fixture)
    else:
        sensor_records = read_sensor_frame_manifest(sensor_manifest_path)
        sources = apply_sensor_frame_manifest(
            read_source_streams(args.fixture, use_sensor_manifest=False),
            sensor_records,
            path=sensor_manifest_path,
        )
    packs = _read_evidence_packs(
        args.evidence,
        canonical_oracle=args.evidence_lane == "teacher_oracle",
        expected_variant=args.expected_variant,
    )
    distributed = distributed_env(env)
    sealed_shard_map: QuestionShardMap | None = None
    sealed_lineage: QAShardLineage | None = None
    if args.evidence_lane == "teacher_oracle":
        sealed_shard_map, sealed_lineage = _read_sealed_qa_contract(
            args.approved_shard_map,
            args.qa_shard_lineage,
        )
        _validate_teacher_oracle_contract(
            args,
            env,
            packs,
            sealed_shard_map,
            sealed_lineage,
        )
        if sealed_shard_map.world_size != distributed.world_size:
            raise TransformersCliUsageError(
                detail=(
                    "approved teacher-oracle shard map world_size does not "
                    "match runtime"
                ),
            )
    if args.evidence_lane == "student":
        if args.model_fingerprint is None:
            raise TransformersCliUsageError(
                detail="student QA requires --model-fingerprint",
            )
        if args.frame_assets_manifest is None:
            raise TransformersCliUsageError(
                detail="student QA requires --frame-assets-manifest",
            )
        if args.lineage_config is None:
            raise TransformersCliUsageError(
                detail="student QA requires --lineage-config",
            )
        if args.sensor_frame_manifest is None:
            raise TransformersCliUsageError(
                detail="student QA requires --sensor-frame-manifest",
            )
        if args.memory_manifest is None:
            raise TransformersCliUsageError(
                detail="student QA requires --memory-manifest",
            )
        if args.inference_sources is None:
            raise TransformersCliUsageError(
                detail="student QA requires --inference-sources",
            )
        if args.inference_producer is None:
            raise TransformersCliUsageError(
                detail="student QA requires --inference-producer",
            )
        _ = _sha256_file(args.model_fingerprint)
        _ = _sha256_file(args.frame_assets_manifest)

    def validate_lineage() -> EvidenceLineage | None:
        return validate_evidence_lineage(
            args.evidence,
            args.evidence_lane,
            args.evidence_lineage,
            args.checkpoint,
            args.typed_memory,
            args.inference_manifest,
            config_path=args.lineage_config,
            sensor_path=args.sensor_frame_manifest,
            data_root=args.fixture,
            memory_manifest_path=args.memory_manifest,
            inference_sources_path=args.inference_sources,
            frame_assets_path=args.frame_assets_manifest,
            inference_producer_path=args.inference_producer,
            sources=sources,
            sensor_records=sensor_records,
        )

    validation_seal: TypedMemoryValidationSeal | None = None
    if args.evidence_lane == "student":
        validation_seal = synchronize_typed_memory_validation(
            validate_lineage,
            args.inference_manifest,
            distributed,
        )
        evidence_lineage = validation_seal.lineage
    elif args.evidence_lane == "teacher_oracle":
        evidence_lineage = None
    else:
        evidence_lineage = validate_lineage()
    validate_evidence_trace_lane(packs, args.evidence_lane)
    validate_external_evidence_packs(packs, questions)
    if sealed_shard_map is None:
        rank_packs = packs_for_rank(packs, distributed)
    else:
        rank_packs = packs_for_rank_from_map(packs, distributed, sealed_shard_map)
    rank_video_ids = frozenset(
        video_id
        for pack in rank_packs
        for video_id in (
            questions[pack.question_id].video_ids
            or (questions[pack.question_id].video_id,)
        )
    )
    typed_spatial_records: tuple[RetrievalMemoryRecord, ...] = ()
    if args.evidence_lane == "student":
        if args.memory_manifest is None:
            raise TransformersCliUsageError(
                detail="student evidence requires --memory-manifest",
            )
        rank_memory_ids_by_store: dict[RetrievalStore, frozenset[str]] = {
            "episodic": frozenset(
                item.memory_id
                for pack in rank_packs
                for item in pack.evidence
                if item.source_store == "episodic"
            ),
            "semantic": frozenset(
                item.memory_id
                for pack in rank_packs
                for item in pack.evidence
                if item.source_store == "semantic"
            ),
            "visual": frozenset(
                item.memory_id
                for pack in rank_packs
                for item in pack.evidence
                if item.source_store == "visual"
            ),
        }
        canonical_records: tuple[RetrievalMemoryRecord, ...] = ()
        rank_read_error: Exception | None = None
        try:
            canonical_records = read_retrieval_memory_artifacts(
                args.memory_manifest,
                video_ids=rank_video_ids,
                memory_ids_by_store=rank_memory_ids_by_store,
            )
            validate_student_evidence_against_memory(
                rank_packs,
                canonical_records,
            )
        except Exception as exc:  # noqa: BLE001 - synchronize failure across ranks.
            rank_read_error = exc
        synchronize_rank_memory_read(rank_read_error, distributed)
        if evidence_lineage is None:
            raise TransformersCliUsageError(
                detail="student evidence requires valid evidence lineage",
            )
        if validation_seal is None:
            raise TransformersCliUsageError(
                detail="student evidence requires typed validation seal",
            )
        synchronize_typed_memory_postread(
            validation_seal,
            args.memory_manifest,
            args.inference_manifest,
            distributed,
        )
        typed_spatial_records = tuple(
            record for record in canonical_records if record.source_store == "spatial"
        )
    typed_geometry_by_video: dict[str, list[RetrievalMemoryRecord]] = {}
    for record in typed_spatial_records:
        if record.geometry is not None and record.geometry.get("record_type") in {
            "event",
            "object",
        }:
            typed_geometry_by_video.setdefault(record.video_id, []).append(record)
    if validation_process_group is not None:
        try:
            validation_process_group.barrier()
            validation_process_group.destroy_process_group()
        except Exception as exc:
            raise TransformersCliUsageError(
                detail=f"cannot close distributed QA validation group: {exc}",
            ) from exc
    frame_root = Path(env.get("SMVQA_FRAME_ROOT", args.fixture / "frames"))
    match args.backend:
        case "mock":
            backend: QABackend = MockQABackend()
        case "gemma4" | "real":
            if env.get(REMOTE_ENV_FLAG) != "1":
                raise RemoteOnlyError(action="qa_transformers real model")
            backend = Gemma4QABackend(model_path=args.model)

    written = rank_output_path(args.out, distributed)
    if sealed_shard_map is None or sealed_lineage is None:
        _bind_resume_manifest(args, written)
        predictions = list(load_rank_progress(written))
    else:
        predictions = list(
            sealed_load_rank_progress(
                written,
                sealed_shard_map,
                distributed,
                sealed_lineage,
            )
        )
    _validate_rank_progress(rank_packs, predictions, completed=written.exists())
    completed_by_question = {
        prediction.question_id: prediction for prediction in predictions
    }
    for pack in rank_packs:
        question = _question_for_pack(pack, questions)
        video_frames = sample_video_frames(
            sources,
            question,
            pack,
            frame_root=frame_root,
            max_frames=32,
        )
        missing_frame_refs = tuple(
            frame.frame_ref
            for frame in video_frames
            if frame.path is None or not frame.path.is_file()
        )
        if args.require_frames and (not video_frames or missing_frame_refs):
            raise TransformersCliUsageError(
                detail=(
                    f"required QA frame missing: {question.question_id}: "
                    f"{','.join(missing_frame_refs) or 'no selected frames'}"
                ),
            )
        wearer_pose = causal_wearer_pose(sources, question, pack)
        geometry_proofs = geometry_proofs_for_question(
            question,
            pack,
            wearer_yaw_degrees=None if wearer_pose is None else wearer_pose[0],
            wearer_yaw_uncertainty_degrees=(
                0.0 if wearer_pose is None else wearer_pose[1]
            ),
            spatial_records=(
                tuple(
                    record
                    for video_id in (question.video_ids or (question.video_id,))
                    for record in typed_geometry_by_video.get(video_id, ())
                    if record.end_time <= question.question_time
                )
                if args.evidence_lane == "student"
                else None
            ),
        )
        prompt = build_qa_prompt(
            question,
            pack,
            video_frames,
            geometry_proofs,
        )
        input_frame_refs = tuple(
            f"{frame.video_id}/{frame.frame_ref}" for frame in video_frames
        )
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        if completed := completed_by_question.get(pack.question_id):
            current_proofs = {proof.proof_id: proof for proof in geometry_proofs}
            resumed_proofs = tuple(
                current_proofs[proof_id].model_dump(mode="json")
                for proof_id in completed.geometry_proof_ids
                if proof_id in current_proofs and current_proofs[proof_id].answerable
            )
            if (
                completed.input_frame_refs != input_frame_refs
                or completed.prompt_sha256 != prompt_sha256
                or len(resumed_proofs) != len(completed.geometry_proof_ids)
                or completed.geometry_proofs != resumed_proofs
                or (
                    any(proof.answerable for proof in geometry_proofs)
                    and not completed.geometry_proof_ids
                )
            ):
                raise QAShardError(
                    detail=(
                        "resumed prediction does not match current prompt/frames/"
                        f"geometry: {pack.question_id}"
                    ),
                )
            continue
        raw_outputs: list[str] = []
        raw_output_path = _raw_output_path(args.out, question.question_id)
        try:
            prediction: PredictionRecord | None = None
            last_parse_error: QAParseError | None = None
            for _attempt in range(2):
                raw_outputs.extend(
                    backend.raw_outputs(prompt, question, pack, video_frames),
                )
                _write_raw_outputs(raw_output_path, raw_outputs)
                try:
                    prediction = parse_qa_output(
                        question=question,
                        raw_outputs=raw_outputs,
                        prompt_token_count=len(prompt.split()),
                        raw_model_output_path=str(raw_output_path),
                        evidence_pack=pack,
                        geometry_proofs=geometry_proofs,
                        input_frame_refs=input_frame_refs,
                        prompt_sha256=prompt_sha256,
                    )
                except QAParseError as exc:
                    last_parse_error = exc
                    continue
                break
            if prediction is None:
                if last_parse_error is None:
                    raise TransformersCliUsageError(
                        detail=f"no QA output for {question.question_id}",
                    )
                raise last_parse_error
            predictions.append(prediction)
            if sealed_shard_map is None or sealed_lineage is None:
                checkpoint_rank(written, predictions)
            else:
                sealed_checkpoint_rank(
                    written,
                    predictions,
                    sealed_shard_map,
                    distributed,
                    sealed_lineage,
                )
        except (QABackendUnavailableError, QAParseError) as exc:
            raise TransformersCliUsageError(detail=str(exc)) from exc
    if sealed_shard_map is None or sealed_lineage is None:
        complete_rank(written, predictions)
        if distributed.world_size == 1:
            return TransformersCliResult(written=written, predictions=len(predictions))
        if distributed.rank == 0:
            wait_for_shards(args.out, distributed.world_size, env)
            merge_shards(args.out, packs, distributed.world_size)
            return TransformersCliResult(written=args.out, predictions=len(packs))
        return TransformersCliResult(written=written, predictions=len(predictions))
    sealed_complete_rank(
        written,
        predictions,
        sealed_shard_map,
        distributed,
        sealed_lineage,
    )
    if distributed.rank == 0:
        merge_sealed_shards(args.out, packs, sealed_shard_map, sealed_lineage)
        return TransformersCliResult(written=args.out, predictions=len(packs))
    return TransformersCliResult(written=written, predictions=len(predictions))


def causal_wearer_pose(
    sources: Sequence[SourceStreamExample],
    question: QuestionRequest,
    pack: EvidencePack,
) -> tuple[float, float] | None:
    """Return latest causal trusted yaw/std in the spatial evidence frame."""
    spatial = tuple(
        item
        for item in pack.evidence
        if item.source_store == "spatial" and item.geometry is not None
    )
    video_ids = {item.video_id for item in spatial}
    coordinate_frames: set[str] = set()
    for item in spatial:
        geometry = item.geometry
        if geometry is not None and isinstance(
            frame := geometry.get("coordinate_frame"),
            str,
        ):
            coordinate_frames.add(frame)
    if len(video_ids) != 1 or len(coordinate_frames) != 1:
        return None
    video_id = next(iter(video_ids))
    coordinate_frame = next(iter(coordinate_frames))
    source = next((item for item in sources if item.video_id == video_id), None)
    if source is None or video_id not in (question.video_ids or (question.video_id,)):
        return None
    trusted = tuple(
        sample
        for sample in source.pose_samples
        if is_trusted_causal_pose(
            sample,
            cutoff_time=question.question_time,
            coordinate_frame=coordinate_frame,
        )
    )
    if not trusted:
        return None
    pose = max(trusted, key=lambda sample: sample.timestamp)
    covariance = cast("tuple[float, ...]", pose.pose_covariance_xyz_m_rpy_deg)
    return (cast("float", pose.yaw_degrees), math.sqrt(covariance[35]))


def parse_cli_args(argv: Sequence[str]) -> TransformersCliArgs:  # noqa: PLR0912,PLR0915
    model: str | None = None
    fixture: Path | None = None
    evidence: Path | None = None
    evidence_lane: EvidenceLane | None = None
    evidence_lineage: Path | None = None
    checkpoint: Path | None = None
    typed_memory: Path | None = None
    inference_manifest: Path | None = None
    model_fingerprint: Path | None = None
    frame_assets_manifest: Path | None = None
    lineage_config: Path | None = None
    sensor_frame_manifest: Path | None = None
    sensor_audit: Path | None = None
    memory_manifest: Path | None = None
    inference_sources: Path | None = None
    inference_producer: Path | None = None
    approved_shard_map: Path | None = None
    qa_shard_lineage: Path | None = None
    expected_variant: OracleVariant | None = None
    require_frames = False
    out: Path | None = None
    backend: TransformersBackendName = "gemma4"
    index = 0
    while index < len(argv):
        option = argv[index]
        if option == "--require-frames":
            require_frames = True
            index += 1
            continue
        if option in {
            "--model",
            "--fixture",
            "--evidence",
            "--evidence-lane",
            "--evidence-lineage",
            "--checkpoint",
            "--typed-memory",
            "--inference-manifest",
            "--model-fingerprint",
            "--frame-assets-manifest",
            "--lineage-config",
            "--sensor-frame-manifest",
            "--sensor-audit",
            "--memory-manifest",
            "--inference-sources",
            "--inference-producer",
            "--approved-shard-map",
            "--qa-shard-lineage",
            "--expected-variant",
            "--out",
            "--backend",
        }:
            if index + 1 >= len(argv):
                raise TransformersCliUsageError(detail=f"missing value for {option}")
            value = argv[index + 1]
            if option == "--model":
                model = value
            elif option == "--fixture":
                fixture = Path(value)
            elif option == "--evidence":
                evidence = Path(value)
            elif option == "--evidence-lane":
                evidence_lane = _parse_evidence_lane(value)
            elif option == "--evidence-lineage":
                evidence_lineage = Path(value)
            elif option == "--checkpoint":
                checkpoint = Path(value)
            elif option == "--typed-memory":
                typed_memory = Path(value)
            elif option == "--inference-manifest":
                inference_manifest = Path(value)
            elif option == "--model-fingerprint":
                model_fingerprint = Path(value)
            elif option == "--frame-assets-manifest":
                frame_assets_manifest = Path(value)
            elif option == "--lineage-config":
                lineage_config = Path(value)
            elif option == "--sensor-frame-manifest":
                sensor_frame_manifest = Path(value)
            elif option == "--sensor-audit":
                sensor_audit = Path(value)
            elif option == "--memory-manifest":
                memory_manifest = Path(value)
            elif option == "--inference-sources":
                inference_sources = Path(value)
            elif option == "--inference-producer":
                inference_producer = Path(value)
            elif option == "--approved-shard-map":
                approved_shard_map = Path(value)
            elif option == "--qa-shard-lineage":
                qa_shard_lineage = Path(value)
            elif option == "--expected-variant":
                expected_variant = _parse_oracle_variant(value)
            elif option == "--out":
                out = Path(value)
            elif option == "--backend":
                backend = _parse_backend(value)
            index += 2
            continue
        raise TransformersCliUsageError(detail=f"unknown option: {option}")
    if model is None:
        raise TransformersCliUsageError(detail="qa_transformers requires --model")
    if fixture is None:
        raise TransformersCliUsageError(detail="qa_transformers requires --fixture")
    if evidence is None:
        raise TransformersCliUsageError(detail="qa_transformers requires --evidence")
    if evidence_lane is None:
        raise TransformersCliUsageError(
            detail="qa_transformers requires --evidence-lane",
        )
    if out is None:
        raise TransformersCliUsageError(detail="qa_transformers requires --out")
    return TransformersCliArgs(
        model=model,
        fixture=fixture,
        evidence=evidence,
        evidence_lane=evidence_lane,
        evidence_lineage=evidence_lineage,
        checkpoint=checkpoint,
        typed_memory=typed_memory,
        inference_manifest=inference_manifest,
        require_frames=require_frames,
        out=out,
        backend=backend,
        model_fingerprint=model_fingerprint,
        frame_assets_manifest=frame_assets_manifest,
        lineage_config=lineage_config,
        sensor_frame_manifest=sensor_frame_manifest,
        sensor_audit=sensor_audit,
        memory_manifest=memory_manifest,
        inference_sources=inference_sources,
        approved_shard_map=approved_shard_map,
        qa_shard_lineage=qa_shard_lineage,
        expected_variant=expected_variant,
        inference_producer=inference_producer,
    )


def main(argv: Sequence[str] | None = None) -> int:
    owned_process_group: _DistributedProcessGroup | None = None
    try:
        if cache_root := os.environ.get("WORLDMM_TRITON_CACHE_ROOT"):
            rank = distributed_env(os.environ).rank
            cache_dir = Path(cache_root) / f"rank-{rank:05d}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
        args = parse_cli_args(sys.argv[1:] if argv is None else argv)
        if args.evidence_lane == "student":
            owned_process_group = initialize_qa_validation_process_group(os.environ)
        result = run_transformers_cli(
            args,
            env=os.environ,
            validation_process_group=owned_process_group,
        )
    except (
        TransformersCliUsageError,
        RemoteOnlyError,
        QAShardError,
        SensorFrameManifestError,
        TransformersGenerationError,
        OSError,
        ValidationError,
    ) as exc:
        _ = sys.stderr.write(f"{exc}\n")
        return 2
    finally:
        if owned_process_group is not None and owned_process_group.is_initialized():
            with suppress(Exception):
                owned_process_group.destroy_process_group()
    _ = sys.stdout.write(f"wrote {result.written}\npredictions={result.predictions}\n")
    return 0


def _read_evidence_packs(
    path: Path,
    *,
    canonical_oracle: bool = False,
    expected_variant: OracleVariant | None = None,
) -> tuple[EvidencePack, ...]:
    packs: list[EvidencePack] = []
    canonical_variants: set[OracleVariant] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            if canonical_oracle:
                canonical = load_canonical_oracle_evidence_pack(line)
                canonical_variants.add(canonical.variant)
                packs.append(canonical_oracle_to_evidence_pack(canonical))
            else:
                packs.append(EvidencePack.model_validate_json(line))
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            detail = f"{path}: line {line_number}: {exc}"
            raise TransformersCliUsageError(detail=detail) from exc
    if canonical_oracle:
        if expected_variant is None:
            raise TransformersCliUsageError(
                detail="teacher_oracle requires --expected-variant"
            )
        if canonical_variants != {expected_variant}:
            raise TransformersCliUsageError(
                detail=(
                    "teacher-oracle evidence must contain exactly the expected variant"
                )
            )
    return tuple(packs)


def _read_sealed_qa_contract(
    shard_map_path: Path | None,
    lineage_path: Path | None,
) -> tuple[QuestionShardMap, QAShardLineage]:
    """Load the immutable EXP-0005 shard contract before any QA work begins."""
    if shard_map_path is None:
        raise TransformersCliUsageError(
            detail="teacher_oracle requires --approved-shard-map",
        )
    if lineage_path is None:
        raise TransformersCliUsageError(
            detail="teacher_oracle requires --qa-shard-lineage",
        )
    try:
        shard_map = QuestionShardMap.model_validate_json(
            shard_map_path.read_text(encoding="utf-8"),
        )
        lineage = QAShardLineage.model_validate_json(
            lineage_path.read_text(encoding="utf-8"),
        )
    except (OSError, ValidationError) as exc:
        raise TransformersCliUsageError(
            detail=f"invalid sealed teacher-oracle QA contract: {exc}",
        ) from exc
    if shard_map.world_size != lineage.world_size:
        raise TransformersCliUsageError(
            detail=(
                "sealed teacher-oracle QA lineage world_size does not match shard map"
            ),
        )
    if (
        shard_map.approved_salt != lineage.approved_salt
        or shard_map.sha256 != lineage.question_map_sha256
    ):
        raise TransformersCliUsageError(
            detail="sealed teacher-oracle QA lineage does not match shard map",
        )
    return shard_map, lineage


def read_teacher_oracle_pre_evaluation_lineage(
    path: Path | None,
) -> OracleQAPreEvaluationLineage:
    """Read only the output-free contract required before teacher QA begins."""
    if path is None:
        raise TransformersCliUsageError(
            detail="teacher_oracle requires --evidence-lineage"
        )
    try:
        return OracleQAPreEvaluationLineage.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise TransformersCliUsageError(
            detail=f"invalid teacher-oracle pre-evaluation lineage: {path}: {exc}"
        ) from exc


def _validate_teacher_oracle_contract(
    args: TransformersCliArgs,
    env: Mapping[str, str],
    packs: Sequence[EvidencePack],
    shard_map: QuestionShardMap,
    shard_lineage: QAShardLineage,
) -> None:
    """Bind the live teacher QA invocation to approved evidence and runtime lineage."""
    if args.expected_variant is None:
        raise TransformersCliUsageError(
            detail="teacher_oracle requires --expected-variant"
        )
    oracle_lineage = read_teacher_oracle_pre_evaluation_lineage(args.evidence_lineage)
    qa_input = next(
        (
            item
            for item in oracle_lineage.qa_inputs
            if item.variant == args.expected_variant
        ),
        None,
    )
    evidence_sha256 = _sha256_file(args.evidence)
    if (
        qa_input is None
        or qa_input.evidence_sha256 != evidence_sha256
        or shard_lineage.evidence_sha256 != evidence_sha256
    ):
        raise TransformersCliUsageError(
            detail="teacher-oracle evidence bytes do not match approved QA inputs"
        )
    _validate_teacher_oracle_shard_lineage(
        shard_map,
        shard_lineage,
        oracle_lineage.shared_qa_lineage,
    )
    validate_teacher_oracle_live_contract(args, env, oracle_lineage)
    if qa_input.pre_evaluation_sha256 != oracle_lineage.shared_qa_lineage.sha256:
        raise TransformersCliUsageError(
            detail=(
                "teacher-oracle QA inputs do not match approved pre-evaluation lineage"
            )
        )
    if not packs:
        raise TransformersCliUsageError(
            detail="teacher-oracle evidence must not be empty"
        )


def _validate_teacher_oracle_shard_lineage(
    shard_map: QuestionShardMap,
    shard_lineage: QAShardLineage,
    shared: SharedQALineage,
) -> None:
    if (
        shard_lineage.approved_salt != shared.approved_salt
        or shard_lineage.world_size != shared.world_size
        or shard_lineage.question_map_sha256 != shared.question_map_sha256
        or shard_lineage.model_sha256 != shared.model_sha256
        or shard_lineage.prompt_sha256 != shared.prompt_sha256
        or shard_lineage.decoding_sha256 != shared.decoding_sha256
        or shard_lineage.runtime_sha256 != shared.runtime_sha256
        or shard_lineage.seed != shared.seed
    ):
        raise TransformersCliUsageError(
            detail=(
                "teacher-oracle shard lineage does not match approved "
                "QA runtime lineage"
            )
        )
    if shard_map.sha256 != shared.question_map_sha256:
        raise TransformersCliUsageError(
            detail=(
                "teacher-oracle shard map does not match approved QA runtime lineage"
            )
        )


def validate_teacher_oracle_live_contract(
    args: TransformersCliArgs,
    env: Mapping[str, str],
    oracle_lineage: OracleQAPreEvaluationLineage,
) -> None:
    shared = oracle_lineage.shared_qa_lineage
    _validate_teacher_oracle_backend_and_frames(args, oracle_lineage, shared)
    live_lineage = _live_teacher_oracle_lineage(args, env)
    approved_inventories = (
        shared.python_inventory_sha256,
        shared.torch_inventory_sha256,
        shared.transformers_inventory_sha256,
    )
    if any(inventory is None for inventory in approved_inventories):
        raise TransformersCliUsageError(
            detail="teacher-oracle approved runtime inventories are required"
        )
    if approved_inventories != live_lineage[4:]:
        raise TransformersCliUsageError(
            detail=(
                "teacher-oracle live runtime inventories do not match approved contract"
            )
        )
    if (
        shared.prompt_sha256 != live_lineage[0]
        or shared.decoding_sha256 != live_lineage[1]
        or shared.runtime_sha256 != live_lineage[2]
        or shared.seed != live_lineage[3]
        or oracle_lineage.shared_input_sha256 != _fixture_data_sha256(args.fixture)
    ):
        raise TransformersCliUsageError(
            detail="teacher-oracle live QA lineage does not match approved contract"
        )


def _validate_teacher_oracle_backend_and_frames(
    args: TransformersCliArgs,
    oracle_lineage: OracleQAPreEvaluationLineage,
    shared: SharedQALineage,
) -> None:
    if args.backend == "mock":
        raise TransformersCliUsageError(detail="teacher_oracle forbids MockQABackend")
    if args.backend not in {"gemma4", "real"}:
        raise TransformersCliUsageError(
            detail="teacher_oracle requires the live Gemma4 backend"
        )
    if args.model_fingerprint is None:
        raise TransformersCliUsageError(
            detail="teacher_oracle requires --model-fingerprint"
        )
    if _sha256_file(args.model_fingerprint) != shared.model_sha256:
        raise TransformersCliUsageError(
            detail=(
                "teacher-oracle model bytes do not match approved QA runtime lineage"
            )
        )
    _verify_checksum_inventory(
        args.model_fingerprint, Path(args.model), "teacher-oracle model"
    )
    if (
        not args.require_frames
        or args.sensor_frame_manifest is None
        or args.sensor_audit is None
    ):
        raise TransformersCliUsageError(
            detail=(
                "teacher_oracle requires --require-frames, --sensor-audit, and "
                "--sensor-frame-manifest"
            )
        )
    if args.frame_assets_manifest is None:
        raise TransformersCliUsageError(
            detail="teacher_oracle requires --frame-assets-manifest"
        )
    required = (
        (args.sensor_audit, oracle_lineage.sensor_audit_sha256, "sensor audit"),
        (
            args.sensor_frame_manifest,
            oracle_lineage.sensor_manifest_sha256,
            "sensor manifest",
        ),
        (
            args.frame_assets_manifest,
            oracle_lineage.selected_frame_inventory_sha256,
            "selected frame inventory",
        ),
    )
    for path, approved, description in required:
        if _sha256_file(path) != approved:
            raise TransformersCliUsageError(
                detail=f"teacher-oracle {description} does not match approved lineage"
            )
    _verify_checksum_inventory(
        args.frame_assets_manifest, None, "teacher-oracle selected frame assets"
    )
    if _checksum_inventory_assets_sha256(args.frame_assets_manifest) != (
        oracle_lineage.selected_frame_assets_sha256
    ):
        raise TransformersCliUsageError(
            detail="teacher-oracle selected frame assets do not match approved lineage"
        )


def _live_teacher_oracle_lineage(
    args: TransformersCliArgs, env: Mapping[str, str]
) -> tuple[str, str, str, int, str, str, str]:
    prompt_sha256 = _sha256_file(Path(__file__).with_name("qa_prompt.py"))
    decoding_sha256 = hashlib.sha256(
        json.dumps(
            {
                "backend": args.backend,
                "parse_attempt_limit": 2,
                "parser_sha256": _sha256_file(Path(__file__).with_name("qa.py")),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    runtime_sha256 = hashlib.sha256(
        json.dumps(
            {
                "python_executable": str(Path(sys.executable).resolve()),
                "python_version": sys.version,
                "python_prefix": sys.prefix,
                "torch_version": _runtime_package_version("torch"),
                "transformers_version": _runtime_package_version("transformers"),
                "torch_module_sha256": _runtime_module_sha256("torch"),
                "transformers_module_sha256": _runtime_module_sha256("transformers"),
                "transformers_backend_sha256": _sha256_file(
                    Path(__file__).with_name("transformers_backend.py")
                ),
                "qa_transformers_sha256": _sha256_file(Path(__file__)),
                "frame_contract": "sample_video_frames-v1",
                "require_frames": args.require_frames,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    try:
        seed = int(env["WORLDMM_QA_SEED"])
    except (KeyError, ValueError) as exc:
        raise TransformersCliUsageError(
            detail="teacher_oracle requires integer WORLDMM_QA_SEED"
        ) from exc
    return (
        prompt_sha256,
        decoding_sha256,
        runtime_sha256,
        seed,
        _runtime_inventory_sha256("python"),
        _runtime_inventory_sha256("torch"),
        _runtime_inventory_sha256("transformers"),
    )


def _runtime_package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:
        raise TransformersCliUsageError(
            detail=f"teacher_oracle requires installed {package} runtime"
        ) from exc


def _runtime_module_sha256(package: str) -> str:
    try:
        module = importlib.import_module(package)
        module_path = Path(module.__file__ or "").resolve(strict=True)
    except (ImportError, OSError) as exc:
        raise TransformersCliUsageError(
            detail=f"teacher_oracle cannot bind {package} runtime bytes"
        ) from exc
    return _sha256_file(module_path)


def _runtime_inventory_sha256(
    runtime: Literal["python", "torch", "transformers"],
) -> str:
    """Hash every installed runtime file, not only package entry points."""
    paths: tuple[Path, ...]
    if runtime == "python":
        stdlib = Path(sysconfig.get_path("stdlib") or "").resolve(strict=True)
        paths = (Path(sys.executable).resolve(strict=True), *stdlib.rglob("*"))
    else:
        try:
            distribution = importlib.metadata.distribution(runtime)
        except importlib.metadata.PackageNotFoundError as exc:
            raise TransformersCliUsageError(
                detail=f"teacher_oracle requires installed {runtime} runtime"
            ) from exc
        located = (
            Path(str(distribution.locate_file(file)))
            for file in distribution.files or ()
        )
        paths = tuple(path.resolve(strict=True) for path in located if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _questions_by_id(fixture: Path) -> dict[str, QuestionRequest]:
    return {
        question.question_id: question for question in read_fixture_questions(fixture)
    }


def _question_for_pack(
    pack: EvidencePack,
    questions: Mapping[str, QuestionRequest],
) -> QuestionRequest:
    question = questions.get(pack.question_id)
    if question is None:
        raise TransformersCliUsageError(detail=f"unknown question: {pack.question_id}")
    return question


def validate_evidence_lineage(  # noqa: PLR0912,PLR0913
    evidence_path: Path,
    expected_lane: EvidenceLane,
    lineage_path: Path | None,
    checkpoint_path: Path | None,
    typed_memory_path: Path | None,
    inference_manifest_path: Path | None,
    *,
    config_path: Path | None = None,
    sensor_path: Path | None = None,
    data_root: Path | None = None,
    memory_manifest_path: Path | None = None,
    inference_sources_path: Path | None = None,
    frame_assets_path: Path | None = None,
    inference_producer_path: Path | None = None,
    sources: Sequence[SourceStreamExample] | None = None,
    sensor_records: Sequence[SensorFrameManifestRecord] | None = None,
    expected_trust_digests: Mapping[str, str] | None = None,
) -> EvidenceLineage | None:
    """Bind student evidence to the exact artifacts that produced it."""
    if lineage_path is None:
        if expected_lane == "heuristic":
            return None
        raise TransformersCliUsageError(
            detail="student evidence requires --evidence-lineage",
        )
    try:
        lineage = EvidenceLineage.model_validate_json(
            lineage_path.read_text(encoding="utf-8"),
        )
    except (OSError, ValidationError) as exc:
        raise TransformersCliUsageError(
            detail=f"invalid evidence lineage: {lineage_path}: {exc}",
        ) from exc
    if lineage.lane != expected_lane:
        raise TransformersCliUsageError(
            detail=(
                "evidence lineage lane mismatch: "
                f"expected {expected_lane}, got {lineage.lane}"
            ),
        )
    actual_sha256 = _sha256_file(evidence_path)
    if lineage.evidence_sha256 != actual_sha256:
        raise TransformersCliUsageError(
            detail="evidence lineage does not match evidence_sha256",
        )
    if expected_lane != "student":
        return lineage
    if checkpoint_path is None:
        raise TransformersCliUsageError(
            detail="student evidence requires --checkpoint",
        )
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if lineage.checkpoint_sha256 != checkpoint_sha256:
        raise TransformersCliUsageError(
            detail="evidence lineage does not match checkpoint_sha256",
        )
    if typed_memory_path is None:
        raise TransformersCliUsageError(
            detail="student evidence requires --typed-memory",
        )
    typed_memory_sha256 = _sha256_file(typed_memory_path)
    if lineage.typed_memory_sha256 != typed_memory_sha256:
        raise TransformersCliUsageError(
            detail="evidence lineage does not match typed_memory_sha256",
        )
    if inference_manifest_path is None:
        raise TransformersCliUsageError(
            detail="student evidence requires --inference-manifest",
        )
    inference_manifest_sha256 = _sha256_file(inference_manifest_path)
    if lineage.inference_manifest_sha256 != inference_manifest_sha256:
        raise TransformersCliUsageError(
            detail="evidence lineage does not match inference_manifest_sha256",
        )
    if config_path is not None and lineage.config_sha256 != _sha256_file(config_path):
        raise TransformersCliUsageError(
            detail="evidence lineage does not match config_sha256",
        )
    if sensor_path is not None and lineage.sensor_sha256 != _sha256_file(sensor_path):
        raise TransformersCliUsageError(
            detail="evidence lineage does not match sensor_sha256",
        )
    if data_root is not None and lineage.data_sha256 != _fixture_data_sha256(data_root):
        raise TransformersCliUsageError(
            detail="evidence lineage does not match data_sha256",
        )
    if memory_manifest_path is not None:
        _validate_memory_artifact_lineage(lineage, memory_manifest_path)
    _validate_production_inference_manifest(
        inference_manifest_path,
        checkpoint_sha256,
        typed_memory_sha256,
        lineage.sensor_sha256,
        typed_memory_path,
        inference_sources_path=inference_sources_path,
        frame_assets_path=frame_assets_path,
        inference_producer_path=inference_producer_path,
        sources=sources,
        sensor_records=sensor_records,
    )
    if expected_trust_digests is not None:
        allowed = {
            "model_contract_sha256",
            "student_architecture_sha256",
            "model_load_consensus_payload_sha256",
            "model_load_consensus_file_sha256",
        }
        if unknown := tuple(sorted(set(expected_trust_digests) - allowed)):
            raise TransformersCliUsageError(
                detail=f"unknown student trust digest: {unknown[0]}",
            )
        for field, expected in expected_trust_digests.items():
            if getattr(lineage, field) != expected:
                raise TransformersCliUsageError(
                    detail=f"evidence lineage does not match trusted {field}",
                )
    required_context = {
        "--lineage-config": config_path,
        "--sensor-frame-manifest": sensor_path,
        "--fixture": data_root,
        "--memory-manifest": memory_manifest_path,
        "--inference-sources": inference_sources_path,
        "--frame-assets-manifest": frame_assets_path,
        "--inference-producer": inference_producer_path,
        "source records": sources,
        "sensor records": sensor_records,
    }
    if missing := tuple(
        name for name, value in required_context.items() if value is None
    ):
        raise TransformersCliUsageError(
            detail=f"student evidence validation requires {', '.join(missing)}",
        )
    return lineage


def _validate_memory_artifact_lineage(
    lineage: EvidenceLineage,
    memory_manifest_path: Path,
) -> None:
    for field, actual in memory_artifact_hashes(memory_manifest_path).items():
        if getattr(lineage, field) != actual:
            raise TransformersCliUsageError(
                detail=f"evidence lineage does not match {field}",
            )


def synchronize_typed_memory_validation(
    validate: Callable[[], EvidenceLineage | None],
    inference_manifest_path: Path | None,
    distributed: DistributedEnv,
    *,
    distributed_api: _DistributedObjectBroadcaster | None = None,
) -> TypedMemoryValidationSeal:
    """Run full validation on rank 0 and broadcast its strict result seal.

    Direct callers and environments without an initialized process group retain
    the existing full, single-process validation behavior.
    """
    broadcaster = _initialized_distributed_broadcaster(distributed_api)
    if broadcaster is None:
        return _run_typed_memory_validation(validate, inference_manifest_path)

    rank = broadcaster.get_rank()
    world_size = broadcaster.get_world_size()
    try:
        broadcaster.barrier()
    except Exception as exc:
        raise TransformersCliUsageError(
            detail=f"distributed typed-memory validation barrier failed: {exc}",
        ) from exc
    payload: list[object] = [None]
    if rank == 0:
        try:
            seal = _run_typed_memory_validation(validate, inference_manifest_path)
            outcome: BaseModel = _DistributedValidationSuccess(seal=seal)
        except Exception as exc:  # noqa: BLE001 - peers need the same failure.
            detail = (
                exc.detail
                if isinstance(exc, TransformersCliUsageError)
                else f"{type(exc).__name__}: {exc}"
            )
            outcome = _DistributedValidationFailure(
                detail=(detail or type(exc).__name__)[:4096],
            )
        payload[0] = _canonical_model_json(outcome)
    try:
        broadcaster.broadcast_object_list(payload, src=0)
    except Exception as exc:
        raise TransformersCliUsageError(
            detail=f"distributed typed-memory validation broadcast failed: {exc}",
        ) from exc
    if (rank, world_size) != (distributed.rank, distributed.world_size):
        raise TransformersCliUsageError(
            detail=(
                "distributed validation rank/world mismatch: "
                f"process_group={rank}/{world_size} "
                f"environment={distributed.rank}/{distributed.world_size}"
            ),
        )
    return _validated_distributed_outcome(payload[0])


def synchronize_typed_memory_postread(
    expected: TypedMemoryValidationSeal,
    memory_manifest_path: Path,
    inference_manifest_path: Path | None,
    distributed: DistributedEnv,
    *,
    distributed_api: _DistributedObjectBroadcaster | None = None,
) -> None:
    """Recheck sealed artifact digests after rank-local streaming reads."""

    def validate_postread() -> EvidenceLineage:
        _validate_memory_artifact_lineage(expected.lineage, memory_manifest_path)
        return expected.lineage

    actual = synchronize_typed_memory_validation(
        validate_postread,
        inference_manifest_path,
        distributed,
        distributed_api=distributed_api,
    )
    if actual != expected:
        raise TransformersCliUsageError(
            detail="typed-memory validation seal changed during rank-local read",
        )


def synchronize_rank_memory_read(
    error: Exception | None,
    distributed: DistributedEnv,
    *,
    distributed_api: _DistributedObjectBroadcaster | None = None,
) -> None:
    """Make every rank fail promptly when any rank-local projection fails."""
    broadcaster = _initialized_distributed_broadcaster(distributed_api)
    if broadcaster is None:
        if error is not None:
            raise error
        return
    rank = broadcaster.get_rank()
    world_size = broadcaster.get_world_size()
    if error is None:
        local: BaseModel = _DistributedRankReadSuccess(rank=rank)
    else:
        detail = (
            error.detail
            if isinstance(error, TransformersCliUsageError)
            else f"{type(error).__name__}: {error}"
        )
        local = _DistributedRankReadFailure(
            rank=rank,
            detail=(detail or type(error).__name__)[:4096],
        )
    outcomes: list[object] = [None] * world_size
    try:
        broadcaster.all_gather_object(outcomes, _canonical_model_json(local))
    except Exception as exc:
        raise TransformersCliUsageError(
            detail=f"distributed rank-memory validation gather failed: {exc}",
        ) from exc
    if (rank, world_size) != (distributed.rank, distributed.world_size):
        raise TransformersCliUsageError(
            detail=(
                "distributed validation rank/world mismatch: "
                f"process_group={rank}/{world_size} "
                f"environment={distributed.rank}/{distributed.world_size}"
            ),
        )
    failures = tuple(
        (expected_rank, detail)
        for expected_rank, payload in enumerate(outcomes)
        if (detail := _validated_rank_read_outcome(payload, expected_rank)) is not None
    )
    if failures:
        failed_rank, detail = failures[0]
        raise TransformersCliUsageError(
            detail=f"rank {failed_rank} memory projection failed: {detail}",
        )


def _validated_rank_read_outcome(payload: object, expected_rank: int) -> str | None:
    if (
        not isinstance(payload, str)
        or len(payload.encode()) > MEMORY_MANIFEST_MAX_BYTES
    ):
        raise TransformersCliUsageError(
            detail="invalid distributed rank-memory validation payload",
        )
    try:
        loaded = cast("object", json.loads(payload))
    except json.JSONDecodeError as exc:
        raise TransformersCliUsageError(
            detail="invalid distributed rank-memory validation payload",
        ) from exc
    if not isinstance(loaded, dict):
        raise TransformersCliUsageError(
            detail="invalid distributed rank-memory validation payload",
        )
    outcome = cast("dict[str, object]", loaded)
    try:
        if outcome.get("status") == "ok":
            success = _DistributedRankReadSuccess.model_validate(outcome)
            if success.rank != expected_rank:
                raise TransformersCliUsageError(
                    detail="distributed rank-memory validation rank mismatch",
                )
            return None
        if outcome.get("status") == "error":
            failure = _DistributedRankReadFailure.model_validate(outcome)
            if failure.rank != expected_rank:
                raise TransformersCliUsageError(
                    detail="distributed rank-memory validation rank mismatch",
                )
            return failure.detail
    except ValidationError as exc:
        raise TransformersCliUsageError(
            detail="invalid distributed rank-memory validation payload",
        ) from exc
    raise TransformersCliUsageError(
        detail="invalid distributed rank-memory validation payload",
    )


def _run_typed_memory_validation(
    validate: Callable[[], EvidenceLineage | None],
    inference_manifest_path: Path | None,
) -> TypedMemoryValidationSeal:
    lineage = validate()
    if lineage is None or lineage.lane != "student":
        raise TransformersCliUsageError(
            detail="student evidence requires valid evidence lineage",
        )
    if inference_manifest_path is None:
        raise TransformersCliUsageError(
            detail="student evidence requires --inference-manifest",
        )
    return _typed_memory_validation_seal(lineage, inference_manifest_path)


def _typed_memory_validation_seal(
    lineage: EvidenceLineage,
    inference_manifest_path: Path,
) -> TypedMemoryValidationSeal:
    try:
        raw_manifest = inference_manifest_path.read_bytes()
    except OSError as exc:
        raise TransformersCliUsageError(
            detail=f"invalid inference manifest: {inference_manifest_path}: {exc}",
        ) from exc
    inference_manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    if lineage.inference_manifest_sha256 != inference_manifest_sha256:
        raise TransformersCliUsageError(
            detail="inference manifest changed during typed-memory validation",
        )
    try:
        loaded = cast("object", json.loads(raw_manifest))
    except json.JSONDecodeError as exc:
        raise TransformersCliUsageError(
            detail=f"invalid typed-memory validation seal inputs: {exc}",
        ) from exc
    if not isinstance(loaded, dict):
        raise TransformersCliUsageError(
            detail="invalid typed-memory validation seal inputs: not an object",
        )
    typed_memory_sha256 = lineage.typed_memory_sha256
    if typed_memory_sha256 is None:
        raise TransformersCliUsageError(
            detail="invalid typed-memory validation seal inputs: missing digest",
        )
    manifest = cast("dict[str, object]", loaded)
    try:
        seal = TypedMemoryValidationSeal(
            lineage=lineage,
            typed_memory_sha256=typed_memory_sha256,
            inference_manifest_sha256=inference_manifest_sha256,
            record_count=cast("int", manifest["record_count"]),
            actual_bytes=cast("int", manifest["actual_bytes"]),
            window_count=cast("int", manifest["window_count"]),
            max_window_bytes=cast("int", manifest["max_window_bytes"]),
            window_seconds=float(cast("float", manifest["window_seconds"])),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise TransformersCliUsageError(
            detail=f"invalid typed-memory validation seal inputs: {exc}",
        ) from exc
    if seal.typed_memory_sha256 != seal.lineage.typed_memory_sha256:
        raise TransformersCliUsageError(
            detail="typed-memory validation seal digest mismatch",
        )
    return seal


def _initialized_distributed_broadcaster(
    explicit: _DistributedObjectBroadcaster | None,
) -> _DistributedObjectBroadcaster | None:
    if explicit is None:
        try:
            explicit = cast(
                "_DistributedObjectBroadcaster",
                cast("object", importlib.import_module("torch.distributed")),
            )
        except ImportError:
            return None
    try:
        return (
            explicit if explicit.is_available() and explicit.is_initialized() else None
        )
    except (AttributeError, RuntimeError):
        return None


def initialize_qa_validation_process_group(
    env: Mapping[str, str],
) -> _DistributedProcessGroup | None:
    """Initialize a temporary CPU group only for the torchrun QA entry point."""
    distributed = distributed_env(env)
    if distributed.world_size == 1:
        return None
    try:
        process_group = cast(
            "_DistributedProcessGroup",
            cast("object", importlib.import_module("torch.distributed")),
        )
    except ImportError:
        return None
    if not process_group.is_available() or process_group.is_initialized():
        return None
    raw_timeout = env.get("WORLDMM_QA_SHARD_TIMEOUT_SECONDS", "3600")
    try:
        timeout_seconds = int(raw_timeout)
    except ValueError as exc:
        raise TransformersCliUsageError(
            detail="WORLDMM_QA_SHARD_TIMEOUT_SECONDS must be an integer",
        ) from exc
    if not 1 <= timeout_seconds <= QA_DISTRIBUTED_TIMEOUT_MAX_SECONDS:
        raise TransformersCliUsageError(
            detail=("WORLDMM_QA_SHARD_TIMEOUT_SECONDS must be in [1, 86400]"),
        )
    try:
        process_group.init_process_group(
            "gloo",
            init_method="env://",
            timeout=timedelta(seconds=timeout_seconds),
        )
    except Exception as exc:
        raise TransformersCliUsageError(
            detail=f"cannot initialize distributed QA validation: {exc}",
        ) from exc
    return process_group


def _validated_distributed_outcome(payload: object) -> TypedMemoryValidationSeal:
    if (
        not isinstance(payload, str)
        or len(payload.encode()) > MEMORY_MANIFEST_MAX_BYTES
    ):
        raise TransformersCliUsageError(
            detail="invalid distributed typed-memory validation payload",
        )
    try:
        loaded = cast("object", json.loads(payload))
    except json.JSONDecodeError as exc:
        raise TransformersCliUsageError(
            detail="invalid distributed typed-memory validation payload",
        ) from exc
    if not isinstance(loaded, dict):
        raise TransformersCliUsageError(
            detail="invalid distributed typed-memory validation payload",
        )
    outcome = cast("dict[str, object]", loaded)
    status = outcome.get("status")
    try:
        if status == "error":
            failure = _DistributedValidationFailure.model_validate(outcome)
            raise TransformersCliUsageError(detail=failure.detail)
        if status == "ok":
            return _DistributedValidationSuccess.model_validate(outcome).seal
    except ValidationError as exc:
        raise TransformersCliUsageError(
            detail="invalid distributed typed-memory validation payload",
        ) from exc
    raise TransformersCliUsageError(
        detail="invalid distributed typed-memory validation payload",
    )


def _canonical_model_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_production_inference_manifest(  # noqa: PLR0912,PLR0913
    path: Path,
    checkpoint_sha256: str,
    typed_memory_sha256: str,
    sensor_sha256: str | None,
    typed_memory_path: Path,
    *,
    inference_sources_path: Path | None = None,
    frame_assets_path: Path | None = None,
    inference_producer_path: Path | None = None,
    sources: Sequence[SourceStreamExample] | None = None,
    sensor_records: Sequence[SensorFrameManifestRecord] | None = None,
) -> None:
    try:
        loaded = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as exc:
        raise TransformersCliUsageError(
            detail=f"invalid inference manifest: {path}: {exc}",
        ) from exc
    if not isinstance(loaded, dict):
        raise TransformersCliUsageError(
            detail=f"invalid inference manifest object: {path}",
        )
    manifest = cast("dict[str, object]", loaded)
    schema_version = manifest.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise TransformersCliUsageError(
            detail=f"inference manifest schema_version must be integer 1: {path}",
        )
    if (
        manifest.get("production_ready") is not True
        or manifest.get("result_class") != "student"
        or manifest.get("producer") != "spatial-student"
    ):
        raise TransformersCliUsageError(
            detail=f"inference manifest is not production student output: {path}",
        )
    if manifest.get("checkpoint_sha256") != checkpoint_sha256:
        raise TransformersCliUsageError(
            detail=f"inference manifest checkpoint_sha256 mismatch: {path}",
        )
    if manifest.get("records_sha256") != typed_memory_sha256:
        raise TransformersCliUsageError(
            detail=f"inference manifest records_sha256 mismatch: {path}",
        )
    if manifest.get("sensor_sha256") != sensor_sha256:
        raise TransformersCliUsageError(
            detail=f"inference manifest sensor_sha256 mismatch: {path}",
        )
    for field in ("sources_sha256", "frame_assets_sha256", "producer_sha256"):
        digest = manifest.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != SHA256_HEX_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise TransformersCliUsageError(
                detail=f"inference manifest {field} must be SHA-256: {path}",
            )
    window_seconds = manifest.get("window_seconds")
    if (
        not isinstance(window_seconds, (int, float))
        or isinstance(window_seconds, bool)
        or float(window_seconds) != DEFAULT_TYPED_MEMORY_WINDOW_SECONDS
    ):
        raise TransformersCliUsageError(
            detail=(
                "inference manifest window_seconds must be "
                f"{DEFAULT_TYPED_MEMORY_WINDOW_SECONDS}: {path}"
            ),
        )
    for field in ("record_count", "byte_budget_per_window", "window_count"):
        value = manifest.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise TransformersCliUsageError(
                detail=f"inference manifest {field} must be a positive integer: {path}",
            )
    for field in ("max_window_bytes", "actual_bytes"):
        value = manifest.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise TransformersCliUsageError(
                detail=(
                    f"inference manifest {field} must be a non-negative integer: {path}"
                ),
            )
    byte_budget_per_window = cast("int", manifest["byte_budget_per_window"])
    max_window_bytes = cast("int", manifest["max_window_bytes"])
    try:
        artifact = validate_typed_memory_artifact(
            typed_memory_path,
            byte_budget_per_window=byte_budget_per_window,
            window_seconds=float(window_seconds),
            sources=sources,
            sensor_records=sensor_records,
        )
    except TypedMemoryWriterError as exc:
        raise TransformersCliUsageError(
            detail=f"invalid typed memory artifact: {typed_memory_path}: {exc}",
        ) from exc
    if max_window_bytes > byte_budget_per_window:
        raise TransformersCliUsageError(
            detail=f"inference manifest max_window_bytes exceeds budget: {path}",
        )
    if manifest["actual_bytes"] != artifact.actual_bytes:
        raise TransformersCliUsageError(
            detail=f"inference manifest actual_bytes mismatch: {path}",
        )
    if manifest["record_count"] != artifact.record_count:
        raise TransformersCliUsageError(
            detail=f"inference manifest record_count mismatch: {path}",
        )
    if manifest["window_count"] != artifact.window_count:
        raise TransformersCliUsageError(
            detail=f"inference manifest window_count mismatch: {path}",
        )
    if max_window_bytes != artifact.max_window_bytes:
        raise TransformersCliUsageError(
            detail=f"inference manifest max_window_bytes mismatch: {path}",
        )
    expected_origin_artifacts = {
        "sources_sha256": inference_sources_path,
        "frame_assets_sha256": frame_assets_path,
        "producer_sha256": inference_producer_path,
    }
    for field, artifact_path in expected_origin_artifacts.items():
        if artifact_path is not None and manifest[field] != _sha256_file(artifact_path):
            raise TransformersCliUsageError(
                detail=f"inference manifest {field} mismatch: {path}",
            )


def validate_evidence_trace_lane(
    packs: Sequence[EvidencePack],
    expected_lane: EvidenceLane,
) -> None:
    if expected_lane not in {"student", "teacher_oracle"}:
        return
    for pack in packs:
        if pack.retrieval_trace.policy_route == "legacy-missing-trace":
            raise TransformersCliUsageError(
                detail=f"student evidence requires retrieval_trace: {pack.question_id}",
            )


def validate_external_evidence_packs(
    packs: Sequence[EvidencePack],
    questions: Mapping[str, QuestionRequest],
) -> None:
    """Fail closed before externally materialized evidence enters QA."""
    pack_question_ids = tuple(pack.question_id for pack in packs)
    seen: set[str] = set()
    duplicates: set[str] = set()
    for question_id in pack_question_ids:
        if question_id in seen:
            duplicates.add(question_id)
        seen.add(question_id)
    if duplicates:
        raise TransformersCliUsageError(
            detail=f"duplicate evidence pack: {sorted(duplicates)[0]}",
        )
    actual = set(pack_question_ids)
    expected = set(questions)
    if unknown := sorted(actual - expected):
        raise TransformersCliUsageError(
            detail=f"unknown evidence pack question: {unknown[0]}",
        )
    if missing := sorted(expected - actual):
        raise TransformersCliUsageError(
            detail=f"missing evidence pack: {missing[0]}",
        )

    for pack in packs:
        question = questions[pack.question_id]
        if detail := evidence_pack_validation_error(question, pack):
            raise TransformersCliUsageError(
                detail=f"invalid evidence pack {pack.question_id}: {detail}",
            )


def validate_spatial_evidence_against_typed_memory(
    packs: Sequence[EvidencePack],
    typed_records: Sequence[RetrievalMemoryRecord],
    *,
    max_frame_refs: int = RETRIEVAL_FRAME_REF_CAP,
) -> None:
    """Require exact typed projections after deterministic global ref capping."""
    _validate_evidence_projections(
        packs,
        typed_records,
        validated_stores=frozenset({"spatial"}),
        max_frame_refs=max_frame_refs,
        source_label="typed spatial",
    )


def validate_student_evidence_against_memory(
    packs: Sequence[EvidencePack],
    memory_records: Sequence[RetrievalMemoryRecord],
    *,
    max_frame_refs: int = RETRIEVAL_FRAME_REF_CAP,
) -> None:
    """Require every student evidence item to be an exact store projection."""
    _validate_evidence_projections(
        packs,
        memory_records,
        validated_stores=frozenset(
            {"episodic", "semantic", "visual", "spatial"},
        ),
        max_frame_refs=max_frame_refs,
        source_label="canonical",
    )


def _validate_evidence_projections(
    packs: Sequence[EvidencePack],
    records: Sequence[RetrievalMemoryRecord],
    *,
    validated_stores: frozenset[str],
    max_frame_refs: int,
    source_label: str,
) -> None:
    if not 0 <= max_frame_refs <= RETRIEVAL_FRAME_REF_CAP:
        raise TransformersCliUsageError(
            detail=(
                "student evidence max_frame_refs must be between 0 and "
                f"{RETRIEVAL_FRAME_REF_CAP}"
            ),
        )
    scoped_records = tuple(
        record for record in records if record.source_store in validated_stores
    )
    index = {
        (record.source_store, record.memory_id): record for record in scoped_records
    }
    if len(index) != len(scoped_records):
        duplicate_detail = (
            "typed spatial retrieval records contain duplicate memory IDs"
            if source_label == "typed spatial"
            else "canonical retrieval records contain duplicate store/memory IDs"
        )
        raise TransformersCliUsageError(
            detail=duplicate_detail,
        )
    for pack in packs:
        actual_frame_ref_count = sum(len(item.frame_refs) for item in pack.evidence)
        if actual_frame_ref_count != pack.retrieval_trace.frame_ref_count:
            raise TransformersCliUsageError(
                detail=(
                    "student evidence frame-ref trace differs from evidence: "
                    f"{pack.question_id}"
                ),
            )
        if actual_frame_ref_count > max_frame_refs:
            raise TransformersCliUsageError(
                detail=(f"student evidence exceeds max_frame_refs: {pack.question_id}"),
            )
        remaining_frame_refs = max_frame_refs
        for item in pack.evidence:
            if item.source_store not in validated_stores:
                raise TransformersCliUsageError(
                    detail=(
                        "student evidence references store outside validated "
                        f"projection set: {pack.question_id}:{item.source_store}"
                    ),
                )
            expected = index.get((item.source_store, item.memory_id))
            if expected is None:
                unknown_detail = (
                    "student spatial evidence references unknown typed memory"
                    if source_label == "typed spatial"
                    else "student evidence references unknown canonical memory"
                )
                raise TransformersCliUsageError(
                    detail=(f"{unknown_detail}: {pack.question_id}:{item.memory_id}"),
                )
            actual_projection = (
                item.video_id,
                item.snippet,
                item.frame_refs,
                item.start_time,
                item.end_time,
                item.geometry,
            )
            expected_projection = (
                expected.video_id,
                expected.snippet,
                cap_frame_refs(expected.frame_refs, remaining_frame_refs),
                expected.start_time,
                expected.end_time,
                expected.geometry,
            )
            if actual_projection != expected_projection:
                mismatch_detail = (
                    "student spatial evidence differs from typed memory"
                    if source_label == "typed spatial"
                    else "student evidence differs from canonical memory"
                )
                raise TransformersCliUsageError(
                    detail=(f"{mismatch_detail}: {pack.question_id}:{item.memory_id}"),
                )
            remaining_frame_refs -= len(item.frame_refs)


def _parse_backend(value: str) -> TransformersBackendName:
    match value:
        case "gemma4" | "real" | "mock":
            return value
        case other:
            raise TransformersCliUsageError(detail=f"unknown backend: {other}")


def _parse_evidence_lane(value: str) -> EvidenceLane:
    match value:
        case "heuristic" | "student" | "teacher_oracle":
            return value
        case other:
            raise TransformersCliUsageError(
                detail=f"unknown evidence lane: {other}",
            )


def _parse_oracle_variant(value: str) -> OracleVariant:
    match value:
        case "E0" | "T0" | "T1":
            return value
        case _:
            raise TransformersCliUsageError(
                detail="--expected-variant must be E0, T0, or T1"
            )


def _validate_rank_progress(
    packs: Sequence[EvidencePack],
    predictions: Sequence[PredictionRecord],
    *,
    completed: bool,
) -> None:
    expected = {pack.question_id for pack in packs}
    actual = [prediction.question_id for prediction in predictions]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for question_id in actual:
        if question_id in seen:
            duplicates.add(question_id)
        seen.add(question_id)
    if duplicates:
        raise QAShardError(
            detail=f"duplicate QA checkpoint question: {sorted(duplicates)[0]}",
        )
    unexpected = set(actual) - expected
    if unexpected:
        raise QAShardError(
            detail=f"unexpected QA checkpoint question: {sorted(unexpected)[0]}",
        )
    if completed and set(actual) != expected:
        missing = sorted(expected - set(actual))
        raise QAShardError(
            detail=f"incomplete final QA shard; missing: {', '.join(missing)}",
        )


def qa_resume_manifest_path(out: Path) -> Path:
    return out.with_name(f"{out.name}.manifest.json")


def _bind_resume_manifest(args: TransformersCliArgs, written: Path) -> None:
    path = qa_resume_manifest_path(args.out)
    expected = qa_resume_manifest(args)
    if path.exists():
        _validate_resume_manifest(path, expected)
        return
    if _prediction_progress_exists(args.out, written):
        raise QAShardError(
            detail=f"QA resume manifest missing for existing predictions: {path}",
        )
    _install_manifest_atomic(path, expected)
    _validate_resume_manifest(path, expected)


def _qa_trust_digests(args: TransformersCliArgs) -> dict[str, str]:
    if args.evidence_lane != "student" or args.evidence_lineage is None:
        return {
            "model_contract_sha256": "",
            "student_architecture_sha256": "",
            "model_load_consensus_payload_sha256": "",
            "model_load_consensus_file_sha256": "",
        }
    try:
        lineage = EvidenceLineage.model_validate_json(
            args.evidence_lineage.read_text(encoding="utf-8"),
        )
    except (OSError, ValidationError) as exc:
        raise QAShardError(
            detail=f"invalid student evidence lineage for QA resume: {exc}",
        ) from exc
    return {
        "model_contract_sha256": lineage.model_contract_sha256 or "",
        "student_architecture_sha256": lineage.student_architecture_sha256 or "",
        "model_load_consensus_payload_sha256": (
            lineage.model_load_consensus_payload_sha256 or ""
        ),
        "model_load_consensus_file_sha256": (
            lineage.model_load_consensus_file_sha256 or ""
        ),
    }


def qa_resume_manifest(args: TransformersCliArgs) -> dict[str, str]:

    return {
        "manifest_version": QA_RESUME_MANIFEST_VERSION,
        "prompt_schema_version": QA_PROMPT_SCHEMA_VERSION,
        "evidence_sha256": _sha256_file(args.evidence),
        "evidence_lane": args.evidence_lane,
        "evidence_lineage_sha256": (
            _sha256_file(args.evidence_lineage)
            if args.evidence_lineage is not None
            else ""
        ),
        **_qa_trust_digests(args),
        "expected_variant": args.expected_variant or "",
        "checkpoint_sha256": (
            _sha256_file(args.checkpoint) if args.checkpoint is not None else ""
        ),
        "typed_memory_sha256": (
            _sha256_file(args.typed_memory) if args.typed_memory is not None else ""
        ),
        "inference_manifest_sha256": (
            _sha256_file(args.inference_manifest)
            if args.inference_manifest is not None
            else ""
        ),
        "inference_sources_sha256": (
            _sha256_file(args.inference_sources)
            if args.inference_sources is not None
            else ""
        ),
        "inference_producer_sha256": (
            _sha256_file(args.inference_producer)
            if args.inference_producer is not None
            else ""
        ),
        "require_frames": "true" if args.require_frames else "false",
        "questions_sha256": _sha256_file(args.fixture / "questions.jsonl"),
        "sources_sha256": _sha256_file(args.fixture / "sources.jsonl"),
        "backend": args.backend,
        "model": args.model,
        "model_fingerprint_sha256": (
            _sha256_file(args.model_fingerprint)
            if args.model_fingerprint is not None
            else ""
        ),
        "frame_assets_sha256": (
            _sha256_file(args.frame_assets_manifest)
            if args.frame_assets_manifest is not None
            else ""
        ),
        "lineage_config_sha256": (
            _sha256_file(args.lineage_config) if args.lineage_config is not None else ""
        ),
        "sensor_frame_manifest_sha256": (
            _sha256_file(args.sensor_frame_manifest)
            if args.sensor_frame_manifest is not None
            else ""
        ),
        "sensor_audit_sha256": (
            _sha256_file(args.sensor_audit) if args.sensor_audit is not None else ""
        ),
        "memory_manifest_sha256": (
            _sha256_file(args.memory_manifest)
            if args.memory_manifest is not None
            else ""
        ),
    }


def _fixture_data_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("sources.jsonl", "questions.jsonl"):
        digest.update(name.encode() + b"\0")
        with (root / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def memory_artifact_hashes(manifest_path: Path) -> dict[str, str]:  # noqa: PLR0912
    try:
        if manifest_path.is_symlink():
            raise TransformersCliUsageError(
                detail=f"memory manifest must not be a symlink: {manifest_path}",
            )
        with manifest_path.open("rb") as stream:
            manifest_bytes = stream.read(MEMORY_MANIFEST_MAX_BYTES + 1)
        if len(manifest_bytes) > MEMORY_MANIFEST_MAX_BYTES:
            raise TransformersCliUsageError(
                detail=f"memory manifest exceeds 64 KiB: {manifest_path}",
            )
        loaded = cast(
            "object",
            json.loads(manifest_bytes),
        )
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise TransformersCliUsageError(
            detail=f"invalid memory manifest: {manifest_path}: {exc}",
        ) from exc
    if not isinstance(loaded, dict):
        raise TransformersCliUsageError(
            detail=f"invalid memory manifest object: {manifest_path}",
        )
    payload = cast("dict[str, object]", loaded)
    expected_keys = {
        "schema_version",
        "episodic_memory",
        "semantic_memory",
        "visual_memory",
        "spatial_memory",
    }
    if set(payload) != expected_keys or type(payload.get("schema_version")) is not int:
        raise TransformersCliUsageError(
            detail=f"memory manifest keys/schema mismatch: {manifest_path}",
        )
    if payload["schema_version"] != 1:
        raise TransformersCliUsageError(
            detail=f"unsupported memory manifest schema: {payload['schema_version']}",
        )
    spatial = payload.get("spatial_memory")
    if not isinstance(spatial, dict):
        raise TransformersCliUsageError(
            detail=f"memory manifest missing spatial_memory: {manifest_path}",
        )
    spatial_payload = cast("dict[str, object]", spatial)
    if set(spatial_payload) != {"path"}:
        raise TransformersCliUsageError(
            detail=f"memory manifest has invalid spatial_memory: {manifest_path}",
        )
    raw_paths = {
        "episodic_memory_sha256": payload.get("episodic_memory"),
        "semantic_memory_sha256": payload.get("semantic_memory"),
        "visual_memory_sha256": payload.get("visual_memory"),
        "typed_memory_sha256": spatial_payload.get("path"),
    }
    if any(not isinstance(path, str) for path in raw_paths.values()):
        raise TransformersCliUsageError(
            detail=f"memory manifest contains invalid artifact paths: {manifest_path}",
        )
    try:
        resolved_manifest = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise TransformersCliUsageError(
            detail=f"memory manifest unavailable: {manifest_path}: {exc}",
        ) from exc
    if resolved_manifest != manifest_path.absolute():
        raise TransformersCliUsageError(
            detail=f"memory manifest path contains a symlink: {manifest_path}",
        )
    root = resolved_manifest.parent
    expected_paths = {
        "episodic_memory_sha256": root / "episodic.jsonl",
        "semantic_memory_sha256": root / "worldmm_sv/semantic.jsonl",
        "visual_memory_sha256": root / "worldmm_sv/visual.jsonl",
        "typed_memory_sha256": root / "typed_memory.jsonl",
    }
    resolved_paths: dict[str, Path] = {}
    for field, raw_path in raw_paths.items():
        candidate = Path(cast("str", raw_path))
        if not candidate.is_absolute():
            candidate = root / candidate
        expected = expected_paths[field]
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise TransformersCliUsageError(
                detail=f"memory artifact unavailable: {candidate}: {exc}",
            ) from exc
        if candidate.is_symlink() or resolved != expected or not resolved.is_file():
            raise TransformersCliUsageError(
                detail=(
                    f"memory manifest path mismatch for {field}: "
                    f"expected {expected}, got {candidate}"
                ),
            )
        resolved_paths[field] = resolved
    try:
        return {
            "memory_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            **{
                field: _sha256_file(resolved_path)
                for field, resolved_path in resolved_paths.items()
            },
        }
    except OSError as exc:
        raise TransformersCliUsageError(
            detail=f"memory artifact unreadable: {exc}",
        ) from exc


def _prediction_progress_exists(out: Path, written: Path) -> bool:
    direct = {out, partial_output_path(out), written, partial_output_path(written)}
    if any(path.exists() for path in direct):
        return True
    if not out.parent.is_dir():
        return False
    rank_prefix = f"{out.stem}.rank"
    return any(
        child.name.startswith(rank_prefix)
        and (
            child.name.endswith(out.suffix)
            or child.name.endswith(f"{out.suffix}.partial")
        )
        for child in out.parent.iterdir()
    )


def _install_manifest_atomic(path: Path, manifest: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            _ = output.write(
                json.dumps(
                    dict(manifest),
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
            )
            output.flush()
            os.fsync(output.fileno())
        # Identical ranks may concurrently bind the shared output.
        with suppress(FileExistsError):
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_resume_manifest(
    path: Path,
    expected: Mapping[str, str],
) -> None:
    try:
        loaded_object = cast(
            "object",
            json.loads(path.read_text(encoding="utf-8")),
        )
    except (json.JSONDecodeError, OSError) as exc:
        raise QAShardError(detail=f"invalid QA resume manifest: {path}: {exc}") from exc
    if not isinstance(loaded_object, dict):
        raise QAShardError(detail=f"invalid QA resume manifest object: {path}")
    loaded_values = cast("dict[object, object]", loaded_object)
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in loaded_values.items()
    ):
        raise QAShardError(detail=f"invalid QA resume manifest fields: {path}")
    loaded = cast("dict[str, str]", loaded_values)
    mismatched = sorted(
        key for key, value in expected.items() if loaded.get(key) != value
    )
    unexpected = sorted(set(loaded) - set(expected))
    if mismatched or unexpected:
        fields = ", ".join((*mismatched, *unexpected))
        raise QAShardError(
            detail=f"QA resume manifest mismatch ({fields}): {path}",
        )


class _ChecksumInventoryError(ValueError):
    """Raised when a checksum inventory is malformed or does not verify."""


def _checksum_inventory_error(detail: str) -> NoReturn:
    raise _ChecksumInventoryError(detail)


def _read_checksum_inventory(inventory: Path) -> tuple[tuple[str, str], ...]:
    lines = inventory.read_text(encoding="utf-8").splitlines()
    if not lines:
        _checksum_inventory_error("empty inventory")
    return tuple(_parse_checksum_inventory_record(line) for line in lines)


def _parse_checksum_inventory_record(line: str) -> tuple[str, str]:
    digest, separator, filename = line.partition("  ")
    if (
        not separator
        or len(digest) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in digest)
        or not filename
    ):
        _checksum_inventory_error("invalid checksum record")
    return digest, filename


def _validate_checksum_inventory(
    checksums: Sequence[tuple[str, str]],
    root: Path | None,
) -> None:
    resolved_root = root.resolve(strict=True) if root is not None else None
    for digest, filename in checksums:
        asset = Path(filename).resolve(strict=True)
        if resolved_root is not None:
            try:
                _ = asset.relative_to(resolved_root)
            except ValueError:
                _checksum_inventory_error("inventory entry escapes model reference")
        if _sha256_file(asset) != digest:
            detail = f"checksum mismatch: {asset}"
            _checksum_inventory_error(detail)


def _verify_checksum_inventory(
    inventory: Path, root: Path | None, description: str
) -> None:
    try:
        checksums = _read_checksum_inventory(inventory)
        _validate_checksum_inventory(checksums, root)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TransformersCliUsageError(
            detail=f"invalid {description} inventory: {inventory}: {exc}"
        ) from exc


def _checksum_inventory_assets_sha256(inventory: Path) -> str:
    """Hash verified selected-frame assets separately from their inventory file."""
    try:
        checksums = _read_checksum_inventory(inventory)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TransformersCliUsageError(
            detail=(
                f"invalid teacher-oracle selected frame inventory: {inventory}: {exc}"
            )
        ) from exc
    digest = hashlib.sha256()
    for expected, filename in checksums:
        asset = Path(filename).resolve(strict=True)
        actual = _sha256_file(asset)
        if actual != expected:
            raise TransformersCliUsageError(
                detail=f"teacher-oracle selected frame asset changed: {asset}"
            )
        digest.update(str(asset).encode("utf-8"))
        digest.update(b"\0")
        digest.update(actual.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_output_path(out: Path, question_id: str) -> Path:
    digest = hashlib.sha256(question_id.encode()).hexdigest()[:16]
    return out.parent / f"{out.stem}_raw_model_outputs" / f"q_{digest}.json"


def _write_raw_outputs(path: Path, raw_outputs: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _ = temporary.write_text(
            json.dumps(
                {"raw_outputs": tuple(raw_outputs)},
                ensure_ascii=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        _ = temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
