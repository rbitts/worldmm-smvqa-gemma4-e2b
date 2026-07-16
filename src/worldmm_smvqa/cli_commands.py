from __future__ import annotations

# allow: SIZE_OK - command router module predates this change; split by command group
# when the CLI surface grows again.
import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

from worldmm_smvqa.chunking import (
    build_chunks,
    read_source_streams,
    write_fixture_chunks,
)
from worldmm_smvqa.cli_args import (
    CliPublicationError,
    CliUsageError,
    CommandResult,
    ParsedArgs,
)
from worldmm_smvqa.config import (
    REMOTE_ENV_FLAG,
    RemoteOnlyError,
    load_config,
    require_remote,
)
from worldmm_smvqa.fixture_cli import prepare_fixture_stdout, validate_schema_stdout
from worldmm_smvqa.fixtures import read_fixture_questions
from worldmm_smvqa.memory_alignment_config import load_memory_alignment_config
from worldmm_smvqa.memory_sources import write_fixture_source_memories
from worldmm_smvqa.metrics import (
    InvalidPredictionError,
    evaluate_prediction_files,
    metrics_stdout,
    write_metrics,
)
from worldmm_smvqa.preflight import (
    write_preflight_report,
    write_teacher_oracle_preflight_report,
)
from worldmm_smvqa.qa import (
    Gemma4QABackend,
    MockQABackend,
    QARetrievalOptions,
    run_qa,
    write_predictions_jsonl,
)
from worldmm_smvqa.remote_plan import plan_stdout, write_remote_plan
from worldmm_smvqa.report import write_report
from worldmm_smvqa.retrieval import (
    RetrievalOptions,
    build_fixture_retrieval_stores,
    injected_future_memory,
    parse_retrieval_stores,
    read_retrieval_memory_artifacts,
    retrieve_evidence,
)
from worldmm_smvqa.sensor_audit import write_sensor_audit_report
from worldmm_smvqa.sensor_frames import write_sensor_frame_manifest
from worldmm_smvqa.smoke import run_smoke_pipeline, smoke_stdout
from worldmm_smvqa.worldmm.episodic import write_fixture_episodic_memory
from worldmm_smvqa.worldmm.llm_memory_io import (
    LLMMemoryBindings,
    distributed_env,
    memory_bindings,
    partition_by_video,
    qwen_bindings,
    write_distributed_jsonl,
    write_llm_episodic_memory,
    write_llm_semantic_memory,
    write_llm_visual_memory,
)
from worldmm_smvqa.worldmm.semantic import write_fixture_semantic_memory
from worldmm_smvqa.worldmm.spatial_compression import (
    SpatialCompressionManifest,
    build_compressed_spatial_memory,
)
from worldmm_smvqa.worldmm.spatial_diagnostics import (
    write_spatial_retrieval_diagnostics,
)
from worldmm_smvqa.worldmm.spatial_types import SpatialTokenRecord
from worldmm_smvqa.worldmm.visual import write_fixture_visual_memory

if TYPE_CHECKING:
    from worldmm_smvqa.retrieval_types import EvidencePack, RetrievalMemoryRecord
    from worldmm_smvqa.schema import QuestionRequest, StreamChunk

SUPPORTED_BUILD_STORES: Final = frozenset(
    {"episodic", "semantic", "visual", "spatial"},
)
SUPPORTED_MEMORY_BACKENDS: Final = frozenset({"mock", "qwen", "memory"})
DEFAULT_RETRIEVAL_STORES: Final = "episodic,semantic,visual,spatial"
DEFAULT_MAX_FRAME_REFS: Final = 32


def handle_prepare_fixture(args: ParsedArgs) -> CommandResult:
    return CommandResult(stdout=prepare_fixture_stdout(args.config, args.out))


def handle_validate_schema(args: ParsedArgs) -> CommandResult:
    return CommandResult(stdout=validate_schema_stdout(args.config, args.input))


def handle_preflight(args: ParsedArgs) -> CommandResult:
    input_dir = args.input or args.fixture
    if input_dir is None:
        raise CliUsageError(detail="preflight requires --input/--fixture")
    if args.out is None:
        raise CliUsageError(detail="preflight requires --out")
    configured_frame_root = os.environ.get("SMVQA_FRAME_ROOT")
    report = write_preflight_report(
        input_dir,
        args.out,
        frame_root=(
            None if configured_frame_root is None else Path(configured_frame_root)
        ),
    )
    return CommandResult(
        stdout=(
            f"wrote {args.out}\nstatus={report.status} "
            f"errors={len(report.errors)} warnings={len(report.warnings)}\n"
        ),
        exit_code=1 if report.errors else 0,
    )


def handle_validate_teacher_oracle_inputs(args: ParsedArgs) -> CommandResult:
    if args.sensor_audit is None:
        raise CliUsageError(
            detail="validate-teacher-oracle-inputs requires --sensor-audit"
        )
    if args.experiment_config is None:
        raise CliUsageError(
            detail="validate-teacher-oracle-inputs requires --experiment-config"
        )
    if args.out is None:
        raise CliUsageError(detail="validate-teacher-oracle-inputs requires --out")
    try:
        report = write_teacher_oracle_preflight_report(
            args.sensor_audit,
            args.experiment_config,
            args.out,
        )
    except OSError as exc:
        raise CliPublicationError(
            command="validate-teacher-oracle-inputs",
            path=args.out,
            reason=exc.strerror or type(exc).__name__,
        ) from exc
    return CommandResult(
        stdout=(
            f"wrote {args.out}\nstatus={report.status} "
            f"blockers={len(report.blockers)}\n"
        ),
        exit_code=1 if report.blockers else 0,
    )


def handle_audit_sensors(args: ParsedArgs) -> CommandResult:
    if args.sensor_manifest is None:
        raise CliUsageError(detail="audit-sensors requires --sensor-manifest")
    if args.observations is None:
        raise CliUsageError(detail="audit-sensors requires --observations")
    if args.frame_root is None:
        raise CliUsageError(detail="audit-sensors requires --frame-root")
    if args.out is None:
        raise CliUsageError(detail="audit-sensors requires --out")
    try:
        report = write_sensor_audit_report(
            args.sensor_manifest,
            args.observations,
            args.frame_root,
            args.out,
        )
    except OSError as exc:
        raise CliPublicationError(
            command="audit-sensors",
            path=args.out,
            reason=exc.strerror or type(exc).__name__,
        ) from exc
    return CommandResult(
        stdout=(
            f"wrote {args.out}\n"
            f"state={report.operational_state} "
            f"decision={report.provider_gate_decision} "
            f"errors={len(report.issues)}\n"
        ),
        exit_code=1 if report.issues else 0,
    )


def handle_build_memory(args: ParsedArgs) -> CommandResult:
    if args.backend == "memory":
        return _handle_memory_build(args)
    _config = load_config(args.config)
    if args.backend not in SUPPORTED_MEMORY_BACKENDS:
        raise CliUsageError(detail=f"unsupported build-memory backend: {args.backend}")
    if args.stage == "sensor-frames":
        return _handle_sensor_frame_manifest_build(args)
    if args.stage == "chunk":
        return _handle_chunk_build(args)
    if args.stage == "source-memories":
        return _handle_source_memory_build(args)
    if args.stage is not None:
        raise CliUsageError(detail=f"unsupported build-memory stage: {args.stage}")
    if args.store is None:
        raise CliUsageError(detail="build-memory requires --stage or --store/--stores")
    stores = _requested_stores(args.store)
    if not stores:
        raise CliUsageError(
            detail="build-memory requires at least one non-empty --store/--stores value"
        )
    invalid_stores = stores - SUPPORTED_BUILD_STORES
    if invalid_stores:
        ordered = ",".join(sorted(invalid_stores))
        raise CliUsageError(detail=f"unsupported build-memory store: {ordered}")
    if stores == frozenset({"episodic"}):
        return _handle_episodic_build(args)
    if stores <= {"semantic", "visual", "spatial"}:
        return _handle_semantic_visual_build(args)
    raise CliUsageError(detail="build-memory stores must not mix episodic with others")


def _handle_memory_build(args: ParsedArgs) -> CommandResult:
    if args.stage is not None:
        raise CliUsageError(
            detail="build-memory --backend memory does not support stages"
        )
    if args.store is None:
        raise CliUsageError(
            detail="build-memory --backend memory requires --store/--stores"
        )
    stores = _requested_stores(args.store)
    if not stores:
        raise CliUsageError(
            detail="build-memory requires at least one non-empty --store/--stores value"
        )
    if stores == frozenset({"episodic"}):
        if args.out is None:
            raise CliUsageError(detail="build-memory --store episodic requires --out")
    elif stores and stores <= {"semantic", "visual"}:
        if args.out is None:
            raise CliUsageError(detail="build-memory semantic/visual requires --out")
        if "semantic" in stores and args.input is None:
            raise CliUsageError(
                detail=(
                    "build-memory --backend memory semantic requires "
                    "--input <episodic.jsonl>"
                )
            )
    else:
        invalid = ",".join(sorted(stores - {"episodic", "semantic", "visual"}))
        if invalid:
            raise CliUsageError(
                detail=f"unsupported build-memory --backend memory store: {invalid}"
            )
        detail = (
            "build-memory --backend memory stores must not mix episodic with others"
        )
        raise CliUsageError(detail=detail)

    _ = load_memory_alignment_config(args.config, Path.cwd())
    if os.environ.get(REMOTE_ENV_FLAG) != "1":
        raise RemoteOnlyError(action="build-memory memory")
    bindings = memory_bindings(os.environ)
    if stores == frozenset({"episodic"}):
        return _handle_episodic_build(args, bindings)
    return _handle_semantic_visual_build(args, bindings)


def handle_retrieve(args: ParsedArgs) -> CommandResult:
    _config = load_config(args.config)
    if args.out is None:
        raise CliUsageError(detail="retrieve requires --out")
    if args.question is None:
        raise CliUsageError(detail="retrieve requires --question")
    fixture_dir = args.fixture or Path("tests/fixtures/tiny_smvqa")
    stores = parse_retrieval_stores(args.store or DEFAULT_RETRIEVAL_STORES)
    chunks = _retrieval_chunks(fixture_dir, args.retrieval_protocol)
    _validate_max_frame_refs(args.max_frame_refs)
    question = _read_fixture_question(fixture_dir, args.question)
    memories = (
        read_retrieval_memory_artifacts(args.input)
        if args.input is not None
        else build_fixture_retrieval_stores(fixture_dir)
    )
    if args.inject_future_memory:
        memories = (*memories, injected_future_memory(question))
    pack = retrieve_evidence(
        question,
        memories,
        enabled_stores=stores,
        options=RetrievalOptions(
            chunks=chunks,
            max_frame_refs=args.max_frame_refs,
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _ = args.out.write_text(pack.model_dump_json(indent=2) + "\n", encoding="utf-8")
    store_text = ",".join(pack.requested_stores)
    selected_text = ",".join(pack.selected_stores)
    return CommandResult(
        stdout=(
            f"wrote {args.out}\n"
            f"stores={store_text} selected_stores={selected_text} "
            f"protocol={args.retrieval_protocol} "
            f"evidence={len(pack.evidence)} "
            f"causal_filtered_count={pack.causal_filtered_count}\n"
        ),
    )


def handle_retrieve_batch(args: ParsedArgs) -> CommandResult:
    _config = load_config(args.config)
    if args.out is None:
        raise CliUsageError(detail="retrieve-batch requires --out")
    fixture_dir = args.fixture or Path("tests/fixtures/tiny_smvqa")
    stores = parse_retrieval_stores(args.store or DEFAULT_RETRIEVAL_STORES)
    chunks = _retrieval_chunks(fixture_dir, args.retrieval_protocol)
    _validate_max_frame_refs(args.max_frame_refs)
    questions = read_fixture_questions(fixture_dir)
    memories = (
        read_retrieval_memory_artifacts(args.input)
        if args.input is not None
        else build_fixture_retrieval_stores(fixture_dir)
    )
    memories_by_video: dict[str, list[RetrievalMemoryRecord]] = {}
    for memory in memories:
        memories_by_video.setdefault(memory.video_id, []).append(memory)
    chunks_by_video: dict[str, list[StreamChunk]] = {}
    for chunk in chunks or ():
        chunks_by_video.setdefault(chunk.video_id, []).append(chunk)
    packs: list[EvidencePack] = []
    for question in questions:
        video_ids = tuple(
            dict.fromkeys(question.video_ids or (question.video_id,)),
        )
        question_memories = tuple(
            memory
            for video_id in video_ids
            for memory in memories_by_video.get(video_id, ())
        )
        question_chunks = (
            None
            if chunks is None
            else tuple(
                chunk
                for video_id in video_ids
                for chunk in chunks_by_video.get(video_id, ())
            )
        )
        packs.append(
            retrieve_evidence(
                question,
                (
                    (*question_memories, injected_future_memory(question))
                    if args.inject_future_memory
                    else question_memories
                ),
                enabled_stores=stores,
                options=RetrievalOptions(
                    chunks=question_chunks,
                    max_frame_refs=args.max_frame_refs,
                ),
            ),
        )
    _write_evidence_packs_atomic(args.out, tuple(packs))
    return CommandResult(
        stdout=(
            f"wrote {args.out}\n"
            f"evidence_packs={len(packs)} memories={len(memories)} "
            f"protocol={args.retrieval_protocol}\n"
        ),
    )


def handle_qa(args: ParsedArgs) -> CommandResult:
    config = load_config(args.config)
    if args.local and args.backend != "mock":
        require_remote(config, "qa real model", os.environ)
    if args.real_model or args.backend in {"gemma4", "real"}:
        require_remote(config, "qa real model", os.environ)
        backend = Gemma4QABackend(
            model_path=config.values.get("remote", {}).get("model_path"),
        )
    elif args.backend == "mock":
        backend = MockQABackend()
    else:
        raise CliUsageError(detail=f"unknown qa backend: {args.backend}")
    if args.out is None:
        raise CliUsageError(detail="qa requires --out")
    fixture_dir = args.fixture or Path("tests/fixtures/tiny_smvqa")
    chunks = _retrieval_chunks(fixture_dir, args.retrieval_protocol)
    _validate_max_frame_refs(args.max_frame_refs)
    predictions = run_qa(
        fixture_dir,
        backend,
        retrieval_options=QARetrievalOptions(
            enabled_stores=parse_retrieval_stores(
                args.store or DEFAULT_RETRIEVAL_STORES,
            ),
            chunks=chunks,
            use_chunk_protocol=args.retrieval_protocol == "worldmm-smvqa",
            max_frame_refs=args.max_frame_refs,
            frame_root=fixture_dir / "frames",
            spatial_env=os.environ,
        ),
    )
    write_predictions_jsonl(predictions, args.out)
    return CommandResult(
        stdout=f"wrote {args.out}\npredictions={len(predictions)}\n",
    )


def handle_evaluate(args: ParsedArgs) -> CommandResult:
    _config = load_config(args.config)
    if args.pred is None:
        raise CliUsageError(detail="evaluate requires --pred")
    if args.labels is None:
        raise CliUsageError(detail="evaluate requires --labels")
    if args.out is None:
        raise CliUsageError(detail="evaluate requires --out")
    metrics = evaluate_prediction_files(args.pred, args.labels)
    if metrics.diagnostics.causal_violation_count:
        raise InvalidPredictionError(
            question_id="predictions",
            detail=(
                "causal evidence violations: "
                f"{metrics.diagnostics.causal_violation_count}"
            ),
        )
    write_metrics(metrics, args.out)
    return CommandResult(stdout=metrics_stdout(metrics, args.out))


def handle_diagnose_spatial(args: ParsedArgs) -> CommandResult:
    _config = load_config(args.config)
    if args.input is None:
        raise CliUsageError(detail="diagnose-spatial requires --input")
    if args.labels is None:
        raise CliUsageError(detail="diagnose-spatial requires --labels")
    if args.out is None:
        raise CliUsageError(detail="diagnose-spatial requires --out")
    write_spatial_retrieval_diagnostics(args.input, args.labels, args.out)
    return CommandResult(stdout=f"wrote {args.out}\n")


class _MockDagRunner(Protocol):
    def __call__(
        self,
        fixture: Path,
        *,
        student_architecture: Path = ...,
    ) -> object: ...


def handle_mock_dag(args: ParsedArgs) -> CommandResult:
    _config = load_config(args.config)
    if args.fixture is None:
        raise CliUsageError(detail="mock-dag requires --fixture")
    run_local_mock_dag = cast(
        "_MockDagRunner",
        importlib.import_module("worldmm_smvqa.mock_dag").run_local_mock_dag,
    )

    if args.student_architecture is None:
        _ = run_local_mock_dag(args.fixture)
    else:
        _ = run_local_mock_dag(
            args.fixture,
            student_architecture=args.student_architecture,
        )
    return CommandResult(stdout="")


def handle_report(args: ParsedArgs) -> CommandResult:
    _config = load_config(args.config)
    if args.run_manifest is None:
        raise CliUsageError(detail="report requires --run-manifest")
    if args.out is None:
        raise CliUsageError(detail="report requires --out")
    write_report(args.run_manifest, args.out)
    return CommandResult(stdout=f"wrote {args.out}\n")


def handle_smoke(args: ParsedArgs) -> CommandResult:
    _config = load_config(args.config)
    if args.out is None:
        raise CliUsageError(detail="smoke requires --out")
    fixture_dir = args.fixture or Path("tests/fixtures/tiny_smvqa")
    if args.ablation_protocol is not None:
        _validate_ablation_protocol(args.ablation_protocol)
    result = run_smoke_pipeline(
        fixture_dir,
        args.out,
        os.environ,
        ablation_stores=args.ablation_stores,
        ablation_protocol=args.ablation_protocol,
    )
    return CommandResult(stdout=smoke_stdout(args.out, result))


def handle_launch_remote(args: ParsedArgs) -> CommandResult:
    if args.out is None:
        raise CliUsageError(detail="launch-remote requires --out")
    if args.profile != "teacher-oracle":
        raise CliUsageError(detail="launch-remote requires --profile teacher-oracle")
    if args.experiment_config is None:
        raise CliUsageError(detail="launch-remote requires --experiment-config")
    if args.sensor_audit is not None:
        raise CliUsageError(
            detail=(
                "launch-remote does not accept --sensor-audit; "
                "use the reviewed experiment config"
            )
        )
    if args.submit and args.dry_run:
        raise CliUsageError(
            detail="launch-remote accepts only one of --dry-run/--submit",
        )
    if not args.submit and not args.dry_run:
        raise CliUsageError(detail="launch-remote requires --dry-run or --submit")
    config = load_config(args.config)
    remote = config.values.get("remote")
    configured_experiment = (
        remote.get("experiment_config") if isinstance(remote, dict) else None
    )
    if not isinstance(configured_experiment, str) or args.experiment_config.resolve(
        strict=False
    ) != Path(configured_experiment).resolve(strict=False):
        raise CliUsageError(
            detail="--experiment-config must match remote.experiment_config"
        )
    try:
        result = write_remote_plan(
            config,
            args.out,
            os.environ,
            submit=args.submit,
            plan_profile="teacher-oracle",
        )
    except OSError as exc:
        raise CliPublicationError(
            command="launch-remote",
            path=args.out,
            reason=exc.strerror or type(exc).__name__,
        ) from exc
    return CommandResult(stdout=plan_stdout(result))


def _handle_sensor_frame_manifest_build(args: ParsedArgs) -> CommandResult:
    if args.out is None:
        raise CliUsageError(
            detail="build-memory --stage sensor-frames requires --out",
        )
    fixture_dir = args.fixture or Path("tests/fixtures/tiny_smvqa")
    summary = write_sensor_frame_manifest(
        read_source_streams(fixture_dir, use_sensor_manifest=False),
        args.out,
    )
    return CommandResult(
        stdout=(
            f"wrote {summary.path}\n"
            f"sensor_rate_hz={summary.sensor_rate_hz:g} "
            f"sources={summary.source_count} "
            f"source_frames={summary.source_frame_count} "
            f"selected_frames={summary.selected_frame_count}\n"
        ),
    )


def _handle_chunk_build(args: ParsedArgs) -> CommandResult:
    if args.out is None:
        raise CliUsageError(detail="build-memory --stage chunk requires --out")
    fixture_dir = args.fixture or Path("tests/fixtures/tiny_smvqa")
    summary = write_fixture_chunks(fixture_dir, args.out)
    granularities = ",".join(summary.granularities)
    return CommandResult(
        stdout=(
            f"wrote {summary.path}\n"
            f"chunks={summary.chunks} granularities={granularities}\n"
        ),
    )


def _handle_source_memory_build(args: ParsedArgs) -> CommandResult:
    if args.out is None:
        raise CliUsageError(
            detail="build-memory --stage source-memories requires --out",
        )
    fixture_dir = args.fixture or Path("tests/fixtures/tiny_smvqa")
    summary = write_fixture_source_memories(fixture_dir, args.out)
    stores = ",".join(summary.stores)
    return CommandResult(
        stdout=(
            f"wrote {summary.path}\nsource_memories={summary.records} stores={stores}\n"
        ),
    )


def _handle_episodic_build(
    args: ParsedArgs, bindings: LLMMemoryBindings | None = None
) -> CommandResult:
    if args.out is None:
        raise CliUsageError(detail="build-memory --store episodic requires --out")
    fixture_dir = args.fixture or Path("tests/fixtures/tiny_smvqa")
    if args.backend in {"qwen", "memory"}:
        active_bindings = bindings or _llm_bindings(args)
        summary = write_llm_episodic_memory(
            fixture_dir, args.out, active_bindings.generate
        )
    else:
        summary = write_fixture_episodic_memory(fixture_dir, args.out)
    return CommandResult(
        stdout=(
            f"wrote {summary.path}\n"
            f"nodes={summary.nodes} edges={summary.edges} "
            f"contains={summary.contains_edges}\n"
        ),
    )


def _handle_semantic_visual_build(
    args: ParsedArgs, bindings: LLMMemoryBindings | None = None
) -> CommandResult:
    if args.out is None:
        raise CliUsageError(detail="build-memory semantic/visual requires --out")
    fixture_dir = args.fixture or Path("tests/fixtures/tiny_smvqa")
    stores = _requested_stores(args.store or "")
    active_bindings = (
        bindings or _llm_bindings(args)
        if args.backend in {"qwen", "memory"} and stores & {"semantic", "visual"}
        else None
    )
    args.out.mkdir(parents=True, exist_ok=True)
    semantic_records = 0
    visual_records = 0
    spatial_records = 0
    if "semantic" in stores:
        if active_bindings is not None:
            if args.input is None:
                detail = (
                    "build-memory --backend qwen semantic requires "
                    "--input <episodic.jsonl>"
                )
                raise CliUsageError(
                    detail=detail,
                )
            semantic_records = write_llm_semantic_memory(
                args.input,
                args.out / "semantic.jsonl",
                active_bindings.generate,
            ).records
        else:
            semantic_records = write_fixture_semantic_memory(
                fixture_dir,
                args.out / "semantic.jsonl",
            ).records
    if "visual" in stores:
        if active_bindings is not None:
            visual_records = write_llm_visual_memory(
                fixture_dir,
                args.out / "visual.jsonl",
                frame_root=_frame_root(fixture_dir),
                caption=active_bindings.caption,
            ).records
        else:
            visual_records = write_fixture_visual_memory(
                fixture_dir,
                args.out / "visual.jsonl",
            ).records
    if "spatial" in stores:
        spatial_records = _write_fixture_spatial_memory(
            fixture_dir,
            args.out / "spatial.jsonl",
        )
    return CommandResult(
        stdout=(
            f"wrote {args.out}\n"
            f"semantic_records={semantic_records} visual_records={visual_records} "
            f"spatial_records={spatial_records}\n"
        ),
    )


def _requested_stores(value: str) -> frozenset[str]:
    return frozenset(part.strip() for part in value.split(",") if part.strip())


def _write_evidence_packs_atomic(
    path: Path,
    packs: tuple[EvidencePack, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _ = temporary.write_text(
            "".join(f"{pack.model_dump_json()}\n" for pack in packs),
            encoding="utf-8",
        )
        _ = temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _llm_bindings(args: ParsedArgs) -> LLMMemoryBindings:
    require_remote(load_config(args.config), "build-memory llm", os.environ)
    return qwen_bindings(os.environ)


def _frame_root(fixture_dir: Path) -> Path:
    return Path(os.environ.get("SMVQA_FRAME_ROOT", str(fixture_dir / "frames")))


def _write_fixture_spatial_memory(fixture_dir: Path, output: Path) -> int:
    sources = partition_by_video(read_source_streams(fixture_dir))
    chunks = build_chunks(sources)
    clip_chunks = tuple(chunk for chunk in chunks if chunk.granularity == "clip_30s")
    result = build_compressed_spatial_memory(
        sources,
        clip_chunks,
        measure_legacy=True,
    )
    written = write_distributed_jsonl(result.records, output)
    rank, world_size = distributed_env()
    manifest = SpatialCompressionManifest(
        experiment=result.experiment,
        rank=rank,
        world_size=world_size,
        source_count=len(sources),
        record_count=len(result.records),
        token_count=sum(
            isinstance(record, SpatialTokenRecord) for record in result.records
        ),
        trajectory_count=sum(
            not isinstance(record, SpatialTokenRecord) for record in result.records
        ),
        candidate_count=result.candidate_count,
        raw_record_count=result.raw_record_count,
        raw_bytes=result.raw_bytes,
        compressed_bytes=result.compressed_bytes,
    )
    stats_path = output.with_name(f"{output.stem}.stats{output.suffix}")
    _ = write_distributed_jsonl((manifest,), stats_path)
    return sum(1 for line in written.read_text(encoding="utf-8").splitlines() if line)


def _retrieval_chunks(
    fixture_dir: Path,
    retrieval_protocol: str,
) -> tuple[StreamChunk, ...] | None:
    if retrieval_protocol == "worldmm-smvqa":
        return build_chunks(read_source_streams(fixture_dir))
    if retrieval_protocol == "legacy-round-robin":
        return None
    raise CliUsageError(detail=f"unsupported retrieval protocol: {retrieval_protocol}")


def _validate_max_frame_refs(max_frame_refs: int) -> None:
    if max_frame_refs < 0 or max_frame_refs > DEFAULT_MAX_FRAME_REFS:
        raise CliUsageError(
            detail=f"--max-frame-refs must be between 0 and {DEFAULT_MAX_FRAME_REFS}",
        )


def _validate_ablation_protocol(ablation_protocol: str) -> None:
    if ablation_protocol != "legacy-round-robin":
        raise CliUsageError(
            detail=f"unsupported ablation protocol: {ablation_protocol}",
        )


def _read_fixture_question(fixture_dir: Path, question_id: str) -> QuestionRequest:
    canonical_id = _canonical_question_id(question_id)
    for question in read_fixture_questions(fixture_dir):
        if question.question_id == canonical_id:
            return question
    raise CliUsageError(detail=f"unknown question: {question_id}")


def _canonical_question_id(question_id: str) -> str:
    if question_id == "q_object_001":
        return "q_fake_001"
    return question_id
