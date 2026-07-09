from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from worldmm_smvqa.chunking import write_fixture_chunks
from worldmm_smvqa.cli_args import CliUsageError, CommandResult, ParsedArgs
from worldmm_smvqa.config import load_config, require_remote
from worldmm_smvqa.fixture_cli import prepare_fixture_stdout, validate_schema_stdout
from worldmm_smvqa.fixtures import read_fixture_questions
from worldmm_smvqa.memory_sources import write_fixture_source_memories
from worldmm_smvqa.metrics import (
    evaluate_prediction_files,
    metrics_stdout,
    write_metrics,
)
from worldmm_smvqa.qa import (
    Gemma4QABackend,
    MockQABackend,
    run_qa,
    write_predictions_jsonl,
)
from worldmm_smvqa.remote_plan import plan_stdout, write_remote_plan
from worldmm_smvqa.report import write_report
from worldmm_smvqa.retrieval import (
    build_fixture_retrieval_stores,
    injected_future_memory,
    parse_retrieval_stores,
    retrieve_evidence,
)
from worldmm_smvqa.smoke import run_smoke_pipeline, smoke_stdout
from worldmm_smvqa.worldmm.episodic import write_fixture_episodic_memory
from worldmm_smvqa.worldmm.semantic import write_fixture_semantic_memory
from worldmm_smvqa.worldmm.visual import write_fixture_visual_memory

if TYPE_CHECKING:
    from worldmm_smvqa.schema import QuestionRequest

SUPPORTED_BUILD_STORES: Final = frozenset({"episodic", "semantic", "visual"})


def handle_prepare_fixture(args: ParsedArgs) -> CommandResult:
    return CommandResult(stdout=prepare_fixture_stdout(args.config, args.out))


def handle_validate_schema(args: ParsedArgs) -> CommandResult:
    return CommandResult(stdout=validate_schema_stdout(args.config, args.input))


def handle_build_memory(args: ParsedArgs) -> CommandResult:
    _config = load_config(args.config)
    if args.stage == "chunk":
        return _handle_chunk_build(args)
    if args.stage == "source-memories":
        return _handle_source_memory_build(args)
    if args.stage is not None:
        raise CliUsageError(detail=f"unsupported build-memory stage: {args.stage}")
    if args.store is None:
        raise CliUsageError(detail="build-memory requires --stage or --store/--stores")
    stores = _requested_stores(args.store)
    invalid_stores = stores - SUPPORTED_BUILD_STORES
    if invalid_stores:
        ordered = ",".join(sorted(invalid_stores))
        raise CliUsageError(detail=f"unsupported build-memory store: {ordered}")
    if stores == frozenset({"episodic"}):
        return _handle_episodic_build(args)
    if stores <= {"semantic", "visual"}:
        return _handle_semantic_visual_build(args)
    raise CliUsageError(detail="build-memory stores must not mix episodic with others")


def handle_retrieve(args: ParsedArgs) -> CommandResult:
    _config = load_config(args.config)
    if args.out is None:
        raise CliUsageError(detail="retrieve requires --out")
    if args.question is None:
        raise CliUsageError(detail="retrieve requires --question")
    fixture_dir = args.fixture or Path("tests/fixtures/tiny_smvqa")
    stores = parse_retrieval_stores(args.store or "episodic,semantic,visual")
    question = _read_fixture_question(fixture_dir, args.question)
    memories = build_fixture_retrieval_stores(fixture_dir)
    if args.inject_future_memory:
        memories = (*memories, injected_future_memory(question))
    pack = retrieve_evidence(question, memories, enabled_stores=stores)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _ = args.out.write_text(pack.model_dump_json(indent=2) + "\n", encoding="utf-8")
    store_text = ",".join(pack.requested_stores)
    selected_text = ",".join(pack.selected_stores)
    return CommandResult(
        stdout=(
            f"wrote {args.out}\n"
            f"stores={store_text} selected_stores={selected_text} "
            f"evidence={len(pack.evidence)} "
            f"causal_filtered_count={pack.causal_filtered_count}\n"
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
    predictions = run_qa(fixture_dir, backend)
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
    write_metrics(metrics, args.out)
    return CommandResult(stdout=metrics_stdout(metrics, args.out))


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
    result = run_smoke_pipeline(fixture_dir, args.out, os.environ)
    return CommandResult(stdout=smoke_stdout(args.out, result))


def handle_launch_remote(args: ParsedArgs) -> CommandResult:
    config = load_config(args.config)
    if args.out is None:
        raise CliUsageError(detail="launch-remote requires --out")
    if args.submit and args.dry_run:
        raise CliUsageError(
            detail="launch-remote accepts only one of --dry-run/--submit",
        )
    if not args.submit and not args.dry_run:
        raise CliUsageError(detail="launch-remote requires --dry-run or --submit")
    result = write_remote_plan(config, args.out, os.environ, submit=args.submit)
    return CommandResult(stdout=plan_stdout(result))


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
            f"wrote {summary.path}\n"
            f"source_memories={summary.records} stores={stores}\n"
        ),
    )


def _handle_episodic_build(args: ParsedArgs) -> CommandResult:
    if args.out is None:
        raise CliUsageError(detail="build-memory --store episodic requires --out")
    fixture_dir = args.fixture or Path("tests/fixtures/tiny_smvqa")
    summary = write_fixture_episodic_memory(fixture_dir, args.out)
    return CommandResult(
        stdout=(
            f"wrote {summary.path}\n"
            f"nodes={summary.nodes} edges={summary.edges} "
            f"contains={summary.contains_edges}\n"
        ),
    )


def _handle_semantic_visual_build(args: ParsedArgs) -> CommandResult:
    if args.out is None:
        raise CliUsageError(detail="build-memory semantic/visual requires --out")
    fixture_dir = args.fixture or Path("tests/fixtures/tiny_smvqa")
    stores = _requested_stores(args.store or "")
    args.out.mkdir(parents=True, exist_ok=True)
    semantic_records = 0
    visual_records = 0
    if "semantic" in stores:
        semantic = write_fixture_semantic_memory(
            fixture_dir,
            args.out / "semantic.jsonl",
        )
        semantic_records = semantic.records
    if "visual" in stores:
        visual = write_fixture_visual_memory(fixture_dir, args.out / "visual.jsonl")
        visual_records = visual.records
    return CommandResult(
        stdout=(
            f"wrote {args.out}\n"
            f"semantic_records={semantic_records} visual_records={visual_records}\n"
        ),
    )


def _requested_stores(value: str) -> frozenset[str]:
    return frozenset(part.strip() for part in value.split(",") if part.strip())


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
