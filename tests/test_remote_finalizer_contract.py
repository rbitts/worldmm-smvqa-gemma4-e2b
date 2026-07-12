from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from worldmm_smvqa.chunking import read_source_streams
from worldmm_smvqa.fixtures import read_fixture_questions
from worldmm_smvqa.qa_transformers import (
    TransformersCliArgs,
    memory_artifact_hashes,
    qa_resume_manifest,
)
from worldmm_smvqa.remote_script import dag_stage_script_text
from worldmm_smvqa.report import RemoteRunManifest
from worldmm_smvqa.retrieval_types import EvidenceLineage
from worldmm_smvqa.schema import PredictionRecord
from worldmm_smvqa.sensor_frames import write_sensor_frame_manifest
from worldmm_smvqa.worldmm.typed_memory import (
    ObjectGeometry,
    ObjectMemoryRecord,
    SpatialUncertainty,
    ValidityInterval,
    canonical_jsonl_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/tiny_smvqa"
DIGEST = "a" * 64


def test_generated_finalizer_revalidates_and_writes_probe_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output/probe-finalizer"
    paths, env = _prepare_finalizer_run(output)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _finalizer_python(),
            _sha(paths["finalization_inputs"]),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = RemoteRunManifest.model_validate_json(
        (output / "summary/remote_manifest.json").read_text(encoding="utf-8"),
    )
    assert manifest.result_class == "contract_probe"
    assert manifest.execution_profile == "probe"
    assert manifest.evidence_lineage_sha256 == _sha(paths["evidence_lineage"])
    assert manifest.run_identity_sha256 == _sha(
        output / "summary/run_identity.json",
    )
    assert manifest.finalization_inputs_sha256 == _sha(
        paths["finalization_inputs"],
    )
    assert {metric.experiment for metric in manifest.metrics} == {"PROBE"}
    assert (output / "summary/final_report.md").is_file()


def test_generated_finalizer_rejects_changed_sealed_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output/tampered-finalizer"
    paths, env = _prepare_finalizer_run(output)
    _ = paths["summary"].write_text("tampered after seal\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _finalizer_python(),
            _sha(paths["finalization_inputs"]),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "finalization input seal no longer matches current artifacts" in result.stderr
    )
    assert not (output / "summary/run_identity.json").exists()
    assert not (output / "summary/remote_manifest.json").exists()


def _prepare_finalizer_run(
    output: Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    paths = _prepare_finalizer_artifacts(output)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "WORLDMM_OUTPUT_ROOT": str(output),
        "WORLDMM_APPROVAL_FILE": str(paths["approval"]),
        "RUN_FIXTURE": str(FIXTURE),
        "GEMMA_MODEL_PATH": "/approved/gemma",
        "WORLDMM_EXECUTION_REPO": str(output / "code_snapshot"),
        "WORLDMM_SENSOR_FRAME_MANIFEST": str(paths["sensor"]),
        "WORLDMM_SPATIAL_INFER_EXE": str(paths["inference_producer"]),
        "WORLDMM_EXECUTION_PROFILE": "probe",
        "WORLDMM_RUN_ID": "probe-finalizer",
        "SLURM_JOB_ID": "12345",
    }
    qa_args = TransformersCliArgs(
        model=env["GEMMA_MODEL_PATH"],
        fixture=FIXTURE,
        evidence=paths["evidence"],
        evidence_lane="student",
        evidence_lineage=paths["evidence_lineage"],
        checkpoint=paths["checkpoint"],
        typed_memory=paths["typed_memory"],
        inference_manifest=paths["inference_manifest"],
        inference_sources=paths["inference_sources"],
        inference_producer=paths["inference_producer"],
        require_frames=True,
        out=paths["predictions"],
        backend="gemma4",
        model_fingerprint=paths["gemma_model_fingerprint"],
        frame_assets_manifest=paths["frame_assets"],
        lineage_config=paths["config"],
        sensor_frame_manifest=paths["sensor"],
        memory_manifest=paths["memory_manifest"],
    )
    _ = paths["qa_resume_manifest"].write_text(
        json.dumps(qa_resume_manifest(qa_args), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_qa_completion(paths)
    _write_finalization_inputs(paths)
    return paths, env


def _prepare_finalizer_artifacts(output: Path) -> dict[str, Path]:
    paths = {
        "approval": output / "approval.json",
        "deployed_code": output / "diagnostics/deployed_code.sha256",
        "deployed_code_files": output / "diagnostics/deployed_code.files.sha256",
        "env_contract": output / "diagnostics/env_contract.json",
        "frame_assets": output / "diagnostics/frame_assets.sha256",
        "preflight_inputs": output / "diagnostics/preflight_inputs.sha256",
        "gemma_model_fingerprint": output / "diagnostics/gemma_model.sha256",
        "checkpoint": output / "checkpoints/spatial_student.pt",
        "inference_manifest": output / "memory/typed_memory.inference.json",
        "inference_sources": output / "inference_inputs/sources.jsonl",
        "inference_producer": output / "bin/spatial-infer",
        "typed_memory": output / "memory/typed_memory.jsonl",
        "memory_manifest": output / "memory/memory_manifest.json",
        "episodic_memory": output / "memory/episodic.jsonl",
        "semantic_memory": output / "memory/worldmm_sv/semantic.jsonl",
        "visual_memory": output / "memory/worldmm_sv/visual.jsonl",
        "memory_inputs": output / "retrieval/memory_inputs.sha256",
        "memory_hashes": output / "retrieval/memory_inputs.json",
        "evidence": output / "retrieval/evidence_packs.jsonl",
        "evidence_lineage": output / "retrieval/evidence_packs.jsonl.lineage.json",
        "predictions": output / "qa/predictions.jsonl",
        "qa_resume_manifest": output / "qa/predictions.jsonl.manifest.json",
        "qa_completion": output / "qa/completed.json",
        "finalization_inputs": output / "summary/finalization_inputs.sha256",
        "summary": output / "summary/summary.txt",
        "metrics": output / "metrics/metrics.json",
        "sensor": output / "manifests/sensor_frames.jsonl",
        "config": output / "code_snapshot/configs/remote.example.yaml",
        "sources": FIXTURE / "sources.jsonl",
        "questions": FIXTURE / "questions.jsonl",
        "labels": FIXTURE / "labels.jsonl",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    (output / "summary").mkdir(parents=True, exist_ok=True)
    for name in (
        "approval",
        "deployed_code",
        "deployed_code_files",
        "env_contract",
        "frame_assets",
        "preflight_inputs",
        "gemma_model_fingerprint",
        "checkpoint",
        "finalization_inputs",
        "summary",
    ):
        _ = paths[name].write_text(f"{name}\n", encoding="utf-8")
    _ = paths["config"].write_bytes((ROOT / "configs/remote.example.yaml").read_bytes())
    snapshot_root = output / "code_snapshot"
    relative_config = paths["config"].relative_to(snapshot_root).as_posix()
    _ = paths["deployed_code"].write_text(
        f"{_sha(paths['config'])}  ./{relative_config}\n",
        encoding="utf-8",
    )
    filename_digest = hashlib.sha256(f"./{relative_config}\0".encode()).hexdigest()
    _ = paths["deployed_code_files"].write_text(
        f"{filename_digest}  -\n",
        encoding="utf-8",
    )
    sources = read_source_streams(FIXTURE, use_sensor_manifest=False)
    _ = write_sensor_frame_manifest(sources, paths["sensor"])
    _ = paths["inference_sources"].write_bytes(paths["sources"].read_bytes())
    _ = paths["inference_producer"].write_bytes(b"spatial-infer")
    record = ObjectMemoryRecord(
        memory_id="memory-1",
        source_video_id="fake_video_001",
        entity_id="mug",
        instance_id="mug-1",
        local_frame_id="source_world",
        geometry_uncertainty=SpatialUncertainty(
            covariance_xyz=(
                (0.01, 0.0, 0.0),
                (0.0, 0.01, 0.0),
                (0.0, 0.0, 0.01),
            ),
            standard_deviation_m=0.1,
        ),
        validity=ValidityInterval(start_time=8.0, end_time=8.0),
        first_seen_time=8.0,
        last_seen_time=8.0,
        observation_count=1,
        confidence=0.9,
        provenance="observed",
        evidence_refs=("fake_video_001_frame_0008",),
        geometry=ObjectGeometry(centroid=(0.5, 1.5, 1.0), extent=(0.2, 0.2, 0.3)),
        semantic_label="mug",
    )
    _ = paths["typed_memory"].write_bytes(canonical_jsonl_bytes(record))
    for name in ("episodic_memory", "semantic_memory", "visual_memory"):
        _ = paths[name].write_text("{}\n", encoding="utf-8")
    _ = paths["memory_manifest"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodic_memory": str(paths["episodic_memory"]),
                "semantic_memory": str(paths["semantic_memory"]),
                "visual_memory": str(paths["visual_memory"]),
                "spatial_memory": {"path": str(paths["typed_memory"])},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _ = paths["memory_hashes"].write_text(
        json.dumps(
            memory_artifact_hashes(paths["memory_manifest"]),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    memory_inputs = (
        paths["memory_manifest"],
        paths["episodic_memory"],
        paths["semantic_memory"],
        paths["visual_memory"],
        paths["typed_memory"],
        paths["memory_hashes"],
    )
    _ = paths["memory_inputs"].write_text(
        "".join(f"{_sha(path)}  {path}\n" for path in memory_inputs),
        encoding="utf-8",
    )
    _ = paths["inference_manifest"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "production_ready": True,
                "result_class": "student",
                "producer": "spatial-student",
                "sources_sha256": _sha(paths["inference_sources"]),
                "frame_assets_sha256": _sha(paths["frame_assets"]),
                "producer_sha256": _sha(paths["inference_producer"]),
                "checkpoint_sha256": _sha(paths["checkpoint"]),
                "sensor_sha256": _sha(paths["sensor"]),
                "records_sha256": _sha(paths["typed_memory"]),
                "record_count": 1,
                "byte_budget_per_window": 4096,
                "actual_bytes": paths["typed_memory"].stat().st_size,
                "window_count": 1,
                "max_window_bytes": paths["typed_memory"].stat().st_size,
                "window_seconds": 30.0,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    _ = paths["evidence"].write_text("{}\n", encoding="utf-8")
    lineage = EvidenceLineage(
        lane="student",
        producer="spatial-student",
        evidence_sha256=_sha(paths["evidence"]),
        checkpoint_sha256=_sha(paths["checkpoint"]),
        inference_manifest_sha256=_sha(paths["inference_manifest"]),
        config_sha256=_sha(paths["config"]),
        sensor_sha256=_sha(paths["sensor"]),
        data_sha256=_fixture_digest(FIXTURE),
        **memory_artifact_hashes(paths["memory_manifest"]),
    )
    _ = paths["evidence_lineage"].write_text(
        lineage.model_dump_json() + "\n",
        encoding="utf-8",
    )
    predictions = (
        PredictionRecord(
            question_id=question.question_id,
            answerable=True,
            ranked_choices=tuple(
                choice.choice_id for choice in question.answer_choices
            ),
            answer=question.answer_choices[0].choice_id,
            confidence=0.5,
            supporting_memory_ids=(),
            input_frame_refs=(f"{question.video_id}/fake-frame",),
            prompt_sha256=DIGEST,
            prompt_token_count=1,
            raw_model_output_path=None,
        )
        for question in read_fixture_questions(FIXTURE)
    )
    _ = paths["predictions"].write_text(
        "".join(f"{prediction.model_dump_json()}\n" for prediction in predictions),
        encoding="utf-8",
    )
    _ = paths["metrics"].write_text(
        json.dumps({"Ans-F1": 1.0, "QA-Acc": 2.0, "QA-MRR": 3.0}) + "\n",
        encoding="utf-8",
    )
    return paths


def _write_qa_completion(paths: dict[str, Path]) -> None:
    _ = paths["qa_completion"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "predictions_sha256": _sha(paths["predictions"]),
                "qa_resume_manifest_sha256": _sha(paths["qa_resume_manifest"]),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_finalization_inputs(paths: dict[str, Path]) -> None:
    sealed = tuple(
        path
        for name, path in paths.items()
        if name not in {"finalization_inputs", "metrics"}
    )
    _ = paths["finalization_inputs"].write_text(
        "".join(f"{_sha(path)}  {path}\n" for path in sealed),
        encoding="utf-8",
    )


def _finalizer_python() -> str:
    blocks = _embedded_python_blocks(dag_stage_script_text())
    marker = "RemoteRunManifest, render_report"
    return next(block for block in blocks if marker in block)


def _embedded_python_blocks(script: str) -> tuple[str, ...]:
    blocks: list[str] = []
    current: list[str] | None = None
    waiting_for_command_end = False
    for line in script.splitlines():
        if current is None and "<<'PY'" in line:
            current = []
            waiting_for_command_end = line.rstrip().endswith("\\")
        elif current is not None and line == "PY":
            blocks.append("\n".join(current) + "\n")
            current = None
        elif current is not None and waiting_for_command_end:
            waiting_for_command_end = line.rstrip().endswith("\\")
        elif current is not None:
            current.append(line)
    return tuple(blocks)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("sources.jsonl", "questions.jsonl", "labels.jsonl"):
        digest.update(name.encode() + b"\0")
        digest.update((root / name).read_bytes())
    return digest.hexdigest()
