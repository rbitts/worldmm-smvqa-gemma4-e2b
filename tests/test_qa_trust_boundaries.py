from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from worldmm_smvqa import qa_transformers
from worldmm_smvqa.chunking import read_source_streams
from worldmm_smvqa.fixtures import read_fixture_questions
from worldmm_smvqa.qa import QAParseError, parse_qa_output
from worldmm_smvqa.qa_prompt import (
    SPATIAL_EVIDENCE_PLACEHOLDER,
    build_qa_prompt,
)
from worldmm_smvqa.qa_transformers import (
    TransformersCliArgs,
    TransformersCliUsageError,
    causal_wearer_pose,
    memory_artifact_hashes,
    parse_cli_args,
    read_teacher_oracle_pre_evaluation_lineage,
    run_transformers_cli,
    validate_evidence_lineage,
    validate_evidence_trace_lane,
    validate_external_evidence_packs,
    validate_spatial_evidence_against_typed_memory,
    validate_student_evidence_against_memory,
    validate_teacher_oracle_live_contract,
)
from worldmm_smvqa.retrieval import (
    RetrievalOptions,
    read_typed_spatial_retrieval_records,
    retrieve_evidence,
)
from worldmm_smvqa.retrieval_types import (
    RETRIEVAL_FRAME_REF_CAP,
    EvidenceItem,
    EvidenceLineage,
    EvidencePack,
    OracleEvidenceLineage,
    OracleQAInputLineage,
    OracleQAPreEvaluationLineage,
    OracleVariantLineage,
    RetrievalMemoryRecord,
    RetrievalStore,
    SharedQALineage,
)
from worldmm_smvqa.schema import (
    AnswerChoice,
    FrameMetadata,
    PoseSample,
    PredictionRecord,
    QuestionRequest,
    SourceStreamExample,
)
from worldmm_smvqa.sensor_frames import (
    SensorFrameManifestError,
    read_sensor_frame_manifest,
    write_sensor_frame_manifest,
)
from worldmm_smvqa.smoke import run_smoke_pipeline
from worldmm_smvqa.worldmm.geometry_executor import (
    GeometryQuery,
    execute_geometry,
    geometry_proofs_for_question,
)
from worldmm_smvqa.worldmm.typed_memory import (
    ObjectGeometry,
    ObjectMemoryRecord,
    SpatialUncertainty,
    ValidityInterval,
    canonical_jsonl_bytes,
)

FIXTURE = Path("tests/fixtures/tiny_smvqa")


def _typed_object(memory_id: str = "typed-object") -> ObjectMemoryRecord:
    return ObjectMemoryRecord(
        memory_id=memory_id,
        source_video_id="video-1",
        entity_id=f"entity-{memory_id}",
        instance_id=f"instance-{memory_id}",
        local_frame_id="room-1",
        geometry_uncertainty=SpatialUncertainty(
            covariance_xyz=(
                (0.01, 0.0, 0.0),
                (0.0, 0.01, 0.0),
                (0.0, 0.0, 0.01),
            ),
            standard_deviation_m=0.1,
        ),
        validity=ValidityInterval(start_time=1.0, end_time=2.0),
        first_seen_time=1.0,
        last_seen_time=2.0,
        observation_count=1,
        confidence=0.9,
        provenance="observed",
        geometry=ObjectGeometry(centroid=(1, 2, 3), extent=(1, 1, 1)),
        semantic_label="mug",
    )


def _fixture_typed_object() -> ObjectMemoryRecord:
    return ObjectMemoryRecord(
        memory_id="fixture-object",
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
        geometry=ObjectGeometry(centroid=(1.0, 2.0, 3.0), extent=(1.0, 1.0, 1.0)),
        semantic_label="mug",
    )


@pytest.mark.parametrize("empty_trace", [False, True], ids=["missing", "empty"])
def test_require_frames_rejects_text_only_mock(
    tmp_path: Path,
    *,
    empty_trace: bool,
) -> None:
    smoke_dir = tmp_path / "smoke"
    _ = run_smoke_pipeline(FIXTURE, smoke_dir, {})
    evidence = smoke_dir / "evidence_packs.jsonl"
    if empty_trace:
        payloads = [
            cast("dict[str, object]", json.loads(line))
            for line in evidence.read_text().splitlines()
        ]
        for payload in payloads:
            trace = cast("dict[str, object]", payload["retrieval_trace"])
            trace["eligible_shard_ids"] = []
            trace["selected_clip_ids"] = []
        _ = evidence.write_text(
            "".join(f"{json.dumps(payload)}\n" for payload in payloads),
            encoding="utf-8",
        )
    args = TransformersCliArgs(
        model="mock",
        fixture=FIXTURE,
        evidence=evidence,
        evidence_lane="heuristic",
        evidence_lineage=None,
        checkpoint=None,
        typed_memory=None,
        inference_manifest=None,
        require_frames=True,
        out=tmp_path / "predictions.jsonl",
        backend="mock",
    )

    with pytest.raises(TransformersCliUsageError, match="required QA frame missing"):
        _ = run_transformers_cli(args, env={})


def test_require_frames_records_input_frame_refs(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "smoke"
    _ = run_smoke_pipeline(FIXTURE, smoke_dir, {})
    evidence = smoke_dir / "evidence_packs.jsonl"
    first_pack = evidence.read_text(encoding="utf-8").splitlines()[0]
    _ = evidence.write_text(f"{first_pack}\n", encoding="utf-8")
    qa_fixture = tmp_path / "fixture"
    qa_fixture.mkdir()
    for name in ("sources.jsonl", "questions.jsonl"):
        lines = (FIXTURE / name).read_text(encoding="utf-8").splitlines()
        selected = lines if name == "sources.jsonl" else lines[:1]
        _ = (qa_fixture / name).write_text(
            "".join(f"{line}\n" for line in selected),
            encoding="utf-8",
        )
    frame_root = tmp_path / "frames"
    for source in read_source_streams(FIXTURE):
        for frame in source.frame_metadata:
            path = frame_root / source.video_id / f"{frame.frame_ref}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_bytes(b"mock-frame")
    output = tmp_path / "predictions.jsonl"
    args = TransformersCliArgs(
        model="mock",
        fixture=qa_fixture,
        evidence=evidence,
        evidence_lane="heuristic",
        evidence_lineage=None,
        checkpoint=None,
        typed_memory=None,
        inference_manifest=None,
        require_frames=True,
        out=output,
        backend="mock",
    )

    result = run_transformers_cli(
        args,
        env={"SMVQA_FRAME_ROOT": str(frame_root)},
    )

    predictions = tuple(
        PredictionRecord.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    )
    assert result.predictions == len(predictions)
    assert predictions
    assert all(prediction.input_frame_refs for prediction in predictions)
    assert all(
        "/" in frame_ref
        for prediction in predictions
        for frame_ref in prediction.input_frame_refs
    )
    assert all(prediction.prompt_sha256 for prediction in predictions)


def test_qa_cli_requires_explicit_evidence_lane() -> None:
    with pytest.raises(
        TransformersCliUsageError,
        match="requires --evidence-lane",
    ):
        _ = parse_cli_args(
            (
                "--model",
                "model",
                "--fixture",
                "fixture",
                "--evidence",
                "evidence.jsonl",
                "--out",
                "predictions.jsonl",
            ),
        )


def test_qa_cli_parses_inference_origin_artifacts() -> None:
    args = parse_cli_args(
        (
            "--model",
            "model",
            "--fixture",
            "fixture",
            "--evidence",
            "evidence.jsonl",
            "--evidence-lane",
            "student",
            "--inference-sources",
            "inference_inputs/sources.jsonl",
            "--inference-producer",
            "bin/spatial-infer",
            "--out",
            "predictions.jsonl",
        ),
    )

    assert args.inference_sources == Path("inference_inputs/sources.jsonl")
    assert args.inference_producer == Path("bin/spatial-infer")


def test_student_spatial_evidence_must_match_typed_projection(
    tmp_path: Path,
) -> None:
    typed_memory = tmp_path / "typed-memory.jsonl"
    _ = typed_memory.write_bytes(canonical_jsonl_bytes(_typed_object()))
    (record,) = read_typed_spatial_retrieval_records(typed_memory)
    tampered_geometry = dict(record.geometry or {})
    tampered_geometry["x"] = 999.0
    pack = EvidencePack(
        question_id="q",
        video_id=record.video_id,
        requested_stores=("spatial",),
        selected_stores=("spatial",),
        evidence_budget=1,
        evidence=(
            EvidenceItem(
                memory_id=record.memory_id,
                video_id=record.video_id,
                snippet=record.snippet,
                frame_refs=record.frame_refs,
                source_store="spatial",
                start_time=record.start_time,
                end_time=record.end_time,
                retrieval_score=1.0,
                geometry=tampered_geometry,
            ),
        ),
        causal_filtered_count=0,
    )

    with pytest.raises(
        TransformersCliUsageError,
        match="spatial evidence differs from typed memory",
    ):
        validate_spatial_evidence_against_typed_memory((pack,), (record,))
    non_spatial_item = pack.evidence[0].model_copy(update={"source_store": "semantic"})
    non_spatial_pack = pack.model_copy(update={"evidence": (non_spatial_item,)})
    with pytest.raises(
        TransformersCliUsageError,
        match="outside validated projection set",
    ):
        validate_spatial_evidence_against_typed_memory((non_spatial_pack,), (record,))


def test_student_spatial_evidence_matches_global_frame_ref_cap(
    tmp_path: Path,
) -> None:
    refs = tuple(f"frame-{index:02d}" for index in range(40))
    typed_memory = tmp_path / "typed-memory.jsonl"
    typed = _typed_object().model_copy(
        update={"evidence_refs": refs, "observation_count": len(refs)},
    )
    _ = typed_memory.write_bytes(canonical_jsonl_bytes(typed))
    (record,) = read_typed_spatial_retrieval_records(typed_memory)
    question = QuestionRequest(
        question_id="q-cap",
        video_id="video-1",
        question="Where was the mug?",
        question_time=3.0,
        answer_choices=(),
    )
    pack = retrieve_evidence(
        question,
        (record,),
        enabled_stores=frozenset({"spatial"}),
        options=RetrievalOptions(evidence_budget=1),
    )

    assert pack.evidence[0].frame_refs == refs[:RETRIEVAL_FRAME_REF_CAP]
    validate_spatial_evidence_against_typed_memory((pack,), (record,))

    shortened_item = pack.evidence[0].model_copy(
        update={"frame_refs": pack.evidence[0].frame_refs[:-1]},
    )
    shortened_pack = pack.model_copy(
        update={
            "evidence": (shortened_item,),
            "retrieval_trace": pack.retrieval_trace.model_copy(
                update={"frame_ref_count": RETRIEVAL_FRAME_REF_CAP - 1},
            ),
        },
    )
    with pytest.raises(
        TransformersCliUsageError,
        match="spatial evidence differs from typed memory",
    ):
        validate_spatial_evidence_against_typed_memory(
            (shortened_pack,),
            (record,),
        )


@pytest.mark.parametrize(
    "store",
    ["episodic", "semantic", "visual", "spatial"],
)
def test_student_evidence_must_match_every_canonical_store_projection(
    store: RetrievalStore,
) -> None:
    record = RetrievalMemoryRecord(
        memory_id=f"{store}:memory",
        source_store=store,
        video_id="video-1",
        start_time=1.0,
        end_time=2.0,
        snippet=f"canonical {store} snippet",
        frame_refs=(f"{store}-frame",),
        geometry={"record_type": "object"} if store == "spatial" else None,
    )
    question = QuestionRequest(
        question_id=f"q-{store}",
        video_id="video-1",
        question="What happened?",
        question_time=3.0,
        answer_choices=(),
    )
    pack = retrieve_evidence(
        question,
        (record,),
        enabled_stores=frozenset((record.source_store,)),
        options=RetrievalOptions(evidence_budget=1),
    )

    validate_student_evidence_against_memory((pack,), (record,))

    tampered = pack.model_copy(
        update={
            "evidence": (pack.evidence[0].model_copy(update={"snippet": "tampered"}),),
        },
    )
    with pytest.raises(
        TransformersCliUsageError,
        match="student evidence differs from canonical memory",
    ):
        validate_student_evidence_against_memory((tampered,), (record,))
    mismatched_store: RetrievalStore = "spatial" if store != "spatial" else "semantic"
    store_tampered = pack.model_copy(
        update={
            "evidence": (
                pack.evidence[0].model_copy(update={"source_store": mismatched_store}),
            ),
        },
    )
    with pytest.raises(
        TransformersCliUsageError,
        match="unknown canonical memory",
    ):
        validate_student_evidence_against_memory((store_tampered,), (record,))

    geometry_tampered = pack.model_copy(
        update={
            "evidence": (
                pack.evidence[0].model_copy(
                    update={"geometry": {"record_type": "bad"}}
                ),
            ),
        },
    )
    with pytest.raises(
        TransformersCliUsageError,
        match="differs from canonical memory",
    ):
        validate_student_evidence_against_memory((geometry_tampered,), (record,))


def test_explicit_qa_sensor_manifest_rejects_stale_source_inventory(
    tmp_path: Path,
) -> None:
    smoke_dir = tmp_path / "smoke"
    _ = run_smoke_pipeline(FIXTURE, smoke_dir, {})
    sources = read_source_streams(FIXTURE, use_sensor_manifest=False)
    sensor_manifest = tmp_path / "sensor.jsonl"
    _ = write_sensor_frame_manifest(sources, sensor_manifest)
    extra_frame = FrameMetadata(
        frame_ref="unexpected-frame",
        timestamp=100.5,
        description="inventory drift",
    )
    first = sources[0].model_copy(
        update={
            "frame_refs": (*sources[0].frame_refs, extra_frame.frame_ref),
            "frame_metadata": tuple(
                sorted(
                    (*sources[0].frame_metadata, extra_frame),
                    key=lambda frame: frame.timestamp,
                ),
            ),
        },
    )
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _ = (fixture / "sources.jsonl").write_text(
        "".join(f"{source.model_dump_json()}\n" for source in (first, *sources[1:])),
        encoding="utf-8",
    )
    _ = (fixture / "questions.jsonl").write_bytes(
        (FIXTURE / "questions.jsonl").read_bytes(),
    )
    args = TransformersCliArgs(
        model="mock",
        fixture=fixture,
        evidence=smoke_dir / "evidence_packs.jsonl",
        evidence_lane="heuristic",
        evidence_lineage=None,
        checkpoint=None,
        typed_memory=None,
        inference_manifest=None,
        require_frames=False,
        out=tmp_path / "predictions.jsonl",
        backend="mock",
        sensor_frame_manifest=sensor_manifest,
    )

    with pytest.raises(SensorFrameManifestError, match="stale or modified"):
        _ = run_transformers_cli(args, env={})


def test_student_evidence_lineage_is_required_and_binds_evidence(  # noqa: PLR0915
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    _ = evidence.write_text("evidence\n", encoding="utf-8")

    with pytest.raises(
        TransformersCliUsageError,
        match="student evidence requires --evidence-lineage",
    ):
        _ = validate_evidence_lineage(evidence, "student", None, None, None, None)

    checkpoint = tmp_path / "spatial_student.pt"
    _ = checkpoint.write_bytes(b"checkpoint")
    typed_memory = tmp_path / "typed_memory.jsonl"
    valid_typed_memory = canonical_jsonl_bytes(_fixture_typed_object())
    _ = typed_memory.write_bytes(valid_typed_memory)
    episodic_memory = tmp_path / "episodic.jsonl"
    semantic_memory = tmp_path / "worldmm_sv/semantic.jsonl"
    visual_memory = tmp_path / "worldmm_sv/visual.jsonl"
    semantic_memory.parent.mkdir()
    _ = episodic_memory.write_text("episodic\n", encoding="utf-8")
    _ = semantic_memory.write_text("semantic\n", encoding="utf-8")
    _ = visual_memory.write_text("visual\n", encoding="utf-8")
    memory_manifest = tmp_path / "memory_manifest.json"
    _ = memory_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodic_memory": str(episodic_memory),
                "semantic_memory": str(semantic_memory),
                "visual_memory": str(visual_memory),
                "spatial_memory": {"path": str(typed_memory)},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    typed_memory_sha256 = hashlib.sha256(typed_memory.read_bytes()).hexdigest()
    config = tmp_path / "config.yaml"
    sensor = tmp_path / "sensor.jsonl"
    _ = config.write_text("runtime: remote\n", encoding="utf-8")
    raw_sources = read_source_streams(FIXTURE, use_sensor_manifest=False)
    _ = write_sensor_frame_manifest(raw_sources, sensor)
    sensor_records = read_sensor_frame_manifest(sensor)
    data_root = FIXTURE
    data_digest = hashlib.sha256()
    for name in ("sources.jsonl", "questions.jsonl"):
        path = data_root / name
        data_digest.update(name.encode() + b"\0")
        data_digest.update(path.read_bytes())
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    sensor_sha256 = hashlib.sha256(sensor.read_bytes()).hexdigest()
    digest = data_digest.hexdigest()
    inference_sources = tmp_path / "inference_sources.jsonl"
    frame_assets = tmp_path / "frame_assets.sha256"
    inference_producer = tmp_path / "spatial-infer"
    _ = inference_sources.write_bytes((FIXTURE / "sources.jsonl").read_bytes())
    _ = frame_assets.write_text("frame-assets\n", encoding="utf-8")
    _ = inference_producer.write_bytes(b"spatial-infer")
    inference_manifest = tmp_path / "typed_memory.manifest.json"
    manifest_payload = {
        "schema_version": 1,
        "production_ready": True,
        "result_class": "student",
        "producer": "spatial-student",
        "sources_sha256": hashlib.sha256(inference_sources.read_bytes()).hexdigest(),
        "frame_assets_sha256": hashlib.sha256(frame_assets.read_bytes()).hexdigest(),
        "producer_sha256": hashlib.sha256(
            inference_producer.read_bytes(),
        ).hexdigest(),
        "checkpoint_sha256": checkpoint_sha256,
        "records_sha256": typed_memory_sha256,
        "sensor_sha256": sensor_sha256,
        "record_count": 1,
        "byte_budget_per_window": 1024,
        "window_count": 1,
        "max_window_bytes": typed_memory.stat().st_size,
        "actual_bytes": typed_memory.stat().st_size,
        "window_seconds": 30.0,
    }
    _ = inference_manifest.write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    lineage = EvidenceLineage(
        lane="student",
        producer="spatial-student",
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        model_contract_sha256="a" * 64,
        student_architecture_sha256="b" * 64,
        model_load_consensus_payload_sha256="c" * 64,
        model_load_consensus_file_sha256="d" * 64,
        checkpoint_sha256=checkpoint_sha256,
        inference_manifest_sha256=hashlib.sha256(
            inference_manifest.read_bytes(),
        ).hexdigest(),
        config_sha256=config_sha256,
        sensor_sha256=sensor_sha256,
        data_sha256=digest,
        **memory_artifact_hashes(memory_manifest),
    )
    for field in (
        "model_contract_sha256",
        "student_architecture_sha256",
        "model_load_consensus_payload_sha256",
        "model_load_consensus_file_sha256",
    ):
        missing = lineage.model_dump()
        del missing[field]
        with pytest.raises(ValidationError, match=field):
            _ = EvidenceLineage.model_validate(missing)
    lineage_path = tmp_path / "evidence.lineage.json"
    _ = lineage_path.write_text(lineage.model_dump_json(), encoding="utf-8")
    with pytest.raises(
        TransformersCliUsageError,
        match="student evidence requires --checkpoint",
    ):
        _ = validate_evidence_lineage(
            evidence,
            "student",
            lineage_path,
            None,
            None,
            None,
        )
    with pytest.raises(
        TransformersCliUsageError,
        match="student evidence requires --typed-memory",
    ):
        _ = validate_evidence_lineage(
            evidence,
            "student",
            lineage_path,
            checkpoint,
            None,
            None,
        )
    with pytest.raises(
        TransformersCliUsageError,
        match="student evidence requires --inference-manifest",
    ):
        _ = validate_evidence_lineage(
            evidence,
            "student",
            lineage_path,
            checkpoint,
            typed_memory,
            None,
        )
    _ = validate_evidence_lineage(
        evidence,
        "student",
        lineage_path,
        checkpoint,
        typed_memory,
        inference_manifest,
        config_path=config,
        sensor_path=sensor,
        data_root=data_root,
        memory_manifest_path=memory_manifest,
        inference_sources_path=inference_sources,
        frame_assets_path=frame_assets,
        inference_producer_path=inference_producer,
        sources=raw_sources,
        sensor_records=sensor_records,
        expected_trust_digests={
            "model_contract_sha256": "a" * 64,
            "student_architecture_sha256": "b" * 64,
            "model_load_consensus_payload_sha256": "c" * 64,
            "model_load_consensus_file_sha256": "d" * 64,
        },
    )
    for origin, field in (
        (inference_sources, "sources_sha256"),
        (frame_assets, "frame_assets_sha256"),
        (inference_producer, "producer_sha256"),
    ):
        original = origin.read_bytes()
        _ = origin.write_bytes(original + b"changed")
        with pytest.raises(TransformersCliUsageError, match=f"{field} mismatch"):
            _ = validate_evidence_lineage(
                evidence,
                "student",
                lineage_path,
                checkpoint,
                typed_memory,
                inference_manifest,
                config_path=config,
                sensor_path=sensor,
                data_root=data_root,
                memory_manifest_path=memory_manifest,
                inference_sources_path=inference_sources,
                frame_assets_path=frame_assets,
                inference_producer_path=inference_producer,
                sources=raw_sources,
                sensor_records=sensor_records,
            )
        _ = origin.write_bytes(original)
    with pytest.raises(
        TransformersCliUsageError,
        match="student evidence validation requires --lineage-config",
    ):
        _ = validate_evidence_lineage(
            evidence,
            "student",
            lineage_path,
            checkpoint,
            typed_memory,
            inference_manifest,
        )
    _ = episodic_memory.write_text("changed episodic\n", encoding="utf-8")
    with pytest.raises(TransformersCliUsageError, match="episodic_memory_sha256"):
        _ = validate_evidence_lineage(
            evidence,
            "student",
            lineage_path,
            checkpoint,
            typed_memory,
            inference_manifest,
            memory_manifest_path=memory_manifest,
        )
    _ = episodic_memory.write_text("episodic\n", encoding="utf-8")
    alternate_typed_memory = tmp_path / "alternate_typed_memory.jsonl"
    _ = alternate_typed_memory.write_bytes(valid_typed_memory)
    _ = memory_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodic_memory": str(episodic_memory),
                "semantic_memory": str(semantic_memory),
                "visual_memory": str(visual_memory),
                "spatial_memory": {"path": str(alternate_typed_memory)},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TransformersCliUsageError, match="typed_memory_sha256"):
        _ = memory_artifact_hashes(memory_manifest)
    _ = memory_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodic_memory": str(episodic_memory),
                "semantic_memory": str(semantic_memory),
                "visual_memory": str(visual_memory),
                "spatial_memory": {"path": str(typed_memory)},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    actual_memory_root = tmp_path / "actual-memory"
    linked_memory_root = tmp_path / "linked-memory"
    actual_memory_root.mkdir()
    linked_memory_root.symlink_to(actual_memory_root, target_is_directory=True)
    _ = (actual_memory_root / "memory_manifest.json").write_text(
        memory_manifest.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(TransformersCliUsageError, match="path contains a symlink"):
        _ = memory_artifact_hashes(linked_memory_root / "memory_manifest.json")
    _ = config.write_text("runtime: changed\n", encoding="utf-8")
    with pytest.raises(TransformersCliUsageError, match="config_sha256"):
        _ = validate_evidence_lineage(
            evidence,
            "student",
            lineage_path,
            checkpoint,
            typed_memory,
            inference_manifest,
            config_path=config,
            sensor_path=sensor,
            data_root=data_root,
        )
    _ = config.write_text("runtime: remote\n", encoding="utf-8")

    _ = evidence.write_text("changed\n", encoding="utf-8")
    with pytest.raises(
        TransformersCliUsageError,
        match="does not match evidence_sha256",
    ):
        _ = validate_evidence_lineage(
            evidence,
            "student",
            lineage_path,
            checkpoint,
            typed_memory,
            inference_manifest,
        )

    _ = evidence.write_text("evidence\n", encoding="utf-8")
    _ = checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(
        TransformersCliUsageError,
        match="does not match checkpoint_sha256",
    ):
        _ = validate_evidence_lineage(
            evidence,
            "student",
            lineage_path,
            checkpoint,
            typed_memory,
            inference_manifest,
        )

    _ = checkpoint.write_bytes(b"checkpoint")
    _ = typed_memory.write_text("changed typed memory\n", encoding="utf-8")
    with pytest.raises(
        TransformersCliUsageError,
        match="does not match typed_memory_sha256",
    ):
        _ = validate_evidence_lineage(
            evidence,
            "student",
            lineage_path,
            checkpoint,
            typed_memory,
            inference_manifest,
        )

    _ = typed_memory.write_bytes(valid_typed_memory)
    proxy_payload = {**manifest_payload, "production_ready": False}
    _ = inference_manifest.write_text(json.dumps(proxy_payload), encoding="utf-8")
    proxy_lineage = lineage.model_copy(
        update={
            "inference_manifest_sha256": hashlib.sha256(
                inference_manifest.read_bytes(),
            ).hexdigest(),
        },
    )
    _ = lineage_path.write_text(proxy_lineage.model_dump_json(), encoding="utf-8")
    with pytest.raises(
        TransformersCliUsageError,
        match="not production student output",
    ):
        _ = validate_evidence_lineage(
            evidence,
            "student",
            lineage_path,
            checkpoint,
            typed_memory,
            inference_manifest,
        )

    for update, message in (
        ({"schema_version": True}, "schema_version must be integer 1"),
        ({"sensor_sha256": "2" * 64}, "sensor_sha256 mismatch"),
    ):
        invalid_payload = {**manifest_payload, **update}
        _ = inference_manifest.write_text(
            json.dumps(invalid_payload),
            encoding="utf-8",
        )
        invalid_lineage = lineage.model_copy(
            update={
                "inference_manifest_sha256": hashlib.sha256(
                    inference_manifest.read_bytes(),
                ).hexdigest(),
            },
        )
        _ = lineage_path.write_text(
            invalid_lineage.model_dump_json(),
            encoding="utf-8",
        )
        with pytest.raises(TransformersCliUsageError, match=message):
            _ = validate_evidence_lineage(
                evidence,
                "student",
                lineage_path,
                checkpoint,
                typed_memory,
                inference_manifest,
            )


def test_legacy_evidence_is_allowed_only_in_explicit_heuristic_lane(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "legacy.jsonl"
    _ = evidence.write_text("legacy\n", encoding="utf-8")

    _ = validate_evidence_lineage(evidence, "heuristic", None, None, None, None)
    with pytest.raises(ValidationError, match="student evidence lineage missing"):
        _ = EvidenceLineage(
            lane="student",
            producer="spatial-student",
            evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        )


def test_teacher_oracle_qa_starts_from_an_output_free_pre_evaluation_lineage(
    tmp_path: Path,
) -> None:
    digest = "a" * 64
    shared = SharedQALineage(
        approved_salt="approved",
        world_size=1,
        question_map_sha256=digest,
        model_sha256=digest,
        prompt_sha256=digest,
        decoding_sha256=digest,
        runtime_sha256=digest,
        python_inventory_sha256=digest,
        torch_inventory_sha256=digest,
        transformers_inventory_sha256=digest,
        seed=0,
    )
    inputs = tuple(
        OracleQAInputLineage(
            variant=variant,
            memory_sha256=digest,
            evidence_sha256=digest,
            pre_evaluation_sha256=shared.sha256,
        )
        for variant in ("E0", "T0", "T1")
    )
    lineage = OracleQAPreEvaluationLineage(
        producer="offline-teacher",
        sensor_audit_sha256=digest,
        object_semantic_sha256=digest,
        geometry_sha256=digest,
        place_sha256=digest,
        typed_memory_sha256=digest,
        shared_input_sha256=digest,
        sensor_manifest_sha256=digest,
        selected_frame_inventory_sha256=digest,
        selected_frame_assets_sha256=digest,
        qa_inputs=inputs,
        shared_qa_lineage=shared,
    )
    path = tmp_path / "pre-evaluation-lineage.json"
    _ = path.write_text(lineage.model_dump_json(), encoding="utf-8")

    parsed = read_teacher_oracle_pre_evaluation_lineage(path)

    assert parsed == lineage
    assert "predictions_sha256" not in path.read_text(encoding="utf-8")
    assert "metrics_sha256" not in path.read_text(encoding="utf-8")
    assert "finalization_receipt_sha256" not in path.read_text(encoding="utf-8")


def test_teacher_oracle_qa_rejects_post_evaluation_lineage_at_startup(
    tmp_path: Path,
) -> None:
    digest = "a" * 64
    shared = SharedQALineage(
        approved_salt="approved",
        world_size=1,
        question_map_sha256=digest,
        model_sha256=digest,
        prompt_sha256=digest,
        decoding_sha256=digest,
        runtime_sha256=digest,
        seed=0,
    )
    post_evaluation = OracleEvidenceLineage(
        producer="offline-teacher",
        sensor_audit_sha256=digest,
        object_semantic_sha256=digest,
        geometry_sha256=digest,
        place_sha256=digest,
        typed_memory_sha256=digest,
        shared_input_sha256=digest,
        variants=tuple(
            OracleVariantLineage(
                variant=variant,
                memory_sha256=digest,
                evidence_sha256=digest,
                predictions_sha256=digest,
                metrics_sha256=digest,
                pre_evaluation_sha256=shared.sha256,
                finalization_receipt_sha256=digest,
                finalization_receipt_file_sha256=digest,
            )
            for variant in ("E0", "T0", "T1")
        ),
        shared_qa_lineage=shared,
    )
    path = tmp_path / "post-evaluation-lineage.json"
    _ = path.write_text(post_evaluation.model_dump_json(), encoding="utf-8")

    with pytest.raises(TransformersCliUsageError, match="pre-evaluation lineage"):
        _ = read_teacher_oracle_pre_evaluation_lineage(path)


def test_teacher_oracle_live_contract_accepts_matching_and_rejects_inventory_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "a" * 64
    shared = SharedQALineage(
        approved_salt="approved",
        world_size=1,
        question_map_sha256=digest,
        model_sha256=digest,
        prompt_sha256=digest,
        decoding_sha256=digest,
        runtime_sha256=digest,
        python_inventory_sha256=digest,
        torch_inventory_sha256=digest,
        transformers_inventory_sha256=digest,
        seed=7,
    )
    lineage = OracleQAPreEvaluationLineage(
        producer="offline-teacher",
        sensor_audit_sha256=digest,
        object_semantic_sha256=digest,
        geometry_sha256=digest,
        place_sha256=digest,
        typed_memory_sha256=digest,
        shared_input_sha256=digest,
        sensor_manifest_sha256=digest,
        selected_frame_inventory_sha256=digest,
        selected_frame_assets_sha256=digest,
        qa_inputs=tuple(
            OracleQAInputLineage(
                variant=variant,
                memory_sha256=digest,
                evidence_sha256=digest,
                pre_evaluation_sha256=shared.sha256,
            )
            for variant in ("E0", "T0", "T1")
        ),
        shared_qa_lineage=shared,
    )
    args = TransformersCliArgs(
        model="model",
        fixture=tmp_path,
        evidence=tmp_path / "evidence.jsonl",
        evidence_lane="teacher_oracle",
        evidence_lineage=tmp_path / "lineage.json",
        checkpoint=None,
        typed_memory=None,
        inference_manifest=None,
        require_frames=True,
        out=tmp_path / "predictions.jsonl",
        backend="gemma4",
    )

    def accept_backend_and_frames(*_args: object) -> None:
        return None

    def fixture_digest(_fixture: Path) -> str:
        return digest

    def matching_lineage(
        *_args: object,
    ) -> tuple[str, str, str, int, str, str, str]:
        return (digest, digest, digest, 7, digest, digest, digest)

    monkeypatch.setattr(
        qa_transformers,
        "_validate_teacher_oracle_backend_and_frames",
        accept_backend_and_frames,
    )
    monkeypatch.setattr(
        qa_transformers,
        "_fixture_data_sha256",
        fixture_digest,
    )
    matching = matching_lineage()
    monkeypatch.setattr(
        qa_transformers,
        "_live_teacher_oracle_lineage",
        matching_lineage,
    )

    validate_teacher_oracle_live_contract(args, {}, lineage)

    def drifted_lineage(
        *_args: object,
    ) -> tuple[str, str, str, int, str, str, str]:
        return (*matching[:6], "b" * 64)

    monkeypatch.setattr(
        qa_transformers,
        "_live_teacher_oracle_lineage",
        drifted_lineage,
    )
    with pytest.raises(
        TransformersCliUsageError,
        match="live runtime inventories do not match",
    ):
        validate_teacher_oracle_live_contract(args, {}, lineage)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"record_count": True}, "record_count must be a positive integer"),
        ({"byte_budget_per_window": 0}, "byte_budget_per_window must be"),
        ({"window_count": 0}, "window_count must be a positive integer"),
        ({"max_window_bytes": False}, "max_window_bytes must be"),
        ({"actual_bytes": -1}, "actual_bytes must be a non-negative integer"),
        ({"window_seconds": 0}, "window_seconds must be"),
        ({"max_window_bytes": 4097}, "max_window_bytes exceeds budget"),
        ({"actual_bytes": 2}, "actual_bytes mismatch"),
        ({"record_count": 2}, "record_count mismatch"),
    ],
)
def test_student_manifest_rejects_invalid_accounting(
    tmp_path: Path,
    update: dict[str, object],
    message: str,
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    _ = evidence.write_text("evidence\n", encoding="utf-8")
    checkpoint = tmp_path / "student.pt"
    _ = checkpoint.write_bytes(b"checkpoint")
    typed_memory = tmp_path / "typed_memory.jsonl"
    _ = typed_memory.write_bytes(canonical_jsonl_bytes(_typed_object()))
    digest = "1" * 64
    manifest = {
        "schema_version": 1,
        "production_ready": True,
        "result_class": "student",
        "producer": "spatial-student",
        "sources_sha256": "a" * 64,
        "frame_assets_sha256": "b" * 64,
        "producer_sha256": "c" * 64,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "records_sha256": hashlib.sha256(typed_memory.read_bytes()).hexdigest(),
        "sensor_sha256": digest,
        "record_count": 1,
        "byte_budget_per_window": 4096,
        "window_count": 1,
        "max_window_bytes": typed_memory.stat().st_size,
        "actual_bytes": typed_memory.stat().st_size,
        "window_seconds": 30.0,
        **update,
    }
    manifest_path = tmp_path / "inference.json"
    _ = manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    lineage = EvidenceLineage(
        lane="student",
        producer="spatial-student",
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        model_contract_sha256="a" * 64,
        student_architecture_sha256="b" * 64,
        model_load_consensus_payload_sha256="c" * 64,
        model_load_consensus_file_sha256="d" * 64,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        typed_memory_sha256=hashlib.sha256(typed_memory.read_bytes()).hexdigest(),
        inference_manifest_sha256=hashlib.sha256(
            manifest_path.read_bytes(),
        ).hexdigest(),
        config_sha256=digest,
        sensor_sha256=digest,
        data_sha256=digest,
        memory_manifest_sha256=digest,
        episodic_memory_sha256=digest,
        semantic_memory_sha256=digest,
        visual_memory_sha256=digest,
    )
    lineage_path = tmp_path / "lineage.json"
    _ = lineage_path.write_text(lineage.model_dump_json(), encoding="utf-8")

    with pytest.raises(TransformersCliUsageError, match=message):
        _ = validate_evidence_lineage(
            evidence,
            "student",
            lineage_path,
            checkpoint,
            typed_memory,
            manifest_path,
        )


def test_student_evidence_rejects_legacy_missing_trace() -> None:
    question = read_fixture_questions(FIXTURE)[0]
    pack = _pack(question)

    validate_evidence_trace_lane((pack,), "heuristic")
    with pytest.raises(
        TransformersCliUsageError,
        match="student evidence requires retrieval_trace",
    ):
        validate_evidence_trace_lane((pack,), "student")


def test_external_pack_set_rejects_unknown_duplicate_and_missing_questions() -> None:
    questions = read_fixture_questions(FIXTURE)[:2]
    by_id = {question.question_id: question for question in questions}
    first = _pack(questions[0])
    second = _pack(questions[1])

    with pytest.raises(TransformersCliUsageError, match="duplicate evidence pack"):
        validate_external_evidence_packs((first, first, second), by_id)

    unknown = first.model_copy(update={"question_id": "unknown-question"})
    with pytest.raises(
        TransformersCliUsageError,
        match="unknown evidence pack question",
    ):
        validate_external_evidence_packs((unknown, second), by_id)

    with pytest.raises(TransformersCliUsageError, match="missing evidence pack"):
        validate_external_evidence_packs((first,), by_id)


@pytest.mark.parametrize(
    ("pack_update", "item_update", "message"),
    [
        ({"video_id": "off-scope-video"}, {}, "pack video_id.*outside"),
        ({}, {"video_id": "off-scope-video"}, "evidence video_id.*outside"),
        ({}, {"start_time": 3.0, "end_time": 2.0}, "start_time exceeds"),
        ({}, {"end_time": 46.0}, "ends after question_time"),
    ],
)
def test_external_pack_rejects_scope_and_causal_time_violations(
    pack_update: dict[str, object],
    item_update: dict[str, object],
    message: str,
) -> None:
    question = read_fixture_questions(FIXTURE)[0]
    pack = _pack(question)
    if item_update:
        pack = pack.model_copy(
            update={
                "evidence": (pack.evidence[0].model_copy(update=item_update),),
            },
        )
    if pack_update:
        pack = pack.model_copy(update=pack_update)

    with pytest.raises(TransformersCliUsageError, match=message):
        validate_external_evidence_packs(
            (pack,),
            {question.question_id: question},
        )


def test_geometry_answer_requires_answerable_executor_proof_citation() -> None:
    question = read_fixture_questions(FIXTURE)[4]
    proof = execute_geometry(
        (
            {
                "entity_id": "opaque-entity-1",
                "label": "mug",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "coordinate_frame": "source_world",
                "uncertainty_m": 0.0,
                "provenance": "object_geometry",
                "evidence_refs": ["opaque-spatial-1"],
            },
            {
                "entity_id": "opaque-entity-2",
                "label": "notebook",
                "x": 0.5,
                "y": 0.0,
                "z": 0.0,
                "coordinate_frame": "source_world",
                "uncertainty_m": 0.0,
                "provenance": "object_geometry",
                "evidence_refs": ["opaque-spatial-2"],
            },
        ),
        GeometryQuery(
            operation="distance",
            coordinate_frame="source_world",
            subject="opaque-entity-1",
            object="opaque-entity-2",
        ),
    )
    without_citation = (
        '{"answerable":true,"ranked_choices":["B","C","A","D"],'
        '"answer":"B","confidence":0.9,"supporting_memory_ids":[],'
        '"geometry_proof_ids":[]}'
    )

    with pytest.raises(QAParseError, match="requires a geometry proof ID"):
        _ = parse_qa_output(
            question=question,
            raw_outputs=(without_citation,),
            prompt_token_count=10,
            raw_model_output_path=None,
            geometry_proofs=(proof,),
        )

    with_citation = without_citation.replace(
        '"geometry_proof_ids":[]',
        f'"geometry_proof_ids":["{proof.proof_id}"]',
    )
    prediction = parse_qa_output(
        question=question,
        raw_outputs=(with_citation,),
        prompt_token_count=10,
        raw_model_output_path=None,
        geometry_proofs=(proof,),
    )
    assert prediction.geometry_proof_ids == (proof.proof_id,)

    contradictory = with_citation.replace(
        '"ranked_choices":["B","C","A","D"],"answer":"B"',
        '"ranked_choices":["A","B","C","D"],"answer":"A"',
    )
    with pytest.raises(QAParseError, match="contradicts cited geometry proof"):
        _ = parse_qa_output(
            question=question,
            raw_outputs=(contradictory,),
            prompt_token_count=10,
            raw_model_output_path=None,
            geometry_proofs=(proof,),
        )


def test_unanswerable_geometry_proof_requires_model_abstention() -> None:
    question = read_fixture_questions(FIXTURE)[4]
    proof = execute_geometry(
        (),
        GeometryQuery(
            operation="distance",
            coordinate_frame="source_world",
            subject="missing-1",
            object="missing-2",
        ),
    )
    assert not proof.answerable
    answer = (
        '{"answerable":true,"ranked_choices":["B","C","A","D"],'
        '"answer":"B","confidence":0.9,"supporting_memory_ids":[],'
        '"geometry_proof_ids":[]}'
    )
    with pytest.raises(QAParseError, match="requires model abstention"):
        _ = parse_qa_output(
            question=question,
            raw_outputs=(answer,),
            prompt_token_count=10,
            raw_model_output_path=None,
            geometry_proofs=(proof,),
        )

    abstention = answer.replace(
        '"answerable":true',
        '"answerable":false',
    ).replace('"answer":"B"', '"answer":null')
    prediction = parse_qa_output(
        question=question,
        raw_outputs=(abstention,),
        prompt_token_count=10,
        raw_model_output_path=None,
        geometry_proofs=(proof,),
    )
    assert not prediction.answerable


def test_causal_wearer_pose_requires_trusted_source_frame_and_covariance() -> None:
    question = read_fixture_questions(FIXTURE)[4]
    source = read_source_streams(FIXTURE, use_sensor_manifest=False)[0]
    covariance = (0.0,) * 35 + (4.0,)
    poses = tuple(
        sample.model_copy(
            update={
                "source": "vio",
                "processing_mode": "online_causal",
                "observed_through_time": sample.timestamp,
                "coordinate_frame": "source_world",
                "pose_covariance_xyz_m_rpy_deg": covariance,
            },
        )
        for sample in source.pose_samples
    )
    source = source.model_copy(update={"pose_samples": poses})
    spatial = EvidenceItem(
        memory_id="spatial-pose",
        video_id=source.video_id,
        snippet="spatial pose evidence",
        frame_refs=(),
        source_store="spatial",
        start_time=1.0,
        end_time=2.0,
        retrieval_score=1.0,
        geometry={"coordinate_frame": "source_world"},
    )
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("spatial",),
        selected_stores=("spatial",),
        evidence_budget=1,
        evidence=(spatial,),
        causal_filtered_count=0,
    )

    assert causal_wearer_pose((source,), question, pack) == (90.0, 2.0)
    for rejected_source, processing_mode in (
        ("slam", "offline"),
        ("ground_truth", "online_causal"),
        ("model_only", "online_causal"),
    ):
        untrusted = source.model_copy(
            update={
                "pose_samples": tuple(
                    sample.model_copy(
                        update={
                            "source": rejected_source,
                            "processing_mode": processing_mode,
                        },
                    )
                    for sample in poses
                ),
            },
        )
        assert causal_wearer_pose((untrusted,), question, pack) is None

    missing_certificate = source.model_copy(
        update={
            "pose_samples": tuple(
                sample.model_copy(update={"observed_through_time": None})
                for sample in poses
            ),
        },
    )
    assert causal_wearer_pose((missing_certificate,), question, pack) is None

    future_certificate = source.model_copy(
        update={
            "pose_samples": tuple(
                sample.model_copy(
                    update={
                        "observed_through_time": max(
                            sample.timestamp,
                            question.question_time + 1.0,
                        ),
                    },
                )
                for sample in poses
            ),
        },
    )
    assert causal_wearer_pose((future_certificate,), question, pack) is None


def test_source_pose_90_degrees_reaches_direction_proof_without_unit_conversion() -> (
    None
):
    covariance = (0.0,) * 36
    source = SourceStreamExample(
        video_id="video-direction",
        start_time=0.0,
        end_time=3.0,
        pose_samples=(
            PoseSample(
                timestamp=1.0,
                x=0.0,
                y=0.0,
                z=1.5,
                yaw_degrees=90.0,
                source="vio",
                processing_mode="online_causal",
                observed_through_time=1.0,
                coordinate_frame="room:1",
                pose_covariance_xyz_m_rpy_deg=covariance,
            ),
        ),
    )
    question = QuestionRequest(
        question_id="direction-90-degrees",
        video_id=source.video_id,
        question="What direction is target:1 from origin:1?",
        question_time=2.0,
        answer_choices=(
            AnswerChoice(choice_id="A", text="front", choice_ltype="answer"),
            AnswerChoice(choice_id="B", text="behind", choice_ltype="answer"),
        ),
    )
    evidence = tuple(
        EvidenceItem(
            memory_id=entity_id,
            video_id=source.video_id,
            snippet=entity_id,
            frame_refs=(f"frame-{entity_id}",),
            source_store="spatial",
            start_time=1.0,
            end_time=1.0,
            retrieval_score=1.0,
            geometry={
                "entity_id": entity_id,
                "label": label,
                "x": x,
                "y": 0.0,
                "z": 0.0,
                "coordinate_frame": "room:1",
                "uncertainty_m": 0.01,
                "provenance": "observed",
            },
        )
        for entity_id, label, x in (
            ("target:1", "target", 2.0),
            ("origin:1", "origin", 0.0),
        )
    )
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("spatial",),
        selected_stores=("spatial",),
        evidence_budget=2,
        evidence=evidence,
        causal_filtered_count=0,
    )

    wearer_pose = causal_wearer_pose((source,), question, pack)
    assert wearer_pose is not None
    assert wearer_pose == (90.0, 0.0)
    proofs = geometry_proofs_for_question(
        question,
        pack,
        wearer_yaw_degrees=wearer_pose[0],
        wearer_yaw_uncertainty_degrees=wearer_pose[1],
    )

    assert len(proofs) == 1
    assert proofs[0].answerable
    assert proofs[0].value == "front"


def test_prompt_withholds_spatial_payload_but_keeps_non_spatial_snippet() -> None:
    question = read_fixture_questions(FIXTURE)[0]
    pack = EvidencePack(
        question_id=question.question_id,
        video_id=question.video_id,
        requested_stores=("spatial", "semantic"),
        selected_stores=("spatial", "semantic"),
        evidence_budget=2,
        evidence=(
            EvidenceItem(
                memory_id="opaque-spatial-id",
                video_id=question.video_id,
                snippet="subject left_of object at x=9.875 distance_m=7.25",
                frame_refs=(),
                source_store="spatial",
                start_time=1.0,
                end_time=2.0,
                retrieval_score=1.0,
                geometry={"relation": "left_of", "distance_m": 7.25, "x": 9.875},
            ),
            EvidenceItem(
                memory_id="semantic-id",
                video_id=question.video_id,
                snippet="the lamp was switched on",
                frame_refs=(),
                source_store="semantic",
                start_time=1.0,
                end_time=2.0,
                retrieval_score=0.9,
            ),
        ),
        causal_filtered_count=0,
    )

    prompt = build_qa_prompt(question, pack)

    assert "opaque-spatial-id" in prompt
    assert SPATIAL_EVIDENCE_PLACEHOLDER in prompt
    assert "left_of" not in prompt
    assert "9.875" not in prompt
    assert "7.25" not in prompt
    assert "the lamp was switched on" in prompt


def test_support_ids_without_trusted_pack_are_rejected() -> None:
    question = read_fixture_questions(FIXTURE)[0]
    raw = (
        '{"answerable":true,"ranked_choices":["A","B","C","D"],'
        '"answer":"A","confidence":0.9,'
        '"supporting_memory_ids":["fabricated"]}'
    )

    with pytest.raises(QAParseError, match="require a trusted evidence pack"):
        _ = parse_qa_output(
            question=question,
            raw_outputs=(raw,),
            prompt_token_count=10,
            raw_model_output_path=None,
        )


def _pack(question: QuestionRequest) -> EvidencePack:
    question_id = question.question_id
    video_id = question.video_id
    return EvidencePack(
        question_id=question_id,
        video_id=video_id,
        requested_stores=("semantic",),
        selected_stores=("semantic",),
        evidence_budget=1,
        evidence=(
            EvidenceItem(
                memory_id=f"memory-{question_id}",
                video_id=video_id,
                snippet="safe semantic evidence",
                frame_refs=(),
                source_store="semantic",
                start_time=1.0,
                end_time=2.0,
                retrieval_score=1.0,
            ),
        ),
        causal_filtered_count=0,
    )
