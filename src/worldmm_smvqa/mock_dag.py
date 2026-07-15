# ruff: noqa: ANN401, EM101, TRY003
# pyright: reportAny=false
# pyright: reportExplicitAny=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportCallInDefaultInitializer=false
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from worldmm_smvqa.chunking import build_chunks
from worldmm_smvqa.memory_sources import build_source_memories
from worldmm_smvqa.model_contract import (
    encode_declared_json,
    load_model_boundary_contract,
    load_student_architecture,
    raw_sha256,
)
from worldmm_smvqa.qa import MockQABackend, parse_qa_output
from worldmm_smvqa.qa_prompt import build_qa_prompt
from worldmm_smvqa.retrieval import (
    RetrievalOptions,
    build_retrieval_records,
    retrieve_evidence,
)
from worldmm_smvqa.schema import FrameMetadata, QuestionRequest, SourceStreamExample
from worldmm_smvqa.spatial_train import (
    LocalMockCheckpointAuthorizationV1,
    StudentConfig,
    build_student,
    compute_losses,
    load_checkpoint,
    save_checkpoint_atomic,
)
from worldmm_smvqa.worldmm.llm_memory import (
    CLIP_PROMPT,
    SHARD_PROMPT,
    TRIPLET_PROMPT,
    build_llm_episodic_graph,
    build_llm_semantic_memory,
    build_llm_visual_memory,
)
from worldmm_smvqa.worldmm.typed_memory import (
    ObjectGeometry,
    ObjectMemoryRecord,
    SpatialUncertainty,
    ValidityInterval,
)

if TYPE_CHECKING:
    from worldmm_smvqa.worldmm.visual import VisualMemoryRecord


@dataclass(frozen=True, slots=True)
class MockDagResult:
    fixture_sha256: str
    student_architecture_sha256: str
    observations: int
    qwen_store_counts: dict[str, int]
    selected_stores: tuple[str, ...]
    loss: float
    gradient_parameter_count: int
    optimizer_steps: int
    parameter_changed: bool
    checkpoint_round_trip: bool
    prediction_answerable: bool


class MockDagError(RuntimeError):
    pass


def run_local_mock_dag(  # noqa: PLR0915
    fixture: Path,
    *,
    student_architecture: Path = Path("configs/spatial/student_architecture_v1.json"),
) -> MockDagResult:
    """Run the canonical CPU-only fixture through production-owned consumers."""
    contract, fixture_sha256 = load_model_boundary_contract(fixture)
    architecture, architecture_sha256 = load_student_architecture(
        student_architecture,
        expected_model_contract_sha256=fixture_sha256,
    )
    transaction = cast("dict[str, Any]", contract["transaction"])
    raw_source = SourceStreamExample.model_validate(transaction["video"])
    source = raw_source.model_copy(
        update={
            "frame_metadata": tuple(
                FrameMetadata(
                    frame_ref=frame_ref,
                    timestamp=float(index) + 0.5,
                    description=raw_source.captions[0],
                )
                for index, frame_ref in enumerate(raw_source.frame_refs)
            ),
        },
    )
    question = QuestionRequest.model_validate(transaction["question"])
    observations = cast("list[dict[str, Any]]", transaction["observations"])

    chunks = build_chunks((source,))
    source_memories = build_source_memories(chunks)
    prompt_trace: list[str] = []

    def generate(prompt: str) -> str:
        prompt_trace.append(prompt)
        if prompt.startswith(CLIP_PROMPT.split("{", maxsplit=1)[0]):
            return '{"summary":"A red mug is on the table."}'
        if prompt.startswith(SHARD_PROMPT.split("{", maxsplit=1)[0]):
            return '{"summary":"The clip shows a red mug on a table."}'
        if prompt.startswith(TRIPLET_PROMPT.split("{", maxsplit=1)[0]):
            return (
                '{"triplets":[{"subject":"red mug","predicate":"on","object":"table"}]}'
            )
        return '{"action":"keep"}'

    episodic = build_llm_episodic_graph(chunks, source_memories, generate)
    semantic = build_llm_semantic_memory(episodic, generate)
    visual = _build_visual_fixture(source)
    if not episodic or not semantic or not visual or not prompt_trace:
        raise MockDagError("production Qwen memory consumers produced an empty store")

    spatial = (_spatial_record(source.video_id),)
    retrieval_records = build_retrieval_records(episodic, semantic, visual, spatial)
    pack = retrieve_evidence(
        question,
        retrieval_records,
        enabled_stores=frozenset(("episodic", "semantic", "visual", "spatial")),
        options=RetrievalOptions(evidence_budget=6),
    )
    selected = set(pack.selected_stores)
    if "spatial" not in selected or not selected.intersection(
        {"episodic", "semantic", "visual"}
    ):
        raise MockDagError("retrieval did not join Qwen-owned and spatial evidence")

    torch = __import__("torch")
    config = StudentConfig(
        input_dim=architecture.input_dim,
        teacher_dim=architecture.teacher_dim,
        geometry_dim=architecture.geometry_dim,
        association_classes=architecture.association_classes,
        hidden_dim=architecture.hidden_dim,
        learning_rate=architecture.learning_rate,
        rate_normalizer_bytes=architecture.rate_normalizer_bytes,
        teacher_cache_sha256=hashlib.sha256(
            json.dumps(observations, separators=(",", ":")).encode(),
        ).hexdigest(),
        model_contract_sha256=fixture_sha256,
    )
    batch = _student_batch(torch, observations)
    model = build_student(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    optimizer.zero_grad(set_to_none=True)
    losses = compute_losses(
        model(batch["features"]),
        batch,
        config,
        aggregate_distributed=False,
    )
    total = losses["total"]
    if not math.isfinite(float(total.detach().item())):
        raise MockDagError("student total loss is not finite")
    total.backward()
    gradient_count = sum(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.parameters()
    )
    if gradient_count == 0:
        raise MockDagError("student produced no finite non-empty gradients")
    optimizer.step()
    changed = any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, model.parameters(), strict=True)
    )
    if not changed:
        raise MockDagError("optimizer step did not change parameters")

    local_payload = {
        "schema_version": "local-mock-authorization-v1",
        "fixture_sha256": fixture_sha256,
        "student_architecture_sha256": architecture_sha256,
        "test_transaction_id": question.question_id,
        "code_tree_sha256": raw_sha256(Path(__file__).read_bytes()),
    }
    authorization = LocalMockCheckpointAuthorizationV1(
        kind="local_mock_v1",
        local_authorization_sha256=raw_sha256(
            encode_declared_json(local_payload, "local-mock-authorization-v1"),
        ),
        model_contract_sha256=fixture_sha256,
        student_architecture_sha256=architecture_sha256,
    )
    with tempfile.TemporaryDirectory(prefix="worldmm-mock-dag-") as temporary:
        checkpoint = Path(temporary) / "student.pt"
        save_checkpoint_atomic(
            checkpoint,
            model=model,
            optimizer=optimizer,
            config=config,
            authorization=authorization,
            next_epoch=1,
            global_step=1,
        )
        restored = build_student(config)
        restored_optimizer = torch.optim.AdamW(
            restored.parameters(),
            lr=config.learning_rate,
        )
        counters = load_checkpoint(
            checkpoint,
            model=restored,
            optimizer=restored_optimizer,
            device=torch.device("cpu"),
            expected_config=config,
            expected_authorization=authorization,
        )
        round_trip = counters == (1, 1) and all(
            torch.equal(left, right)
            for left, right in zip(
                model.state_dict().values(),
                restored.state_dict().values(),
                strict=True,
            )
        )

    prompt = build_qa_prompt(question, pack)
    backend = MockQABackend()
    raw_outputs = backend.raw_outputs(prompt, question, pack)
    prediction = parse_qa_output(
        question=question,
        raw_outputs=raw_outputs,
        prompt_token_count=len(prompt.split()),
        raw_model_output_path=None,
        evidence_pack=pack,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )
    return MockDagResult(
        fixture_sha256=fixture_sha256,
        student_architecture_sha256=architecture_sha256,
        observations=len(observations),
        qwen_store_counts={
            "episodic": len(episodic),
            "semantic": len(semantic),
            "visual": len(visual),
        },
        selected_stores=pack.selected_stores,
        loss=float(total.detach().item()),
        gradient_parameter_count=gradient_count,
        optimizer_steps=1,
        parameter_changed=changed,
        checkpoint_round_trip=round_trip,
        prediction_answerable=prediction.answerable,
    )


def _build_visual_fixture(
    source: SourceStreamExample,
) -> tuple[VisualMemoryRecord, ...]:
    with tempfile.TemporaryDirectory(prefix="worldmm-mock-frames-") as temporary:
        frame_root = Path(temporary)
        video_root = frame_root / source.video_id
        video_root.mkdir(parents=True)
        for frame_ref in source.frame_refs:
            _ = (video_root / f"{frame_ref}.jpg").write_bytes(b"fixture-only")
        return build_llm_visual_memory(
            (source,),
            frame_root=frame_root,
            caption=lambda _path: "red mug on table",
        )


def _student_batch(torch: Any, observations: list[dict[str, Any]]) -> dict[str, object]:
    return {
        "sample_id": [item["observation_id"] for item in observations],
        "features": torch.tensor(
            [item["features"] for item in observations], dtype=torch.float32
        ),
        "teacher_embedding": torch.tensor(
            [item["teacher_embedding"] for item in observations],
            dtype=torch.float32,
        ),
        "type_target": torch.tensor(
            [item["type_target"] for item in observations], dtype=torch.long
        ),
        "geometry_target": torch.tensor(
            [item["geometry_target"] for item in observations],
            dtype=torch.float32,
        ),
        "association_target": torch.tensor(
            [item["association_target"] for item in observations],
            dtype=torch.long,
        ),
        "uncertainty_target": torch.tensor(
            [item["uncertainty_target"] for item in observations],
            dtype=torch.float32,
        ),
        "byte_cost": torch.tensor(
            [item["byte_cost"] for item in observations], dtype=torch.float32
        ),
    }


def _spatial_record(video_id: str) -> ObjectMemoryRecord:
    return ObjectMemoryRecord(
        memory_id="mock-spatial-red-mug",
        source_video_id=video_id,
        entity_id="red-mug",
        instance_id="red-mug-1",
        local_frame_id="source_world",
        geometry_uncertainty=SpatialUncertainty(
            covariance_xyz=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
            standard_deviation_m=0.1,
        ),
        validity=ValidityInterval(start_time=0.5, end_time=2.0),
        first_seen_time=0.5,
        last_seen_time=1.5,
        observation_count=2,
        confidence=0.95,
        provenance="observed",
        evidence_refs=("obs-0", "obs-1"),
        geometry=ObjectGeometry(centroid=(1.0, 0.0, 0.0), extent=(0.1, 0.1, 0.2)),
        semantic_label="red mug",
    )
