from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from worldmm_smvqa.chunking import build_chunks
from worldmm_smvqa.retrieval import (
    RetrievalOptions,
    build_retrieval_records,
    retrieve_evidence,
)
from worldmm_smvqa.schema import (
    ObjectMetadata,
    PoseSample,
    QuestionRequest,
    SourceStreamExample,
    StreamChunk,
)
from worldmm_smvqa.worldmm.spatial_compression import (
    CompactJsonSpatialMemoryCodec,
    DeltaTopKSpatialTokenDecoder,
    ExperimentOption,
    ObjectToken,
    RelationToken,
    SpatialCompressionError,
    SpatialCompressionOptions,
    SpatialExperimentConfig,
    SpatialGeometryFeatureSet,
    SpatialMemoryCodec,
    SpatialTokenCandidate,
    StructuredSpatialGeometryEncoder,
    ZoneToken,
    build_compressed_spatial_memory,
    build_spatial_token_candidates,
    decode_spatial_token,
    register_spatial_geometry_encoder,
    register_spatial_projection_head,
    register_spatial_token_decoder,
    register_spatial_token_selector,
    resolve_spatial_experiment_config,
    resolve_spatial_memory_model,
    select_spatial_tokens,
)
from worldmm_smvqa.worldmm.spatial_types import (
    SpatialRelationKind,
    SpatialTokenRecord,
)


def _clips(source: SourceStreamExample) -> tuple[StreamChunk, ...]:
    return tuple(
        chunk for chunk in build_chunks((source,)) if chunk.granularity == "clip_30s"
    )


def _repeated_source() -> SourceStreamExample:
    detections = tuple(
        detection
        for index in range(5)
        for detection in (
            ObjectMetadata(
                label="mug",
                confidence=0.9,
                start_time=float(index),
                end_time=float(index) + 0.8,
                x=0.12,
                y=1.0,
                z=0.75,
            ),
            ObjectMetadata(
                label="notebook",
                confidence=0.95,
                start_time=float(index),
                end_time=float(index) + 0.8,
                x=1.02,
                y=0.0,
                z=0.25,
            ),
        )
    )
    return SourceStreamExample(
        video_id="compressed",
        start_time=0.0,
        end_time=60.0,
        object_detections=detections,
        pose_samples=(
            PoseSample(timestamp=0.0, x=0.0, y=0.0, z=1.5),
            PoseSample(timestamp=30.0, x=0.0, y=0.0, z=1.5),
        ),
    )


def test_compression_removes_repeated_geometry_and_inverse_relations() -> None:
    # Given: two static objects detected once per second at unchanged coordinates.
    source = _repeated_source()

    # When: compact spatial memory is built.
    result = build_compressed_spatial_memory(
        (source,),
        _clips(source),
        env={},
        measure_legacy=True,
    )
    tokens = tuple(
        record for record in result.records if isinstance(record, SpatialTokenRecord)
    )
    decoded = tuple(decode_spatial_token(record) for record in tokens)
    objects = tuple(item for item in decoded if isinstance(item, ObjectToken))
    relations = tuple(item for item in decoded if isinstance(item, RelationToken))

    # Then: repeated observations collapse and only canonical relation directions stay.
    assert {item.object_label for item in objects} == {"mug", "notebook"}
    assert {item.relation for item in relations} == {
        "above",
        "in_front_of",
        "left_of",
        "near",
    }
    assert result.raw_record_count > result.candidate_count
    assert result.compressed_bytes < result.raw_bytes / 2


def test_object_tokens_bound_quantization_error() -> None:
    # Given: geometry that is not aligned to the 0.25m codec grid.
    source = _repeated_source()

    # When: the object token is encoded then decoded.
    result = build_compressed_spatial_memory(
        (source,),
        _clips(source),
        env={},
    )
    mug = next(
        decoded
        for record in result.records
        if isinstance(record, SpatialTokenRecord)
        and isinstance((decoded := decode_spatial_token(record)), ObjectToken)
        and decoded.object_label == "mug"
    )

    # Then: nearest-grid error stays within half one quantization step.
    assert mug.x == pytest.approx(0.0)
    assert abs(mug.x - 0.12) <= 0.125
    assert abs(mug.y - 1.0) <= 0.125
    assert abs(mug.z - 0.75) <= 0.125
    assert mug.uncertainty_m == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("relation", "canonical_relation"),
    [
        ("right_of", "left_of"),
        ("behind", "in_front_of"),
        ("below", "above"),
    ],
)
def test_codec_canonicalizes_inverse_relation_endpoints(
    relation: SpatialRelationKind,
    canonical_relation: SpatialRelationKind,
) -> None:
    # Given: a direct codec caller supplies an inverse relation direction.
    codec = CompactJsonSpatialMemoryCodec()
    token = RelationToken(
        scale_m=0.25,
        subject="mug",
        relation=relation,
        object="notebook",
        zone_id="desk",
        distance_m=0.5,
        subject_instance_id="mug-1",
        object_instance_id="notebook-1",
        valid_from=1.0,
        valid_to=2.0,
    )

    # When: the compact payload is encoded and decoded.
    decoded = codec.decode(
        SpatialTokenRecord(
            memory_id="relation",
            video_id="video",
            start_time=1.0,
            end_time=2.0,
            token=codec.encode(token),
            importance=1.0,
        ),
    )

    # Then: canonical direction preserves meaning by swapping both endpoints.
    assert isinstance(decoded, RelationToken)
    assert decoded.relation == canonical_relation
    assert decoded.subject == "notebook"
    assert decoded.object == "mug"
    assert decoded.subject_instance_id == "notebook-1"
    assert decoded.object_instance_id == "mug-1"
    assert decoded.distance_m == pytest.approx(0.5)
    assert decoded.valid_from == pytest.approx(1.0)
    assert decoded.valid_to == pytest.approx(2.0)


def test_moved_object_emits_causal_delta_not_future_state() -> None:
    # Given: one object moves after an early question.
    source = SourceStreamExample(
        video_id="moving",
        start_time=0.0,
        end_time=60.0,
        object_detections=(
            ObjectMetadata(
                label="mug",
                confidence=0.9,
                start_time=5.0,
                end_time=6.0,
                x=0.0,
                y=0.0,
                z=1.0,
            ),
            ObjectMetadata(
                label="mug",
                confidence=0.9,
                start_time=40.0,
                end_time=41.0,
                x=2.0,
                y=0.0,
                z=1.0,
            ),
        ),
    )
    result = build_compressed_spatial_memory(
        (source,),
        _clips(source),
        env={},
    )
    memories = build_retrieval_records((), (), (), result.records)
    question = QuestionRequest(
        question_id="before-move",
        video_id="moving",
        question="Where was the mug?",
        question_time=30.0,
        answer_choices=(),
    )

    # When: retrieval runs before the second observation.
    pack = retrieve_evidence(
        question,
        memories,
        enabled_stores=frozenset({"spatial"}),
        options=RetrievalOptions(evidence_budget=6),
    )

    # Then: the early token survives and the future delta is filtered.
    object_ids = {
        item.memory_id for item in pack.evidence if ":object:mug:" in item.memory_id
    }
    assert object_ids == {"spatial_token:moving:object:mug:5"}
    assert pack.causal_filtered_count == 1


def test_relation_deltas_are_causal_and_emit_returned_state() -> None:
    # Given: low-confidence A-left-B changes direction, then returns after upgrades.
    source = SourceStreamExample(
        video_id="relation-return",
        start_time=0.0,
        end_time=30.0,
        object_detections=tuple(
            detection
            for timestamp, mug_x, notebook_x, confidence in (
                (0.0, 0.25, 1.25, 0.2),
                (10.0, 1.5, 0.5, 1.0),
                (20.0, 0.25, 1.25, 1.0),
            )
            for detection in (
                ObjectMetadata(
                    label="mug",
                    confidence=confidence,
                    start_time=timestamp,
                    end_time=timestamp + 0.5,
                    x=mug_x,
                    y=0.0,
                    z=1.0,
                ),
                ObjectMetadata(
                    label="notebook",
                    confidence=confidence,
                    start_time=timestamp,
                    end_time=timestamp + 0.5,
                    x=notebook_x,
                    y=0.0,
                    z=1.0,
                ),
            )
        ),
    )

    # When: the decoder builds relation candidates over the complete stream.
    candidates = build_spatial_token_candidates((source,), env={}).candidates
    first_left_candidate = next(
        candidate
        for candidate in candidates
        if candidate.record.start_time == 0.0
        and isinstance(
            (token := decode_spatial_token(candidate.record)),
            RelationToken,
        )
        and token.relation == "left_of"
        and token.subject == "mug"
        and token.object == "notebook"
    )
    selected = build_compressed_spatial_memory(
        (source,),
        _clips(source),
        env={},
        options=SpatialCompressionOptions(
            max_tokens_per_window=100,
            max_bytes_per_window=100_000,
            min_keep_score=0.0,
        ),
    ).records
    left_relations = tuple(
        (candidate, token)
        for candidate in selected
        if isinstance(candidate, SpatialTokenRecord)
        and isinstance((token := decode_spatial_token(candidate)), RelationToken)
        and token.relation == "left_of"
        and token.subject == "mug"
        and token.object == "notebook"
    )

    # Then: future confidence does not alter the first score and A-left-B re-emits.
    assert tuple(item.start_time for item, _token in left_relations) == (
        0.0,
        20.0,
    )
    assert first_left_candidate.features["confidence"] == pytest.approx(0.2)


def test_selector_enforces_per_window_token_budget() -> None:
    # Given: more distinct geometry candidates than the device window budget.
    source = SourceStreamExample(
        video_id="budget",
        start_time=0.0,
        end_time=30.0,
        object_detections=tuple(
            ObjectMetadata(
                label=f"object_{index}",
                confidence=0.9,
                start_time=float(index),
                end_time=float(index) + 0.5,
                x=float(index),
                y=0.0,
                z=1.0,
            )
            for index in range(5)
        ),
        pose_samples=(PoseSample(timestamp=0.0, x=0.0, y=0.0, z=1.5),),
    )

    # When: the static token budget is two.
    result = build_compressed_spatial_memory(
        (source,),
        _clips(source),
        env={},
        options=SpatialCompressionOptions(max_tokens_per_window=2),
    )
    tokens = tuple(
        record for record in result.records if isinstance(record, SpatialTokenRecord)
    )

    # Then: prefix-causal admission keeps the first zone and one object.
    assert len(tokens) == 2
    assert sum(":zone:" in record.memory_id for record in tokens) == 1
    assert sum(":object:" in record.memory_id for record in tokens) == 1
    assert result.experiment.token_budget == 2


class _FeatureScoreSelector:
    def score(self, features: Mapping[str, float]) -> float:
        return features["score"]


def _scored_candidate(
    memory_id: str,
    *,
    score: float,
    end_time: float = 1.0,
    frame_refs: tuple[str, ...] = (),
) -> SpatialTokenCandidate:
    return SpatialTokenCandidate(
        record=SpatialTokenRecord(
            memory_id=memory_id,
            video_id="byte-budget",
            start_time=end_time - 0.5,
            end_time=end_time,
            token='["Z",25,"zone",0,0,0]',  # noqa: S106
            importance=0.0,
            frame_refs=frame_refs,
        ),
        features={"score": score},
    )


def _selected_jsonl_bytes(candidate: SpatialTokenCandidate) -> int:
    score = candidate.features["score"]
    record = candidate.record.model_copy(update={"importance": round(score, 6)})
    return len(record.model_dump_json().encode("utf-8")) + 1


def test_selector_enforces_strict_serialized_byte_budget() -> None:
    # Given: two equal-size writes that individually fit but jointly exceed budget.
    candidates = (
        _scored_candidate("high", score=0.9),
        _scored_candidate("low_", score=0.8),
    )
    candidate_bytes = tuple(_selected_jsonl_bytes(item) for item in candidates)
    assert candidate_bytes[0] == candidate_bytes[1]
    byte_budget = sum(candidate_bytes) - 1

    # When: admission applies actual serialized JSONL byte cost.
    selected = select_spatial_tokens(
        candidates,
        selector=_FeatureScoreSelector(),
        options=SpatialCompressionOptions(
            max_tokens_per_window=2,
            max_bytes_per_window=byte_budget,
        ),
    )
    selected_bytes = sum(
        len(record.model_dump_json().encode("utf-8")) + 1 for record in selected
    )

    # Then: greedy selection never crosses the hard per-window cap.
    assert tuple(record.memory_id for record in selected) == ("high",)
    assert selected_bytes <= byte_budget
    assert selected_bytes + candidate_bytes[1] > byte_budget


def test_same_timestamp_selection_prefers_score_per_serialized_byte() -> None:
    # Given: compact candidate has lower score but greater value per serialized byte.
    compact = _scored_candidate("compact", score=0.6)
    bloated = _scored_candidate(
        "bloated",
        score=0.9,
        frame_refs=("x" * 1_000,),
    )
    assert 0.6 / _selected_jsonl_bytes(compact) > (0.9 / _selected_jsonl_bytes(bloated))

    # When: only one simultaneous write can be retained.
    selected = select_spatial_tokens(
        (bloated, compact),
        selector=_FeatureScoreSelector(),
        options=SpatialCompressionOptions(max_tokens_per_window=1),
    )

    # Then: greedy admission maximizes selector value per actual byte.
    assert tuple(record.memory_id for record in selected) == ("compact",)


def test_selector_rejects_non_positive_byte_budget() -> None:
    with pytest.raises(SpatialCompressionError, match="max_bytes_per_window"):
        _ = select_spatial_tokens(
            (),
            selector=_FeatureScoreSelector(),
            options=SpatialCompressionOptions(max_bytes_per_window=0),
        )


def test_window_selection_is_prefix_causal() -> None:
    # Given: an early keep-worthy token and a better future token in one window.
    candidates = tuple(
        SpatialTokenCandidate(
            record=SpatialTokenRecord(
                memory_id=memory_id,
                video_id="causal-window",
                start_time=end_time - 0.5,
                end_time=end_time,
                token='["Z",25,"zone",0,0,0]',  # noqa: S106
                importance=0.0,
            ),
            features={"score": score},
        )
        for memory_id, end_time, score in (
            ("early", 1.0, 0.6),
            ("future", 20.0, 1.0),
        )
    )

    # When: one token can be admitted in the 30-second window.
    selected = select_spatial_tokens(
        candidates,
        selector=_FeatureScoreSelector(),
        options=SpatialCompressionOptions(max_tokens_per_window=1),
    )

    # Then: future importance cannot retroactively evict past memory state.
    assert tuple(record.memory_id for record in selected) == ("early",)


def test_dropped_state_is_reconsidered_on_later_observation() -> None:
    # Given: the first observation of one unchanged state misses the write gate.
    candidates = tuple(
        SpatialTokenCandidate(
            record=SpatialTokenRecord(
                memory_id=memory_id,
                video_id="gate-retry",
                start_time=end_time - 0.5,
                end_time=end_time,
                token='["O",25,"mug","zone",0,0,0,90,"o"]',  # noqa: S106
                importance=0.0,
            ),
            features={"score": score},
            state_key="object:mug",
            state_signature="same-state",
            state_position=(0.0, 0.0, 0.0),
            dedup_radius_m=0.5,
        )
        for memory_id, end_time, score in (
            ("dropped", 1.0, 0.4),
            ("retried", 2.0, 0.8),
        )
    )

    # When: selection updates retained state only after a token is admitted.
    selected = select_spatial_tokens(
        candidates,
        selector=_FeatureScoreSelector(),
        options=SpatialCompressionOptions(max_tokens_per_window=1),
    )

    # Then: candidate generation alone cannot suppress the later eligible write.
    assert tuple(record.memory_id for record in selected) == ("retried",)


def test_decode_spatial_token_rejects_malformed_payload() -> None:
    # Given: a persisted token with an invalid field shape.
    record = SpatialTokenRecord(
        memory_id="bad",
        video_id="video",
        start_time=0.0,
        end_time=0.0,
        token='["O",25]',  # noqa: S106
        importance=1.0,
    )

    # When / Then: artifact parsing fails with a typed error.
    with pytest.raises(SpatialCompressionError, match="expected 9 or 15 token fields"):
        _ = decode_spatial_token(record)


def test_custom_selector_path_is_loaded_from_environment(tmp_path: Path) -> None:
    # Given: a selector that strongly prefers zone tokens.
    selector = tmp_path / "selector.json"
    _ = selector.write_text(
        (
            '{"version":"linear-v1","feature_names":'
            '["kind_object","kind_relation","kind_zone","confidence",'
            '"geometry_reliability","frame_grounded","metric_relation","recency"],'
            '"weights":[0,0,10,0,0,0,0,0],"bias":-5}'
        ),
        encoding="utf-8",
    )
    source = _repeated_source()

    # When: the device budget keeps one token.
    result = build_compressed_spatial_memory(
        (source,),
        _clips(source),
        env={
            "WORLDMM_SPATIAL_SELECTOR_PATH": str(selector),
            "WORLDMM_SPATIAL_TOKEN_BUDGET": "1",
        },
    )
    (token,) = tuple(
        record for record in result.records if isinstance(record, SpatialTokenRecord)
    )

    # Then: persisted selection follows model weights, not hard-coded token kind.
    assert ":zone:" in token.memory_id
    assert result.experiment.selector_path == str(selector)


class _Cut3RStubEncoder:
    name: str = "cut3r-test"

    def __init__(self, _options: Mapping[str, ExperimentOption]) -> None:
        pass

    def encode(self, source: SourceStreamExample) -> SpatialGeometryFeatureSet:
        baseline = StructuredSpatialGeometryEncoder().encode(source)
        zone_ids = {zone.zone_id: f"cut3r_{zone.zone_id}" for zone in baseline.zones}
        return SpatialGeometryFeatureSet(
            encoder=self.name,
            projection_head=None,
            zones=tuple(
                zone.model_copy(update={"zone_id": zone_ids[zone.zone_id]})
                for zone in baseline.zones
            ),
            anchors=tuple(
                anchor.model_copy(update={"zone_id": zone_ids[anchor.zone_id]})
                for anchor in baseline.anchors
            ),
            relations=tuple(
                relation.model_copy(update={"zone_id": zone_ids[relation.zone_id]})
                for relation in baseline.relations
            ),
            extra_features={
                anchor.memory_id: {"cut3r_confidence": 0.9}
                for anchor in baseline.anchors
            },
            latent_state=(0.9,),
        )


class _LinearProjectionStub:
    name: str = "linear-test"

    def __init__(self, _options: Mapping[str, ExperimentOption]) -> None:
        pass

    def project(
        self,
        features: SpatialGeometryFeatureSet,
    ) -> SpatialGeometryFeatureSet:
        assert features.latent_state == (0.9,)
        return SpatialGeometryFeatureSet(
            encoder=features.encoder,
            projection_head=self.name,
            zones=features.zones,
            anchors=features.anchors,
            relations=features.relations,
            extra_features={
                key: {"latent_0": values.get("cut3r_confidence", 0.0) * 2.0}
                for key, values in features.extra_features.items()
            },
            latent_state=(1.8,),
        )


class _GraphTokenDecoderStub:
    name: str = "graph-test"

    def __init__(self, _options: Mapping[str, ExperimentOption]) -> None:
        pass

    def decode_candidates(
        self,
        source: SourceStreamExample,
        geometry: SpatialGeometryFeatureSet,
        codec: SpatialMemoryCodec,
        options: SpatialCompressionOptions,
    ) -> tuple[SpatialTokenCandidate, ...]:
        latent_state = cast("tuple[float]", geometry.latent_state)
        (latent_0,) = latent_state
        assert latent_0 == 1.8
        baseline = DeltaTopKSpatialTokenDecoder().decode_candidates(
            source,
            geometry,
            codec,
            options,
        )
        return tuple(
            SpatialTokenCandidate(
                record=candidate.record.model_copy(
                    update={"token_decoder": self.name},
                ),
                features={
                    **candidate.features,
                    "decoder_bias": 1.0,
                    "decoder_latent_0": latent_0,
                },
                state_key=candidate.state_key,
                state_signature=candidate.state_signature,
                state_position=candidate.state_position,
                dedup_radius_m=candidate.dedup_radius_m,
            )
            for candidate in baseline
        )


def test_encoder_projection_and_token_decoder_are_independently_swappable() -> None:
    # Given: CUT3R-like features, a learned projection boundary, and another decoder.
    register_spatial_geometry_encoder("cut3r-test", _Cut3RStubEncoder)
    register_spatial_projection_head("linear-test", _LinearProjectionStub)
    register_spatial_token_decoder("graph-test", _GraphTokenDecoderStub)
    model = resolve_spatial_memory_model(
        {},
        config=SpatialExperimentConfig(
            name="cut3r-graph-test",
            encoder="cut3r-test",
            projection_head="linear-test",
            token_decoder="graph-test",  # noqa: S106
        ),
    )
    source = _repeated_source()

    # When: projected features are decoded into spatial-memory candidates.
    built = build_spatial_token_candidates((source,), experiment=model)
    object_candidate = next(
        candidate
        for candidate in built.candidates
        if ":object:" in candidate.record.memory_id
    )
    zone_token = next(
        token
        for candidate in built.candidates
        if isinstance((token := decode_spatial_token(candidate.record)), ZoneToken)
    )

    # Then: model provenance and projected/decoder features survive independently.
    assert object_candidate.record.encoder == "cut3r-test"
    assert object_candidate.record.projection_head == "linear-test"
    assert object_candidate.record.token_decoder == "graph-test"  # noqa: S105
    assert object_candidate.features["latent_0"] == pytest.approx(1.8)
    assert object_candidate.features["decoder_bias"] == 1.0
    assert object_candidate.features["decoder_latent_0"] == pytest.approx(1.8)
    assert zone_token.zone_id.startswith("cut3r_zone_")


class _ZoneOnlySelector:
    def score(self, features: Mapping[str, float]) -> float:
        return features.get("kind_zone", 0.0)


def _zone_selector_factory(
    options: Mapping[str, ExperimentOption],
    path: Path | None,
) -> _ZoneOnlySelector:
    assert options == {"mode": "zone-only"}
    assert path is None
    return _ZoneOnlySelector()


def test_selector_is_swappable_through_experiment_config() -> None:
    # Given: a registered non-linear-head selector chosen only by experiment config.
    register_spatial_token_selector("zone-only-test", _zone_selector_factory)
    model = resolve_spatial_memory_model(
        {},
        config=SpatialExperimentConfig(
            name="selector-swap-test",
            selector="zone-only-test",
            selector_options={"mode": "zone-only"},
            token_budget=1,
            window_seconds=15.0,
        ),
    )

    # When: compression applies the configured selector and window policy.
    result = build_compressed_spatial_memory(
        (_repeated_source(),),
        _clips(_repeated_source()),
        env={},
        experiment=model,
    )
    tokens = tuple(
        record for record in result.records if isinstance(record, SpatialTokenRecord)
    )

    # Then: config controls selector implementation and the effective window size.
    assert tokens
    assert all(":zone:" in token.memory_id for token in tokens)
    assert model.options.window_seconds == 15.0


class _BadProvenanceProjection:
    name: str = "bad-provenance-test"

    def __init__(self, _options: Mapping[str, ExperimentOption]) -> None:
        pass

    def project(
        self,
        features: SpatialGeometryFeatureSet,
    ) -> SpatialGeometryFeatureSet:
        return SpatialGeometryFeatureSet(
            encoder=features.encoder,
            projection_head="wrong-projection",
            zones=features.zones,
            anchors=features.anchors,
            relations=features.relations,
            extra_features=features.extra_features,
        )


def test_plugin_provenance_mismatch_fails_before_artifact_write() -> None:
    # Given: a plugin that returns provenance different from its registry identity.
    register_spatial_projection_head(
        "bad-provenance-test",
        _BadProvenanceProjection,
    )
    model = resolve_spatial_memory_model(
        {},
        config=SpatialExperimentConfig(projection_head="bad-provenance-test"),
    )

    # When / Then: model output cannot silently claim another component.
    with pytest.raises(SpatialCompressionError, match="projection provenance"):
        _ = build_spatial_token_candidates((_repeated_source(),), experiment=model)


def test_decoder_options_change_object_delta_policy() -> None:
    # Given: one object moves less than the default 2x quantization threshold.
    source = SourceStreamExample(
        video_id="decoder-options",
        start_time=0.0,
        end_time=30.0,
        object_detections=(
            ObjectMetadata(
                label="mug",
                confidence=0.9,
                start_time=1.0,
                end_time=2.0,
                x=0.0,
                y=0.0,
                z=1.0,
            ),
            ObjectMetadata(
                label="mug",
                confidence=0.9,
                start_time=3.0,
                end_time=4.0,
                x=0.4,
                y=0.0,
                z=1.0,
            ),
        ),
    )
    default = resolve_spatial_memory_model({}, config=SpatialExperimentConfig())
    strict = resolve_spatial_memory_model(
        {},
        config=SpatialExperimentConfig(
            decoder_options={"object_delta_multiplier": 1.0},
        ),
    )

    # When: both policies apply admission-aware delta state to the same stream.
    default_records = build_compressed_spatial_memory(
        (source,),
        _clips(source),
        experiment=default,
    ).records
    strict_records = build_compressed_spatial_memory(
        (source,),
        _clips(source),
        experiment=strict,
    ).records

    # Then: an explicit movement event survives both dedup policies.
    assert sum(":object:mug:" in item.memory_id for item in default_records) == 2
    assert sum(":object:mug:" in item.memory_id for item in strict_records) == 2


def test_unchanged_object_refreshes_once_per_budget_window() -> None:
    source = SourceStreamExample(
        video_id="last-seen",
        start_time=0.0,
        end_time=120.0,
        object_detections=(
            ObjectMetadata(
                label="mug",
                confidence=0.9,
                start_time=1.0,
                end_time=2.0,
                x=1.0,
                y=2.0,
                z=1.0,
            ),
            ObjectMetadata(
                label="mug",
                confidence=0.9,
                start_time=100.0,
                end_time=101.0,
                x=1.0,
                y=2.0,
                z=1.0,
            ),
        ),
    )

    records = build_compressed_spatial_memory(
        (source,),
        _clips(source),
        env={},
        options=SpatialCompressionOptions(
            max_tokens_per_window=100,
            max_bytes_per_window=100_000,
            min_keep_score=0.0,
        ),
    ).records
    object_records = tuple(
        (record, token)
        for record in records
        if isinstance(record, SpatialTokenRecord)
        and isinstance((token := decode_spatial_token(record)), ObjectToken)
    )

    assert tuple(record.end_time for record, _token in object_records) == (2.0, 101.0)
    assert len({token.instance_id for _record, token in object_records}) == 1
    assert tuple(token.change_type for _record, token in object_records) == (
        "appeared",
        "observed",
    )


def test_zone_candidate_uses_first_causal_pose_not_future_revisit() -> None:
    source = SourceStreamExample(
        video_id="zone-prefix",
        start_time=0.0,
        end_time=120.0,
        pose_samples=(
            PoseSample(timestamp=1.0, x=0.0, y=0.0, z=1.5),
            PoseSample(timestamp=100.0, x=1.0, y=0.0, z=1.5),
        ),
    )

    candidates = build_spatial_token_candidates((source,), env={}).candidates
    zone_record, zone = next(
        (candidate.record, token)
        for candidate in candidates
        if isinstance((token := decode_spatial_token(candidate.record)), ZoneToken)
    )

    assert zone_record.end_time == 1.0
    assert zone.x == 0.0
    selected = build_compressed_spatial_memory(
        (source,),
        _clips(source),
        env={},
    ).records
    assert any(
        isinstance(record, SpatialTokenRecord)
        and isinstance(decode_spatial_token(record), ZoneToken)
        for record in selected
    )


def test_final_artifact_byte_cap_includes_trajectory_records() -> None:
    source = SourceStreamExample(
        video_id="trajectory-budget",
        start_time=0.0,
        end_time=60.0,
        pose_samples=(
            PoseSample(timestamp=1.0, x=0.0, y=0.0, z=1.5),
            PoseSample(timestamp=59.0, x=0.0, y=0.0, z=1.5),
        ),
    )

    result = build_compressed_spatial_memory(
        (source,),
        _clips(source),
        env={},
        options=SpatialCompressionOptions(max_bytes_per_window=1),
    )

    assert result.records == ()
    assert result.compressed_bytes == 0


def test_builtin_components_reject_unknown_options() -> None:
    # Given / When / Then: config typos cannot silently produce the baseline model.
    with pytest.raises(SpatialCompressionError, match="unknown option typo"):
        _ = resolve_spatial_memory_model(
            {},
            config=SpatialExperimentConfig(encoder_options={"typo": True}),
        )


def test_unknown_spatial_model_component_fails_before_memory_build() -> None:
    # Given / When / Then: experiment typos fail at the configuration boundary.
    with pytest.raises(SpatialCompressionError, match="unknown geometry encoder"):
        _ = resolve_spatial_memory_model(
            {},
            config=SpatialExperimentConfig(encoder="missing-encoder"),
        )


def test_effective_config_resolution_does_not_construct_plugins(tmp_path: Path) -> None:
    # Given: a config whose plugin and component cannot load on a CPU head node.
    selector_path = tmp_path / "selector.json"
    config = SpatialExperimentConfig(
        encoder="gpu-only-test",
        plugins=("module_that_must_not_be_imported",),
    )

    # When: only the effective run manifest config is resolved.
    resolved = resolve_spatial_experiment_config(
        {"WORLDMM_SPATIAL_SELECTOR_PATH": str(selector_path)},
        config=config,
    )

    # Then: environment overrides resolve without importing or constructing plugins.
    assert resolved.encoder == "gpu-only-test"
    assert resolved.plugins == ("module_that_must_not_be_imported",)
    assert resolved.selector_path == str(selector_path)


def test_byte_budget_propagates_from_environment_and_experiment_config() -> None:
    # Given / When: byte budgets come from either runtime env or explicit config.
    env = {"WORLDMM_SPATIAL_BYTE_BUDGET": "1024"}
    env_config = resolve_spatial_experiment_config(env)
    env_model = resolve_spatial_memory_model(env)
    explicit_model = resolve_spatial_memory_model(
        {},
        config=SpatialExperimentConfig(byte_budget=512),
    )

    # Then: effective config and runtime admission options carry the same budget.
    assert env_config.byte_budget == 1024
    assert env_model.config.byte_budget == 1024
    assert env_model.options.max_bytes_per_window == 1024
    assert explicit_model.config.byte_budget == 512
    assert explicit_model.options.max_bytes_per_window == 512
