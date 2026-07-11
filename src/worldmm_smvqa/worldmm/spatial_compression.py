from __future__ import annotations

import importlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, Self, override, runtime_checkable

from pydantic import Field, JsonValue, TypeAdapter, ValidationError, model_validator

from worldmm_smvqa.schema import FrozenModel, SourceStreamExample, StreamChunk
from worldmm_smvqa.worldmm.spatial import (
    build_object_anchors,
    build_object_state_snapshots,
    build_trajectory_summaries,
    build_zones,
    derive_relations,
)
from worldmm_smvqa.worldmm.spatial_types import (
    SpatialAnchorRecord,
    SpatialChangeType,
    SpatialCoordinateFrame,
    SpatialProvenance,
    SpatialRelationKind,
    SpatialRelationRecord,
    SpatialTokenRecord,
    WearerTrajectorySummaryRecord,
    ZoneRecord,
)

type SpatialCompressedRecord = SpatialTokenRecord | WearerTrajectorySummaryRecord
type RawStaticRecord = ZoneRecord | SpatialAnchorRecord | SpatialRelationRecord
type SpatialTokenKind = Literal["object", "relation", "zone"]
type DecodedSpatialToken = ObjectToken | RelationToken | ZoneToken
type ExperimentOption = JsonValue

FEATURE_NAMES: Final = (
    "kind_object",
    "kind_relation",
    "kind_zone",
    "confidence",
    "geometry_reliability",
    "frame_grounded",
    "metric_relation",
    "recency",
)
DEFAULT_QUANTIZATION_M: Final = 0.25
DEFAULT_TOKEN_BUDGET: Final = 16
DEFAULT_BYTE_BUDGET: Final = 4096
DEFAULT_WINDOW_SECONDS: Final = 30.0
DEFAULT_MIN_KEEP_SCORE: Final = 0.5
CONFIDENCE_SCALE: Final = 100
STATE_CONFIDENCE_SCALE: Final = 10
LEGACY_OBJECT_TOKEN_FIELDS: Final = 9
OBJECT_TOKEN_FIELDS: Final = 15
LEGACY_RELATION_TOKEN_FIELDS: Final = 7
RELATION_TOKEN_FIELDS: Final = 12
SPATIAL_SELECTOR_PATH_ENV: Final = "WORLDMM_SPATIAL_SELECTOR_PATH"
SPATIAL_TOKEN_BUDGET_ENV: Final = "WORLDMM_SPATIAL_TOKEN_BUDGET"  # noqa: S105
SPATIAL_BYTE_BUDGET_ENV: Final = "WORLDMM_SPATIAL_BYTE_BUDGET"
SPATIAL_QUANTIZATION_ENV: Final = "WORLDMM_SPATIAL_QUANTIZATION_M"
SPATIAL_EXPERIMENT_ENV: Final = "WORLDMM_SPATIAL_EXPERIMENT_CONFIG"
STRUCTURED_ENCODER: Final = "structured-v1"
IDENTITY_PROJECTION: Final = "identity-v1"
COMPACT_JSON_DECODER: Final = "compact-json-v1"
DELTA_TOPK_DECODER: Final = "delta-topk-v1"
LINEAR_SELECTOR: Final = "linear-v1"
OBJECT_DELTA_MULTIPLIER_OPTION: Final = "object_delta_multiplier"

_PROVENANCE_CODE: Final[dict[SpatialProvenance, str]] = {
    "object_geometry": "o",
    "gaze": "g",
    "pose": "p",
    "slam_pose": "s",
}
_CODE_PROVENANCE: Final[dict[str, SpatialProvenance]] = {
    value: key for key, value in _PROVENANCE_CODE.items()
}
_PROVENANCE_RELIABILITY: Final[dict[SpatialProvenance, float]] = {
    "object_geometry": 1.0,
    "gaze": 0.8,
    "slam_pose": 0.6,
    "pose": 0.4,
}
_RELATION_CODE: Final[dict[SpatialRelationKind, str]] = {
    "near": "N",
    "left_of": "L",
    "in_front_of": "F",
    "above": "U",
    "right_of": "L",
    "behind": "F",
    "below": "U",
}
_CODE_RELATION: Final[dict[str, SpatialRelationKind]] = {
    "N": "near",
    "L": "left_of",
    "F": "in_front_of",
    "U": "above",
}
_INVERSE_RELATION: Final[dict[SpatialRelationKind, SpatialRelationKind]] = {
    "left_of": "right_of",
    "in_front_of": "behind",
    "above": "below",
    "near": "near",
    "right_of": "left_of",
    "behind": "in_front_of",
    "below": "above",
}
_CHANGE_CODE: Final[dict[SpatialChangeType, str]] = {
    "appeared": "A",
    "observed": "O",
    "moved": "M",
}
_CODE_CHANGE: Final[dict[str, SpatialChangeType]] = {
    value: key for key, value in _CHANGE_CODE.items()
}
_TOKEN_PAYLOAD_ADAPTER: Final = TypeAdapter(list[object])


@dataclass(frozen=True, slots=True)
class SpatialCompressionError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"SpatialCompressionError: {self.detail}"


@dataclass(frozen=True, slots=True)
class SpatialCompressionOptions:
    quantization_m: float = DEFAULT_QUANTIZATION_M
    max_tokens_per_window: int = DEFAULT_TOKEN_BUDGET
    max_bytes_per_window: int = DEFAULT_BYTE_BUDGET
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    min_keep_score: float = DEFAULT_MIN_KEEP_SCORE


class SpatialExperimentConfig(FrozenModel):
    name: str = "source-compact-v1"
    encoder: str = STRUCTURED_ENCODER
    projection_head: str = IDENTITY_PROJECTION
    token_decoder: str = DELTA_TOPK_DECODER
    codec: str = COMPACT_JSON_DECODER
    selector: str = LINEAR_SELECTOR
    selector_path: str | None = None
    token_budget: int = DEFAULT_TOKEN_BUDGET
    byte_budget: int = DEFAULT_BYTE_BUDGET
    quantization_m: float = DEFAULT_QUANTIZATION_M
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    min_keep_score: float = DEFAULT_MIN_KEEP_SCORE
    plugins: tuple[str, ...] = ()
    encoder_options: dict[str, ExperimentOption] = Field(default_factory=dict)
    projection_options: dict[str, ExperimentOption] = Field(default_factory=dict)
    decoder_options: dict[str, ExperimentOption] = Field(default_factory=dict)
    codec_options: dict[str, ExperimentOption] = Field(default_factory=dict)
    selector_options: dict[str, ExperimentOption] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SpatialGeometryFeatureSet:
    encoder: str
    projection_head: str | None
    zones: tuple[ZoneRecord, ...]
    anchors: tuple[SpatialAnchorRecord, ...]
    relations: tuple[SpatialRelationRecord, ...]
    extra_features: Mapping[str, Mapping[str, float]]
    # Transient provider state stays in-process; persisted records remain typed.
    latent_state: object | None = None


@runtime_checkable
class SpatialGeometryEncoder(Protocol):
    name: str

    def encode(self, source: SourceStreamExample) -> SpatialGeometryFeatureSet:
        """Build geometry records and optional selector features for one source."""
        ...


@runtime_checkable
class SpatialProjectionHead(Protocol):
    name: str

    def project(
        self,
        features: SpatialGeometryFeatureSet,
    ) -> SpatialGeometryFeatureSet:
        """Map backend-specific geometry features into the decoder feature space."""
        ...


class SpatialSelectorModel(FrozenModel):
    # ponytail: linear head exports as tiny JSON; replace through decoder/selector
    # experiment only when QA retention shows this capacity is insufficient.
    version: Literal["linear-v1"] = "linear-v1"
    feature_names: tuple[str, ...] = FEATURE_NAMES
    weights: tuple[float, ...]
    bias: float

    @model_validator(mode="after")
    def _require_matching_features(self) -> Self:
        if not self.feature_names or len(self.feature_names) != len(
            set(self.feature_names),
        ):
            msg = "feature_names must be non-empty and unique"
            raise ValueError(msg)
        if len(self.weights) != len(self.feature_names):
            msg = "weights must match feature_names"
            raise ValueError(msg)
        return self

    def score(self, features: Mapping[str, float]) -> float:
        """Return one bounded keep score for a spatial-token candidate."""
        logit = self.bias + sum(
            weight * features.get(name, 0.0)
            for name, weight in zip(
                self.feature_names,
                self.weights,
                strict=True,
            )
        )
        if logit >= 0.0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)


@runtime_checkable
class SpatialTokenSelector(Protocol):
    def score(self, features: Mapping[str, float]) -> float:
        """Return one bounded keep score for a spatial-token candidate."""
        ...


DEFAULT_SELECTOR: Final = SpatialSelectorModel(
    weights=(1.6, 1.8, 0.3, 1.0, 0.8, 0.2, 0.4, 0.2),
    bias=-1.5,
)


@dataclass(frozen=True, slots=True)
class SpatialTokenCandidate:
    record: SpatialTokenRecord
    features: Mapping[str, float]
    state_key: str | None = None
    state_signature: str | None = None
    state_position: tuple[float, float, float] | None = None
    dedup_radius_m: float = 0.0


@dataclass(frozen=True, slots=True)
class SpatialCandidateBuild:
    candidates: tuple[SpatialTokenCandidate, ...]
    raw_records: tuple[RawStaticRecord, ...]


@dataclass(frozen=True, slots=True)
class SpatialCompressionResult:
    experiment: SpatialExperimentConfig
    records: tuple[SpatialCompressedRecord, ...]
    candidate_count: int
    raw_record_count: int
    raw_bytes: int
    compressed_bytes: int


class SpatialCompressionManifest(FrozenModel):
    experiment: SpatialExperimentConfig
    rank: int = Field(ge=0)
    world_size: int = Field(gt=0)
    source_count: int = Field(ge=0)
    record_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    trajectory_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    raw_record_count: int = Field(ge=0)
    raw_bytes: int = Field(ge=0)
    compressed_bytes: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class ZoneToken:
    scale_m: float
    zone_id: str
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class ObjectToken:
    scale_m: float
    object_label: str
    zone_id: str
    x: float
    y: float
    z: float
    confidence: float
    provenance: SpatialProvenance
    instance_id: str | None = None
    coordinate_frame: SpatialCoordinateFrame = "source_world"
    uncertainty_m: float | None = None
    change_type: SpatialChangeType = "observed"
    valid_from: float | None = None
    valid_to: float | None = None


@dataclass(frozen=True, slots=True)
class RelationToken:
    scale_m: float
    subject: str
    relation: SpatialRelationKind
    object: str
    zone_id: str
    distance_m: float | None
    subject_instance_id: str | None = None
    object_instance_id: str | None = None
    coordinate_frame: SpatialCoordinateFrame = "source_world"
    valid_from: float | None = None
    valid_to: float | None = None


@runtime_checkable
class SpatialMemoryCodec(Protocol):
    name: str

    def encode(self, token: DecodedSpatialToken) -> str:
        """Encode one semantic spatial token into a persisted representation."""
        ...

    def decode(self, record: SpatialTokenRecord) -> DecodedSpatialToken:
        """Decode one persisted token for retrieval and diagnostics."""
        ...


class StructuredSpatialGeometryEncoder:
    name: str = STRUCTURED_ENCODER
    options: dict[str, ExperimentOption]

    def __init__(self, options: Mapping[str, ExperimentOption] | None = None) -> None:
        """Keep encoder options for reproducible experiment manifests."""
        self.options = dict(options or {})
        _require_known_options(self.options, frozenset(), self.name)

    def encode(self, source: SourceStreamExample) -> SpatialGeometryFeatureSet:
        """Use source object geometry, gaze, and pose through existing builders."""
        zones = build_zones(source)
        anchors = build_object_anchors(source)
        # ponytail: existing pairwise relation builder; use a spatial index when
        # dense CUT3R anchors make relation construction the measured bottleneck.
        return SpatialGeometryFeatureSet(
            encoder=self.name,
            projection_head=None,
            zones=zones,
            anchors=anchors,
            relations=derive_relations(anchors),
            extra_features={},
        )


class IdentitySpatialProjectionHead:
    name: str = IDENTITY_PROJECTION
    options: dict[str, ExperimentOption]

    def __init__(self, options: Mapping[str, ExperimentOption] | None = None) -> None:
        """Keep projection options for reproducible experiment manifests."""
        self.options = dict(options or {})
        _require_known_options(self.options, frozenset(), self.name)

    def project(
        self,
        features: SpatialGeometryFeatureSet,
    ) -> SpatialGeometryFeatureSet:
        """Keep structured scalar features unchanged for the baseline model."""
        return SpatialGeometryFeatureSet(
            encoder=features.encoder,
            projection_head=self.name,
            zones=features.zones,
            anchors=features.anchors,
            relations=features.relations,
            extra_features=features.extra_features,
            latent_state=features.latent_state,
        )


class CompactJsonSpatialMemoryCodec:
    name: str = COMPACT_JSON_DECODER
    options: dict[str, ExperimentOption]

    def __init__(self, options: Mapping[str, ExperimentOption] | None = None) -> None:
        """Keep codec options for reproducible experiment manifests."""
        self.options = dict(options or {})
        _require_known_options(self.options, frozenset(), self.name)

    def encode(self, token: DecodedSpatialToken) -> str:
        """Encode compact integer geometry in a versioned JSON array."""
        # ponytail: JSON stays inspectable for benchmark artifacts; add a binary
        # codec plugin only after on-device storage/latency profiling requires it.
        scale_cm = _scale_cm(token.scale_m)
        match token:
            case ZoneToken():
                return _token_json(
                    "Z",
                    scale_cm,
                    token.zone_id,
                    _quantize(token.x, token.scale_m),
                    _quantize(token.y, token.scale_m),
                    _quantize(token.z, token.scale_m),
                )
            case ObjectToken():
                return _token_json(
                    "O",
                    scale_cm,
                    token.object_label,
                    token.zone_id,
                    _quantize(token.x, token.scale_m),
                    _quantize(token.y, token.scale_m),
                    _quantize(token.z, token.scale_m),
                    round(token.confidence * CONFIDENCE_SCALE),
                    _PROVENANCE_CODE[token.provenance],
                    token.instance_id or "",
                    token.coordinate_frame,
                    _CHANGE_CODE[token.change_type],
                    _time_millis(token.valid_from),
                    _time_millis(token.valid_to),
                    (
                        -1
                        if token.uncertainty_m is None
                        else _quantize_uncertainty(
                            token.uncertainty_m,
                            token.scale_m,
                        )
                    ),
                )
            case RelationToken():
                (
                    subject,
                    relation,
                    object_label,
                    subject_instance_id,
                    object_instance_id,
                ) = _canonical_relation_values(
                    token.subject,
                    token.relation,
                    token.object,
                    token.subject_instance_id,
                    token.object_instance_id,
                )
                return _token_json(
                    "R",
                    scale_cm,
                    subject,
                    _RELATION_CODE[relation],
                    object_label,
                    token.zone_id,
                    (
                        -1
                        if token.distance_m is None
                        else _quantize(token.distance_m, token.scale_m)
                    ),
                    subject_instance_id or "",
                    object_instance_id or "",
                    token.coordinate_frame,
                    _time_millis(token.valid_from),
                    _time_millis(token.valid_to),
                )

    def decode(self, record: SpatialTokenRecord) -> DecodedSpatialToken:
        """Decode the built-in compact JSON token representation."""
        return _decode_compact_json(record)


@runtime_checkable
class SpatialTokenDecoder(Protocol):
    name: str

    def decode_candidates(
        self,
        source: SourceStreamExample,
        geometry: SpatialGeometryFeatureSet,
        codec: SpatialMemoryCodec,
        options: SpatialCompressionOptions,
    ) -> tuple[SpatialTokenCandidate, ...]:
        """Decode projected geometry into candidates before budget selection."""
        ...


class DeltaTopKSpatialTokenDecoder:
    name: str = DELTA_TOPK_DECODER
    options: dict[str, ExperimentOption]
    object_delta_multiplier: float

    def __init__(self, options: Mapping[str, ExperimentOption] | None = None) -> None:
        """Keep decoder options for reproducible experiment manifests."""
        self.options = dict(options or {})
        _require_known_options(
            self.options,
            frozenset({OBJECT_DELTA_MULTIPLIER_OPTION}),
            self.name,
        )
        self.object_delta_multiplier = _option_float(
            self.options,
            OBJECT_DELTA_MULTIPLIER_OPTION,
            2.0,
        )
        if (
            not math.isfinite(self.object_delta_multiplier)
            or self.object_delta_multiplier <= 0.0
        ):
            raise SpatialCompressionError(
                detail=f"{self.name}.{OBJECT_DELTA_MULTIPLIER_OPTION} must be positive",
            )

    def decode_candidates(
        self,
        source: SourceStreamExample,
        geometry: SpatialGeometryFeatureSet,
        codec: SpatialMemoryCodec,
        options: SpatialCompressionOptions,
    ) -> tuple[SpatialTokenCandidate, ...]:
        """Generate causal delta tokens from zones, anchors, and relations."""
        return (
            *_zone_candidates(
                source,
                geometry.zones,
                options,
                encoder=geometry.encoder,
                projection_head=geometry.projection_head or IDENTITY_PROJECTION,
                token_decoder=self.name,
                codec=codec,
            ),
            *_object_candidates(
                source,
                geometry.anchors,
                options,
                encoder=geometry.encoder,
                projection_head=geometry.projection_head or IDENTITY_PROJECTION,
                token_decoder=self.name,
                codec=codec,
                extra_features=geometry.extra_features,
                object_delta_multiplier=self.object_delta_multiplier,
            ),
            *_relation_candidates(
                source,
                geometry.anchors,
                geometry.relations,
                options,
                encoder=geometry.encoder,
                projection_head=geometry.projection_head or IDENTITY_PROJECTION,
                token_decoder=self.name,
                codec=codec,
                extra_features=geometry.extra_features,
            ),
        )


@dataclass(frozen=True, slots=True)
class SpatialMemoryModel:
    config: SpatialExperimentConfig
    options: SpatialCompressionOptions
    encoder: SpatialGeometryEncoder
    projection_head: SpatialProjectionHead
    token_decoder: SpatialTokenDecoder
    codec: SpatialMemoryCodec
    selector: SpatialTokenSelector


type GeometryEncoderFactory = Callable[
    [Mapping[str, ExperimentOption]],
    SpatialGeometryEncoder,
]
type ProjectionHeadFactory = Callable[
    [Mapping[str, ExperimentOption]],
    SpatialProjectionHead,
]
type TokenDecoderFactory = Callable[
    [Mapping[str, ExperimentOption]],
    SpatialTokenDecoder,
]
type CodecFactory = Callable[
    [Mapping[str, ExperimentOption]],
    SpatialMemoryCodec,
]
type SelectorFactory = Callable[
    [Mapping[str, ExperimentOption], Path | None],
    SpatialTokenSelector,
]


def _linear_selector_factory(
    options: Mapping[str, ExperimentOption],
    path: Path | None,
) -> SpatialTokenSelector:
    _require_known_options(options, frozenset(), LINEAR_SELECTOR)
    return load_spatial_selector(path)


_GEOMETRY_ENCODER_FACTORIES: dict[str, GeometryEncoderFactory] = {
    STRUCTURED_ENCODER: StructuredSpatialGeometryEncoder,
}
_PROJECTION_HEAD_FACTORIES: dict[str, ProjectionHeadFactory] = {
    IDENTITY_PROJECTION: IdentitySpatialProjectionHead,
}
_TOKEN_DECODER_FACTORIES: dict[str, TokenDecoderFactory] = {
    DELTA_TOPK_DECODER: DeltaTopKSpatialTokenDecoder,
}
_CODEC_FACTORIES: dict[str, CodecFactory] = {
    COMPACT_JSON_DECODER: CompactJsonSpatialMemoryCodec,
}
_SELECTOR_FACTORIES: dict[str, SelectorFactory] = {
    LINEAR_SELECTOR: _linear_selector_factory,
}
_CODEC_DECODE_OPTIONS: dict[str, Mapping[str, ExperimentOption]] = {}


def register_spatial_geometry_encoder(
    name: str,
    factory: GeometryEncoderFactory,
) -> None:
    if not name:
        raise SpatialCompressionError(detail="geometry encoder name is empty")
    _GEOMETRY_ENCODER_FACTORIES[name] = factory


def register_spatial_projection_head(
    name: str,
    factory: ProjectionHeadFactory,
) -> None:
    if not name:
        raise SpatialCompressionError(detail="projection head name is empty")
    _PROJECTION_HEAD_FACTORIES[name] = factory


def register_spatial_token_decoder(
    name: str,
    factory: TokenDecoderFactory,
) -> None:
    if not name:
        raise SpatialCompressionError(detail="token decoder name is empty")
    _TOKEN_DECODER_FACTORIES[name] = factory


def register_spatial_memory_codec(
    name: str,
    factory: CodecFactory,
) -> None:
    if not name:
        raise SpatialCompressionError(detail="codec name is empty")
    _CODEC_FACTORIES[name] = factory


def register_spatial_token_selector(
    name: str,
    factory: SelectorFactory,
) -> None:
    if not name:
        raise SpatialCompressionError(detail="selector name is empty")
    _SELECTOR_FACTORIES[name] = factory


def load_spatial_experiment_config(path: Path) -> SpatialExperimentConfig:
    try:
        return SpatialExperimentConfig.model_validate_json(
            path.read_text(encoding="utf-8"),
        )
    except (OSError, ValidationError) as exc:
        raise SpatialCompressionError(
            detail=f"invalid spatial experiment {path}: {exc}",
        ) from exc


def load_spatial_plugins(config: SpatialExperimentConfig) -> None:
    for module_name in config.plugins:
        try:
            _ = importlib.import_module(module_name)
        except ImportError as exc:
            raise SpatialCompressionError(
                detail=f"failed to import spatial plugin {module_name}: {exc}",
            ) from exc
    _CODEC_DECODE_OPTIONS[config.codec] = dict(config.codec_options)


def resolve_spatial_memory_model(
    env: Mapping[str, str],
    *,
    config: SpatialExperimentConfig | None = None,
) -> SpatialMemoryModel:
    selected_config = resolve_spatial_experiment_config(env, config=config)
    load_spatial_plugins(selected_config)
    encoder_factory = _component_factory(
        _GEOMETRY_ENCODER_FACTORIES,
        selected_config.encoder,
        "geometry encoder",
    )
    projection_factory = _component_factory(
        _PROJECTION_HEAD_FACTORIES,
        selected_config.projection_head,
        "projection head",
    )
    decoder_factory = _component_factory(
        _TOKEN_DECODER_FACTORIES,
        selected_config.token_decoder,
        "token decoder",
    )
    codec_factory = _component_factory(
        _CODEC_FACTORIES,
        selected_config.codec,
        "codec",
    )
    options = SpatialCompressionOptions(
        quantization_m=selected_config.quantization_m,
        max_tokens_per_window=selected_config.token_budget,
        max_bytes_per_window=selected_config.byte_budget,
        window_seconds=selected_config.window_seconds,
        min_keep_score=selected_config.min_keep_score,
    )
    selector_path = (
        Path(selected_config.selector_path) if selected_config.selector_path else None
    )
    selector_factory = _SELECTOR_FACTORIES.get(selected_config.selector)
    if selector_factory is None:
        raise SpatialCompressionError(
            detail=f"unknown selector: {selected_config.selector}",
        )
    selector = selector_factory(selected_config.selector_options, selector_path)
    codec = codec_factory(selected_config.codec_options)
    token_decoder = decoder_factory(selected_config.decoder_options)
    projection_head = projection_factory(selected_config.projection_options)
    encoder = encoder_factory(selected_config.encoder_options)
    return SpatialMemoryModel(
        config=selected_config,
        options=options,
        encoder=encoder,
        projection_head=projection_head,
        token_decoder=token_decoder,
        codec=codec,
        selector=selector,
    )


def resolve_spatial_experiment_config(
    env: Mapping[str, str],
    *,
    config: SpatialExperimentConfig | None = None,
) -> SpatialExperimentConfig:
    """Resolve effective config without importing or constructing model plugins."""
    selected_config = config or _experiment_config_from_env(env)
    selector_path = (
        Path(selected_config.selector_path)
        if selected_config.selector_path
        else _selector_path(env)
    )
    _validate_options(
        SpatialCompressionOptions(
            quantization_m=selected_config.quantization_m,
            max_tokens_per_window=selected_config.token_budget,
            max_bytes_per_window=selected_config.byte_budget,
            window_seconds=selected_config.window_seconds,
            min_keep_score=selected_config.min_keep_score,
        ),
    )
    return selected_config.model_copy(
        update={
            "selector_path": str(selector_path) if selector_path is not None else None,
        },
    )


def load_spatial_selector(path: Path | None) -> SpatialSelectorModel:
    if path is None:
        return DEFAULT_SELECTOR
    try:
        return SpatialSelectorModel.model_validate_json(
            path.read_text(encoding="utf-8"),
        )
    except (OSError, ValidationError) as exc:
        raise SpatialCompressionError(
            detail=f"invalid selector model {path}: {exc}",
        ) from exc


def build_compressed_spatial_memory(  # noqa: PLR0913
    sources: Sequence[SourceStreamExample],
    clip_chunks: Sequence[StreamChunk],
    *,
    env: Mapping[str, str] | None = None,
    experiment: SpatialMemoryModel | None = None,
    options: SpatialCompressionOptions | None = None,
    measure_legacy: bool = False,
) -> SpatialCompressionResult:
    runtime_env = os.environ if env is None else env
    selected_experiment = experiment or resolve_spatial_memory_model(runtime_env)
    selected_options = options or selected_experiment.options
    built = build_spatial_token_candidates(
        sources,
        experiment=selected_experiment,
        options=selected_options,
    )
    static_records = select_spatial_tokens(
        built.candidates,
        selector=selected_experiment.selector,
        options=selected_options,
    )
    zones = tuple(
        record for record in built.raw_records if isinstance(record, ZoneRecord)
    )
    trajectory_chunks = tuple(
        chunk for chunk in clip_chunks if _has_zone_overlap(chunk, zones)
    )
    trajectories = _select_trajectory_summaries(
        build_trajectory_summaries(trajectory_chunks, zones),
        static_records,
        selected_options,
    )
    compressed_records = (*static_records, *trajectories)
    if measure_legacy:
        raw_anchors = tuple(
            record
            for record in built.raw_records
            if isinstance(record, SpatialAnchorRecord)
        )
        raw_records: Sequence[FrozenModel] = (
            *built.raw_records,
            *build_object_state_snapshots(clip_chunks, raw_anchors),
            *trajectories,
        )
    else:
        raw_records = built.raw_records
    return SpatialCompressionResult(
        experiment=selected_experiment.config.model_copy(
            update={
                "token_budget": selected_options.max_tokens_per_window,
                "byte_budget": selected_options.max_bytes_per_window,
                "quantization_m": selected_options.quantization_m,
                "window_seconds": selected_options.window_seconds,
                "min_keep_score": selected_options.min_keep_score,
            },
        ),
        records=compressed_records,
        candidate_count=len(built.candidates),
        raw_record_count=len(raw_records),
        raw_bytes=_jsonl_bytes(raw_records),
        compressed_bytes=_jsonl_bytes(compressed_records),
    )


def _select_trajectory_summaries(
    trajectories: Sequence[WearerTrajectorySummaryRecord],
    static_records: Sequence[SpatialTokenRecord],
    options: SpatialCompressionOptions,
) -> tuple[WearerTrajectorySummaryRecord, ...]:
    counts: dict[tuple[str, int], int] = {}
    byte_counts: dict[tuple[str, int], int] = {}
    for record in static_records:
        key = (
            record.video_id,
            math.floor(record.end_time / options.window_seconds),
        )
        counts[key] = counts.get(key, 0) + 1
        byte_counts[key] = byte_counts.get(key, 0) + _jsonl_bytes((record,))
    selected: list[WearerTrajectorySummaryRecord] = []
    for record in trajectories:
        key = (
            record.video_id,
            math.floor(record.end_time / options.window_seconds),
        )
        record_bytes = _jsonl_bytes((record,))
        if counts.get(key, 0) >= options.max_tokens_per_window:
            continue
        if byte_counts.get(key, 0) + record_bytes > options.max_bytes_per_window:
            continue
        selected.append(record)
        counts[key] = counts.get(key, 0) + 1
        byte_counts[key] = byte_counts.get(key, 0) + record_bytes
    return tuple(selected)


def build_spatial_token_candidates(
    sources: Sequence[SourceStreamExample],
    *,
    env: Mapping[str, str] | None = None,
    experiment: SpatialMemoryModel | None = None,
    options: SpatialCompressionOptions | None = None,
) -> SpatialCandidateBuild:
    runtime_env = os.environ if env is None else env
    selected_experiment = experiment or resolve_spatial_memory_model(runtime_env)
    selected_options = options or selected_experiment.options
    _validate_options(selected_options)
    candidates: list[SpatialTokenCandidate] = []
    raw_records: list[RawStaticRecord] = []
    for source in sources:
        encoded = selected_experiment.encoder.encode(source)
        if encoded.encoder != selected_experiment.encoder.name:
            raise SpatialCompressionError(
                detail=(
                    f"{source.video_id}: encoder provenance {encoded.encoder} does not "
                    f"match {selected_experiment.encoder.name}"
                ),
            )
        geometry = selected_experiment.projection_head.project(encoded)
        _validate_projected_geometry(
            source,
            geometry,
            encoder=selected_experiment.encoder.name,
            projection_head=selected_experiment.projection_head.name,
        )
        raw_records.extend(
            (*geometry.zones, *geometry.anchors, *geometry.relations),
        )
        candidates.extend(
            selected_experiment.token_decoder.decode_candidates(
                source,
                geometry,
                selected_experiment.codec,
                selected_options,
            ),
        )
    return SpatialCandidateBuild(
        candidates=tuple(candidates),
        raw_records=tuple(raw_records),
    )


def select_spatial_tokens(
    candidates: Sequence[SpatialTokenCandidate],
    *,
    selector: SpatialTokenSelector,
    options: SpatialCompressionOptions,
) -> tuple[SpatialTokenRecord, ...]:
    _validate_options(options)
    scored: list[tuple[SpatialTokenCandidate, SpatialTokenRecord, int]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.record.video_id,
            item.record.end_time,
            item.record.start_time,
            item.record.memory_id,
        ),
    ):
        if (
            not math.isfinite(candidate.dedup_radius_m)
            or candidate.dedup_radius_m < 0.0
        ):
            raise SpatialCompressionError(
                detail=f"{candidate.record.memory_id}: invalid dedup radius",
            )
        score = selector.score(candidate.features)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise SpatialCompressionError(
                detail=f"{candidate.record.memory_id}: selector score must be in [0,1]",
            )
        if score < options.min_keep_score:
            continue
        importance = round(score, 6)
        record = candidate.record.model_copy(update={"importance": importance})
        scored.append((candidate, record, _jsonl_bytes((record,))))
    selected: list[SpatialTokenRecord] = []
    retained: dict[tuple[str, str], SpatialTokenCandidate] = {}
    window_counts: dict[tuple[str, int], int] = {}
    window_bytes: dict[tuple[str, int], int] = {}
    for candidate, record, record_bytes in sorted(
        scored,
        key=lambda item: (
            item[1].video_id,
            item[1].end_time,
            # ponytail: causal admission allows score/byte ranking only among
            # simultaneously available records; add knapsack only if measured.
            -(item[1].importance / item[2]),
            -item[1].importance,
            item[1].memory_id,
        ),
    ):
        state_key = (
            (record.video_id, candidate.state_key)
            if candidate.state_key is not None
            else None
        )
        if state_key is not None:
            previous = retained.get(state_key)
            if _same_selected_state(previous, candidate) and not _freshness_due(
                previous,
                candidate,
                options.window_seconds,
            ):
                continue
        window = math.floor(record.end_time / options.window_seconds)
        window_key = (record.video_id, window)
        if window_counts.get(window_key, 0) >= options.max_tokens_per_window:
            continue
        if (
            window_bytes.get(window_key, 0) + record_bytes
            > options.max_bytes_per_window
        ):
            continue
        selected.append(record)
        window_counts[window_key] = window_counts.get(window_key, 0) + 1
        window_bytes[window_key] = window_bytes.get(window_key, 0) + record_bytes
        if state_key is not None:
            retained[state_key] = candidate
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                item.video_id,
                item.start_time,
                item.end_time,
                item.memory_id,
            ),
        ),
    )


def _same_selected_state(
    previous: SpatialTokenCandidate | None,
    current: SpatialTokenCandidate,
) -> bool:
    if (
        previous is None
        or current.state_signature is None
        or previous.state_signature != current.state_signature
    ):
        return False
    if previous.state_position is None or current.state_position is None:
        return True
    return math.dist(previous.state_position, current.state_position) <= (
        current.dedup_radius_m
    )


def _freshness_due(
    previous: SpatialTokenCandidate | None,
    current: SpatialTokenCandidate,
    window_seconds: float,
) -> bool:
    if (
        previous is None
        or current.state_key is None
        or not current.state_key.startswith("object:")
    ):
        return False
    # ponytail: one unchanged refresh per budget window bounds steady-state growth.
    return math.floor(previous.record.end_time / window_seconds) != math.floor(
        current.record.end_time / window_seconds,
    )


def decode_spatial_token(record: SpatialTokenRecord) -> DecodedSpatialToken:
    factory = _CODEC_FACTORIES.get(record.codec)
    if factory is None and os.environ.get(SPATIAL_EXPERIMENT_ENV):
        _ = resolve_spatial_memory_model(os.environ)
        factory = _CODEC_FACTORIES.get(record.codec)
    if factory is None:
        raise SpatialCompressionError(
            detail=f"{record.memory_id}: unknown spatial codec {record.codec}",
        )
    return factory(_CODEC_DECODE_OPTIONS.get(record.codec, {})).decode(record)


def _decode_compact_json(record: SpatialTokenRecord) -> DecodedSpatialToken:
    try:
        payload = _TOKEN_PAYLOAD_ADAPTER.validate_json(record.token)
    except ValidationError as exc:
        raise SpatialCompressionError(
            detail=f"{record.memory_id}: invalid token JSON",
        ) from exc
    if not payload:
        raise SpatialCompressionError(
            detail=f"{record.memory_id}: token must be a non-empty JSON array",
        )
    kind = _string(payload, 0, record)
    if kind == "Z":
        _require_length(payload, 6, record)
        scale_m = _scale_m(payload, record)
        return ZoneToken(
            scale_m=scale_m,
            zone_id=_string(payload, 2, record),
            x=_coordinate(payload, 3, scale_m, record),
            y=_coordinate(payload, 4, scale_m, record),
            z=_coordinate(payload, 5, scale_m, record),
        )
    if kind == "O":
        _require_length_in(
            payload,
            (LEGACY_OBJECT_TOKEN_FIELDS, OBJECT_TOKEN_FIELDS),
            record,
        )
        extended = len(payload) == OBJECT_TOKEN_FIELDS
        scale_m = _scale_m(payload, record)
        confidence_pct = _integer(payload, 7, record)
        provenance_code = _string(payload, 8, record)
        if not 0 <= confidence_pct <= CONFIDENCE_SCALE:
            raise SpatialCompressionError(
                detail=f"{record.memory_id}: confidence must be in [0,100]",
            )
        try:
            provenance = _CODE_PROVENANCE[provenance_code]
        except KeyError as exc:
            raise SpatialCompressionError(
                detail=f"{record.memory_id}: invalid provenance code",
            ) from exc
        instance_id = _string(payload, 9, record) or None if extended else None
        coordinate_frame = (
            _coordinate_frame(payload, 10, record) if extended else "source_world"
        )
        change_type = _change_type(payload, 11, record) if extended else "observed"
        return ObjectToken(
            scale_m=scale_m,
            object_label=_string(payload, 2, record),
            zone_id=_string(payload, 3, record),
            x=_coordinate(payload, 4, scale_m, record),
            y=_coordinate(payload, 5, scale_m, record),
            z=_coordinate(payload, 6, scale_m, record),
            confidence=confidence_pct / CONFIDENCE_SCALE,
            provenance=provenance,
            instance_id=instance_id,
            coordinate_frame=coordinate_frame,
            change_type=change_type,
            valid_from=(_optional_time(payload, 12, record) if extended else None),
            valid_to=_optional_time(payload, 13, record) if extended else None,
            uncertainty_m=(
                None
                if not extended or _integer(payload, 14, record) < 0
                else _integer(payload, 14, record) * scale_m
            ),
        )
    if kind == "R":
        _require_length_in(
            payload,
            (LEGACY_RELATION_TOKEN_FIELDS, RELATION_TOKEN_FIELDS),
            record,
        )
        extended = len(payload) == RELATION_TOKEN_FIELDS
        scale_m = _scale_m(payload, record)
        relation_code = _string(payload, 3, record)
        try:
            relation = _CODE_RELATION[relation_code]
        except KeyError as exc:
            raise SpatialCompressionError(
                detail=f"{record.memory_id}: invalid relation code",
            ) from exc
        distance_bin = _integer(payload, 6, record)
        return RelationToken(
            scale_m=scale_m,
            subject=_string(payload, 2, record),
            relation=relation,
            object=_string(payload, 4, record),
            zone_id=_string(payload, 5, record),
            distance_m=None if distance_bin < 0 else distance_bin * scale_m,
            subject_instance_id=(
                (_string(payload, 7, record) or None) if extended else None
            ),
            object_instance_id=(
                (_string(payload, 8, record) or None) if extended else None
            ),
            coordinate_frame=(
                _coordinate_frame(payload, 9, record) if extended else "source_world"
            ),
            valid_from=(_optional_time(payload, 10, record) if extended else None),
            valid_to=_optional_time(payload, 11, record) if extended else None,
        )
    raise SpatialCompressionError(
        detail=f"{record.memory_id}: unknown token kind {kind}",
    )


def spatial_token_snippet(record: SpatialTokenRecord) -> str:
    token = decode_spatial_token(record)
    match token:
        case ZoneToken():
            return (
                f"zone {token.zone_id} centered near "
                f"({_format_float(token.x)},{_format_float(token.y)},"
                f"{_format_float(token.z)}) compact_geometry"
            )
        case ObjectToken():
            return (
                f"{token.object_label} instance={token.instance_id or 'unknown'} "
                f"anchored in {token.zone_id} near "
                f"({_format_float(token.x)},{_format_float(token.y)},"
                f"{_format_float(token.z)}) provenance={token.provenance} "
                f"frame={token.coordinate_frame} change={token.change_type} "
                f"last_seen={_format_float(record.end_time)} "
                "compact_geometry"
            )
        case RelationToken():
            inverse = _INVERSE_RELATION[token.relation]
            distance = (
                ""
                if token.distance_m is None
                else f" distance_m={_format_float(token.distance_m)}"
            )
            return (
                f"{token.subject}[{token.subject_instance_id or 'unknown'}] "
                f"{token.relation} "
                f"{token.object}[{token.object_instance_id or 'unknown'}]; "
                f"{token.object} {inverse} {token.subject} in {token.zone_id}"
                f"{distance} frame={token.coordinate_frame} compact_geometry"
            )


def spatial_token_geometry(
    record: SpatialTokenRecord,
) -> dict[str, float | str]:
    token = decode_spatial_token(record)
    match token:
        case ZoneToken():
            return {
                "codec": record.codec,
                "encoder": record.encoder,
                "projection_head": record.projection_head,
                "token_decoder": record.token_decoder,
                "x": token.x,
                "y": token.y,
                "z": token.z,
            }
        case ObjectToken():
            object_geometry: dict[str, float | str] = {
                "codec": record.codec,
                "encoder": record.encoder,
                "projection_head": record.projection_head,
                "token_decoder": record.token_decoder,
                "entity_id": token.instance_id or record.memory_id,
                "instance_id": token.instance_id or record.memory_id,
                "label": token.object_label,
                "object_label": token.object_label,
                "x": token.x,
                "y": token.y,
                "z": token.z,
                "provenance": token.provenance,
                "coordinate_frame": token.coordinate_frame,
                "change_type": token.change_type,
                "last_seen_time": record.end_time,
            }
            if token.uncertainty_m is not None:
                object_geometry["uncertainty_m"] = token.uncertainty_m
            if token.valid_from is not None:
                object_geometry["valid_from"] = token.valid_from
            if token.valid_to is not None:
                object_geometry["valid_to"] = token.valid_to
            return object_geometry
        case RelationToken():
            relation_geometry: dict[str, float | str] = {
                "codec": record.codec,
                "encoder": record.encoder,
                "projection_head": record.projection_head,
                "relation": token.relation,
                "token_decoder": record.token_decoder,
                "coordinate_frame": token.coordinate_frame,
                "subject": token.subject,
                "object": token.object,
            }
            if token.subject_instance_id is not None:
                relation_geometry["subject_instance_id"] = token.subject_instance_id
            if token.object_instance_id is not None:
                relation_geometry["object_instance_id"] = token.object_instance_id
            if token.distance_m is not None:
                relation_geometry["distance_m"] = token.distance_m
            return relation_geometry


def spatial_token_relation_record(
    record: SpatialTokenRecord,
) -> SpatialRelationRecord | None:
    token = decode_spatial_token(record)
    if not isinstance(token, RelationToken):
        return None
    return SpatialRelationRecord(
        memory_id=record.memory_id,
        video_id=record.video_id,
        subject=token.subject,
        relation=token.relation,
        object=token.object,
        zone_id=token.zone_id,
        start_time=record.start_time,
        end_time=record.end_time,
        distance_m=token.distance_m,
        subject_instance_id=token.subject_instance_id,
        object_instance_id=token.object_instance_id,
        coordinate_frame=token.coordinate_frame,
        valid_from=token.valid_from,
        valid_to=token.valid_to,
    )


def _zone_candidates(  # noqa: PLR0913
    source: SourceStreamExample,
    zones: Sequence[ZoneRecord],
    options: SpatialCompressionOptions,
    *,
    encoder: str,
    projection_head: str,
    token_decoder: str,
    codec: SpatialMemoryCodec,
) -> tuple[SpatialTokenCandidate, ...]:
    seen: set[str] = set()
    candidates: list[SpatialTokenCandidate] = []
    for zone in sorted(zones, key=lambda item: (item.video_id, item.zone_id)):
        if zone.video_id != source.video_id:
            raise SpatialCompressionError(
                detail=f"{zone.zone_id}: zone/source video_id mismatch",
            )
        if not zone.visit_intervals:
            raise SpatialCompressionError(
                detail=f"{zone.zone_id}: zone has no causal visit interval",
            )
        if zone.zone_id in seen:
            continue
        seen.add(zone.zone_id)
        availability_time = zone.visit_intervals[0][0]
        centroid = (
            zone.visit_centroids[0]
            if zone.visit_centroids
            else (zone.centroid_x, zone.centroid_y, zone.centroid_z)
        )
        cell_x, cell_y = zone.cell
        token = ZoneToken(
            scale_m=options.quantization_m,
            zone_id=zone.zone_id,
            x=centroid[0],
            y=centroid[1],
            z=centroid[2],
        )
        record = SpatialTokenRecord(
            memory_id=(
                f"spatial_token:{source.video_id}:zone:{cell_x}:{cell_y}:"
                f"{_format_float(availability_time)}"
            ),
            video_id=source.video_id,
            encoder=encoder,
            projection_head=projection_head,
            token_decoder=token_decoder,
            codec=codec.name,
            start_time=availability_time,
            end_time=availability_time,
            token=codec.encode(token),
            importance=0.0,
        )
        candidates.append(
            SpatialTokenCandidate(
                record=record,
                features=_features(
                    kind="zone",
                    confidence=0.8,
                    reliability=0.8,
                    frame_grounded=False,
                    metric_relation=False,
                    recency=_window_recency(
                        availability_time,
                        options.window_seconds,
                    ),
                ),
                state_key=f"zone:{zone.zone_id}",
                state_signature=record.token,
            ),
        )
    return tuple(candidates)


def _object_candidates(  # noqa: PLR0913
    source: SourceStreamExample,
    anchors: Sequence[SpatialAnchorRecord],
    options: SpatialCompressionOptions,
    *,
    encoder: str,
    projection_head: str,
    token_decoder: str,
    codec: SpatialMemoryCodec,
    extra_features: Mapping[str, Mapping[str, float]],
    object_delta_multiplier: float,
) -> tuple[SpatialTokenCandidate, ...]:
    candidates: list[SpatialTokenCandidate] = []
    memory_ids: set[str] = set()
    for anchor in sorted(anchors, key=lambda item: (item.start_time, item.memory_id)):
        token = ObjectToken(
            scale_m=options.quantization_m,
            object_label=anchor.object_label,
            zone_id=anchor.zone_id,
            x=anchor.x,
            y=anchor.y,
            z=anchor.z,
            confidence=anchor.confidence,
            provenance=anchor.provenance,
            instance_id=anchor.instance_id,
            coordinate_frame=anchor.coordinate_frame,
            uncertainty_m=(
                None
                if anchor.uncertainty_m is None
                else anchor.uncertainty_m
                + (math.sqrt(3.0) * options.quantization_m / 2.0)
            ),
            change_type=anchor.change_type,
            valid_from=anchor.valid_from,
            valid_to=anchor.valid_to,
        )
        identity = anchor.instance_id or anchor.memory_id
        memory_id = (
            f"spatial_token:{source.video_id}:object:{anchor.object_label}:"
            f"{_format_float(anchor.start_time)}"
        )
        memory_id = _unique_candidate_memory_id(memory_id, identity, memory_ids)
        memory_ids.add(memory_id)
        record = SpatialTokenRecord(
            memory_id=memory_id,
            video_id=source.video_id,
            encoder=encoder,
            projection_head=projection_head,
            token_decoder=token_decoder,
            codec=codec.name,
            start_time=anchor.start_time,
            end_time=anchor.end_time,
            token=codec.encode(token),
            importance=0.0,
            frame_refs=anchor.frame_refs[:1],
        )
        candidates.append(
            SpatialTokenCandidate(
                record=record,
                features=_features(
                    kind="object",
                    confidence=anchor.confidence,
                    reliability=_PROVENANCE_RELIABILITY[anchor.provenance],
                    frame_grounded=bool(anchor.frame_refs),
                    metric_relation=False,
                    recency=_window_recency(
                        anchor.end_time,
                        options.window_seconds,
                    ),
                    extra=extra_features.get(anchor.memory_id, {}),
                ),
                state_key=f"object:{anchor.object_label}:{identity}",
                state_signature=_token_json(
                    anchor.zone_id,
                    round(anchor.confidence * STATE_CONFIDENCE_SCALE),
                    anchor.provenance,
                    bool(anchor.frame_refs),
                    _time_millis(anchor.valid_from),
                ),
                state_position=(anchor.x, anchor.y, anchor.z),
                dedup_radius_m=(options.quantization_m * object_delta_multiplier),
            ),
        )
    return tuple(candidates)


def _relation_candidates(  # noqa: PLR0913
    source: SourceStreamExample,
    anchors: Sequence[SpatialAnchorRecord],
    relations: Sequence[SpatialRelationRecord],
    options: SpatialCompressionOptions,
    *,
    encoder: str,
    projection_head: str,
    token_decoder: str,
    codec: SpatialMemoryCodec,
    extra_features: Mapping[str, Mapping[str, float]],
) -> tuple[SpatialTokenCandidate, ...]:
    candidates: list[SpatialTokenCandidate] = []
    memory_ids: set[str] = set()
    for relation in sorted(
        relations,
        key=lambda item: (item.end_time, item.start_time, item.memory_id),
    ):
        subject_anchor = _relation_endpoint(
            anchors,
            label=relation.subject,
            instance_id=relation.subject_instance_id,
            relation=relation,
        )
        object_anchor = _relation_endpoint(
            anchors,
            label=relation.object,
            instance_id=relation.object_instance_id,
            relation=relation,
        )
        if subject_anchor is None or object_anchor is None:
            continue
        (
            subject,
            relation_kind,
            object_label,
            subject_instance_id,
            object_instance_id,
        ) = _canonical_relation(relation)
        if subject == object_label and subject_instance_id == object_instance_id:
            continue
        relation_confidence = min(
            subject_anchor.confidence,
            object_anchor.confidence,
        )
        relation_reliability = min(
            _PROVENANCE_RELIABILITY[subject_anchor.provenance],
            _PROVENANCE_RELIABILITY[object_anchor.provenance],
        )
        token = RelationToken(
            scale_m=options.quantization_m,
            subject=subject,
            relation=relation_kind,
            object=object_label,
            zone_id=relation.zone_id,
            distance_m=relation.distance_m,
            subject_instance_id=subject_instance_id,
            object_instance_id=object_instance_id,
            coordinate_frame=relation.coordinate_frame,
            valid_from=relation.valid_from,
            valid_to=relation.valid_to,
        )
        subject_identity = subject_instance_id or subject
        object_identity = object_instance_id or object_label
        memory_id = (
            f"spatial_token:{source.video_id}:relation:{subject}:"
            f"{relation_kind}:{object_label}:"
            f"{_format_float(relation.start_time)}"
        )
        memory_id = _unique_candidate_memory_id(
            memory_id,
            f"{subject_identity}:{object_identity}",
            memory_ids,
        )
        memory_ids.add(memory_id)
        record = SpatialTokenRecord(
            memory_id=memory_id,
            video_id=source.video_id,
            encoder=encoder,
            projection_head=projection_head,
            token_decoder=token_decoder,
            codec=codec.name,
            start_time=relation.start_time,
            end_time=relation.end_time,
            token=codec.encode(token),
            importance=0.0,
        )
        candidates.append(
            SpatialTokenCandidate(
                record=record,
                features=_features(
                    kind="relation",
                    confidence=relation_confidence,
                    reliability=relation_reliability,
                    frame_grounded=False,
                    metric_relation=relation.distance_m is not None,
                    recency=_window_recency(
                        relation.end_time,
                        options.window_seconds,
                    ),
                    extra=extra_features.get(relation.memory_id, {}),
                ),
                state_key=(
                    f"relation:{subject_identity}:{relation_kind}:{object_identity}"
                ),
                state_signature=record.token,
            ),
        )
    return tuple(candidates)


def _canonical_relation(
    relation: SpatialRelationRecord,
) -> tuple[str, SpatialRelationKind, str, str | None, str | None]:
    return _canonical_relation_values(
        relation.subject,
        relation.relation,
        relation.object,
        relation.subject_instance_id,
        relation.object_instance_id,
    )


def _canonical_relation_values(
    subject: str,
    relation: SpatialRelationKind,
    object_label: str,
    subject_instance_id: str | None,
    object_instance_id: str | None,
) -> tuple[str, SpatialRelationKind, str, str | None, str | None]:
    match relation:
        case "right_of":
            return (
                object_label,
                "left_of",
                subject,
                object_instance_id,
                subject_instance_id,
            )
        case "behind":
            return (
                object_label,
                "in_front_of",
                subject,
                object_instance_id,
                subject_instance_id,
            )
        case "below":
            return (
                object_label,
                "above",
                subject,
                object_instance_id,
                subject_instance_id,
            )
        case "near":
            left = (subject, subject_instance_id or "")
            right = (object_label, object_instance_id or "")
            if left <= right:
                return (
                    subject,
                    "near",
                    object_label,
                    subject_instance_id,
                    object_instance_id,
                )
            return (
                object_label,
                "near",
                subject,
                object_instance_id,
                subject_instance_id,
            )
        case other:
            return (
                subject,
                other,
                object_label,
                subject_instance_id,
                object_instance_id,
            )


def _unique_candidate_memory_id(
    base: str,
    suffix: str,
    seen: set[str],
) -> str:
    if base not in seen:
        return base
    candidate = f"{base}:{suffix}"
    duplicate_index = 2
    while candidate in seen:
        candidate = f"{base}:{suffix}:{duplicate_index}"
        duplicate_index += 1
    return candidate


def _relation_endpoint(
    anchors: Sequence[SpatialAnchorRecord],
    *,
    label: str,
    instance_id: str | None,
    relation: SpatialRelationRecord,
) -> SpatialAnchorRecord | None:
    valid_from = (
        relation.valid_from if relation.valid_from is not None else relation.start_time
    )
    valid_to = relation.valid_to if relation.valid_to is not None else relation.end_time
    matches = tuple(
        anchor
        for anchor in anchors
        if anchor.object_label == label
        and (instance_id is None or anchor.instance_id == instance_id)
        and anchor.start_time <= valid_to
        and anchor.end_time >= valid_from
        and anchor.end_time <= relation.end_time
    )
    if not matches:
        return None
    return max(matches, key=lambda anchor: (anchor.end_time, anchor.memory_id))


def _features(  # noqa: PLR0913
    *,
    kind: SpatialTokenKind,
    confidence: float,
    reliability: float,
    frame_grounded: bool,
    metric_relation: bool,
    recency: float,
    extra: Mapping[str, float] | None = None,
) -> Mapping[str, float]:
    return {
        "kind_object": float(kind == "object"),
        "kind_relation": float(kind == "relation"),
        "kind_zone": float(kind == "zone"),
        "confidence": confidence,
        "geometry_reliability": reliability,
        "frame_grounded": float(frame_grounded),
        "metric_relation": float(metric_relation),
        "recency": recency,
        **dict(extra or {}),
    }


def _validate_projected_geometry(
    source: SourceStreamExample,
    geometry: SpatialGeometryFeatureSet,
    *,
    encoder: str,
    projection_head: str,
) -> None:
    if geometry.encoder != encoder:
        raise SpatialCompressionError(
            detail=(
                f"{source.video_id}: projected encoder provenance "
                f"{geometry.encoder} does not match {encoder}"
            ),
        )
    if geometry.projection_head != projection_head:
        raise SpatialCompressionError(
            detail=(
                f"{source.video_id}: projection provenance "
                f"{geometry.projection_head} does not match {projection_head}"
            ),
        )
    for record in (*geometry.zones, *geometry.anchors, *geometry.relations):
        if record.video_id != source.video_id:
            raise SpatialCompressionError(
                detail=(
                    f"{source.video_id}: projected geometry contains "
                    f"video_id {record.video_id}"
                ),
            )


def _require_known_options(
    options: Mapping[str, ExperimentOption],
    allowed: frozenset[str],
    component: str,
) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise SpatialCompressionError(
            detail=f"{component}: unknown option {unknown[0]}",
        )


def _option_float(
    options: Mapping[str, ExperimentOption],
    name: str,
    default: float,
) -> float:
    if name not in options:
        return default
    value = options[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpatialCompressionError(detail=f"{name} must be a number")
    return float(value)


def _validate_options(options: SpatialCompressionOptions) -> None:
    if not math.isfinite(options.quantization_m) or options.quantization_m <= 0.0:
        raise SpatialCompressionError(detail="quantization_m must be positive")
    if options.max_tokens_per_window <= 0:
        raise SpatialCompressionError(
            detail="max_tokens_per_window must be positive",
        )
    if options.max_bytes_per_window <= 0:
        raise SpatialCompressionError(
            detail="max_bytes_per_window must be positive",
        )
    if not math.isfinite(options.window_seconds) or options.window_seconds <= 0.0:
        raise SpatialCompressionError(detail="window_seconds must be positive")
    if not math.isfinite(options.min_keep_score) or not (
        0.0 <= options.min_keep_score <= 1.0
    ):
        raise SpatialCompressionError(detail="min_keep_score must be in [0,1]")
    _ = _scale_cm(options.quantization_m)


def _experiment_config_from_env(
    env: Mapping[str, str],
) -> SpatialExperimentConfig:
    config_path = env.get(SPATIAL_EXPERIMENT_ENV, "").strip()
    if config_path:
        return load_spatial_experiment_config(Path(config_path))
    selector_path = env.get(SPATIAL_SELECTOR_PATH_ENV, "").strip()
    return SpatialExperimentConfig(
        selector_path=selector_path or None,
        token_budget=_env_int(
            env,
            SPATIAL_TOKEN_BUDGET_ENV,
            DEFAULT_TOKEN_BUDGET,
        ),
        byte_budget=_env_int(
            env,
            SPATIAL_BYTE_BUDGET_ENV,
            DEFAULT_BYTE_BUDGET,
        ),
        quantization_m=_env_float(
            env,
            SPATIAL_QUANTIZATION_ENV,
            DEFAULT_QUANTIZATION_M,
        ),
    )


def _component_factory[ComponentT](
    factories: Mapping[
        str,
        Callable[[Mapping[str, ExperimentOption]], ComponentT],
    ],
    name: str,
    component_type: str,
) -> Callable[[Mapping[str, ExperimentOption]], ComponentT]:
    factory = factories.get(name)
    if factory is None:
        raise SpatialCompressionError(
            detail=f"unknown {component_type}: {name}",
        )
    return factory


def _selector_path(env: Mapping[str, str]) -> Path | None:
    value = env.get(SPATIAL_SELECTOR_PATH_ENV, "").strip()
    return Path(value) if value else None


def _env_float(
    env: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    value = env.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise SpatialCompressionError(detail=f"{name} must be a number") from exc


def _env_int(
    env: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    value = env.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SpatialCompressionError(detail=f"{name} must be an integer") from exc


def _scale_cm(quantization_m: float) -> int:
    scale_cm = round(quantization_m * 100)
    if scale_cm <= 0 or not math.isclose(
        quantization_m,
        scale_cm / 100.0,
        abs_tol=1e-9,
    ):
        raise SpatialCompressionError(
            detail="quantization_m must be a positive whole-centimeter value",
        )
    return scale_cm


def _quantize(value: float, quantization_m: float) -> int:
    return round(value / quantization_m)


def _quantize_uncertainty(value: float, quantization_m: float) -> int:
    """Round uncertainty upward so the compact token never becomes overconfident."""
    return math.ceil(value / quantization_m)


def _time_millis(value: float | None) -> int:
    return -1 if value is None else round(value * 1_000)


def _token_json(*parts: object) -> str:
    return json.dumps(parts, separators=(",", ":"), ensure_ascii=True)


def _window_recency(timestamp: float, window_seconds: float) -> float:
    window_offset = timestamp % window_seconds
    return min(1.0, max(0.0, window_offset / window_seconds))


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


def _jsonl_bytes(records: Sequence[FrozenModel]) -> int:
    return sum(len(record.model_dump_json().encode("utf-8")) + 1 for record in records)


def _require_length(
    payload: list[object],
    expected: int,
    record: SpatialTokenRecord,
) -> None:
    if len(payload) != expected:
        raise SpatialCompressionError(
            detail=(
                f"{record.memory_id}: expected {expected} token fields, "
                f"received {len(payload)}"
            ),
        )


def _require_length_in(
    payload: list[object],
    expected: tuple[int, ...],
    record: SpatialTokenRecord,
) -> None:
    if len(payload) not in expected:
        choices = " or ".join(str(length) for length in expected)
        raise SpatialCompressionError(
            detail=(
                f"{record.memory_id}: expected {choices} token fields, "
                f"received {len(payload)}"
            ),
        )


def _coordinate_frame(
    payload: list[object],
    index: int,
    record: SpatialTokenRecord,
) -> SpatialCoordinateFrame:
    value = _string(payload, index, record)
    if value != "source_world":
        raise SpatialCompressionError(
            detail=f"{record.memory_id}: invalid coordinate frame {value}",
        )
    return "source_world"


def _change_type(
    payload: list[object],
    index: int,
    record: SpatialTokenRecord,
) -> SpatialChangeType:
    code = _string(payload, index, record)
    try:
        return _CODE_CHANGE[code]
    except KeyError as exc:
        raise SpatialCompressionError(
            detail=f"{record.memory_id}: invalid change type",
        ) from exc


def _optional_time(
    payload: list[object],
    index: int,
    record: SpatialTokenRecord,
) -> float | None:
    milliseconds = _integer(payload, index, record)
    return None if milliseconds < 0 else milliseconds / 1_000.0


def _scale_m(payload: list[object], record: SpatialTokenRecord) -> float:
    scale_cm = _integer(payload, 1, record)
    if scale_cm <= 0:
        raise SpatialCompressionError(
            detail=f"{record.memory_id}: scale must be positive",
        )
    return scale_cm / 100.0


def _coordinate(
    payload: list[object],
    index: int,
    scale_m: float,
    record: SpatialTokenRecord,
) -> float:
    return _integer(payload, index, record) * scale_m


def _string(
    payload: list[object],
    index: int,
    record: SpatialTokenRecord,
) -> str:
    try:
        value = payload[index]
    except IndexError as exc:
        raise SpatialCompressionError(
            detail=f"{record.memory_id}: missing token field {index}",
        ) from exc
    if not isinstance(value, str):
        raise SpatialCompressionError(
            detail=f"{record.memory_id}: token field {index} must be a string",
        )
    return value


def _integer(
    payload: list[object],
    index: int,
    record: SpatialTokenRecord,
) -> int:
    try:
        value = payload[index]
    except IndexError as exc:
        raise SpatialCompressionError(
            detail=f"{record.memory_id}: missing token field {index}",
        ) from exc
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpatialCompressionError(
            detail=f"{record.memory_id}: token field {index} must be an integer",
        )
    return value


def _format_float(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")
