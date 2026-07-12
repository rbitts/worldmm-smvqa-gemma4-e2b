from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, Field, model_validator

from worldmm_smvqa.schema import FrozenModel

if TYPE_CHECKING:
    from worldmm_smvqa.retrieval_types import EvidencePack
    from worldmm_smvqa.schema import QuestionRequest

type GeometryOperation = Literal[
    "distance",
    "relative_direction",
    "near",
    "last_seen",
    "count",
]
type GeometryValue = bool | float | int | str
type GeometryInput = GeometryEntityFact | BaseModel | Mapping[str, object]

_GROUNDED = frozenset(
    {"observed", "object_geometry", "multi_view_fused", "human_confirmed"},
)
_VECTOR_DIMENSIONS = 3
_COUNT_INVALIDATING_EVENT_KINDS = frozenset(
    {"appeared", "disappeared", "moved"},
)


class GeometryEntityFact(FrozenModel):
    entity_id: str
    label: str
    provenance: str
    evidence_refs: tuple[str, ...]
    source_video_id: str | None = None
    x: float | None = None
    y: float | None = None
    z: float | None = None
    coordinate_frame: str | None = None
    uncertainty_m: float | None = Field(default=None, ge=0.0)
    last_seen_time: float | None = None
    time_uncertainty_s: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_geometry(self) -> GeometryEntityFact:
        position = (self.x, self.y, self.z)
        if any(value is not None for value in position) and not all(
            value is not None for value in position
        ):
            msg = "entity position requires x, y, and z"
            raise ValueError(msg)
        values = (
            *position,
            self.uncertainty_m,
            self.last_seen_time,
            self.time_uncertainty_s,
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            msg = "geometry values must be finite"
            raise ValueError(msg)
        return self


class GeometryQuery(FrozenModel):
    operation: GeometryOperation
    coordinate_frame: str = Field(min_length=1)
    subject: str | None = None
    object: str | None = None
    entity_label: str | None = None
    wearer_yaw_degrees: float | None = None
    wearer_yaw_uncertainty_degrees: float = Field(default=0.0, ge=0.0)
    near_threshold_m: float = Field(default=1.5, gt=0.0)
    max_uncertainty_m: float = Field(default=0.5, ge=0.0)
    entity_index_complete: bool = False

    @model_validator(mode="after")
    def _require_finite_options(self) -> GeometryQuery:
        values = (
            self.wearer_yaw_degrees,
            self.wearer_yaw_uncertainty_degrees,
            self.near_threshold_m,
            self.max_uncertainty_m,
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            msg = "geometry query values must be finite"
            raise ValueError(msg)
        return self


class GeometryProof(FrozenModel):
    proof_id: str
    answerable: bool
    value: GeometryValue | None
    entity_ids: tuple[str, ...]
    subject_entity_id: str | None = None
    object_entity_id: str | None = None
    operation: GeometryOperation
    coordinate_frame: str
    uncertainty: float | None = Field(default=None, ge=0.0)
    uncertainty_unit: Literal["meters", "seconds", "count"] | None = None
    provenance: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    reason: str | None = None

    @model_validator(mode="after")
    def _require_consistent_answer(self) -> GeometryProof:
        if self.answerable and (self.value is None or self.reason is not None):
            msg = "answerable proof requires value and no reason"
            raise ValueError(msg)
        if not self.answerable and (self.value is not None or self.reason is None):
            msg = "unanswerable proof requires null value and reason"
            raise ValueError(msg)
        return self


@dataclass(frozen=True, slots=True)
class _Result:
    value: GeometryValue | None = None
    uncertainty: float | None = None
    unit: Literal["meters", "seconds", "count"] | None = None
    reason: str | None = None


def execute_geometry(
    records: Sequence[GeometryInput],
    query: GeometryQuery,
) -> GeometryProof:
    """Execute one operation; yaw=0 faces +Y and +X is wearer-right."""
    facts, invalid = _normalize(records)
    if query.operation == "count":
        return _count(facts, invalid, query)
    if query.operation == "last_seen":
        return _last_seen(facts, query)
    return _pair(facts, query)


def plan_geometry_query(  # noqa: PLR0911,PLR0913
    question_text: str,
    records: Sequence[GeometryInput],
    *,
    coordinate_frame: str = "world",
    wearer_yaw_degrees: float | None = None,
    wearer_yaw_uncertainty_degrees: float = 0.0,
    entity_index_complete: bool = False,
) -> GeometryQuery | None:
    """Plan the small deterministic operator subset; ambiguous text returns None."""
    text = question_text.casefold()
    operation = _operation(text)
    if operation is None:
        return None
    facts, _invalid = _normalize(records)
    latest_facts = _latest(facts)
    explicit_entity_ids = frozenset(
        fact.entity_id
        for fact in latest_facts
        if _mention_position(text, fact.entity_id) is not None
    )
    mentions = _mentioned_entities(
        text,
        latest_facts,
        allow_ambiguous_label=operation == "count",
    )
    if mentions is None:
        return None
    if operation == "count":
        labels = tuple(dict.fromkeys(fact.label for fact in mentions))
        if len(labels) != 1:
            return None
        return GeometryQuery(
            operation=operation,
            coordinate_frame=coordinate_frame,
            entity_label=labels[0],
            entity_index_complete=entity_index_complete,
        )
    entity_ids = tuple(dict.fromkeys(fact.entity_id for fact in mentions))
    required = 1 if operation == "last_seen" else 2
    if len(entity_ids) != required:
        return None
    if (
        required > 1
        and not entity_index_complete
        and not set(entity_ids).issubset(explicit_entity_ids)
    ):
        return None
    return GeometryQuery(
        operation=operation,
        coordinate_frame=coordinate_frame,
        subject=entity_ids[0],
        object=None if required == 1 else entity_ids[1],
        wearer_yaw_degrees=wearer_yaw_degrees,
        wearer_yaw_uncertainty_degrees=wearer_yaw_uncertainty_degrees,
        entity_index_complete=entity_index_complete,
    )


def geometry_proofs_for_question(  # noqa: PLR0913
    question: QuestionRequest,
    evidence_pack: EvidencePack,
    *,
    coordinate_frame: str | None = None,
    wearer_yaw_degrees: float | None = None,
    wearer_yaw_uncertainty_degrees: float = 0.0,
    spatial_records: Sequence[GeometryInput] | None = None,
    entity_index_complete: bool = False,
) -> tuple[GeometryProof, ...]:
    """Build the one supported proof explicitly requested by a QA question."""
    retrieved_records = tuple(
        item
        for item in evidence_pack.evidence
        if item.source_store == "spatial" and item.geometry is not None
    )
    question_text = question.question.casefold()
    operation = _operation(question_text)
    persisted_records = tuple(spatial_records) if spatial_records is not None else ()
    certified_complete_index = spatial_records is not None and entity_index_complete
    transition_events = tuple(
        record
        for record in persisted_records
        if _event_kind(record) in _COUNT_INVALIDATING_EVENT_KINDS
    )
    records = (
        tuple(record for record in persisted_records if _is_object_record(record))
        if spatial_records is not None
        else retrieved_records
    )
    facts, _invalid = _normalize(records)
    fact_frames = tuple(
        dict.fromkeys(
            fact.coordinate_frame
            for fact in facts
            if fact.coordinate_frame is not None
        ),
    )
    effective_frame = (
        coordinate_frame
        if coordinate_frame is not None
        else fact_frames[0]
        if len(fact_frames) == 1
        else "ambiguous-coordinate-frame"
    )
    query = plan_geometry_query(
        question.question,
        facts,
        coordinate_frame=effective_frame,
        wearer_yaw_degrees=wearer_yaw_degrees,
        wearer_yaw_uncertainty_degrees=wearer_yaw_uncertainty_degrees,
        entity_index_complete=certified_complete_index,
    )
    if operation == "count" and transition_events:
        count_query = query or GeometryQuery(
            operation="count",
            coordinate_frame=effective_frame,
            entity_index_complete=certified_complete_index,
        )
        selected = tuple(
            fact
            for fact in _latest(facts)
            if count_query.entity_label is None
            or fact.label == count_query.entity_label
        )
        return (
            _proof(
                count_query,
                selected,
                _Result(
                    reason=(
                        "dynamic count requires consolidated object state"
                    ),
                ),
            ),
        )
    location_intent = _last_seen_location_intent(question_text)
    if operation == "last_seen" and (
        location_intent or not _last_seen_time_intent(question_text)
    ):
        last_seen_query = query or GeometryQuery(
            operation="last_seen",
            coordinate_frame=effective_frame,
            entity_index_complete=certified_complete_index,
        )
        selected = tuple(
            fact
            for fact in _latest(facts)
            if last_seen_query.subject is None
            or fact.entity_id == last_seen_query.subject
        )
        return (
            _proof(
                last_seen_query,
                selected,
                _Result(
                    reason=(
                        "last-seen location proof is not implemented"
                        if location_intent
                        else "last-seen proof requires an explicit time intent"
                    ),
                ),
                subject_entity_id=last_seen_query.subject,
            ),
        )
    if query is not None:
        if query.operation == "last_seen" and _newer_transition_event_exists(
            transition_events,
            facts,
            query.subject,
        ):
            return (
                _proof(
                    query,
                    facts,
                    _Result(
                        reason=(
                            "last-seen state is stale relative to a typed "
                            "change event"
                        ),
                    ),
                    subject_entity_id=query.subject,
                ),
            )
        return (execute_geometry(facts, query),)
    if operation is None:
        return ()
    fallback_query = GeometryQuery(
        operation=operation,
        coordinate_frame=effective_frame,
        wearer_yaw_degrees=wearer_yaw_degrees,
        wearer_yaw_uncertainty_degrees=wearer_yaw_uncertainty_degrees,
        entity_index_complete=certified_complete_index,
    )
    return (
        _proof(
            fallback_query,
            facts,
            _Result(reason="geometry query planning failed or selector is ambiguous"),
        ),
    )


def _pair(  # noqa: PLR0911,PLR0912
    facts: Sequence[GeometryEntityFact],
    query: GeometryQuery,
) -> GeometryProof:
    if query.subject is None or query.object is None:
        return _proof(query, (), _Result(reason="two entity selectors required"))
    subject, reason = _one(facts, query.subject)
    object_, object_reason = _one(facts, query.object)
    selected = tuple(item for item in (subject, object_) if item is not None)
    if reason or object_reason or subject is None or object_ is None:
        return _proof(
            query,
            selected,
            _Result(reason=reason or object_reason or "entity not found"),
            subject_entity_id=None if subject is None else subject.entity_id,
            object_entity_id=None if object_ is None else object_.entity_id,
        )
    if subject.entity_id == object_.entity_id:
        return _proof(
            query,
            selected,
            _Result(reason="operation requires two distinct entities"),
            subject_entity_id=subject.entity_id,
            object_entity_id=object_.entity_id,
        )
    if (
        subject.source_video_id is not None
        and object_.source_video_id is not None
        and subject.source_video_id != object_.source_video_id
    ):
        return _proof(
            query,
            selected,
            _Result(reason="entities belong to different source videos"),
            subject_entity_id=subject.entity_id,
            object_entity_id=object_.entity_id,
        )
    invalid = _invalid_metric_fact(subject, query) or _invalid_metric_fact(
        object_, query
    )
    if invalid:
        return _proof(
            query,
            selected,
            _Result(reason=invalid),
            subject_entity_id=subject.entity_id,
            object_entity_id=object_.entity_id,
        )

    sx, sy, sz = _position(subject)
    ox, oy, oz = _position(object_)
    dx, dy, dz = sx - ox, sy - oy, sz - oz
    distance = math.sqrt((dx * dx) + (dy * dy) + (dz * dz))
    uncertainty = math.hypot(
        cast("float", subject.uncertainty_m),
        cast("float", object_.uncertainty_m),
    )
    if not math.isfinite(distance) or not math.isfinite(uncertainty):
        return _proof(
            query,
            selected,
            _Result(reason="derived metric geometry is not finite"),
            subject_entity_id=subject.entity_id,
            object_entity_id=object_.entity_id,
        )
    if query.operation == "distance":
        return _proof(
            query,
            selected,
            _Result(value=distance, uncertainty=uncertainty, unit="meters"),
            subject_entity_id=subject.entity_id,
            object_entity_id=object_.entity_id,
        )
    if query.operation == "near":
        if abs(distance - query.near_threshold_m) <= uncertainty:
            return _proof(
                query,
                selected,
                _Result(reason="uncertainty interval crosses near threshold"),
                subject_entity_id=subject.entity_id,
                object_entity_id=object_.entity_id,
            )
        return _proof(
            query,
            selected,
            _Result(
                value=distance < query.near_threshold_m,
                uncertainty=uncertainty,
                unit="meters",
            ),
            subject_entity_id=subject.entity_id,
            object_entity_id=object_.entity_id,
        )
    if query.wearer_yaw_degrees is None:
        return _proof(
            query,
            selected,
            _Result(reason="finite wearer yaw required"),
            subject_entity_id=subject.entity_id,
            object_entity_id=object_.entity_id,
        )

    yaw = math.radians(query.wearer_yaw_degrees)
    yaw_uncertainty = math.radians(query.wearer_yaw_uncertainty_degrees)
    direction_uncertainty = uncertainty + (
        distance * abs(math.sin(min(yaw_uncertainty, math.pi / 2)))
    )
    right_m = (dx * math.cos(yaw)) - (dy * math.sin(yaw))
    front_m = (dx * math.sin(yaw)) + (dy * math.cos(yaw))
    if (
        max(abs(right_m), abs(front_m)) <= direction_uncertainty
        or abs(abs(right_m) - abs(front_m)) <= direction_uncertainty
    ):
        return _proof(
            query,
            selected,
            _Result(reason="direction is ambiguous within positional uncertainty"),
            subject_entity_id=subject.entity_id,
            object_entity_id=object_.entity_id,
        )
    if abs(right_m) > abs(front_m):
        value = "right" if right_m > 0 else "left"
    else:
        value = "front" if front_m > 0 else "behind"
    return _proof(
        query,
        selected,
        _Result(value=value, uncertainty=direction_uncertainty, unit="meters"),
        subject_entity_id=subject.entity_id,
        object_entity_id=object_.entity_id,
    )


def _last_seen(  # noqa: PLR0911
    facts: Sequence[GeometryEntityFact],
    query: GeometryQuery,
) -> GeometryProof:
    if not query.entity_index_complete:
        return _proof(
            query,
            (),
            _Result(reason="last-seen requires a complete entity index"),
        )
    if query.subject is None:
        return _proof(query, (), _Result(reason="entity selector required"))
    fact, reason = _one(facts, query.subject)
    if fact is None:
        return _proof(query, (), _Result(reason=reason))
    if reason:
        return _proof(
            query,
            (fact,),
            _Result(reason=reason),
            subject_entity_id=fact.entity_id,
        )
    if fact.provenance not in _GROUNDED:
        return _proof(
            query,
            (fact,),
            _Result(reason="entity provenance is not geometry-grounded"),
            subject_entity_id=fact.entity_id,
        )
    if fact.last_seen_time is None:
        return _proof(
            query,
            (fact,),
            _Result(reason="last-seen time is missing"),
            subject_entity_id=fact.entity_id,
        )
    return _proof(
        query,
        (fact,),
        _Result(
            value=fact.last_seen_time,
            uncertainty=fact.time_uncertainty_s,
            unit="seconds",
        ),
        subject_entity_id=fact.entity_id,
    )


def _count(
    facts: Sequence[GeometryEntityFact],
    invalid: int,
    query: GeometryQuery,
) -> GeometryProof:
    latest, conflicts = _latest_state(facts)
    selected = tuple(
        fact
        for fact in latest
        if query.entity_label is None or fact.label == query.entity_label
    )
    if not query.entity_index_complete:
        return _proof(
            query,
            selected,
            _Result(reason="count requires a complete entity index"),
        )
    if invalid:
        return _proof(
            query,
            selected,
            _Result(reason="one or more retrieved records are invalid"),
        )
    if conflicts:
        conflicting_ids = ", ".join(
            f"{video_id}/{entity_id}" if video_id else entity_id
            for video_id, entity_id in sorted(conflicts)
        )
        return _proof(
            query,
            selected,
            _Result(
                reason=f"conflicting latest records: {conflicting_ids}"
            ),
        )
    if not selected:
        return _proof(
            query,
            (),
            _Result(reason="no matching grounded entity records"),
        )
    if any(fact.provenance not in _GROUNDED for fact in selected):
        return _proof(
            query,
            selected,
            _Result(reason="one or more entities are not geometry-grounded"),
        )
    return _proof(
        query,
        selected,
        _Result(value=len(selected), uncertainty=0.0, unit="count"),
    )


def _normalize(
    records: Sequence[GeometryInput],
) -> tuple[tuple[GeometryEntityFact, ...], int]:
    facts: list[GeometryEntityFact] = []
    invalid = 0
    for record in records:
        try:
            facts.append(_fact(record))
        except (TypeError, ValueError):
            invalid += 1
    return tuple(facts), invalid


def _is_object_record(record: GeometryInput) -> bool:
    record_type = _record_value(record, "record_type")
    return record_type is None or record_type == "object"


def _event_kind(record: GeometryInput) -> str | None:
    if _record_value(record, "record_type") != "event":
        return None
    value = _record_value(record, "event_kind")
    return value if isinstance(value, str) else None


def _newer_transition_event_exists(
    events: Sequence[GeometryInput],
    facts: Sequence[GeometryEntityFact],
    subject: str | None,
) -> bool:
    if subject is None:
        return False
    matching = tuple(fact for fact in _latest(facts) if fact.entity_id == subject)
    if len(matching) != 1 or matching[0].last_seen_time is None:
        return False
    last_seen_time = matching[0].last_seen_time
    return any(
        subject in _record_text_items(event, "involved_entity_ids")
        and (event_time := _record_time(event)) is not None
        and event_time > last_seen_time
        for event in events
    )


def _record_value(record: GeometryInput, key: str) -> object:
    if isinstance(record, GeometryEntityFact):
        return None
    raw = dict(record) if isinstance(record, Mapping) else record.model_dump()
    value = raw.get(key)
    if value is not None:
        return value
    geometry = _mapping(raw.get("geometry"))
    return None if geometry is None else geometry.get(key)


def _record_text_items(record: GeometryInput, key: str) -> tuple[str, ...]:
    value = _record_value(record, key)
    if isinstance(value, str):
        try:
            decoded = cast("object", json.loads(value))
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            decoded_items = cast("Sequence[object]", decoded)
            if all(isinstance(item, str) for item in decoded_items):
                return tuple(cast("Sequence[str]", decoded_items))
        # Legacy event projections joined identifiers with spaces.
        return tuple(item for item in value.split() if item)
    if isinstance(value, (list, tuple)):
        items = cast("Sequence[object]", value)
        if all(isinstance(item, str) for item in items):
            return tuple(cast("Sequence[str]", items))
    return ()


def _record_time(record: GeometryInput) -> float | None:
    for key in ("last_seen_time", "end_time"):
        value = _record_value(record, key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            time = float(value)
            return time if math.isfinite(time) else None
    validity = _record_value(record, "validity")
    validity_mapping = _mapping(validity)
    if validity_mapping is None:
        return None
    value = validity_mapping.get("end_time")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        time = float(value)
        return time if math.isfinite(time) else None
    return None


def _fact(record: GeometryInput) -> GeometryEntityFact:
    if isinstance(record, GeometryEntityFact):
        return record
    raw = dict(record) if isinstance(record, Mapping) else record.model_dump()
    geometry = _mapping(raw.get("geometry"))
    if geometry is not None:
        raw.update(geometry)
    centroid = raw.get("centroid")
    position = _vector3(centroid) if centroid is not None else (None, None, None)
    uncertainty = _mapping(raw.get("geometry_uncertainty"))
    uncertainty_m = _number(raw.get("uncertainty_m"))
    if uncertainty_m is None and uncertainty is not None:
        uncertainty_m = _number(uncertainty.get("standard_deviation_m"))
    entity_id = _text(raw, "entity_id", "instance_id", "object_id", "memory_id")
    label = _text(raw, "label", "semantic_label", "object_label", "object")
    provenance = _text(raw, "provenance", "geometry_source")
    evidence_refs = _text_items(raw.get("frame_refs"))
    if not evidence_refs:
        evidence_refs = _text_items(raw.get("evidence_refs"))
    memory_id = raw.get("memory_id")
    if isinstance(memory_id, str) and memory_id not in evidence_refs:
        evidence_refs = (*evidence_refs, memory_id)
    return GeometryEntityFact(
        entity_id=entity_id,
        label=label,
        provenance=provenance,
        evidence_refs=evidence_refs,
        source_video_id=_maybe_text(raw, "video_id", "source_video_id"),
        x=_number(raw.get("x")) if raw.get("x") is not None else position[0],
        y=_number(raw.get("y")) if raw.get("y") is not None else position[1],
        z=_number(raw.get("z")) if raw.get("z") is not None else position[2],
        coordinate_frame=_maybe_text(raw, "coordinate_frame", "local_frame_id"),
        uncertainty_m=uncertainty_m,
        last_seen_time=_number(raw.get("last_seen_time", raw.get("end_time"))),
        time_uncertainty_s=_number(raw.get("time_uncertainty_s")) or 0.0,
    )


def _one(
    facts: Sequence[GeometryEntityFact],
    selector: str,
) -> tuple[GeometryEntityFact | None, str]:
    latest, conflicts = _latest_state(facts)
    matches = [fact for fact in latest if fact.entity_id == selector]
    if len(matches) == 1 and _entity_key(matches[0]) in conflicts:
        return matches[0], f"conflicting latest records: {matches[0].entity_id}"
    if not matches:
        matches = [fact for fact in latest if fact.label == selector]
    if not matches:
        return None, f"entity not found: {selector}"
    if len(matches) > 1:
        return None, f"ambiguous entity selector: {selector}"
    if _entity_key(matches[0]) in conflicts:
        return matches[0], f"conflicting latest records: {matches[0].entity_id}"
    return matches[0], ""


def _latest(
    facts: Sequence[GeometryEntityFact],
) -> tuple[GeometryEntityFact, ...]:
    return _latest_state(facts)[0]


def _latest_state(
    facts: Sequence[GeometryEntityFact],
) -> tuple[tuple[GeometryEntityFact, ...], frozenset[tuple[str, str]]]:
    by_id: dict[tuple[str, str], list[GeometryEntityFact]] = {}
    for fact in facts:
        by_id.setdefault(_entity_key(fact), []).append(fact)

    latest: list[GeometryEntityFact] = []
    conflicts: set[tuple[str, str]] = set()
    for entity_key, entity_facts in sorted(by_id.items()):
        latest_time = max(
            -math.inf if fact.last_seen_time is None else fact.last_seen_time
            for fact in entity_facts
        )
        candidates = sorted(
            (
                fact
                for fact in entity_facts
                if (-math.inf if fact.last_seen_time is None else fact.last_seen_time)
                == latest_time
            ),
            key=_canonical_fact,
        )
        reference = candidates[0]
        reference_geometry = _geometry_state(reference)
        if any(
            _geometry_state(candidate) != reference_geometry
            for candidate in candidates[1:]
        ):
            conflicts.add(entity_key)
        latest.append(reference)
    return tuple(latest), frozenset(conflicts)


def _entity_key(fact: GeometryEntityFact) -> tuple[str, str]:
    return (fact.source_video_id or "", fact.entity_id)


def _canonical_fact(fact: GeometryEntityFact) -> str:
    return json.dumps(
        fact.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _geometry_state(
    fact: GeometryEntityFact,
) -> tuple[float | None, float | None, float | None, str | None]:
    return (fact.x, fact.y, fact.z, fact.coordinate_frame)


def _invalid_metric_fact(fact: GeometryEntityFact, query: GeometryQuery) -> str:
    if fact.provenance not in _GROUNDED:
        return f"entity is not geometry-grounded: {fact.entity_id}"
    if fact.coordinate_frame != query.coordinate_frame:
        return f"coordinate frame mismatch: {fact.entity_id}"
    if fact.x is None or fact.y is None or fact.z is None:
        return f"entity position is missing: {fact.entity_id}"
    if fact.uncertainty_m is None:
        return f"entity uncertainty is missing: {fact.entity_id}"
    if fact.uncertainty_m > query.max_uncertainty_m:
        return f"entity uncertainty exceeds limit: {fact.entity_id}"
    return ""


def _position(fact: GeometryEntityFact) -> tuple[float, float, float]:
    return cast("tuple[float, float, float]", (fact.x, fact.y, fact.z))


def _proof(
    query: GeometryQuery,
    facts: Sequence[GeometryEntityFact],
    result: _Result,
    *,
    subject_entity_id: str | None = None,
    object_entity_id: str | None = None,
) -> GeometryProof:
    payload: dict[str, object] = {
        "query": query.model_dump(mode="json"),
        "answerable": result.reason is None,
        "value": result.value,
        "entity_ids": tuple(sorted(fact.entity_id for fact in facts)),
        "subject_entity_id": subject_entity_id,
        "object_entity_id": object_entity_id,
        "operation": query.operation,
        "coordinate_frame": query.coordinate_frame,
        "uncertainty": result.uncertainty,
        "uncertainty_unit": result.unit,
        "provenance": tuple(sorted({fact.provenance for fact in facts})),
        "evidence_refs": tuple(
            sorted({ref for fact in facts for ref in fact.evidence_refs}),
        ),
        "reason": result.reason,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    proof_id = f"geometry-proof:{hashlib.sha256(canonical).hexdigest()[:20]}"
    return GeometryProof(
        proof_id=proof_id,
        answerable=result.reason is None,
        value=result.value,
        entity_ids=tuple(sorted(fact.entity_id for fact in facts)),
        subject_entity_id=subject_entity_id,
        object_entity_id=object_entity_id,
        operation=query.operation,
        coordinate_frame=query.coordinate_frame,
        uncertainty=result.uncertainty,
        uncertainty_unit=result.unit,
        provenance=tuple(sorted({fact.provenance for fact in facts})),
        evidence_refs=tuple(
            sorted({ref for fact in facts for ref in fact.evidence_refs}),
        ),
        reason=result.reason,
    )


def _text(raw: Mapping[str, object], *keys: str) -> str:
    value = _maybe_text(raw, *keys)
    if value is None:
        msg = f"required text missing: {keys[0]}"
        raise ValueError(msg)
    return value


def _maybe_text(raw: Mapping[str, object], *keys: str) -> str | None:
    return next(
        (value for key in keys if isinstance((value := raw.get(key)), str) and value),
        None,
    )


def _number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = "geometry number must be int or float"
        raise TypeError(msg)
    return float(value)


def _mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, BaseModel):
        return cast("Mapping[str, object]", value.model_dump())
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    return None


def _vector3(value: object) -> tuple[float | None, float | None, float | None]:
    if not isinstance(value, (list, tuple)):
        msg = "geometry centroid must contain exactly three numbers"
        raise TypeError(msg)
    items = cast("Sequence[object]", value)
    if len(items) != _VECTOR_DIMENSIONS:
        msg = "geometry centroid must contain exactly three numbers"
        raise TypeError(msg)
    return (_number(items[0]), _number(items[1]), _number(items[2]))


def _text_items(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        items = cast("Sequence[object]", value)
        if all(isinstance(item, str) for item in items):
            return tuple(cast("Sequence[str]", items))
    msg = "evidence_refs must contain strings"
    raise TypeError(msg)


def _operation(text: str) -> GeometryOperation | None:
    if any(term in text for term in ("last seen", "last saw", "last observed")):
        return "last_seen"
    if any(term in text for term in ("how many", "number of", "count of")):
        return "count"
    if any(term in text for term in ("how far", "distance")):
        return "distance"
    if any(term in text for term in (" near ", "close to", "nearby")):
        return "near"
    if any(
        term in text
        for term in (
            "left of",
            "right of",
            "in front of",
            "behind",
            "direction",
            "relative to",
        )
    ):
        return "relative_direction"
    return None


def _last_seen_time_intent(text: str) -> bool:
    return any(
        term in text
        for term in ("when", "what time", "at what time", "timestamp")
    )


def _last_seen_location_intent(text: str) -> bool:
    return any(
        term in text
        for term in ("where", "which room", "what room", "location")
    )


def _mentioned_entities(
    text: str,
    facts: Sequence[GeometryEntityFact],
    *,
    allow_ambiguous_label: bool,
) -> tuple[GeometryEntityFact, ...] | None:
    matches: list[tuple[int, GeometryEntityFact]] = []
    explicit_ids: set[str] = set()
    for fact in facts:
        position = _mention_position(text, fact.entity_id)
        if position is not None:
            explicit_ids.add(fact.entity_id)
            matches.append((position, fact))
    labels = tuple(dict.fromkeys(fact.label for fact in facts))
    for label in labels:
        position = _mention_position(text, label, allow_plural=True)
        if position is None:
            continue
        label_facts = tuple(
            fact
            for fact in facts
            if fact.label == label and fact.entity_id not in explicit_ids
        )
        if len(label_facts) > 1 and not allow_ambiguous_label:
            return None
        matches.extend((position, fact) for fact in label_facts)
    matches.sort(key=lambda item: (item[0], item[1].entity_id))
    return tuple(dict.fromkeys(fact for _position, fact in matches))


def _mention_position(
    text: str,
    value: str,
    *,
    allow_plural: bool = False,
) -> int | None:
    suffix = "(?:s|es)?" if allow_plural else ""
    match = re.search(
        rf"(?<!\w){re.escape(value.casefold())}{suffix}(?!\w)",
        text,
    )
    return None if match is None else match.start()
