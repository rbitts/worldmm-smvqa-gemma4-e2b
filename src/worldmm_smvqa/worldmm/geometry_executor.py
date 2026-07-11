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


class GeometryEntityFact(FrozenModel):
    entity_id: str
    label: str
    provenance: str
    evidence_refs: tuple[str, ...]
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
    near_threshold_m: float = Field(default=1.5, gt=0.0)
    max_uncertainty_m: float = Field(default=0.5, ge=0.0)
    entity_index_complete: bool = False

    @model_validator(mode="after")
    def _require_finite_options(self) -> GeometryQuery:
        values = (
            self.wearer_yaw_degrees,
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


def plan_geometry_query(
    question_text: str,
    records: Sequence[GeometryInput],
    *,
    coordinate_frame: str = "world",
    wearer_yaw_degrees: float | None = None,
) -> GeometryQuery | None:
    """Plan the small deterministic operator subset; ambiguous text returns None."""
    text = question_text.casefold()
    operation = _operation(text)
    if operation is None:
        return None
    facts, _invalid = _normalize(records)
    mentions = _mentioned_entities(
        text,
        _latest(facts),
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
        )
    entity_ids = tuple(dict.fromkeys(fact.entity_id for fact in mentions))
    required = 1 if operation == "last_seen" else 2
    if len(entity_ids) != required:
        return None
    return GeometryQuery(
        operation=operation,
        coordinate_frame=coordinate_frame,
        subject=entity_ids[0],
        object=None if required == 1 else entity_ids[1],
        wearer_yaw_degrees=wearer_yaw_degrees,
    )


def geometry_proofs_for_question(
    question: QuestionRequest,
    evidence_pack: EvidencePack,
    *,
    coordinate_frame: str = "world",
    wearer_yaw_degrees: float | None = None,
) -> tuple[GeometryProof, ...]:
    """Build the one supported proof explicitly requested by a QA question."""
    records = tuple(
        item
        for item in evidence_pack.evidence
        if item.source_store == "spatial" and item.geometry is not None
    )
    facts, _invalid = _normalize(records)
    query = plan_geometry_query(
        question.question,
        facts,
        coordinate_frame=coordinate_frame,
        wearer_yaw_degrees=wearer_yaw_degrees,
    )
    return () if query is None else (execute_geometry(facts, query),)


def _pair(  # noqa: PLR0911
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
    right_m = (dx * math.cos(yaw)) - (dy * math.sin(yaw))
    front_m = (dx * math.sin(yaw)) + (dy * math.cos(yaw))
    if (
        max(abs(right_m), abs(front_m)) <= uncertainty
        or abs(abs(right_m) - abs(front_m)) <= uncertainty
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
        _Result(value=value, uncertainty=uncertainty, unit="meters"),
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


def _count(  # noqa: PLR0911
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
        return _proof(
            query,
            selected,
            _Result(
                reason=(f"conflicting latest records: {', '.join(sorted(conflicts))}")
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
    if any(fact.coordinate_frame != query.coordinate_frame for fact in selected):
        return _proof(
            query,
            selected,
            _Result(reason="entity coordinate frames do not match query frame"),
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
    evidence_refs = _text_items(raw.get("evidence_refs"))
    memory_id = raw.get("memory_id")
    if isinstance(memory_id, str) and memory_id not in evidence_refs:
        evidence_refs = (*evidence_refs, memory_id)
    return GeometryEntityFact(
        entity_id=entity_id,
        label=label,
        provenance=provenance,
        evidence_refs=evidence_refs,
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
    if matches and matches[0].entity_id in conflicts:
        return matches[0], f"conflicting latest records: {matches[0].entity_id}"
    if not matches:
        matches = [fact for fact in latest if fact.label == selector]
    if not matches:
        return None, f"entity not found: {selector}"
    if len(matches) > 1:
        return None, f"ambiguous entity selector: {selector}"
    if matches[0].entity_id in conflicts:
        return matches[0], f"conflicting latest records: {matches[0].entity_id}"
    return matches[0], ""


def _latest(
    facts: Sequence[GeometryEntityFact],
) -> tuple[GeometryEntityFact, ...]:
    return _latest_state(facts)[0]


def _latest_state(
    facts: Sequence[GeometryEntityFact],
) -> tuple[tuple[GeometryEntityFact, ...], frozenset[str]]:
    by_id: dict[str, list[GeometryEntityFact]] = {}
    for fact in facts:
        by_id.setdefault(fact.entity_id, []).append(fact)

    latest: list[GeometryEntityFact] = []
    conflicts: set[str] = set()
    for entity_id, entity_facts in sorted(by_id.items()):
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
            conflicts.add(entity_id)
        latest.append(reference)
    return tuple(latest), frozenset(conflicts)


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
