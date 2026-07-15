"""Strict, single-source signed-attestation encoding.

This module deliberately implements the small I-JSON profile used by remote
approval documents.  It rejects ambiguous JSON before a document reaches any
signature check.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Final, Self, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, FiniteFloat, JsonValue, TypeAdapter, model_validator

from worldmm_smvqa.schema import FrozenModel

DOMAIN = b"worldmm-signed-attestation-v1"
SAFE_INTEGER = (1 << 53) - 1
_SURROGATE_MIN = 0xD800
_SURROGATE_MAX = 0xDFFF
_DECIMAL_NOTATION_MIN = 1e-6
_DECIMAL_NOTATION_MAX = 1e21
type CanonicalJSONValue = (
    None
    | bool
    | int
    | float
    | str
    | list[CanonicalJSONValue]
    | tuple[CanonicalJSONValue, ...]
    | Mapping[str, CanonicalJSONValue]
)
type JSONValue = CanonicalJSONValue
_JSON_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_DUPLICATE_MEMBER = "duplicate JSON member: {key!r}"
_SAFE_INTEGER_EXCEEDED = "I-JSON integer exceeds IEEE-754 safe range"
_NONFINITE_NUMBER = "I-JSON number is non-finite"
_INVALID_STRICT_JSON = "invalid strict JSON"
_LONE_SURROGATE = "I-JSON string contains a lone surrogate"
_UNSAFE_NUMBER = "I-JSON number is outside the safe range"
_NONSTRING_OBJECT_KEY = "attestation object key is not a string"
_NON_JSON_PAYLOAD = "non-I-JSON attestation payload"
_PAYLOAD_DIGEST_MISMATCH = "attestation payload_sha256 mismatch"
_INVALID_PURPOSE = "invalid attestation purpose"
_UNPADDED_BASE64URL = "base64url must be unpadded"
_INVALID_BASE64URL = "invalid base64url"

_APPROVED_PURPOSES = frozenset(
    {
        "teacher_cache_production",
        "identity_index",
        "oracle-preflight-self-check:capability_runner",
        "oracle-preflight-self-check:signer_vectors",
        "oracle-preflight-self-check:resolver",
        "oracle-preflight-self-check:quality",
    }
)
_ED25519_PUBLIC_KEY_SIZE_BYTES = 32
_ED25519_PUBLIC_KEY_SIZE_MESSAGE = "Ed25519 public key must be exactly 32 raw bytes"
_UNSUPPORTED_ENVELOPE_VERSION = "unsupported attestation envelope version"
_PUBLIC_KEY_ENCODING_MESSAGE = "attestation public key must be raw unpadded base64url"
_KEY_ID_DERIVATION_MESSAGE = "attestation key_id must derive from the raw public key"
_INVALID_KEY_WINDOW = "attestation key not_after must be after not_before"
_UNSUPPORTED_KEY_REGISTRY_VERSION = "unsupported attestation key registry version"
_DUPLICATE_KEY_ID = "attestation key registry key_id must be unique"
_UNAPPROVED_PURPOSE = "attestation purpose is not approved"
_UNAUTHORIZED_KEY = "attestation key is not authorized for purpose"
_SIGNATURE_VERIFICATION_FAILED = "signed attestation envelope verification failed"


def key_id_from_public_key(public_key: bytes) -> str:
    """Return the canonical key ID: raw unpadded base64url Ed25519 public key."""
    if len(public_key) != _ED25519_PUBLIC_KEY_SIZE_BYTES:
        raise AttestationError(_ED25519_PUBLIC_KEY_SIZE_MESSAGE)
    return b64url_encode(public_key)


class SignedAttestationEnvelopeV1(FrozenModel):
    """The sole portable signed-attestation envelope."""

    version: str = "signed-attestation-envelope-v1"
    key_id: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    payload: JsonValue
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    signature_b64url: str = Field(min_length=1)
    issued_at: FiniteFloat = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_envelope(self) -> Self:
        if self.version != "signed-attestation-envelope-v1":
            raise ValueError(_UNSUPPORTED_ENVELOPE_VERSION)
        if self.purpose not in _APPROVED_PURPOSES:
            raise ValueError(_INVALID_PURPOSE)
        if self.payload_sha256 != payload_sha256(self.signing_payload()):
            raise ValueError(_PAYLOAD_DIGEST_MISMATCH)
        _ = b64url_decode(self.signature_b64url)
        return self

    def signing_payload(self) -> dict[str, JsonValue]:
        """Return the unsigned envelope fields covered by the signature."""
        return {
            "version": self.version,
            "key_id": self.key_id,
            "purpose": self.purpose,
            "payload": self.payload,
            "payload_sha256": self.payload_sha256,
            "issued_at": self.issued_at,
        }


class ImmutableAttestationKeyV1(FrozenModel):
    """An immutable, purpose-scoped raw Ed25519 public-key entry."""

    key_id: str = Field(min_length=1)
    public_key_b64url: str = Field(min_length=1)
    purposes: tuple[str, ...] = Field(min_length=1)
    not_before: FiniteFloat = Field(ge=0.0)
    not_after: FiniteFloat = Field(gt=0.0)
    revoked: bool = False

    @model_validator(mode="after")
    def _validate_key(self) -> Self:
        public_key = b64url_decode(self.public_key_b64url)
        if b64url_encode(public_key) != self.public_key_b64url:
            raise ValueError(_PUBLIC_KEY_ENCODING_MESSAGE)
        if self.key_id != key_id_from_public_key(public_key):
            raise ValueError(_KEY_ID_DERIVATION_MESSAGE)
        if not set(self.purposes) <= _APPROVED_PURPOSES:
            raise ValueError(_INVALID_PURPOSE)
        if self.not_after <= self.not_before:
            raise ValueError(_INVALID_KEY_WINDOW)
        return self


class ImmutableAttestationKeyRegistryV1(FrozenModel):
    """Immutable registry with one canonical raw-key identity per entry."""

    registry_version: str = "immutable-attestation-key-registry-v1"
    keys: tuple[ImmutableAttestationKeyV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_registry(self) -> Self:
        if self.registry_version != "immutable-attestation-key-registry-v1":
            raise ValueError(_UNSUPPORTED_KEY_REGISTRY_VERSION)
        if len({key.key_id for key in self.keys}) != len(self.keys):
            raise ValueError(_DUPLICATE_KEY_ID)
        return self


def verify_signed_attestation_envelope(
    envelope: SignedAttestationEnvelopeV1,
    registry: ImmutableAttestationKeyRegistryV1,
    *,
    purpose: str,
) -> None:
    """Verify canonical framing, raw-key identity, purpose, and signature."""
    if purpose not in _APPROVED_PURPOSES or envelope.purpose != purpose:
        raise AttestationError(_UNAPPROVED_PURPOSE)
    key = next((item for item in registry.keys if item.key_id == envelope.key_id), None)
    if (
        key is None
        or key.revoked
        or purpose not in key.purposes
        or not key.not_before <= envelope.issued_at <= key.not_after
    ):
        raise AttestationError(_UNAUTHORIZED_KEY)
    try:
        Ed25519PublicKey.from_public_bytes(b64url_decode(key.public_key_b64url)).verify(
            b64url_decode(envelope.signature_b64url),
            signing_bytes(with_payload_sha256(envelope.signing_payload()), purpose),
        )
    except (ValueError, InvalidSignature) as exc:
        raise AttestationError(_SIGNATURE_VERIFICATION_FAILED) from exc


class AttestationError(ValueError):
    """An attestation cannot be represented or verified safely."""


def _duplicate_object(
    pairs: list[tuple[str, JsonValue]],
) -> dict[str, JsonValue]:
    value: dict[str, JsonValue] = {}
    for key, item in pairs:
        if key in value:
            raise AttestationError(_DUPLICATE_MEMBER.format(key=key))
        value[key] = item
    return value


def loads_strict(raw: bytes | str) -> JsonValue:
    """Parse I-JSON while rejecting duplicate members and unsafe integers."""

    def parse_int(token: str) -> int:
        value = int(token)
        if abs(value) > SAFE_INTEGER:
            raise AttestationError(_SAFE_INTEGER_EXCEEDED)
        return value

    def parse_float(token: str) -> float:
        value = float(Decimal(token))
        if not math.isfinite(value):
            raise AttestationError(_NONFINITE_NUMBER)
        return value

    try:
        return _JSON_VALUE_ADAPTER.validate_python(
            json.loads(
                raw,
                object_pairs_hook=_duplicate_object,
                parse_int=parse_int,
                parse_float=parse_float,
            )
        )
    except (json.JSONDecodeError, ValueError, ArithmeticError) as exc:
        if isinstance(exc, AttestationError):
            raise
        raise AttestationError(_INVALID_STRICT_JSON) from exc


def _quote(value: str) -> str:
    if any(_SURROGATE_MIN <= ord(character) <= _SURROGATE_MAX for character in value):
        raise AttestationError(_LONE_SURROGATE)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _number(value: float) -> str:
    if not math.isfinite(value) or abs(value) > SAFE_INTEGER:
        raise AttestationError(_UNSAFE_NUMBER)
    if value == 0:
        return "0"
    # ECMAScript/JCS use decimal notation in this interval.  Decimal built from
    # repr preserves Python's shortest round-trippable IEEE-754 spelling.
    magnitude = abs(value)
    if _DECIMAL_NOTATION_MIN <= magnitude < _DECIMAL_NOTATION_MAX:
        rendered = format(Decimal(repr(value)), "f").rstrip("0").rstrip(".")
        return rendered or "0"
    mantissa, exponent = repr(value).lower().split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent_value = int(exponent)
    exponent_sign = "+" if exponent_value >= 0 else ""
    return f"{mantissa}e{exponent_sign}{exponent_value}"


def _render_scalar(item: CanonicalJSONValue) -> str | None:
    if item is None:
        return "null"
    if isinstance(item, bool):
        return str(item).lower()
    if isinstance(item, int):
        if abs(item) > SAFE_INTEGER:
            raise AttestationError(_SAFE_INTEGER_EXCEEDED)
        return str(item)
    if isinstance(item, float):
        return _number(item)
    if isinstance(item, str):
        return _quote(item)
    return None


def _render_array(values: Iterable[CanonicalJSONValue]) -> str:
    return "[" + ",".join(_render(value) for value in values) + "]"


def _render_object(value: Mapping[str, CanonicalJSONValue]) -> str:
    keys = sorted(value, key=lambda key: key.encode("utf-16be"))
    rendered = ",".join(f"{_quote(key)}:{_render(value[key])}" for key in keys)
    return "{" + rendered + "}"


def _render(item: CanonicalJSONValue) -> str:
    scalar = _render_scalar(item)
    if scalar is not None:
        return scalar
    if isinstance(item, list | tuple):
        return _render_array(item)
    if isinstance(item, Mapping):
        return _render_object(item)
    raise AttestationError(_NON_JSON_PAYLOAD)


def canonicalize(value: object) -> bytes:
    """Return RFC-8785-style canonical I-JSON bytes with UTF-16 key ordering."""
    canonical_value = cast(
        "CanonicalJSONValue",
        _JSON_VALUE_ADAPTER.validate_python(value),
    )
    return _render(canonical_value).encode("utf-8")


def payload_sha256(value: Mapping[str, JsonValue]) -> str:
    """Digest the unsigned payload, excluding transport-only envelope fields."""
    payload = {
        key: item
        for key, item in value.items()
        if key not in {"signature", "payload_sha256"}
    }
    return hashlib.sha256(canonicalize(payload)).hexdigest()


def with_payload_sha256(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = dict(value)
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def require_payload_sha256(value: Mapping[str, JsonValue]) -> None:
    digest = value.get("payload_sha256")
    if not isinstance(digest, str) or digest != payload_sha256(value):
        raise AttestationError(_PAYLOAD_DIGEST_MISMATCH)


def signing_bytes(value: Mapping[str, JsonValue], purpose: str) -> bytes:
    """Frame an unsigned SignedAttestationEnvelopeV1 payload exactly once."""
    if not purpose or not purpose.isascii() or "\x00" in purpose:
        raise AttestationError(_INVALID_PURPOSE)
    require_payload_sha256(value)
    unsigned = {key: item for key, item in value.items() if key != "signature"}
    payload = canonicalize(unsigned)
    return (
        DOMAIN
        + b"\x00"
        + purpose.encode("ascii")
        + b"\x00"
        + len(payload).to_bytes(8, "big")
        + payload
    )


def b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def b64url_decode(value: str) -> bytes:
    if not value or "=" in value:
        raise AttestationError(_UNPADDED_BASE64URL)
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (binascii.Error, ValueError) as exc:
        raise AttestationError(_INVALID_BASE64URL) from exc
