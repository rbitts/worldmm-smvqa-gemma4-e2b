"""Fail-closed decoding for the Slurm allocation accounting contract.

The contract deliberately consumes only allocation rows (`sacct -X`) and duplicates
(`-D`).  A successful producer has exactly one, cluster-qualified allocation row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Never, override

ACCOUNTING_FIELDS: Final[tuple[str, ...]] = (
    "JobIDRaw",
    "Cluster",
    "State%64",
    "ExitCode",
    "Restarts",
    "SLUID",
    "OriginalSLUID",
)
ACCOUNTING_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    field.split("%", 1)[0] for field in ACCOUNTING_FIELDS
)
# The fields above are available together on supported production Slurm versions.
MINIMUM_SLURM_VERSION: Final[tuple[int, int]] = (23, 2)
_SUCCESS_STATE: Final[str] = "COMPLETED"
_STATES: Final[frozenset[str]] = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "CONFIGURING",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PENDING",
        "PREEMPTED",
        "REQUEUED",
        "RESIZING",
        "REVOKED",
        "RUNNING",
        "SIGNALING",
        "SPECIAL_EXIT",
        "STAGE_OUT",
        "STOPPED",
        "SUSPENDED",
        "TIMEOUT",
    }
)
_VERSION_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:slurm\s+)?(\d+)\.(\d+)(?:\.\d+)?$", re.IGNORECASE
)
_EXIT_CODE_RE: Final[re.Pattern[str]] = re.compile(r"^\d+:\d+$")
_JOB_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[1-9]\d*$")
_CANCELLED_RE: Final[re.Pattern[str]] = re.compile(r"^CANCELLED(?: by \d+)?$")
_EXPECTED_ALLOCATION_ROWS: Final[int] = 1
_SACCT_DELIMITER: Final[str] = "|"
_SACCT_TRUNCATION_MARKER: Final[str] = "+"
_SUCCESS_EXIT_CODE: Final[str] = "0:0"
_NO_RESTARTS: Final[int] = 0
_CANCELLED_STATE: Final[str] = "CANCELLED"

_NONTERMINAL_STATES: Final[frozenset[str]] = frozenset(
    {
        "CONFIGURING",
        "PENDING",
        "RESIZING",
        "RUNNING",
        "SIGNALING",
        "STAGE_OUT",
        "SUSPENDED",
    }
)


@dataclass(frozen=True, slots=True)
class SlurmAccountingError(ValueError):
    """Accounting validation failure with a stable diagnostic detail."""

    detail: str

    @override
    def __str__(self) -> str:
        return f"SlurmAccountingError: {self.detail}"


def _fail(detail: str) -> Never:
    raise SlurmAccountingError(detail=detail)


@dataclass(frozen=True, slots=True)
class SacctCapability:
    """Capabilities recorded by preflight before any producer is submitted."""

    version: tuple[int, int]
    fields: frozenset[str]


@dataclass(frozen=True, slots=True)
class AccountingRecord:
    job_id_raw: str
    cluster: str
    state: str
    exit_code: str
    restarts: int
    sluid: str
    original_sluid: str


def sacct_command(*, sacct: str, cluster: str, job_id: str) -> tuple[str, ...]:
    """Return the lossless, allocation-only sacct invocation for one job."""
    _validate_cluster(cluster)
    _validate_job_id(job_id)
    return (
        sacct,
        "-D",
        "-X",
        "-n",
        "-P",
        f"--clusters={cluster}",
        f"--jobs={job_id}",
        f"--format={','.join(ACCOUNTING_FIELDS)}",
    )


def parse_sacct_version(value: str) -> tuple[int, int]:
    match = _VERSION_RE.fullmatch(value.strip())
    if match is None:
        _fail(f"unparseable Slurm version: {value!r}")
    return (int(match.group(1)), int(match.group(2)))


def preflight_capability(
    *, version_output: str, helpformat_output: str
) -> SacctCapability:
    """Validate the version and every required accounting field before submission."""
    version = parse_sacct_version(version_output)
    if version < MINIMUM_SLURM_VERSION:
        required = ".".join(map(str, MINIMUM_SLURM_VERSION))
        _fail(f"Slurm {required}+ is required, got {version_output!r}")
    fields = frozenset(
        token.strip()
        for token in re.split(r"[\s|,]+", helpformat_output)
        if token.strip()
    )
    missing = sorted(ACCOUNTING_FIELD_NAMES - fields)
    if missing:
        _fail(f"sacct lacks required accounting fields: {', '.join(missing)}")
    return SacctCapability(version=version, fields=fields)


def decode_accounting(
    payload: str,
    *,
    cluster: str,
    job_id: str,
    expected_sluid: str | None = None,
) -> AccountingRecord:
    """Decode and validate exactly one non-hidden accounting allocation row.

    Duplicate, requeued, restarted, federated, truncated, or mismatched records
    are not resolvable into a scientific success and therefore fail closed.
    """
    _validate_cluster(cluster)
    _validate_job_id(job_id)
    if expected_sluid is not None and not expected_sluid:
        _fail("expected SLUID must be non-empty when supplied")
    rows = [line for line in payload.splitlines() if line.strip()]
    if len(rows) != _EXPECTED_ALLOCATION_ROWS:
        _fail(f"expected exactly one allocation row, got {len(rows)}")
    columns = rows[0].split(_SACCT_DELIMITER)
    if len(columns) != len(ACCOUNTING_FIELDS):
        _fail(f"expected {len(ACCOUNTING_FIELDS)} sacct columns, got {len(columns)}")
    if any(_SACCT_TRUNCATION_MARKER in value for value in columns):
        _fail("truncated sacct value ('+') is not accepted")
    (
        job_id_raw,
        record_cluster,
        state,
        exit_code,
        restarts,
        sluid,
        original_sluid,
    ) = (column.strip() for column in columns)
    record = AccountingRecord(
        job_id_raw=job_id_raw,
        cluster=record_cluster,
        state=_normalize_state(state),
        exit_code=exit_code,
        restarts=_parse_restarts(restarts),
        sluid=sluid,
        original_sluid=original_sluid,
    )
    if record.job_id_raw != job_id:
        identity_detail = (
            "accounting allocation identity mismatch: "
            f"{record.job_id_raw!r} != {job_id!r}"
        )
        _fail(identity_detail)
    if record.cluster != cluster:
        _fail(
            f"accounting federation/cluster mismatch: {record.cluster!r} != {cluster!r}"
        )
    if not _EXIT_CODE_RE.fullmatch(record.exit_code):
        _fail(f"invalid sacct ExitCode: {record.exit_code!r}")
    if not record.sluid or not record.original_sluid:
        _fail("hidden or incomplete accounting identity")
    if record.sluid != record.original_sluid:
        _fail("accounting SLUID must equal OriginalSLUID")
    if expected_sluid is not None and record.sluid != expected_sluid:
        _fail("accounting SLUID lineage mismatch")
    if record.restarts != _NO_RESTARTS:
        _fail("restarted/requeued allocation is ambiguous")
    return record


def require_success(record: AccountingRecord) -> AccountingRecord:
    """Require a normal Slurm completion, not merely an exit-code-shaped row."""
    if record.state != _SUCCESS_STATE or record.exit_code != _SUCCESS_EXIT_CODE:
        failure_detail = (
            "producer did not succeed: "
            f"state={record.state} exit_code={record.exit_code}"
        )
        _fail(failure_detail)
    return record


def is_nonterminal(record: AccountingRecord) -> bool:
    """Return whether a valid allocation record may still settle before its deadline."""
    return record.state in _NONTERMINAL_STATES


def is_cancelled(record: AccountingRecord) -> bool:
    """Return whether the allocation reached Slurm's unambiguous cancelled state."""
    return record.state == _CANCELLED_STATE


def require_cancelled(record: AccountingRecord) -> AccountingRecord:
    """Require cancellation rather than treating every failed state alike."""
    if not is_cancelled(record):
        _fail(
            "allocation was not cancelled: "
            f"state={record.state} exit_code={record.exit_code}"
        )
    return record


def _normalize_state(value: str) -> str:
    if not value:
        _fail("hidden accounting state")
    if _CANCELLED_RE.fullmatch(value):
        return _CANCELLED_STATE
    if value not in _STATES:
        _fail(f"unknown or non-lossless Slurm state: {value!r}")
    return value


def _parse_restarts(value: str) -> int:
    if not re.fullmatch(r"\d+", value):
        _fail(f"invalid Restarts value: {value!r}")
    return int(value)


def _validate_cluster(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        _fail(f"unsafe cluster name: {value!r}")


def _validate_job_id(value: str) -> None:
    if not _JOB_ID_RE.fullmatch(value):
        _fail(f"unsafe allocation job ID: {value!r}")
