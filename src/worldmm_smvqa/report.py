from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Literal, override

from pydantic import ConfigDict, Field, ValidationError, field_validator

from worldmm_smvqa.schema import FrozenModel

REQUIRED_REPORT_SECTIONS: Final = (
    "Local code/config changed",
    "Remote command used",
    "Remote job ID or process reference",
    "Remote artifact path on company storage",
    "Key metrics or failure reason",
    "What was not copied locally",
)
SENSITIVE_MARKERS: Final = (
    "authorization",
    "bearer ",
    "password",
    "secret",
    "token",
)
MARKDOWN_SPECIALS: Final = "\\*_{}[]<>()#+-.!|"


@dataclass(frozen=True, slots=True)
class IncompleteRemoteManifestError(Exception):
    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"IncompleteRemoteManifest: {self.path}: {self.detail}"


class ReportMetric(FrozenModel):
    name: str = Field(min_length=1)
    value: float


class RemoteRunManifest(FrozenModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    baseline_name: Literal["WorldMM-SMVQA"]
    remote_status: Literal["pending", "failed", "complete"]
    local_changes: tuple[str, ...] = Field(min_length=1)
    remote_command: str = Field(min_length=1)
    remote_job_reference: str = Field(min_length=1)
    remote_artifact_path: str = Field(min_length=1)
    metrics: tuple[ReportMetric, ...] = ()
    failure_reason: str | None = None
    not_copied_locally: tuple[str, ...] = Field(min_length=1)

    @field_validator(
        "remote_command",
        "remote_job_reference",
        "remote_artifact_path",
        "failure_reason",
        mode="after",
    )
    @classmethod
    def _strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip()


def write_report(run_manifest: Path, output: Path) -> None:
    manifest = read_run_manifest(run_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(render_report(manifest), encoding="utf-8")


def read_run_manifest(path: Path) -> RemoteRunManifest:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IncompleteRemoteManifestError(path=path, detail=str(exc)) from exc
    try:
        return RemoteRunManifest.model_validate_json(raw)
    except ValidationError as exc:
        raise IncompleteRemoteManifestError(
            path=path,
            detail=_validation_detail(exc),
        ) from exc


def render_report(manifest: RemoteRunManifest) -> str:
    lines = [
        "# WorldMM-SMVQA Final Result Handoff",
        "",
        f"remote_status: {_plain(manifest.remote_status)}",
        f"baseline_name: {_plain(manifest.baseline_name)}",
        "",
        "## Local code/config changed",
        *_bullets(manifest.local_changes),
        "",
        "## Remote command used",
        _code_block(_plain(manifest.remote_command)),
        "",
        "## Remote job ID or process reference",
        _markdown_text(manifest.remote_job_reference),
        "",
        "## Remote artifact path on company storage",
        _markdown_text(manifest.remote_artifact_path),
        "",
        "## Key metrics or failure reason",
        *_metric_lines(manifest),
        "",
        "## What was not copied locally",
        *_bullets(manifest.not_copied_locally),
        "",
    ]
    return "\n".join(lines)


def _metric_lines(manifest: RemoteRunManifest) -> tuple[str, ...]:
    if manifest.metrics:
        return tuple(
            f"- {_markdown_text(metric.name)}: {metric.value:.6f}"
            for metric in manifest.metrics
        )
    reason = manifest.failure_reason or "Remote benchmark not run."
    return (f"No remote metrics reported. {_markdown_text(reason)}",)


def _bullets(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"- {_markdown_text(value)}" for value in values)


def _code_block(value: str) -> str:
    return f"```bash\n{value}\n```"


def _validation_detail(exc: ValidationError) -> str:
    lines = tuple(
        line
        for line in str(exc).splitlines()
        if "input_value=" not in line and "For further information" not in line
    )
    return " ".join(lines)


def _plain(value: str) -> str:
    text = " ".join(value.replace("`", "'").split())
    lowered = text.lower()
    if any(marker in lowered for marker in SENSITIVE_MARKERS):
        return "[redacted sensitive manifest text]"
    return text


def _markdown_text(value: str) -> str:
    return "".join(
        f"\\{character}" if character in MARKDOWN_SPECIALS else character
        for character in _plain(value)
    )
