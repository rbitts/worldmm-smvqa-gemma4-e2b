from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Literal, Self, override

from pydantic import (
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

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
METRIC_PERCENT_MAX: Final = 100.0


@dataclass(frozen=True, slots=True)
class IncompleteRemoteManifestError(Exception):
    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"IncompleteRemoteManifest: {self.path}: {self.detail}"


class ReportMetric(FrozenModel):
    experiment: Literal["PROBE", "E1", "E2", "E3"]
    name: str = Field(min_length=1)
    value: float = Field(allow_inf_nan=False)


class RemoteRunManifest(FrozenModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    baseline_name: Literal["WorldMM-SMVQA"]
    remote_status: Literal["pending", "failed", "complete"]
    result_class: Literal[
        "contract_probe",
        "mock",
        "heuristic",
        "student",
        "official",
    ]
    experiment_id: str = Field(min_length=1)
    execution_profile: Literal["probe", "full", "not-run"] = "not-run"
    lane: str = Field(min_length=1)
    split_id: str = Field(min_length=1)
    code_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    typed_memory_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    inference_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evidence_lineage_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    predictions_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metrics_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    qa_resume_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    run_identity_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    finalization_inputs_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
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

    @field_validator(
        "remote_command",
        "remote_job_reference",
        "remote_artifact_path",
        mode="after",
    )
    @classmethod
    def _require_nonblank_remote_text(cls, value: str) -> str:
        if not value:
            msg = "remote execution identity must not be blank"
            raise ValueError(msg)
        return value

    @field_validator("experiment_id", "lane", "split_id", mode="after")
    @classmethod
    def _strip_identity(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "result identity values must not be blank"
            raise ValueError(msg)
        return stripped

    @model_validator(mode="after")
    def _require_complete_learned_identity(self) -> Self:
        if self.remote_status != "complete":
            return self
        if self.result_class == "official":
            msg = (
                "official completion requires per-experiment immutable manifests "
                "(not implemented)"
            )
            raise ValueError(msg)
        if self.result_class in {"mock", "heuristic"}:
            msg = f"{self.result_class} result cannot claim remote completion"
            raise ValueError(msg)
        if self.result_class not in {"student", "contract_probe"}:
            return self
        expected_profile = (
            "probe" if self.result_class == "contract_probe" else "full"
        )
        if self.execution_profile != expected_profile:
            msg = (
                f"complete {self.result_class} result requires "
                f"execution_profile={expected_profile}"
            )
            raise ValueError(msg)
        digest_fields = (
            "code_sha256",
            "checkpoint_sha256",
            "typed_memory_sha256",
            "inference_manifest_sha256",
            "evidence_sha256",
            "evidence_lineage_sha256",
            "model_sha256",
            "prompt_sha256",
            "predictions_sha256",
            "metrics_sha256",
            "qa_resume_manifest_sha256",
            "run_identity_sha256",
            "finalization_inputs_sha256",
        )
        if missing := tuple(
            name for name in digest_fields if getattr(self, name) is None
        ):
            msg = (
                f"complete {self.result_class} result missing digests: "
                f"{', '.join(missing)}"
            )
            raise ValueError(msg)
        if self.lane != "student":
            msg = f"complete {self.result_class} result requires lane=student"
            raise ValueError(msg)
        if re.fullmatch(r"[0-9a-f]{64}", self.split_id) is None:
            msg = (
                f"complete {self.result_class} result requires a SHA-256 split_id"
            )
            raise ValueError(msg)
        metric_keys = tuple((metric.experiment, metric.name) for metric in self.metrics)
        if len(metric_keys) != len(set(metric_keys)):
            msg = f"complete {self.result_class} result contains duplicate metrics"
            raise ValueError(msg)
        expected_experiment = (
            "PROBE" if self.result_class == "contract_probe" else "E1"
        )
        if any(
            experiment != expected_experiment for experiment, _name in metric_keys
        ):
            msg = (
                f"complete {self.result_class} result may contain only "
                f"{expected_experiment} metrics"
            )
            raise ValueError(msg)
        required_names = {"Ans-F1", "QA-Acc", "QA-MRR"}
        core = {
            metric.name: metric.value
            for metric in self.metrics
            if metric.name in required_names
        }
        if core.keys() != required_names:
            msg = (
                f"complete {self.result_class} result requires "
                "Ans-F1, QA-Acc, and QA-MRR"
            )
            raise ValueError(msg)
        if any(not 0.0 <= value <= METRIC_PERCENT_MAX for value in core.values()):
            msg = (
                f"complete {self.result_class} core metrics must be between 0 and 100"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _require_status_payload(self) -> Self:
        if self.remote_status != "complete" and self.metrics:
            msg = "pending or failed result must not contain metrics"
            raise ValueError(msg)
        if self.remote_status == "failed" and not self.failure_reason:
            msg = "failed result requires failure_reason"
            raise ValueError(msg)
        if self.remote_status == "complete" and not self.metrics:
            msg = "complete result requires metrics"
            raise ValueError(msg)
        if (
            self.remote_status == "complete"
            and self.evidence_lineage_sha256 is None
        ):
            msg = "complete result requires evidence_lineage_sha256"
            raise ValueError(msg)
        return self


def write_report(run_manifest: Path, output: Path) -> None:
    manifest = read_run_manifest(run_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        _ = temporary.write_text(render_report(manifest), encoding="utf-8")
        _ = temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


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
        f"result_class: {_plain(manifest.result_class)}",
        f"experiment_id: {_plain(manifest.experiment_id)}",
        f"execution_profile: {_plain(manifest.execution_profile)}",
        f"lane: {_plain(manifest.lane)}",
        f"split_id: {_plain(manifest.split_id)}",
        f"code_sha256: {_plain(manifest.code_sha256 or 'not-recorded')}",
        (f"checkpoint_sha256: {_plain(manifest.checkpoint_sha256 or 'not-recorded')}"),
        (
            "typed_memory_sha256: "
            f"{_plain(manifest.typed_memory_sha256 or 'not-recorded')}"
        ),
        (
            "inference_manifest_sha256: "
            f"{_plain(manifest.inference_manifest_sha256 or 'not-recorded')}"
        ),
        f"evidence_sha256: {_plain(manifest.evidence_sha256 or 'not-recorded')}",
        (
            "evidence_lineage_sha256: "
            f"{_plain(manifest.evidence_lineage_sha256 or 'not-recorded')}"
        ),
        f"model_sha256: {_plain(manifest.model_sha256 or 'not-recorded')}",
        f"prompt_sha256: {_plain(manifest.prompt_sha256 or 'not-recorded')}",
        f"predictions_sha256: {_plain(manifest.predictions_sha256 or 'not-recorded')}",
        f"metrics_sha256: {_plain(manifest.metrics_sha256 or 'not-recorded')}",
        (
            "qa_resume_manifest_sha256: "
            f"{_plain(manifest.qa_resume_manifest_sha256 or 'not-recorded')}"
        ),
        (
            "run_identity_sha256: "
            f"{_plain(manifest.run_identity_sha256 or 'not-recorded')}"
        ),
        (
            "finalization_inputs_sha256: "
            f"{_plain(manifest.finalization_inputs_sha256 or 'not-recorded')}"
        ),
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
            f"- [{metric.experiment}] {_markdown_text(metric.name)}: {metric.value:.6f}"
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
        line.split("[type=", maxsplit=1)[0].rstrip() if "input_value=" in line else line
        for line in str(exc).splitlines()
        if "For further information" not in line
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
