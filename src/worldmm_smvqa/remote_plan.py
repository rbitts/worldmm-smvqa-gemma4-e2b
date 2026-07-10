from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict, override

from worldmm_smvqa.config import AppConfig, MissingRemoteConfigError
from worldmm_smvqa.remote_script import script_text

APPROVAL_ENV: Final = "WORLDMM_SMVQA_REMOTE_APPROVED"
REMOTE_SCRIPT_NAME: Final = "run_worldmm_smvqa.sh"
EXPECTED_OUTPUTS_NAME: Final = "expected_outputs.json"
COPYBACK_POLICY_NAME: Final = "copyback_policy.txt"

REMOTE_FIELDS: Final = {
    "bastion_host": "BASTION_HOST",
    "head_node": "HEAD_NODE",
    "data_root": "SMVQA_DATA_ROOT",
    "model_path": "GEMMA_MODEL_PATH",
    "output_root": "WORLDMM_OUTPUT_ROOT",
}

class ExpectedOutputs(TypedDict):
    remote_job_reference: str
    metrics: list[str]
    outputs: dict[str, str]
    copyback_allowed: list[str]
    copyback_forbidden: list[str]


SubmitMode = Literal["dry-run", "approved-plan-only"]


@dataclass(frozen=True, slots=True)
class ExplicitApprovalRequiredError(Exception):
    flag: str

    @override
    def __str__(self) -> str:
        return f"ExplicitApprovalRequired: set {self.flag}=1 to use --submit"


@dataclass(frozen=True, slots=True)
class RemotePlanResult:
    script: Path
    expected_outputs: Path
    copyback_policy: Path
    mode: SubmitMode


def write_remote_plan(
    config: AppConfig,
    out_dir: Path,
    env: Mapping[str, str],
    *,
    submit: bool,
) -> RemotePlanResult:
    if submit and env.get(APPROVAL_ENV) != "1":
        raise ExplicitApprovalRequiredError(flag=APPROVAL_ENV)

    _require_remote_placeholders(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    script = out_dir / REMOTE_SCRIPT_NAME
    expected_outputs = out_dir / EXPECTED_OUTPUTS_NAME
    copyback_policy = out_dir / COPYBACK_POLICY_NAME

    _ = script.write_text(script_text(), encoding="utf-8")
    script.chmod(0o755)
    _ = expected_outputs.write_text(
        json.dumps(_expected_outputs(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _ = copyback_policy.write_text(_copyback_policy_text(), encoding="utf-8")

    mode: SubmitMode = "approved-plan-only" if submit else "dry-run"
    return RemotePlanResult(
        script=script,
        expected_outputs=expected_outputs,
        copyback_policy=copyback_policy,
        mode=mode,
    )


def plan_stdout(result: RemotePlanResult) -> str:
    plan_dir = shlex.quote(str(result.script.parent))
    remote_plan_dir = "$WORLDMM_REMOTE_REPO/remote-plan"
    sync_repo = (
        'rsync -az -e "ssh -J $BASTION_HOST" '
        "--exclude .git --exclude .venv --exclude .omo ./ "
        '"$HEAD_NODE:$WORLDMM_REMOTE_REPO/"'
    )
    sync_plan = (
        f'rsync -az -e "ssh -J $BASTION_HOST" {plan_dir}/ '
        f'"$HEAD_NODE:{remote_plan_dir}/"'
    )
    remote_shell_command = (
        'cd "$WORLDMM_REMOTE_REPO" && '
        "mkdir -p remote-plan/logs && "
        f"/opt/slurm/bin/sbatch --parsable remote-plan/{REMOTE_SCRIPT_NAME}"
    )
    remote_command = (
        'ssh -J "$BASTION_HOST" "$HEAD_NODE" '
        f"{shlex.quote(remote_shell_command)}"
    )
    return (
        f"remote plan mode={result.mode}\n"
        f"wrote {result.script}\n"
        f"wrote {result.expected_outputs}\n"
        f"wrote {result.copyback_policy}\n"
        "# dry-run/plan only; no ssh, remote shell, or job submission opened locally\n"
        f"{sync_repo}\n"
        f"{sync_plan}\n"
        f"{remote_command}\n"
    )


def _require_remote_placeholders(config: AppConfig) -> None:
    remote = config.values.get("remote", {})
    for key, env_name in REMOTE_FIELDS.items():
        if remote.get(key) != f"${{{env_name}}}":
            raise MissingRemoteConfigError(name=env_name)


def _expected_outputs() -> ExpectedOutputs:
    return {
        "remote_job_reference": "slurm-${SLURM_JOB_ID}",
        "metrics": ["Ans-F1", "QA-Acc", "QA-MRR"],
        "outputs": {
            "source_manifest": "$WORLDMM_OUTPUT_ROOT/manifests/source_roots.txt",
            "question_manifest": "$WORLDMM_OUTPUT_ROOT/manifests/question_ids.txt",
            "chunk_manifest": "$WORLDMM_OUTPUT_ROOT/chunks/source_chunks.jsonl",
            "caption_ocr_object_frame_refs": (
                "$WORLDMM_OUTPUT_ROOT/source_refs/source_memories.jsonl"
            ),
            "episodic_memory": "$WORLDMM_OUTPUT_ROOT/memory/episodic.jsonl",
            "semantic_memory": "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv/semantic.jsonl",
            "visual_memory": "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv/visual.jsonl",
            "spatial_memory": "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv/spatial.jsonl",
            "retrieval_evidence": "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl",
            "retrieval_trace_evidence_packs": (
                "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl"
            ),
            "predictions": "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl",
            "metrics": "$WORLDMM_OUTPUT_ROOT/metrics/official_metrics.json",
            "spatial_diagnostics": (
                "$WORLDMM_OUTPUT_ROOT/diagnostics/spatial_diagnostics.json"
            ),
            "ablation_without_spatial": (
                "$WORLDMM_OUTPUT_ROOT/ablation/without_spatial/"
                "metrics/official_metrics.json"
            ),
            "ablation_protocol_legacy": (
                "$WORLDMM_OUTPUT_ROOT/ablation/protocol_legacy_round_robin/"
                "metrics/official_metrics.json"
            ),
            "ablation_without_spatial_predictions": (
                "$WORLDMM_OUTPUT_ROOT/ablation/without_spatial/"
                "qa/predictions.jsonl"
            ),
            "ablation_protocol_legacy_predictions": (
                "$WORLDMM_OUTPUT_ROOT/ablation/protocol_legacy_round_robin/"
                "qa/predictions.jsonl"
            ),
            "slurm_stdout": (
                "$WORLDMM_OUTPUT_ROOT/logs/slurm-${SLURM_JOB_ID}.out"
            ),
            "slurm_stderr": (
                "$WORLDMM_OUTPUT_ROOT/logs/slurm-${SLURM_JOB_ID}.err"
            ),
            "memory_manifest": "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json",
            "job_metadata": "$WORLDMM_OUTPUT_ROOT/summary/job.json",
            "slurm_job_id": "$WORLDMM_OUTPUT_ROOT/summary/slurm_job_id.txt",
            "summary": "$WORLDMM_OUTPUT_ROOT/summary/summary.txt",
            "remote_manifest": (
                "$WORLDMM_OUTPUT_ROOT/summary/remote_manifest.json"
            ),
            "final_report": "$WORLDMM_OUTPUT_ROOT/summary/final_report.md",
        },
        "copyback_allowed": [
            "metrics",
            "logs",
            "plots",
            "summaries",
            "small sample outputs",
        ],
        "copyback_forbidden": [
            "full datasets",
            "model weights",
            "checkpoints",
            "sensitive artifacts",
        ],
    }


def _copyback_policy_text() -> str:
    return (
        "Only metrics/logs/plots/summaries/small samples may be copied locally.\n"
        "Forbidden copyback: no full datasets, no model weights, no checkpoints, "
        "no sensitive artifacts.\n"
        "Store full benchmark data, model artifacts, checkpoints, and logs on "
        "approved company storage under $WORLDMM_OUTPUT_ROOT.\n"
    )
