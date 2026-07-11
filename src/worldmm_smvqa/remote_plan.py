from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict, override

from worldmm_smvqa.config import AppConfig, MissingRemoteConfigError
from worldmm_smvqa.remote_script import (
    dag_stage_script_text,
    dag_submit_script_text,
    script_text,
)

APPROVAL_ENV: Final = "WORLDMM_SMVQA_REMOTE_APPROVED"
REMOTE_SCRIPT_NAME: Final = "run_worldmm_smvqa.sh"
DAG_SUBMIT_SCRIPT_NAME: Final = "submit_worldmm_smvqa_dag.sh"
DAG_STAGE_SCRIPT_NAME: Final = "run_worldmm_smvqa_stage.sh"
EXPECTED_OUTPUTS_NAME: Final = "expected_outputs.json"
COPYBACK_POLICY_NAME: Final = "copyback_policy.txt"
DEFAULT_REMOTE_REPO: Final = "/repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b"

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
    dag_submit_script = out_dir / DAG_SUBMIT_SCRIPT_NAME
    dag_stage_script = out_dir / DAG_STAGE_SCRIPT_NAME
    expected_outputs = out_dir / EXPECTED_OUTPUTS_NAME
    copyback_policy = out_dir / COPYBACK_POLICY_NAME

    _ = script.write_text(script_text(), encoding="utf-8")
    script.chmod(0o755)
    _ = dag_submit_script.write_text(dag_submit_script_text(), encoding="utf-8")
    dag_submit_script.chmod(0o755)
    _ = dag_stage_script.write_text(dag_stage_script_text(), encoding="utf-8")
    dag_stage_script.chmod(0o755)
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
    remote_repo_ref = f"${{WORLDMM_REMOTE_REPO:-{DEFAULT_REMOTE_REPO}}}"
    remote_plan_dir = f"{remote_repo_ref}/remote-plan"
    sync_repo = (
        'rsync -az -e "ssh -J $BASTION_HOST" '
        "--exclude .git --exclude .venv --exclude .omo --exclude '.env*' ./ "
        f'"$HEAD_NODE:{remote_repo_ref}/"'
    )
    sync_plan = (
        f'rsync -az -e "ssh -J $BASTION_HOST" --exclude \'.env*\' '
        f'{plan_dir}/ '
        f'"$HEAD_NODE:{remote_plan_dir}/"'
    )
    remote_repo = f'"${{WORLDMM_REMOTE_REPO:-{DEFAULT_REMOTE_REPO}}}"'
    legacy_remote_shell_command = (
        f"cd {remote_repo} && "
        "mkdir -p remote-plan/logs && "
        f"/opt/slurm/bin/sbatch --parsable remote-plan/{REMOTE_SCRIPT_NAME}"
    )
    legacy_remote_command = (
        f'ssh -J "$BASTION_HOST" "$HEAD_NODE" '
        f"{shlex.quote(legacy_remote_shell_command)}"
    )
    remote_shell_command = (
        f"cd {remote_repo} && "
        "mkdir -p remote-plan/logs && "
        f"{f'{APPROVAL_ENV}=1 ' if result.mode == 'approved-plan-only' else ''}"
        f"bash remote-plan/{DAG_SUBMIT_SCRIPT_NAME}"
    )
    remote_command = (
        f'ssh -J "$BASTION_HOST" "$HEAD_NODE" {shlex.quote(remote_shell_command)}'
    )
    return (
        f"remote plan mode={result.mode}\n"
        f"wrote {result.script}\n"
        f"wrote {result.expected_outputs}\n"
        f"wrote {result.copyback_policy}\n"
        f"wrote {result.script.parent / DAG_SUBMIT_SCRIPT_NAME}\n"
        f"wrote {result.script.parent / DAG_STAGE_SCRIPT_NAME}\n"
        "# dry-run/plan only; no ssh, remote shell, or job submission opened locally\n"
        f"# legacy single-job compatibility: {legacy_remote_command}\n"
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
        "remote_job_reference": (
            "$WORLDMM_OUTPUT_ROOT/summary/dag_jobs.env#REPORT_JOB_ID"
        ),
        "metrics": ["Ans-F1", "QA-Acc", "QA-MRR"],
        "outputs": {
            "sensor_frame_manifest": (
                "$WORLDMM_OUTPUT_ROOT/manifests/sensor_frames.jsonl"
            ),
            "chunk_manifest": (
                "$WORLDMM_OUTPUT_ROOT/manifests/source_chunks.jsonl"
            ),
            "source_memories": (
                "$WORLDMM_OUTPUT_ROOT/manifests/source_memories.jsonl"
            ),
            "retrieval_evidence": (
                "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl"
            ),
            "predictions": "$WORLDMM_OUTPUT_ROOT/qa/predictions.jsonl",
            "metrics": "$WORLDMM_OUTPUT_ROOT/metrics/official_metrics.json",
            "stage_logs": "$WORLDMM_OUTPUT_ROOT/logs/*-*.out",
            "summary": "$WORLDMM_OUTPUT_ROOT/summary/summary.txt",
            "dag_job_manifest": "$WORLDMM_OUTPUT_ROOT/summary/dag_jobs.env",
            "preflight_report": "$WORLDMM_OUTPUT_ROOT/diagnostics/preflight.json",
            "teacher_cache_report": (
                "$WORLDMM_OUTPUT_ROOT/diagnostics/teacher_cache.json"
            ),
            "teacher_cache": "$WORLDMM_OUTPUT_ROOT/teacher/cache.jsonl",
            "student_teacher_cache": (
                "$WORLDMM_OUTPUT_ROOT/training/student_teacher_cache.jsonl"
            ),
            "utility_labels": (
                "$WORLDMM_OUTPUT_ROOT/training/selector_rows.jsonl"
            ),
            "spatial_selector": "$WORLDMM_OUTPUT_ROOT/training/selector.json",
            "spatial_checkpoint": (
                "$WORLDMM_OUTPUT_ROOT/checkpoints/spatial_student.pt"
            ),
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
