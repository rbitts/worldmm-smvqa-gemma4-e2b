from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pydantic import TypeAdapter

from worldmm_smvqa.remote_plan import ExpectedOutputs

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ENV_NAMES = frozenset(
    {
        "SMVQA_DATA_ROOT",
        "GEMMA_MODEL_PATH",
        "WORLDMM_OUTPUT_ROOT",
        "BASTION_HOST",
        "HEAD_NODE",
    },
)
REQUIRED_STAGES = (
    "fetch Gemma 4 E2B model",
    "prepare source manifests",
    "build 30s/30m chunks",
    "generate/load captions OCR object frame refs",
    "build WorldMM stores with the Qwen memory constructor",
    "retrieve all QA under causal cutoff",
    "run Gemma 4 E2B QA",
    "evaluate official metrics",
    "run ablation lanes",
    "write summary",
)


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in REMOTE_ENV_NAMES | {
        "WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST",
        "WORLDMM_SMVQA_REMOTE_APPROVED",
    }:
        _ = env.pop(name, None)
    return subprocess.run(
        ["uv", "run", "worldmm-smvqa", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launch_remote_dry_run_writes_full_plan_contract(  # noqa: PLR0915
    tmp_path: Path,
) -> None:
    # Given: a remote full benchmark plan output directory.
    out_dir = tmp_path / "remote_plan"

    # When: launch-remote is run as a local dry-run.
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    # Then: deterministic artifacts describe every remote stage and copyback rule.
    assert result.returncode == 0, result.stderr
    script = out_dir / "run_worldmm_smvqa.sh"
    expected = out_dir / "expected_outputs.json"
    policy = out_dir / "copyback_policy.txt"
    assert script.is_file()
    assert expected.is_file()
    assert policy.is_file()

    script_text = script.read_text(encoding="utf-8")
    for stage in REQUIRED_STAGES:
        assert stage in script_text
    assert "ssh " not in script_text
    assert "#SBATCH --partition=gpu-vtt-queue" in script_text
    assert "#SBATCH --nodes=10" in script_text
    assert "#SBATCH --gpus-per-node=8" in script_text
    assert "/opt/slurm/bin/srun" in script_text
    assert "python -m torch.distributed.run" in script_text
    assert 'cd "$WORLDMM_REMOTE_REPO"' in script_text
    assert 'source "$WORLDMM_REMOTE_REPO/.venv/bin/activate"' in script_text
    assert 'hf download "$WORLDMM_MODEL_ID" --local-dir "$GEMMA_MODEL_PATH"' in (
        script_text
    )
    assert "$WORLDMM_OUTPUT_ROOT" in script_text
    assert "$SMVQA_DATA_ROOT" in script_text
    assert "$GEMMA_MODEL_PATH" in script_text
    assert 'run_distributed_memory "episodic"' in script_text
    assert 'run_distributed_memory "semantic,visual"' in script_text
    assert "worldmm-smvqa build-memory --stores spatial" in script_text
    assert "WORLDMM_MEMORY_SHARD_TIMEOUT_SECONDS" in script_text
    _assert_qwen_memory_contract(script_text)
    assert '--input "$WORLDMM_OUTPUT_ROOT/memory/memory_manifest.json"' in script_text
    assert "worldmm-smvqa retrieve-batch" in script_text
    assert "while IFS= read -r question_id" not in script_text
    assert "--stores episodic,semantic,visual,spatial" in script_text
    assert "--retrieval-protocol worldmm-smvqa" in script_text
    assert "--max-frame-refs 32" in script_text
    assert 'run_ablation_lane "without_spatial"' in script_text
    assert "--stores episodic,semantic,visual" in script_text
    assert 'run_ablation_lane "protocol_legacy_round_robin"' in script_text
    assert '--retrieval-protocol "$protocol"' in script_text
    assert '  "legacy-round-robin"' in script_text

    manifest = TypeAdapter(ExpectedOutputs).validate_json(
        expected.read_text(encoding="utf-8"),
    )
    assert manifest["remote_job_reference"] == "slurm-${SLURM_JOB_ID}"
    assert manifest["metrics"] == ["Ans-F1", "QA-Acc", "QA-MRR"]
    assert manifest["outputs"]["spatial_memory"] == (
        "$WORLDMM_OUTPUT_ROOT/memory/worldmm_sv/spatial.jsonl"
    )
    assert manifest["outputs"]["retrieval_trace_evidence_packs"] == (
        "$WORLDMM_OUTPUT_ROOT/retrieval/evidence_packs.jsonl"
    )
    assert manifest["outputs"]["spatial_diagnostics"] == (
        "$WORLDMM_OUTPUT_ROOT/diagnostics/spatial_diagnostics.json"
    )
    assert manifest["outputs"]["ablation_without_spatial"] == (
        "$WORLDMM_OUTPUT_ROOT/ablation/without_spatial/"
        "metrics/official_metrics.json"
    )
    assert manifest["outputs"]["ablation_protocol_legacy"] == (
        "$WORLDMM_OUTPUT_ROOT/ablation/protocol_legacy_round_robin/"
        "metrics/official_metrics.json"
    )
    assert manifest["outputs"]["slurm_stdout"] == (
        "$WORLDMM_OUTPUT_ROOT/logs/slurm-${SLURM_JOB_ID}.out"
    )
    assert manifest["outputs"]["slurm_job_id"] == (
        "$WORLDMM_OUTPUT_ROOT/summary/slurm_job_id.txt"
    )
    for value in manifest["outputs"].values():
        assert isinstance(value, str)
        assert value.startswith("$WORLDMM_OUTPUT_ROOT/")

    policy_text = policy.read_text(encoding="utf-8")
    assert "metrics/logs/plots/summaries/small samples" in policy_text
    assert "no full datasets" in policy_text
    assert "no model weights" in policy_text
    assert "no checkpoints" in policy_text


def _assert_qwen_memory_contract(script_text: str) -> None:
    assert "--backend qwen" in script_text
    assert "WORLDMM_MEMORY_MODEL_ID:=Qwen/Qwen3-VL-8B-Instruct" in script_text
    assert (
        "WORLDMM_MEMORY_MODEL_PATH:="
        "/repo/VTteam/bongh.park/outputs/models/qwen3-vl"
    ) in script_text
    assert 'hf download "$WORLDMM_MEMORY_MODEL_ID" --local-dir' in script_text
    assert "WORLDMM_VISUAL_ENCODER_ID" not in script_text
    assert "WORLDMM_VISUAL_ENCODER_PATH" not in script_text
    assert "vlm2vec" not in script_text.lower()
    assert 'memory_args+=(--input "$WORLDMM_MEMORY_INPUT")' in script_text
    assert '  "$WORLDMM_OUTPUT_ROOT/memory/episodic.jsonl"' in script_text


def test_launch_remote_uses_slurm_rendezvous_and_per_node_rank(
    tmp_path: Path,
) -> None:
    # Given: a generated company Slurm workload.
    out_dir = tmp_path / "remote_plan"
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )
    assert result.returncode == 0, result.stderr
    script_text = (out_dir / "run_worldmm_smvqa.sh").read_text(encoding="utf-8")

    # When: shell syntax is checked without starting Slurm.
    proof = subprocess.run(
        ["bash", "-n", str(out_dir / "run_worldmm_smvqa.sh")],
        text=True,
        capture_output=True,
        check=False,
    )

    # Then: each Slurm node starts one torch.distributed.run agent.
    assert proof.returncode == 0, proof.stderr
    assert '/opt/slurm/bin/scontrol show hostnames "$SLURM_JOB_NODELIST"' in (
        script_text
    )
    assert 'MASTER_ADDR="${worldmm_hosts[0]}"' in script_text
    assert 'MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"' in script_text
    assert '--ntasks="$WORLDMM_REMOTE_NODES"' in script_text
    assert '--nproc-per-node "$WORLDMM_GPUS_PER_NODE"' in script_text
    assert '--node-rank "$SLURM_NODEID"' in script_text
    assert '--master-addr "$MASTER_ADDR"' in script_text
    assert '--master-port "$MASTER_PORT"' in script_text
    assert "\neval " not in script_text


def test_launch_remote_submit_requires_explicit_env_approval(tmp_path: Path) -> None:
    # Given: submit was requested without the required approval env value.
    out_dir = tmp_path / "remote_plan"

    # When: launch-remote tries to submit.
    result = run_cli(
        "launch-remote",
        "--submit",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    # Then: the typed approval guard fails before any plan or remote process starts.
    assert result.returncode != 0
    assert "ExplicitApprovalRequired" in result.stderr
    assert not out_dir.exists()
    combined = f"{result.stdout}\n{result.stderr}"
    assert "ssh " not in combined
    assert "REMOTE_JOB_LAUNCHER" not in combined


def test_launch_remote_dry_run_replaces_stale_artifacts(tmp_path: Path) -> None:
    # Given: a stale artifact from an interrupted earlier run.
    out_dir = tmp_path / "remote_plan"
    out_dir.mkdir()
    stale_script = out_dir / "run_worldmm_smvqa.sh"
    _ = stale_script.write_text("stale\n", encoding="utf-8")

    # When: the dry-run is repeated.
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        "configs/remote.example.yaml",
        "--out",
        str(out_dir),
    )

    # Then: generated files are replaced by current deterministic content.
    assert result.returncode == 0, result.stderr
    assert stale_script.read_text(encoding="utf-8") != "stale\n"
    script_text = stale_script.read_text(encoding="utf-8")
    assert 'run_distributed_memory "semantic,visual"' in script_text
    assert "build-memory --stores spatial" in script_text


def test_launch_remote_config_requires_remote_placeholders(tmp_path: Path) -> None:
    # Given: a malformed remote config missing WORLDMM_OUTPUT_ROOT.
    config = tmp_path / "missing-output-root.yaml"
    _ = config.write_text(
        """runtime:
  location: remote
remote:
  bastion_host: ${BASTION_HOST}
  head_node: ${HEAD_NODE}
  data_root: ${SMVQA_DATA_ROOT}
  model_path: ${GEMMA_MODEL_PATH}
""",
        encoding="utf-8",
    )

    # When: a dry-run plan is requested.
    result = run_cli(
        "launch-remote",
        "--dry-run",
        "--config",
        str(config),
        "--out",
        str(tmp_path / "remote_plan"),
    )

    # Then: the boundary reports a typed missing remote config error.
    assert result.returncode != 0
    assert "MissingRemoteConfig: WORLDMM_OUTPUT_ROOT" in result.stderr
