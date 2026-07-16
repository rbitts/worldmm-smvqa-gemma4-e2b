from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from worldmm_smvqa.memory_alignment_plan import (
    PlanRenderError,
    render_comparison_plan,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    contract = root / "configs" / "spatial" / "model_boundary_contract_v2.json"
    contract.parent.mkdir(parents=True)
    contract_bytes = b'{"schema_version":"model-boundary-contract-v2"}\n'
    contract.write_bytes(contract_bytes)
    digest = _sha(contract_bytes)
    config = root / "memory.yaml"
    config.write_text(
        "schema_version: memory-alignment-config-v1\n"
        "runtime:\n"
        "  location: remote\n"
        "memory_alignment:\n"
        "  backend: memory\n"
        "  artifact_role: memory_builder\n"
        "  model_family: gemma\n"
        "  model_variant: Gemma-4-E2B-IT\n"
        "  model_path: ${WORLDMM_MEMORY_MODEL_PATH}\n"
        "  contract_id: worldmm-smvqa-memory-v2\n"
        "  contract_path: configs/spatial/model_boundary_contract_v2.json\n"
        f"  contract_sha256: {digest}\n",
        encoding="utf-8",
    )
    baseline = root / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "schema_version": "sealed-memory-bundle-v1",
                "role": "baseline",
                "contract_selection": {
                    "schema_version": "contract-selection-v1",
                    "version": "v1",
                    "contract_id": "worldmm-smvqa-local-boundaries-v1",
                    "contract_path": "configs/spatial/model_boundary_contract_v1.json",
                    "expected_contract_file_sha256": "a" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    candidate = root / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "schema_version": "sealed-memory-bundle-v1",
                "role": "candidate",
                "contract_selection": {
                    "schema_version": "contract-selection-v1",
                    "version": "v2",
                    "contract_id": "worldmm-smvqa-memory-v2",
                    "contract_path": "configs/spatial/model_boundary_contract_v2.json",
                    "expected_contract_file_sha256": digest,
                },
            }
        ),
        encoding="utf-8",
    )
    cohort = root / "cohort.json"
    cohort.write_text(
        json.dumps({"schema_version": "memory-comparison-cohort-v1"}),
        encoding="utf-8",
    )
    return config, baseline, candidate, cohort


def test_render_plan_is_deterministic_non_executable_and_review_only(
    tmp_path: Path,
) -> None:
    config, baseline, candidate, cohort = _write_inputs(tmp_path)

    rendered = render_comparison_plan(
        config=config,
        repository_root=tmp_path,
        baseline_manifest=baseline,
        candidate_manifest=candidate,
        cohort=cohort,
        out=tmp_path / "out",
    )

    plan = json.loads(rendered.plan_path.read_text(encoding="utf-8"))
    assert plan["schema_version"] == "memory-alignment-plan-v1"
    assert plan["submission"] is False
    assert [row["comparison_id"] for row in plan["comparisons"]] == [
        "visual_primary",
        "episodic_primary",
        "semantic_primary",
        "semantic_rebuild",
    ]
    assert plan["scientific_protocol"] == {
        "confidence_interval": "paired_bootstrap_95_percentile",
        "confidence_interval_indexes": ["249", "9749"],
        "decision": "pass only when every CI lower bound is at least -0.05",
        "k": "6",
        "non_inferiority_threshold": "-0.05",
        "paired_bootstrap_replicates": "10000",
        "protocol_id": "memory-recall6-paired-bootstrap-v1",
    }
    assert set(plan["inputs"]) == {
        "config_sha256",
        "baseline_manifest_sha256",
        "candidate_manifest_sha256",
        "cohort_sha256",
    }
    forbidden_keys = {
        "command",
        "host",
        "url",
        "environment",
        "secret",
        "scheduler",
        "submit_command",
        "execute",
    }
    assert not forbidden_keys & _all_keys(plan)
    review = rendered.review_path.read_text(encoding="utf-8")
    assert [line for line in review.splitlines() if line.startswith("# ")] == [
        "# Status",
        "# Inputs",
        "# Comparisons",
        "# Scientific protocol",
        "# Blockers",
        "# Deferred",
    ]
    assert review.endswith("Submission supported: no\n")


def test_render_plan_refuses_to_clobber_existing_output(tmp_path: Path) -> None:
    config, baseline, candidate, cohort = _write_inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "sentinel"
    sentinel.write_text("owned", encoding="utf-8")

    with pytest.raises(PlanRenderError, match="output already exists"):
        render_comparison_plan(
            config=config,
            repository_root=tmp_path,
            baseline_manifest=baseline,
            candidate_manifest=candidate,
            cohort=cohort,
            out=out,
        )

    assert sentinel.read_text(encoding="utf-8") == "owned"
    assert list(out.iterdir()) == [sentinel]


def test_render_plan_validates_before_creating_output(tmp_path: Path) -> None:
    config, baseline, candidate, cohort = _write_inputs(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "contract_sha256: ", "contract_sha256: 0"
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    with pytest.raises(PlanRenderError, match="digest"):
        render_comparison_plan(
            config=config,
            repository_root=tmp_path,
            baseline_manifest=baseline,
            candidate_manifest=candidate,
            cohort=cohort,
            out=out,
        )

    assert not out.exists()


def test_render_plan_rejects_contract_path_escape(tmp_path: Path) -> None:
    config, baseline, candidate, cohort = _write_inputs(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "configs/spatial/model_boundary_contract_v2.json", "../contract.json"
        ),
        encoding="utf-8",
    )

    with pytest.raises(PlanRenderError, match="repository-relative"):
        render_comparison_plan(
            config=config,
            repository_root=tmp_path,
            baseline_manifest=baseline,
            candidate_manifest=candidate,
            cohort=cohort,
            out=tmp_path / "out",
        )


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key).lower() for key in value} | set().union(
            *(_all_keys(child) for child in value.values()),
            set(),
        )
    if isinstance(value, list):
        return set().union(*(_all_keys(child) for child in value), set())
    return set()
