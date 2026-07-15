from __future__ import annotations

import pytest

from worldmm_smvqa.slurm_accounting import (
    SlurmAccountingError,
    decode_accounting,
    preflight_capability,
    require_success,
    sacct_command,
)


def test_sacct_contract_is_cluster_qualified_lossless_and_allocation_only() -> None:
    command = sacct_command(sacct="sacct", cluster="cluster-a", job_id="123")
    assert command == (
        "sacct",
        "-D",
        "-X",
        "-n",
        "-P",
        "--clusters=cluster-a",
        "--jobs=123",
        "--format=JobIDRaw,Cluster,State%64,ExitCode,Restarts,SLUID,OriginalSLUID",
    )


def test_accounting_accepts_only_one_normal_success() -> None:
    record = decode_accounting(
        "123|cluster-a|COMPLETED|0:0|0|1001|1001\n",
        cluster="cluster-a",
        job_id="123",
        expected_sluid="1001",
    )
    assert require_success(record).state == "COMPLETED"


@pytest.mark.parametrize(
    "payload",
    [
        "123|cluster-a|CANCELLED by 1001|0:0|0|1001|1001\n",
        "123|cluster-a|OUT_OF_MEMORY|0:0|0|1001|1001\n",
        "123|cluster-a|NODE_FAIL|0:0|0|1001|1001\n",
        "123|cluster-a|TIMEOUT|0:0|0|1001|1001\n",
        "123|cluster-a|PREEMPTED|0:0|0|1001|1001\n",
        "123|cluster-a|COMPLETED|1:0|0|1001|1001\n",
    ],
)
def test_terminal_non_successes_are_not_success(payload: str) -> None:
    record = decode_accounting(payload, cluster="cluster-a", job_id="123")
    with pytest.raises(
        SlurmAccountingError,
        match="producer did not succeed",
    ):
        _ = require_success(record)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("", "expected exactly one allocation row"),
        (
            "123|cluster-a|COMPLETED+|0:0|0|1001|1001\n",
            "truncated sacct value",
        ),
        (
            "123|cluster-b|COMPLETED|0:0|0|1001|1001\n",
            "accounting federation/cluster mismatch",
        ),
        (
            "123|cluster-a|COMPLETED|0:0|1|1001|1001\n",
            "restarted/requeued allocation",
        ),
        (
            "123|cluster-a|COMPLETED|0:0|0|1001|1002\n",
            "SLUID must equal OriginalSLUID",
        ),
        (
            (
                "123|cluster-a|COMPLETED|0:0|0|1001|1001\n"
                "123|cluster-a|COMPLETED|0:0|0|1001|1001\n"
            ),
            "expected exactly one allocation row",
        ),
        (
            "124|cluster-a|COMPLETED|0:0|0|1001|1001\n",
            "accounting allocation identity mismatch",
        ),
        (
            "123|cluster-a|MYSTERY|0:0|0|1001|1001\n",
            "unknown or non-lossless Slurm state",
        ),
    ],
)
def test_ambiguous_or_untrusted_records_fail_closed(
    payload: str,
    match: str,
) -> None:
    with pytest.raises(SlurmAccountingError, match=match):
        _ = decode_accounting(payload, cluster="cluster-a", job_id="123")


def test_capability_requires_version_and_all_identity_fields() -> None:
    fields = "JobIDRaw Cluster State ExitCode Restarts SLUID OriginalSLUID"
    assert preflight_capability(
        version_output="slurm 23.02.7", helpformat_output=fields
    )
    with pytest.raises(
        SlurmAccountingError,
        match=r"Slurm 23\.2\+ is required",
    ):
        _ = preflight_capability(
            version_output="slurm 22.05.9",
            helpformat_output=fields,
        )
    with pytest.raises(
        SlurmAccountingError,
        match="lacks required accounting fields",
    ):
        _ = preflight_capability(
            version_output="slurm 23.02.7", helpformat_output="JobIDRaw"
        )
