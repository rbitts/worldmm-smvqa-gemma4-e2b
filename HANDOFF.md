# WorldMM-SMVQA EXP-0005 operator runbook

| Field | Value |
| --- | --- |
| Experiment | EXP-0005 teacher-oracle object/location ceiling |
| State | Local contract only; no company command has been authorized or run |
| Page ID | SM-OPERATIONS-HANDOFF |
| Confluence parent | SM-OPERATIONS |
| Canonical generated scripts | `submit_teacher_oracle_preflight.sh`, `submit_teacher_oracle_provider_gate.sh`, `submit_teacher_oracle_downstream.sh`, `run_teacher_oracle_stage.sh` |

This runbook is not approval to use company compute. Every `REPLACE_*` value is fail-closed: leave it unchanged to stop before remote submission. Expensive company CPU/GPU jobs require separate human approval for the exact phase; `WORLDMM_SMVQA_REMOTE_APPROVED=1` is an additional command-scoped guard, not approval itself. Do not manually invoke `sbatch`, alter dependencies, synthesize markers/reports, substitute job IDs, or submit Phase B early.

## Student contract/load profile

This runbook remains authoritative for EXP-0005. The separate `student` profile does
not alter its graph, renderer, approval, or artifact bytes. Before any student remote
operation, review the immutable model-boundary fixture, student architecture, accepted
provider lock, submission manifest, and exact full/probe physical matrix. A pending or
fake provider lock may generate test-only non-runnable plans but cannot authorize
release.

Every student `EvidenceLineage`, QA resume manifest, completion manifest, and report
must agree on `model_contract_sha256`, `student_architecture_sha256`,
`model_load_consensus_payload_sha256`, and
`model_load_consensus_file_sha256`. Missing, stale, or mismatched identity is a no-go
before optimizer construction. A failed attempt is diagnostic only and must be retried
under a new `WORLDMM_RUN_ID`; never reuse or overwrite its output root. Copy back only
approved lightweight metrics, reports, receipts, manifests, diagnostics, redacted logs,
and plots—not datasets, weights, checkpoints, memory stores, predictions, or evidence
packs.
## Memory-alignment candidate boundary

The opt-in `memory` backend, model-neutral v2 envelope, trusted sealed-bundle
evaluator, and render-only comparison plan are separate from EXP-0005 and do
not authorize company execution. Existing qwen/v1 behavior remains the
production-compatible baseline. Candidate construction requires the reviewed
Gemma config and exact v2 contract digest; comparison requires externally
produced trusted sealed baseline/candidate bundles and a fixed cohort.

`worldmm-memory-alignment render-plan` only creates a new no-clobber review
directory with `comparison-plan.json` and `review.md`. It cannot submit, invoke
a scheduler, load a model, or generate evaluation inputs. Bundle production,
remote evaluation, QA attestation, publication, promotion/rollback, and Android
work require separately approved operational plans.

Set the reviewed run identity explicitly; never derive it from the clock:

```bash
export WORLDMM_RUN_ID=REPLACE_WITH_APPROVED_RUN_ID
```

## Fixed EXP-0005 semantics and topology

E0 is the shared semantic **object-presence** control: persisted `object_presence_v1` records contain no geometry, place, entity, or identity field. T0 adds only selected geometry; T1 adds geometry and place. This checked-in configuration literally selects `t1_location_mode=frame_bound_place`: place is only the same-observation object→place assertion, and cross-observation last-location/last-seen/count is forbidden. Thus identity is absent. A separately approved `stable_last_location` configuration must add the identity producer and identity capability/resource exactly once; it must not duplicate it through an environment variable.

Phase A is preflight, then the fixed geometry/semantic/place producers, then `afterany` provider gate, then `afterany` terminal finalizer. Phase B does not exist until the terminal finalizer creates a valid continuation: per variant materialize → retrieve → label-blind QA, then evaluator → finalizer. The names and IDs are exclusively those in generated `operator_contract.json`, `dag_jobs.preflight.env`, `dag_jobs.provider.env`, and `dag_jobs.env`.
Phase-A provider submission must carry the descriptor-verified `PREFLIGHT_JOB_ID` from `dag_jobs.preflight.env`; its manifest also carries one immutable, fsynced expectation digest per producer. The gate rejects any mismatch among that manifest, the producer attempt marker, and the expectation. The preflight seal is an exclusive-create descriptor binding of the measured audit, measured validation, and selected sensor inventory—not a reusable validation receipt. Phase-B authority is the single closed signed binding schema emitted by the gate and verified against the terminal `go` record.

## 1. Local dry-run only

This creates artifacts and performs no SSH, remote shell, or Slurm action:

```bash
export WORLDMM_PLAN_OUT="$PWD/.artifacts/exp-0005-remote-plan"
export WORLDMM_EXECUTION_PROFILE=teacher-oracle
export WORLDMM_EXPERIMENT_CONFIG=configs/spatial/exp_0005_teacher_oracle.example.json
bash scripts/remote/run_worldmm_smvqa.sh
```

Review `$WORLDMM_PLAN_OUT/operator_contract.json`, `$WORLDMM_PLAN_OUT/approval_blockers.json`, and all four canonical scripts. The ordered `operations` rows are authoritative: execute only their `step_id`, `host`, and `argv`; every row carries its prerequisites, artifacts, retry rule, monitoring query, cancellation rule, and copyback commands. Placeholders, absent fields, an unapproved resource, or a contract/script digest mismatch are blockers. Transfer the reviewed generated `remote-plan/` separately with the approved immutable code snapshot; do not deploy into a reused snapshot.

## 2. Immutable environment and CPU preflight

The final remote environment is `$WORLDMM_REMOTE_REPO/.env.worldmm`. Before *every* source, bind its expected SHA-256 outside the file, verify owner/mode/type/non-symlink/digest, then source it. Run this exact shell fragment on the head node; do not source any environment first:

```bash
export WORLDMM_REMOTE_REPO=REPLACE_APPROVED_IMMUTABLE_SNAPSHOT
export WORLDMM_ATTESTED_RUNTIME_ROOT="$WORLDMM_REMOTE_REPO/.venv"
export WORLDMM_REMOTE_ENV_FILE="$WORLDMM_REMOTE_REPO/.env.worldmm"
export EXPECTED_ENV_SHA256=REPLACE_REVIEWED_ENV_SHA256
verify_immutable_env() {
  local f=$1 expected=$2 actual mode uid
  test -f "$f" && test ! -L "$f" || { echo "unsafe env path" >&2; return 1; }
  read -r mode uid < <(stat -c '%a %u' "$f")
  test "$mode" = 600 && test "$uid" = "$(id -u)" || { echo "unsafe env owner/mode" >&2; return 1; }
  actual=$(sha256sum "$f" | cut -d ' ' -f1)
  test "$actual" = "$expected" || { echo "env digest mismatch" >&2; return 1; }
}
verify_immutable_env "$WORLDMM_REMOTE_ENV_FILE" "$EXPECTED_ENV_SHA256"
set -a; source "$WORLDMM_REMOTE_ENV_FILE"; set +a
: "${WORLDMM_RUN_ID:?}" "${WORLDMM_OUTPUT_ROOT:?}" "${WORLDMM_APPROVAL_FILE:?}" \
  "${WORLDMM_EXPERIMENT_CONFIG:?}" "${WORLDMM_RESOURCE_CONFIG_SHA256:?}"
```

The mode-0600 operator-owned env is immutable for both phases and contains no labels, questions, choices, answers, evidence, private key, or key path. It must bind the code snapshot, experiment/config/resource/plan/script digests, sensor/frame/byte/split/policy digests, provider executable/config digests, signer registry digest, output root, accounting cluster and settle policy.

The untracked company-side `$WORLDMM_REMOTE_REPO/.env.worldmm` is mandatory. Verify and use the same unchanged file for both phases.

Run CPU preflight only after its separate approval:

```bash
WORLDMM_SMVQA_REMOTE_APPROVED=1 WORLDMM_DAG_PHASE=preflight \
  bash "$WORLDMM_REMOTE_REPO/remote-plan/submit_teacher_oracle_preflight.sh"
source "$WORLDMM_OUTPUT_ROOT/summary/dag_jobs.preflight.env"
/opt/slurm/bin/squeue -j "$PREFLIGHT_JOB_ID"
/opt/slurm/bin/sacct -D -X -n -P --clusters="$WORLDMM_SLURM_CLUSTER" --jobs="$PREFLIGHT_JOB_ID" --format=JobIDRaw,Cluster,State%64,ExitCode,Restarts,SLUID,OriginalSLUID
```

Wait for the generated preflight receipt and review its sensor/capability/signer/JCS/Ed25519/resolver/StageSpec diagnostics. It must prove selected-inventory RGB/intrinsics coverage, the exact sealed-root mask `RESOLVE_IN_ROOT|RESOLVE_NO_SYMLINKS|RESOLVE_NO_MAGICLINKS|RESOLVE_NO_XDEV` (never `RESOLVE_BENEATH`), and a strict pass. Failure creates no Phase-A approval or provider job.

## 3. Approval schema and signing

Create a canonical RFC 8785 JCS JSON approval using the organization’s authorized Ed25519 signer, domain-separated for its declared purpose. Do not sign pretty JSON or use an alternate serializer/algorithm. The immutable signer registry must identify an allowlisted, unrevoked, time-valid key authorized for `phase_a_approval` or `phase_b_approval` respectively. Preserve the exact JCS payload bytes and detached signature bytes beside the approval; record SHA-256 of each byte sequence in the approval ledger.

Phase-A payload fields are exactly: schema version; purpose; experiment/profile; run ID; output root; code snapshot; plan; generated script; experiment config; resource config; sensor audit; validation receipt; frame assets; provider executable/stage/config; split; policy; byte budget; signer registry; accounting settle policy; configured producer ordered tuple; and their declared digests. Its producer tuple is exactly `geometry,semantic,place` for this frame-bound configuration.

After a Go terminal, make the Phase-B payload with both exact file-byte digests. The Go branch deliberately creates **both** the signed continuation receipt and the `ProviderGateTerminalV1` Go record; they coexist and are separately bound by Phase-B approval:
```bash
sha256sum "$WORLDMM_OUTPUT_ROOT/summary/teacher_oracle_continue.json" \
          "$WORLDMM_OUTPUT_ROOT/summary/teacher_oracle_terminal.json"
```
Set `continue_receipt_sha256` to the first output and `terminal_sha256` to the second, calculated over the files exactly as written (not parsed/reformatted JSON). A missing, non-Go, unsigned, mismatched, or already-consumed continuation or terminal is a Phase-B blocker.

## 4. Phase A provider/gate/terminal

After separate Phase-A approval, repeat the immutable-environment fragment above, export the receipt key only for this command, and submit:

```bash
WORLDMM_SMVQA_REMOTE_APPROVED=1 WORLDMM_DAG_PHASE=provider-gate \
WORLDMM_CONTINUE_RECEIPT_KEY_ID=REPLACE_AUTHORIZED_KEY_ID \
WORLDMM_CONTINUE_RECEIPT_SIGNING_KEY=REPLACE_OPERATOR_OWNED_MODE_0600_NON_SYMLINK_KEY \
  bash "$WORLDMM_REMOTE_REPO/remote-plan/submit_teacher_oracle_provider_gate.sh"
source "$WORLDMM_OUTPUT_ROOT/summary/dag_jobs.provider.env"
/opt/slurm/bin/squeue -j "$PROVIDER_GEOMETRY_JOB_ID,$PROVIDER_SEMANTIC_JOB_ID,$PROVIDER_PLACE_JOB_ID,$PROVIDER_GATE_JOB_ID,$PROVIDER_GATE_TERMINAL_JOB_ID"
/opt/slurm/bin/sacct -D -X -n -P --clusters="$WORLDMM_SLURM_CLUSTER" --jobs="$PROVIDER_GEOMETRY_JOB_ID,$PROVIDER_SEMANTIC_JOB_ID,$PROVIDER_PLACE_JOB_ID,$PROVIDER_GATE_JOB_ID,$PROVIDER_GATE_TERMINAL_JOB_ID" --format=JobIDRaw,Cluster,State%64,ExitCode,Restarts,SLUID,OriginalSLUID
```

The gate’s authoritative accounting query is `/opt/slurm/bin/sacct -D -X -n -P --clusters=<cluster> --jobs=<comma-separated IDs> --format=JobIDRaw,Cluster,State%64,ExitCode,Restarts,SLUID,OriginalSLUID`; it requires exactly one top-level row per expected producer plus `COMPLETED`, `0:0`, owner/read-only non-symlink success marker, rehashed output manifest/files, and exact stage/job/attempt/input/resource/code digests. It alone evaluates the provider contract. `FAILED`, `OUT_OF_MEMORY`, `NODE_FAIL`, `TIMEOUT`, `PREEMPTED`, unknown/nonzero, missing/ambiguous accounting, marker/artifact/digest/attempt mismatch all produce canonical gate failure and terminal early report. `CANCELLED` is `cancelled/not_decidable` only with a matching sealed cancellation intent; otherwise it is failed/not-decidable. Never poll as a fallback or create a replacement gate/report.

On Go, gate and terminal outputs coexist: the gate writes the signed `summary/teacher_oracle_continue.json` and the afterany terminal writes `summary/teacher_oracle_terminal.json` with `provider_gate_decision=go`. Every non-Go result writes only the terminal with the early `provider_gate_terminal_v1` profile; gate crash/OOM/node failure/timeout/cancel still releases terminal through `afterany`. In all non-continuation cases, `dag_jobs.env` and Phase-B IDs must be absent.

To cancel a producer, first atomically seal `CancellationIntentV1` for the exact current stage/job/attempt, then execute the `cancellation` operator row’s exact argv:
```bash
/opt/slurm/bin/scancel "$JOB_ID"
```
Do not cancel gate or terminal. Emergency cancellation of gate/terminal is an owner-approved incident only, ends `cancelled/not_decidable`, creates no continuation, and is never resumed.

## 5. Phase B and terminal recovery

Only after the Phase-B approval and byte-digest checks, repeat immutable env validation and submit:

```bash
WORLDMM_SMVQA_REMOTE_APPROVED=1 WORLDMM_DAG_PHASE=downstream \
WORLDMM_PHASE_B_APPROVAL_FILE=REPLACE_SIGNED_PHASE_B_APPROVAL \
  bash "$WORLDMM_REMOTE_REPO/remote-plan/submit_teacher_oracle_downstream.sh"
source "$WORLDMM_OUTPUT_ROOT/summary/dag_jobs.env"
/opt/slurm/bin/squeue -j "$MATERIALIZE_E0_JOB_ID,$RETRIEVE_E0_JOB_ID,$QA_E0_JOB_ID,$MATERIALIZE_T0_JOB_ID,$RETRIEVE_T0_JOB_ID,$QA_T0_JOB_ID,$MATERIALIZE_T1_JOB_ID,$RETRIEVE_T1_JOB_ID,$QA_T1_JOB_ID,$EVALUATE_JOB_ID,$FINALIZE_JOB_ID"
```

The downstream submitter revalidates approval/continuation/terminal/resource/preflight identity, atomically consumes the continuation once, and only then issues the first `sbatch`. Any check failure means nonzero and zero jobs. A full success produces `oracle_variants_terminal_v1`; it must not be confused with an early terminal report. Retry only as immutable `attempt-N` under the same bindings; resource/config/world-size changes require a new run, approval, and output root.

## 6. Copyback and recovery

For a non-Go/cancel/failure branch, wait for terminal and copy only the early profile:

```bash
rsync -av --files-from=<(printf '%s\n' summary/teacher_oracle_terminal.json summary/dag_jobs.provider.env) \
  "$HEAD_NODE:$WORLDMM_OUTPUT_ROOT/" "./exp-0005-$WORLDMM_RUN_ID-early/"
```

For a full terminal, copy only lightweight reviewed artifacts:

```bash
rsync -av --files-from=<(printf '%s\n' summary/teacher_oracle_terminal.json summary/final_report.md summary/remote_manifest.json summary/dag_jobs.env oracle/E0/metrics.json oracle/T0/metrics.json oracle/T1/metrics.json) \
  "$HEAD_NODE:$WORLDMM_OUTPUT_ROOT/" "./exp-0005-$WORLDMM_RUN_ID-full/"
```

Never copy datasets, frames, labels, raw evidence packs, provider cache/output, weights, or checkpoints. Copyback does not repair a failed run. Record the terminal manifest, exact job states, approval byte digests, continuation/terminal byte digests, and any incident ID; then stop. No remote command is executed by this document by default.

## 7. Unknown submission reconciliation and cancellation

Do not cancel a gate or terminal job. For a failed `sbatch` response, reconcile before any retry, stale-state check, or lock removal; this command queries the exact cluster job-name/comment descriptor and releases the phase lock only after every descriptor proves one job or no job:

```bash
bash "$WORLDMM_REMOTE_REPO/remote-plan/submit_worldmm_smvqa_dag.sh" \
  --reconcile-unknown-sbatch
```

For a producer-only partial submission, the submitter creates one exclusive, fsynced `CancellationIntentV1` for each exact run/stage/job/attempt before calling `scancel`. Preserve `summary/cancellation_intent.*.json` and `summary/dag_submit.*.attempts`; only a matching intent may be recorded as `cancelled/not_decidable`.
## Appendix: deferred legacy student integrity checks

These checks are retained only for a future separately approved student run; they are not part of EXP-0005:

```bash
: "${BASTION_HOST:?set local ProxyJump host}"
: "${HEAD_NODE:?set local Slurm head host}"
env_file="$WORLDMM_REMOTE_REPO/.env.worldmm"
test "$(stat -c %a "$env_file")" = 600
export SMVQA_FRAME_ROOT="$WORLDMM_OUTPUT_ROOT/inference_inputs/frames"
test -s "$WORLDMM_SENSOR_FRAME_MANIFEST"
spatial_infer_clean() {
  env -u WORLDMM_SPATIAL_INFER_EXE "$@"
}
```

For that deferred lane, review `env_contract.json.effective_teacher_resources`. These operational fields are not validated by comparing them to `run_identity.json`; verify them against the terminal manifest and final report.
