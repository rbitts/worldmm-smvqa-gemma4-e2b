#!/usr/bin/env bash
set -euo pipefail

: "${WORLDMM_PLAN_OUT:?WORLDMM_PLAN_OUT is required}"
: "${WORLDMM_EXECUTION_PROFILE:?WORLDMM_EXECUTION_PROFILE=teacher-oracle is required}"
: "${WORLDMM_EXPERIMENT_CONFIG:?WORLDMM_EXPERIMENT_CONFIG is required}"
if [ "$WORLDMM_EXECUTION_PROFILE" != "teacher-oracle" ]; then
  printf 'WORLDMM_EXECUTION_PROFILE must be teacher-oracle\n' >&2
  exit 2
fi
# This wrapper only renders an operator plan.  Remote authorization belongs to the
# generated phase scripts and must be command-scoped there.
if [ "${WORLDMM_SMVQA_REMOTE_APPROVED:-0}" = "1" ]; then
  printf 'run_worldmm_smvqa.sh only renders a dry-run plan; do not pass remote approval here\n' >&2
  exit 2
fi


exec worldmm-smvqa launch-remote --dry-run \
  --profile "$WORLDMM_EXECUTION_PROFILE" \
  --experiment-config "$WORLDMM_EXPERIMENT_CONFIG" \
  --config "${WORLDMM_REMOTE_CONFIG:-configs/remote.example.yaml}" \
  --out "$WORLDMM_PLAN_OUT"
