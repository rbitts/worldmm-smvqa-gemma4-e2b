#!/usr/bin/env bash
set -euo pipefail

: "${WORLDMM_PLAN_OUT:?WORLDMM_PLAN_OUT is required}"

worldmm-smvqa launch-remote --dry-run \
  --config "${WORLDMM_REMOTE_CONFIG:-configs/remote.example.yaml}" \
  --out "$WORLDMM_PLAN_OUT"
