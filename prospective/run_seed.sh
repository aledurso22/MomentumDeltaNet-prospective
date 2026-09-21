#!/usr/bin/env bash
# One seed, all six arms in lockstep, on one GPU. Run one of these per seed.
#
#   EXPECTED_COMMIT=<sha> CUDA_VISIBLE_DEVICES=0 ./prospective/run_seed.sh 501
set -euo pipefail

SEED="${1:?usage: run_seed.sh <seed>}"
OUT_ROOT="${OUT_ROOT:-$HOME/mdn-bridge}"
STEPS="${STEPS:-4000}"

cd "$(dirname "$0")/.."
HEAD_SHA="$(git rev-parse HEAD)"
if [ -n "${EXPECTED_COMMIT:-}" ] && [ "$HEAD_SHA" != "$EXPECTED_COMMIT" ]; then
  echo "refusing: HEAD is $HEAD_SHA, expected $EXPECTED_COMMIT" >&2
  exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing: worktree is dirty" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$OUT_ROOT/$STAMP-seed$SEED"
mkdir -p "$RUN_DIR"

echo "commit : $HEAD_SHA"
echo "seed   : $SEED"
echo "gpus   : ${CUDA_VISIBLE_DEVICES:-all}"
echo "out    : $RUN_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

python -u -m prospective.train \
  --seed "$SEED" --steps "$STEPS" \
  --out "$RUN_DIR/results.json" 2>&1 | tee "$RUN_DIR/train.log"
