#!/usr/bin/env bash
# One Slurm job, several GPUs, one seed per GPU as a DIRECT CHILD PROCESS.
#
# This is the S5 wave's pattern, and it exists because separate co-resident
# jobs on this node do not survive each other: submitting three sbatch jobs to
# pgi15-gpu5 ended with the first COMPLETED and the other two FAILED at the
# same elapsed second, when node cleanup for the finishing job took its
# neighbours with it. No sbatch, no salloc, no srun below this line.
#
#   SEEDS="501 502 503 504" bash prospective/run_wave.sh -- <extra train.py flags>
#
# Resumable: with RUN_TAG set, each seed writes to $OUT_ROOT/$RUN_TAG-seed<s>
# (a stable directory), --resume is passed, and train.log is appended to, so
# resubmitting the same command continues from each seed's checkpoint.
# STEPS, when set, is passed as --steps (flags after -- override it).
#
# v2 writes to mdn-bridge-runs-v2 so v1 (reduced-mixer) results stay separate;
# train.py defaults ARE the production config, so no task flags are needed.
set -euo pipefail

SEEDS="${SEEDS:-501 502 503}"
STEPS="${STEPS:-}"
OUT_ROOT="${OUT_ROOT:-$HOME/mdn-bridge-runs-v2}"
[ "${1:-}" = "--" ] && shift

cd "$(dirname "$0")/.."
HEAD_SHA="$(git rev-parse HEAD)"
if [ -n "${EXPECTED_COMMIT:-}" ] && [ "$HEAD_SHA" != "$EXPECTED_COMMIT" ]; then
  echo "refusing: HEAD is $HEAD_SHA, expected $EXPECTED_COMMIT" >&2; exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing: worktree is dirty" >&2; exit 1
fi

# de-duplicated GPU tokens actually visible to this job
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a TOKENS <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t TOKENS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
read -r -a SEED_ARR <<< "$SEEDS"
if [ "${#SEED_ARR[@]}" -gt "${#TOKENS[@]}" ]; then
  echo "refusing: ${#SEED_ARR[@]} seeds but only ${#TOKENS[@]} GPU tokens" >&2
  exit 1
fi

STAMP="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RESUME=""
if [ -n "${RUN_TAG:-}" ]; then RESUME="--resume"; fi
echo "commit : $HEAD_SHA"
echo "seeds  : ${SEED_ARR[*]}"
echo "tokens : ${TOKENS[*]}"
echo "out    : $OUT_ROOT/$STAMP-*"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

pids=()
for i in "${!SEED_ARR[@]}"; do
  seed="${SEED_ARR[$i]}"
  token="${TOKENS[$i]}"
  run_dir="$OUT_ROOT/$STAMP-seed$seed"
  mkdir -p "$run_dir"
  echo "launching seed $seed on gpu token $token -> $run_dir"
  CUDA_VISIBLE_DEVICES="$token" \
    python -u -m prospective.train --seed "$seed" ${STEPS:+--steps "$STEPS"} \
      --out "$run_dir/results.json" ${RESUME:+"$RESUME"} "$@" >> "$run_dir/train.log" 2>&1 &
  pids+=($!)
done

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "seed ${SEED_ARR[$i]} FAILED" >&2; status=1
  else
    echo "seed ${SEED_ARR[$i]} ok"
  fi
done
exit "$status"
