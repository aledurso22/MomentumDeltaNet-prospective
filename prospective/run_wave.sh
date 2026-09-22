#!/usr/bin/env bash
# One Slurm job, several GPUs, one seed per GPU as a DIRECT CHILD PROCESS.
#
# This is the S5 wave's pattern, and it exists because separate co-resident
# jobs on this node do not survive each other: submitting three sbatch jobs to
# pgi15-gpu5 ended with the first COMPLETED and the other two FAILED at the
# same elapsed second, when node cleanup for the finishing job took its
# neighbours with it. No sbatch, no salloc, no srun below this line.
#
#   SEEDS="501 502 503" bash prospective/run_wave.sh -- <extra train.py flags>
set -euo pipefail

SEEDS="${SEEDS:-501 502 503}"
STEPS="${STEPS:-4000}"
OUT_ROOT="${OUT_ROOT:-$HOME/mdn-bridge-runs}"
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

STAMP="$(date +%Y%m%d-%H%M%S)"
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
    python -u -m prospective.train --seed "$seed" --steps "$STEPS" \
      --out "$run_dir/results.json" "$@" > "$run_dir/train.log" 2>&1 &
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
