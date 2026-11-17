#!/usr/bin/env bash
# The v4 geometry/inertia PILOT, run directly on a node you already hold
# (an interactive allocation, or from pilot.sbatch). Four paired seeds, one
# per GPU; within each seed the four arms train in lockstep on identical
# batches from identical init (run_wave.sh: direct child processes, no srun).
# Arms, steps, eval/checkpoint cadence and seeds come from
# `python -m prospective.experiment_pilot flags|seeds`.
#
#   EXPECTED_COMMIT=<sha> RUN_TAG=pilot-<sha10> bash prospective/run_pilot.sh
#
# Resumable: rerun the same command with the same RUN_TAG and every seed
# continues from its last checkpoint (every 200 updates).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the verified commit}"
: "${RUN_TAG:?set RUN_TAG (stable; reuse it to resume)}"
export OUT_ROOT="${OUT_ROOT:-$HOME/mdn-pilot-runs}"
export EXPECTED_COMMIT RUN_TAG
unset STEPS                         # the pilot flags carry --steps
export PYTHONPATH="$REPO/flash-linear-attention${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

SEEDS="$(python -m prospective.experiment_pilot seeds)"
read -r -a FLAGS <<< "$(python -m prospective.experiment_pilot flags)"
export SEEDS
read -r -a SEED_ARR <<< "$SEEDS"
NGPU="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a _vis <<< "$CUDA_VISIBLE_DEVICES"; NGPU="${#_vis[@]}"
fi
if [ "$NGPU" -lt "${#SEED_ARR[@]}" ]; then
  echo "refusing: ${#SEED_ARR[@]} seeds but $NGPU GPUs visible" >&2; exit 2
fi
mkdir -p "$OUT_ROOT"
echo "=== pilot on $(hostname), job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "flags  : ${FLAGS[*]}"
exec bash prospective/run_wave.sh -- "${FLAGS[@]}"
