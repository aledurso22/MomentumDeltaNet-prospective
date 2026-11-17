#!/usr/bin/env bash
#SBATCH --job-name=mdn-verify
#SBATCH --partition=pgi15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --time=03:00:00
#SBATCH --output=/Users/durso/mdn-bridge-verify/slurm-%x-%j.out
#
# VERIFICATION ONLY: CPU tests, GPU integration tests (both Triton precision
# modes), profiling of the four-arm pilot configuration. It never starts a
# training run.
#
#   EXPECTED_COMMIT=<sha> sbatch --nodelist=pgi15-gpu5 prospective/verify.sbatch
#
# Accepted GPUs: RTX 3090 or RTX 4090 (ALLOWED_GPUS in verify_gpu.sh). The
# profile is only valid for the GPU model it ran on: profile on the model the
# pilot will use. Inside an existing interactive allocation, run
# verify_gpu.sh directly instead (README / v4 report).
#
# Same conventions as seed.sbatch (partition pgi15, cpu=10, mem=50G, the
# /Local/durso checkout and venv); the node is chosen on the command line.
# The job refuses any commit but the expected one. Artefacts go to the SHARED home, so they can be fetched from the head
# node without logging into the GPU node.
set -euo pipefail

REPO="${REPO:-/Local/durso/mdn-bridge}"
VENV="${VENV:-/Local/durso/.venv-torch}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the commit to verify}"

[ -d "$REPO/.git" ] || { echo "no checkout at $REPO on $(hostname)" >&2; exit 2; }
[ -f "$VENV/bin/activate" ] || { echo "no venv at $VENV on $(hostname)" >&2; exit 2; }
cd "$REPO"
source "$VENV/bin/activate"

HEAD_SHA="$(git rev-parse HEAD)"
if [ "$HEAD_SHA" != "$EXPECTED_COMMIT" ]; then
  echo "refusing: HEAD is $HEAD_SHA, expected $EXPECTED_COMMIT" >&2; exit 2
fi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
# the accepted models are checked (and the hardware recorded) by verify_gpu.sh

export OUT="/Users/durso/mdn-bridge-verify/${HEAD_SHA:0:10}-${SLURM_JOB_ID}"
echo "=== job $SLURM_JOB_ID on $(hostname), $GPU_NAME"
echo "commit : $HEAD_SHA"
echo "out    : $OUT"
exec bash prospective/verify_gpu.sh
