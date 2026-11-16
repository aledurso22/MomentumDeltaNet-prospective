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
# modes), profiling. It never starts a training run.
#
#   EXPECTED_COMMIT=<sha> sbatch --nodelist=<rtx4090-node> prospective/verify.sbatch
#
# Same conventions as seed.sbatch (partition pgi15, cpu=10, mem=50G, the
# /Local/durso checkout and venv); the node is chosen on the command line
# because the 4090 node is not the 3090 node pgi15-gpu5. The job refuses to
# run on any GPU that is not an RTX 4090, and on any commit but the expected
# one. Artefacts go to the SHARED home, so they can be fetched from the head
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
case "$GPU_NAME" in
  *4090*) ;;
  *) echo "refusing: GPU is '$GPU_NAME', this job is for an RTX 4090" >&2; exit 2 ;;
esac

export OUT="/Users/durso/mdn-bridge-verify/${HEAD_SHA:0:10}-${SLURM_JOB_ID}"
echo "=== job $SLURM_JOB_ID on $(hostname), $GPU_NAME"
echo "commit : $HEAD_SHA"
echo "out    : $OUT"
exec bash prospective/verify_gpu.sh
