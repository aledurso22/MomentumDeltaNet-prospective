#!/usr/bin/env bash
#SBATCH --job-name=mdn-bridge
#SBATCH --partition=pgi15
#SBATCH --nodelist=pgi15-gpu5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --output=/Local/durso/mdn-bridge/slurm-%x-%j.out
#
# Detached run: survives logout, SSH drops and the laptop being closed.
#
#   SEED=501 sbatch prospective/seed.sbatch
#
# Resources match allocation 67366/67378 exactly (cpu=10, gpu=1, mem=50G),
# with a real time limit rather than the 40 minutes that killed 67366.
set -euo pipefail

SEED="${SEED:-501}"
REPO="${REPO:-/Local/durso/mdn-bridge}"
VENV="${VENV:-/Local/durso/.venv-torch}"
export OUT_ROOT="${OUT_ROOT:-/Local/durso/mdn-bridge-runs}"

cd "$REPO"
source "$VENV/bin/activate"

echo "=== job $SLURM_JOB_ID on $(hostname) ==="
echo "seed   : $SEED"
echo "out    : $OUT_ROOT"

# run_seed.sh enforces the commit guard and the clean-worktree guard
exec env EXPECTED_COMMIT="${EXPECTED_COMMIT:-$(git rev-parse HEAD)}" \
     ./prospective/run_seed.sh "$SEED"
