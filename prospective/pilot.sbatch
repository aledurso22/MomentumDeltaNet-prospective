#!/usr/bin/env bash
#SBATCH --job-name=mdn-pilot
#SBATCH --partition=pgi15
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --time=03:00:00
#SBATCH --output=/Users/durso/mdn-pilot-runs/slurm-%x-%j.out
#
# The v4 geometry/inertia PILOT: four paired seeds, one per GPU, each seed a
# direct child process training the four pilot arms in lockstep on identical
# batches from identical init (run_wave.sh; the S5 pattern, no srun below).
# Arms, steps, eval and checkpoint cadence and seeds all come from
# `python -m prospective.experiment_pilot flags|seeds`.
#
# Batch wrapper around run_pilot.sh. Submit ONLY after the verification job for this commit has passed and its
# profile (on this GPU model) says how long 2,000 updates take:
#
#   EXPECTED_COMMIT=<sha> RUN_TAG=pilot-<sha10> \
#     sbatch --nodelist=<four-gpu-node> prospective/pilot.sbatch
#
# Resumable: the run directories are $OUT_ROOT/$RUN_TAG-seed<s> and every
# seed checkpoints every 200 steps; resubmitting the same command with the
# same RUN_TAG continues each seed from its checkpoint. The time limit is a
# placeholder to be set from the measured profile.
set -euo pipefail

REPO="${REPO:-/Local/durso/mdn-bridge}"
VENV="${VENV:-/Local/durso/.venv-torch}"
[ -d "$REPO/.git" ] || { echo "no checkout at $REPO on $(hostname)" >&2; exit 2; }
cd "$REPO"
source "$VENV/bin/activate"
export OUT_ROOT="${OUT_ROOT:-/Users/durso/mdn-pilot-runs}"
exec bash prospective/run_pilot.sh
