#!/usr/bin/env bash
#SBATCH --job-name=mdn-heldout
#SBATCH --partition=pgi15
#SBATCH --nodelist=pgi15-gpu5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --chdir=/Users/durso
#SBATCH --output=/Users/durso/mdn-pilot-runs/slurm-%x-%j.out
# Existing source checkout and environment are node-local to pgi15-gpu5.
# This is a maximum wall time, not an estimated duration.
set -euo pipefail
cd /Local/durso/mdn-bridge
source /Local/durso/.venv-torch/bin/activate
exec bash prospective/run_eval_pilot.sh
