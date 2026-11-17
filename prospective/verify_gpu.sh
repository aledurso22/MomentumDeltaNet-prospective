#!/usr/bin/env bash
# The gate before any wave: unit tests, integration in both Triton precision
# modes, then profiling of the v4 PILOT configuration only (its four arms,
# 2,000 steps, its eval and checkpoint cadence, all taken from
# `experiment_pilot flags`). No other arm set is profiled. NEVER launches
# training.
#
#   OUT=<dir> CUDA_VISIBLE_DEVICES=0 bash prospective/verify_gpu.sh
#
# Uses the fla checked out in THIS repository (PYTHONPATH), so the kernels
# are the pinned upstream ones whatever is pip-installed. Every stage runs
# even if an earlier one fails (so one failure does not hide the others);
# per-stage exit codes go to $OUT/status.txt and the script exits non-zero if
# any stage failed.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "refusing: tracked files are modified" >&2; exit 1
fi
OUT="${OUT:-$HOME/mdn-bridge-verify/$(git rev-parse --short=10 HEAD)-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
export VERIFY_OUT="$OUT"
# accepted GPU models (substrings of the nvidia-smi name); the numerical
# checks and tolerances do not depend on which one is detected
ALLOWED_GPUS="${ALLOWED_GPUS:-RTX 3090,RTX 4090}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
ok=0; IFS=',' read -r -a _allowed <<< "$ALLOWED_GPUS"
for g in "${_allowed[@]}"; do case "$GPU_NAME" in *"$g"*) ok=1 ;; esac; done
if [ "$ok" != 1 ]; then
  echo "refusing: GPU '$GPU_NAME' is not one of: $ALLOWED_GPUS" >&2; exit 2
fi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id \
  --format=csv > "$OUT/hardware.csv"
{ echo "detected_gpu_0: $GPU_NAME"; echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}  node: $(hostname)"
  lscpu | grep -E "Model name|^CPU\(s\)" || true; } > "$OUT/hardware.txt"
cat "$OUT/hardware.txt"
export PYTHONPATH="$REPO/flash-linear-attention${PYTHONPATH:+:$PYTHONPATH}"
# persistent Triton cache, one directory per commit and precision mode
# (shared with diag_chunk_gates.sh); never deleted, so reruns at the same
# commit do not recompile. FLA_CACHE_RESULTS=1 also keeps autotune choices.
CACHE_ROOT="${TRITON_CACHE_ROOT:-$HOME/.cache/mdn-triton}/$(git rev-parse --short=10 HEAD)"
export FLA_CACHE_RESULTS="${FLA_CACHE_RESULTS:-1}"
mkdir -p "$CACHE_ROOT/default" "$CACHE_ROOT/ieee"

{
  echo "commit $(git rev-parse HEAD)"
  echo "host   $(hostname)"
  python - <<'EOF'
import os, torch, triton, einops, fla
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("triton", triton.__version__)
print("fla from", os.path.dirname(fla.__file__))
try:
    import causal_conv1d; print("causal_conv1d", causal_conv1d.__version__)
except Exception:
    print("causal_conv1d not installed -> ShortConvolution uses its Triton backend")
print("gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
EOF
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  echo "triton cache root $CACHE_ROOT (persistent); FLA_CACHE_RESULTS=$FLA_CACHE_RESULTS"
} 2>&1 | tee "$OUT/env.txt"

stage() {   # stage <name> <command...>; never aborts the script
  local name="$1"; shift
  echo "== $name"
  set +e
  "$@" 2>&1 | tee "$OUT/$name.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "$name $rc" | tee -a "$OUT/status.txt"
}
: > "$OUT/status.txt"

read -r -a PILOT_FLAGS <<< "$(python -m prospective.experiment_pilot flags)"
echo "pilot flags: ${PILOT_FLAGS[*]}" | tee -a "$OUT/env.txt"

stage unit python -m pytest -q -p no:cacheprovider prospective/test_rules.py \
  prospective/test_writers.py prospective/test_experiment_v3.py prospective/test_model_cpu.py \
  prospective/test_coefficients.py prospective/test_task.py prospective/test_projection.py \
  prospective/test_experiment_pilot.py prospective/test_train_cpu.py
stage integration_default env TRITON_CACHE_DIR="$CACHE_ROOT/default" \
  python -m pytest -s -q -p no:cacheprovider prospective/test_integration.py
stage integration_ieee env TRITON_CACHE_DIR="$CACHE_ROOT/ieee" TRITON_F32_DEFAULT=ieee \
  python -m pytest -s -q -p no:cacheprovider prospective/test_integration.py
stage profile env TRITON_CACHE_DIR="$CACHE_ROOT/default" python -u -m prospective.profile_gpu "${PILOT_FLAGS[@]}" \
  --budget-minutes 60 --out "$OUT/profile.json"
echo "artefacts in $OUT"
cat "$OUT/status.txt"
! grep -qv ' 0$' "$OUT/status.txt"
