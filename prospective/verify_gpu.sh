#!/usr/bin/env bash
# The gate before any wave: unit tests, integration in both Triton precision
# modes, then profiling. NEVER launches training.
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
export PYTHONPATH="$REPO/flash-linear-attention${PYTHONPATH:+:$PYTHONPATH}"
CACHE_ROOT="${TMPDIR:-/tmp}/triton-cache-$$"

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

stage unit python -m pytest -q -p no:cacheprovider prospective/test_rules.py \
  prospective/test_writers.py prospective/test_experiment_v3.py prospective/test_model_cpu.py \
  prospective/test_coefficients.py prospective/test_task.py
stage integration_default env TRITON_CACHE_DIR="$CACHE_ROOT/default" \
  python -m pytest -s -q -p no:cacheprovider prospective/test_integration.py
stage integration_ieee env TRITON_CACHE_DIR="$CACHE_ROOT/ieee" TRITON_F32_DEFAULT=ieee \
  python -m pytest -s -q -p no:cacheprovider prospective/test_integration.py
stage profile python -u -m prospective.profile_gpu --out "$OUT/profile.json"
rm -rf "$CACHE_ROOT"
echo "artefacts in $OUT"
cat "$OUT/status.txt"
! grep -qv ' 0$' "$OUT/status.txt"
