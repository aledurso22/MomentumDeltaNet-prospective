#!/usr/bin/env bash
# Targeted diagnostic for the two IEEE failures of
# test_official_chunk_layer_vs_fp64 (T = 52, 128). One GPU; no test suite,
# no profiler, no training. See prospective/diag_chunk_gates.py.
#
#   CUDA_VISIBLE_DEVICES=0 bash prospective/diag_chunk_gates.sh           # IEEE (the failing mode)
#   CUDA_VISIBLE_DEVICES=0 DIAG_MODE=default bash prospective/diag_chunk_gates.sh
#
# Triton cache: PERSISTENT, one directory per commit and precision mode,
# $TRITON_CACHE_ROOT/<sha10>/<mode> (default root ~/.cache/mdn-triton), shared
# with verify_gpu.sh, so reruns at the same commit do not recompile.
# FLA_CACHE_RESULTS=1 also keeps the autotune choices there.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
SHA10="$(git rev-parse --short=10 HEAD)"
MODE="${DIAG_MODE:-ieee}"
case "$MODE" in
  ieee) export TRITON_F32_DEFAULT=ieee ;;
  default) unset TRITON_F32_DEFAULT ;;
  *) echo "DIAG_MODE must be ieee or default" >&2; exit 2 ;;
esac
export PYTHONPATH="$REPO/flash-linear-attention${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="${TRITON_CACHE_ROOT:-$HOME/.cache/mdn-triton}/$SHA10/$MODE"
export FLA_CACHE_RESULTS="${FLA_CACHE_RESULTS:-1}"
OUT="${OUT:-$HOME/mdn-bridge-verify/$SHA10-diag-$MODE-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT" "$TRITON_CACHE_DIR"
echo "commit $(git rev-parse HEAD)  mode $MODE  cache $TRITON_CACHE_DIR  out $OUT"
python -u -m prospective.diag_chunk_gates --T 52 128 --out "$OUT" 2>&1 | tee "$OUT/diag.log"
