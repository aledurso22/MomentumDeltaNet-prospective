#!/usr/bin/env bash
# Four final checkpoints, one per allocated GPU. Evaluation only.
set -euo pipefail
cd "$(dirname "$0")/.."
MDN_REPO="$PWD"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the evaluator commit}"
if [ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]; then
  echo "Wrong checkout: expected $EXPECTED_COMMIT" >&2
  exit 2
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Tracked files have local changes; refusing evaluation." >&2
  exit 2
fi
export PYTHONPATH="$MDN_REPO/flash-linear-attention${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
# Reuse the existing native/reference frontend's cache; no MDN chunk kernel
# is called by this evaluator. Source model code has not changed.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/mdn-triton/47fe52a670/default}"
export FLA_CACHE_RESULTS=1

MDN_SOURCE_ROOT="${SOURCE_ROOT:-$HOME/mdn-pilot-runs}"
MDN_SOURCE_TAG="${SOURCE_TAG:-pilot-47fe52a670}"
MDN_EVAL_ROOT="${EVAL_ROOT:-$HOME/mdn-pilot-heldout}"
MDN_EVAL_TAG="${EVAL_TAG:-heldout-47fe52a670-v1}"
MDN_OUT="$MDN_EVAL_ROOT/$MDN_EVAL_TAG"
MDN_SEEDS=(701 702 703 704)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a MDN_GPU_TOKENS <<< "$CUDA_VISIBLE_DEVICES"
else
  read -r -a MDN_GPU_TOKENS <<< "$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
fi
if [ "${#MDN_GPU_TOKENS[@]}" -lt 4 ]; then
  echo "Need four visible GPUs, one for each seed." >&2
  exit 2
fi
for MDN_SEED in "${MDN_SEEDS[@]}"; do
  MDN_CHECKPOINT="$MDN_SOURCE_ROOT/$MDN_SOURCE_TAG-seed$MDN_SEED/ckpt.pt"
  test -f "$MDN_CHECKPOINT" || { echo "Missing checkpoint: $MDN_CHECKPOINT" >&2; exit 2; }
  python -c 'import sys; from prospective.eval_pilot import load_checkpoint; load_checkpoint(sys.argv[1], expected_seed=int(sys.argv[2]))' "$MDN_CHECKPOINT" "$MDN_SEED"
done
mkdir -p "$MDN_EVAL_ROOT"
mkdir "$MDN_OUT"  # intentionally fails if this follow-up already exists
printf 'Evaluation only: %s\nSource: %s/%s-seed*\nOutput: %s\n' \
  "$EXPECTED_COMMIT" "$MDN_SOURCE_ROOT" "$MDN_SOURCE_TAG" "$MDN_OUT"

MDN_PIDS=()
mdn_stop_children() {
  for MDN_PID in "${MDN_PIDS[@]}"; do kill "$MDN_PID" 2>/dev/null || true; done
  wait || true
  exit 130
}
trap mdn_stop_children INT TERM
for MDN_I in 0 1 2 3; do
  MDN_SEED="${MDN_SEEDS[$MDN_I]}"
  CUDA_VISIBLE_DEVICES="${MDN_GPU_TOKENS[$MDN_I]}" \
    python -u -m prospective.eval_pilot evaluate \
      --checkpoint "$MDN_SOURCE_ROOT/$MDN_SOURCE_TAG-seed$MDN_SEED/ckpt.pt" \
      --out "$MDN_OUT/seed$MDN_SEED.json" \
      --device cuda --batches 24 --batch-size 32 \
      > "$MDN_OUT/seed$MDN_SEED.log" 2>&1 &
  MDN_PIDS+=("$!")
  printf 'Evaluating seed %s -> %s/seed%s.log\n' "$MDN_SEED" "$MDN_OUT" "$MDN_SEED"
done
MDN_FAILED=0
for MDN_I in 0 1 2 3; do
  if wait "${MDN_PIDS[$MDN_I]}"; then
    printf 'seed %s ok\n' "${MDN_SEEDS[$MDN_I]}"
  else
    printf 'seed %s FAILED: inspect its log\n' "${MDN_SEEDS[$MDN_I]}" >&2
    MDN_FAILED=1
  fi
done
trap - INT TERM
python -m prospective.eval_pilot analyse "$MDN_OUT/seed*.json" | tee "$MDN_OUT/summary.txt"
exit "$MDN_FAILED"
