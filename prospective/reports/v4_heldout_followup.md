# Final-checkpoint follow-up to the v4 geometry pilot

The completed pilot reached approximately 100% accuracy in all four arms by
step 300. Its final scores do not establish an advantage for geometry or
learned motion. All four seeds reached step 2000 without recorded arm failure.

This is a **follow-up stress test**, using the delay/load generators already
present in `task.HELDOUT`. These suites were discussed in the superseded v3
plan but were not preregistered endpoints of the v4 pilot. No new training,
hyperparameter selection, model surgery, or checkpoint selection is performed.

## Fixed protocol

- Final step-2000 checkpoints for seeds 701, 702, 703, 704 from
  `pilot-47fe52a670-seed<seed>/ckpt.pt`.
- All four saved arms, reconstructed from each checkpoint's configuration and
  embedded instance specification. Loading is strict; failed/incomplete
  checkpoints and unexpected task/vocabulary configurations are refused.
- Existing `delay8`, `delay16`, `load12`, `load16` suites, unchanged. The first
  two add distractor writes after revisions; the last two increase initial
  memory load. Token IDs, value vocabulary, query writes, read-after-update,
  and lag definitions remain those of the pilot.
- Per seed and suite: 24 batches of 32 sequences. Every arm receives exactly
  the same examples. The seed is `177000 + 10 * training_seed + suite_index`,
  where suites are sorted lexicographically. This differs from both the
  training and original evaluation random streams. Example hashes are saved.
- Evaluate the complete saved network in FP32 with the reference recurrent
  backend. No gradients or optimizer are used. Checkpoint bytes and loaded
  model state tensors must remain unchanged.
- Report revised, untouched, and overall accuracy with correct/total counts,
  per seed and mean across seeds. Report descriptive paired differences for
  geometry versus Euclidean, motion versus geometry, and each versus native.
  Retain all suites; do not select a favorable one as a new primary endpoint.
- Additional original metrics and lag-bin counts are stored. Undefined
  quantities are JSON null, never zero: delay suites have no short-lag samples
  and hence no value for the original pilot primary metric.
- Outputs go under `~/mdn-pilot-heldout/`, separately from the original pilot.
  Existing output files/directories are not overwritten.

These evaluations test generalization to greater memory load and distractor
delay. They do not directly test a smoothly drifting teacher or establish
whether zero drag is preferable to learned drag.

## Run on the current four-GPU allocation

After fetching the evaluator commit on pgi15-gpu5:

```bash
cd /Local/durso/mdn-bridge
source /Local/durso/.venv-torch/bin/activate
EXPECTED_COMMIT=<evaluator-sha> bash prospective/run_eval_pilot.sh
```

This reads the existing checkpoints without resuming training. The launcher
runs one seed per GPU, records per-seed progress logs, and prints the combined
summary after the four processes finish. The GPU execution time is unmeasured;
it performs only inference, not the pilot's 2000 optimizer updates.

For a session-independent batch, submit `prospective/eval_pilot.sbatch` on
pgi15-gpu5 with `EXPECTED_COMMIT` exported. An interactive allocation holding
all four GPUs must finish before that batch can start. The existing
`/Users/durso/mdn-pilot-runs` directory must exist for Slurm's log file.

Individual evaluation and later analysis are also available:

```bash
python -m prospective.eval_pilot evaluate \
  --checkpoint /path/to/seed701/ckpt.pt --out /separate/path/seed701.json
python -m prospective.eval_pilot analyse '/separate/path/seed*.json'
```

## Validation status

Validation: **19 CPU tests passed** (8 evaluator checks and 11 task checks);
both launcher scripts also pass `bash -n`. Focused CPU tests use the actual
layer with the existing CPU stand-ins for
Triton components. They check reconstruction, shared examples, no training or
state changes, correct counts, undefined metrics, failure on non-finite logits,
and refusal to overwrite results. This verifies evaluator behavior, not CUDA
numerics. GPU checkpoint evaluation remains to be run on the cluster.
