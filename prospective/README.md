# Prospective write rules on Momentum DeltaNet

Base: this repository at **c6e77fa261fb0c002fae1a14b6209a5b28d2edc9**
(`HuuYuLong/MomentumDeltaNet`, ICML 2026). Branch `prospective-rules` changes
nothing under `flash-linear-attention/` or `flame/`.

Reports: [`reports/v3_memory_motion_plan.md`](reports/v3_memory_motion_plan.md)
(next experiment: joint memory–motion writer vs controls),
[`reports/v2_official_mixer.md`](reports/v2_official_mixer.md) (restored
mixer; its wave is not launched) and
[`reports/v1_reduced_mixer.md`](reports/v1_reduced_mixer.md) (superseded, with
erratum; code at `8d39b8a`).

| file | what |
|---|---|
| `coefficients.py` | the (M, gamma, T) -> (a, b, c, d) map, taken unchanged from the S5 study |
| `rules.py` | the six arms as recurrent references, line-for-line against `naive.py` |
| `reference.py` | the `reference` backend: chunk_mode_rule's preprocessing + `rules.py` (no fla import) |
| `mixer.py` | `ProspectiveMDN`: the official `MomentumDeltaNet` layer with a recurrent-backend hook |
| `writers.py` | v3 writers: `joint` (constant-velocity Kalman), `rls` (full key covariance), `geometric` (fixed-gain inertial) |
| `experiment_v3.py`, `specs/v3/` | v3 protocol: arms, primary score, equal-budget dev grids, selection, confirmation, analysis |
| `model.py` | shared small backbone around the mixer; `build()` gives identical shared init |
| `task.py` | MQAR with revision; lag metric with counts; legacy v1 metrics |
| `train.py` | six arms in lockstep, FP32, production config as defaults |
| `profile_gpu.py` | measured per-arm time/memory and the per-seed wave estimate |
| `layer_ref.py` | pure-PyTorch functional layer in any dtype: the FP64 reference for GPU checks |
| `verify_gpu.sh` | gate before a wave: unit tests, integration (default + IEEE dots), profile; never trains |
| `verify.sbatch` | the verification-only Slurm job (one RTX 4090; refuses other GPUs) |
| `run_wave.sh`, `run_seed.sh`, `seed.sbatch` | launchers (output in `mdn-bridge-runs-v2`) |
| `aggregate.py`, `summary.py`, `parse_log.py` | analysis; never mix mixer versions |
| `test_coefficients.py`, `test_rules.py`, `test_task.py` | CPU tests |
| `test_writers.py`, `test_experiment_v3.py` | v3 writers vs dense FP64 / global Kalman references; protocol checks |
| `test_model_cpu.py`, `cpu_stub/` | model-level CPU plumbing tests with the real layer source and CPU stand-ins for Triton modules |
| `test_integration.py` | GPU tests against the official layer and chunk kernel, errors vs FP64 to JSON |

`tss` and `generalized` are the SAME code path and the SAME equation: TSS is
`M = gamma = 0`. Its start point `T = h` gives the filter `Y = 2R - R_prev`,
not the identity.

Not done: chunked/cached execution (whole-sequence only), and a `chunk.py`
kernel for the modified rules.
