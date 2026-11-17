# Part 2, v4 — Geometry and learned inertia: training pilot

**Status: implemented and CPU-checked. Every GPU check, every cost number and
every training result is PENDING until the cluster jobs below have run.**

This pilot supersedes the v3 plan ([`v3_memory_motion_plan.md`](v3_memory_motion_plan.md)).
The v3 code (`writers.py`, `experiment_v3.py`, `specs/v3/`) and its reports are
kept unchanged; none of it is launched. The pilot uses no Kalman/joint writer,
no Nesterov arm, none of the old residual filters, and no development grid.

## 1. Questions

1. Does historical prediction geometry help with learned representations?
   Contrast: `geometric_projection` vs `euclidean_projection`. The only
   difference is the correction direction b.
2. Does letting the inertial coefficients learn add anything beyond that
   geometry? Contrast: `geometric_motion_learned` vs `geometric_projection`.
   Same b; motion is added.

Each projection arm is also reported against `native`.

## 2. Arms

All four share the restored official frontend and readout: projections,
causal short convolutions + SiLU, the output correction
q ← q − e^{D_log} k, in-kernel-style l2 normalisation of q and k, the
sigmoid-gated RMS norm and o_proj. Only the recurrent backend differs.

In math orientation W ∈ ℝ^{dv×dk}; the code stores S = Wᵀ.
`o_t = K^{-1/2} · W_t q_t`: the corrected memory is read with the official
output scaling.

| arm | recurrence |
|---|---|
| `native` | official MDN rule with its learned gates (α, β, μ, η) |
| `euclidean_projection` | W_t = W_{t−1} + (v_t − W_{t−1}k_t) b_tᵀ, with b_t = k_t / (k_tᵀk_t) |
| `geometric_projection` | same update; C_0 = .03 I, C_t = C_{t−1} + k_t k_tᵀ, b_t = C_t⁻¹k_t / (k_tᵀC_t⁻¹k_t) |
| `geometric_motion_learned` | same b_t, G = 1: U_t = W_{t−1} + V_{t−1}; D_t = (v_t − U_t k_t) b_tᵀ; W_t = U_t + D_t; V_t = (1−d)V_{t−1} + (A−d)D_t; d = rA |

The geometric writers:
- **Not the RLS writer.** b_t has no observation-noise term in its
  denominator, so `writers.rls_writer` is not used.
- **Inverse maintained incrementally.** C⁻¹ is updated by Sherman–Morrison
  and differentiated through. There is no forgetting (λ = 1); c0 = .03 is
  the initial ridge.
- **No carried state.** W_0 = V_0 = 0 and C_0⁻¹ = I/.03 at the start of every
  sequence. Execution is whole-sequence only.

The projection arms use the normalised unscaled key k (never MDN's p = αk),
and no MDN gate. Their nine MDN-gate parameters per block are kept, so that
shared initialisation is identical across arms. They are never read and are
declared and tested as unused (`mixer.unused_by_design`).

### Learned inertial coefficients

- **What is learned:** A ∈ [0, 1] and r ∈ [0, 1], one of each per head and
  layer (2 × 4 × 2 = 16 numbers), with d = rA.
- **Initialisation:** A = .003125, r = .5, so d = .0015625. This is the
  tested modest-drag setting of the local screens.
- **They are not gates:** they are outer-trained parameters, constant over
  the tokens of a forward pass, with no input dependence.
- **Training:**
  - stored directly and clamped to [0, 1] after every optimizer step (`Model.project_`);
  - excluded from weight decay, and nothing else is (`model.NO_DECAY_LEAVES`);
  - for every other arm, the optimizer is the old single-group AdamW, bitwise (tested).
- **Boundaries:**
  - r = 0 is zero drag;
  - A = 0 is no motion (d = 0, V stays 0);
  - r = 1 is also no motion (A − d = 0, V stays 0).

  The [0, 1] box is the experimental parameter domain, not a stability
  theorem for the network.
- **Recording:** A, r, d and A − d are recorded per head and layer at every
  evaluation (`curves[*].inertia`) and at the end (`inertia`).

Motion that is cancelled (A − d ≈ 0) is reported as such. It is not reported
as an inertial benefit.

> **Flag (not changed, as instructed: no sweep, direct parameters).** AdamW
> moves each parameter by roughly lr = 3e-3 per step, and A starts at
> 3.125e-3. In a CPU smoke run of the exact pilot configuration, some heads'
> A was clamped to 0 after 2 updates, while others roughly doubled. The
> initial value therefore does not pin the regime for long. The learned
> trajectory of A, d and A − d is what the pilot reports.

### Zero keys

- **Exact zero:** k_tᵀ x_t = 0 happens only for k_t = 0. There b_t := 0
  (no correction), and the Sherman–Morrison step is already the identity.
- **No threshold:** the guard is exactly `den == 0`. Tiny nonzero keys get
  the exact projection.
- **Motion arm:** at a zero key W_t = U_t, so the motion still advances.
- **Tested:** outputs and every gradient stay finite, including in a layer
  whose key projection is identically zero.

### Recurrent storage

Persistent state per sequence, head and layer at dk = 32, dv = 64:

| arm | carried state | floats |
|---|---|---|
| native | S, momentum | 4096 |
| euclidean_projection | W | 2048 |
| geometric_projection | W, C⁻¹ | 3072 |
| geometric_motion_learned | W, V, C⁻¹ | 5120 |

The no-motion writers do not allocate a velocity matrix. The carried state is
asserted to equal these numbers. Training memory (activations kept for
backprop) is measured separately by the profiler.

## 3. Verification

### CPU (done)

Local result: 188 passed, 1 skipped. The skip is the CUDA-only file.

`test_projection.py` (36 tests):
- **Dense FP64 reference:** all three rules match an independent dense
  reference (explicit C_t, a linear solve, per-(batch, head) loops, dk ≠ dv),
  at the initial and at interior (A, r).
- **Existing geometric writer:** `geometric_projection` equals
  `writers.geometric_writer` at A = 0, G = 1, c0 = .03, λ = 1, in outputs and
  in q, k, v gradients.
- **Zero drag:** r = 0 equals the same writer at the same A.
- **Frozen local recurrence:** at initialisation, every pilot rule equals the
  completed screens' recurrence (`prospective_rtrl_experiment` @ `cf8f87c`,
  `diagnostics/five_writer_screen.py::Standalone`). It is checked against a
  transcription in the test, and also against the original file imported
  unchanged when that checkout is present (it was, locally).
- **No-motion boundaries:** A = 0 and r = 1 are bitwise
  `geometric_projection`, per head, including mixed heads.
- **Projection property:** W_t k_t = v_t after every write, in all three arms.
- **Gradients:**
  - `gradcheck` passes through C⁻¹ and through A and r;
  - detaching C⁻¹ changes ∂/∂k;
  - A and r receive nonzero gradients at initialisation.
- **Causality:** exact.
- **Zero keys:** handled as described above.
- **C⁻¹:** bitwise symmetric and positive definite (FP64 and FP32 at
  production size), equal to the dense inverse.
- **State:** resets between calls; the storage accounting is checked.

`test_model_cpu.py` (real official layer source, CPU stand-ins for the Triton
modules), for every arm, including the pilot arms:
- FP32 matches the FP64 functional layer, in outputs, inputs and every
  parameter gradient;
- exact model causality, identical shared initialisation, and exactly the
  declared unused parameters.

It also checks, for the pilot arms:
- leaf initialisation, and that the layer at initialisation runs exactly the
  declared recurrence on the frontend outputs;
- the boundary identities at layer level;
- the weight-decay exclusion, and that other arms' optimizer is unchanged;
- clamping, gradients reaching A and r, and zero raw keys.

`test_train_cpu.py`:
- **Resume:** checkpoint + resume reproduces an uninterrupted run bitwise:
  every parameter, the AdamW state, the batch generator, the curves and the
  coefficients.
- **Refusals:** a resume with a different configuration is refused.
- **Failures:** a non-finite update is recorded and not applied.

`test_experiment_pilot.py`:
- exactly the four arms;
- the spec file and launch flags give the existing configuration, with fresh seeds;
- the analysis flags near-chance seeds and cancelled motion.

### GPU (PENDING)

`test_integration.py` keeps every existing check. Its every-arm loops (FP32
reference vs FP64 layer, causality, shared initialisation, unused parameters)
now include the three pilot arms. It adds:
- the motion arm vs FP64 at interior (A, r), graded at the declared 1e-4
  norm-relative;
- bitwise no-motion boundaries on GPU;
- finite zero-key layers;
- a few updates of the exact pilot spec with a checkpoint and a resume: the
  batch stream must be identical; parameter differences are recorded.

## 4. Pilot protocol (`experiment_pilot.py`)

- **Model:** width 128, 2 layers, 4 heads, dk 32, dv 64, batch 16, FP32
  (TF32 off), reference backend.
- **Task:** the existing 52-token production MQAR (8 pairs, 6 revisions,
  12 queries, 24 keys, 32 values), read-after-update; queries also write.
- **Training:**
  - the existing AdamW (lr 3e-3, wd .01 except A and r, clip 1.0), no sweep;
  - 2,000 updates;
  - evaluation every 100 updates on the fixed 24 × 32 set;
  - checkpoint every 200 updates.
- **Pairing:** the four arms train in lockstep in one process per seed, on
  identical batches from identical shared initialisation.
- **Seeds:** four fresh paired seeds, 701–704, one per GPU on a four-GPU node.
- **Metrics:**
  - primary ½(rev_lag_le8 + untouched), each seed scored as the mean of the
    last 5 evaluations;
  - also both components, revised, overall, and the counts;
  - learning curves;
  - learned A, d, A − d.
- **Analysis:** per-seed paired differences, descriptive only. This is a
  pilot.
- **Exclusions and labels:**
  - a seed where either arm of a pair stays near chance (overall ≤ 5 %;
    chance is 3.1 %) or failed is *inconclusive* for that pair, and is listed;
  - missing seeds are listed;
  - if A − d ≈ 0 in every head, the analysis prints that no difference may be
    called inertial.

## 5. Cost (PENDING)

The cluster has not measured this yet. The verification job profiles exactly
the pilot configuration: four arms, 2,000 updates, 20 evaluations and 10
checkpoints, taken from `experiment_pilot flags`. No other arm set is
profiled.

It reports per arm:
- forward, backward and update time;
- evaluation time;
- peak allocated and reserved memory;
- persistent state.

It also reports:
- the checkpoint write time;
- the per-seed estimate, and whether it fits the 60-minute target;
- if not, how many updates would.

The profile is valid only for the GPU model it ran on. The pilot runs on
pgi15-gpu5 (4 × RTX 3090), so verification and profiling run there too.
`verify_gpu.sh` accepts an RTX 3090 or 4090 and records the detected hardware
(`hardware.txt`, `hardware.csv`). Numerical checks and tolerances do not
depend on the GPU model.
