# Part 2, v3 — Joint memory–motion writer: implementation, checks, protocol

**Question.** Can a fast memory that estimates both its current contents and
their ongoing change, with separate corrections derived from accumulated
information, improve fast selective revision without damaging untouched
associations?

**What is and is not being tested.**

* The S5 study (Part 1) already tested *damped inertial prospective
  dynamics* on an SSM recurrence.
* The v2 MDN arms (`tss`, `generalized`, `qhm`, `nesterov`) *filter or shift
  the write residual* before native MDN momentum. They are preserved
  unchanged (`rules.py`), but the eight-seed v2 rerun of those arms will
  **not** be launched.
* v3 replaces the memory update itself with writers that carry their own
  information state.
* The small nonlinear experiments
  (`prospective_rtrl_experiment/results/moving_memory_tracker/`) support
  tracking under constant drift. They do not establish a benefit in a
  trained memory architecture, which is what v3 tests.
* The joint update is a constant-velocity Kalman / trajectory Gauss–Newton
  construction. We test its usefulness as a learned memory layer; we do not
  claim to have invented Kalman filtering.

Status: **implementation and CPU verification complete; GPU verification,
profiling, development runs and confirmation runs pending.** No v3 training
result exists.

---

## 1. Arms

All six run inside the restored official MDN mixer (`mixer.py`), in the same
small model: width 128, 2 layers, 4 heads, key dim 32, value dim 64, FP32,
whole-sequence.

| # | arm | what it is | memory update |
|---|---|---|---|
| 1 | `gated` | momentum-off gated delta rule under the common frontend; not a reproduction of the published GDN | `rules.py`, μ = 0 |
| 2 | `native` | native MDN | `rules.py` |
| 3 | `nesterov` | the v2 Nesterov writer: residual at the lookahead matrix | `rules.py` |
| 4 | `rls` | state-only full key-covariance writer (exponentially weighted RLS); not the published KDN or MesaNet | `writers.py` |
| 5 | `geometric` | the original fixed-gain geometric inertial writer | `writers.py` |
| 6 | `joint` | joint memory–motion writer | `writers.py` |

### 1.1 Shared vs algorithm-specific components

**Shared by all six:** the official q/k/v projections, causal short
convolutions (SiLU), output correction `q ← q − e^{D_log} k` on the
un-normalised q and k, the l2 normalisation of q and k, the read scale
`K^−½`, the gated RMS output norm with `g_proj`, `o_proj`, and the backbone.

**Algorithm-specific gates.** The MDN arms use the official data-dependent
gates (α, β, μ, η with the θ constraint). The writers use **none** of them.
They take the normalised, unscaled key `k̂`, never MDN's auxiliary
`p = α k̂`, and fixed per-arm hyperparameters. No native momentum or
retention is applied after a writer's update.

**Unused parameters, by design.** Every arm keeps every official parameter,
so shared initialisation is bit-identical. The ones an arm never reads get no
gradient, so AdamW leaves them at their initial values. They are tested on
CPU, recorded in each `results.json` (`unused_parameters`), and tested on GPU:

| arm | unused (each of the 2 blocks) |
|---|---|
| `rls`, `geometric`, `joint` | `a_proj, b_proj, m_proj, e_proj, A_log, dt_bias, mu_bias, Mu_log, log_factor` |
| `gated` | `m_proj, Mu_log, mu_bias` (μ forced to 0) |
| `native`, `nesterov` | none |

### 1.2 Orientation

The derivations use `W, Z ∈ ℝ^{dv×dk}` with local prediction `W k`. The
repository stores memory as `S ∈ ℝ^{dk×dv}` with `o = qᵀS` (`naive.py`). The
writers store `S = Wᵀ` and `S_z = Zᵀ`, so `W k → kᵀS`, `r aᵀ → a rᵀ` and
`W q → (qᵀS)ᵀ`. The dense FP64 references in `test_writers.py` work in the
derivation's own `W` orientation and share no code with `writers.py`. Every
test uses unequal key and value dims.

---

## 2. The three writers (`prospective/writers.py`)

### 2.1 `joint`: constant-velocity Kalman memory–motion writer

One covariance `P ∈ ℝ^{2dk×2dk}` (64×64 here) per sequence and head. It is
shared by all `dv` rows of `[W Z]`, and no global Jacobian or covariance over
the entries of `W` is ever formed.

```
first prior (declared):  W_1⁻ = 0, Z_1⁻ = 0, P_1⁻ = P_0 = diag(p0 I, p0/T0² I)
                         -- the initial values ARE the t = 1 prior; no
                            prediction step precedes the first observation
prediction (t ≥ 2):      Ŵ_t = W_{t−1} + Z_{t−1},  Ẑ_t = Z_{t−1},
                         P_t⁻ = F P_{t−1} Fᵀ / ρ,  F = [[I, I], [0, I]]
correction:              h = [k; 0], b = P⁻h, s = σ² + hᵀb, [a; c] = b/s,
                         r = v − Ŵk,  W = Ŵ + r aᵀ,  Z = Ẑ + r cᵀ,
                         P = P⁻ − b bᵀ / s
read:                    o_t = W_t q_t   (current posterior)
```

ρ discounts information: it inflates `P`, never the stored motion. The
update is used as written, with no extra scalar normalisation from the
nonlinear experiments. `P` is differentiated through during outer training,
and so is the transport. `F P Fᵀ` is formed as `(A+D) + (B+Bᵀ)`, `B + D`,
which keeps `P` **bitwise** symmetric. Nothing clips eigenvalues, symmetrises
`P` or resets a trajectory.

### 2.2 `rls`: state-only full key-covariance writer

The same observation weighting (σ²) and information-forgetting convention
(`P⁻ = P/ρ`) as the joint writer, with transition `I`, memory `W` only and
`P ∈ ℝ^{dk×dk}`. The first prior is `P_0 = p0 I`. It uses full key
correlations, not diagonal uncertainty (tested).

### 2.3 `geometric`: the original fixed-gain geometric inertial writer

Source: the `full` arm of
`prospective_rtrl_experiment/diagnostics/moving_memory_tracker.py`
(`V⁺ = V + A R`, `W⁺ = W + V⁺ + G R`, `R = (y − Wk)dᵀ`,
`d = C⁻¹k / kᵀC⁻¹k`). In posterior variables:

```
ξ_t     = (u_t k_t − v_t) dᵀ_t
W_t⁺    = u_t − G ξ_t          ← read here: o_t = W_t⁺ q_t
Z_t⁺    = Z_{t−1} − A ξ_t
u_{t+1} = W_t⁺ + Z_t⁺          (u_1 = 0, Z_0 = 0)
```

**Timing.** The tracker read `u_n` *before* its update. The layer is
read-after-update, so the read is `W_t⁺ q_t`, not `u_{t+1} q_t` (tested).
Eliminating the posterior variables recovers both of the tracker's
recurrences exactly: the two-state form and
`u_{t+1} = 2u_t − u_{t−1} − Aξ_t − G(ξ_t − ξ_{t−1})` (tested).

**`C_t`, an adaptation that needs your sign-off.** The tracker used a
*fixed* `C = 0.1 I + KKᵀ` over a public dictionary of all addresses. A
trained layer has no such dictionary. The causal analogue implemented is

```
C_t = λ C_{t−1} + k_t k_tᵀ,  C_0 = c0 I,  reference c0 = 0.1, λ = 1,
```

kept as `C⁻¹` by Sherman–Morrison and checked against a dense solve on an
explicitly accumulated `C`. Including the current key leaves `d_t` unchanged,
because `(C + kkᵀ)⁻¹k ∥ C⁻¹k` (tested). The reference point is `A = 0.05,
G = 1, c0 = 0.1, λ = 1`. Any other `(A, G)` and any `λ < 1` are *later
tuning* and are labelled that way in the development grid.

---

## 3. Verification

| check | file | status |
|---|---|---|
| joint vs dense FP64 (explicit `F`, W orientation), outputs and all states, ρ ∈ {1, .97, .9}, T0 ∈ {1, 2, 4, 16}: ≤ 1e-12 | `test_writers.py` | **passed** (CPU) |
| joint factorisation = explicit global Kalman filter over vec([W Z]) (transition `I⊗F`, observation `I⊗hᵀ`, prior `I⊗P0`), mean and covariance `I⊗P`, 7 ticks: ≤ 1e-11 | `test_writers.py` | **passed** (CPU) |
| joint timing: initial values are the first prior; t = 2 prior is one prediction; read is the posterior, not the prior | `test_writers.py` | **passed** (CPU) |
| ρ enters only `P`; motion transport is unchanged | `test_writers.py` | **passed** (CPU) |
| rls vs dense FP64 (ρ ∈ {1, .98, .9}); at ρ = 1 equals closed-form ridge least squares; covariance is full | `test_writers.py` | **passed** (CPU) |
| geometric vs dense FP64 (explicit `C_t`, dense solve) for the reference point and two tuned settings, incl. λ < 1 | `test_writers.py` | **passed** (CPU) |
| geometric: both original recurrences recovered; reads the corrected memory, not the next prediction; `d` invariant to including the current key | `test_writers.py` | **passed** (CPU) |
| `gradcheck` (FP64) through covariance and transport, w.r.t. q, k, v, all three writers | `test_writers.py` | **passed** (CPU) |
| covariance is differentiated, not detached (dL/dk changes if P is detached) | `test_writers.py` | **passed** (CPU) |
| P (joint, rls) and C⁻¹ (geometric) bitwise symmetric and positive definite at every tick: random and repeated keys, ρ ∈ {1, .99, .95}, T0 ∈ {1, 4, 16}; FP64 T = 300 and FP32 at production size (dk = 32, T = 52) | `test_writers.py` | **passed** (CPU) |
| states reset between calls; batch sequences independent | `test_writers.py` | **passed** (CPU) |
| model level (real layer source, CPU stand-ins for Triton modules): FP32 vs FP64 functional layer, all gradients, all nine arms; exact causality; identical shared init; unused parameters exactly as declared; writer gradients reach the shared frontend | `test_model_cpu.py` | **passed** (CPU) |
| task: lag bins partition `revised`; held-out suites keep semantics (targets ∉ inputs, independent lag, distractors never queried); production generator unchanged from v1 | `test_task.py` | **passed** (CPU) |
| protocol consistency (equal budgets, disjoint fresh seeds, declared primary and margin) | `test_experiment_v3.py` | **passed** (CPU) |
| writers in the real Triton-frontend layer vs FP64, outputs and every gradient; causality; shared init; unused parameters | `test_integration.py` | **pending GPU** |

CPU total: 122 passed (torch 2.15 CPU).

---

## 4. Task and endpoints (declared before any v3 run)

Production configuration, unchanged: 8 pairs, 6 revisions, 12 queries, 24
keys, 32 values, length 52. Read-after-update and query writes are kept. The
inner learner only ever receives projected values of observed input tokens:
`y` is a separate tensor, and the answer is never an input (tested).

* **Primary score** = ½ (`rev_lag_le8` + `untouched`). Both components and
  their sample counts are reported separately.
* **Non-inferiority on untouched**, margin **1.0 pp**. In a primary
  comparison, joint − comparator untouched accuracy must have a one-sided
  95 % bootstrap lower bound above −1.0 pp. A primary-score benefit that
  fails this is not claimed as a benefit.
* **Also reported:** all-revised, overall, and accuracy by actual token lag
  (2–8, 10–16, 18–24, 26–32, ≥ 34), each with counts.
  `legacy_immediate_revised` is for historical comparison only.
* **Held-out suites,** evaluated once at the end on their own fixed sets
  (`task.HELDOUT`), never used for training or selection:
  * `delay8` and `delay16`: 8 or 16 distractor writes of never-queried keys
    between the revisions and the queries.
  * `load12` and `load16`: 12 or 16 initial pairs.

  The delay suites have no lag ≤ 8 queries, so for them all-revised,
  untouched and overall are reported. The production generator is untouched
  (tested). No smooth-drift test is part of this benchmark.

## 5. Protocol (`prospective/experiment_v3.py`)

| phase | content |
|---|---|
| development | seeds 9001, 9002. **Equal budget: 6 configurations per arm.** gated/native/nesterov: lr ∈ {7e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3}. rls: lr ∈ {1e-3, 3e-3} × ρ ∈ {1, .98, .9}. joint: lr ∈ {1e-3, 3e-3} × (ρ, T0) ∈ {(1, 4), (.98, 4), (.98, 16)}. geometric: lr ∈ {1e-3, 3e-3} × (A, G) ∈ {(.05, 1) reference, (.0125, 1), (.05, .5)}. σ² = 1, p0 = 10 fixed |
| selection | per arm, highest mean dev tail primary score; ties go to grid order. `select` refuses to lock with any config missing a dev seed, and writes `locked.json` with the commit |
| confirmation | seeds 601–608, fresh. All six arms in lockstep per seed, locked settings, 4000 steps, batch 16, eval every 200 on 24×32, `--heldout` |
| analysis | `analyse`: per-seed tail-5 means; main contrasts joint − native and joint − rls; secondary contrasts explain whether gated, nesterov or geometric already suffice. Each contrast reports paired mean, 95 % bootstrap CI, exact sign test, and untouched non-inferiority. Missing and stalled runs are listed and kept |

Dev specs are committed in `prospective/specs/v3/`.

**Decision for you before development starts:** the dev grids above are a
proposal. The equal-budget rule is fixed; the grid values are not yet
established optima for this task.

## 6. GPU cost

**Pending.** `profile_gpu.py` now profiles all nine arms by default. Per arm
it reports forward, backward, optimizer step, evaluation, peak allocated and
reserved memory, and *persistent* recurrent state per sequence, which is
separate from training memory. It also gives per-seed estimates for the v2
six and the v3 six in lockstep. Persistent state per sequence (FP32, 2 layers
× 4 heads): native 128 KiB; joint 2·32·64 + 64² = 8,192 numbers per head
(256 KiB per sequence). That does not establish training throughput; the
measurement does. The historical 62-minute figure is not used.

Timing comparisons are made only within one GPU type (`provenance.gpu` is
recorded). Dev and confirmation seeds run as independent processes, one seed
per GPU, on 3090s and 4090s via `run_wave.sh` (one Slurm job per node,
children per GPU).

## 7. Launch (only after §3 GPU checks and §6 profiling)

```bash
# development: 6 arms x 2 seeds = 12 independent runs (6 lockstep configs each)
for arm in gated native nesterov rls geometric joint; do
  EXPECTED_COMMIT=<sha> SEEDS="9001 9002" bash prospective/run_wave.sh -- \
    --spec prospective/specs/v3/dev_$arm.json --experiment v3-memory-motion --heldout
done
python -m prospective.experiment_v3 select '<runs>/*/results.json' prospective/specs/v3/locked.json
python -m prospective.experiment_v3 confirm-spec prospective/specs/v3/locked.json prospective/specs/v3/confirm.json
# commit locked.json + confirm.json, then: confirmation, 8 fresh seeds
EXPECTED_COMMIT=<sha> SEEDS="601 602 603 604" bash prospective/run_wave.sh -- \
  --spec prospective/specs/v3/confirm.json --experiment v3-memory-motion --heldout
EXPECTED_COMMIT=<sha> SEEDS="605 606 607 608" bash prospective/run_wave.sh -- \
  --spec prospective/specs/v3/confirm.json --experiment v3-memory-motion --heldout
python -m prospective.experiment_v3 analyse prospective/specs/v3/locked.json '<runs>/*/results.json'
```
