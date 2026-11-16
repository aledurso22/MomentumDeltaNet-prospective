# Part 2, v2 — Six recurrent rules inside the official MDN mixer

**What this rerun tests:** prospective residual filtering *inside an official
Momentum DeltaNet mixer, in a small shared model*. It does not implement
inertial NLA on the fast matrix: the filter acts on the write residual `R_t`
before it enters the momentum accumulator, and `S` and `M` keep the official
update.

v1 (reduced mixer) is preserved, with an erratum, in
[`v1_reduced_mixer.md`](v1_reduced_mixer.md). Its numbers are not comparable
with v2 and are never aggregated with them (`aggregate.py` selects runs by
`provenance.mixer_version`).

Status: **implementation and CPU-side verification done; GPU verification,
profiling and the wave are pending.** No v2 training result exists yet.

---

## 1. What was restored

`prospective/mixer.py::ProspectiveMDN` **subclasses** the official
`fla.layers.momentum_deltanet.MomentumDeltaNet` (upstream `c6e77fa`,
unmodified). `__init__` is the official one, so every parameter and its
initialisation are official. Only `forward` is overridden, in three stages:

| stage | source | contents |
|---|---|---|
| `frontend` | copied line-for-line from the official forward | q/k/v projections → causal `ShortConvolution` (k = 4, SiLU) on each; head split; `a,b,m,e` gate projections; `log_μ = −e^{Mu_log}·softplus(m + mu_bias)`, `log_α = −e^{A_log}·softplus(a + dt_bias)`, `η = tanh(e/τ) + 1`, `β = σ(b)`; θ = arctan(η·e^{log_factor}) constraint `β ← sin²θ·β`, `log_α ← log cos²θ + log_α`; `log_μ ≥ −2`; output correction `q ← q − e^{D_log}·k` on the **un-normalised** q, k |
| backend | **the hook** | `official_chunk`: `chunk_mode_rule` exactly as the official layer calls it (native only). `reference`: in-kernel preprocessing reproduced (`q, k ← x/√(Σx²+1e-6)`, `p = α·k̂`, scale `K^−½`), then `rules.recurrent_ref(arm, …)` |
| `readout` | copied | `FusedRMSNormGated(o, g_proj(h))` (sigmoid gate) → `o_proj` |

Configuration: the official `MomentumDeltaNetConfig` defaults as the official
model passes them (`norm_eps 1e-6`, `conv_size 4`, `min_log_mu −2`,
`use_gate`, `use_output_correction`, `use_p_times_alpha`, `use_short_conv`),
with **width 128, 2 layers, 4 heads, key dim 32, `expand_v = 2` → value dim
64** per head (256 total). `recurrent_ref` carries `S, M ∈ ℝ^{K×V}` with
K ≠ V, and the v1 reshape that assumed K = V is gone.

The official *model*-level init (`normal(0, 0.02)` on the mixer's Linear and
Conv1d weights) is applied by `official_model_init_()`. The surrounding v1
backbone (embedding, pre-LN residual blocks, GELU MLP, final LayerNorm, head)
is kept unchanged and shared.

### 1.1 Recurrent-backend hook and the dispatch bypass

All six arms run the **same** frontend and readout code, and differ only in the
declared rule passed to `recurrent_ref`:
`gated`, `native`, `tss`, `generalized`, `qhm`, `nesterov`. The update
equations are unchanged from v1. One numerical-path change: Nesterov now
receives the normalised key `k̂` explicitly (`n=`), instead of recovering it as
`p/α`. It is the same quantity, but `p/α` is `0/0` once `α` underflows in
FP32 (tested).

The official forward sets `mode = 'fused_recurrent'` whenever `q_len ≤ 64`
and then asserts `mode == 'chunk'` in training. `fused_recurrent` has no
backward. At our length (52) the official layer therefore **cannot train**
(asserted in a test). `ProspectiveMDN.forward` ignores that dispatch and
always uses the declared backend. Training uses `reference`.

### 1.2 Execution scope and precision

* **Whole-sequence only.** Padding masks, caches and `cu_seqlens` raise. The
  filter arms' history (`y_{t−1}, y_{t−2}, R_{t−1}`) is not carried across
  calls, so `recurrent_ref` raises if a filter arm is given an initial state
  or asked for a final one.
* **FP32.** No autocast; `torch.backends.{cuda.matmul,cudnn}.allow_tf32 =
  False`, recorded in each `results.json`. All arms run the same dtype path.
  The official chunk kernel is used only for verification, never training.
  Its precision, inspected in the pinned tree: 91 `tl.dot` calls in
  `chunk_o/wy_fast/chunk_delta_h.py` set no precision (Triton default, TF32
  on sm_80+ unless `TRITON_F32_DEFAULT` changes it *and* the installed Triton
  honours that); `solve_tril` passes `input_precision=FLA_TRIL_PRECISION`
  explicitly (default `ieee`, unaffected by `TRITON_F32_DEFAULT`); cumsum
  kernels pass `allow_tf32=False`. Whether the default dots actually ran in
  IEEE is measured on the GPU by a probe kernel, not assumed (§3).
* **Optimizer unchanged from v1:** AdamW over all parameters with one weight
  decay. The official layer marks gate parameters `_no_weight_decay`, but the
  official trainer (flame/torchtitan `build_optimizers`) ignores the flag, so
  uniform decay *is* the official behaviour.

---

## 2. Task semantics

Production configuration, now the CLI defaults: **8 initial pairs,
6 revisions, 12 queries, 24 keys, 32 values** → length 52, vocabulary 58,
chance 1/32. A given seed produces exactly the v1 token stream (tested against
`8d39b8a:prospective/task.py`).

* **Read-after-update, kept.** The official layer writes each token before it
  reads it out. There is no query write-mask, so `[k QUERY]` steps write too.
  This is the official convention and the comparison keeps it. The task is not
  a read-only probe.
* **Targets are separate from inputs.** `y` is its own tensor. Scored
  positions are QUERY inputs, and the following input is never a value token
  (tested).
* **Metrics** (each pooled over the fixed eval set, with `n_<metric>`):

| metric | definition |
|---|---|
| `overall`, `revised`, `untouched` | as v1 |
| `rev_lag_le8` | revised, **lag ≤ 8 tokens**, from the rewrite's value token to the query's readout token |
| `rev_lag_gt8` | revised, lag > 8 |
| `legacy_immediate_revised` | v1's `immediate_revised`: lag ≤ 16 **and** first half of queries |
| `legacy_later_revised` | v1's `later_revised` |

At the production configuration `rev_lag_le8` occurs only for queries 0 and 2
on late rewrites, so it is the smallest bucket. Its count is printed every
eval and stored.

---

## 3. Verification

| check | file | status |
|---|---|---|
| coefficient map reproduces S5 golden values (1e-12) | `test_coefficients.py` | **passed** (CPU) |
| native rule = official `naive.py`, bitwise, incl. K ≠ V | `test_rules.py` | **passed** (CPU) |
| QHM ν=1 ≡ native; TSS ≡ generalized at M=γ=0; generalized(0, h, 0) ≈ native; gated ≡ native at μ=0 | `test_rules.py` | **passed** (CPU) |
| Nesterov residual = evaluation at the explicit lookahead *matrix* `L = αS − βμM` (independent float64 restatement, 1e-12) | `test_rules.py` | **passed** (CPU) |
| TSS start filter is `2R − R_prev` | `test_rules.py` | **passed** (CPU) |
| filter arms refuse state carry | `test_rules.py` | **passed** (CPU) |
| reference preprocessing = Triton l2norm semantics + `p = α k̂` | `test_rules.py` | **passed** (CPU) |
| same tokens as v1; legacy metric = v1 metric; lag recomputed independently; targets ∉ inputs | `test_task.py` | **passed** (CPU) |
| Triton default-dot precision, measured by a probe kernel vs FP64 | `test_integration.py` | pending GPU |
| frontend copy vs the unmodified official forward: output bitwise; grads no further apart than the official path's own run-to-run difference | `test_integration.py` | pending GPU |
| official layer asserts at T=52 in training (why bypass) | `test_integration.py` | pending GPU |
| recurrence: ref32 and chunk vs ref64, output + all 7 input grads, T = 52, 128 | `test_integration.py` | pending GPU |
| whole layer, reference FP32 and official chunk FP32 each vs an FP64 pure-PyTorch layer (`layer_ref.py`): output, dL/dh, every parameter grad, T = 52, 128; reference for all six arms | `test_integration.py` | pending GPU |
| official fused_recurrent forward at T=52 vs FP64 (recorded only: its l2norm is x/(‖x‖+1e-6)) | `test_integration.py` | pending GPU |
| boundary identities in the full layer (QHM bitwise, TSS≡gen bitwise, gen boundary ≤1e-6, gated) | `test_integration.py` | pending GPU |
| causality, all six arms, full model, exact | `test_integration.py` | pending GPU |
| shared tensors bit-identical across arms; extras are arm leaves only; official init present | `test_integration.py` | pending GPU |
| leaves move; TSS M, γ pinned at 0 | `test_integration.py` | pending GPU |

Model-level CPU checks (`test_model_cpu.py`) run the real official layer
source with CPU stand-ins for its Triton modules (`cpu_stub/`): every arm
against the FP64 functional layer (< 1e-4, outputs and all gradients), exact
causality, identical shared init. Plumbing only, not Triton numerics.

CPU status: 43 passed (torch 2.15 CPU, macOS).

**Error reporting and pass criterion.** Every GPU comparison is made against
FP64 and records, per tensor, the maximum absolute error, the reference norm
and the norm-relative error, for outputs, input gradients and every parameter
gradient. These go to `integration_<mode>.json`. The criterion, declared
before any GPU run, is norm-relative ≤ 1e-4 per tensor. It applies to the
FP32 reference backend always, and to the chunk kernel only when the probe
shows IEEE default dots and `FLA_TRIL_PRECISION=ieee`. Under TF32 dots no
bound for accumulated gradients has been derived, so chunk errors are
recorded but not graded. No tolerance is loosened after seeing results.

---

## 4. GPU cost

**Not yet measured.** `prospective/profile_gpu.py` (stage 4 of
`verify_gpu.sh`) builds the six models exactly as `train.py` does (one
process, all resident, FP32, batch 16). After 20 warm-up lockstep steps it
records, per arm, the median synchronised forward / backward / update time
over 50 steps and the peak allocated / reserved memory. It also times one full
eval (24 × 32) per arm and the CPU batch generation. The estimate is

    per seed = 4000 × (batch_gen + Σ_arms step) + 20 × Σ_arms eval

Fill this section from `profile.json`. The v1 figure (~62 min on a 3090) is
historical context only.

---

## 5. Training-wave configuration — **not to be launched**

> The eight-seed rerun of the six v2 arms below is **cancelled**. The arms
> are preserved, but the next main experiment is v3
> ([`v3_memory_motion_plan.md`](v3_memory_motion_plan.md)). This section is
> kept as the record of the configuration that was proposed.

| | |
|---|---|
| code | the commit `verify_gpu.sh` passed on; `EXPECTED_COMMIT` set in the launcher |
| model | width 128, 2 layers, 4 heads, K 32, V 64, official mixer, `reference` backend, FP32 |
| task | 8 pairs, 6 revisions, 12 queries, 24 keys, 32 values (defaults) |
| optimisation | AdamW lr 3e-3, wd 0.01, clip 1.0, batch 16, 4000 steps, constant LR (as v1) |
| eval | every 200 steps, fixed 24 × 32 set, pooled, with counts |
| seeds | 501–508, **one independent seed per GPU** as child processes of one Slurm job (`run_wave.sh`); no model sharding |
| output | `…/mdn-bridge-runs-v2/<stamp>-seed<s>/results.json`, `mixer_version = v2-official-mdn-mixer` |
| analysis | `python -m prospective.aggregate` (v2 only; exact sign tests) |

```bash
# 1. gate, on one RTX 4090
EXPECTED_COMMIT=<sha> sbatch --nodelist=<rtx4090-node> prospective/verify.sbatch
# 2. wave, e.g. 4 GPUs, twice
EXPECTED_COMMIT=<sha> SEEDS="501 502 503 504" bash prospective/run_wave.sh
EXPECTED_COMMIT=<sha> SEEDS="505 506 507 508" bash prospective/run_wave.sh
```

---

## 6. Claims carried from v1, corrected

* TSS starts at `Y_t = 2R_t − R_{t−1}`, not at the native identity filter.
* The Jury conditions certify the isolated filter `z² − az + b`, not the
  memory feedback loop.
* `M = γ = 0` vs learned `(M, γ, T)` does not isolate damping.
* `gated` is the momentum-off gated-delta-rule ablation under this common
  frontend, not a reproduction of the published Gated DeltaNet.

Remaining known differences from the full official model: the backbone
(LayerNorm + GELU MLP, not RMSNorm + SwiGLU), and the model's scale and task.
No coefficient sweeps or optimizer variants were added in this restoration.
