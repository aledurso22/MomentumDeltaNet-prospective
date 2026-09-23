# Part 2 — The Momentum DeltaNet bridge: protocol, equations, implementation, results

*Part 1, the three S5 arms, is `docs/S5_THREE_ARM_REFERENCE.md` in
`github.com/aledurso22/final-prospective-s5`.*

The same second-order prospective filter, moved from an SSM **recurrence** to a
linear-attention **write residual**, in the official Momentum DeltaNet codebase.
Written to be read against the source; every code block is copied from the tree.

Upstream base: `HuuYuLong/MomentumDeltaNet` at **`c6e77fa261fb0c002fae1a14b6209a5b28d2edc9`**
(ICML 2026), pushed here as branch `upstream-base` for diffing. Our work is
branch `prospective-rules`; `fla/` is untouched.

---

## 1. Why this experiment exists

Part 1 measured the law on the S5 recurrence and found it costs 1.00 pp. The
six-arm ladder on a synthetic 8×8 fast weight measured it on the write residual
and found +7.53 pp on immediate revision. **Those two results differ in two
variables at once** — architecture *and* placement — so neither transfers.

This bridge holds architecture fixed at the real MDN and placement fixed at the
drive, leaving one question: does the effect survive in a real architecture?

It answers "is the effect real", **not** "is this practical at scale" (§6).

---

## 2. The native rule we branch from

`fla/ops/momentum_delta_rule/naive.py`, the repository's own recurrent
reference. Four lines:

```python
w_t = -(v_t.unsqueeze(-2) - p_t.unsqueeze(-2) @ S_prev)   # write residual
Mt  = mu_i * M_prev + (eta_i * k_t).unsqueeze(-1) @ w_t   # momentum on it
St  = alpha_i * S_prev - beta_i * Mt                      # state update
out[:, i] = (q_t.unsqueeze(-1) * St).sum(-2)              # read out
```

Orientation notes that matter when reading the diff:

* `S` is **K×V** here (Part 1's `W` was V×K).
* `alpha` is folded into `p = alpha · Norm(k)` *before* the kernel, so the
  residual is already evaluated at the decayed state.
* `eta` is folded into `k` **before** the outer product.
* `alpha` forget, `beta` write strength, `mu` momentum decay, `eta` momentum
  gain — all four are token-dependent gates.

---

## 3. The six arms

Each changes **exactly one** of those four lines. `prospective/rules.py`:

```python
ARMS = ("gated", "native", "tss", "generalized", "qhm", "nesterov")
FILTER_ARMS = ("tss", "generalized")
NO_MOMENTUM_ARMS = ("gated",)
```

```python
if arm in NO_MOMENTUM_ARMS:
    log_mu = torch.full_like(log_mu, float("-inf"))       # gated: mu = 0

for i in range(T):
    ...
    if arm == "nesterov":                       # ← LINE 1: where R is evaluated
        n_t = (p_t / alpha_i.view(B, H, 1)).unsqueeze(-2)
        w_t = -(v_t.unsqueeze(-2) - p_t.unsqueeze(-2) @ S_prev
                + (beta_i * mu_i) * (n_t @ M_prev))
    else:
        w_t = -(v_t.unsqueeze(-2) - p_t.unsqueeze(-2) @ S_prev)

    R_t = k_t.unsqueeze(-1) @ w_t               # rank-1, eta NOT folded in

    if arm in FILTER_ARMS:                      # ← LINE 2: the momentum drive
        y_t = a * y_prev - b * y_prev2 + c * R_t - d * R_prev
        y_prev2, y_prev = y_prev, y_t
        M_t = mu_i * M_prev + eta_i * y_t
    else:
        M_t = mu_i * M_prev + (eta[:, i].unsqueeze(-1) * k_t).unsqueeze(-1) @ w_t

    if arm == "qhm":                            # ← LINE 3: the final combine
        S_t = alpha_i * S_prev - beta_i * (nu * M_t + (1.0 - nu) * eta_i * R_t)
    else:
        S_t = alpha_i * S_prev - beta_i * M_t
```

| arm | line | change | extra params | extra carry |
|---|---|---|---|---|
| `gated` | — | `mu := 0`, momentum carry off | 0 | 0 |
| `native` | — | their expression, unmodified | 0 | 0 |
| `tss` | drive | `R_t → y_t`, at `M = γ = 0` | 1 (`T`) | `y, y_prev, R_prev` |
| `generalized` | drive | `R_t → y_t`, all three free | 3 | `y, y_prev, R_prev` |
| `qhm` | combine | mixes `M_t` with raw `η R_t` | 1 (`ν`) | 0 |
| `nesterov` | residual | evaluated at lookahead `L_t` | 0 | 0 |

### 3.1 Three implementation details that are load-bearing

**`R_t` deliberately excludes `eta`.** The official line folds `η` into `k`
before the outer product. `η` varies per token, so filtering `ηR` is *not*
`η·filter(R)`. The filter must see the unscaled residual, with `η` applied
after. The `else` branch keeps their exact expression, which is what makes the
bitwise checks in §5 possible.

**Nesterov carries `p_t / alpha_i`.** The lookahead is
`L_t = α S_prev − β μ M_prev`, but `α` is already inside `p`, so the correction
term needs it divided back out.

**Gated DeltaNet needs no second implementation.** The official `naive.py` says
so itself — *"For GatedDeltaNet: p = alpha * Norm(k), mu = 0"*. It is the
momentum rule with the carry switched off, which is why it costs zero extra
code and serves as the calibration arm.

---

## 4. The coefficient map — imported from Part 1, not rewritten

`prospective/coefficients.py`:

```python
def prospective_coefficients(mass, gamma, response, h=H_TOKEN):
    """(a, b, c, d) from (M, gamma, T). Formed once, no branch on values."""
    q = mass + h * (gamma + response)
    alpha, beta, delta = mass / q, (h * h) / q, (h * response) / q
    return (1.0 + alpha - beta, alpha, beta + delta, delta)


def tss_coefficients(response, h=H_TOKEN):
    """The M = gamma = 0 face. Kept as a call into the SAME map so the two
    arms cannot drift apart."""
    zero = torch.zeros_like(response)
    return prospective_coefficients(zero, zero, response, h)
```

**TSS is a call into the generalized function**, not a second equation. Checked
against golden values generated from `s5/discrete_recurrence.py` to **1e-12**,
and separately checked to reduce to literal Eq. (17):

```
a = 1 − h/T,    b = 0,    c = 1 + h/T,    d = 1
```

### 4.1 Recurrence placement vs drive placement

Both studies run the identical difference operator

```
y_t = (1 + α − β) y_{t-1} − α y_{t-2} + (β + δ) u_t − δ u_{t-1}
```

with `α = M/q`, `β = h²/q`, `δ = hT/q`, `q = M + h(γ+T)`. **The entire
difference is what `u` is:**

| | target `u_t` | consequence |
|---|---|---|
| **S5, Part 1** | `Y_t = F s_{t-1} + G x_t`, `F = 1 + (T/h)(λ̄−1)` | `λ̄` enters `a₁, a₂`; prospective parameters and SSM mode share one characteristic polynomial |
| **MDN, here** | `R_t`, the write residual | `a, b, c, d` depend on `(M, γ, T)` **only**; the gates never enter; memory recurrence untouched |

That gate-independence is why the Jury conditions can be certified at a
checkpoint, and why this arm is comparable in cost to a two-tap operator rather
than harder.

### 4.2 Admissible set and the projection

```python
G_MIN = 2.0 ** -10          # gamma + T >= G_MIN * h
DELTA_FILTER = 1e-3         # 4M + 2h(gamma+T) >= h^2 (1 + DELTA_FILTER)

@torch.no_grad()
def project_leaves(mass, gamma, response, h=H_TOKEN, freeze_mass_gamma=False):
    if freeze_mass_gamma:
        mass.zero_(); gamma.zero_()
    else:
        mass.clamp_(min=0.0); gamma.clamp_(min=0.0)
    response.clamp_(min=0.0)
    deficit = (G_MIN * h) - (gamma + response)
    response.add_(deficit.clamp(min=0.0))
    need = (h * h) * (1.0 + DELTA_FILTER) - 4.0 * mass - 2.0 * h * gamma
    response.add_((need / (2.0 * h) - response).clamp(min=0.0))
```

Of the three Jury conditions on `z² − az + b`, two hold identically:
`1 − a + b = β > 0` and `1 − b = 1 − α > 0`. Only
`1 + a + b > 0 ⟺ 4M + 2h(γ+T) > h²` can bind, and that is what the repair
enforces. For TSS (`M = γ = 0`) it reduces to `T > h/2`, the drive-side
stability bound.

---

## 5. The parameterization bug, and why it matters methodologically

The first eight-seed-capable run produced a flat null. The cause was mine, and
it is worth recording because it is a general trap.

Original: `M = softplus(raw_M)` with `raw_M` initialized so `M ≈ 1e-4`, putting
the arm exactly at the native boundary so it would "have to earn its
difference". But

```
d(softplus)/d(raw) = sigmoid(raw),   and at M ≈ 1e-4 that is itself ≈ 1e-4
```

so the gradient reaching `M` is scaled by 1e-4 and its effective learning rate
sits four orders of magnitude below the rest of the model. Measured over 4000
steps: **`M` went 1e-4 → 3e-4 and `γ` 1e-4 → 3e-4.** The generalized arm ran as
TSS for the entire run — the three arms that were supposed to differ were one
model under three names, which is exactly why `native`, `tss` and `generalized`
landed within a point of each other. QHM had the same fault: `ν` started at
`sigmoid(8)` (derivative 3.3e-4) and ended at 0.9964.

**No smooth positive reparameterization avoids this** — every one flattens as
the leaf approaches zero. The fix is the one the earlier ladder already used:
**store the leaves directly and project after each optimizer step.**

```python
self.fil_T     = nn.Parameter(torch.full((n_heads,), GEN_T0 if gen else TSS_T0))
self.fil_M     = nn.Parameter(torch.full((n_heads,), GEN_M0 if gen else 0.0),
                              requires_grad=gen)
self.fil_gamma = nn.Parameter(torch.full((n_heads,), GEN_GAMMA0 if gen else 0.0),
                              requires_grad=gen)
```

```python
opts[a].step()
models[a].project_()          # leaves are stored directly
```

Effect, measured over **40** steps after the fix: `M` 0.025 → 0.057, `ν` → 0.961.
Before the fix, 4000 steps moved `M` by 2e-4.

**The cost of the fix, stated plainly:** generalized no longer starts at the
native boundary. It starts at `GEN_M0 = 0.025, GEN_GAMMA0 = 1.0, GEN_T0 = h`,
which is the S5 study's successful point. TSS and QHM still start exactly at
their native points, because direct storage has no trap there. Three regression
tests cover this: the leaves move under optimization, TSS stays pinned at
exactly zero, and projection leaves the coefficients Jury-stable from arbitrary
negative starts.

---

## 6. Which execution path, and why not the fast one

The MDN repository has three paths. We use the slowest, deliberately:

| path | file | differentiable | used |
|---|---|---|---|
| PyTorch token loop | our `rules.py`, mirroring `naive.py` | yes, via autograd | **yes** |
| recurrent Triton | `fused_recurrent.py` | **no** | no |
| chunkwise Triton | `chunk.py` | yes | no |

The original plan was to modify `fused_recurrent` only. That is impossible:

```python
@staticmethod
@input_guard
def backward(ctx, do, dst, dmt):
    raise NotImplementedError(
        "Backward pass is not implemented yet and we do not have plans to "
        "implement it because we haven't figured out how to compute dg "
        "without materializing the full hidden states for all time steps.")
```

`fused_recurrent_mode_rule` is a **decoding** path with no backward, and
upstream states no intention of adding one. So training goes through the
PyTorch token loop, where autograd supplies the backward.

**Consequence, and the honest limit of this study:** a Python per-token loop
with retained activations bounds the model size and context. This is the
correct *mathematics* in the official architecture at small scale, **not** a
fast implementation, and it says nothing about throughput or memory at 400M.
Writing `chunk.py` kernels is the work that follows a positive go/no-go — and
Nesterov is the arm that breaks first there, since moving the residual
evaluation point is what invalidates the WY/UT derivation.

### 6.1 Equivalence against the official reference

`prospective/test_rules.py` loads `naive.py` by path (the `fla/__init__` pulls
`transformers`, which this check does not need) and compares:

| check | result |
|---|---|
| our `native` path vs theirs | **bitwise 0** |
| QHM at `ν = 1` | **bitwise 0** |
| Nesterov at `μ → 0` | **bitwise 0** |
| `tss` path ≡ `generalized` path at `M = γ = 0` | **bitwise 0** |
| generalized at native boundary (`M=0, γ=h, T=0`) | `< 1e-6` relative, float32 |
| gradients reach `M`, `γ`, `T` | yes |

The three bitwise rows are the arms whose expression is character-identical to
theirs. The `1e-6` row is **not** a loosened tolerance hiding a defect: the
filter path multiplies by `η` *after* the outer product where the official line
folds `η` into `k` first, and the official reference force-casts every input to
float32. Both facts are documented at the assertion.

18 tests total, all passing.

---

## 7. Task, model, training

### 7.1 MQAR with revision

`prospective/task.py`. Plain retrieval would miss the effect: the earlier
ladder moved immediate revision by +7.53 while retention and recall went
slightly the *other* way. So the probe writes key–value pairs, **rewrites** some
of them, then queries.

```
token layout:  [k v] × n_pairs   [k v'] × n_revisions   [k QUERY] × n_queries
```

| metric | scored on | the ladder's name |
|---|---|---|
| `overall` | all queries | recall |
| `revised` | keys that were overwritten | revision |
| `untouched` | written once, never touched | retention |
| `immediate_revised` | revised, rewrite ≤ `IMMEDIATE_GAP = 8` tokens back | immediate revision |
| `later_revised` | revised, rewrite further back | later revised |

Production settings: `n_keys 24`, `n_pairs 8`, `n_revisions 6`, `n_queries 12`,
`n_values 32` → sequence length **52**, vocabulary **58** (24 keys + 32 values +
`QUERY` + `PAD`). Chance on a query is 1/32 ≈ **3.1%**.

An earlier configuration used `n_keys 64, n_pairs 16` and produced ~13% overall
after 4000 steps — the model was failing the *memory* task, leaving no headroom
for a *revision* effect. That run is excluded by config, not by hand (§8.2).

### 7.2 Model

`prospective/model.py`. Deliberately minimal; everything outside the mixer is
shared across arms.

```python
Model(arm, vocab, d_model=128, n_heads=4, n_layers=2)
  embed  → [Block × 2] → LayerNorm → Linear head
  Block  = h + Mixer(LN(h));  h + MLP(LN(h))
  Mixer  = q,k,v projections (k L2-normalized), 4 gates from one Linear,
           recurrent_ref(arm, ...), output projection
```

Gates, per head per token:

```python
log_alpha = -F.softplus(g[..., 0])     # forget
log_mu    = -F.softplus(g[..., 1])     # momentum decay
beta      = torch.sigmoid(g[..., 2])   # write strength
eta       = torch.sigmoid(g[..., 3])   # momentum gain
p         = log_alpha.exp().unsqueeze(-1) * k
```

### 7.3 Shared-init verification

```python
def build(arm, vocab, seed, **kw):
    torch.manual_seed(seed)
    return Model(arm, vocab, **kw)
```

Verified: **zero differing tensors** across all six models at initialization, on
every tensor they share. Parameter counts differ only by each arm's own leaves
(3 per head per layer for the filter arms, 1 for QHM, 0 otherwise).

### 7.4 Training

`prospective/train.py`. **All six arms are stepped inside one loop on one
batch:**

```python
for step in range(1, args.steps + 1):
    b = batch(args.bsz)                       # ONE batch, all arms see it
    for a in arms:
        loss, _ = loss_of(models[a], b)
        opts[a].zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(models[a].parameters(), args.clip)
        opts[a].step()
        models[a].project_()
```

So a difference between two arms cannot come from data order, initialization or
schedule.

| setting | value |
|---|---|
| optimizer | AdamW, `lr = 3e-3`, `weight_decay = 0.01` |
| gradient clip | 1.0 (global norm) |
| batch | 16 |
| steps | 4000 |
| eval | every 200 steps, 24 batches × 32 sequences, **fixed** eval set |
| seeds | 501–508 |
| schedule | none — constant LR |

No LR schedule and no warm-up. That is a real difference from Part 1 and a fair
thing to question (§9).

### 7.5 Execution

`prospective/run_wave.sh` — **one Slurm job, one child process per GPU**:

```bash
CUDA_VISIBLE_DEVICES="$token" \
  python -u -m prospective.train --seed "$seed" --steps "$STEPS" ... &
```

This is not a stylistic choice. Three separate `sbatch` jobs on the same node
ended with the first `COMPLETED` and the other two `FAILED` at the *identical
elapsed second* — node cleanup for the finishing job takes its co-resident
neighbours with it. The launcher refuses if seeds outnumber GPU tokens and
exits non-zero if any child fails, so a partial wave cannot be read as a
complete one.

Wall clock: 4000 steps × six arms ≈ **62 min** on one RTX 3090; four seeds in
parallel on four GPUs, so eight seeds is two hours.

---

## 8. Results, eight seeds

### 8.1 Per arm

Mean over seeds of each seed's final-five-eval mean. "Escaped" counts seeds
finishing above 5% (chance is 3.1%).

| arm | imm. revised | later revised | revised | untouched | overall | escaped |
|---|---|---|---|---|---|---|
| Literal Nesterov | 15.15 | 14.28 | 14.67 | 11.69 | **13.18** | **8/8** |
| Generalized (M, γ, T) | 15.11 | 14.06 | 14.53 | 11.53 | 13.03 | 7/8 |
| Native MDN | 12.79 | 12.29 | 12.51 | 10.30 | 11.40 | 7/8 |
| QHM | 10.95 | 10.59 | 10.74 | 8.91 | 9.83 | 7/8 |
| Gated DeltaNet | 4.33 | 4.38 | 4.36 | 4.01 | 4.18 | 1/8 |
| **Zucchet / TSS** | **3.47** | 3.52 | 3.50 | 3.42 | 3.46 | **0/8** |

### 8.2 Paired contrasts, exact sign test

The outcome is **bimodal** — a seed either escapes the plateau or sits at
chance — so a t-test's assumptions do not hold and the headline test is the
exact one-sided sign test at 0.5. Means are reported beside it, not instead.

Generalized minus, on immediate revision:

| contrast | mean Δ | wins | p |
|---|---|---|---|
| Zucchet / TSS | +11.64 | **8/8** | **0.0039** |
| Gated DeltaNet | +10.78 | **8/8** | **0.0039** |
| QHM | +4.16 | 6/8 | 0.1445 |
| Native MDN | +2.32 | 3/8 | 0.8555 |
| Literal Nesterov | −0.04 | 5/8 | 0.3633 |

Against Native, which is the baseline that matters:

| minus native | mean Δ | wins | p |
|---|---|---|---|
| Literal Nesterov | +2.37 | 4/8 | 0.6367 |
| Generalized | +2.32 | 3/8 | 0.8555 |
| QHM | −1.84 | 5/8 | 0.3633 |
| Gated DeltaNet | −8.46 | 1/8 | 0.9961 |
| Zucchet / TSS | −9.32 | **0/8** | 1.0000 |

Read the other way: **native beats TSS 8/8 at p = 0.0039** and Gated DeltaNet
7/8 at p = 0.0352.

`prospective/aggregate.py` selects runs by **config**, not path — the runs
directory also holds the 64-key study and the 800-step probe — and lists what it
skipped rather than averaging it in.

### 8.3 What this establishes

**The undamped arm never trains.** 0/8 seeds above chance; generalized beats it 8/8 at
p = 0.0039 on *all five* metrics. This is the controlled comparison: same code
path, same `T` initialization, the only difference being whether `M` and `γ` may
leave zero.

**The law does not beat the baseline.** Native wins 5 of 8 seeds, and the
mean +2.32 rides on two seeds where *native* failed to escape. Generalized −
native is 3/8, 3/8, 3/8, 5/8, 4/8 across the five metrics — never significant,
never even consistent in sign. And generalized is indistinguishable from literal
Nesterov, mean difference −0.04.

**A three-seed read of this same experiment showed generalized ahead by 6 pp.**
Eight seeds show that was the escape lottery. Recorded because it is the clearest
available argument for why the second wave was worth an hour.

**Calibration passed.** Gated DeltaNet is beaten by native at p = 0.0352, which
is the direction the MDN paper reports.

### 8.4 Why TSS fails, mechanistically

The filter's DC gain is exactly 1 in every configuration. What differs is the
high-frequency response:

| arm | at learned parameters | Nyquist / DC |
|---|---|---|
| TSS (`M = γ = 0`), `T = 0.79–1.28` | high-pass | **2.3 – 4.5×** |
| Generalized, `M = 0.05–0.47`, `γ = 0.9–1.9` | low-pass | **0.44 – 0.98×** |

and **damping, not mass, does the work** — the same conclusion Part 1 reaches
analytically, where at `λ̄ = 1` the S5 recurrence factors as
`(z−1)(z−(α+δ))` with `α+δ = (M+hT)/(M+hT+hγ)`, which equals 1 for *any* mass
whenever `γ = 0`:

```
gamma=1, T=1:   M=0.00 → 1.00    M=0.10 → 0.88    M=0.40 → 0.65
M=0,     T=1:   gamma=0 → 3.00   gamma=1 → 1.00
```

TSS is pinned at *both* zero, so it gets neither. It amplifies the fastest
component of the residual stream by 2.3–4.5× while holding DC at 1 — the same
failure family as the matched-`Γ_k` S5 arm, whose Nyquist/DC was 1143× at the
median mode.

**Why it scored 79.17 in the earlier ladder and 3.47 here** — a hypothesis, not
a result. That ladder **continued from a sha256-verified pretrained native
checkpoint** for 200 updates. The memory already worked, so `R` was small and
structured and a 3× boost of its fast component cost little while the lead term
bought real anticipation. Here, training is from random initialization for 4000
steps: `R` is large and spiky (a rank-1 outer product that jumps at every write
and is masked to zero on queries), and amplifying that into the momentum
accumulator prevents the memory from forming at all.

The gain arithmetic is solid; the causal link to from-scratch failure is
inference. **It is directly testable** with this harness: run TSS from a
pretrained native checkpoint. If it recovers, the explanation holds.

---

## 9. What a referee should push on

1. **From-scratch, not continuation.** The result that motivated this bridge
   came from continuation training. §8.4 argues that difference explains the
   reversal — which means the bridge does not actually replicate the original
   setting, and the continuation arm has not been run.
2. **No LR schedule, no warm-up**, unlike Part 1. Constant `3e-3` throughout.
   The plateau-then-escape structure (all arms at chance until ~step 600) may be
   an optimization artifact that a warm-up or a lower LR would remove.
3. **Bimodal outcomes.** Reported honestly via the sign test, but it means the
   *means* in §8.1 are not a summary of a unimodal distribution and should not
   be read as one.
4. **Five metrics, no pre-registered primary.** Generalized − QHM reaches
   7/8, p = 0.035 on three of the five metrics but only 6/8, p = 0.14 on
   immediate revision. With five correlated metrics in play that pattern needs a
   declared primary before it is claimed.
5. **Generalized does not start at native**, since a boundary start cannot train
   (§5). TSS and QHM do. The arms therefore do not all begin identical to
   baseline.
6. **Tiny model, short context.** `d_model 128`, 2 layers, sequence length 52,
   because the token loop bounds it. Nothing here speaks to 400M or to long
   context.
7. **The task is synthetic.** MQAR-with-revision, not language modelling. LM
   validation loss was never measured, and §7.1 argues it would be the wrong
   endpoint — but that argument is ours, not established.
8. **`ESCAPE = 5.0` is a threshold we chose.** It sits well above chance (3.1%)
   and the arms are far from it in both directions, but it is a free parameter
   in the "escaped" column. The sign tests do not depend on it.

---

## 10. Combined conclusion

Across Part 1 and Part 2, four settings: two architectures × two placements of
the same coefficient map.

* **Zero damping is unusable** (the arms are at `M = γ = 0`, but the binding
  variable is `γ`). Non-finite on the first update in S5
  (companion roots 1.16–1.63 at `T = 0.05`); 0/8 seeds above chance in MDN, beaten
  8/8 at p = 0.0039.
* **Finite mass and damping makes the law run**, and the learned regime is the
  predicted one in S5 — `|r_slow| ≈ 0.999`, `|r_fast| ≈ 0.05`, 55–62 of 64 modes,
  with `γ` rising away from the stability boundary.
* **In neither architecture does it beat the baseline.** −1.00 pp against Native
  S5 at 3/3 seeds; statistically indistinguishable from Native MDN at 8 seeds,
  and from literal Nesterov.

What mass and damping buy is a law that runs at all rather than one that
diverges at initialization or never leaves chance — the difference between two
named points on one coefficient map, not an advantage over the model they are
added to.
