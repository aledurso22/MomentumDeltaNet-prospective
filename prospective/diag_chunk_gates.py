"""Targeted diagnostic for test_integration.py::test_official_chunk_layer_vs_fp64
(IEEE mode, T = 52 and 128): the whole-layer chunk outputs agree with FP64
to ~5e-6, but d/d A_log, dt_bias reach ~6e-3 and d/d log_factor 1.7e-2.

Question: is this a BACKWARD error of the chunk kernel, or rounding noise
amplified by CANCELLATION in the sum over tokens that forms a gate
parameter's gradient? Tolerances are not touched; this only measures.

Same inputs everywhere. The test's layer, hidden states and upstream
gradient are rebuilt with the test's seeds. Then:

1. Reproduce the failing comparison in this process (anchor).
2. Fix ONE set of recurrence inputs: the FP64 frontend outputs, cast to FP32
   (the FP64 truth uses exactly these values, promoted). Fix ONE upstream
   gradient: the actual full-layer gradient at the recurrence output, dL/do,
   from the FP64 readout at the exact recurrence output.
3. Recurrence input gradients from these implementations on those inputs:
       chunk            the official chunk kernel, as shipped
       chunk_epsfree    the same, except that the backward-only regulariser
                        in prepare_uyz_repr_bwd_kernel,
                            d_x = -d_log_c_jt (1 - x) / (x + 1e-6),
                        is replaced by the exact  / x  (x = 1 - exp(...);
                        0 where x == 0). The forward has no such eps.
       reference_fp32   our token loop in FP32
       truth_fp64       our token loop in FP64 (the official equation)
       truth_fp64_beta_eps  the same at beta + 1e-6. The chunk kernel uses
                        beta ONLY as log(beta + 1e-6) (forward) and
                        log(beta + 1e-6), 1 / (beta + 1e-6) (backward), i.e.
                        it implements exactly this function. Comparing with
                        it separates a MODEL difference from kernel error.
4. Propagate each set through the EXACT FP64 frontend vector-Jacobian
   product to the parameters, so the recurrence backward is the only
   difference; then substitute the chunk gradient of ONE input at a time
   (attribution).
5. Cancellation, for A_log, dt_bias, log_factor: per-token contributions
   c_bt = sum_X dL/dX_bt * dX_bt/dp (FP64, forward-mode), the cancellation
   factor kappa = sum|c| / |sum c|, and how each implementation's per-token
   errors e_bt add up: |sum e| against the random-walk scale ||e||_2 and the
   coherent bound sum|e|.
   Also corr(e, c): errors proportional to the true contributions vs not.
6. The x = 1 - exp(log_c_{j-1} - log_ct_i) distribution on the chunk's valid
   lower triangle (where eps / x and FP32 cancellation in 1 - exp matter).

    TRITON_F32_DEFAULT=ieee python -m prospective.diag_chunk_gates --T 52 128 --out DIR
"""

import argparse
import contextlib
import importlib.util
import json
import os
import time

import torch
import triton
import triton.language as tl

import fla.ops.momentum_delta_rule.wy_fast as WY

from .layer_ref import frontend_ref, mixer_forward_ref, params_as, readout_ref
from .mixer import ProspectiveMDN
from .reference import reference_recurrence
from .train import set_fp32

DEV = torch.device("cuda")
F64 = torch.float64
D_MODEL, N_HEADS, HEAD_DIM, EXPAND_V = 128, 4, 32, 2
TOL = 1e-4                       # the declared tolerance, unchanged
INPUTS = ("q", "k", "v", "log_alpha", "log_mu", "beta", "eta")
GATE_PARAMS = ("A_log", "dt_bias", "log_factor")
BT = 64
BETA_EPS = 1e-6                  # the kernel's log(beta + 1e-6)
REGULARISED = ("    b_d_x = tl.where(mask_tril, - b_d_log_c_jt * (1 - b_x) / "
               "(b_x + eps), 0.0)")
EXACT = ("    b_d_x = tl.where(mask_tril & (b_x != 0), - b_d_log_c_jt * (1 - b_x)"
         " / tl.where(b_x != 0, b_x, 1.0), 0.0)")


# ----------------------------------------------------------- as in the test
def layer():
    torch.manual_seed(0)
    m = ProspectiveMDN("native", hidden_size=D_MODEL, num_heads=N_HEADS,
                       head_dim=HEAD_DIM, expand_v=EXPAND_V,
                       backend="official_chunk")
    m.official_model_init_()
    return m.to(DEV)


def hidden(B, T, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(B, T, D_MODEL, device=DEV, generator=g)


def upstream(T, seed):
    return torch.randn(4, T, D_MODEL, device=DEV,
                       generator=torch.Generator(device=DEV).manual_seed(seed + 1))


def err(x, ref):
    x, ref = x.detach().double(), ref.detach().double()
    d = x - ref
    nr = ref.norm().item()
    return dict(max_abs=d.abs().max().item(), ref_norm=nr,
                norm_rel=d.norm().item() / nr if nr > 0 else float("nan"),
                nonfinite=int((~torch.isfinite(x)).sum()))


@triton.jit
def _dot_probe(a_ptr, b_ptr, c_ptr, N: tl.constexpr):
    o = tl.arange(0, N)
    a = tl.load(a_ptr + o[:, None] * N + o[None, :])
    b = tl.load(b_ptr + o[:, None] * N + o[None, :])
    tl.store(c_ptr + o[:, None] * N + o[None, :], tl.dot(a, b))


def dot_mode():
    g = torch.Generator(device=DEV).manual_seed(0)
    a = torch.randn(32, 32, device=DEV, generator=g)
    b = torch.randn(32, 32, device=DEV, generator=g)
    c = torch.empty_like(a)
    _dot_probe[(1,)](a, b, c, N=32)
    e = err(c, a.double() @ b.double())["norm_rel"]
    return ("ieee" if e < 1e-5 else "tf32"), e


# ------------------------------------------------- the eps-free variant
def load_epsfree_kernel(out_dir):
    """A copy of wy_fast.py with ONLY the regularised d_x line made exact,
    imported from a real file (Triton needs the source)."""
    src = open(WY.__file__).read()
    if src.count(REGULARISED) != 1:
        raise SystemExit("the regularised d_x line was not found exactly once "
                         "in " + WY.__file__)
    path = os.path.join(out_dir, "wy_fast_epsfree.py")
    with open(path, "w") as fh:
        fh.write(src.replace(REGULARISED, EXACT))
    spec = importlib.util.spec_from_file_location("diag_wy_fast_epsfree", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.prepare_uyz_repr_bwd_kernel


@contextlib.contextmanager
def bwd_kernel(kernel):
    old = WY.prepare_uyz_repr_bwd_kernel
    WY.prepare_uyz_repr_bwd_kernel = kernel      # the wrapper looks it up here
    try:
        yield
    finally:
        WY.prepare_uyz_repr_bwd_kernel = old


# ------------------------------------------------------------- one length
def run_T(m, T, epsfree, out_dir):
    seed = 20 + T
    h, g = hidden(4, T, seed), upstream(T, seed)
    res = dict(T=T)

    # 1. anchor: the failing comparison, reproduced
    h32 = h.clone().requires_grad_(True)
    m.zero_grad(set_to_none=True)
    m(h32)[0].backward(g)
    P = params_as(m, F64)
    mixer_forward_ref(m, h.double(), P).backward(g.double())
    res["anchor_layer"] = {n: err(p.grad, P[n].grad)["norm_rel"]
                           for n, p in m.named_parameters()
                           if P[n].grad is not None and p.grad is not None}

    # 2. fixed inputs and the actual full-layer upstream at the recurrence output
    P = params_as(m, F64)
    h64 = h.double()
    f64 = frontend_ref(m, h64, P)                          # graph to P
    f_in = {k: f64[k].detach().float() for k in INPUTS}    # the common inputs
    leaves64 = {k: f_in[k].double().requires_grad_(True) for k in INPUTS}
    o_true = reference_recurrence("native", leaves64, {})
    o_leaf = o_true.detach().requires_grad_(True)
    dO = torch.autograd.grad(readout_ref(m, o_leaf, h64, P), o_leaf, g.double())[0]
    res["upstream_norm"] = dO.norm().item()

    # 3. recurrence gradients, four implementations
    grads, outs = {}, {}
    truth = torch.autograd.grad(o_true, [leaves64[k] for k in INPUTS], dO)
    grads["truth_fp64"] = dict(zip(INPUTS, truth))
    lv = {k: f_in[k].double().requires_grad_(True) for k in INPUTS}
    o_eps = reference_recurrence("native", dict(lv, beta=lv["beta"] + BETA_EPS), {})
    outs["truth_fp64_beta_eps"] = o_eps
    grads["truth_fp64_beta_eps"] = dict(zip(INPUTS, torch.autograd.grad(
        o_eps, [lv[k] for k in INPUTS], dO)))
    lv = {k: f_in[k].clone().requires_grad_(True) for k in INPUTS}
    o = reference_recurrence("native", lv, {})
    outs["reference_fp32"] = o
    grads["reference_fp32"] = dict(zip(INPUTS, torch.autograd.grad(
        o, [lv[k] for k in INPUTS], dO.float())))
    for name, ctx in (("chunk", contextlib.nullcontext()),
                      ("chunk_epsfree", bwd_kernel(epsfree))):
        with ctx:
            lv = {k: f_in[k].clone().requires_grad_(True) for k in INPUTS}
            o = m.backend_official_chunk(lv)
            outs[name] = o
            gs = torch.autograd.grad(o, [lv[k] for k in INPUTS], dO.float())
        grads[name] = dict(zip(INPUTS, gs))
    res["forward_output"] = {n: err(o, o_true) for n, o in outs.items()}
    res["forward_output_vs_beta_eps_model"] = {
        n: err(outs[n], o_eps) for n in ("chunk", "chunk_epsfree", "reference_fp32")}
    res["chunk_forward_identical_across_variants"] = bool(
        torch.equal(outs["chunk"], outs["chunk_epsfree"]))
    res["recurrence_grads"] = {
        n: {k: err(grads[n][k], grads["truth_fp64"][k]) for k in INPUTS}
        for n in ("chunk", "chunk_epsfree", "reference_fp32", "truth_fp64_beta_eps")}
    res["recurrence_grads_vs_beta_eps_model"] = {
        n: {k: err(grads[n][k], grads["truth_fp64_beta_eps"][k]) for k in INPUTS}
        for n in ("chunk", "chunk_epsfree")}

    # 4. exact FP64 propagation to the parameters
    names = [n for n in P if any(n.startswith(s) for s in (
        "q_", "k_", "v_", "a_", "b_", "m_", "e_")) or n in (
        "A_log", "dt_bias", "mu_bias", "Mu_log", "log_factor", "D_log")]
    outs_f = [f64[k] for k in INPUTS]

    def vjp(gin):
        gs = torch.autograd.grad(outs_f, [P[n] for n in names],
                                 [gin[k].double() for k in INPUTS],
                                 retain_graph=True, allow_unused=True)
        return {n: (x if x is not None else torch.zeros_like(P[n]))
                for n, x in zip(names, gs)}
    p_truth = vjp(grads["truth_fp64"])
    p_vjp = {v: vjp(grads[v]) for v in ("chunk", "chunk_epsfree", "reference_fp32",
                                         "truth_fp64_beta_eps")}
    res["param_grads"] = {
        v: {n: err(x, p_truth[n])["norm_rel"] for n, x in p_vjp[v].items()}
        for v in p_vjp}
    res["param_grads_vs_beta_eps_model"] = {
        v: {n: err(x, p_vjp["truth_fp64_beta_eps"][n])["norm_rel"]
            for n, x in p_vjp[v].items()}
        for v in ("chunk", "chunk_epsfree")}
    res["attribution_chunk"] = {}
    for k in INPUTS:
        mixed = dict(grads["truth_fp64"], **{k: grads["chunk"][k]})
        pg = vjp(mixed)
        res["attribution_chunk"][k] = {n: err(pg[n], p_truth[n])["norm_rel"]
                                       for n in GATE_PARAMS + ("a_proj.weight",)}

    # 5. cancellation for the per-head gate parameters (forward-mode Jacobians)
    Pd = {n: t.detach() for n, t in P.items()}
    res["cancellation"] = {}
    for pname in GATE_PARAMS:
        def gates(x, pname=pname):
            f = frontend_ref(m, h64, dict(Pd, **{pname: x}))
            return f["log_alpha"], f["beta"]
        _, (dla, dbe) = torch.func.jvp(gates, (Pd[pname],),
                                       (torch.ones_like(Pd[pname]),))

        def contrib(gs):           # [B, T, H] per-token contributions
            return gs["log_alpha"].double() * dla + gs["beta"].double() * dbe
        c = contrib(grads["truth_fp64"])
        c_eps = contrib(grads["truth_fp64_beta_eps"])
        rows = []
        for hh in range(N_HEADS):
            ch = c[..., hh].flatten()
            row = dict(head=hh, n_tokens=ch.numel(), sum=ch.sum().item(),
                       kappa=(ch.abs().sum() / ch.sum().abs()).item(),
                       amplification=(ch.norm() / ch.sum().abs()).item())
            # errors vs the official equation, and the chunk kernel's own error
            # vs the function it implements (the beta + 1e-6 model)
            for v, e_full in (
                    ("chunk", contrib(grads["chunk"]) - c),
                    ("chunk_epsfree", contrib(grads["chunk_epsfree"]) - c),
                    ("reference_fp32", contrib(grads["reference_fp32"]) - c),
                    ("truth_fp64_beta_eps", c_eps - c),
                    ("chunk_vs_beta_eps_model", contrib(grads["chunk"]) - c_eps)):
                e = e_full[..., hh].flatten()
                row[v] = dict(
                    rel_err=(e.sum().abs() / ch.sum().abs()).item(),
                    per_token_rel=(e.norm() / ch.norm()).item(),
                    sum_over_random=(e.sum().abs() / e.norm()).item(),
                    sum_over_coherent=(e.sum().abs() / e.abs().sum()).item(),
                    corr_with_c=torch.corrcoef(torch.stack([e, ch]))[0, 1].item())
            rows.append(row)
        res["cancellation"][pname] = rows

    # 6. x = 1 - exp(log_c_{j-1} - log_ct_i) on each chunk's valid tril
    la, lm, be = (f_in[k].double() for k in ("log_alpha", "log_mu", "beta"))
    xs = []
    for c0 in range(0, T, BT):
        sl = slice(c0, min(c0 + BT, T))
        logc = torch.log(be[:, sl] + 1e-6) + torch.cumsum(lm[:, sl] - la[:, sl], 1)
        logct = torch.logcumsumexp(logc, 1)                         # [B, L, H]
        prev = torch.cat([torch.full_like(logct[:, :1], -float("inf")),
                          logct[:, :-1]], 1)
        x = 1 - torch.exp(prev[:, None, :, :] - logct[:, :, None, :])   # [B,i,j,H]
        L = logct.shape[1]
        tril = torch.ones(L, L, dtype=torch.bool, device=DEV).tril()
        xs.append(x[:, tril].flatten())
    x = torch.cat(xs)
    res["x_distribution"] = dict(
        n=x.numel(), min=x.min().item(),
        quantiles={q: x.quantile(q).item() for q in (0.001, 0.01, 0.1, 0.5)},
        frac_below={t: (x < t).double().mean().item()
                    for t in (1e-3, 1e-4, 1e-5, 1e-6)})

    bq = f_in["beta"].double().flatten()
    res["beta_distribution"] = dict(
        min=bq.min().item(),
        quantiles={q: bq.quantile(q).item() for q in (0.001, 0.01, 0.1, 0.5)},
        frac_below={t: (bq < t).double().mean().item()
                    for t in (1e-2, 1e-3, 1e-4, 1e-5)})

    with open(os.path.join(out_dir, f"diag_chunk_gates_T{T}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def report(res):
    T = res["T"]
    print(f"\n================ T = {T}")
    print("anchor (the failing whole-layer comparison, this process): "
          + "  ".join(f"{n}={res['anchor_layer'][n]:.2e}" for n in GATE_PARAMS))
    print("forward output vs FP64: " + "  ".join(
        f"{n}={e['norm_rel']:.2e}" for n, e in res["forward_output"].items())
          + f"   chunk forward identical across variants: "
          f"{res['chunk_forward_identical_across_variants']}")
    print("forward output vs the FP64 beta+1e-6 model: " + "  ".join(
        f"{n}={e['norm_rel']:.2e}" for n, e in res["forward_output_vs_beta_eps_model"].items()))
    print("\nrecurrence input gradients vs FP64 (same inputs, actual upstream), norm_rel")
    print(f"  {'input':>10} " + " ".join(f"{v:>15}" for v in res["recurrence_grads"]))
    for k in INPUTS:
        print(f"  {k:>10} " + " ".join(f"{res['recurrence_grads'][v][k]['norm_rel']:15.2e}"
                                       for v in res["recurrence_grads"]))
    print("  vs the FP64 beta+1e-6 model:")
    for k in INPUTS:
        print(f"  {k:>10} " + " ".join(
            f"{res['recurrence_grads_vs_beta_eps_model'][v][k]['norm_rel']:15.2e}"
            for v in res["recurrence_grads_vs_beta_eps_model"]))
    print("\nparameter gradients, recurrence grads propagated by the exact FP64 "
          "frontend VJP, norm_rel")
    show = GATE_PARAMS + ("mu_bias", "Mu_log", "a_proj.weight", "b_proj.weight",
                          "m_proj.weight", "q_proj.weight")
    print(f"  {'param':>14} " + " ".join(f"{v:>15}" for v in res["param_grads"]))
    for n in show:
        print(f"  {n:>14} " + " ".join(f"{res['param_grads'][v][n]:15.2e}"
                                       for v in res["param_grads"]))
    print("  vs the FP64 beta+1e-6 model (chunk, chunk_epsfree):")
    for n in show:
        print(f"  {n:>14} " + " ".join(f"{res['param_grads_vs_beta_eps_model'][v][n]:15.2e}"
                                       for v in res["param_grads_vs_beta_eps_model"]))
    print("\nattribution: ONLY this input's gradient from the chunk kernel, norm_rel")
    print(f"  {'input':>10} " + " ".join(f"{n:>14}" for n in GATE_PARAMS + ("a_proj.weight",)))
    for k, row in res["attribution_chunk"].items():
        print(f"  {k:>10} " + " ".join(f"{row[n]:14.2e}" for n in row))
    print("\ncancellation per head: kappa = sum|c|/|sum c|; for each implementation "
          "rel_err of the head's gradient, per-token rel error, and |sum e| / ||e||_2 "
          "(~1: random-walk), |sum e| / sum|e| (~1: coherent)")
    for p, rows in res["cancellation"].items():
        for r in rows:
            parts = []
            for v in ("chunk", "reference_fp32", "truth_fp64_beta_eps",
                      "chunk_vs_beta_eps_model"):
                d = r[v]
                parts.append(f"{v}: {d['rel_err']:.1e} (tok {d['per_token_rel']:.1e}, "
                             f"rw {d['sum_over_random']:.1f}, coh {d['sum_over_coherent']:.2f},"
                             f" corr {d['corr_with_c']:+.2f})")
            print(f"  {p:>10} h{r['head']} kappa {r['kappa']:7.1f} A {r['amplification']:6.1f} | "
                  + " | ".join(parts))
    xd = res["x_distribution"]
    print(f"\nx = 1 - exp(log_c_(j-1) - log_ct_i) on the valid tril (n={xd['n']}): "
          f"min {xd['min']:.2e}, quantiles " + ", ".join(
              f"{q}:{v:.2e}" for q, v in xd["quantiles"].items())
          + "; fraction below " + ", ".join(f"{t}:{v:.3f}" for t, v in xd["frac_below"].items()))

    bd = res["beta_distribution"]
    print(f"beta: min {bd['min']:.2e}, quantiles " + ", ".join(
        f"{q}:{v:.2e}" for q, v in bd["quantiles"].items()) + "; fraction below "
          + ", ".join(f"{t}:{v:.3f}" for t, v in bd["frac_below"].items()))

    # verdict, mechanical
    pg = res["param_grads"]
    worst = {v: max(pg[v][n] for n in GATE_PARAMS) for v in pg}
    worst_eps = {v: max(res["param_grads_vs_beta_eps_model"][v][n] for n in GATE_PARAMS)
                 for v in res["param_grads_vs_beta_eps_model"]}
    print("\nverdict:")
    print(f"  gate-parameter worst norm_rel vs the official equation: chunk "
          f"{worst['chunk']:.1e}, FP64 beta+1e-6 model {worst['truth_fp64_beta_eps']:.1e};"
          f" chunk vs the beta+1e-6 model {worst_eps['chunk']:.1e}")
    if worst["chunk"] > TOL and worst_eps["chunk"] <= TOL:
        print("  -> MODEL difference: the kernel computes the recurrence at beta + 1e-6, "
              "and its gradients match that function's exact FP64 gradients within "
              f"{TOL}. Not a backward error; the per-token deviation is amplified by "
              "the cancellation (kappa) in the gate-parameter sums.")
    elif worst["chunk"] > TOL and worst["chunk_epsfree"] <= TOL:
        print(f"  the gate-parameter discrepancy ({worst['chunk']:.1e}) disappears "
              f"({worst['chunk_epsfree']:.1e} <= {TOL}) when the backward-only eps in "
              "d_x is removed: a systematic BACKWARD approximation in the kernel, "
              "not cancellation.")
    elif worst["chunk"] > TOL:
        # the kernel's own error (vs the function it implements), per head:
        # random rounding sums like a random walk (|sum e| / ||e||_2 ~ 1) and
        # the parameter error is then per-token error x amplification A;
        # a systematic error adds coherently (|sum e| / ||e||_2 -> sqrt(N))
        rows = [r for rows in res["cancellation"].values() for r in rows]
        rw = sorted(r["chunk_vs_beta_eps_model"]["sum_over_random"] for r in rows)
        med_rw = rw[len(rw) // 2]
        sqrt_n = rows[0]["n_tokens"] ** 0.5
        print(f"  eps-free backward: {worst['chunk_epsfree']:.1e} (> {TOL}: the "
              "backward-only eps is not the cause).")
        print(f"  chunk error vs its own function: median |sum e|/||e||_2 over "
              f"heads = {med_rw:.1f} (random walk ~1, fully coherent ~{sqrt_n:.0f}); "
              "per head: parameter error = per-token error x A x that ratio (table).")
        if med_rw <= 3:
            print("  -> ROUNDING amplified by CANCELLATION: the kernel's per-token gate "
                  "gradient errors are incoherent; the gate-parameter sums cancel "
                  "(kappa, A above) and amplify them. Not a formula error.")
        else:
            print("  -> COHERENT per-token error in the kernel's gate gradients: "
                  "consistent with a systematic BACKWARD error, not with rounding. "
                  "Investigate the gate path of prepare_uyz_repr_bwd_kernel / "
                  "chunk_mode_rule_bwd_dqkyz.")
    else:
        print(f"  not reproduced here: chunk worst {worst['chunk']:.1e} <= {TOL}")
    print(f"  reference_fp32 worst {worst['reference_fp32']:.1e} (same cancellation, "
          "exact algorithm: the floor that kappa alone produces in FP32)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, nargs="+", default=[52, 128])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_fp32()
    mode, probe = dot_mode()
    meta = dict(gpu=torch.cuda.get_device_name(DEV), torch=torch.__version__,
                triton=triton.__version__,
                TRITON_F32_DEFAULT=os.environ.get("TRITON_F32_DEFAULT", "unset"),
                FLA_TRIL_PRECISION=os.environ.get("FLA_TRIL_PRECISION", "ieee"),
                TRITON_CACHE_DIR=os.environ.get("TRITON_CACHE_DIR", "unset"),
                probe_default_dot=mode, probe_norm_rel=probe)
    print(json.dumps(meta, indent=1))
    with open(os.path.join(args.out, "diag_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    epsfree = load_epsfree_kernel(args.out)
    m = layer()
    for T in args.T:
        t0 = time.time()
        report(run_T(m, T, epsfree, args.out))
        print(f"(T={T}: {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
