"""Integration checks for the restored official mixer. Needs CUDA + Triton + fla.

`verify_gpu.sh` runs this file twice, once with the Triton default and once
with TRITON_F32_DEFAULT=ieee, and keeps both JSON error reports.

Triton precision, as inspected in the pinned fla tree (not assumed):

* `fla/ops/momentum_delta_rule/{chunk_o,wy_fast,chunk_delta_h}.py`: 91
  `tl.dot` calls, none with `allow_tf32` / `input_precision`. They take
  Triton's default, which is TF32 on sm_80+ unless TRITON_F32_DEFAULT says
  otherwise. Whether this Triton build honours that variable is MEASURED by
  `test_triton_default_dot_precision` (a probe kernel vs FP64), not assumed.
* `fla/ops/utils/solve_tril.py`: every dot passes
  `input_precision=FLA_TRIL_PRECISION` explicitly (default 'ieee').
  TRITON_F32_DEFAULT cannot change it; FLA_TRIL_PRECISION can.
* `fla/ops/utils/cumsum.py`, `logcumsumexp.py`: `allow_tf32=False`.
* `fused_recurrent.py`, `l2norm.py`, `convolution.py`, `fused_norm_gate.py`:
  no dots in the paths used here.

So a run is "IEEE FP32 throughout" only if the probe reports IEEE AND
FLA_TRIL_PRECISION is unset or 'ieee'. That is recorded as `all_ieee`.

Error reporting: every comparison is against an FP64 reference (`layer_ref`
for whole layers, `reference.py` for the recurrence) and records the maximum
absolute error, the reference norm, and the norm-relative error
||x - x64|| / ||x64|| for outputs, input gradients and every parameter
gradient. Results go to $VERIFY_OUT/integration_<mode>.json.

Finiteness: every compared tensor (output, input gradient, parameter
gradient) and its FP64 reference must be finite everywhere, in BOTH
precision modes and whether or not the comparison is graded. A NaN or Inf
fails the test; non-finite counts are written to the JSON report.

Pass criterion, declared before any GPU run: norm-relative error <= 1e-4 per
tensor. It is applied to the FP32 reference backend always, and to the
official chunk backend only when `all_ieee`. With TF32 dots there is no
derived bound for accumulated gradients, so chunk errors are RECORDED and not
graded; nothing is loosened to obtain a pass.
"""

import json
import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
triton = pytest.importorskip("triton")
pytest.importorskip("fla")
import triton.language as tl  # noqa: E402

from fla.layers.momentum_deltanet import MomentumDeltaNet  # noqa: E402

from prospective.layer_ref import mixer_forward_ref, params_as  # noqa: E402
from prospective.mixer import ALL_ARMS, ProspectiveMDN, unused_by_design  # noqa: E402
from prospective.model import (ARM_LEAVES, build, shared_parameter_check,  # noqa: E402
                               unused_parameters)
from prospective.reference import reference_recurrence  # noqa: E402
from prospective.task import PRODUCTION_TASK, make_batch  # noqa: E402
from prospective.train import set_fp32  # noqa: E402

DEV = torch.device("cuda")
REQUESTED = os.environ.get("TRITON_F32_DEFAULT", "").lower() or "default"
TRIL = os.environ.get("FLA_TRIL_PRECISION", "ieee")
TOL = 1e-4          # declared before any GPU run; see module docstring
D_MODEL, N_HEADS, HEAD_DIM, EXPAND_V = 128, 4, 32, 2
set_fp32()

REPORT = {"requested_TRITON_F32_DEFAULT": REQUESTED,
          "FLA_TRIL_PRECISION": TRIL, "tolerance_norm_rel": TOL,
          "gpu": torch.cuda.get_device_name(DEV), "torch": torch.__version__,
          "triton": triton.__version__, "comparisons": {}}


def _write_report():
    out = os.environ.get("VERIFY_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"integration_{REQUESTED}.json"), "w") as fh:
            json.dump(REPORT, fh, indent=1)


@pytest.fixture(scope="session", autouse=True)
def _report_at_end():
    yield
    _write_report()


def err(x, ref):
    x, ref = x.detach().double(), ref.detach().double()
    d = x - ref
    nr = ref.norm().item()
    return dict(max_abs=d.abs().max().item(), ref_norm=nr,
                norm_rel=(d.norm().item() / nr) if nr > 0 else
                (0.0 if d.norm().item() == 0 else float("inf")),
                nonfinite=int((~torch.isfinite(x)).sum().item()),
                nonfinite_ref=int((~torch.isfinite(ref)).sum().item()),
                numel=x.numel())


def record(name, rows, graded):
    """Print and store a comparison. EVERY comparison, graded or not,
    requires the measured tensor and its FP64 reference to be finite
    everywhere; only the relative-error tolerance depends on `graded`."""
    REPORT["comparisons"][name] = dict(graded=graded, rows=rows)
    print(f"\n  {name}  ({'graded' if graded else 'recorded only'};"
          " finiteness always required)")
    print(f"    {'tensor':>28} {'max_abs':>10} {'||ref||':>10} {'norm_rel':>10}"
          f" {'nonfinite':>12}")
    for n, e in rows.items():
        print(f"    {n:>28} {e['max_abs']:10.2e} {e['ref_norm']:10.2e} "
              f"{e['norm_rel']:10.2e} {e['nonfinite']:5d}/{e['numel']:<6d}")
    nonfinite = {n: (e["nonfinite"], e["nonfinite_ref"]) for n, e in rows.items()
                 if e["nonfinite"] or e["nonfinite_ref"]}
    assert not nonfinite, f"non-finite (measured, reference) counts: {nonfinite}"
    if graded:
        bad = {n: e["norm_rel"] for n, e in rows.items()
               if not e["norm_rel"] <= TOL}
        assert not bad, bad


# ------------------------------------------------------- precision, measured
@triton.jit
def _dot_probe(a_ptr, b_ptr, c_ptr, N: tl.constexpr):
    o = tl.arange(0, N)
    a = tl.load(a_ptr + o[:, None] * N + o[None, :])
    b = tl.load(b_ptr + o[:, None] * N + o[None, :])
    tl.store(c_ptr + o[:, None] * N + o[None, :], tl.dot(a, b))


def default_dot_precision():
    g = torch.Generator(device=DEV).manual_seed(0)
    N = 32
    a = torch.randn(N, N, device=DEV, generator=g)
    b = torch.randn(N, N, device=DEV, generator=g)
    c = torch.empty_like(a)
    _dot_probe[(1,)](a, b, c, N=N)
    e = err(c, a.double() @ b.double())["norm_rel"]
    # TF32 operands (10-bit mantissa) give ~1e-3 here; IEEE FP32 ~1e-7
    return ("ieee" if e < 1e-5 else "tf32"), e


MODE, PROBE_ERR = default_dot_precision()
ALL_IEEE = MODE == "ieee" and TRIL == "ieee"
REPORT.update(probe_default_dot=MODE, probe_norm_rel=PROBE_ERR,
              all_ieee=ALL_IEEE)


def test_triton_default_dot_precision():
    print(f"\n  default tl.dot: {MODE} (probe norm_rel {PROBE_ERR:.2e}); "
          f"FLA_TRIL_PRECISION={TRIL}; all_ieee={ALL_IEEE}")
    if REQUESTED == "ieee":
        assert MODE == "ieee", ("TRITON_F32_DEFAULT=ieee was set but this "
                                "Triton build still uses TF32 for default dots")


# ------------------------------------------------------------ helpers
def layer(arm="native", backend="reference", seed=0):
    torch.manual_seed(seed)
    m = ProspectiveMDN(arm, hidden_size=D_MODEL, num_heads=N_HEADS,
                       head_dim=HEAD_DIM, expand_v=EXPAND_V, backend=backend)
    m.official_model_init_()
    return m.to(DEV)


def hidden(B, T, seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(B, T, D_MODEL, device=DEV, generator=g)


def layer_errors(m, T, seed):
    """Output, dL/dh and every parameter grad of the FP32 layer `m` vs the
    FP64 functional reference built from the same parameters."""
    h = hidden(4, T, seed)
    g = torch.randn(4, T, D_MODEL, device=DEV,
                    generator=torch.Generator(device=DEV).manual_seed(seed + 1))
    h32 = h.clone().requires_grad_(True)
    m.zero_grad(set_to_none=True)
    o32 = m(h32)[0]
    o32.backward(g)
    P = params_as(m, torch.float64)
    h64 = h.double().requires_grad_(True)
    o64 = mixer_forward_ref(m, h64, P)
    o64.backward(g.double())
    rows = {"output": err(o32, o64), "d/d input": err(h32.grad, h64.grad)}
    for n, p in m.named_parameters():
        if p.grad is None and P[n].grad is None:
            continue
        rows["d/d " + n] = err(p.grad if p.grad is not None
                               else torch.zeros_like(p), P[n].grad)
    return rows


# ------------------------------------------------------------ the mixer shape
def test_value_dimension_is_expanded():
    m = layer()
    assert (m.head_k_dim, m.head_v_dim) == (32, 64)
    assert (m.key_dim, m.value_dim) == (128, 256)
    assert m.v_conv1d.weight.shape[0] == 256
    o, _, _ = m(hidden(2, 52))
    assert o.shape == (2, 52, D_MODEL) and o.dtype == torch.float32


def test_everything_is_fp32():
    m = build("native", vocab=58, seed=0).to(DEV)
    assert all(p.dtype == torch.float32 for p in m.parameters())
    assert not torch.is_autocast_enabled()
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


def test_whole_sequence_only():
    m = layer()
    h = hidden(1, 8)
    with pytest.raises(NotImplementedError):
        m(h, attention_mask=torch.ones(1, 8, device=DEV, dtype=torch.long))
    with pytest.raises(NotImplementedError):
        m(h, use_cache=True)


# --------------------------------------------- the copy of the official layer
def test_frontend_copy_is_bitwise_official():
    """Our forward with the official_chunk backend vs the UNMODIFIED official
    forward at T = 128 (where the official layer itself uses chunk)."""
    m = layer("native", "official_chunk").train()
    h1 = hidden(2, 128).requires_grad_(True)
    h2 = h1.detach().clone().requires_grad_(True)
    o1 = m(h1)[0]
    g1 = torch.autograd.grad(o1.sum(), [h1] + list(m.parameters()))
    o2 = MomentumDeltaNet.forward(m, h2)[0]
    g2 = torch.autograd.grad(o2.sum(), [h2] + list(m.parameters()))
    # the official path run again: its own run-to-run floor (0 if the
    # kernels are deterministic). Ours may not differ by more than that.
    h3 = h1.detach().clone().requires_grad_(True)
    o3 = MomentumDeltaNet.forward(m, h3)[0]
    g3 = torch.autograd.grad(o3.sum(), [h3] + list(m.parameters()))
    names = ["input"] + [n for n, _ in m.named_parameters()]
    # finiteness first: NaN would make every comparison below vacuous
    nonfinite = {f"{tag}:{n}": int((~torch.isfinite(t)).sum())
                 for tag, gs in (("ours", g1), ("official", g2), ("official_rerun", g3))
                 for n, t in zip(names, gs) if not torch.isfinite(t).all()}
    nonfinite.update({f"output{i}": int((~torch.isfinite(o)).sum())
                      for i, o in enumerate((o1, o2, o3)) if not torch.isfinite(o).all()})
    assert not nonfinite, nonfinite
    assert torch.equal(o1, o2)
    diff = {n: (a - b).abs().max().item() for n, a, b in zip(names, g1, g2)}
    floor = {n: (a - b).abs().max().item() for n, a, b in zip(names, g2, g3)}
    REPORT["comparisons"]["frontend_copy_grad_max_abs"] = dict(
        ours_vs_official=diff, official_run_to_run=floor)
    bad = {n: (diff[n], floor[n]) for n in names if diff[n] > floor[n]}
    assert not bad, bad


def test_official_layer_cannot_train_at_length_52():
    m = layer("native", "official_chunk").train()
    with pytest.raises(AssertionError):
        MomentumDeltaNet.forward(m, hidden(2, 52))


# ------------------------------------------------- whole layer vs FP64
@pytest.mark.parametrize("T", [52, 128])
def test_reference_layer_vs_fp64(T):
    record(f"layer reference FP32 vs FP64, T={T}",
           layer_errors(layer("native", "reference"), T, 20 + T), graded=True)


@pytest.mark.parametrize("T", [52, 128])
def test_official_chunk_layer_vs_fp64(T):
    record(f"layer official_chunk FP32 vs FP64, T={T}",
           layer_errors(layer("native", "official_chunk"), T, 20 + T),
           graded=ALL_IEEE)


def test_reference_layer_vs_fp64_every_arm():
    """Every arm: v2 rules, v3 writers, v4 pilot projection writers (the
    learned-inertia arm at its declared initial A, r)."""
    for arm in ALL_ARMS:
        m = layer(arm, "reference")
        with torch.no_grad():
            if arm == "generalized":
                m.fil_M.fill_(0.3); m.fil_gamma.fill_(0.5); m.fil_T.fill_(1.2)
            if arm == "qhm":
                m.nu.fill_(0.6)
        record(f"layer reference FP32 vs FP64, arm={arm}, T=52",
               layer_errors(m, 52, 77), graded=True)


def test_official_fused_recurrent_forward_at_52():
    """At T = 52 in eval the official forward runs fused_recurrent. Its
    in-kernel l2norm is x / (||x|| + 1e-6), not chunk's x / sqrt(sum + 1e-6),
    so this is recorded, not graded."""
    m = layer("native", "reference").eval()
    h = hidden(4, 52, seed=3)
    with torch.no_grad():
        theirs = MomentumDeltaNet.forward(m, h)[0]
    P = params_as(m, torch.float64)
    with torch.no_grad():
        o64 = mixer_forward_ref(m, h.double(), P)
    record("official fused_recurrent forward vs FP64, T=52",
           {"output": err(theirs, o64)}, graded=False)


# ---------------------------------------------- recurrence: kernel level
@pytest.mark.parametrize("T", [52, 128])
def test_recurrence_kernel_vs_fp64(T):
    m = layer("native", "official_chunk")
    with torch.no_grad():
        f = m.frontend(hidden(4, T, seed=T))
    f32 = {k: v.detach().clone().requires_grad_(True) for k, v in f.items()}
    fc = {k: v.detach().clone().requires_grad_(True) for k, v in f.items()}
    f64 = {k: v.detach().double().requires_grad_(True) for k, v in f.items()}
    o_r = reference_recurrence("native", f32, {})
    o_c = m.backend_official_chunk(fc)
    o_64 = reference_recurrence("native", f64, {})
    g = torch.randn_like(o_c)
    gr = torch.autograd.grad(o_r, list(f32.values()), g)
    gc = torch.autograd.grad(o_c, list(fc.values()), g, allow_unused=True)
    g64 = torch.autograd.grad(o_64, list(f64.values()), g.double())
    rows_r = {"output": err(o_r, o_64)}
    rows_c = {"output": err(o_c, o_64)}
    for n, a, c, t in zip(f, gr, gc, g64):
        assert c is not None, f"chunk backward returned no grad for {n}"
        rows_r["d/d " + n] = err(a, t)
        rows_c["d/d " + n] = err(c, t)
    record(f"recurrence reference FP32 vs FP64, T={T}", rows_r, graded=True)
    record(f"recurrence chunk FP32 vs FP64, T={T}", rows_c, graded=ALL_IEEE)


@pytest.mark.parametrize("T", [52, 116])
def test_chunk_gate_gradients_finite_with_partial_last_chunk(T):
    """Regression for the logcumsumexp backward in prepare_uyz_repr_bwd_kernel
    (wy_fast.py): padded rows of a partial last chunk were not masked, and
    exp(log_c_before) overflowed there whenever the in-chunk
    log_c = log(beta + 1e-6) + cumsum(log_mu - log_alpha) exceeded 88.72,
    giving NaN d/d log_alpha, log_mu, beta. The precondition (overflow
    regime reached) is asserted, so the test cannot pass vacuously."""
    m = layer("native", "official_chunk")
    with torch.no_grad():
        f = m.frontend(hidden(4, T, seed=300 + T))
    la, lm, be = f["log_alpha"], f["log_mu"], f["beta"]
    last = slice((T // 64) * 64, T)
    logc = torch.log(be[:, last] + 1e-6) + torch.cumsum(lm[:, last] - la[:, last], 1)
    REPORT["comparisons"][f"partial chunk T={T}: max in-chunk log_c"] = logc.max().item()
    assert T % 64 != 0 and logc.max().item() > 88.72, "overflow regime not reached"
    fc = {k: v.detach().clone().requires_grad_(True) for k, v in f.items()}
    f64 = {k: v.detach().double().requires_grad_(True) for k, v in f.items()}
    o_c = m.backend_official_chunk(fc)
    o_64 = reference_recurrence("native", f64, {})
    g = torch.randn_like(o_c)
    gc = torch.autograd.grad(o_c, list(fc.values()), g, allow_unused=True)
    g64 = torch.autograd.grad(o_64, list(f64.values()), g.double())
    rows = {"output": err(o_c, o_64)}
    for n, c, t in zip(f, gc, g64):
        assert c is not None, n
        rows["d/d " + n] = err(c, t)
    record(f"recurrence chunk FP32 vs FP64, partial last chunk, T={T}", rows,
           graded=ALL_IEEE)


# ------------------------------------------ arm-boundary identities, in layer
def test_layer_qhm_at_nu_one_is_native_bitwise():
    n, q = layer("native"), layer("qhm")
    assert float(q.nu) == 1.0
    h = hidden(2, 52, seed=5)
    with torch.no_grad():
        assert torch.equal(n(h)[0], q(h)[0])


def test_layer_tss_is_generalized_at_zero_mass_bitwise():
    t, g = layer("tss"), layer("generalized")
    with torch.no_grad():
        g.fil_M.zero_(); g.fil_gamma.zero_(); g.fil_T.copy_(t.fil_T)
        h = hidden(2, 52, seed=6)
        assert torch.equal(t(h)[0], g(h)[0])


def test_layer_generalized_at_native_boundary():
    n, g = layer("native"), layer("generalized")
    with torch.no_grad():
        g.fil_M.zero_(); g.fil_gamma.fill_(1.0); g.fil_T.zero_()
        h = hidden(2, 52, seed=7)
        e = err(g(h)[0], n(h)[0])
    REPORT["comparisons"]["generalized boundary vs native"] = e
    assert e["norm_rel"] <= 1e-6


def test_layer_tss_start_is_not_native():
    n, t = layer("native"), layer("tss")
    h = hidden(2, 52, seed=8)
    with torch.no_grad():
        assert err(t(h)[0], n(h)[0])["norm_rel"] > 1e-3


def test_layer_gated_is_native_with_mu_zero():
    m = layer("native")
    with torch.no_grad():
        f = m.frontend(hidden(2, 52, seed=4))
    f_off = dict(f, log_mu=torch.full_like(f["log_mu"], float("-inf")))
    assert torch.equal(reference_recurrence("gated", f, {}),
                       reference_recurrence("native", f_off, {}))


# ------------------------------------------------------------ model-level
def _nontrivial_leaves(m):
    with torch.no_grad():
        for blk in m.blocks:
            x = blk.mix
            if getattr(x, "arm", None) == "generalized":
                x.fil_M.fill_(0.3); x.fil_gamma.fill_(0.5); x.fil_T.fill_(1.2)
            elif hasattr(x, "fil_T"):
                x.fil_T.fill_(1.5)
            if hasattr(x, "nu"):
                x.nu.fill_(0.6)
            if hasattr(x, "inertia_A"):
                x.inertia_A.fill_(0.3); x.inertia_r.fill_(0.4)


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_future_tokens_cannot_change_earlier_outputs(arm):
    m = build(arm, vocab=58, seed=0).to(DEV).eval()
    _nontrivial_leaves(m)
    gen = torch.Generator().manual_seed(9)
    x = make_batch(4, **PRODUCTION_TASK, generator=gen)["x"].to(DEV)
    with torch.no_grad():
        o = m(x)
        for t in (0, 9, 27, 50):
            x2 = x.clone()
            x2[:, t + 1:] = torch.randint(0, 58, x2[:, t + 1:].shape,
                                          generator=gen).to(DEV)
            o2 = m(x2)
            assert torch.equal(o[:, :t + 1], o2[:, :t + 1]), (arm, t)
            assert not torch.equal(o[:, t + 1:], o2[:, t + 1:]), (arm, t)


def test_shared_tensors_initialise_identically_across_arms():
    models = {a: build(a, vocab=58, seed=501) for a in ALL_ARMS}
    ref = models["native"]
    for a, m in models.items():
        assert shared_parameter_check(ref, m) == [], a
        assert set(ref.state_dict()) <= set(m.state_dict()), a
        extra = set(m.state_dict()) - set(ref.state_dict())
        assert all(n.split(".")[-1] in ARM_LEAVES for n in extra), (a, extra)
    mix = ref.blocks[0].mix
    assert abs(mix.q_proj.weight.std().item() - 0.02) < 2e-3
    assert abs(mix.q_conv1d.weight.std().item() - 0.02) < 5e-3


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_unused_parameters_on_gpu(arm):
    m = build(arm, vocab=58, seed=0).to(DEV)
    b = make_batch(4, **PRODUCTION_TASK, generator=torch.Generator().manual_seed(0))
    torch.nn.functional.cross_entropy(m(b["x"].to(DEV)).reshape(-1, 58),
                                      b["y"].to(DEV).reshape(-1),
                                      ignore_index=-100).backward()
    got = set(unused_parameters(m))
    want = {f"blocks.{i}.mix.{n}" for i in range(2) for n in unused_by_design(arm)}
    REPORT.setdefault("unused_parameters", {})[arm] = sorted(got)
    assert got == want, got ^ want


def test_leaves_actually_move_under_optimization():
    m = build("generalized", vocab=58, seed=0).to(DEV)
    mix = m.blocks[0].mix
    before = (mix.fil_M.detach().clone(), mix.fil_gamma.detach().clone())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    b = make_batch(8, **PRODUCTION_TASK, generator=torch.Generator().manual_seed(0))
    x, y = b["x"].to(DEV), b["y"].to(DEV)
    for _ in range(25):
        loss = torch.nn.functional.cross_entropy(
            m(x).reshape(-1, 58), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        m.project_()
    assert (mix.fil_M - before[0]).abs().max().item() > 1e-3
    assert (mix.fil_gamma - before[1]).abs().max().item() > 1e-3


def test_tss_leaves_stay_pinned_at_zero():
    m = build("tss", vocab=58, seed=0).to(DEV)
    mix = m.blocks[0].mix
    opt = torch.optim.AdamW(m.parameters(), lr=3e-2)
    x = torch.randint(0, 58, (4, 52), device=DEV)
    for _ in range(10):
        m(x).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        m.project_()
    assert torch.equal(mix.fil_M, torch.zeros_like(mix.fil_M))
    assert torch.equal(mix.fil_gamma, torch.zeros_like(mix.fil_gamma))
    assert (mix.fil_T > 0.5).all()


# ------------------------------------------------- v4 pilot: projection arms
PILOT_MOTION = "geometric_motion_learned"


def test_pilot_motion_arm_vs_fp64_at_interior_coefficients():
    m = layer(PILOT_MOTION, "reference")
    with torch.no_grad():
        m.inertia_A.copy_(torch.tensor([0.3, 0.05, 0.9, 0.003125]))
        m.inertia_r.copy_(torch.tensor([0.4, 0.0, 0.7, 0.5]))
    record("layer reference FP32 vs FP64, arm=geometric_motion_learned "
           "(interior A, r), T=52", layer_errors(m, 52, 78), graded=True)


@pytest.mark.parametrize("A,r", [(0.0, 0.5), (0.3, 1.0)])
def test_pilot_no_motion_boundaries_bitwise_on_gpu(A, r):
    g, mo = layer("geometric_projection"), layer(PILOT_MOTION)
    with torch.no_grad():
        mo.inertia_A.fill_(A); mo.inertia_r.fill_(r)
        h = hidden(2, 52, seed=12)
        assert torch.equal(g(h)[0], mo(h)[0])


def test_pilot_zero_keys_stay_finite_on_gpu():
    for arm in ("euclidean_projection", "geometric_projection", PILOT_MOTION):
        m = layer(arm)
        with torch.no_grad():
            m.k_proj.weight.zero_()
        h = hidden(2, 52, seed=13).requires_grad_(True)
        o = m(h)[0]
        o.sum().backward()
        assert torch.isfinite(o).all() and torch.isfinite(h.grad).all(), arm


def test_pilot_spec_trains_checkpoints_and_resumes_on_gpu(tmp_path):
    """A few updates of the exact pilot spec (tiny eval set), checkpoint,
    resume. Bitwise resume is proven on CPU (test_train_cpu.py); here the
    batch stream must be identical and the parameter differences between an
    uninterrupted and a resumed run are RECORDED (GPU kernels need not be
    run-to-run deterministic)."""
    from prospective import experiment_pilot as P
    from prospective.train import parser, run
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = P.flags() + ["--eval-batches", "2", "--eval-every", "2",
                        "--ckpt-every", "2", "--device", "cuda", "--seed", "1"]
    spec_i = base.index("--spec") + 1
    base[spec_i] = os.path.join(root, base[spec_i])

    def go(d, steps, *extra):
        return run(parser().parse_args(base + ["--steps", str(steps), "--out",
                                               str(d / "results.json"), *extra]))
    a = go(tmp_path / "a", 4)
    go(tmp_path / "b", 2)
    go(tmp_path / "b", 4, "--resume")
    assert a["arms"] == list(P.ARMS) and not a["failed"]
    assert [x["step"] for x in a["curves"]["native"]] == [2, 4]
    ca = torch.load(tmp_path / "a" / "ckpt.pt", weights_only=False)
    cb = torch.load(tmp_path / "b" / "ckpt.pt", weights_only=False)
    assert torch.equal(ca["gen"], cb["gen"])
    diffs = {arm: max((t - cb["models"][arm][n]).abs().max().item()
                      for n, t in sd.items()) for arm, sd in ca["models"].items()}
    REPORT["comparisons"]["pilot resume vs uninterrupted, max |param diff|"] = diffs
    print("\n  resume vs uninterrupted, max |param diff|:", diffs)
    for c in (ca, cb):
        for arm, sd in c["models"].items():
            assert all(torch.isfinite(t).all() for t in sd.values()), arm
    assert "inertia" in a and set(a["inertia"]) == {PILOT_MOTION}
