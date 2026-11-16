"""Every arm must reproduce the official native reference at its native point."""

import importlib.util
import os

import torch

# Load the official reference BY PATH: `fla/__init__` pulls in transformers,
# which this equivalence check does not need.
_NAIVE = os.path.join(os.path.dirname(__file__), os.pardir,
                      "flash-linear-attention", "fla", "ops",
                      "momentum_delta_rule", "naive.py")
_spec = importlib.util.spec_from_file_location("fla_mdr_naive", _NAIVE)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
recurrent_momentum_delta_rule_ref = _mod.recurrent_momentum_delta_rule_ref
from prospective.coefficients import H_TOKEN
from prospective.rules import recurrent_ref

B, T, H, DK, DV = 2, 24, 3, 8, 8


# The official reference casts every input to float32 unconditionally, so the
# cross-checks against it run in float32; only the gradient test uses float64.
def inputs(seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    k = torch.nn.functional.normalize(r(B, T, H, DK), dim=-1)
    log_alpha = -torch.rand(B, T, H, generator=g, dtype=dtype) * 0.1
    p = log_alpha.exp().unsqueeze(-1) * k
    return dict(q=r(B, T, H, DK), k=k, v=r(B, T, H, DV), p=p,
                log_alpha=log_alpha,
                log_mu=-torch.rand(B, T, H, generator=g, dtype=dtype) * 0.5,
                beta=torch.rand(B, T, H, generator=g, dtype=dtype),
                eta=torch.rand(B, T, H, generator=g, dtype=dtype))


def native(d):
    return recurrent_momentum_delta_rule_ref(
        d["q"], d["k"], d["v"], d["log_alpha"], d["log_mu"], d["p"],
        beta=d["beta"], eta=d["eta"])[0]


def _dev(a, b):
    return (a - b).abs().max().item()


def test_our_native_path_matches_theirs():
    d = inputs()
    assert _dev(recurrent_ref("native", **d)[0], native(d)) == 0.0


def test_qhm_at_nu_one_is_native():
    d = inputs(1)
    got = recurrent_ref("qhm", nu=torch.tensor(1.0, dtype=torch.float64), **d)[0]
    assert _dev(got, native(d)) == 0.0


def test_generalized_at_native_boundary_is_native():
    """M = 0, gamma = h, T = 0 gives (a,b,c,d) = (0,0,1,0), i.e. y_t = R_t."""
    d = inputs(2)
    z = torch.zeros(H, dtype=torch.float64)
    got = recurrent_ref("generalized", mass=z, gamma=z + H_TOKEN, response=z,
                        **d)[0]
    # not bitwise: the filter path multiplies by eta AFTER the outer product,
    # where the official line folds eta into k first
    assert _dev(got, native(d)) < 1e-6 * native(d).abs().max().item()


def test_tss_is_the_generalized_path_at_zero_mass():
    d = inputs(3, dtype=torch.float64)
    z = torch.zeros(H, dtype=torch.float64)
    T_ = torch.full((H,), 2.0, dtype=torch.float64)
    x = recurrent_ref("tss", mass=z, gamma=z, response=T_, **d)[0]
    y = recurrent_ref("generalized", mass=z, gamma=z, response=T_, **d)[0]
    assert _dev(x, y) == 0.0


def test_tss_actually_changes_the_output():
    d = inputs(4)
    z = torch.zeros(H, dtype=torch.float64)
    got = recurrent_ref("tss", mass=z, gamma=z,
                        response=torch.full((H,), 2.0, dtype=torch.float64),
                        **d)[0]
    assert _dev(got, native(d)) > 1e-3


def test_nesterov_is_native_when_momentum_is_off():
    """mu = 0 makes the lookahead L_t equal the decayed state, so the arm
    collapses onto native. This is the only exact point Nesterov has."""
    d = inputs(5)
    d["log_mu"] = torch.full_like(d["log_mu"], -60.0)   # mu ~ 0
    assert _dev(recurrent_ref("nesterov", **d)[0], native(d)) < 1e-12


def test_gradients_flow_to_every_new_leaf():
    d = inputs(6, dtype=torch.float64)
    z = torch.zeros(H, dtype=torch.float64, requires_grad=True)
    m = torch.full((H,), 0.7, dtype=torch.float64, requires_grad=True)
    g = torch.full((H,), 0.3, dtype=torch.float64, requires_grad=True)
    t = torch.full((H,), 1.5, dtype=torch.float64, requires_grad=True)
    recurrent_ref("generalized", mass=m, gamma=g, response=t, **d)[0].sum().backward()
    for leaf in (m, g, t):
        assert leaf.grad is not None and leaf.grad.abs().max() > 0


def test_gated_is_the_momentum_rule_with_the_carry_off():
    """Gated DeltaNet is arm `gated`: the same path with mu = 0. It must equal
    the official native reference evaluated at mu = 0, bitwise."""
    d = inputs(7)
    got = recurrent_ref("gated", **d)[0]
    d_off = dict(d, log_mu=torch.full_like(d["log_mu"], float("-inf")))
    assert _dev(got, native(d_off)) == 0.0


def test_gated_differs_from_native_when_momentum_is_on():
    d = inputs(8)
    assert _dev(recurrent_ref("gated", **d)[0], native(d)) > 1e-3


def test_every_arm_runs_and_all_six_are_present():
    from prospective.rules import ARMS
    assert len(ARMS) == 6
    d = inputs(9)
    z = torch.zeros(H, dtype=d["v"].dtype)
    extra = {"tss": dict(mass=z, gamma=z, response=z + 2.0),
             "generalized": dict(mass=z + 0.7, gamma=z + 0.3, response=z + 1.5),
             "qhm": dict(nu=torch.tensor(0.6, dtype=d["v"].dtype))}
    for arm in ARMS:
        o, _ = recurrent_ref(arm, **extra.get(arm, {}), **d)
        assert torch.isfinite(o).all(), arm


def test_projection_keeps_coefficients_jury_stable():
    from prospective.coefficients import (jury_ok, project_leaves,
                                          prospective_coefficients)
    g = torch.Generator().manual_seed(3)
    M = torch.rand(200, generator=g) * 2 - 1.0      # includes negatives
    gam = torch.rand(200, generator=g) * 2 - 1.0
    T = torch.rand(200, generator=g) * 2 - 1.0
    project_leaves(M, gam, T)
    a, b, _, _ = prospective_coefficients(M, gam, T)
    assert jury_ok(a, b).all()


# ---------------------------------------------------------------- v2 checks
def _nesterov_explicit(d):
    """Independent restatement: build L_t = alpha S - beta mu M as a MATRIX,
    read it with the undecayed key, w = Norm(k) L - v. Everything else is the
    native rule. float64, no shared code with rules.py."""
    q, k, v = d["q"] * d["q"].shape[-1] ** -0.5, d["k"], d["v"]
    B_, T_, H_, DK_ = k.shape
    S = torch.zeros(B_, H_, DK_, v.shape[-1], dtype=v.dtype)
    M = torch.zeros_like(S)
    out = torch.zeros_like(v)
    for i in range(T_):
        al = d["log_alpha"][:, i].exp()[..., None, None]
        mu = d["log_mu"][:, i].exp()[..., None, None]
        be = d["beta"][:, i][..., None, None]
        et = d["eta"][:, i][..., None, None]
        L = al * S - be * mu * M                       # the declared lookahead
        w = torch.einsum("bhk,bhkv->bhv", k[:, i], L) - v[:, i]
        M = mu * M + et * torch.einsum("bhk,bhv->bhkv", k[:, i], w)
        S = al * S - be * M
        out[:, i] = torch.einsum("bhk,bhkv->bhv", q[:, i], S)
    return out


def test_nesterov_residual_is_evaluation_at_the_lookahead_matrix():
    d = inputs(10, dtype=torch.float64)
    ref = _nesterov_explicit(d)
    got = recurrent_ref("nesterov", n=d["k"], **d)[0]
    assert _dev(got, ref) < 1e-12 * ref.abs().max().item()
    # the p / alpha fallback is the same quantity
    got2 = recurrent_ref("nesterov", **d)[0]
    assert _dev(got2, ref) < 1e-12 * ref.abs().max().item()
    # and it is NOT the native residual (momentum is on)
    assert _dev(recurrent_ref("native", **d)[0], ref) > 1e-6


def test_nesterov_explicit_key_survives_alpha_underflow():
    """p = alpha Norm(k) underflows to 0 when log_alpha < ~-104 in float32;
    p / alpha is then 0/0. The explicit key is unaffected."""
    d = inputs(11)
    d["log_alpha"] = d["log_alpha"].clone()
    d["log_alpha"][:, 5] = -120.0
    d["p"] = d["log_alpha"].exp().unsqueeze(-1) * d["k"]
    assert torch.isfinite(recurrent_ref("nesterov", n=d["k"], **d)[0]).all()
    assert not torch.isfinite(recurrent_ref("nesterov", **d)[0]).all()


def test_tss_starts_at_two_R_minus_previous_R_not_identity():
    """At its start point (M = gamma = 0, T = h) TSS is y_t = 2 R_t - R_(t-1).
    The native identity filter y_t = R_t is (a,b,c,d) = (0,0,1,0)."""
    from prospective.coefficients import TSS_T0, tss_coefficients
    a, b, c, d = tss_coefficients(torch.tensor([TSS_T0], dtype=torch.float64))
    assert (a.item(), b.item(), c.item(), d.item()) == (0.0, 0.0, 2.0, 1.0)


def test_filter_arms_refuse_state_carry():
    import pytest
    d = inputs(12)
    z = torch.zeros(H)
    S0 = torch.zeros(B, H, DK, DV)
    for arm in ("tss", "generalized"):
        kw = dict(mass=z, gamma=z, response=z + 1.0)
        with pytest.raises(NotImplementedError):
            recurrent_ref(arm, initial_S=S0, **kw, **d)
        with pytest.raises(NotImplementedError):
            recurrent_ref(arm, output_final_state=True, **kw, **d)


def test_unequal_key_and_value_dims():
    """v2 runs K = 32, V = 64. Native must still equal the official reference
    bitwise, and every arm must run at K != V."""
    from prospective.rules import ARMS
    g = torch.Generator().manual_seed(13)
    dk, dv = 8, 16
    k = torch.nn.functional.normalize(torch.randn(B, T, H, dk, generator=g), dim=-1)
    la = -torch.rand(B, T, H, generator=g) * 0.1
    d = dict(q=torch.randn(B, T, H, dk, generator=g), k=k,
             v=torch.randn(B, T, H, dv, generator=g),
             p=la.exp().unsqueeze(-1) * k, log_alpha=la,
             log_mu=-torch.rand(B, T, H, generator=g) * 0.5,
             beta=torch.rand(B, T, H, generator=g),
             eta=torch.rand(B, T, H, generator=g))
    assert _dev(recurrent_ref("native", **d)[0], native(d)) == 0.0
    z = torch.zeros(H)
    extra = {"tss": dict(mass=z, gamma=z, response=z + 2.0),
             "generalized": dict(mass=z + 0.7, gamma=z + 0.3, response=z + 1.5),
             "qhm": dict(nu=torch.tensor(0.6)), "nesterov": dict(n=k)}
    for arm in ARMS:
        o, _ = recurrent_ref(arm, **extra.get(arm, {}), **d)
        assert o.shape == (B, T, H, dv) and torch.isfinite(o).all(), arm


def test_reference_backend_preprocessing():
    """reference_recurrence = l2norm(q), l2norm(k), p = alpha l2norm(k), then
    the rule; native on it equals the official naive reference bitwise."""
    from prospective.reference import l2norm_ref, reference_recurrence
    g = torch.Generator().manual_seed(14)
    f = dict(q=torch.randn(B, T, H, DK, generator=g) * 3,
             k=torch.randn(B, T, H, DK, generator=g) * 3,
             v=torch.randn(B, T, H, DV, generator=g),
             log_alpha=-torch.rand(B, T, H, generator=g),
             log_mu=-torch.rand(B, T, H, generator=g),
             beta=torch.rand(B, T, H, generator=g),
             eta=torch.rand(B, T, H, generator=g) * 2)
    got = reference_recurrence("native", f, {})
    qn, kn = l2norm_ref(f["q"]), l2norm_ref(f["k"])
    want = recurrent_momentum_delta_rule_ref(
        qn, kn, f["v"], f["log_alpha"], f["log_mu"],
        (kn * f["log_alpha"].exp().unsqueeze(-1)), beta=f["beta"],
        eta=f["eta"])[0]
    assert _dev(got, want) == 0.0
    # Triton l2norm semantics: x / sqrt(sum x^2 + 1e-6), not F.normalize
    x = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    assert abs(l2norm_ref(x)[0, 0].item() - 3.0 / (25.0 + 1e-6) ** 0.5) < 1e-15
