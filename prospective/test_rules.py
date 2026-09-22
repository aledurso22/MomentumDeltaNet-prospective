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


def test_leaves_actually_move_under_optimization():
    """The regression that cost seed 501: with M = softplus(raw) started at the
    native boundary, d(softplus)/d(raw) ~ 1e-4 froze M and gamma in place and
    the generalized arm ran as TSS for 4000 steps. Stored directly, they move."""
    import importlib
    model_mod = importlib.import_module("prospective.model")
    m = model_mod.build("generalized", vocab=40, seed=0, d_model=32,
                        n_heads=2, n_layers=1)
    mix = m.blocks[0].mix
    before = (mix.fil_M.detach().clone(), mix.fil_gamma.detach().clone())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 40, (4, 24))
    tgt = torch.randint(0, 40, (4, 24))
    for _ in range(25):
        loss = torch.nn.functional.cross_entropy(
            m(x).reshape(-1, 40), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        m.project_()
    moved_M = (mix.fil_M - before[0]).abs().max().item()
    moved_g = (mix.fil_gamma - before[1]).abs().max().item()
    assert moved_M > 1e-3, moved_M
    assert moved_g > 1e-3, moved_g


def test_tss_leaves_stay_pinned_at_zero():
    import importlib
    model_mod = importlib.import_module("prospective.model")
    m = model_mod.build("tss", vocab=40, seed=0, d_model=32, n_heads=2,
                        n_layers=1)
    mix = m.blocks[0].mix
    opt = torch.optim.AdamW(m.parameters(), lr=3e-2)
    x = torch.randint(0, 40, (4, 24))
    for _ in range(10):
        m(x).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        m.project_()
    assert torch.equal(mix.fil_M, torch.zeros_like(mix.fil_M))
    assert torch.equal(mix.fil_gamma, torch.zeros_like(mix.fil_gamma))
    # and T must be repaired above the drive-stability bound h/2
    assert (mix.fil_T > 0.5).all(), mix.fil_T


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
