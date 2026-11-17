"""Model-level checks on CPU, with the REAL official layer source and CPU
stand-ins for its Triton modules (see cpu_stub/README.md).

Plumbing only: shapes, every arm, exact causality, identical shared init,
gradients reaching the parameters, the FP64 functional reference agreeing
with the module. Triton numerics are checked by test_integration.py on GPU.
Skipped where the real `fla` imports (there, test_integration.py runs).
"""

import os
import sys

import pytest
import torch

SKIP_REASON = "real fla available: model checks run in test_integration.py"
try:
    import fla
    REAL_FLA = "cpu_stub" not in (fla.__file__ or "")
except Exception:
    REAL_FLA = False
if REAL_FLA:
    pytest.skip(SKIP_REASON, allow_module_level=True)
if "fla" not in sys.modules:
    for _k in [k for k in sys.modules if k == "fla" or k.startswith("fla.")]:
        del sys.modules[_k]
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cpu_stub"))

from prospective.layer_ref import mixer_forward_ref, params_as  # noqa: E402
from prospective.model import ARM_LEAVES, build, shared_parameter_check  # noqa: E402
from prospective.mixer import ALL_ARMS, ProspectiveMDN, unused_by_design  # noqa: E402
from prospective.model import unused_parameters  # noqa: E402
from prospective.task import PRODUCTION_TASK, make_batch  # noqa: E402
from prospective.writers import WRITER_ARMS  # noqa: E402


def rel(a, b):
    b = b.detach().double()
    return ((a.detach().double() - b).norm() / b.norm()).item()


def layer(arm, seed=0):
    torch.manual_seed(seed)
    m = ProspectiveMDN(arm, hidden_size=128, num_heads=4, head_dim=32,
                       expand_v=2)
    m.official_model_init_()
    return m


def test_real_official_layer_is_what_runs():
    from fla.layers.momentum_deltanet import MomentumDeltaNet
    assert ProspectiveMDN.__mro__[1] is MomentumDeltaNet
    assert MomentumDeltaNet.forward.__code__.co_filename.endswith(
        os.path.join("flash-linear-attention", "fla", "layers",
                     "momentum_deltanet.py"))


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_layer_fp32_matches_fp64_functional_reference(arm):
    m = layer(arm)
    h = torch.randn(2, 52, 128, generator=torch.Generator().manual_seed(1))
    g = torch.randn(2, 52, 128, generator=torch.Generator().manual_seed(2))
    h32 = h.clone().requires_grad_(True)
    o = m(h32)[0]
    o.backward(g)
    P = params_as(m, torch.float64)
    h64 = h.double().requires_grad_(True)
    o64 = mixer_forward_ref(m, h64, P)
    o64.backward(g.double())
    assert rel(o, o64) < 1e-4
    assert rel(h32.grad, h64.grad) < 1e-4
    for n, p in m.named_parameters():
        if P[n].grad is not None and P[n].grad.norm() > 0:
            assert p.grad is not None, n
            assert rel(p.grad, P[n].grad) < 1e-4, (n, rel(p.grad, P[n].grad))


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_model_causality_exact(arm):
    m = build(arm, vocab=58, seed=0).eval()
    gen = torch.Generator().manual_seed(9)
    x = make_batch(3, **PRODUCTION_TASK, generator=gen)["x"]
    with torch.no_grad():
        o = m(x)
        for t in (0, 9, 27, 50):
            x2 = x.clone()
            x2[:, t + 1:] = torch.randint(0, 58, x2[:, t + 1:].shape, generator=gen)
            o2 = m(x2)
            assert torch.equal(o[:, :t + 1], o2[:, :t + 1]), (arm, t)
            assert not torch.equal(o[:, t + 1:], o2[:, t + 1:]), (arm, t)


def test_shared_init_identical_across_arms():
    models = {a: build(a, vocab=58, seed=501) for a in ALL_ARMS}
    ref = models["native"]
    for a, m in models.items():
        assert shared_parameter_check(ref, m) == [], a
        assert set(ref.state_dict()) <= set(m.state_dict()), a
        extra = set(m.state_dict()) - set(ref.state_dict())
        assert all(n.split(".")[-1] in ARM_LEAVES for n in extra), (a, extra)


def test_whole_sequence_guard():
    with pytest.raises(NotImplementedError):
        layer("native")(torch.randn(1, 4, 128), use_cache=True)


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_unused_parameters_are_exactly_the_declared_ones(arm):
    m = build(arm, vocab=58, seed=0)
    b = make_batch(4, **PRODUCTION_TASK, generator=torch.Generator().manual_seed(0))
    torch.nn.functional.cross_entropy(m(b["x"]).reshape(-1, 58), b["y"].reshape(-1),
                                      ignore_index=-100).backward()
    unused = set(unused_parameters(m))
    want = {f"blocks.{i}.mix.{n}" for i in range(2) for n in unused_by_design(arm)}
    assert unused == want, unused ^ want


@pytest.mark.parametrize("arm", WRITER_ARMS)
def test_writer_gradients_reach_shared_frontend(arm):
    m = layer(arm)
    m(torch.randn(2, 52, 128)).__getitem__(0).sum().backward()
    for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "q_conv1d.weight",
              "k_conv1d.weight", "v_conv1d.weight", "D_log", "g_proj.weight",
              "o_norm.weight", "o_proj.weight"):
        p = dict(m.named_parameters())[n]
        assert p.grad is not None and p.grad.abs().max() > 0, n


# ------------------------------------------------ v4 pilot: learned inertia
from prospective.model import NO_DECAY_LEAVES, make_optimizer, param_groups  # noqa: E402
from prospective.projection import (INERTIA_A0, INERTIA_R0,  # noqa: E402
                                    PROJECTION_ARMS, projection_writer)
from prospective.reference import l2norm_ref  # noqa: E402

MOTION = "geometric_motion_learned"


def test_inertia_leaves_initialise_as_declared():
    m = build(MOTION, vocab=58, seed=0)
    for blk in m.blocks:
        assert torch.equal(blk.mix.inertia_A, torch.full((4,), INERTIA_A0))
        assert torch.equal(blk.mix.inertia_r, torch.full((4,), INERTIA_R0))
    c = m.inertia()
    assert set(c) == {"layer0", "layer1"}
    assert c["layer0"]["d"] == pytest.approx([INERTIA_A0 * INERTIA_R0] * 4)
    assert build("geometric_projection", vocab=58, seed=0).inertia() == {}


def test_layer_motion_arm_at_init_is_the_frozen_recurrence_on_frontend_outputs():
    """In the layer, the learned arm at init runs exactly the declared
    recurrence on the shared frontend's normalised q, k and v."""
    m = layer(MOTION)
    h = torch.randn(2, 52, 128, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        f = m.frontend(h)
        o = m.backend_reference(f)
        want = projection_writer(l2norm_ref(f["q"]), l2norm_ref(f["k"]), f["v"],
                                 "geometric", 32 ** -0.5,
                                 A=torch.full((4,), INERTIA_A0),
                                 r=torch.full((4,), INERTIA_R0))
    assert torch.equal(o, want)


@pytest.mark.parametrize("A,r", [(0.0, 0.5), (0.3, 1.0)])
def test_layer_no_motion_boundaries_bitwise(A, r):
    g, mo = layer("geometric_projection"), layer(MOTION)
    with torch.no_grad():
        mo.inertia_A.fill_(A); mo.inertia_r.fill_(r)
        h = torch.randn(2, 52, 128, generator=torch.Generator().manual_seed(4))
        assert torch.equal(g(h)[0], mo(h)[0])


def test_inertia_excluded_from_weight_decay_only():
    m = build(MOTION, vocab=58, seed=0)
    groups = param_groups(m, 0.01)
    nd = {id(p) for p in groups[1]["params"]}
    names = {n for n, p in m.named_parameters() if id(p) in nd}
    assert names == {f"blocks.{i}.mix.{x}" for i in range(2) for x in NO_DECAY_LEAVES}
    assert groups[0]["weight_decay"] == 0.01 and groups[1]["weight_decay"] == 0.0
    for arm in ("native", "euclidean_projection", "geometric_projection", "qhm"):
        assert len(param_groups(build(arm, vocab=58, seed=0), 0.01)) == 1, arm


def test_non_pilot_arms_optimise_exactly_as_before():
    """A single-group make_optimizer is the old plain AdamW, bitwise."""
    b = make_batch(4, **PRODUCTION_TASK, generator=torch.Generator().manual_seed(1))
    outs = []
    for mk in (lambda m: make_optimizer(m, 3e-3, 0.01),
               lambda m: torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)):
        m = build("native", vocab=58, seed=0)
        o = mk(m)
        for _ in range(2):
            loss = torch.nn.functional.cross_entropy(m(b["x"]).reshape(-1, 58),
                                                     b["y"].reshape(-1), ignore_index=-100)
            o.zero_grad(); loss.backward(); o.step()
        outs.append(m.state_dict())
    assert all(torch.equal(outs[0][n], outs[1][n]) for n in outs[0])


def test_inertia_projection_keeps_the_box_and_no_decay_holds():
    m = build(MOTION, vocab=58, seed=0)
    mix = m.blocks[0].mix
    with torch.no_grad():
        mix.inertia_A.copy_(torch.tensor([-0.2, 0.5, 1.7, 0.0]))
        mix.inertia_r.copy_(torch.tensor([1.3, -0.1, 0.5, 0.2]))
    m.project_()
    assert mix.inertia_A.tolist() == [0.0, 0.5, 1.0, 0.0]
    assert mix.inertia_r.tolist() == pytest.approx([1.0, 0.0, 0.5, 0.2])
    # with zero gradient, AdamW with decay 0 leaves them where they are
    opt = make_optimizer(m, 3e-3, 0.5)
    before = mix.inertia_A.detach().clone()
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.equal(mix.inertia_A, before)


def test_inertia_gets_gradient_in_the_model():
    m = build(MOTION, vocab=58, seed=0)
    b = make_batch(4, **PRODUCTION_TASK, generator=torch.Generator().manual_seed(0))
    torch.nn.functional.cross_entropy(m(b["x"]).reshape(-1, 58), b["y"].reshape(-1),
                                      ignore_index=-100).backward()
    for blk in m.blocks:
        assert blk.mix.inertia_A.grad.abs().min() > 0
        assert blk.mix.inertia_r.grad.abs().min() > 0


@pytest.mark.parametrize("arm", PROJECTION_ARMS)
def test_zero_raw_keys_in_the_layer_stay_finite(arm):
    """A key projection of exactly zero (l2norm -> 0) writes nothing; the
    layer's output and every gradient stay finite."""
    m = layer(arm)
    with torch.no_grad():
        m.k_proj.weight.zero_()            # every raw key is 0 -> conv -> silu(0) = 0
    h = torch.randn(2, 52, 128, requires_grad=True)
    o = m(h)[0]
    o.sum().backward()
    assert torch.isfinite(o).all() and torch.isfinite(h.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
