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

try:
    import fla  # noqa: F401
    pytest.skip("real fla available: model checks run in test_integration.py",
                allow_module_level=True)
except Exception:
    for _k in [k for k in sys.modules if k == "fla" or k.startswith("fla.")]:
        del sys.modules[_k]
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cpu_stub"))

from prospective.layer_ref import mixer_forward_ref, params_as  # noqa: E402
from prospective.model import ARM_LEAVES, build, shared_parameter_check  # noqa: E402
from prospective.mixer import ProspectiveMDN  # noqa: E402
from prospective.rules import ARMS  # noqa: E402
from prospective.task import PRODUCTION_TASK, make_batch  # noqa: E402

ALL_ARMS = ARMS


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
