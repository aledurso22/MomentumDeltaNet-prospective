"""The pilot's projection writers against independent references.

* a dense FP64 reference in the derivation's orientation (W in R^{dv x dk},
  explicit C_t and a linear solve, per-(batch, head) loops), sharing no code
  with projection.py;
* the existing v3 geometric writer at A = 0, G = 1 (outputs and gradients);
* the frozen local recurrence of the completed screens
  (prospective_rtrl_experiment @ cf8f87c, diagnostics/five_writer_screen.py,
  `Standalone`), transcribed below and, where that checkout exists, also
  imported unchanged.
Every test uses unequal key/value dimensions.
"""

import importlib.util
import os

import numpy as np
import pytest
import torch

from prospective.projection import (C0, INERTIA_A0, INERTIA_R0,
                                    PROJECTION_ARMS, coefficients,
                                    normalized_direction, projection_writer,
                                    run_projection, state_numbers)
from prospective.writers import geometric_writer

B, T, H, DK, DV = 2, 11, 3, 5, 7
F64 = torch.float64
SCALE = DK ** -0.5


def data(seed=0, T_=T, dk=DK, dv=DV, dtype=F64, B_=B, H_=H):
    g = torch.Generator().manual_seed(seed)
    k = torch.nn.functional.normalize(torch.randn(B_, T_, H_, dk, generator=g, dtype=dtype), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(B_, T_, H_, dk, generator=g, dtype=dtype), dim=-1)
    v = torch.randn(B_, T_, H_, dv, generator=g, dtype=dtype)
    return q, k, v


def heads(*vals, dtype=F64):
    return torch.tensor(vals, dtype=dtype)


def maxrel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


# ------------------------------------------------------------ dense FP64 ref
def dense(q, k, v, geometry, A=None, r=None, c0=C0, scale=SCALE):
    Bn, Tn, Hn, K = k.shape
    out = torch.zeros_like(v)
    for b in range(Bn):
        for h in range(Hn):
            W = torch.zeros(v.shape[-1], K, dtype=F64)
            V = torch.zeros_like(W)
            C = c0 * torch.eye(K, dtype=F64)
            for t in range(Tn):
                kt = k[b, t, h]
                if geometry == "geometric":
                    C = C + torch.outer(kt, kt)
                    x = torch.linalg.solve(C, kt)
                else:
                    x = kt
                bt = x / (kt @ x)
                if A is None:
                    W = W + torch.outer(v[b, t, h] - W @ kt, bt)
                else:
                    a_, d_ = A[h], r[h] * A[h]
                    U = W + V
                    D = torch.outer(v[b, t, h] - U @ kt, bt)
                    W = U + D
                    V = (1 - d_) * V + (a_ - d_) * D
                out[b, t, h] = scale * (W @ q[b, t, h])
    return out


@pytest.mark.parametrize("geometry", ["euclidean", "geometric"])
def test_no_motion_matches_dense_fp64(geometry):
    q, k, v = data(1)
    assert maxrel(projection_writer(q, k, v, geometry, SCALE),
                  dense(q, k, v, geometry)) < 1e-12


@pytest.mark.parametrize("A,r", [((0.003125,) * 3, (0.5,) * 3),
                                 ((0.3, 0.0, 0.9), (0.4, 0.7, 1.0)),
                                 ((1.0, 0.5, 0.2), (0.0, 0.25, 0.6))])
def test_motion_matches_dense_fp64(A, r):
    q, k, v = data(2)
    A, r = heads(*A), heads(*r)
    assert maxrel(projection_writer(q, k, v, "geometric", SCALE, A=A, r=r),
                  dense(q, k, v, "geometric", A, r)) < 1e-11


def test_written_key_is_recalled_exactly():
    """b^T k = 1, so W_t k_t = v_t after the write, for all three rules
    (for the motion arm: W k = U k + (v - U k) = v)."""
    q, k, v = data(3)
    for arm in PROJECTION_ARMS:
        _, states = run_projection(arm, q, k, v, dict(A=heads(.3, .1, .9), r=heads(.2, .5, 0.)),
                                   SCALE, return_states=True)
        for t, st in enumerate(states):
            recalled = (k[:, t].unsqueeze(-2) @ st["S"]).squeeze(-2)
            assert (recalled - v[:, t]).abs().max() < 1e-12, (arm, t)


# --------------------------------------------- vs the existing v3 writer
def test_geometric_projection_is_v3_geometric_writer_at_A0_G1():
    """Outputs AND gradients (q, k, v) against writers.geometric_writer with
    A = 0, G = 1, c0 = .03, lam = 1 (no forgetting)."""
    q, k, v = (x.requires_grad_(True) for x in data(4))
    o1 = run_projection("geometric_projection", q, k, v, {}, SCALE)
    o2 = geometric_writer(q, k, v, A=0.0, G=1.0, c0=C0, lam=1.0, scale=SCALE)
    assert maxrel(o1, o2) < 1e-13
    g = torch.randn_like(o1)
    g1 = torch.autograd.grad(o1, (q, k, v), g)
    g2 = torch.autograd.grad(o2, (q, k, v), g)
    for a, b in zip(g1, g2):
        assert maxrel(a, b) < 1e-12


def test_zero_drag_is_v3_geometric_writer_with_the_same_A():
    """r = 0 (d = 0) is the v3 fixed-gain geometric writer with G = 1 at the
    same A, c0 = .03, lam = 1."""
    q, k, v = data(5)
    for a_ in (0.003125, 0.05, 0.4):
        o1 = projection_writer(q, k, v, "geometric", SCALE, A=heads(a_, a_, a_), r=heads(0, 0, 0))
        o2 = geometric_writer(q, k, v, A=a_, G=1.0, c0=C0, lam=1.0, scale=SCALE)
        assert maxrel(o1, o2) < 1e-12, a_


# --------------------------------------------- boundary identities (exact)
def test_no_motion_boundaries_are_bitwise_geometric_projection():
    """A = 0 (d = 0, V stays 0) and r = 1 (A - d = 0, V stays 0), per head."""
    q, k, v = data(6)
    ref = run_projection("geometric_projection", q, k, v, {}, SCALE)
    for A, r in (((0., 0., 0.), (.5, 0., 1.)), ((.3, .9, .003125), (1., 1., 1.))):
        o = projection_writer(q, k, v, "geometric", SCALE, A=heads(*A), r=heads(*r))
        assert torch.equal(o, ref), (A, r)
    # mixed: only the boundary heads coincide
    o = projection_writer(q, k, v, "geometric", SCALE, A=heads(0., .3, .3), r=heads(.5, 1., .5))
    assert torch.equal(o[:, :, :2], ref[:, :, :2])
    assert not torch.equal(o[:, :, 2], ref[:, :, 2])


def test_motion_is_active_at_initialisation():
    q, k, v = data(7)
    ref = run_projection("geometric_projection", q, k, v, {}, SCALE)
    o = projection_writer(q, k, v, "geometric", SCALE,
                          A=heads(INERTIA_A0, INERTIA_A0, INERTIA_A0),
                          r=heads(INERTIA_R0, INERTIA_R0, INERTIA_R0))
    assert maxrel(o, ref) > 1e-6


# ------------------------------------------ the frozen local recurrence
class Standalone:
    """Transcribed from prospective_rtrl_experiment @ cf8f87c,
    diagnostics/five_writer_screen.py::Standalone (fam geo_damped /
    geo_no_motion / euclidean_zero), NumPy float64; dims parameterised."""

    def __init__(self, kind, c, dk, dv):
        self.kind, self.c = kind, c
        self.W = np.zeros((dv, dk))
        self.V = np.zeros((dv, dk))
        self.Ci = np.eye(dk) / c["lam"] if c.get("lam") is not None else None

    def step(self, k, y):
        c = self.c
        if self.Ci is not None:
            ck = self.Ci @ k
            self.Ci -= np.outer(ck, ck) / (1.0 + k @ ck)
            x = self.Ci @ k
            b = x / (k @ x)
        else:
            b = k / (k @ k)
        if self.kind == "nomotion":
            self.W = self.W + c["G"] * np.outer(y - self.W @ k, b)
            return
        U = self.W + self.V
        D = np.outer(y - U @ k, b)
        self.W = U + c["G"] * D
        self.V = (1.0 - c["d"]) * self.V + (c["A"] - c["d"] * c["G"]) * D


def run_standalone(cls_or_factory, q, k, v):
    Bn, Tn, Hn, K = k.shape
    out = torch.zeros_like(v)
    for b in range(Bn):
        for h in range(Hn):
            s = cls_or_factory(K, v.shape[-1])
            for t in range(Tn):
                s.step(k[b, t, h].numpy(), v[b, t, h].numpy())
                out[b, t, h] = SCALE * torch.from_numpy(s.W @ q[b, t, h].numpy())
    return out


FROZEN = {   # pilot arm -> (kind, frozen config) of the local screens
    "euclidean_projection": ("nomotion_euc", dict(G=1.0, lam=None)),
    "geometric_projection": ("nomotion", dict(G=1.0, lam=0.03)),
    "geometric_motion_learned": ("motion", dict(A=0.003125, G=1.0,
                                                d=0.003125 / 2, lam=0.03)),
}


def _kind(kind):
    return "nomotion" if kind.startswith("nomotion") else "motion"


@pytest.mark.parametrize("arm", PROJECTION_ARMS)
def test_matches_frozen_local_recurrence_at_initialisation(arm):
    """At the declared initial values (A = .003125, r = .5 -> d = .0015625,
    c0 = lam_screen = .03) every pilot rule IS the frozen screen rule."""
    q, k, v = data(8)
    kind, c = FROZEN[arm]
    ref = run_standalone(lambda dk, dv: Standalone(_kind(kind), c, dk, dv), q, k, v)
    leaves = dict(A=torch.full((H,), INERTIA_A0, dtype=F64),
                  r=torch.full((H,), INERTIA_R0, dtype=F64))
    assert maxrel(run_projection(arm, q, k, v, leaves, SCALE), ref) < 1e-12


SCREEN = os.environ.get("PROSPECTIVE_RTRL_DIR", os.path.expanduser(
    "~/Documents/prospective_rtrl_experiment"))
SCREEN_FILE = os.path.join(SCREEN, "diagnostics", "five_writer_screen.py")


@pytest.mark.skipif(not os.path.exists(SCREEN_FILE),
                    reason="local screen checkout not present (cluster)")
@pytest.mark.parametrize("arm", PROJECTION_ARMS)
def test_matches_imported_frozen_screen_code(arm):
    spec = importlib.util.spec_from_file_location("five_writer_screen", SCREEN_FILE)
    F = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(F)
    fam = {"euclidean_projection": "euclidean_zero",
           "geometric_projection": "geo_no_motion",
           "geometric_motion_learned": "geo_damped"}[arm]
    kind, c = FROZEN[arm]
    if fam == "euclidean_zero":       # the screen's motion family at A = d = 0
        c = dict(A=0.0, G=1.0, d=0.0, lam=None)

    def make(dk, dv):
        F.DK, F.DV = dk, dv
        return F.Standalone(fam, c)
    q, k, v = data(9)
    leaves = dict(A=torch.full((H,), INERTIA_A0, dtype=F64),
                  r=torch.full((H,), INERTIA_R0, dtype=F64))
    assert maxrel(run_projection(arm, q, k, v, leaves, SCALE),
                  run_standalone(make, q, k, v)) < 1e-12


# ---------------------------------------------------------------- gradients
def test_gradcheck_through_geometry_and_coefficients():
    q, k, v = data(10, T_=5, dk=3, dv=4, B_=1, H_=2)
    A = heads(0.3, 0.6).requires_grad_(True)
    r = heads(0.4, 0.2).requires_grad_(True)
    q, k, v = (x.requires_grad_(True) for x in (q, k, v))
    assert torch.autograd.gradcheck(
        lambda q_, k_, v_, A_, r_: projection_writer(q_, k_, v_, "geometric", 0.7, A=A_, r=r_),
        (q, k, v, A, r))
    for geo in ("euclidean", "geometric"):
        assert torch.autograd.gradcheck(
            lambda q_, k_, v_: projection_writer(q_, k_, v_, geo, 0.7), (q, k, v))


def test_geometry_is_differentiated_not_detached():
    """dL/dk through C^-1 is part of the gradient: detaching C^-1 changes it."""
    q, k, v = data(11)
    k.requires_grad_(True)
    g = torch.autograd.grad(projection_writer(q, k, v, "geometric", SCALE).sum(), k)[0]

    def detached(q, k, v):
        Bn, Tn, Hn, K = k.shape
        S = v.new_zeros(Bn, Hn, K, v.shape[-1])
        Ci = (torch.eye(K, dtype=F64) / C0).expand(Bn, Hn, K, K)
        out = torch.empty_like(v)
        for t in range(Tn):
            kt = k[:, t]
            ck = (Ci @ kt.unsqueeze(-1)).squeeze(-1)
            Ci = (Ci - (ck[..., :, None] * ck[..., None, :])
                  / (1 + (kt * ck).sum(-1))[..., None, None]).detach()
            b = normalized_direction((Ci @ kt.unsqueeze(-1)).squeeze(-1), kt)
            S = S + b[..., :, None] * (v[:, t] - (kt.unsqueeze(-2) @ S).squeeze(-2))[..., None, :]
            out[:, t] = SCALE * (q[:, t].unsqueeze(-2) @ S).squeeze(-2)
        return out
    gd = torch.autograd.grad(detached(q, k, v).sum(), k)[0]
    assert maxrel(gd, g) > 1e-3


def test_coefficient_gradients_are_nonzero_at_initialisation():
    q, k, v = data(12, dtype=torch.float32)
    A = torch.full((H,), INERTIA_A0, requires_grad=True)
    r = torch.full((H,), INERTIA_R0, requires_grad=True)
    o = projection_writer(q, k, v, "geometric", SCALE, A=A, r=r)
    gA, gr = torch.autograd.grad((o * torch.randn_like(o)).sum(), (A, r))
    assert (gA.abs() > 0).all() and (gr.abs() > 0).all()


# ---------------------------------------------------------------- causality
@pytest.mark.parametrize("arm", PROJECTION_ARMS)
def test_causal(arm):
    q, k, v = data(13)
    leaves = dict(A=heads(.3, .5, .1), r=heads(.2, .9, .5))
    o = run_projection(arm, q, k, v, leaves, SCALE)
    for t in (0, 4, 9):
        q2, k2, v2 = data(99)
        q3 = torch.cat([q[:, :t + 1], q2[:, t + 1:]], 1)
        k3 = torch.cat([k[:, :t + 1], k2[:, t + 1:]], 1)
        v3 = torch.cat([v[:, :t + 1], v2[:, t + 1:]], 1)
        o3 = run_projection(arm, q3, k3, v3, leaves, SCALE)
        assert torch.equal(o[:, :t + 1], o3[:, :t + 1])
        assert not torch.equal(o[:, t + 1:], o3[:, t + 1:])


# ---------------------------------------------------------------- zero keys
@pytest.mark.parametrize("arm", PROJECTION_ARMS)
def test_zero_keys_write_nothing_and_keep_gradients_finite(arm):
    q, k, v = data(14)
    k = k.clone()
    k[:, 3] = 0.0
    k[0, 7, 1] = 0.0
    leaves = dict(A=heads(.3, .5, .1, dtype=F64).requires_grad_(True),
                  r=heads(.2, .9, .5, dtype=F64).requires_grad_(True))
    q, k, v = (x.requires_grad_(True) for x in (q, k, v))
    o, states = run_projection(arm, q, k, v, leaves, SCALE, return_states=True)
    assert torch.isfinite(o).all()
    for t, mask in ((3, (slice(None), slice(None))), (7, (0, 1))):
        prev, cur = states[t - 1], states[t]
        U = prev["S"] + prev["V"] if "V" in prev else prev["S"]
        assert torch.equal(cur["S"][mask], U[mask]), (arm, t)       # D = 0
        if "Cinv" in cur:
            assert torch.equal(cur["Cinv"][mask], prev["Cinv"][mask])
        if "V" in cur:
            d = leaves["r"] * leaves["A"]
            want = (1 - d)[None, :, None, None] * prev["V"]
            assert torch.equal(cur["V"][mask], want[mask])
    grads = torch.autograd.grad((o * torch.randn_like(o)).sum(),
                                (q, k, v) + ((leaves["A"], leaves["r"])
                                             if arm == "geometric_motion_learned" else ()))
    assert all(torch.isfinite(g_).all() for g_ in grads)


def test_normalized_direction_zero_guard_is_exact():
    k = torch.tensor([[0.0, 0.0], [1e-30, 0.0]], dtype=F64)
    b = normalized_direction(k, k)
    assert torch.equal(b[0], torch.zeros(2, dtype=F64))
    assert b[1, 0] == 1e30                  # tiny nonzero keys: exact projection


# --------------------------------------------- storage, symmetry, state
@pytest.mark.parametrize("arm", PROJECTION_ARMS)
def test_carried_state_matches_the_declared_storage(arm):
    """The no-motion writers carry no velocity matrix; the euclidean one no
    covariance either. Carried numbers per (sequence, head) == state_numbers."""
    q, k, v = data(15)
    _, states = run_projection(arm, q, k, v, dict(A=heads(.3, .5, .1), r=heads(.2, .9, .5)),
                               SCALE, return_states=True)
    want = {"euclidean_projection": {"S"}, "geometric_projection": {"S", "Cinv"},
            "geometric_motion_learned": {"S", "Cinv", "V"}}[arm]
    assert set(states[-1]) == want
    per_seq_head = sum(x[0, 0].numel() for x in states[-1].values())
    assert per_seq_head == state_numbers(arm, DK, DV)


def test_storage_numbers_at_pilot_size():
    dk, dv = 32, 64
    assert state_numbers("euclidean_projection", dk, dv) == 2048
    assert state_numbers("geometric_projection", dk, dv) == 2048 + 1024
    assert state_numbers("geometric_motion_learned", dk, dv) == 4096 + 1024
    assert state_numbers("native", dk, dv) == 4096


@pytest.mark.parametrize("dtype", [F64, torch.float32])
def test_inverse_geometry_symmetric_positive_definite(dtype):
    q, k, v = data(16, T_=52, dk=32, dv=64, dtype=dtype, B_=4, H_=4)
    _, states = run_projection("geometric_projection", q, k, v, {}, 32 ** -0.5,
                               return_states=True)
    for st in states:
        Ci = st["Cinv"]
        assert torch.equal(Ci, Ci.transpose(-1, -2))
        assert torch.linalg.eigvalsh(Ci.double()).min() > 0


def test_inverse_matches_dense_inverse():
    q, k, v = data(17)
    _, states = run_projection("geometric_projection", q, k, v, {}, SCALE,
                               return_states=True)
    C = C0 * torch.eye(DK, dtype=F64).expand(B, H, DK, DK)
    for t, st in enumerate(states):
        C = C + k[:, t, :, :, None] * k[:, t, :, None, :]
        assert maxrel(st["Cinv"], torch.linalg.inv(C)) < 1e-10


def test_states_reset_between_calls():
    q, k, v = data(18)
    a = projection_writer(q, k, v, "geometric", SCALE, A=heads(.3, .3, .3), r=heads(.5, .5, .5))
    projection_writer(*data(19), "geometric", SCALE, A=heads(.3, .3, .3), r=heads(.5, .5, .5))
    b = projection_writer(q, k, v, "geometric", SCALE, A=heads(.3, .3, .3), r=heads(.5, .5, .5))
    assert torch.equal(a, b)


def test_coefficient_record():
    c = coefficients(torch.tensor([0.003125, 0.0, 0.5]), torch.tensor([0.5, 0.3, 1.0]))
    assert c["d"] == pytest.approx([0.0015625, 0.0, 0.5])
    assert c["A_minus_d"] == pytest.approx([0.0015625, 0.0, 0.0])


def test_declared_constants():
    assert (C0, INERTIA_A0, INERTIA_R0) == (0.03, 0.003125, 0.5)
    assert INERTIA_R0 * INERTIA_A0 == 0.0015625
