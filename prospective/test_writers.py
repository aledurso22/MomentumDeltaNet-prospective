"""The three v3 writers against independent dense FP64 references.

The references below work in the DERIVATION's orientation (W, Z in
R^{dv x dk}, prediction W k), with explicit matrices and per-(batch, head)
Python loops, and share no code with writers.py. Every test uses unequal
key/value dimensions.
"""

import math

import pytest
import torch

from prospective.writers import (REFERENCE_CFG, geometric_writer,
                                 joint_writer, rls_writer)

B, T, H, DK, DV = 2, 9, 2, 5, 7
F64 = torch.float64


def data(seed=0, T_=T, dk=DK, dv=DV, dtype=F64, B_=B, H_=H):
    g = torch.Generator().manual_seed(seed)
    k = torch.nn.functional.normalize(torch.randn(B_, T_, H_, dk, generator=g, dtype=dtype), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(B_, T_, H_, dk, generator=g, dtype=dtype), dim=-1)
    v = torch.randn(B_, T_, H_, dv, generator=g, dtype=dtype)
    return q, k, v


def F_matrix(K):
    I, Z = torch.eye(K, dtype=F64), torch.zeros(K, K, dtype=F64)
    return torch.cat([torch.cat([I, I], 1), torch.cat([Z, I], 1)], 0)


# ------------------------------------------------------------ dense refs
def joint_dense(q, k, v, rho, s2, p0, T0, scale):
    Bn, Tn, Hn, K = k.shape
    Fm = F_matrix(K)
    out = torch.zeros_like(v)
    Ws, Zs, Ps = {}, {}, {}
    for b in range(Bn):
        for hh in range(Hn):
            W = torch.zeros(v.shape[-1], K, dtype=F64)
            Z = torch.zeros_like(W)
            P = torch.diag(torch.cat([torch.full((K,), p0, dtype=F64), torch.full((K,), p0 / T0 ** 2, dtype=F64)]))
            for t in range(Tn):
                if t > 0:
                    W, Z = W + Z, Z
                    P = Fm @ P @ Fm.T / rho
                kt = k[b, t, hh]
                hv = torch.cat([kt, torch.zeros(K, dtype=F64)])
                bb = P @ hv
                s = s2 + hv @ bb
                r = v[b, t, hh] - W @ kt
                W = W + torch.outer(r, bb[:K] / s)
                Z = Z + torch.outer(r, bb[K:] / s)
                P = P - torch.outer(bb, bb) / s
                out[b, t, hh] = scale * (W @ q[b, t, hh])
                Ws[b, t, hh], Zs[b, t, hh], Ps[b, t, hh] = W, Z, P
    return out, Ws, Zs, Ps


def rls_dense(q, k, v, rho, s2, p0, scale):
    Bn, Tn, Hn, K = k.shape
    out = torch.zeros_like(v)
    for b in range(Bn):
        for hh in range(Hn):
            W = torch.zeros(v.shape[-1], K, dtype=F64)
            P = p0 * torch.eye(K, dtype=F64)
            for t in range(Tn):
                if t > 0:
                    P = P / rho
                kt = k[b, t, hh]
                bb = P @ kt
                s = s2 + kt @ bb
                W = W + torch.outer(v[b, t, hh] - W @ kt, bb / s)
                P = P - torch.outer(bb, bb) / s
                out[b, t, hh] = scale * (W @ q[b, t, hh])
    return out


def geometric_dense(q, k, v, A, G, c0, lam, scale):
    """Posterior form, explicit C_t = lam C_{t-1} + k k^T and a dense solve."""
    Bn, Tn, Hn, K = k.shape
    out = torch.zeros_like(v)
    us, xis = {}, {}
    for b in range(Bn):
        for hh in range(Hn):
            u = torch.zeros(v.shape[-1], K, dtype=F64)
            Z = torch.zeros_like(u)
            C = c0 * torch.eye(K, dtype=F64)
            for t in range(Tn):
                kt = k[b, t, hh]
                C = lam * C + torch.outer(kt, kt)
                y = torch.linalg.solve(C, kt)
                d = y / (kt @ y)
                xi = torch.outer(u @ kt - v[b, t, hh], d)
                us[b, t, hh], xis[b, t, hh] = u, xi
                Wp = u - G * xi
                Z = Z - A * xi
                out[b, t, hh] = scale * (Wp @ q[b, t, hh])
                u = Wp + Z
    return out, us, xis


def maxrel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-300)).item()


SC = DK ** -0.5


# ------------------------------------------------------------ joint
@pytest.mark.parametrize("rho,T0", [(1.0, 1.0), (1.0, 4.0), (0.97, 2.0), (0.9, 16.0)])
def test_joint_matches_dense_fp64(rho, T0):
    q, k, v = data(1)
    got, st = joint_writer(q, k, v, rho, 1.0, 10.0, T0, SC, return_states=True)
    ref, Ws, Zs, Ps = joint_dense(q, k, v, rho, 1.0, 10.0, T0, SC)
    assert maxrel(got, ref) < 1e-12
    for t in (0, T // 2, T - 1):
        for b in range(B):
            for hh in range(H):
                assert maxrel(st[t]["S"][b, hh], Ws[b, t, hh].T) < 1e-12
                assert maxrel(st[t]["Sz"][b, hh], Zs[b, t, hh].T) < 1e-12
                assert maxrel(st[t]["P"][b, hh], Ps[b, t, hh]) < 1e-12


def test_joint_factorisation_equals_global_kalman_filter():
    """The explicit Kalman filter over vec([W Z]) (size 2 dk dv), with
    transition kron(I, F), observation kron(I, h^T), prior kron(I, P0) and
    noise sigma2 I. Its covariance must stay kron(I_dv, P) and its mean
    [W Z], tick by tick."""
    dk, dv, Tn, rho, s2, p0, T0 = 2, 3, 7, 0.95, 0.7, 10.0, 3.0
    q, k, v = data(2, T_=Tn, dk=dk, dv=dv, B_=1, H_=1)
    _, st = joint_writer(q, k, v, rho, s2, p0, T0, dk ** -0.5, return_states=True)
    Id = torch.eye(dv, dtype=F64)
    Phi = torch.kron(Id, F_matrix(dk))
    P0 = torch.diag(torch.cat([torch.full((dk,), p0, dtype=F64), torch.full((dk,), p0 / T0 ** 2, dtype=F64)]))
    Sig = torch.kron(Id, P0)
    th = torch.zeros(dv * 2 * dk, dtype=F64)          # row-major vec of [W Z]
    for t in range(Tn):
        if t > 0:
            th = Phi @ th
            Sig = Phi @ Sig @ Phi.T / rho
        hv = torch.cat([k[0, t, 0], torch.zeros(dk, dtype=F64)])
        Hm = torch.kron(Id, hv.unsqueeze(0))            # [dv, dv*2dk]
        Sy = Hm @ Sig @ Hm.T + s2 * Id
        Kg = Sig @ Hm.T @ torch.linalg.inv(Sy)
        th = th + Kg @ (v[0, t, 0] - Hm @ th)
        Sig = Sig - Kg @ Hm @ Sig
        X = th.view(dv, 2 * dk)
        assert maxrel(st[t]["S"][0, 0], X[:, :dk].T) < 1e-11, t
        assert maxrel(st[t]["Sz"][0, 0], X[:, dk:].T) < 1e-11, t
        assert maxrel(torch.kron(Id, st[t]["P"][0, 0]), Sig) < 1e-11, t


def test_joint_timing_first_prior_and_posterior_read():
    q, k, v = data(3)
    rho, T0 = 0.9, 2.0
    out, st = joint_writer(q, k, v, rho, 1.0, 10.0, T0, SC, return_states=True)
    P0 = torch.block_diag(10.0 * torch.eye(DK, dtype=F64), (10.0 / T0 ** 2) * torch.eye(DK, dtype=F64))
    # initial values ARE the first prior: no prediction before t = 1
    assert torch.equal(st[0]["P_prior"][0, 0], P0)
    assert torch.equal(st[0]["S_prior"], torch.zeros_like(st[0]["S_prior"]))
    # t >= 2 prior is one prediction from the previous posterior
    Fm = F_matrix(DK)
    assert maxrel(st[1]["P_prior"][0, 0], Fm @ st[0]["P"][0, 0] @ Fm.T / rho) < 1e-14
    assert torch.equal(st[1]["S_prior"], st[0]["S"] + st[0]["Sz"])
    for t in range(T):
        post = SC * (q[:, t].unsqueeze(-2) @ st[t]["S"]).squeeze(-2)
        prior = SC * (q[:, t].unsqueeze(-2) @ st[t]["S_prior"]).squeeze(-2)
        assert torch.equal(out[:, t], post)
        assert not torch.allclose(out[:, t], prior)


def test_rho_discounts_information_not_motion():
    """rho enters only P: with the same P the motion transport is identical."""
    q, k, v = data(4)
    _, a = joint_writer(q, k, v, 0.8, 1.0, 10.0, 2.0, SC, return_states=True)
    assert torch.equal(a[1]["Sz_prior"], a[0]["Sz"])
    assert torch.equal(a[1]["S_prior"], a[0]["S"] + a[0]["Sz"])


# ------------------------------------------------------------ rls
@pytest.mark.parametrize("rho", [1.0, 0.98, 0.9])
def test_rls_matches_dense_fp64(rho):
    q, k, v = data(5)
    got = rls_writer(q, k, v, rho, 1.0, 10.0, SC)
    assert maxrel(got, rls_dense(q, k, v, rho, 1.0, 10.0, SC)) < 1e-12


def test_rls_at_rho_one_is_ridge_least_squares():
    """Independent closed form: W_T = (sum v k^T)(sum k k^T + s2/p0 I)^-1."""
    q, k, v = data(6, T_=12)
    s2, p0 = 0.5, 10.0
    _, st = rls_writer(q, k, v, 1.0, s2, p0, SC, return_states=True)
    for b in range(B):
        for hh in range(H):
            Kt, Vt = k[b, :, hh], v[b, :, hh]
            W = (Vt.T @ Kt) @ torch.linalg.inv(Kt.T @ Kt + (s2 / p0) * torch.eye(DK, dtype=F64))
            assert maxrel(st[-1]["S"][b, hh], W.T) < 1e-10


def test_rls_covariance_is_full_not_diagonal():
    q, k, v = data(7)
    _, st = rls_writer(q, k, v, 1.0, 1.0, 10.0, SC, return_states=True)
    P = st[-1]["P"][0, 0]
    assert (P - torch.diag(torch.diag(P))).abs().max() > 1e-3


# ------------------------------------------------------------ geometric
@pytest.mark.parametrize("A,G,c0,lam", [(0.05, 1.0, 0.1, 1.0), (0.2, 0.5, 0.1, 1.0),
                                        (0.05, 1.0, 1.0, 0.95)])
def test_geometric_matches_dense_fp64(A, G, c0, lam):
    q, k, v = data(8)
    got, st = geometric_writer(q, k, v, A, G, c0, lam, SC, return_states=True)
    ref, us, xis = geometric_dense(q, k, v, A, G, c0, lam, SC)
    assert maxrel(got, ref) < 1e-10
    for t in range(T):
        assert maxrel(st[t]["u"][0, 0], us[0, t, 0].T) < 1e-10 or t == 0


def test_geometric_reference_point_is_documented():
    assert REFERENCE_CFG["geometric"] == dict(A=0.05, G=1.0, c0=0.1, lam=1.0)


def test_geometric_eliminates_to_the_original_recurrences():
    """Posterior variables eliminated: (i) the tracker's two-state form
    V+ = V + A R, W+ = W + V+ + G R (R = -xi), whose W is u; (ii) the
    two-step form u_{t+1} = 2u_t - u_{t-1} - A xi_t - G(xi_t - xi_{t-1})."""
    q, k, v = data(9, T_=15)
    A, G = 0.05, 1.0
    _, st = geometric_writer(q, k, v, A, G, 0.1, 1.0, SC, return_states=True)
    xi = [s["xi"] for s in st]
    u = [s["u"] for s in st] + [st[-1]["W_post"] + st[-1]["Z_post"]]
    W, V = torch.zeros_like(u[0]), torch.zeros_like(u[0])
    for t in range(len(xi)):
        assert maxrel(W, u[t]) < 1e-12 or t == 0
        R = -xi[t]
        V = V + A * R
        W = W + V + G * R
    assert maxrel(W, u[-1]) < 1e-12
    for t in range(1, len(xi)):
        u_prev = u[t - 1]
        two = 2 * u[t] - u_prev - A * xi[t] - G * (xi[t] - xi[t - 1])
        assert maxrel(two, u[t + 1]) < 1e-11, t


def test_geometric_reads_corrected_memory_not_next_prediction():
    q, k, v = data(10)
    out, st = geometric_writer(q, k, v, 0.05, 1.0, 0.1, 1.0, SC, return_states=True)
    for t in range(T):
        assert torch.equal(out[:, t], SC * (q[:, t].unsqueeze(-2) @ st[t]["W_post"]).squeeze(-2))
        nxt = SC * (q[:, t].unsqueeze(-2) @ (st[t]["W_post"] + st[t]["Z_post"])).squeeze(-2)
        assert not torch.allclose(out[:, t], nxt)


def test_geometric_direction_ignores_current_key_inclusion():
    g = torch.Generator().manual_seed(11)
    M = torch.randn(DK, DK, generator=g, dtype=F64)
    C = M @ M.T + 0.1 * torch.eye(DK, dtype=F64)
    kk = torch.randn(DK, generator=g, dtype=F64)
    d1 = torch.linalg.solve(C, kk); d1 = d1 / (kk @ d1)
    d2 = torch.linalg.solve(C + torch.outer(kk, kk), kk); d2 = d2 / (kk @ d2)
    assert maxrel(d1, d2) < 1e-12


# ------------------------------------------------------------ gradients
@pytest.mark.parametrize("arm", ["joint", "rls", "geometric"])
def test_gradcheck_through_covariance(arm):
    q, k, v = data(12, T_=5, dk=3, dv=2, B_=1, H_=1)
    q, k, v = (x.clone().requires_grad_(True) for x in (q, k, v))
    fn = {"joint": lambda q, k, v: joint_writer(q, k, v, 0.9, 1.0, 10.0, 2.0, 0.5),
          "rls": lambda q, k, v: rls_writer(q, k, v, 0.9, 1.0, 10.0, 0.5),
          "geometric": lambda q, k, v: geometric_writer(q, k, v, 0.05, 1.0, 0.1, 0.9, 0.5)}[arm]
    assert torch.autograd.gradcheck(fn, (q, k, v), eps=1e-6, atol=1e-7)


def test_covariance_is_differentiated_not_detached():
    """dL/dk has a covariance path: removing it (P computed from detached
    keys) changes the gradient."""
    q, k, v = data(13, T_=6)
    k1 = k.clone().requires_grad_(True)
    joint_writer(q, k1, v, 0.95, 1.0, 10.0, 2.0, SC).sum().backward()

    def detached_P(q, k, v):
        _, st = joint_writer(q, k.detach(), v, 0.95, 1.0, 10.0, 2.0, SC, return_states=True)
        return st

    st = detached_P(q, k, v)
    k2 = k.clone().requires_grad_(True)
    out = []
    S = torch.zeros(B, H, DK, DV, dtype=F64)
    Sz = torch.zeros_like(S)
    for t in range(k.shape[1]):
        Sh, Szh = (S, Sz) if t == 0 else (S + Sz, Sz)
        b = (st[t]["P_prior"][..., :, :DK] @ k2[:, t].unsqueeze(-1)).squeeze(-1)
        s = 1.0 + (k2[:, t] * b[..., :DK]).sum(-1)
        ac = b / s.unsqueeze(-1)
        r = v[:, t] - (k2[:, t].unsqueeze(-2) @ Sh).squeeze(-2)
        S = Sh + ac[..., :DK, None] * r[..., None, :]
        Sz = Szh + ac[..., DK:, None] * r[..., None, :]
        out.append(SC * (q[:, t].unsqueeze(-2) @ S).squeeze(-2))
    torch.stack(out, 1).sum().backward()
    assert (k1.grad - k2.grad).abs().max() > 1e-6


# ------------------------------------------------- symmetry and definiteness
def _trajectories(dtype, T_, dk):
    g = torch.Generator().manual_seed(14)
    rnd = torch.nn.functional.normalize(torch.randn(1, T_, 1, dk, generator=g, dtype=dtype), dim=-1)
    base = torch.nn.functional.normalize(torch.randn(4, dk, generator=g, dtype=dtype), dim=-1)
    cyc = base[torch.arange(T_) % 4].view(1, T_, 1, dk)          # repeated keys
    v = torch.randn(1, T_, 1, 2 * dk, generator=g, dtype=dtype)
    return {"random": rnd, "repeated": cyc}, v


@pytest.mark.parametrize("dtype,T_,dk", [(F64, 300, 8), (torch.float32, 52, 32)])
@pytest.mark.parametrize("rho", [1.0, 0.99, 0.95])
@pytest.mark.parametrize("T0", [1.0, 4.0, 16.0])
def test_joint_covariance_symmetric_positive_definite(dtype, T_, dk, rho, T0):
    keys, v = _trajectories(dtype, T_, dk)
    for name, k in keys.items():
        _, st = joint_writer(k, k, v, rho, 1.0, 10.0, T0, dk ** -0.5, return_states=True)
        for t, s in enumerate(st):
            P = s["P"][0, 0]
            assert torch.equal(P, P.transpose(-1, -2)), (name, t)
            ev = torch.linalg.eigvalsh(P.double()).min().item()
            assert ev > 0, (name, t, ev)
            assert s["s"].min().item() >= 1.0


@pytest.mark.parametrize("dtype,T_,dk", [(F64, 300, 8), (torch.float32, 52, 32)])
@pytest.mark.parametrize("rho", [1.0, 0.99, 0.95])
def test_rls_and_geometric_matrices_symmetric_positive_definite(dtype, T_, dk, rho):
    keys, v = _trajectories(dtype, T_, dk)
    for name, k in keys.items():
        _, st = rls_writer(k, k, v, rho, 1.0, 10.0, dk ** -0.5, return_states=True)
        for t, s in enumerate(st):
            P = s["P"][0, 0]
            assert torch.equal(P, P.transpose(-1, -2)), (name, t)
            assert torch.linalg.eigvalsh(P.double()).min().item() > 0, (name, t)
        _, st = geometric_writer(k, k, v, 0.05, 1.0, 0.1, rho, dk ** -0.5, return_states=True)
        for t, s in enumerate(st):
            Ci = s["Cinv"][0, 0]
            assert torch.equal(Ci, Ci.transpose(-1, -2)), (name, t)
            assert torch.linalg.eigvalsh(Ci.double()).min().item() > 0, (name, t)


def test_all_writer_states_reset_between_calls():
    """Whole-sequence: a second call on the same data gives the same output
    (no state leaks), and sequences in a batch are independent."""
    q, k, v = data(15)
    for fn in (lambda q, k, v: joint_writer(q, k, v, 0.95, 1.0, 10.0, 2.0, SC),
               lambda q, k, v: rls_writer(q, k, v, 0.95, 1.0, 10.0, SC),
               lambda q, k, v: geometric_writer(q, k, v, 0.05, 1.0, 0.1, 1.0, SC)):
        a = fn(q, k, v)
        assert torch.equal(a, fn(q, k, v))
        solo = fn(q[:1], k[:1], v[:1])
        assert maxrel(a[:1], solo) < 1e-14


def test_unit_conventions():
    assert REFERENCE_CFG["joint"]["sigma2"] == 1.0 and REFERENCE_CFG["joint"]["p0"] == 10.0
    assert math.isclose(REFERENCE_CFG["rls"]["p0"], 10.0)
