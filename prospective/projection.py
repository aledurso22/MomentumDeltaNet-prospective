"""Normalised projection writers for the geometry/inertia training pilot.

Three recurrent backends. Each replaces the WHOLE MDN memory update (no MDN
gate alpha, beta, mu, eta is applied). Each takes the shared official
frontend's l2-normalised query q, l2-normalised UNSCALED key k and value v,
and reads the corrected memory with the official output scaling,
o_t = K^-1/2 * q_t^T S_t.

Orientation. Mathematically W in R^{dv x dk} with prediction W k. This
repository stores S = W^T in R^{dk x dv} (o = q^T S), so W k = k^T S and
r b^T -> b r^T.

    euclidean_projection      b_t = k_t / (k_t^T k_t)
    geometric_projection      C_0 = c0 I,  C_t = C_{t-1} + k_t k_t^T,
                              b_t = C_t^-1 k_t / (k_t^T C_t^-1 k_t)
    both (no motion)          W_t = W_{t-1} + (v_t - W_{t-1} k_t) b_t^T

    geometric_motion_learned  the same geometric b_t, G = 1, and
                              U_t = W_{t-1} + V_{t-1}
                              D_t = (v_t - U_t k_t) b_t^T
                              W_t = U_t + D_t
                              V_t = (1 - d) V_{t-1} + (A - d) D_t,   d = r A

c0 = .03 and no forgetting (lam = 1; lam is forgetting, not the ridge c0).
C^-1 is maintained incrementally (Sherman-Morrison) and differentiated
through. W_0 = V_0 = 0, C_0^-1 = I / c0 at the start of every sequence
(whole-sequence only; nothing is carried across calls).

A and r are OUTER-TRAINED parameters, one per head and layer, constant over
the tokens of a forward pass (not input-dependent gates). They are stored
directly, projected onto [0, 1] after every optimiser step and excluded from
weight decay. The family contains zero drag (r = 0) and two no-motion
boundaries: A = 0 (then d = 0 and V stays 0) and r = 1 (then A - d = 0 and V
stays 0). The [0, 1] box is the experimental parameter domain, not a
stability theorem for the network.

Zero keys. If k_t^T x_t = 0 (only for k_t = 0: C^-1 is positive definite)
the correction direction is defined as b_t = 0, i.e. no write; the
Sherman-Morrison step is already the identity there. The guard is exact
(den == 0), so every nonzero key gets the exact projection, however small.

The no-motion writers carry no velocity matrix: the euclidean writer's
persistent state is W, the geometric one's W and C^-1, the motion arm's W, V
and C^-1 (`state_numbers`).
"""

import torch

PROJECTION_ARMS = ("euclidean_projection", "geometric_projection",
                   "geometric_motion_learned")
GEOMETRIC_ARMS = ("geometric_projection", "geometric_motion_learned")
MOTION_ARMS = ("geometric_motion_learned",)

#: C_0 = C0 I (the frozen local screens' lambda = .03)
C0 = 0.03
#: the tested modest-drag setting: A = .003125, d = A / 2, i.e. r = .5
INERTIA_A0, INERTIA_R0 = 0.003125, 0.5
INERTIA_DOMAIN = (0.0, 1.0)
#: parameter names (per head) of the learned inertial coefficients
INERTIA_LEAVES = ("inertia_A", "inertia_r")


def normalized_direction(x, k):
    """b = x / (k^T x), and b = 0 where k^T x == 0 (zero key: no write)."""
    den = (k * x).sum(-1, keepdim=True)
    ok = den != 0
    return torch.where(ok, x / torch.where(ok, den, torch.ones_like(den)),
                       torch.zeros_like(x))


def projection_writer(q, k, v, geometry, scale, c0=C0, A=None, r=None,
                      return_states=False):
    """q, k: [B, T, H, dk] (l2-normalised); v: [B, T, H, dv].
    geometry: 'euclidean' or 'geometric'. A, r: per-head tensors [H] for the
    inertial recurrence, or both None for no motion (no V is allocated).
    return_states: also return, per token, the CARRIED state only."""
    if geometry not in ("euclidean", "geometric"):
        raise ValueError(geometry)
    motion = A is not None
    if motion != (r is not None):
        raise ValueError("A and r go together")
    B, T, H, K = k.shape
    S = v.new_zeros(B, H, K, v.shape[-1])
    if motion:
        A = A.to(v.dtype)
        d = r.to(v.dtype) * A
        cV = (1.0 - d)[None, :, None, None]
        cD = (A - d)[None, :, None, None]
        Vm = torch.zeros_like(S)
    if geometry == "geometric":
        Ci = ((1.0 / c0) * torch.eye(K, dtype=v.dtype, device=v.device)
              ).expand(B, H, K, K)
    out = torch.empty_like(v)
    states = []
    for t in range(T):
        kt = k[:, t]
        if geometry == "geometric":
            ck = (Ci @ kt.unsqueeze(-1)).squeeze(-1)
            Ci = Ci - (ck[..., :, None] * ck[..., None, :]) / (
                1.0 + (kt * ck).sum(-1))[..., None, None]
            x = (Ci @ kt.unsqueeze(-1)).squeeze(-1)
        else:
            x = kt
        b = normalized_direction(x, kt)                          # [B,H,K]
        U = S + Vm if motion else S
        res = v[:, t] - (kt.unsqueeze(-2) @ U).squeeze(-2)       # [B,H,dv]
        D = b[..., :, None] * res[..., None, :]                  # [B,H,K,dv]
        S = U + D
        if motion:
            Vm = cV * Vm + cD * D
        out[:, t] = scale * (q[:, t].unsqueeze(-2) @ S).squeeze(-2)
        if return_states:
            st = dict(S=S)
            if geometry == "geometric":
                st["Cinv"] = Ci
            if motion:
                st["V"] = Vm
            states.append(st)
    return (out, states) if return_states else out


def run_projection(arm, q, k, v, leaves, scale, c0=C0, return_states=False):
    if arm == "euclidean_projection":
        return projection_writer(q, k, v, "euclidean", scale,
                                 return_states=return_states)
    if arm == "geometric_projection":
        return projection_writer(q, k, v, "geometric", scale, c0=c0,
                                 return_states=return_states)
    if arm == "geometric_motion_learned":
        return projection_writer(q, k, v, "geometric", scale, c0=c0,
                                 A=leaves["A"], r=leaves["r"],
                                 return_states=return_states)
    raise ValueError(arm)


def state_numbers(arm, dk, dv):
    """Persistent recurrent-state numbers per sequence, head and layer, for
    every arm of this repository (whole-sequence execution: what a cache
    would have to carry). The MDN native rule carries S and its momentum."""
    own = {"euclidean_projection": dk * dv,
           "geometric_projection": dk * dv + dk * dk,
           "geometric_motion_learned": 2 * dk * dv + dk * dk}
    if arm in own:
        return own[arm]
    from .writers import state_numbers as older
    return older(arm, dk, dv)


def coefficients(A, r):
    """A, d = r A and A - d, as plain lists (one entry per head)."""
    A, r = A.detach().double().cpu(), r.detach().double().cpu()
    d = r * A
    return dict(A=A.tolist(), r=r.tolist(), d=d.tolist(),
                A_minus_d=(A - d).tolist())
