"""Covariance and inertial fast-memory writers, as recurrent backends.

Three writers for the v3 comparison. Each replaces the WHOLE MDN memory
update: no native momentum, retention or write-strength gate is applied
afterwards. Each takes the normalised query q, the normalised UNSCALED key k
(never MDN's auxiliary p = alpha k) and the value v from the shared official
frontend, and reads its current posterior with the same readout convention
as the MDN arms, o_t = K^-1/2 * q_t^T S_t.

Orientation. The derivations use W, Z in R^{dv x dk} with a local prediction
W k. This repository stores memory as S in R^{dk x dv} with o = q^T S (see
`naive.py`), so here S = W^T and Sz = Z^T throughout:

    W k      ->  S^T k  (computed as k^T S)
    r a^T    ->  a r^T
    W q      ->  (q^T S)^T

All three are whole-sequence only: they start from their declared initial
state at t = 1 and return no final state.

----------------------------------------------------------------------------
joint   constant-velocity Kalman / trajectory Gauss-Newton memory-motion writer

    first prior (DECLARED: the initial values ARE the t = 1 prior; no
    prediction step is applied before the first observation)
        W_1^- = 0,  Z_1^- = 0,  P_1^- = P_0 = diag(p0 I, p0 / T0^2 I)
    prediction, t >= 2, from the previous posterior
        W_t^- = W_{t-1} + Z_{t-1},  Z_t^- = Z_{t-1},  P_t^- = F P_{t-1} F^T / rho
        F = [[I, I], [0, I]]
    correction with h_t = [k_t; 0]
        b = P^- h,  s = sigma2 + h^T b,  [a; c] = b / s,  r = v - W^- k
        W_t = W^- + r a^T,  Z_t = Z^- + r c^T,  P_t = P^- - b b^T / s
    read: o_t = W_t q_t (current posterior)

P is ONE (2dk x 2dk) covariance per sequence and head, shared by all dv rows
of [W Z]. That factorisation is exact here: the prediction W k is affine in
each row, every row has the same prior and the same transition, and the
observation noise is isotropic (sigma2 I over the dv outputs). The full
Kalman filter over vec([W Z]) then has covariance P (x) I_dv at every step.
That is tested against the explicit global filter. rho discounts information
(it inflates P); it does not multiply the stored motion Z.

----------------------------------------------------------------------------
rls     state-only full key-covariance writer (exponentially weighted RLS)

The same observation weighting and information-forgetting convention, with
transition I, memory W only and a (dk x dk) covariance:

    W_1^- = 0,  P_1^- = p0 I                      (first prior, as above)
    W_t^- = W_{t-1},  P_t^- = P_{t-1} / rho       (t >= 2)
    b = P^- k,  s = sigma2 + k^T b,  W_t = W^- + r (b/s)^T,  P_t = P^- - b b^T / s

Full key correlations, not diagonal uncertainty. This is the classical RLS /
Kalman memory written as a control; it is NOT a reproduction of the
published KDN or MesaNet architectures.

----------------------------------------------------------------------------
geometric   the original fixed-gain geometric inertial writer

Source: the `full` arm of `prospective_rtrl_experiment/diagnostics/
moving_memory_tracker.py`: V+ = V + A R, W+ = W + V+ + G R, with
R = (y - W k) d^T and d = C^-1 k / (k^T C^-1 k). In posterior variables:

    xi_t    = (u_t k_t - v_t) d_t^T
    W_t^+   = u_t - G xi_t            (corrected memory; READ HERE)
    Z_t^+   = Z_{t-1} - A xi_t        (motion)
    u_{t+1} = W_t^+ + Z_t^+           (next prediction)
    u_1 = 0, Z_0 = 0

Timing in this layer: the tracker READ u_n before its update (read, then
reveal y_n). The layer is read-after-update, so the read is W_t^+ q_t, not
u_{t+1} q_t. Eliminating W^+ and Z^+ recovers the tracker recurrence
u_{t+1} = 2 u_t - u_{t-1} - A xi_t - G (xi_t - xi_{t-1}); tested.

C_t. The tracker used a FIXED C = 0.1 I + K K^T over a public dictionary of
all addresses. A trained layer has no such dictionary, so the causal
analogue used here is the accumulated key information

    C_t = lam C_{t-1} + k_t k_t^T,   C_0 = c0 I,   reference c0 = 0.1, lam = 1

maintained as C^-1 by Sherman-Morrison rank-one updates. Including the
current key does not change d_t: (C + k k^T)^-1 k is parallel to C^-1 k.
Reference gains A = 0.05, G = 1 (the documented reference point). lam < 1
(information forgetting) and any other (A, G, c0) are LATER TUNING and are
labelled as such in the experiment configuration.
"""

import torch

WRITER_ARMS = ("rls", "geometric", "joint")

#: documented reference settings. joint/rls: sigma2 = 1, p0 = 10;
#: joint T0 and rho, rls rho: selected on development data (see
#: experiment_v3.py). geometric: the tracker's reference point.
REFERENCE_CFG = {
    "joint": dict(rho=1.0, sigma2=1.0, p0=10.0, T0=4.0),
    "rls": dict(rho=1.0, sigma2=1.0, p0=10.0),
    "geometric": dict(A=0.05, G=1.0, c0=0.1, lam=1.0),
}


def _eye(n, like):
    return torch.eye(n, dtype=like.dtype, device=like.device)


def joint_writer(q, k, v, rho, sigma2, p0, T0, scale, return_states=False):
    """q, k: [B, T, H, dk] (already l2-normalised); v: [B, T, H, dv]."""
    B, T, H, K = k.shape
    S = v.new_zeros(B, H, K, v.shape[-1])
    Sz = torch.zeros_like(S)
    I = _eye(K, v)
    P0 = torch.block_diag(p0 * I, (p0 / (T0 * T0)) * I).expand(B, H, 2 * K, 2 * K)
    out = torch.empty_like(v)
    states = []
    P = None
    for t in range(T):
        if t == 0:
            Pm, Sh, Szh = P0, S, Sz
        else:
            Sh, Szh = S + Sz, Sz
            A_, Bm, D = P[..., :K, :K], P[..., :K, K:], P[..., K:, K:]
            # F P F^T in an order that keeps P bitwise symmetric
            tl = (A_ + D) + (Bm + Bm.transpose(-1, -2))
            tr = Bm + D
            Pm = torch.cat([torch.cat([tl, tr], -1),
                            torch.cat([tr.transpose(-1, -2), D], -1)], -2) / rho
        kt = k[:, t]
        b = (Pm[..., :, :K] @ kt.unsqueeze(-1)).squeeze(-1)          # [B,H,2K]
        s = sigma2 + (kt * b[..., :K]).sum(-1)                      # [B,H]
        ac = b / s.unsqueeze(-1)
        r = v[:, t] - (kt.unsqueeze(-2) @ Sh).squeeze(-2)            # [B,H,dv]
        S = Sh + ac[..., :K, None] * r[..., None, :]
        Sz = Szh + ac[..., K:, None] * r[..., None, :]
        P = Pm - (b[..., :, None] * b[..., None, :]) / s[..., None, None]
        out[:, t] = scale * (q[:, t].unsqueeze(-2) @ S).squeeze(-2)
        if return_states:
            states.append(dict(S_prior=Sh, Sz_prior=Szh, P_prior=Pm, S=S,
                               Sz=Sz, P=P, s=s))
    return (out, states) if return_states else out


def rls_writer(q, k, v, rho, sigma2, p0, scale, return_states=False):
    B, T, H, K = k.shape
    S = v.new_zeros(B, H, K, v.shape[-1])
    P0 = (p0 * _eye(K, v)).expand(B, H, K, K)
    out = torch.empty_like(v)
    states = []
    P = None
    for t in range(T):
        Pm = P0 if t == 0 else P / rho
        kt = k[:, t]
        b = (Pm @ kt.unsqueeze(-1)).squeeze(-1)
        s = sigma2 + (kt * b).sum(-1)
        r = v[:, t] - (kt.unsqueeze(-2) @ S).squeeze(-2)
        S = S + (b / s.unsqueeze(-1))[..., :, None] * r[..., None, :]
        P = Pm - (b[..., :, None] * b[..., None, :]) / s[..., None, None]
        out[:, t] = scale * (q[:, t].unsqueeze(-2) @ S).squeeze(-2)
        if return_states:
            states.append(dict(P_prior=Pm, S=S, P=P, s=s))
    return (out, states) if return_states else out


def geometric_writer(q, k, v, A, G, c0, lam, scale, return_states=False):
    B, T, H, K = k.shape
    U = v.new_zeros(B, H, K, v.shape[-1])      # u_t (prediction), u_1 = 0
    Sz = torch.zeros_like(U)                    # Z_{t-1}, Z_0 = 0
    Ci = ((1.0 / c0) * _eye(K, v)).expand(B, H, K, K)
    out = torch.empty_like(v)
    states = []
    for t in range(T):
        kt = k[:, t]
        # C_t = lam C_{t-1} + k k^T, as C^-1 (Sherman-Morrison)
        Ci = Ci / lam
        x = (Ci @ kt.unsqueeze(-1)).squeeze(-1)
        Ci = Ci - (x[..., :, None] * x[..., None, :]) / (1.0 + (kt * x).sum(-1))[..., None, None]
        y = (Ci @ kt.unsqueeze(-1)).squeeze(-1)
        d = y / (kt * y).sum(-1, keepdim=True)
        e = (kt.unsqueeze(-2) @ U).squeeze(-2) - v[:, t]            # (u k - v)^T
        xi = d[..., :, None] * e[..., None, :]                      # xi^T, [K, dv]
        Wp = U - G * xi
        Sz = Sz - A * xi
        out[:, t] = scale * (q[:, t].unsqueeze(-2) @ Wp).squeeze(-2)
        if return_states:
            states.append(dict(u=U, xi=xi, W_post=Wp, Z_post=Sz, Cinv=Ci, d=d))
        U = Wp + Sz
    return (out, states) if return_states else out


def run_writer(arm, q, k, v, cfg, scale):
    c = dict(REFERENCE_CFG[arm], **(cfg or {}))
    if arm == "joint":
        return joint_writer(q, k, v, c["rho"], c["sigma2"], c["p0"], c["T0"], scale)
    if arm == "rls":
        return rls_writer(q, k, v, c["rho"], c["sigma2"], c["p0"], scale)
    if arm == "geometric":
        return geometric_writer(q, k, v, c["A"], c["G"], c["c0"], c["lam"], scale)
    raise ValueError(arm)


def state_numbers(arm, dk, dv):
    """Persistent recurrent-state numbers per sequence, head and layer
    (whole-sequence execution; what a cache would have to carry)."""
    return {"gated": dk * dv, "native": 2 * dk * dv, "nesterov": 2 * dk * dv,
            "qhm": 2 * dk * dv, "tss": 5 * dk * dv, "generalized": 5 * dk * dv,
            "rls": dk * dv + dk * dk, "geometric": 2 * dk * dv + dk * dk,
            "joint": 2 * dk * dv + (2 * dk) ** 2}[arm]
