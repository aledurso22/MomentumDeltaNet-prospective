"""The six arms as recurrent references, line-for-line against the official one.

Mirrors `fla/ops/momentum_delta_rule/naive.py::recurrent_momentum_delta_rule_ref`
at pinned commit c6e77fa261fb0c002fae1a14b6209a5b28d2edc9. Native lines, in
their orientation (`S` is K x V, and `alpha` is folded into `p = alpha Norm(k)`):

    w_t  = p_t S_(t-1) - v_t                  # write residual
    R_t  = k_t (x) w_t                        # rank-1, WITHOUT eta
    M_t  = mu_t M_(t-1) + eta_t R_t
    S_t  = alpha_t S_(t-1) - beta_t M_t

Each arm changes exactly ONE of these lines:

    tss / generalized   line 3: eta_t R_t  ->  eta_t y_t, y the filter state
    qhm                 line 4: S_t = alpha_t S_(t-1)
                                      - beta_t [nu M_t + (1-nu) eta_t R_t]
    nesterov            line 1: residual evaluated at the lookahead state
                                L_t = alpha_t S_(t-1) - beta_t mu_t M_(t-1)

`tss` and `generalized` are the SAME code path with different parameters; see
`prospective/coefficients.py` for the provenance of the coefficient map.
"""

import torch

from .coefficients import H_TOKEN, prospective_coefficients

ARMS = ("native", "tss", "generalized", "qhm", "nesterov")
FILTER_ARMS = ("tss", "generalized")


def recurrent_ref(arm, q, k, v, log_alpha, log_mu, p, beta, eta, scale=None,
                  mass=None, gamma=None, response=None, nu=None,
                  h=H_TOKEN, initial_S=None, initial_M=None,
                  output_final_state=False):
    if arm not in ARMS:
        raise ValueError(f"{arm!r} is not an arm of this module")
    q, k, v, p, log_alpha, log_mu, beta, eta = (
        x.to(torch.float32) if x.dtype != torch.float64 else x
        for x in (q, k, v, p, log_alpha, log_mu, beta, eta))
    B, T, H, DK = k.shape
    DV = v.shape[-1]
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    if arm in FILTER_ARMS:
        a, b, c, d = prospective_coefficients(mass, gamma, response, h)
        a, b, c, d = (x.to(v.dtype).view(1, -1, 1, 1) if x.dim() else
                      x.to(v.dtype) for x in (a, b, c, d))
    if arm == "qhm":
        nu = nu.to(v.dtype) if torch.is_tensor(nu) else nu

    S_prev = v.new_zeros(B, H, DK, DV) if initial_S is None else initial_S
    M_prev = v.new_zeros(B, H, DK, DV) if initial_M is None else initial_M
    y_prev = v.new_zeros(B, H, DK, DV)
    y_prev2 = v.new_zeros(B, H, DK, DV)
    R_prev = v.new_zeros(B, H, DK, DV)

    out = torch.zeros_like(v)
    for i in range(T):
        k_t, q_t, v_t, p_t = k[:, i], q[:, i], v[:, i], p[:, i]
        mu_i = log_mu[:, i].exp().view(B, H, 1, 1)
        beta_i = beta[:, i].view(B, H, 1, 1)
        alpha_i = log_alpha[:, i].exp().view(B, H, 1, 1)
        eta_i = eta[:, i].view(B, H, 1, 1)

        if arm == "nesterov":
            # residual at L_t = alpha_t S_(t-1) - beta_t mu_t M_(t-1); alpha is
            # already inside p_t, so the correction carries p_t / alpha_t
            n_t = (p_t / alpha_i.view(B, H, 1)).unsqueeze(-2)
            w_t = -(v_t.unsqueeze(-2) - p_t.unsqueeze(-2) @ S_prev
                    + (beta_i * mu_i) * (n_t @ M_prev))
        else:
            w_t = -(v_t.unsqueeze(-2) - p_t.unsqueeze(-2) @ S_prev)

        R_t = k_t.unsqueeze(-1) @ w_t              # rank-1, no eta

        if arm in FILTER_ARMS:
            y_t = a * y_prev - b * y_prev2 + c * R_t - d * R_prev
            y_prev2, y_prev = y_prev, y_t
            M_t = mu_i * M_prev + eta_i * y_t
        else:
            # byte-for-byte the official expression, including the order in
            # which eta enters the outer product
            M_t = mu_i * M_prev + (eta[:, i].unsqueeze(-1) * k_t).unsqueeze(-1) @ w_t

        if arm == "qhm":
            S_t = alpha_i * S_prev - beta_i * (nu * M_t
                                               + (1.0 - nu) * eta_i * R_t)
        else:
            S_t = alpha_i * S_prev - beta_i * M_t

        out[:, i] = (q_t.unsqueeze(-1) * S_t).sum(-2)
        R_prev, M_prev, S_prev = R_t, M_t, S_t

    final = torch.stack([S_prev, M_prev], dim=0) if output_final_state else None
    return out, final
