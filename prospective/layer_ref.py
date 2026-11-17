"""A pure-PyTorch functional re-implementation of `ProspectiveMDN.forward`,
in any dtype, from the layer's own parameters. No Triton, no fla kernels.

Used to build the FP64 reference that both the FP32 `reference` backend and
the FP32 `official_chunk` backend are measured against: outputs, input
gradients and every parameter gradient. It re-implements, independently of
the Triton code it is checked against,

* `ShortConvolution` (causal depthwise conv, kernel W, then SiLU):
  y_t = silu(sum_i w[i] x_{t-W+1+i}), i.e. F.conv1d with W-1 left padding;
* `FusedRMSNormGated(activation='sigmoid')`:
  y = x * rsqrt(mean(x^2) + eps) * weight * sigmoid(g);

and takes the gates, the output correction and the recurrence from the same
formulas as the official forward (`mixer.py`) and `reference.py`.
"""

import torch
import torch.nn.functional as F

from .reference import reference_recurrence


def params_as(m, dtype):
    """Detached leaf copies of every parameter of `m`, in `dtype`."""
    return {n: p.detach().to(dtype).clone().requires_grad_(p.requires_grad)
            for n, p in m.named_parameters()}


def _lin(P, name, x):
    return x @ P[name + ".weight"].transpose(0, 1)


def _conv_silu(P, name, x):
    w = P[name + ".weight"]                       # [D, 1, W]
    T, W = x.shape[1], w.shape[-1]
    y = F.conv1d(x.transpose(1, 2), w, P.get(name + ".bias"),
                 padding=W - 1, groups=w.shape[0])[..., :T]
    return F.silu(y.transpose(1, 2))


def mixer_forward_ref(m, h, P):
    """`m` supplies configuration (arm, dims, eps, tau); `P` the parameters."""
    B, T, _ = h.shape
    H, dk, dv = m.num_heads, m.head_k_dim, m.head_v_dim
    assert m.use_short_conv and m.use_gate and m.use_output_correction
    q = _conv_silu(P, "q_conv1d", _lin(P, "q_proj", h)).view(B, T, H, dk)
    k = _conv_silu(P, "k_conv1d", _lin(P, "k_proj", h)).view(B, T, H, dk)
    v = _conv_silu(P, "v_conv1d", _lin(P, "v_proj", h)).view(B, T, H, dv)

    a = _lin(P, "a_proj", h)
    b = _lin(P, "b_proj", h)
    mm = _lin(P, "m_proj", h)
    e = _lin(P, "e_proj", h) / m.tau
    log_mu = -P["Mu_log"].exp() * F.softplus(mm + P["mu_bias"])
    log_alpha = -P["A_log"].exp() * F.softplus(a + P["dt_bias"])
    eta = torch.tanh(e) + 1
    beta = torch.sigmoid(b)
    theta = torch.arctan(eta * P["log_factor"].exp())
    beta = torch.sin(theta) ** 2 * beta
    log_alpha = (torch.cos(theta) ** 2).log() + log_alpha
    if m.min_log_mu is not None:
        log_mu = log_mu.clamp_min(m.min_log_mu)
    q = q - P["D_log"].exp()[None, None, :, None] * k

    f = dict(q=q, k=k, v=v, log_alpha=log_alpha, log_mu=log_mu, beta=beta,
             eta=eta)
    leaves = {}
    for name, key in (("fil_M", "mass"), ("fil_gamma", "gamma"),
                      ("fil_T", "response"), ("nu", "nu"),
                      ("inertia_A", "A"), ("inertia_r", "r")):
        if name in P:
            leaves[key] = P[name]
    o = reference_recurrence(m.arm, f, leaves, getattr(m, "writer_cfg", None))

    g = _lin(P, "g_proj", h).view(B, T, H, dv)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + m.o_norm.eps)
    o = o * P["o_norm.weight"] * torch.sigmoid(g)
    return _lin(P, "o_proj", o.reshape(B, T, H * dv))
