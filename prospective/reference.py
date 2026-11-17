"""The `reference` recurrent backend, free of fla/Triton imports.

Given the official frontend's outputs (q and k BEFORE l2 normalisation, after
the output correction), reproduce what `chunk_mode_rule(...,
use_qk_l2norm_in_kernel=True, use_p_times_alpha=True)` does before its
recurrence, then run the declared rule from `rules.py`.
"""

import torch

from .coefficients import H_TOKEN
from .projection import PROJECTION_ARMS, run_projection
from .rules import FILTER_ARMS, recurrent_ref
from .writers import WRITER_ARMS, run_writer

#: `fla.modules.l2norm.l2norm_fwd` default, used in-kernel by chunk_mode_rule
L2_EPS = 1e-6


def l2norm_ref(x, eps=L2_EPS):
    """The Triton l2norm_fwd: y = x * 1/sqrt(sum(x^2) + eps), last dim."""
    return x * (1.0 / torch.sqrt((x * x).sum(-1, keepdim=True) + eps))


def reference_recurrence(arm, f, leaves, writer_cfg=None):
    """The `reference` backend on frontend outputs `f`, in f's dtype (float64
    stays float64, so the same code gives the high-precision truth).

    Reproduces chunk_mode_rule's in-kernel preprocessing: q, k <- l2norm;
    p = alpha * l2norm(k); k is scaled by eta inside the rule; scale K^-1/2."""
    q, k = l2norm_ref(f["q"]), l2norm_ref(f["k"])
    if arm in WRITER_ARMS:
        # the writers use the normalised UNSCALED key and none of the MDN
        # gates (alpha, beta, mu, eta); same dtype rule as recurrent_ref
        q, k, v = (x if x.dtype == torch.float64 else x.to(torch.float32)
                   for x in (q, k, f["v"]))
        return run_writer(arm, q, k, v, writer_cfg, scale=k.shape[-1] ** -0.5)
    if arm in PROJECTION_ARMS:
        # same inputs as the writers: normalised unscaled k, no MDN gates
        q, k, v = (x if x.dtype == torch.float64 else x.to(torch.float32)
                   for x in (q, k, f["v"]))
        return run_projection(arm, q, k, v, leaves, scale=k.shape[-1] ** -0.5)
    p = (k * f["log_alpha"].exp().unsqueeze(-1)).to(q.dtype)
    extra = dict(leaves)
    if arm in FILTER_ARMS:
        extra["h"] = H_TOKEN
    if arm == "nesterov":
        extra["n"] = k
    o, _ = recurrent_ref(arm, q=q, k=k, v=f["v"], log_alpha=f["log_alpha"],
                         log_mu=f["log_mu"], p=p, beta=f["beta"],
                         eta=f["eta"], scale=k.shape[-1] ** -0.5, **extra)
    return o
