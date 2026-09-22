"""A small MDN language model whose token mixer is one of the six arms.

Deliberately minimal and tier-A (see section 5 of the bridge plan): the mixer
is the differentiable token-loop reference, so autograd supplies the backward
that `fused_recurrent` does not have. Everything outside the mixer is shared
across arms and initialized from the same seed, so the ONLY difference between
two arms is the write rule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coefficients import (GEN_GAMMA0, GEN_M0, GEN_T0, H_TOKEN, TSS_T0,
                           project_leaves)
from .rules import ARMS, FILTER_ARMS, recurrent_ref

#: QHM's declared start: nu = 1 is the native function exactly. Stored
#: directly and clamped to [0, 1] after each step, as the ladder did -- a
#: sigmoid start at nu = 1 has derivative ~3e-4 and cannot move.
QHM_NU0, QHM_DOMAIN = 1.0, (0.0, 1.0)


class Mixer(nn.Module):
    def __init__(self, arm, d_model, n_heads):
        super().__init__()
        self.arm, self.n_heads = arm, n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.gates = nn.Linear(d_model, 4 * n_heads, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=False)
        if arm in FILTER_ARMS:
            # stored DIRECTLY, projected after each step; no softplus
            gen = arm == "generalized"
            self.fil_T = nn.Parameter(torch.full((n_heads,),
                                                 GEN_T0 if gen else TSS_T0))
            self.fil_M = nn.Parameter(torch.full((n_heads,),
                                                 GEN_M0 if gen else 0.0),
                                      requires_grad=gen)
            self.fil_gamma = nn.Parameter(torch.full((n_heads,),
                                                     GEN_GAMMA0 if gen else 0.0),
                                          requires_grad=gen)
        if arm == "qhm":
            self.nu = nn.Parameter(torch.full((), QHM_NU0))

    def forward(self, h):
        B, T, D = h.shape
        H, dk = self.n_heads, self.dk
        shape = lambda x: x.view(B, T, H, dk)
        q, k, v = shape(self.q(h)), shape(self.k(h)), shape(self.v(h))
        k = F.normalize(k, dim=-1)
        g = self.gates(h).view(B, T, H, 4)
        log_alpha = -F.softplus(g[..., 0])
        log_mu = -F.softplus(g[..., 1])
        beta = torch.sigmoid(g[..., 2])
        eta = torch.sigmoid(g[..., 3])
        p = log_alpha.exp().unsqueeze(-1) * k
        extra = {}
        if self.arm in FILTER_ARMS:
            extra = dict(mass=self.fil_M, gamma=self.fil_gamma,
                         response=self.fil_T, h=H_TOKEN)
        if self.arm == "qhm":
            extra = dict(nu=self.nu)
        o, _ = recurrent_ref(self.arm, q=q, k=k, v=v, log_alpha=log_alpha,
                             log_mu=log_mu, p=p, beta=beta, eta=eta, **extra)
        return self.out(o.reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, arm, d_model, n_heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.mix = Mixer(arm, d_model, n_heads)
        self.mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                 nn.Linear(2 * d_model, d_model))

    def forward(self, h):
        h = h + self.mix(self.n1(h))
        return h + self.mlp(self.n2(h))


class Model(nn.Module):
    def __init__(self, arm, vocab, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(Block(arm, d_model, n_heads)
                                    for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=True)

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))

    @torch.no_grad()
    def project_(self):
        """Call after every optimizer step. Keeps every arm's own leaves in
        their admissible set; a no-op for arms without leaves."""
        for blk in self.blocks:
            m = blk.mix
            if hasattr(m, "fil_M"):
                project_leaves(m.fil_M.data, m.fil_gamma.data, m.fil_T.data,
                               h=H_TOKEN,
                               freeze_mass_gamma=m.arm != "generalized")
            if hasattr(m, "nu"):
                m.nu.data.clamp_(*QHM_DOMAIN)


def build(arm, vocab, seed, **kw):
    """Shared-backbone construction: the SAME seed gives every arm identical
    embeddings, projections, gates, MLPs and head. Only the arm's own extra
    leaves differ, and each starts at its native point."""
    torch.manual_seed(seed)
    return Model(arm, vocab, **kw)


def shared_parameter_check(a, b):
    """Every tensor present in both models must be bit-identical at init."""
    sa, sb = a.state_dict(), b.state_dict()
    common = set(sa) & set(sb)
    bad = [n for n in sorted(common) if not torch.equal(sa[n], sb[n])]
    return bad
