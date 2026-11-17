"""A small model whose token mixer is the official MDN layer with one of the
six recurrent rules plugged into its backend hook.

The mixer is `mixer.ProspectiveMDN`: the official layer's projections, causal
short convolutions, gate parameterisation and constraints, output correction,
gated RMS norm and output projection, around `rules.recurrent_ref`. The
surrounding backbone (embedding, pre-LN residual blocks with a GELU MLP,
final LayerNorm, head) is the v1 backbone, kept unchanged and shared.

Everything except the arm's own leaves is constructed from the same seed, so
the ONLY difference between two arms is the declared recurrent rule.
"""

import torch
import torch.nn as nn

from .coefficients import H_TOKEN, project_leaves
from .mixer import ALL_ARMS, QHM_DOMAIN, ProspectiveMDN
from .projection import INERTIA_DOMAIN, INERTIA_LEAVES, coefficients

#: production model, as specified for the v2 rerun
MODEL_DEFAULTS = dict(d_model=128, n_heads=4, head_dim=32, expand_v=2,
                      n_layers=2)


class Block(nn.Module):
    def __init__(self, arm, d_model, n_heads, head_dim, expand_v, backend,
                 layer_idx, writer_cfg=None):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.mix = ProspectiveMDN(arm, hidden_size=d_model, num_heads=n_heads,
                                  head_dim=head_dim, expand_v=expand_v,
                                  backend=backend, layer_idx=layer_idx,
                                  writer_cfg=writer_cfg)
        self.mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                 nn.Linear(2 * d_model, d_model))

    def forward(self, h):
        h = h + self.mix(self.n1(h))[0]
        return h + self.mlp(self.n2(h))


class Model(nn.Module):
    def __init__(self, arm, vocab, d_model=128, n_heads=4, head_dim=32,
                 expand_v=2, n_layers=2, backend="reference", writer_cfg=None):
        super().__init__()
        if arm not in ALL_ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(
            Block(arm, d_model, n_heads, head_dim, expand_v, backend, i,
                  writer_cfg)
            for i in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=True)
        # the official model-level init of the mixer's Linear/Conv1d weights,
        # applied in a fixed module order so it draws identically per arm
        for blk in self.blocks:
            blk.mix.official_model_init_()

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))

    def set_backend(self, backend):
        for blk in self.blocks:
            blk.mix.backend = backend

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
            if hasattr(m, "inertia_A"):
                m.inertia_A.data.clamp_(*INERTIA_DOMAIN)
                m.inertia_r.data.clamp_(*INERTIA_DOMAIN)

    def inertia(self):
        """Learned inertial coefficients per layer: A, r, d = r A, A - d (one
        entry per head). Empty for arms without them."""
        return {f"layer{i}": coefficients(blk.mix.inertia_A, blk.mix.inertia_r)
                for i, blk in enumerate(self.blocks)
                if hasattr(blk.mix, "inertia_A")}


ARM_LEAVES = ("fil_M", "fil_gamma", "fil_T", "nu") + INERTIA_LEAVES
#: leaves excluded from weight decay (the pilot's inertial coefficients only;
#: every other parameter keeps the existing uniform AdamW decay)
NO_DECAY_LEAVES = INERTIA_LEAVES


def param_groups(model, weight_decay):
    """The existing AdamW configuration, with NO_DECAY_LEAVES moved to a
    weight_decay = 0 group. Arms without such leaves get the single group
    they always had."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if n.split(".")[-1] in NO_DECAY_LEAVES else decay).append(p)
    groups = [dict(params=decay, weight_decay=weight_decay)]
    if no_decay:
        groups.append(dict(params=no_decay, weight_decay=0.0))
    return groups


def make_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(param_groups(model, weight_decay), lr=lr,
                             weight_decay=weight_decay)


def build(arm, vocab, seed, **kw):
    """Shared-backbone construction: the SAME seed gives every arm identical
    embeddings, mixer frontends/readouts, MLPs and head. Only the arm's own
    leaves differ, and they consume no RNG."""
    torch.manual_seed(seed)
    return Model(arm, vocab, **kw)


#: Optimizer note (the pilot's inertial coefficients are the one exception,
#: see NO_DECAY_LEAVES). The official layer marks A_log, Mu_log, dt_bias, mu_bias,
#: log_factor and D_log `_no_weight_decay`, but the official trainer
#: (flame/torchtitan `build_optimizers`) ignores that flag and applies one
#: weight decay to every parameter. v1 did the same, so train.py keeps a plain
#: AdamW over `model.parameters()`: no new optimizer variant is introduced.


def shared_parameter_check(a, b):
    """Names of tensors present in both models that are NOT bit-identical."""
    sa, sb = a.state_dict(), b.state_dict()
    common = set(sa) & set(sb)
    return [n for n in sorted(common) if not torch.equal(sa[n], sb[n])]


def unused_parameters(model):
    """Trainable parameters that received NO gradient in the last backward.
    For the writer arms these are exactly the MDN-gate-only parameters
    (mixer.NATIVE_GATE_PARAMS) in every block."""
    return [n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None]
