"""The official MomentumDeltaNet layer with an explicit recurrent-backend hook.

`ProspectiveMDN` SUBCLASSES `fla.layers.momentum_deltanet.MomentumDeltaNet`
(pinned upstream c6e77fa, unmodified), so every parameter, its
initialisation, the short convolutions, the gated RMS norm and the output
projection are the official modules themselves. Only `forward` is
overridden, and it is split into three stages:

    frontend(h)  -> q, k, v, log_alpha, log_mu, beta, eta      (official)
    backend(...) -> o                                         (the hook)
    readout(o,h) -> gated RMS norm, o_proj                    (official)

`frontend` and `readout` are line-for-line copies of the official forward
for the whole-sequence case (no padding mask, no cache, no varlen); the
copy is checked BITWISE against the official forward in
`test_integration.py::test_frontend_copy_is_bitwise_official`.

Backends:

    official_chunk  `chunk_mode_rule` exactly as the official layer calls it.
                    Native arm only. Backward-capable; used for verification.
    reference       our PyTorch token loop (`rules.recurrent_ref`), for all
                    six arms. The in-kernel preprocessing of
                    `chunk_mode_rule(use_qk_l2norm_in_kernel=True,
                    use_p_times_alpha=True)` is reproduced explicitly:
                    q, k <- l2norm (eps 1e-6, x / sqrt(sum x^2 + eps)),
                    p = alpha * l2norm(k), scale = K^-1/2.

The official forward switches to `fused_recurrent` whenever q_len <= 64 and
then asserts `mode == 'chunk'` in training, so it cannot train at our
sequence length (52). That dispatch is bypassed here: the backend is the
declared one, always.

Whole-sequence only. Padding masks, caches and `cu_seqlens` raise. The
filter arms' history is not carried across calls (see `rules.py`).
"""

import torch
import torch.nn as nn
from einops import rearrange

from fla.layers.momentum_deltanet import MomentumDeltaNet
from fla.ops.momentum_delta_rule import chunk_mode_rule

from .coefficients import GEN_GAMMA0, GEN_M0, GEN_T0, TSS_T0
from .projection import (INERTIA_A0, INERTIA_R0, MOTION_ARMS,
                         PROJECTION_ARMS)
from .reference import reference_recurrence
from .rules import ARMS, FILTER_ARMS
from .writers import REFERENCE_CFG, WRITER_ARMS

#: every arm the mixer accepts: the six v2 rules, the three v3 writers and
#: the three projection writers of the geometry/inertia pilot
ALL_ARMS = ARMS + WRITER_ARMS + PROJECTION_ARMS
#: official parameters that ONLY the MDN gates use. The writers keep them
#: (so shared init is identical across arms) but never read them; they get
#: no gradient and AdamW leaves them at their initial values.
NATIVE_GATE_PARAMS = ("a_proj.weight", "b_proj.weight", "m_proj.weight",
                      "e_proj.weight", "A_log", "dt_bias", "mu_bias",
                      "Mu_log", "log_factor")
#: the momentum-gate parameters: unused by `gated`, which forces mu = 0
MOMENTUM_GATE_PARAMS = ("m_proj.weight", "Mu_log", "mu_bias")


def unused_by_design(arm):
    """Official parameters an arm keeps (for identical shared init) but never
    reads, so they get no gradient."""
    if arm in WRITER_ARMS or arm in PROJECTION_ARMS:
        return NATIVE_GATE_PARAMS
    if arm == "gated":
        return MOMENTUM_GATE_PARAMS
    return ()

BACKENDS = ("reference", "official_chunk")

#: the official MomentumDeltaNetConfig defaults, as the official model passes
#: them to the layer (the layer's own norm_eps default, 1e-5, is NOT what the
#: model uses)
OFFICIAL_LAYER_KW = dict(mode="chunk", use_gate=True, use_lora_gate=False,
                         use_p_times_alpha=True, use_short_conv=True,
                         conv_size=4, min_log_mu=-2.0, tau_factor=1,
                         use_output_correction=True, norm_eps=1e-6,
                         num_v_heads=None)
#: MomentumDeltaNetConfig.initializer_range
INITIALIZER_RANGE = 0.02

#: QHM's declared start: nu = 1 is the native function exactly. Stored
#: directly and clamped to [0, 1] after each step.
QHM_NU0, QHM_DOMAIN = 1.0, (0.0, 1.0)


class ProspectiveMDN(MomentumDeltaNet):

    def __init__(self, arm, hidden_size, num_heads, head_dim, expand_v=2,
                 backend="reference", layer_idx=None, writer_cfg=None):
        if arm not in ALL_ARMS:
            raise ValueError(arm)
        if backend not in BACKENDS:
            raise ValueError(backend)
        super().__init__(hidden_size=hidden_size, expand_v=expand_v,
                         head_dim=head_dim, num_heads=num_heads,
                         layer_idx=layer_idx, **OFFICIAL_LAYER_KW)
        assert self.num_v_heads == self.num_heads
        self.arm = arm
        self.backend = backend
        # writer hyperparameters are fixed constants, not learned leaves
        self.writer_cfg = (dict(REFERENCE_CFG[arm], **(writer_cfg or {}))
                           if arm in WRITER_ARMS else None)
        # arm leaves come AFTER the official parameters and use torch.full,
        # which consumes no RNG: every shared tensor is drawn identically
        if arm in FILTER_ARMS:
            gen = arm == "generalized"
            self.fil_T = nn.Parameter(torch.full((num_heads,),
                                                 GEN_T0 if gen else TSS_T0))
            self.fil_M = nn.Parameter(torch.full((num_heads,),
                                                 GEN_M0 if gen else 0.0),
                                      requires_grad=gen)
            self.fil_gamma = nn.Parameter(torch.full((num_heads,),
                                                     GEN_GAMMA0 if gen else 0.0),
                                          requires_grad=gen)
        if arm == "qhm":
            self.nu = nn.Parameter(torch.full((), QHM_NU0))
        if arm in MOTION_ARMS:
            # outer-trained, one per head; constant over the tokens of a pass
            self.inertia_A = nn.Parameter(torch.full((num_heads,), INERTIA_A0))
            self.inertia_r = nn.Parameter(torch.full((num_heads,), INERTIA_R0))

    @torch.no_grad()
    def official_model_init_(self):
        """What `MomentumDeltaNetPreTrainedModel._init_weights` does to this
        layer's Linear / Conv1d submodules (normal(0, initializer_range), zero
        bias). Its re-draw of A_log, Mu_log, dt_bias, mu_bias is the same
        distribution as the layer's own __init__, so it is not repeated."""
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(mod.weight, mean=0.0, std=INITIALIZER_RANGE)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    # ---- official, copied ------------------------------------------------
    def frontend(self, hidden_states):
        if self.use_short_conv:
            q, _ = self.q_conv1d(x=self.q_proj(hidden_states), cache=None,
                                 output_final_state=False, cu_seqlens=None)
            k, _ = self.k_conv1d(x=self.k_proj(hidden_states), cache=None,
                                 output_final_state=False, cu_seqlens=None)
            v, _ = self.v_conv1d(x=self.v_proj(hidden_states), cache=None,
                                 output_final_state=False, cu_seqlens=None)
        else:
            q = torch.nn.functional.silu(self.q_proj(hidden_states))
            k = torch.nn.functional.silu(self.k_proj(hidden_states))
            v = torch.nn.functional.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d',
                                       d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        a = self.a_proj(hidden_states).float()
        b = self.b_proj(hidden_states).float()
        m = self.m_proj(hidden_states).float()
        e = self.e_proj(hidden_states).float() / self.tau

        F = torch.nn.functional
        log_mu = - self.Mu_log.float().exp() * F.softplus(m + self.mu_bias)
        log_alpha = - self.A_log.float().exp() * F.softplus(a + self.dt_bias)
        eta = F.tanh(e) + 1
        beta = F.sigmoid(b)

        theta = torch.arctan(eta * self.log_factor.float().exp())
        beta_upper = torch.sin(theta) ** 2
        alpha_upper = torch.cos(theta) ** 2
        beta = beta_upper * beta
        log_alpha = alpha_upper.log() + log_alpha

        if self.min_log_mu is not None:
            log_mu.clamp_min_(self.min_log_mu)

        # output correction is applied to the UN-normalised q and k; the l2
        # normalisation happens afterwards, inside the kernel (or `reference`)
        if self.use_output_correction:
            q = (q - self.D_log.float().exp()[None, None, :, None] * k).to(k.dtype)
        return dict(q=q, k=k, v=v, log_alpha=log_alpha, log_mu=log_mu,
                    beta=beta, eta=eta)

    def readout(self, o, hidden_states):
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d',
                          d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        return self.o_proj(o)

    # ---- the hook --------------------------------------------------------
    def backend_official_chunk(self, f):
        if self.arm != "native":
            raise ValueError("official_chunk implements the native rule only")
        o, _ = chunk_mode_rule(
            q=f["q"], k=f["k"], v=f["v"], log_alpha=f["log_alpha"],
            beta=f["beta"], log_mu=f["log_mu"], eta=f["eta"],
            initial_state=None, output_final_state=False, cu_seqlens=None,
            use_qk_l2norm_in_kernel=True,
            use_p_times_alpha=self.use_p_times_alpha)
        return o

    def leaves(self):
        if self.arm in FILTER_ARMS:
            return dict(mass=self.fil_M, gamma=self.fil_gamma,
                        response=self.fil_T)
        if self.arm == "qhm":
            return dict(nu=self.nu)
        if self.arm in MOTION_ARMS:
            return dict(A=self.inertia_A, r=self.inertia_r)
        return {}

    def backend_reference(self, f):
        assert self.use_p_times_alpha
        return reference_recurrence(self.arm, f, self.leaves(),
                                    self.writer_cfg).to(f["q"].dtype)

    def forward(self, hidden_states, attention_mask=None,
                past_key_values=None, use_cache=False, **kwargs):
        if (attention_mask is not None or past_key_values is not None
                or use_cache or kwargs.get("cu_seqlens") is not None):
            raise NotImplementedError(
                "whole-sequence execution only (no mask/cache/varlen)")
        f = self.frontend(hidden_states)
        if self.backend == "official_chunk":
            o = self.backend_official_chunk(f)
        else:
            o = self.backend_reference(f)
        return self.readout(o, hidden_states), None, None
