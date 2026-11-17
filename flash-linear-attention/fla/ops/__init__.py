# -*- coding: utf-8 -*-

from .attn import parallel_attn
from .comba import chunk_comba, fused_recurrent_comba
from .gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from .momentum_delta_rule import chunk_mode_rule, fused_recurrent_mode_rule

# generalized_delta_rule (dplr/iplr) is not shipped in this trimmed tree
from .kda import chunk_kda, fused_recurrent_kda


__all__ = [
    'parallel_attn',
    'chunk_delta_rule', 'fused_chunk_delta_rule', 'fused_recurrent_delta_rule',
    'chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule',
    'chunk_comba', 'fused_recurrent_comba',
    'chunk_kda', 'fused_recurrent_kda',
    'chunk_mode_rule', 'fused_recurrent_mode_rule',
]
