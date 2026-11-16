def chunk_mode_rule(*a, **k):
    raise RuntimeError("CPU stub: the Triton chunk kernel is not available")


fused_recurrent_mode_rule = chunk_mode_rule
