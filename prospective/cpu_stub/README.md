# CPU stand-in for `fla` (tests only)

Lets the model-level CPU tests (`test_model_cpu.py`) import the REAL official
layer file `flash-linear-attention/fla/layers/momentum_deltanet.py` on a
machine without Triton. Only the Triton-backed modules are replaced:

* `ShortConvolution` -> causal depthwise `F.conv1d` + SiLU
* `FusedRMSNormGated` -> `x * rsqrt(mean x^2 + eps) * w * act(g)`
* `chunk_mode_rule` / `fused_recurrent_mode_rule` -> raise

This checks plumbing (shapes, arms, causality, init, gradients), never Triton
numerics. The GPU tests (`test_integration.py`) use the real kernels. The
stub is used only when the real `fla` cannot be imported.
