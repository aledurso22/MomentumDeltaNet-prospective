# Prospective write rules on Momentum DeltaNet

Base: this repository at **c6e77fa261fb0c002fae1a14b6209a5b28d2edc9**
(`HuuYuLong/MomentumDeltaNet`, ICML 2026). Branch `prospective-rules` is
branched from that commit and changes nothing upstream.

Plan: `docs/MDN_RECURRENT_BRIDGE_PLAN.md` on branch `mdn-recurrent-bridge` of
`final-prospective-s5`.

| file | what |
|---|---|
| `coefficients.py` | the (M, gamma, T) -> (a, b, c, d) map, taken unchanged from the S5 study |
| `rules.py` | the arms as recurrent references, line-for-line against `naive.py` |
| `test_coefficients.py` | the map reproduces the S5 golden values to 1e-12 |
| `test_rules.py` | every arm reproduces the official native reference at its native point |

`tss` and `generalized` are the SAME code path and the SAME equation: TSS is
`M = gamma = 0`. Verified in the S5 repository that `zucchet_coefficients`
equals `generalized_coefficients(mass=0, gamma=0)` to 8.9e-16, so the arm that
failed S5 training is the zero-mass face of the arm that trained.

Not done here: `chunk.py`. This is the recurrent path only, by design -- see
section 5 of the plan.
