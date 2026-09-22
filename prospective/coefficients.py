"""The prospective coefficient map, taken unchanged from the S5 study.

Provenance: `s5/discrete_recurrence.py` in `final-prospective-s5`, branch
`s5-two-compartment-parallel-scan` at 57b7996. The map below is the SAME
algebra as `generalized_coefficients` there; only the target signal differs
(S5 drives the filter with its recurrence target `Y = F s + G x`, this study
drives it with the write residual `R`).

    q     = M + h (gamma + T)
    alpha = M / q      beta = h^2 / q      delta = h T / q

    y_t = a y_(t-1) - b y_(t-2) + c R_t - d R_(t-1)
    a = 1 + alpha - beta      b = alpha
    c = beta + delta          d = delta

Two named points, NOT separate equations:

* TSS      -- M = gamma = 0, T learned. Gives a = 1 - h/T, b = 0,
              c = 1 + h/T, d = 1, i.e. literal TSS Eq. (17) driven by R. This
              is the arm that failed S5 training; `zucchet_coefficients` in the
              S5 repository is `generalized_coefficients` at M = gamma = 0, to
              machine precision (max |diff| 8.9e-16 over a1, a2, c1, c2).
* GENERALIZED -- M, gamma, T all learned. The arm that trained successfully in
              the S5 study.

Admissibility (the S5 study's gate, unchanged): the three strict Jury
conditions of z^2 - a z + b,

    1 - a + b > 0,    1 + a + b > 0,    1 - b > 0.
"""

import torch

H_TOKEN = 1.0
#: TSS start point: M = gamma = 0, T = h
TSS_T0 = H_TOKEN
#: The S5 study's successful generalized start, carried over: a FINITE mass and
#: damping. Not the native boundary -- see `project_leaves` for why a boundary
#: start cannot train.
GEN_M0, GEN_GAMMA0, GEN_T0 = 0.025, 1.0, H_TOKEN
#: declared numerical gaps, as in the six-arm ladder's `filtered.py`
G_MIN = 2.0 ** -10          # gamma + T >= G_MIN * h
DELTA_FILTER = 1e-3         # 4M + 2h(gamma+T) >= h^2 (1 + DELTA_FILTER)


def prospective_coefficients(mass, gamma, response, h=H_TOKEN):
    """(a, b, c, d) from (M, gamma, T). Formed once, no branch on values."""
    q = mass + h * (gamma + response)
    alpha, beta, delta = mass / q, (h * h) / q, (h * response) / q
    return (1.0 + alpha - beta, alpha, beta + delta, delta)


def tss_coefficients(response, h=H_TOKEN):
    """The M = gamma = 0 face. Kept as a call into the SAME map so the two
    arms cannot drift apart."""
    zero = torch.zeros_like(response)
    return prospective_coefficients(zero, zero, response, h)


@torch.no_grad()
def project_leaves(mass, gamma, response, h=H_TOKEN, freeze_mass_gamma=False):
    """Project (M, gamma, T) back into the admissible set, IN PLACE.

    The leaves are stored directly and projected after each optimizer step,
    exactly as the six-arm ladder stored kappa and nu. They are NOT passed
    through softplus. A smooth positive reparameterization multiplies the
    gradient reaching M by d(softplus)/d(raw) = sigmoid(raw), which at M ~ 1e-4
    is itself ~1e-4: starting an arm at the native boundary M = gamma = 0 then
    freezes it there, because its effective learning rate is four orders of
    magnitude below the rest of the model. Measured on seed 501 at 5210f44,
    where M went 1e-4 -> 3e-4 over 4000 steps and the generalized arm ran as
    TSS for the whole run. Every smooth positive transform has this property,
    so the fix is direct storage plus projection, not a different transform.

    The admissible set is the derivation's domain, as in `filtered.py`:
    M, gamma, T >= 0, gamma + T >= G_MIN h, and 4M + 2h(gamma+T) >= h^2
    (1 + DELTA_FILTER), which is the only non-trivial Jury condition here --
    1 - a + b = beta > 0 and 1 - b = 1 - alpha > 0 hold identically.
    """
    if freeze_mass_gamma:
        mass.zero_()
        gamma.zero_()
    else:
        mass.clamp_(min=0.0)
        gamma.clamp_(min=0.0)
    response.clamp_(min=0.0)
    # gamma + T >= G_MIN h
    deficit = (G_MIN * h) - (gamma + response)
    response.add_(deficit.clamp(min=0.0))
    # 4M + 2h(gamma + T) >= h^2 (1 + DELTA_FILTER), repaired through T
    need = (h * h) * (1.0 + DELTA_FILTER) - 4.0 * mass - 2.0 * h * gamma
    response.add_((need / (2.0 * h) - response).clamp(min=0.0))


def jury_ok(a, b):
    """The three strict Jury conditions of z^2 - a z + b, elementwise."""
    return ((1.0 - a + b) > 0) & ((1.0 + a + b) > 0) & ((1.0 - b) > 0)
