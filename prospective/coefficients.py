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


def jury_ok(a, b):
    """The three strict Jury conditions of z^2 - a z + b, elementwise."""
    return ((1.0 - a + b) > 0) & ((1.0 + a + b) > 0) & ((1.0 - b) > 0)
