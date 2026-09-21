"""The coefficient map must reproduce the S5 study's values exactly."""

import json
import os

import torch

from prospective.coefficients import (prospective_coefficients,
                                      tss_coefficients, jury_ok)

GOLDEN = os.path.join(os.path.dirname(__file__), os.pardir,
                      "tests_prospective_golden.json")


def _rows():
    with open(GOLDEN) as fh:
        return json.load(fh)["rows"]


def test_matches_s5_golden_values():
    for r in _rows():
        got = prospective_coefficients(
            torch.tensor(r["M"], dtype=torch.float64),
            torch.tensor(r["gamma"], dtype=torch.float64),
            torch.tensor(r["T"], dtype=torch.float64), r["h"])
        for name, value in zip("abcd", got):
            assert abs(float(value) - r[name]) < 1e-12, (r, name, float(value))


def test_tss_is_the_same_map_not_a_second_equation():
    T = torch.linspace(0.05, 4.0, 40, dtype=torch.float64)
    zero = torch.zeros_like(T)
    for x, y in zip(tss_coefficients(T),
                    prospective_coefficients(zero, zero, T)):
        assert torch.equal(x, y)


def test_tss_reduces_to_literal_eq17():
    """a = 1 - h/T, b = 0, c = 1 + h/T, d = 1."""
    T = torch.linspace(0.05, 4.0, 40, dtype=torch.float64)
    a, b, c, d = tss_coefficients(T)
    assert torch.allclose(a, 1.0 - 1.0 / T, atol=0, rtol=1e-14)
    assert torch.equal(b, torch.zeros_like(b))
    assert torch.allclose(c, 1.0 + 1.0 / T, atol=0, rtol=1e-14)
    assert torch.allclose(d, torch.ones_like(d), atol=0, rtol=1e-14)


def test_tss_is_unstable_below_half_a_token():
    """On the DRIVE the single pole is a = 1 - h/T, stable iff T > h/2. This is
    the placement-dependence the bridge is built to test."""
    a, b, _, _ = tss_coefficients(torch.tensor([0.05, 0.4, 0.51, 2.0],
                                               dtype=torch.float64))
    assert jury_ok(a, b).tolist() == [False, False, True, True]


def test_generalized_native_boundary():
    """M = 0, gamma = h, T = 0 must give y_t = R_t exactly (native)."""
    one = torch.ones((), dtype=torch.float64)
    a, b, c, d = prospective_coefficients(0.0 * one, one, 0.0 * one)
    assert (float(a), float(b), float(c), float(d)) == (0.0, 0.0, 1.0, 0.0)
