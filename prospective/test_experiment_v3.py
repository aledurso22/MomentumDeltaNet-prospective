"""The v3 protocol is internally consistent before any run."""

from prospective import experiment_v3 as E
from prospective.task import NONINFERIORITY_MARGIN_PP, PRIMARY_COMPONENTS


def test_six_arms_and_main_contrasts():
    assert E.ARMS == ("gated", "native", "nesterov", "rls", "geometric", "joint")
    assert E.MAIN_CONTRASTS == (("joint", "native"), ("joint", "rls"))


def test_equal_tuning_budgets():
    assert {len(v) for v in E.DEV_GRID.values()} == {6}
    assert set(E.DEV_GRID) == set(E.ARMS)


def test_geometric_grid_contains_the_documented_reference():
    assert any(c.get("writer") == dict(A=0.05, G=1.0) for c in E.DEV_GRID["geometric"])


def test_seed_sets_are_disjoint_and_fresh():
    v2 = set(range(501, 509))
    assert not set(E.DEV_SEEDS) & set(E.CONFIRM_SEEDS)
    assert not (set(E.DEV_SEEDS) | set(E.CONFIRM_SEEDS)) & v2
    assert len(E.CONFIRM_SEEDS) == 8


def test_primary_and_margin_are_the_declared_ones():
    assert PRIMARY_COMPONENTS == ("rev_lag_le8", "untouched")
    assert E.NONINFERIORITY == ("untouched", NONINFERIORITY_MARGIN_PP) == ("untouched", 1.0)


def test_training_uses_the_restored_small_architecture():
    t = E.TRAIN
    assert (t["d_model"], t["n_layers"], t["n_heads"], t["head_dim"], t["expand_v"]) == (128, 2, 4, 32, 2)


def test_sign_test_and_bootstrap_helpers():
    assert E.sign_p(8, 8) == 1 / 256
    lo, hi = E.bootstrap_ci([1.0] * 8)
    assert lo == hi == 1.0
