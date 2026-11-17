"""The v4 pilot's declared configuration (CPU, no torch model needed)."""

import json
import os
import sys

try:              # train.py imports the mixer: the real fla, or else the CPU
    import fla  # noqa: F401  stub otherwise (cpu_stub/README.md)
except Exception:
    for _k in [k for k in sys.modules if k == "fla" or k.startswith("fla.")]:
        del sys.modules[_k]
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cpu_stub"))
    import fla  # noqa: F401

from prospective import experiment_pilot as P  # noqa: E402
from prospective.experiment_v3 import CONFIRM_SEEDS, DEV_SEEDS  # noqa: E402
from prospective.train import parser  # noqa: E402


def test_exactly_the_four_pilot_arms():
    assert P.ARMS == ("native", "euclidean_projection", "geometric_projection",
                      "geometric_motion_learned")
    banned = {"joint", "nesterov", "rls", "geometric", "tss", "generalized", "qhm", "gated"}
    assert not banned & set(P.ARMS)


def test_spec_file_is_the_declared_spec():
    with open(P.SPEC_PATH) as fh:
        assert json.load(fh) == P.spec()
    assert all("lr" not in e and "writer" not in e for e in P.spec())  # no sweep


def test_flags_give_the_pilot_configuration():
    root = os.path.dirname(os.path.dirname(os.path.abspath(P.__file__)))
    cwd = os.getcwd()
    os.chdir(root)
    try:
        a = parser().parse_args(P.flags())
        assert os.path.exists(a.spec)
    finally:
        os.chdir(cwd)
    assert (a.steps, a.eval_every, a.ckpt_every, a.experiment) == (
        2000, 100, 200, "v4-geometry-pilot")
    # the existing configuration, unchanged
    assert (a.d_model, a.n_layers, a.n_heads, a.head_dim, a.expand_v) == (128, 2, 4, 32, 2)
    assert (a.bsz, a.lr, a.weight_decay, a.clip) == (16, 3e-3, 0.01, 1.0)
    assert (a.n_pairs, a.n_revisions, a.n_queries, a.n_keys, a.n_values) == (8, 6, 12, 24, 32)


def test_seeds_are_fresh_and_paired_one_per_gpu():
    assert len(P.SEEDS) == 4
    old = set(range(501, 509)) | set(CONFIRM_SEEDS) | set(DEV_SEEDS)
    assert not old & set(P.SEEDS)


def _fake(seed, arms=P.ARMS, overall=40.0, A_minus_d=0.0015625):
    curve = lambda bump: [dict(step=s, primary=30 + bump, rev_lag_le8=20 + bump,
                               untouched=40, revised=25, overall=overall,
                               n_primary=None, n_rev_lag_le8=100,
                               inertia={"layer0": dict(A=[.003125], r=[.5], d=[.0015625],
                                                       A_minus_d=[A_minus_d])})
                              for s in (100, 200)]
    return dict(config=dict(experiment=P.EXPERIMENT, seed=seed), arms=list(arms),
                curves={a: curve(i) for i, a in enumerate(arms)}, failed={},
                wall_seconds=60.0, provenance=dict(gpu="x"))


def test_analyse_runs_and_flags_cancelled_motion(tmp_path, capsys):
    for s in P.SEEDS[:3]:
        d = tmp_path / str(s)
        d.mkdir()
        (d / "results.json").write_text(json.dumps(_fake(s, A_minus_d=0.0)))
    P.analyse(str(tmp_path / "*" / "results.json"))
    out = capsys.readouterr().out
    assert "3/4 seeds" in out and "missing" in out and "704" in out
    assert "NOT an inertial benefit" in out
    assert "learned inertia beyond geometry" in out


def test_near_chance_seeds_are_inconclusive(tmp_path, capsys):
    d = tmp_path / "1"
    d.mkdir()
    (d / "results.json").write_text(json.dumps(_fake(P.SEEDS[0], overall=3.0)))
    P.analyse(str(tmp_path / "*" / "results.json"))
    assert "none; inconclusive" in capsys.readouterr().out
