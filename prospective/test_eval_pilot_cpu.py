"""Checkpoint-only follow-up checks with the repository's CPU layer stand-ins.

No training runs here: the fixture labels an untrained tiny model checkpoint
as step 2000 solely to exercise reconstruction and evaluation invariants.
"""

import copy
import hashlib
import json
import os
import sys

import pytest
import torch

try:
    import fla
    REAL_FLA = "cpu_stub" not in (fla.__file__ or "")
except Exception:
    REAL_FLA = False
if REAL_FLA:
    pytest.skip("real fla available: CPU stand-in tests do not apply",
                allow_module_level=True)
if "fla" not in sys.modules:
    for name in [k for k in sys.modules if k == "fla" or k.startswith("fla.")]:
        del sys.modules[name]
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cpu_stub"))

from prospective import eval_pilot  # noqa: E402
from prospective.experiment_pilot import ARMS, EXPERIMENT  # noqa: E402
from prospective.model import Model, build  # noqa: E402
from prospective.task import HELDOUT, make_batch_heldout  # noqa: E402
from prospective.train import fingerprint, model_kwargs, parser  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint_state():
    args = parser().parse_args([
        "--device", "cpu", "--seed", "701", "--experiment", EXPERIMENT,
        "--d-model", "32", "--n-heads", "2", "--head-dim", "8",
        "--n-layers", "1", "--steps", "2000", "--eval-every", "100",
        "--spec", "prospective/specs/pilot/pilot.json",
    ])
    inst = [(a, a, args.lr, None) for a in ARMS]
    state = {
        "fingerprint": fingerprint(args, inst),
        "step": 2000,
        "repo_sha": "47fe52a6705994d191d8ff616aefd03361d1e51a",
        "models": {
            a: build(a, vocab=58, seed=701, **model_kwargs(args)).state_dict()
            for a in ARMS
        },
        "failed": {}, "unused": {}, "curves": {}, "opts": {},
        "gen": torch.Generator().manual_seed(10701).get_state(),
        "segments": [], "wall_seconds": 1.0,
    }
    return state


@pytest.fixture
def checkpoint(tmp_path, checkpoint_state):
    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint_state, path)
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evaluation_pairs_all_arms_and_never_trains_or_mutates(
        tmp_path, checkpoint, checkpoint_state, monkeypatch):
    before = digest(checkpoint)
    seen = {a: [] for a in ARMS}
    models = {}
    original_forward = Model.forward

    def observed_forward(self, x):
        assert not self.training
        assert not torch.is_grad_enabled()
        seen[self.arm].append(x.detach().cpu().clone())
        models[self.arm] = self
        return original_forward(self, x)

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint evaluation attempted training")

    monkeypatch.setattr(Model, "forward", observed_forward)
    monkeypatch.setattr(Model, "project_", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    out = tmp_path / "followup.json"
    report = eval_pilot.evaluate_seed(checkpoint, out, device="cpu",
                                      batches=1, batch_size=2)

    assert digest(checkpoint) == before
    assert set(report["suites"]) == set(HELDOUT)
    for i, suite in enumerate(sorted(HELDOUT)):
        expected = make_batch_heldout(
            2, **HELDOUT[suite],
            generator=torch.Generator().manual_seed(177000 + 10 * 701 + i))
        for arm in ARMS:
            assert torch.equal(seen[arm][i], expected["x"])
            metrics = report["suites"][suite][arm]
            assert metrics["n_overall"] == 24
            assert metrics["n_revised"] == metrics["n_untouched"] == 12
            assert metrics["overall"] == pytest.approx(
                100.0 * metrics["correct_overall"] / metrics["n_overall"])
            if suite.startswith("delay"):
                assert metrics["n_rev_lag_le8"] == 0
                assert metrics["rev_lag_le8"] is None
                assert metrics["primary"] is None
    for arm, model in models.items():
        assert all(torch.equal(t.cpu(), checkpoint_state["models"][arm][name])
                   for name, t in model.state_dict().items())

    def reject_nonstandard_json(value):
        raise AssertionError(f"nonstandard JSON number: {value}")

    assert json.loads(out.read_text(), parse_constant=reject_nonstandard_json) == report
    with pytest.raises((FileExistsError, ValueError)):
        eval_pilot.evaluate_seed(checkpoint, out, device="cpu",
                                 batches=1, batch_size=2)


@pytest.mark.parametrize("corruption", ["step", "failed", "seed", "task", "arms"])
def test_checkpoint_contract_is_checked_before_evaluation(
        tmp_path, checkpoint_state, corruption):
    state = copy.deepcopy(checkpoint_state)
    fp = json.loads(state["fingerprint"])
    if corruption == "step":
        state["step"] = 1800
    elif corruption == "failed":
        state["failed"] = {"native": {"step": 1}}
    elif corruption == "seed":
        fp["cfg"]["seed"] = 999
    elif corruption == "task":
        fp["cfg"]["n_keys"] = 25
    elif corruption == "arms":
        fp["inst"] = fp["inst"][:-1]
    state["fingerprint"] = json.dumps(fp)
    path = tmp_path / "invalid.pt"
    torch.save(state, path)
    with pytest.raises((ValueError, RuntimeError)):
        eval_pilot.load_checkpoint(path)


def test_reconstruction_uses_strict_state_loading(tmp_path, checkpoint_state):
    state = copy.deepcopy(checkpoint_state)
    state["models"][ARMS[0]].pop("embed.weight")
    path = tmp_path / "missing-weight.pt"
    torch.save(state, path)
    with pytest.raises((ValueError, RuntimeError)):
        eval_pilot.evaluate_seed(path, tmp_path / "bad.json", device="cpu",
                                 batches=1, batch_size=2)


def test_nonfinite_logits_fail_instead_of_becoming_accuracy(
        tmp_path, checkpoint, monkeypatch):
    def nonfinite_forward(self, x):
        return torch.full((*x.shape, 58), float("nan"), device=x.device)

    monkeypatch.setattr(Model, "forward", nonfinite_forward)
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        eval_pilot.evaluate_seed(checkpoint, tmp_path / "nonfinite.json",
                                 device="cpu", batches=1, batch_size=2)
