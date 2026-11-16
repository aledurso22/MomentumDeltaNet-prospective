"""Task semantics: targets separate from inputs, honest lag metric, legacy
metric reproduces v1 exactly, and the data stream is unchanged from v1."""

import importlib.util
import os
import subprocess
import tempfile

import pytest
import torch

from prospective.task import (IGNORE, LAG_TOKENS, PRODUCTION_TASK, accuracy,
                              make_batch, masks)

V1_COMMIT = "8d39b8a"     # last commit with the reduced (v1) mixer and task


def batch(seed=0, bsz=64):
    return make_batch(bsz, **PRODUCTION_TASK,
                      generator=torch.Generator().manual_seed(seed))


def test_production_shape():
    b = batch()
    assert b["x"].shape[1] == 52 and b["vocab"] == 58


def test_targets_are_not_model_inputs():
    b = batch(1)
    x, y = b["x"], b["y"]
    scored = y != IGNORE
    n_keys = PRODUCTION_TASK["n_keys"]
    # every scored position is a QUERY input token
    assert (x[scored] == b["query_tok"]).all()
    # the answer is never the input at the scored position or the next one:
    # the token after a QUERY is the next query's key or PAD, never a value
    nxt = torch.roll(x, -1, dims=1)
    nxt_is_value = (nxt >= n_keys) & (nxt < b["query_tok"])
    assert not (scored & nxt_is_value).any()
    assert not (scored & (x == y)).any()
    # y is not a shifted copy of x anywhere
    assert not (scored[:, 1:] & (y[:, 1:] == x[:, :-1])).any()
    # no query appears before the write/rewrite phase ends
    first_query = 2 * (PRODUCTION_TASK["n_pairs"] + PRODUCTION_TASK["n_revisions"])
    assert not scored[:, :first_query].any()


def _lag_from_tokens(x, s, n_keys, query_tok):
    """Independent lag: positions of VALUE tokens written for x[s-1] before s;
    revised iff written twice; lag = s - position of the latest value token."""
    key = int(x[s - 1])
    writes = [t + 1 for t in range(0, s - 1, 2)
              if int(x[t]) == key and n_keys <= int(x[t + 1]) < query_tok]
    return (s - writes[-1]) if len(writes) > 1 else -1


def test_lag_is_measured_in_tokens_from_the_rewrite():
    b = batch(2, bsz=32)
    for r in range(b["x"].shape[0]):
        for s in torch.nonzero(b["y"][r] != IGNORE).flatten().tolist():
            want = _lag_from_tokens(b["x"][r], s, PRODUCTION_TASK["n_keys"],
                                    b["query_tok"])
            assert int(b["lag"][r, s]) == want, (r, s)


def test_lag_metric_is_populated_and_disjoint():
    b = batch(3, bsz=256)
    m = masks(b)
    assert m["rev_lag_le8"].sum() > 0 and m["rev_lag_gt8"].sum() > 0
    assert torch.equal(m["rev_lag_le8"] | m["rev_lag_gt8"], m["revised"])
    assert not (m["rev_lag_le8"] & m["rev_lag_gt8"]).any()
    assert (b["lag"][m["rev_lag_le8"]] <= LAG_TOKENS).all()
    acc = accuracy(torch.zeros(*b["x"].shape, b["vocab"]), b)
    assert acc["n_rev_lag_le8"] == int(m["rev_lag_le8"].sum())


def _v1_task():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        src = subprocess.check_output(
            ["git", "-C", root, "show", f"{V1_COMMIT}:prospective/task.py"],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:
        pytest.skip("v1 task.py not available from git")
    fh = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    fh.write(src)
    fh.close()
    spec = importlib.util.spec_from_file_location("v1_task", fh.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_same_tokens_as_v1_and_legacy_metric_is_v1s():
    v1 = _v1_task()
    for seed in range(4):
        new = batch(seed, bsz=32)
        old = v1.make_batch(32, **PRODUCTION_TASK,
                            generator=torch.Generator().manual_seed(seed))
        assert torch.equal(new["x"], old["x"])
        assert torch.equal(new["y"], old["y"])
        assert torch.equal(new["group"], old["group"])
        m = masks(new)
        scored = new["y"] != IGNORE
        assert torch.equal(m["legacy_immediate_revised"],
                           scored & (old["group"] == 1) & old["immediate"])
        # and the v1 accuracy() keys map onto the legacy metrics
        logits = torch.randn(*new["x"].shape, new["vocab"],
                             generator=torch.Generator().manual_seed(seed))
        a_old, a_new = v1.accuracy(logits, old), accuracy(logits, new)
        for o, n in (("immediate_revised", "legacy_immediate_revised"),
                     ("later_revised", "legacy_later_revised"),
                     ("overall", "overall"), ("revised", "revised"),
                     ("untouched", "untouched")):
            assert a_old[o] == pytest.approx(a_new[n], abs=1e-12), o


def test_v1_immediate_was_not_an_eight_token_window():
    """Documents the v1 discrepancy: its 'immediate' set contains lags up to
    16, so it is not an eight-token window. (At the production config no
    second-half query can have lag <= 8, so the query-index cut only removes
    lags 10-16.)"""
    b = batch(5, bsz=512)
    m = masks(b)
    leg = m["legacy_immediate_revised"]
    assert (b["lag"][leg] > LAG_TOKENS).any()
    assert int(b["lag"][leg].max()) <= 16
