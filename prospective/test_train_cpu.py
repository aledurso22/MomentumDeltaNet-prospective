"""train.py on CPU (stubbed Triton modules, see cpu_stub/README.md):
checkpoint + resume reproduces an uninterrupted run bitwise, a resume with a
different configuration is refused, and a non-finite update is recorded and
not applied. Skipped where the real fla imports (GPU nodes)."""

import json
import os
import sys

import pytest
import torch

SKIP_REASON = "real fla available: CPU stub tests do not apply"
try:
    import fla
    REAL_FLA = "cpu_stub" not in (fla.__file__ or "")
except Exception:
    REAL_FLA = False
if REAL_FLA:
    pytest.skip(SKIP_REASON, allow_module_level=True)
if "fla" not in sys.modules:
    for _k in [k for k in sys.modules if k == "fla" or k.startswith("fla.")]:
        del sys.modules[_k]
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cpu_stub"))

from prospective.model import build, make_optimizer  # noqa: E402
from prospective.task import PRODUCTION_TASK, make_batch  # noqa: E402
from prospective.train import parser, run, update_step  # noqa: E402

SMALL = ["--device", "cpu", "--d-model", "32", "--n-heads", "2", "--head-dim", "8",
         "--bsz", "4", "--eval-bsz", "4", "--eval-batches", "2", "--eval-every", "3",
         "--seed", "11", "--arms", "native", "geometric_motion_learned",
         "geometric_projection"]


def same(x, y):          # NaN-safe (metrics with no samples are NaN)
    return json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)


def go(tmp, steps, *extra):
    out = os.path.join(tmp, "results.json")
    return run(parser().parse_args(SMALL + ["--steps", str(steps), "--out", out,
                                            "--ckpt-every", "3", *extra]))


def test_resume_reproduces_uninterrupted_run_bitwise(tmp_path):
    a = go(str(tmp_path / "a"), 6)
    b_dir = str(tmp_path / "b")
    go(b_dir, 3)                                         # "killed" after step 3
    assert os.path.exists(os.path.join(b_dir, "ckpt.pt"))
    b = go(b_dir, 6, "--resume")
    assert same(a["curves"], b["curves"])
    assert same(a["learned"], b["learned"]) and same(a["inertia"], b["inertia"])
    assert [s["start_step"] for s in b["provenance"]["segments"]] == [1, 4]
    # every parameter and optimiser state, bitwise, from the final checkpoints
    ca = torch.load(os.path.join(str(tmp_path / "a"), "ckpt.pt"), weights_only=False)
    cb = torch.load(os.path.join(b_dir, "ckpt.pt"), weights_only=False)
    assert ca["step"] == cb["step"] == 6
    for arm, sd in ca["models"].items():
        assert all(torch.equal(t, cb["models"][arm][n]) for n, t in sd.items()), arm
        sa, sb = ca["opts"][arm]["state"], cb["opts"][arm]["state"]
        assert all(torch.equal(sa[i][k], sb[i][k]) for i in sa for k in sa[i]), arm
    assert torch.equal(ca["gen"], cb["gen"])
    with open(os.path.join(b_dir, "results.json")) as fh:
        assert same(json.load(fh)["curves"], a["curves"])


def test_resume_without_checkpoint_starts_fresh(tmp_path):
    a = go(str(tmp_path / "a"), 3)
    b = go(str(tmp_path / "b"), 3, "--resume")
    assert same(a["curves"], b["curves"]) and same(a["learned"], b["learned"])


def test_resume_refuses_a_different_configuration(tmp_path):
    d = str(tmp_path / "c")
    go(d, 3)
    with pytest.raises(SystemExit):
        go(d, 6, "--resume", "--lr", "1e-3")


def test_non_finite_update_is_not_applied():
    m = build("geometric_motion_learned", vocab=58, seed=0)
    opt = make_optimizer(m, 3e-3, 0.01)
    b = make_batch(4, **PRODUCTION_TASK, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        m.blocks[0].mix.v_proj.weight[0, 0] = float("nan")
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    _, ok = update_step(m, opt, b, 1.0)
    assert not ok
    for n, p in m.named_parameters():
        assert torch.equal(p, before[n]) or (torch.isnan(p) == torch.isnan(before[n])).all()
    assert not opt.state                                  # no AdamW step taken


def test_rng_state_is_restored_as_a_cpu_byte_tensor():
    """The resume path must hand torch.Generator.set_state a CPU uint8
    tensor whatever device the checkpoint tensors were mapped to (on GPU,
    map_location=cuda moved it to CUDA and set_state raised)."""
    from prospective.train import cpu_rng_state
    g = torch.Generator().manual_seed(3)
    st = g.get_state()
    torch.rand(5, generator=g)
    g.set_state(cpu_rng_state(st.clone()))
    h = torch.Generator().manual_seed(3)
    assert torch.equal(torch.rand(5, generator=g), torch.rand(5, generator=h))
    with pytest.raises(TypeError):
        cpu_rng_state(st.float())


def test_resume_loads_the_checkpoint_on_the_cpu():
    import inspect
    from prospective import train
    src = inspect.getsource(train.run)
    assert 'map_location="cpu"' in src and "map_location=dev" not in src
    assert "gen.set_state(cpu_rng_state(" in src
