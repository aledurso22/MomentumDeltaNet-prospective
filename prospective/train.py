"""Train all six arms together, on identical data, from identical init.

Every arm sees the SAME batches in the SAME order at the same step, and starts
from a shared-seed backbone (see `model.build`). Arms are stepped in lockstep
inside one loop, so a difference between two arms cannot come from data
order, initialisation or schedule.

Precision: FP32 throughout, no autocast, TF32 disabled for torch matmuls and
cuDNN. The flags are recorded in `provenance`.

Writes `results.json` with per-arm curves (each eval pooled over the fixed
eval set, with sample counts) and the provenance of the tree it ran from.
Aggregation is a separate step (`aggregate.py`).

Arms are given either as `--arms` (v2 behaviour: one global lr) or as a
`--spec` JSON list of instances,

    [{"label": "joint", "arm": "joint", "lr": 0.003,
      "writer": {"rho": 0.98, "T0": 4.0}}, ...]

so several configurations (development runs) or several arms with their own
locked settings (confirmation runs) train in lockstep on the same batches.
`--heldout` adds the separately identified held-out suites (task.HELDOUT),
evaluated once at the end on their own fixed sets.
"""

import argparse
import json
import os
import subprocess
import time

import torch
import torch.nn.functional as F

from .model import ARM_LEAVES, MODEL_DEFAULTS, build, unused_parameters
from .rules import ARMS
from .task import (HELDOUT, IGNORE, METRICS, PRODUCTION_TASK, hits,
                   make_batch, make_batch_heldout, primary_score)

#: written into every results.json; aggregate.py refuses to mix versions
MIXER_VERSION = "v2-official-mdn-mixer"


def git_sha(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def set_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def loss_of(model, batch):
    logits = model(batch["x"])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           batch["y"].reshape(-1), ignore_index=IGNORE), logits


def evaluate(model, batches):
    """Pooled over the whole eval set: sum correct / sum count per metric."""
    model.eval()
    tot = {m: [0, 0] for m in METRICS}
    with torch.no_grad():
        for b in batches:
            _, logits = loss_of(model, b)
            for k, (c, n) in hits(logits, b).items():
                tot[k][0] += c
                tot[k][1] += n
    model.train()
    out = {k: (100.0 * c / n if n else float("nan")) for k, (c, n) in tot.items()}
    out.update({f"n_{k}": n for k, (_, n) in tot.items()})
    out["primary"] = primary_score(out)
    return out


def to_dev(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def task_kwargs(args):
    return dict(n_pairs=args.n_pairs, n_revisions=args.n_revisions,
                n_queries=args.n_queries, n_keys=args.n_keys,
                n_values=args.n_values)


def model_kwargs(args):
    return dict(d_model=args.d_model, n_heads=args.n_heads,
                head_dim=args.head_dim, expand_v=args.expand_v,
                n_layers=args.n_layers, backend="reference")


def instances(args):
    """[(label, arm, lr, writer_cfg)] from --spec, or from --arms + --lr."""
    if args.spec:
        with open(args.spec) as fh:
            spec = json.load(fh)
        out = [(e.get("label", e["arm"]), e["arm"], float(e.get("lr", args.lr)),
                e.get("writer")) for e in spec]
    else:
        out = [(a, a, args.lr, None) for a in (args.arms or list(ARMS))]
    labels = [o[0] for o in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate labels in spec: {labels}")
    return out


def leaves_of(model):
    return {n: p.detach().cpu().tolist() for n, p in model.named_parameters()
            if n.split(".")[-1] in ARM_LEAVES}


def run(args):
    set_fp32()
    dev = torch.device(args.device)
    gen = torch.Generator().manual_seed(args.seed + 10_000)
    task = task_kwargs(args)
    vocab = make_batch(2, **task, generator=gen)["vocab"]

    eval_gen = torch.Generator().manual_seed(args.seed + 99_000)
    eval_batches = [to_dev(make_batch(args.eval_bsz, **task,
                                      generator=eval_gen), dev)
                    for _ in range(args.eval_batches)]

    inst = instances(args)
    arms = [lab for lab, _, _, _ in inst]
    models, opts, unused = {}, {}, {}
    for lab, arm, lr, wcfg in inst:
        m = build(arm, vocab, seed=args.seed, writer_cfg=wcfg,
                  **model_kwargs(args)).to(dev)
        models[lab] = m
        opts[lab] = torch.optim.AdamW(m.parameters(), lr=lr,
                                      weight_decay=args.weight_decay)

    curves = {a: [] for a in arms}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        b = to_dev(make_batch(args.bsz, **task, generator=gen), dev)
        for a in arms:                            # ONE batch, all arms see it
            loss, _ = loss_of(models[a], b)
            opts[a].zero_grad(set_to_none=True)
            loss.backward()
            if step == 1:
                unused[a] = unused_parameters(models[a])
            torch.nn.utils.clip_grad_norm_(models[a].parameters(), args.clip)
            opts[a].step()
            models[a].project_()                  # leaves are stored directly
        if step % args.eval_every == 0 or step == args.steps:
            for a in arms:
                acc = evaluate(models[a], eval_batches)
                acc["step"] = step
                curves[a].append(acc)
            line = "  ".join(f"{a}:{curves[a][-1]['rev_lag_le8']:5.1f}"
                             for a in arms)
            n8 = curves[arms[0]][-1]["n_rev_lag_le8"]
            print(f"[{step:5d}] {time.time()-t0:7.1f}s  lag<=8 (n={n8})  {line}",
                  flush=True)
            if args.spec:
                prim = "  ".join(f"{a}:{curves[a][-1]['primary']:5.1f}"
                                 for a in arms)
                print(f"        primary  {prim}", flush=True)
            for a in arms:
                lv = {n.split(".")[-1]: p.detach()
                      for n, p in models[a].named_parameters()
                      if n.split(".")[-1] in ARM_LEAVES}
                if lv:
                    txt = "  ".join(
                        f"{n}={v.min().item():.3f}..{v.max().item():.3f}"
                        for n, v in sorted(lv.items()))
                    print(f"        {a:>12}  {txt}", flush=True)

    heldout = {}
    if args.heldout:
        for i, (suite, cfg) in enumerate(sorted(HELDOUT.items())):
            g = torch.Generator().manual_seed(args.seed + 77_000 + i)
            hb = [to_dev(make_batch_heldout(args.eval_bsz, **cfg, generator=g), dev)
                  for _ in range(args.heldout_batches)]
            for a in arms:
                heldout.setdefault(a, {})[suite] = evaluate(models[a], hb)
            # the delay suites have no lag <= 8 samples by construction, so
            # the primary score is undefined there; report revised/untouched
            print(f"heldout {suite:8s} rev/untouched " + "  ".join(
                f"{a}:{heldout[a][suite]['revised']:5.1f}/"
                f"{heldout[a][suite]['untouched']:5.1f}" for a in arms), flush=True)

    out = dict(
        provenance=dict(
            repo_sha=git_sha(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            mixer_version=MIXER_VERSION, torch=torch.__version__,
            device=str(dev),
            gpu=(torch.cuda.get_device_name(dev) if dev.type == "cuda" else None),
            dtype="float32", autocast=False,
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=torch.backends.cudnn.allow_tf32),
        config={k: v for k, v in vars(args).items()},
        instances=[dict(label=lab, arm=arm, lr=lr, writer=w)
                   for lab, arm, lr, w in inst],
        unused_parameters=unused, heldout=heldout,
        vocab=vocab, arms=arms, curves=curves,
        final={a: curves[a][-1] for a in arms},
        learned={a: leaves_of(models[a]) for a in arms},
        wall_seconds=time.time() - t0)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", args.out)
    return out


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=501)
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--bsz", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    for k, v in MODEL_DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    # production task configuration (v1 report section 7.1), NOT the v1 CLI
    # defaults, which were never used for a reported run
    for k, v in PRODUCTION_TASK.items():
        p.add_argument("--" + k.replace("_", "-"), type=int, default=v)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-bsz", type=int, default=32)
    p.add_argument("--eval-batches", type=int, default=24)
    p.add_argument("--spec", default="", help="JSON list of arm instances")
    p.add_argument("--heldout", action="store_true",
                   help="evaluate the held-out suites at the end")
    p.add_argument("--heldout-batches", type=int, default=8)
    p.add_argument("--experiment", default="v2-rerun",
                   help="recorded label; aggregate.py filters on it")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="")
    return p


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
