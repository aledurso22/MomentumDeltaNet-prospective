"""Train all six arms together, on identical data, from identical init.

Every arm sees the SAME batches in the SAME order at the same step, and starts
from a shared-seed backbone (see `model.build`). Arms are stepped in lockstep
inside one loop rather than in separate runs, so a difference between two arms
cannot come from data order, initialization or schedule.

Writes `results.json` with per-seed, per-arm curves and the provenance of the
tree it ran from. No means are reported here; aggregation is a separate step.
"""

import argparse
import json
import os
import subprocess
import time

import torch
import torch.nn.functional as F

from .model import build
from .rules import ARMS
from .task import IGNORE, accuracy, make_batch


def git_sha(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def loss_of(model, batch):
    logits = model(batch["x"])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           batch["y"].reshape(-1), ignore_index=IGNORE), logits


def evaluate(model, batches):
    model.eval()
    tot = {}
    with torch.no_grad():
        for b in batches:
            _, logits = loss_of(model, b)
            for k, v in accuracy(logits, b).items():
                tot.setdefault(k, []).append(v)
    model.train()
    return {k: sum(v) / len(v) for k, v in tot.items()}


def run(args):
    dev = torch.device(args.device)
    gen = torch.Generator().manual_seed(args.seed + 10_000)
    task = dict(n_pairs=args.n_pairs, n_revisions=args.n_revisions,
                n_queries=args.n_queries, n_keys=args.n_keys,
                n_values=args.n_values)
    probe = make_batch(2, **task, generator=gen)
    vocab = probe["vocab"]

    def batch(bs):
        b = make_batch(bs, **task, generator=gen)
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}

    # a FIXED evaluation set, identical for every arm
    eval_gen = torch.Generator().manual_seed(args.seed + 99_000)
    eval_batches = [{k: (v.to(dev) if torch.is_tensor(v) else v)
                     for k, v in make_batch(args.eval_bsz, **task,
                                            generator=eval_gen).items()}
                    for _ in range(args.eval_batches)]

    arms = args.arms or list(ARMS)
    models, opts = {}, {}
    for a in arms:
        m = build(a, vocab, seed=args.seed, d_model=args.d_model,
                  n_heads=args.n_heads, n_layers=args.n_layers).to(dev)
        models[a] = m
        opts[a] = torch.optim.AdamW(m.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)

    curves = {a: [] for a in arms}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        b = batch(args.bsz)                       # ONE batch, all arms see it
        for a in arms:
            loss, _ = loss_of(models[a], b)
            opts[a].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(models[a].parameters(), args.clip)
            opts[a].step()
            models[a].project_()          # leaves are stored directly
        if step % args.eval_every == 0 or step == args.steps:
            for a in arms:
                acc = evaluate(models[a], eval_batches)
                acc["step"] = step
                curves[a].append(acc)
            line = "  ".join(f"{a}:{curves[a][-1]['immediate_revised']:5.1f}"
                             for a in arms)
            print(f"[{step:5d}] {time.time()-t0:7.1f}s  imm_rev  {line}", flush=True)
            # the leaves, so a short probe shows whether they are moving at all
            for a in arms:
                leaves = {n.split(".")[-1]: p.detach()
                          for n, p in models[a].named_parameters()
                          if n.split(".")[-1] in ("fil_M", "fil_gamma",
                                                  "fil_T", "nu")}
                if leaves:
                    txt = "  ".join(
                        f"{n}={v.min().item():.3f}..{v.max().item():.3f}"
                        for n, v in sorted(leaves.items()))
                    print(f"        {a:>12}  {txt}", flush=True)

    out = dict(
        provenance=dict(repo_sha=git_sha(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), torch=torch.__version__,
            device=str(dev)),
        config={k: v for k, v in vars(args).items()},
        vocab=vocab, arms=arms, curves=curves,
        final={a: curves[a][-1] for a in arms},
        learned={a: {n: p.detach().cpu().tolist()
                     for n, p in models[a].named_parameters()
                     if n.split(".")[-1] in ("fil_M", "fil_gamma", "fil_T",
                                             "nu")} for a in arms},
        wall_seconds=time.time() - t0)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", args.out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=501)
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--bsz", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-pairs", type=int, default=16)
    p.add_argument("--n-revisions", type=int, default=6)
    p.add_argument("--n-queries", type=int, default=12)
    p.add_argument("--n-keys", type=int, default=64)
    p.add_argument("--n-values", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-bsz", type=int, default=32)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="")
    run(p.parse_args())


if __name__ == "__main__":
    main()
