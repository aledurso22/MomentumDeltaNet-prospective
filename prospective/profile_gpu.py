"""Measure, on one GPU, what one seed of the lockstep wave costs.

Builds the models exactly as `train.py` does (the arms must be selected
explicitly with --spec or --arms; there is no default set; same process, all
resident, FP32, reference backend, production task at batch 16), warms up,
then for each arm separately measures synchronised wall time of

    forward   (model + loss)
    backward
    update    (grad clip + finite check + AdamW step + leaf projection),
              the same `update_step` path train.py uses

and the peak allocated / reserved GPU memory during that arm's steps. Then
times one full evaluation (the fixed 24 x 32 eval set) per arm and the CPU
batch generation. The wave estimate is assembled from these measurements:

    per_seed = steps * (batch_gen + sum_arms(fwd + bwd + upd))
             + n_evals * sum_arms(eval)

plus the measured checkpoint write (one per --ckpt-every steps). Nothing is
assumed from earlier hardware. `--budget-minutes` only labels whether the
measured estimate fits; it changes nothing.

    python -m prospective.profile_gpu --spec prospective/specs/pilot/pilot.json \
        --steps 2000 --eval-every 100 --ckpt-every 200 --out profile.json
"""

import json
import os
import statistics as st
import tempfile
import time

import torch

from .model import build, make_optimizer
from .projection import state_numbers
from .task import make_batch
from .train import (evaluate, git_sha, instances, loss_of, model_kwargs,
                    parser, save_checkpoint, set_fp32, task_kwargs, to_dev)


def sync_time(fn, dev):
    torch.cuda.synchronize(dev)
    t = time.perf_counter()
    r = fn()
    torch.cuda.synchronize(dev)
    return time.perf_counter() - t, r


def main():
    p = parser()
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--measure", type=int, default=50)
    p.add_argument("--eval-reps", type=int, default=3)
    p.add_argument("--budget-minutes", type=float, default=60.0)
    args = p.parse_args()
    if not (args.spec or args.arms):
        raise SystemExit("select the arms explicitly (--spec or --arms); "
                         "there is no default arm set")
    set_fp32()
    dev = torch.device(args.device)
    assert dev.type == "cuda", "profile on the GPU the wave will use"
    MiB = 2 ** 20

    gen = torch.Generator().manual_seed(args.seed + 10_000)
    task = task_kwargs(args)
    vocab = make_batch(2, **task, generator=gen)["vocab"]
    eval_gen = torch.Generator().manual_seed(args.seed + 99_000)
    eval_batches = [to_dev(make_batch(args.eval_bsz, **task,
                                      generator=eval_gen), dev)
                    for _ in range(args.eval_batches)]
    inst = instances(args)
    arms = [lab for lab, _, _, _ in inst]
    arm_of = {lab: arm for lab, arm, _, _ in inst}
    models, opts = {}, {}
    for lab, arm, lr, wcfg in inst:
        models[lab] = build(arm, vocab, seed=args.seed, writer_cfg=wcfg,
                            **model_kwargs(args)).to(dev)
        opts[lab] = make_optimizer(models[lab], lr, args.weight_decay)
    n_params = {a: sum(x.numel() for x in m.parameters())
                for a, m in models.items()}

    def step(a, b, rec=None):
        m, o = models[a], opts[a]
        tf, (loss, _) = sync_time(lambda: loss_of(m, b), dev)
        o.zero_grad(set_to_none=True)
        tb, _ = sync_time(loss.backward, dev)

        def upd():           # as train.update_step, after its backward
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), args.clip)
            if bool(torch.isfinite(gn)):
                o.step()
                m.project_()
        tu, _ = sync_time(upd, dev)
        if rec is not None:
            rec["fwd"].append(tf); rec["bwd"].append(tb); rec["upd"].append(tu)

    # batch generation (CPU + H2D), measured on the same generator
    tgen = []
    for _ in range(args.warmup + args.measure):
        t, b = sync_time(lambda: to_dev(make_batch(args.bsz, **task,
                                                   generator=gen), dev), dev)
        tgen.append(t)
    tgen = tgen[args.warmup:]

    print(f"warm-up: {args.warmup} lockstep steps")
    for _ in range(args.warmup):
        b = to_dev(make_batch(args.bsz, **task, generator=gen), dev)
        for a in arms:
            step(a, b)
    torch.cuda.synchronize(dev)
    resident = torch.cuda.memory_allocated(dev)

    rows = {}
    batches = [to_dev(make_batch(args.bsz, **task, generator=gen), dev)
               for _ in range(args.measure)]
    for a in arms:
        torch.cuda.reset_peak_memory_stats(dev)
        rec = dict(fwd=[], bwd=[], upd=[])
        for b in batches:
            step(a, b, rec)
        peak_alloc = torch.cuda.max_memory_allocated(dev)
        peak_res = torch.cuda.max_memory_reserved(dev)
        tev = [sync_time(lambda: evaluate(models[a], eval_batches), dev)[0]
               for _ in range(args.eval_reps)]
        rows[a] = dict(
            fwd_ms=1e3 * st.median(rec["fwd"]), bwd_ms=1e3 * st.median(rec["bwd"]),
            upd_ms=1e3 * st.median(rec["upd"]),
            step_ms=1e3 * st.median([f + b_ + u for f, b_, u in
                                     zip(rec["fwd"], rec["bwd"], rec["upd"])]),
            eval_s=st.median(tev), peak_alloc_mib=peak_alloc / MiB,
            peak_reserved_mib=peak_res / MiB, params=n_params[a],
            # persistent recurrent state, separate from training memory
            state_bytes_per_sequence=4 * args.n_layers * args.n_heads
            * state_numbers(arm_of[a], args.head_dim,
                            int(args.head_dim * args.expand_v)))

    # one checkpoint write of everything train.py saves (models + optimisers)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ckpt.pt")
        tck = []
        for _ in range(3):
            t = time.perf_counter()
            save_checkpoint(path, dict(
                models={a: models[a].state_dict() for a in arms},
                opts={a: opts[a].state_dict() for a in arms}))
            tck.append(time.perf_counter() - t)
        ckpt_mib = os.path.getsize(path) / MiB
    ckpt_s = st.median(tck)

    gen_ms = 1e3 * st.median(tgen)
    n_evals = args.steps // args.eval_every + (args.steps % args.eval_every != 0)
    n_ckpt = (args.steps // args.ckpt_every + (args.steps % args.ckpt_every != 0)
              if args.ckpt_every else 0)
    sum_step_s = sum(r["step_ms"] for r in rows.values()) / 1e3
    sum_eval_s = sum(r["eval_s"] for r in rows.values())
    per_seed_s = (args.steps * (gen_ms / 1e3 + sum_step_s) + n_evals * sum_eval_s
                  + n_ckpt * ckpt_s)
    fits = per_seed_s / 60 <= args.budget_minutes
    estimates = {"profiled": dict(
        arms=arms, lockstep_step_ms=1e3 * sum_step_s + gen_ms,
        eval_round_s=sum_eval_s, n_evals=n_evals, checkpoint_s=ckpt_s,
        n_checkpoints=n_ckpt, per_seed_minutes=per_seed_s / 60,
        budget_minutes=args.budget_minutes, fits_budget=fits,
        steps_in_budget=int(max(0.0, 60 * args.budget_minutes
                                - n_evals * sum_eval_s - n_ckpt * ckpt_s)
                            // (gen_ms / 1e3 + sum_step_s)))}

    w = max(len(a) for a in arms)
    print(f"\n{torch.cuda.get_device_name(dev)}, bsz {args.bsz}, seq "
          f"{2 * (args.n_pairs + args.n_revisions + args.n_queries)}, FP32, "
          f"median of {args.measure} steps after {args.warmup} warm-up")
    print(f"{'arm':>{w}}  {'fwd ms':>8} {'bwd ms':>8} {'upd ms':>7} "
          f"{'step ms':>8} {'eval s':>7} {'peak alloc':>11} {'peak res':>9} "
          f"{'params':>8} {'state/seq':>10}")
    for a, r in rows.items():
        print(f"{a:>{w}}  {r['fwd_ms']:8.1f} {r['bwd_ms']:8.1f} "
              f"{r['upd_ms']:7.1f} {r['step_ms']:8.1f} {r['eval_s']:7.2f} "
              f"{r['peak_alloc_mib']:8.0f} MiB {r['peak_reserved_mib']:6.0f} MiB"
              f" {r['params']:8d} {r['state_bytes_per_sequence'] / 1024:7.0f} KiB")
    print(f"\nbatch generation      {gen_ms:.2f} ms / step")
    print(f"all {len(arms)} profiled models + AdamW state: {resident / MiB:.0f} MiB resident")
    print(f"one lockstep step     {1e3 * sum_step_s + gen_ms:.0f} ms")
    print(f"one eval round        {sum_eval_s:.1f} s  x {n_evals}")
    print(f"checkpoint write      {ckpt_s:.2f} s ({ckpt_mib:.1f} MiB) x {n_ckpt}")
    e = estimates["profiled"]
    print(f"=> one seed, {args.steps} steps, {len(arms)} arms in lockstep: "
          f"{e['per_seed_minutes']:.1f} min on this GPU; budget "
          f"{args.budget_minutes:.0f} min -> {'FITS' if fits else 'DOES NOT FIT'}"
          f" (about {e['steps_in_budget']} steps would)")
    print("(one seed per GPU, seeds in parallel; assumes no CPU contention "
          "between the co-resident processes)")

    out = dict(gpu=torch.cuda.get_device_name(dev), torch=torch.__version__,
               repo_sha=git_sha("."), config=vars(args), arms=rows,
               batch_gen_ms=gen_ms, resident_mib=resident / MiB,
               lockstep_step_ms=1e3 * sum_step_s + gen_ms,
               eval_round_s=sum_eval_s, n_evals=n_evals,
               per_seed_minutes=per_seed_s / 60, checkpoint_mib=ckpt_mib,
               estimates=estimates)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
