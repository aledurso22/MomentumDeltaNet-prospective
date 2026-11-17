"""v4 pilot: historical prediction geometry and learned inertia, trained end
to end. Supersedes the v3 plan (whose code and reports are kept).

Questions:
  1. does historical prediction geometry help with learned representations?
     geometric_projection vs euclidean_projection (same rule, b differs)
  2. does letting the inertial coefficients learn add anything beyond it?
     geometric_motion_learned vs geometric_projection (same b, motion added)
  and, for reference, every projection arm vs native MDN.

A small PILOT: four fresh paired seeds, no hyperparameter sweep (the
existing AdamW configuration), 2,000 updates. Contrasts are reported per
seed and descriptively; nothing here is a confirmatory test. A seed in which
either arm of a pair stays near chance (overall accuracy <= ESCAPE, chance
3.1 %) or failed is inconclusive for that pair and is listed, not dropped
silently. A motion-arm difference is NOT attributed to inertia where the
learned motion is cancelled (A - d ~ 0, which includes A ~ 0).

Single source of the launch configuration (the shell scripts call `flags`
and `seeds`, so the pilot and its profile cannot drift apart):

    python -m prospective.experiment_pilot spec    prospective/specs/pilot/pilot.json
    python -m prospective.experiment_pilot flags   # train.py / profile_gpu.py flags
    python -m prospective.experiment_pilot seeds
    python -m prospective.experiment_pilot analyse 'RUNS/*/results.json'
"""

import glob
import json
import os
import statistics as st
import sys

EXPERIMENT = "v4-geometry-pilot"
ARMS = ("native", "euclidean_projection", "geometric_projection",
        "geometric_motion_learned")
LABELS = {
    "native": "official MDN recurrence, learned gates",
    "euclidean_projection": "normalised Euclidean correction, no motion",
    "geometric_projection": "normalised historical (C^-1) correction, no motion",
    "geometric_motion_learned": "historical correction + learned inertia (A, r)",
}
CONTRASTS = (
    ("geometry", "geometric_projection", "euclidean_projection"),
    ("learned inertia beyond geometry", "geometric_motion_learned",
     "geometric_projection"),
    ("vs native", "euclidean_projection", "native"),
    ("vs native", "geometric_projection", "native"),
    ("vs native", "geometric_motion_learned", "native"),
)
SEEDS = (701, 702, 703, 704)      # fresh: not v2 501-508, v3 601-608 / 9001-2
SPEC_PATH = os.path.join(os.path.dirname(__file__), "specs", "pilot", "pilot.json")
#: everything else is train.py's existing default (width 128, 2 layers, 4
#: heads, dk 32, dv 64, batch 16, AdamW lr 3e-3, wd .01, clip 1, FP32, the
#: 52-token production task with read-after-update queries)
TRAIN_FLAGS = dict(steps=2000, eval_every=100, ckpt_every=200,
                   experiment=EXPERIMENT)
PRIMARY = "primary"               # 0.5 * (rev_lag_le8 + untouched), task.py
REPORTED = ("primary", "rev_lag_le8", "untouched", "revised", "overall")
TAIL = 5                          # per-seed score: mean of the final 5 evals
ESCAPE = 5.0                      # overall % at or below: near chance
MOTION_OFF = 1e-6                 # |A - d| at or below: motion cancelled
UNTOUCHED_MARGIN_PP = 1.0         # descriptive, as in the earlier protocol


def spec():
    return [dict(label=a, arm=a) for a in ARMS]   # lr: train.py default


def flags():
    return ["--spec", os.path.relpath(SPEC_PATH, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))] + [
        x for k, v in TRAIN_FLAGS.items()
        for x in ("--" + k.replace("_", "-"), str(v))]


def tail(curve, metric, n=TAIL):
    vals = [c[metric] for c in curve[-n:] if c.get(metric) == c.get(metric)]
    return st.fmean(vals) if vals else float("nan")


def load(pattern):
    runs = {}
    for p in sorted(glob.glob(pattern)):
        with open(p) as fh:
            r = json.load(fh)
        cfg = r.get("config", {})
        if cfg.get("experiment") != EXPERIMENT or cfg.get("seed") not in SEEDS:
            continue
        if tuple(r["arms"]) != ARMS:
            raise SystemExit(f"{p}: arms {r['arms']} are not the pilot's {ARMS}")
        if cfg["seed"] in runs:
            raise SystemExit(f"seed {cfg['seed']} appears twice: {p}")
        runs[cfg["seed"]] = r
    return runs


def analyse(pattern):
    runs = load(pattern)
    seeds = sorted(runs)
    print(f"{EXPERIMENT}: {len(seeds)}/{len(SEEDS)} seeds {seeds}; "
          f"missing (failed/killed, reported): {sorted(set(SEEDS) - set(runs))}")
    if not seeds:
        return
    for s in seeds:
        r = runs[s]
        last = r["curves"][ARMS[0]][-1]["step"]
        print(f"  seed {s}: last eval step {last}, wall {r['wall_seconds'] / 60:.1f} min, "
              f"gpu {r['provenance'].get('gpu')}, failed arms {r.get('failed', {}) or 'none'}")
    val = {m: {a: {s: tail(runs[s]["curves"][a], m) for s in seeds} for a in ARMS}
           for m in REPORTED}
    near = {a: [s for s in seeds if not val["overall"][a][s] > ESCAPE] for a in ARMS}
    failed = {a: [s for s in seeds if a in runs[s].get("failed", {})] for a in ARMS}

    print(f"\nper-seed tail means (last {TAIL} evals); n = samples per eval set")
    for m in REPORTED:
        n = runs[seeds[0]]["curves"][ARMS[0]][-1].get(f"n_{m}")
        print(f"--- {m}" + (f"  (n={n})" if n is not None else ""))
        for a in ARMS:
            v = [val[m][a][s] for s in seeds]
            print(f"  {a:>25} " + " ".join(f"{x:6.2f}" for x in v)
                  + f"   mean {st.fmean(v):6.2f}")
    print("\nnear chance (overall <= %.1f %%) -> inconclusive:" % ESCAPE)
    for a in ARMS:
        print(f"  {a:>25}: near chance {near[a]}  failed {failed[a]}")

    print("\n--- paired contrasts on the primary score (pilot, descriptive)")
    for tag, a, b in CONTRASTS:
        ok = [s for s in seeds if s not in near[a] + near[b] + failed[a] + failed[b]]
        d = [val[PRIMARY][a][s] - val[PRIMARY][b][s] for s in ok]
        du = [val["untouched"][a][s] - val["untouched"][b][s] for s in ok]
        d8 = [val["rev_lag_le8"][a][s] - val["rev_lag_le8"][b][s] for s in ok]
        head = f"  [{tag}] {a} - {b}: conclusive seeds {ok}"
        if not d:
            print(head + " -> none; inconclusive")
            continue
        print(head)
        print("      primary   " + " ".join(f"{x:+6.2f}" for x in d)
              + f"   mean {st.fmean(d):+6.2f}, higher in {sum(x > 0 for x in d)}/{len(d)}")
        print("      rev<=8    " + " ".join(f"{x:+6.2f}" for x in d8)
              + f"   mean {st.fmean(d8):+6.2f}")
        print("      untouched " + " ".join(f"{x:+6.2f}" for x in du)
              + f"   mean {st.fmean(du):+6.2f} (margin -{UNTOUCHED_MARGIN_PP} pp,"
              f" min {min(du):+.2f})")

    print("\n--- learned inertial coefficients, final eval (per layer, per head)")
    arm = "geometric_motion_learned"
    cancelled_all = True
    for s in seeds:
        inertia = runs[s]["curves"][arm][-1].get("inertia", {})
        for lay, c in sorted(inertia.items()):
            off = [abs(x) <= MOTION_OFF for x in c["A_minus_d"]]
            cancelled_all &= all(off)
            print(f"  seed {s} {lay}: A " + " ".join(f"{x:.4g}" for x in c["A"])
                  + " | d " + " ".join(f"{x:.4g}" for x in c["d"])
                  + " | A-d " + " ".join(f"{x:.4g}" for x in c["A_minus_d"])
                  + f" | motion cancelled in {sum(off)}/{len(off)} heads")
    if cancelled_all:
        print("  motion is cancelled (A - d ~ 0) in EVERY head of every seed: any "
              "difference from geometric_projection is NOT an inertial benefit")

    print("\n--- learning curves: primary, mean over seeds (n seeds per step)")
    steps = sorted({c["step"] for s in seeds for c in runs[s]["curves"][ARMS[0]]})
    print(f"  {'step':>6} " + " ".join(f"{a[:12]:>13}" for a in ARMS))
    for t in steps:
        cells = []
        for a in ARMS:
            v = [c[PRIMARY] for s in seeds for c in runs[s]["curves"][a]
                 if c["step"] == t]
            cells.append(f"{st.fmean(v):8.2f} (n{len(v)})")
        print(f"  {t:6d} " + " ".join(cells))


def main():
    cmd, *rest = sys.argv[1:] or ["help"]
    if cmd == "spec":
        out = rest[0] if rest else SPEC_PATH
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(spec(), fh, indent=1)
            fh.write("\n")
        print("wrote", out)
    elif cmd == "flags":
        print(" ".join(flags()))
    elif cmd == "seeds":
        print(" ".join(map(str, SEEDS)))
    elif cmd == "analyse":
        analyse(*rest)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
