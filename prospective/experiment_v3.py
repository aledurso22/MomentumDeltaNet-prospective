"""v3: joint memory-motion writer vs controls. Protocol, specs and analysis.

Question: can a fast memory that estimates both its current contents and
their ongoing change, with separate corrections derived from accumulated
information, improve fast selective revision without damaging untouched
associations?

Everything that decides the confirmation result is fixed here BEFORE any v3
run: arms, primary score, non-inferiority margin, development grids with
equal budgets, the selection rule, the confirmation seeds and the analysis.
The development phase may only pick one configuration per arm from its
grid; the result is written to a locked protocol file (with the commit) that
the confirmation specs are built from.

    python -m prospective.experiment_v3 dev-specs   OUT_DIR
    python -m prospective.experiment_v3 select      'DEV_RUNS/*/results.json' LOCKED.json
    python -m prospective.experiment_v3 confirm-spec LOCKED.json OUT.json
    python -m prospective.experiment_v3 analyse     LOCKED.json 'CONFIRM_RUNS/*/results.json'
"""

import glob
import json
import math
import os
import random
import statistics as st
import subprocess
import sys

EXPERIMENT = "v3-memory-motion"

#: the six arms, in reporting order
ARMS = ("gated", "native", "nesterov", "rls", "geometric", "joint")
LABELS = {
    "gated": "momentum-off gated delta (common frontend)",
    "native": "native MDN",
    "nesterov": "Nesterov lookahead writer (v2, verified)",
    "rls": "state-only full key-covariance writer (exp.-weighted RLS)",
    "geometric": "fixed-gain geometric inertial writer",
    "joint": "joint memory-motion writer (constant-velocity Kalman)",
}
#: main comparisons; the others explain whether simpler mechanisms suffice
MAIN_CONTRASTS = (("joint", "native"), ("joint", "rls"))
SECONDARY_CONTRASTS = (("joint", "gated"), ("joint", "nesterov"),
                       ("joint", "geometric"), ("rls", "native"))

PRIMARY = "primary"           # 0.5 * (rev_lag_le8 + untouched), task.py
NONINFERIORITY = ("untouched", 1.0)    # metric, margin in percentage points
REPORTED = ("primary", "rev_lag_le8", "untouched", "revised", "overall",
            "rev_lag_01_08", "rev_lag_09_16", "rev_lag_17_24", "rev_lag_25_32",
            "rev_lag_33_up", "legacy_immediate_revised")
TAIL = 5                      # per-seed score: mean of the final 5 evals
ESCAPE = 5.0                  # a run "escaped" if overall > 5 % (chance 3.1 %)

TRAIN = dict(steps=4000, bsz=16, weight_decay=0.01, clip=1.0, eval_every=200,
             eval_batches=24, eval_bsz=32, d_model=128, n_layers=2, n_heads=4,
             head_dim=32, expand_v=2)   # task: production defaults (52 tokens)

DEV_SEEDS = (9001, 9002)
CONFIRM_SEEDS = tuple(range(601, 609))   # fresh: not v2 (501-508), not dev

# ---- development grids: EQUAL budget, 6 configurations per arm --------------
#: arms with no algorithm hyperparameter spend the budget on the learning rate
LR6 = (7e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3)
LR2 = (1e-3, 3e-3)


def _grid():
    g = {a: [dict(lr=lr) for lr in LR6] for a in ("gated", "native", "nesterov")}
    g["rls"] = [dict(lr=lr, writer=dict(rho=r)) for lr in LR2
                for r in (1.0, 0.98, 0.9)]
    g["joint"] = [dict(lr=lr, writer=dict(rho=r, T0=t0)) for lr in LR2
                  for r, t0 in ((1.0, 4.0), (0.98, 4.0), (0.98, 16.0))]
    # (A, G) = (0.05, 1) is the documented reference; the others, and any
    # information forgetting (lam < 1), are later tuning, labelled as such
    g["geometric"] = [dict(lr=lr, writer=dict(A=a, G=gg)) for lr in LR2
                      for a, gg in ((0.05, 1.0), (0.0125, 1.0), (0.05, 0.5))]
    return g


DEV_GRID = _grid()
assert all(len(v) == 6 for v in DEV_GRID.values()), "equal tuning budgets"


def label(arm, i):
    return f"{arm}#{i}"


# ---------------------------------------------------------------- helpers
def tail(curve, metric, n=TAIL):
    vals = [c[metric] for c in curve[-n:] if c.get(metric) == c.get(metric)]
    return st.fmean(vals) if vals else float("nan")


def sign_p(wins, n):
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2.0 ** n


def bootstrap_ci(d, reps=10_000, seed=12345, level=0.95):
    r = random.Random(seed)
    means = sorted(st.fmean(r.choices(d, k=len(d))) for _ in range(reps))
    lo = means[int((1 - level) / 2 * reps)]
    hi = means[int((1 + level) / 2 * reps) - 1]
    return lo, hi


def one_sided_lower(d, reps=10_000, seed=12345, level=0.95):
    r = random.Random(seed)
    means = sorted(st.fmean(r.choices(d, k=len(d))) for _ in range(reps))
    return means[int((1 - level) * reps)]


def load(pattern):
    runs = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as fh:
            r = json.load(fh)
        if r.get("config", {}).get("experiment") == EXPERIMENT:
            runs.append((p, r))
    return runs


# ---------------------------------------------------------------- commands
def dev_specs(out_dir):
    """One spec per arm: its 6 configurations train in lockstep, per seed."""
    os.makedirs(out_dir, exist_ok=True)
    for arm, grid in DEV_GRID.items():
        spec = [dict(label=label(arm, i), arm=arm, **cfg) for i, cfg in enumerate(grid)]
        with open(os.path.join(out_dir, f"dev_{arm}.json"), "w") as fh:
            json.dump(spec, fh, indent=1)
    print(f"wrote {len(DEV_GRID)} dev specs to {out_dir}; run each on seeds {DEV_SEEDS}")


def select(pattern, locked_path):
    """Per arm: the configuration with the highest mean (over dev seeds) tail
    primary score; ties within 1e-9 go to grid order. All runs are listed."""
    runs = load(pattern)
    scores = {}
    for path, r in runs:
        seed = r["config"]["seed"]
        if seed not in DEV_SEEDS:
            print(f"skip {path}: seed {seed} is not a dev seed")
            continue
        for lab in r["arms"]:
            arm, i = lab.split("#")
            scores.setdefault(arm, {}).setdefault(int(i), {})[seed] = tail(
                r["curves"][lab], PRIMARY)
    chosen = {}
    for arm in ARMS:
        per = scores.get(arm, {})
        complete = {i: s for i, s in per.items() if set(s) == set(DEV_SEEDS)}
        if len(complete) != len(DEV_GRID[arm]):
            raise SystemExit(f"{arm}: {len(complete)}/{len(DEV_GRID[arm])} configs "
                             f"have all dev seeds; refusing to lock")
        means = {i: st.fmean(s.values()) for i, s in complete.items()}
        best = max(means.values())
        i = min(i for i, m in means.items() if m >= best - 1e-9)
        chosen[arm] = dict(index=i, **DEV_GRID[arm][i], dev_primary=means)
        print(f"{arm:>10}: config #{i} {DEV_GRID[arm][i]}  "
              + " ".join(f"#{j}={m:.2f}" for j, m in sorted(means.items())))
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    locked = dict(experiment=EXPERIMENT, locked_at_commit=sha, arms=chosen,
                  train=TRAIN, confirm_seeds=CONFIRM_SEEDS,
                  main_contrasts=MAIN_CONTRASTS, primary=PRIMARY,
                  noninferiority=NONINFERIORITY)
    with open(locked_path, "w") as fh:
        json.dump(locked, fh, indent=1)
    print("locked ->", locked_path)


def confirm_spec(locked_path, out_path):
    with open(locked_path) as fh:
        locked = json.load(fh)
    spec = [dict(label=a, arm=a, lr=locked["arms"][a]["lr"],
                 writer=locked["arms"][a].get("writer")) for a in ARMS]
    with open(out_path, "w") as fh:
        json.dump(spec, fh, indent=1)
    print("wrote", out_path)


def analyse(locked_path, pattern):
    with open(locked_path) as fh:
        locked = json.load(fh)
    runs = {r["config"]["seed"]: (p, r) for p, r in load(pattern)
            if r["config"]["seed"] in locked["confirm_seeds"]}
    missing = sorted(set(locked["confirm_seeds"]) - set(runs))
    print(f"{len(runs)} confirmation seeds; missing (failed/killed, reported): {missing}")
    seeds = sorted(runs)
    val = {m: {a: [tail(runs[s][1]["curves"][a], m) for s in seeds] for a in ARMS}
           for m in REPORTED}
    for m in REPORTED:
        print(f"\n--- {m}")
        for a in ARMS:
            v = val[m][a]
            print(f"{a:>10} " + " ".join(f"{x:6.2f}" for x in v) + f"  mean {st.fmean(v):6.2f}")
        ns = [runs[s][1]["curves"][ARMS[0]][-1].get(f"n_{m}") for s in seeds]
        if all(n is not None for n in ns):
            print(f"{'n':>10} " + " ".join(f"{n:6d}" for n in ns))
    print("\n--- stalled runs (overall <= ESCAPE), kept in every statistic")
    for a in ARMS:
        stalled = [s for s, x in zip(seeds, val["overall"][a]) if not x > ESCAPE]
        print(f"{a:>10}: {stalled}")
    print("\n--- contrasts on the primary score; untouched non-inferiority "
          f"margin {NONINFERIORITY[1]} pp (one-sided 95 % bootstrap lower bound)")
    for tag, pairs in (("MAIN", MAIN_CONTRASTS), ("secondary", SECONDARY_CONTRASTS)):
        for a, b in pairs:
            d = [x - y for x, y in zip(val[PRIMARY][a], val[PRIMARY][b])]
            du = [x - y for x, y in zip(val["untouched"][a], val["untouched"][b])]
            wins = sum(x > 0 for x in d)
            lo, hi = bootstrap_ci(d)
            lb_u = one_sided_lower(du)
            print(f"{tag:>9} {a} - {b}: mean {st.fmean(d):+6.2f} [{lo:+.2f}, {hi:+.2f}]"
                  f"  wins {wins}/{len(d)} p={sign_p(wins, len(d)):.4f}"
                  f"  | untouched mean {st.fmean(du):+.2f}, lower {lb_u:+.2f} -> "
                  f"{'non-inferior' if lb_u > -NONINFERIORITY[1] else 'NOT shown non-inferior'}")
    print("\n--- held-out suites (final checkpoint). The delay suites have no "
          "lag <= 8 queries, so the primary score is undefined there; reported: "
          "all-revised / untouched / overall, mean over seeds, with counts")
    first = runs[seeds[0]][1].get("heldout", {})
    for suite in sorted(next(iter(first.values()), {})):
        print(f"  {suite}  (n revised {first[ARMS[0]][suite]['n_revised']}, "
              f"n untouched {first[ARMS[0]][suite]['n_untouched']} per seed)")
        for a in ARMS:
            cells = [st.fmean(runs[s][1]["heldout"][a][suite][m] for s in seeds)
                     for m in ("revised", "untouched", "overall")]
            print(f"    {a:>10}: " + " / ".join(f"{c:6.2f}" for c in cells))

def main():
    cmd, *rest = sys.argv[1:] or ["help"]
    {"dev-specs": dev_specs, "select": select, "confirm-spec": confirm_spec,
     "analyse": analyse}.get(cmd, lambda *a: print(__doc__))(*rest)


if __name__ == "__main__":
    main()
