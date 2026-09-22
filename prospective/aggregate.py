"""Aggregate a wave of results.json into paired contrasts and sign tests.

Stdlib only. Selects runs by CONFIG, not by path, because
/Local/durso/mdn-bridge-runs also holds earlier runs at other task settings
(64 keys, the 800-step probe) whose numbers are not comparable. Runs that do
not match are listed and skipped, never silently averaged in.

The outcome here is bimodal -- a seed either escapes the plateau or does not --
so the headline test is the exact sign test, which assumes nothing about the
distribution. Means are reported alongside, not instead.
"""

import argparse
import glob
import json
import math
import os
import statistics as st

METRICS = ("immediate_revised", "later_revised", "revised", "untouched",
           "overall")
#: a seed counts as having escaped if it clears this, well above 1/32 chance
ESCAPE = 5.0


def sign_p(wins, n):
    """Exact one-sided binomial p at 0.5, P(X >= wins)."""
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (2.0 ** n)


def load(paths, want):
    runs, skipped = {}, []
    for path in paths:
        try:
            r = json.load(open(path))
        except Exception as exc:
            skipped.append((path, f"unreadable: {exc}"))
            continue
        cfg = r.get("config", {})
        bad = [f"{k}={cfg.get(k)!r}!={v!r}" for k, v in want.items()
               if cfg.get(k) != v]
        if bad:
            skipped.append((path, ", ".join(bad)))
            continue
        seed = cfg.get("seed")
        prev = runs.get(seed)
        if prev is None or os.path.getmtime(path) > os.path.getmtime(prev[0]):
            runs[seed] = (path, r)
    return runs, skipped


def tail_mean(curve, metric, tail):
    vals = [c[metric] for c in curve[-tail:]
            if metric in c and c[metric] == c[metric]]
    return st.fmean(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glob", nargs="?",
                    default="/Local/durso/mdn-bridge-runs/*-seed*/results.json")
    ap.add_argument("--tail", type=int, default=5)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-keys", type=int, default=24)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--ref", default="generalized")
    a = ap.parse_args()

    want = {"steps": a.steps, "n_keys": a.n_keys, "n_pairs": a.n_pairs}
    runs, skipped = load(sorted(glob.glob(a.glob)), want)
    if not runs:
        print("no matching runs")
        for p, why in skipped:
            print("  skipped", p, "--", why)
        return

    seeds = sorted(runs)
    arms = runs[seeds[0]][1]["arms"]
    print(f"{len(seeds)} seeds: {seeds}")
    print(f"config: steps={a.steps} n_keys={a.n_keys} n_pairs={a.n_pairs}, "
          f"mean of last {a.tail} evals")
    if skipped:
        print(f"skipped {len(skipped)} non-matching run(s):")
        for p, why in skipped:
            print(f"  {p.split('/')[-2]}  ({why})")
    print()

    val = {m: {arm: [tail_mean(runs[s][1]["curves"][arm], m, a.tail)
                     for s in seeds] for arm in arms} for m in METRICS}

    for m in METRICS:
        if all(x != x for arm in arms for x in val[m][arm]):
            continue
        print(f"--- {m}")
        w = max(len(x) for x in arms)
        head = "  ".join(f"{s:>7}" for s in seeds)
        print(f"{'arm':>{w}}  {head}  {'mean':>7}  escaped")
        for arm in arms:
            v = val[m][arm]
            esc = sum(1 for x in v if x == x and x > ESCAPE)
            print(f"{arm:>{w}}  " + "  ".join(f"{x:7.2f}" for x in v)
                  + f"  {st.fmean(v):7.2f}  {esc}/{len(v)}")
        print(f"\n  paired vs {a.ref}:")
        for arm in arms:
            if arm == a.ref:
                continue
            d = [x - y for x, y in zip(val[m][a.ref], val[m][arm])]
            wins = sum(1 for x in d if x > 0)
            print(f"    {a.ref} - {arm:<14} "
                  + " ".join(f"{x:+6.2f}" for x in d)
                  + f"   mean {st.fmean(d):+6.2f}"
                  + f"   {wins}/{len(d)}  p={sign_p(wins, len(d)):.4f}")
        print()

    print("--- learned leaves (final)")
    for s in seeds:
        r = runs[s][1]
        print(f"  seed {s}")
        for arm, leaves in sorted(r.get("learned", {}).items()):
            if not leaves:
                continue
            parts = []
            for name in sorted(leaves):
                flat = leaves[name]
                flat = flat if isinstance(flat, list) else [flat]
                flat = [v for sub in flat
                        for v in (sub if isinstance(sub, list) else [sub])]
                parts.append(f"{name.split('.')[-1]}="
                             f"{min(flat):.3f}..{max(flat):.3f}")
            print(f"    {arm:>12}  " + "  ".join(parts))


if __name__ == "__main__":
    main()
