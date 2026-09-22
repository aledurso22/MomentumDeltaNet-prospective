"""Read results.json and print what the terminal line cannot show.

Two things the per-step print hides. First, the last eval is ONE noisy draw:
this averages the final `--tail` evals instead. Second, whether each arm's own
parameters actually moved off their native start -- an arm whose leaves never
left the boundary is running as native under a different name.
"""

import argparse
import json
import math

METRICS = ("overall", "revised", "untouched", "immediate_revised",
           "later_revised")
NATIVE_START = {"raw_nu": "nu=1 (sigmoid 8.0)", "raw_T": "T=h",
                "raw_M": "M~0", "raw_g": "gamma~0"}


def softplus(x):
    return math.log1p(math.exp(x)) if x < 30 else x


def tail_mean(curve, tail, key):
    vals = [c[key] for c in curve[-tail:] if key in c and c[key] == c[key]]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def tail_sd(curve, tail, key):
    vals = [c[key] for c in curve[-tail:] if key in c and c[key] == c[key]]
    if len(vals) < 2:
        return float("nan")
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--tail", type=int, default=10,
                    help="how many final evals to average")
    a = ap.parse_args()
    r = json.load(open(a.results))
    arms, curves = r["arms"], r["curves"]
    present = [m for m in METRICS if m in curves[arms[0]][-1]]

    print(f"seed {r['config']['seed']}  steps {r['config']['steps']}  "
          f"sha {r['provenance']['repo_sha'][:9]}  "
          f"{r['wall_seconds']/60:.1f} min")
    print(f"mean of the last {a.tail} evals, +/- sd across them\n")
    w = max(len(x) for x in arms)
    print(" " * (w + 2) + "  ".join(f"{m:>19}" for m in present))
    for arm in arms:
        cells = []
        for m in present:
            cells.append(f"{tail_mean(curves[arm], a.tail, m):8.2f} "
                         f"+/-{tail_sd(curves[arm], a.tail, m):5.2f}")
        print(f"{arm:>{w}}  " + "  ".join(f"{c:>19}" for c in cells))

    print("\nlearned leaves (softplus/sigmoid applied; native start in "
          "brackets):")
    for arm in arms:
        leaves = r.get("learned", {}).get(arm, {})
        if not leaves:
            print(f"  {arm:>{w}}  (none)")
            continue
        parts = []
        for name, vals in sorted(leaves.items()):
            leaf = name.split(".")[-1]
            flat = vals if isinstance(vals, list) else [vals]
            flat = [v for sub in flat for v in
                    (sub if isinstance(sub, list) else [sub])]
            if leaf == "raw_nu":
                shown = [1 / (1 + math.exp(-v)) for v in flat]
            else:
                shown = [softplus(v) for v in flat]
            lo, hi = min(shown), max(shown)
            parts.append(f"{name}={lo:.4f}..{hi:.4f}")
        print(f"  {arm:>{w}}  " + "  ".join(parts))
        print(f"  {'':>{w}}  [{ '; '.join(NATIVE_START[k] for k in sorted(
            {n.split('.')[-1] for n in leaves}) if k in NATIVE_START) }]")


if __name__ == "__main__":
    main()
