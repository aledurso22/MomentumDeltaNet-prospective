"""Recover per-seed curves from train logs, including runs that never wrote
results.json.

Jobs 67448 and 67449 were killed by node cleanup a few seconds before their
final write, so their results.json does not exist -- but every eval they
reached is in the log. This reads the printed lines back. Only
immediate_revised is recoverable this way; the other four metrics live in
results.json alone.
"""

import argparse
import re
import statistics
import sys

EVAL = re.compile(r"^\[\s*(\d+)\]\s+([\d.]+)s\s+imm_rev\s+(.*)$")
ARM = re.compile(r"(\w+):\s*([\d.]+|nan)")
LEAF = re.compile(r"^\s+(\w+)\s+((?:\w+=[-\d.]+\.\.[-\d.]+\s*)+)$")


def parse(path):
    curve, leaves = {}, {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = EVAL.match(line)
            if m:
                step = int(m.group(1))
                for arm, val in ARM.findall(m.group(3)):
                    curve.setdefault(arm, []).append((step, float(val)))
                continue
            m = LEAF.match(line.rstrip("\n"))
            if m:
                leaves[m.group(1)] = m.group(2).strip()
    return curve, leaves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--tail", type=int, default=3,
                    help="how many final evals to average per seed")
    a = ap.parse_args()

    per_seed, arms = {}, []
    for path in a.logs:
        curve, leaves = parse(path)
        if not curve:
            print(f"(no eval lines in {path})", file=sys.stderr)
            continue
        arms = arms or list(curve)
        last = max(s for s, _ in next(iter(curve.values())))
        per_seed[path] = dict(
            last_step=last, leaves=leaves,
            tail={arm: statistics.fmean(v for _, v in pts[-a.tail:])
                  for arm, pts in curve.items()})

    if not per_seed:
        return
    w = max(len(x) for x in arms)
    names = [p.split("/")[-1] for p in per_seed]
    print(f"immediate_revised, mean of the last {a.tail} evals\n")
    print(" " * (w + 2) + "  ".join(f"{n[-18:]:>18}" for n in names)
          + "   mean    n_escaped")
    for arm in arms:
        vals = [d["tail"][arm] for d in per_seed.values()]
        esc = sum(1 for v in vals if v > 5.0)
        print(f"{arm:>{w}}  " + "  ".join(f"{v:18.2f}" for v in vals)
              + f"  {statistics.fmean(vals):6.2f}   {esc}/{len(vals)}")
    print("\nlast step reached: " + ", ".join(
        f"{n[-18:]}={d['last_step']}" for n, d in zip(names, per_seed.values())))
    print("\nfinal leaves per seed:")
    for n, d in zip(names, per_seed.values()):
        print(f"  {n[-18:]}")
        for arm, txt in d["leaves"].items():
            print(f"    {arm:>12}  {txt}")


if __name__ == "__main__":
    main()
