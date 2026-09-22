"""Collapse the S5 15-epoch metrics.jsonl files to per-epoch curves.

Stdlib only: runs on the cluster head node, where /Users/durso is shared and
no venv is needed. Prints the schema first so a wrong field name is diagnosed
from the same run rather than another round trip.
"""

import collections
import glob
import json
import statistics as st
import sys

DEFAULT = [
    ("native",
     "/Users/durso/s5-runs/s5-three-arm-15epoch/20260919-224525/"
     "native_matched_s5/*/metrics.jsonl"),
    ("generalized",
     "/Users/durso/s5-runs/s5-two-compartment-factored/20260921-175242/"
     "generalized_prospective_s5/*/metrics.jsonl"),
]


def records(pairs):
    out = []
    for arm, pattern in pairs:
        for path in sorted(glob.glob(pattern)):
            seed = path.split("/")[-2]
            for line in open(path):
                line = line.strip()
                if line:
                    out.append((arm, seed, json.loads(line)))
    return out


def main():
    pairs = DEFAULT
    if len(sys.argv) > 2:
        pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
    rows = records(pairs)
    if not rows:
        print("no records found", file=sys.stderr)
        return
    print("KEYS:", " ".join(sorted(rows[0][2])))

    def pick(d, *needles):
        for k in d:
            low = k.lower()
            if all(n in low for n in needles):
                return k
        return None

    acc = pick(rows[0][2], "val", "acc") or pick(rows[0][2], "acc")
    loss = pick(rows[0][2], "val", "loss") or pick(rows[0][2], "loss")
    ep = pick(rows[0][2], "epoch") or "epoch"
    print(f"USING: epoch={ep!r} acc={acc!r} loss={loss!r}\n")
    if acc is None:
        return

    by = collections.defaultdict(list)
    for arm, _seed, d in rows:
        if d.get(ep) is not None:
            by[(arm, d[ep])].append((d[acc], d.get(loss)))

    print(f"{'arm':>12} {'ep':>3} {'mean':>9} {'min':>9} {'max':>9} "
          f"{'loss':>9} {'n':>2}")
    for (arm, epoch), vals in sorted(by.items(), key=lambda kv: (kv[0][0],
                                                                kv[0][1])):
        a = [v for v, _ in vals]
        l = [x for _, x in vals if isinstance(x, (int, float))]
        print(f"{arm:>12} {epoch:>3} {st.fmean(a):9.4f} {min(a):9.4f} "
              f"{max(a):9.4f} {(st.fmean(l) if l else float('nan')):9.5f} "
              f"{len(a):>2}")


if __name__ == "__main__":
    main()
