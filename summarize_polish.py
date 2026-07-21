# Merge polish_results_shard*.jsonl and print the polish-test verdict.
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for p in sorted(glob.glob(os.path.join(HERE, "polish_results_shard*.jsonl"))):
    rows += [json.loads(l) for l in open(p)]
by = {(r["seed"], r["func"], r["algo"]): r for r in rows}
seeds = sorted({r["seed"] for r in rows})
NAMES = ["SHADE", "P17", "P17d", "P18"]
FUNCS = [2, 3, 5, 9, 10, 12]
CHECKS = [125, 500, 1250, 2500]
FES = {125: "50k", 500: "200k", 1250: "500k", 2500: "1M"}


def val(s, f, a, g):
    r = by.get((s, f, a))
    return None if r is None else r["traj"].get(str(g), r["final"])


print(f"{len(rows)} runs, {len(seeds)} seeds\n")
print("MEDIANS (polish battlefield: f2 regression + f3/f5/f9/f10/f12 polish)")
for f in FUNCS:
    print(f"\nf{f}:")
    for g in CHECKS:
        meds = {n: float(np.median([val(s, f, n, g) for s in seeds
                                    if val(s, f, n, g) is not None])) for n in NAMES}
        best = min(meds.values())
        print(f"  {FES[g]:>5}: " + "  ".join(
            f"{n}={meds[n]:>9.2f}{'*' if meds[n] == best else ' '}" for n in NAMES))

print("\nPAIRED PER-SEED WINS at 1M FES")
for a, b in [("P17d", "P17"), ("P18", "P17"), ("P17d", "SHADE"),
             ("P18", "SHADE"), ("P17d", "P18")]:
    tw = tl = 0
    cells = []
    for f in FUNCS:
        w = l = 0
        for s in seeds:
            va, vb = val(s, f, a, 2500), val(s, f, b, 2500)
            if va is None or vb is None:
                continue
            if va < vb:
                w += 1
            elif vb < va:
                l += 1
        tw += w
        tl += l
        cells.append(f"f{f}:{w}-{l}")
    print(f"  {a} vs {b}: {tw}-{tl}   " + " ".join(cells))
