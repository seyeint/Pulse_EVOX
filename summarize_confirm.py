# Merge confirm_results_shard*.jsonl and print the confirmatory analysis:
# average rank vs budget (FES-labelled), median finals, head-to-heads, and
# paired per-seed win rates with exact sign-test p-values (no scipy needed).

import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for path in sorted(glob.glob(os.path.join(HERE, "confirm_results_shard*.jsonl"))):
    with open(path) as fh:
        rows += [json.loads(line) for line in fh]

NAMES = ["PSO", "DE", "SHADE", "CLPSO", "P15", "P15e", "P16", "P17"]
CHECKS = [50, 125, 250, 500, 1250, 2500]
FES = {50: "20k", 125: "50k", 250: "100k", 500: "200k", 1250: "500k", 2500: "1M"}
FUNCS = range(1, 13)

by = {}
for r in rows:
    by.setdefault((r["seed"], r["func"], r["algo"]), r)
seeds = sorted({r["seed"] for r in rows})
print(f"{len(rows)} runs loaded, {len(seeds)} seeds\n")


def vals(func, algo, g):
    out = []
    for s in seeds:
        r = by.get((s, func, algo))
        if r is not None:
            out.append(r["traj"].get(str(g), r["final"]))
    return out


def med(func, algo, g):
    v = vals(func, algo, g)
    return float(np.median(v)) if v else float("nan")


print("=" * 100)
print(f"AVERAGE RANK vs BUDGET ({len(seeds)} seeds, shared population, "
      f"medians ranked per function)")
print("=" * 100)
print(f"{'algo':>7}" + "".join(f"{FES[g] + ' FES':>12}" for g in CHECKS))
for n in NAMES:
    line = f"{n:>7}"
    for g in CHECKS:
        meds = {m: med(f, m, g) for f in FUNCS for m in [n]}
        rk = []
        for f in FUNCS:
            order = sorted(NAMES, key=lambda a: med(f, a, g))
            rk.append(order.index(n) + 1)
        line += f"{np.mean(rk):>12.2f}"
    print(line)

print("\nMEDIAN FINAL FITNESS (1M FES)")
for f in FUNCS:
    meds = {n: med(f, n, 2500) for n in NAMES}
    best = min(meds.values())
    print(f"f{f:>2}: " + " ".join(
        f"{n}={meds[n]:>10.1f}{'*' if meds[n] == best else ' '}" for n in NAMES))


def sign_test_p(w, l):
    """Two-sided exact sign test (ties dropped)."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


for g in [125, 2500]:
    print(f"\nPAIRED PER-SEED WINS at {FES[g]} FES  "
          f"(P17 vs opponent, per function; exact sign test)")
    for opp in ["PSO", "DE", "SHADE", "CLPSO", "P15", "P15e", "P16"]:
        tot_w = tot_l = 0
        cells = []
        for f in FUNCS:
            w = l = 0
            for s in seeds:
                ra, rb = by.get((s, f, "P17")), by.get((s, f, opp))
                if ra is None or rb is None:
                    continue
                va = ra["traj"].get(str(g), ra["final"])
                vb = rb["traj"].get(str(g), rb["final"])
                if va < vb:
                    w += 1
                elif vb < va:
                    l += 1
            tot_w += w
            tot_l += l
            cells.append(f"f{f}:{w}-{l}")
        p = sign_test_p(tot_w, tot_l)
        print(f"  P17 vs {opp:>5}: total {tot_w}-{tot_l}  (p={p:.2e})   "
              + " ".join(cells))

print("\nFUNCTION-MEDIAN HEAD-TO-HEAD (wins across 12 functions)")
for g in [125, 500, 2500]:
    line = f"  at {FES[g]:>4} FES: "
    for opp in ["PSO", "DE", "SHADE", "CLPSO", "P15e"]:
        wa = wb = 0
        for f in FUNCS:
            ma, mb = med(f, "P17", g), med(f, opp, g)
            if ma < mb:
                wa += 1
            elif mb < ma:
                wb += 1
        line += f"P17>{opp} {wa}-{wb}   "
    print(line)
