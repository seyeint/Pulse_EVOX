# Figures for the direction-lifetime study. Reads lifetime_sim_results.jsonl.
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "lifetime_sim_results.jsonl"))]
D_BASELINE = 1 / math.sqrt(20)


def decades(r):
    return -math.log10(max(r["final_f"], 1e-300))


def sel(tagpre, **kw):
    out = [r for r in rows if r["tag"].startswith(tagpre)]
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


def med(rs, fn):
    v = [fn(r) for r in rs]
    return np.median(v) if v else np.nan


# ---- fig 1: kappa sweep — progress + alignment ----
kappas = [1, 10, 100, 1000]
mem = [med(sel("kappa", kappa=float(k), variant="memory"), decades) for k in kappas]
fcd = [med(sel("kappa", kappa=float(k), variant="forced"), decades) for k in kappas]
ali = [med(sel("kappa", kappa=float(k), variant="memory"), lambda r: r["align_final"]) for k in kappas]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
x = np.arange(len(kappas))
a1.bar(x - 0.18, mem, 0.36, label="lineage memory", color="#34577c")
a1.bar(x + 0.18, fcd, 0.36, label="forced re-pair (control)", color="#b0b7bf")
a1.set_xticks(x, [f"κ={k}" for k in kappas])
a1.set_ylabel("decades of f-reduction (equal budget)")
a1.set_title("Persistence pays everywhere;\nexcess grows with conditioning")
a1.legend(frameon=False, fontsize=8)
for i, (m, f) in enumerate(zip(mem, fcd)):
    a1.text(i, m + 0.08, f"×{m/f:.2f}", ha="center", fontsize=8, color="#34577c")

a2.plot(kappas, ali, "o-", color="#8c6d1f")
a2.axhline(D_BASELINE, ls="--", color="#999", lw=1)
a2.text(1.2, D_BASELINE + 0.02, "random baseline 1/√d", fontsize=8, color="#777")
a2.set_xscale("log")
a2.set_xlabel("condition number κ")
a2.set_ylabel("final |cos(secant, valley axis)|")
a2.set_title("Secants self-align with the valley floor")
a2.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_kappa.png"), dpi=160)

# ---- fig 2: patience x noise — the retention-starvation cliff ----
gammas = [0.0, 0.05, 0.15, 0.3]
ms = [1, 2, 4, 8]
fig, ax = plt.subplots(figsize=(6.8, 3.6))
w = 0.2
for j, m in enumerate(ms):
    vals = [med(sel("pat", gamma=g, patience=m), decades) for g in gammas]
    ax.bar(np.arange(len(gammas)) + (j - 1.5) * w, vals, w, label=f"patience {m}")
ax.set_xticks(range(len(gammas)), [f"γ={g}" for g in gammas])
ax.set_ylabel("decades (equal budget)")
ax.set_title("Noise + patience ⇒ eviction starvation; noise + patience-1 ⇒ free diffusion")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_patience_noise.png"), dpi=160)

# ---- fig 3: drift — alignment collapse tracks the steering excess ----
omegas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
ratio = []
alis = []
for om in omegas:
    dm = med(sel("drift", omega=om, variant="memory"), decades)
    df = med(sel("drift", omega=om, variant="forced"), decades)
    ratio.append(dm / df)
    alis.append(med(sel("drift", omega=om, variant="memory"), lambda r: r["align_final"]))
ox = [max(o, 3e-5) for o in omegas]
fig, a1 = plt.subplots(figsize=(6.2, 3.4))
a1.semilogx(ox, alis, "o-", color="#8c6d1f", label="alignment")
a1.axhline(D_BASELINE, ls="--", color="#999", lw=1)
a1.set_xlabel("valley rotation speed ω (rad/gen)   [leftmost point = ω=0]")
a1.set_ylabel("final alignment", color="#8c6d1f")
a2 = a1.twinx()
a2.semilogx(ox, ratio, "s-", color="#34577c", label="memory/control ratio")
a2.set_ylabel("progress ratio", color="#34577c")
a1.set_title("Drift destroys the harvested direction (alignment → baseline)")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_drift.png"), dpi=160)
print("figures written")
