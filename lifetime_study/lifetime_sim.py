# Validation sim for the direction-lifetime / amortization theory.
#
# Landscape family (one formula covers sphere, valley, drifting valley):
#   f_t(x) = 0.5 * ( kappa*||x||^2 - (kappa-1)*(u_t . x)^2 )
#   -> curvature 1 along u_t (valley floor), kappa orthogonal. kappa=1 => sphere.
#   Optimum fixed at origin; under drift u_t rotates at omega rad/gen in a fixed
#   random 2-plane (valley orientation rotates, optimum doesn't move).
#
# Algorithm (memory variant) = minimal LinePulse: pairs, uniform-alpha geometric
# crossover along the pair secant, keep-best-2 of {P1,P2,C1,C2}, patience-m
# eviction, broken pairs re-pair among freed pool and fire one ray
# C = P +/- sigma_ray*(X[r2]-X[r3]).  All four candidates evaluated fresh every
# generation (no cached fitness -> no phantom-elitism confound).
#
# Control (forced) = identical operators, but ALL pairs are randomly re-formed
# every generation (no lineage memory); each pair does a ray with probability
# p_ray (matched to the memory run's measured ray fraction, same config+seed),
# else geometric crossover. Isolates persistence as the only variable.
#
# Predictions under test:
#  P1 kappa=1: speedup(memory/forced) ~ 1
#  P2 speedup grows with kappa
#  P3 speedup decays toward 1 as drift omega grows
#  P4 optimal patience m*=1 at zero noise, increases with noise
#  P5 secant alignment |cos(secant,u)| rises above 1/sqrt(d) on valley, not sphere

import json
import os
import sys
import time

import numpy as np

D = 20
POP = 400            # 40 pairs; affine hull spans R^60 generically (pop-1 > d)
NP_ = POP // 2
SIGMA_RAY = 1.0
R0 = 15.0           # initial cloud center norm (optimum near the initial hull —
S0 = 15.0           # contraction regime, where local progress rates are defined)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lifetime_sim_results.jsonl")


def f_true(X, u, kappa):
    n2 = np.sum(X * X, axis=-1)
    pu = X @ u
    return 0.5 * (kappa * n2 - (kappa - 1.0) * pu * pu)


def run(kappa=1.0, omega=0.0, gamma=0.0, patience=1, variant="memory",
        seed=0, gens=1500, p_ray=None, log_every=5):
    rng = np.random.default_rng(seed)
    # fixed rotation plane for drift
    a = rng.standard_normal(D); a /= np.linalg.norm(a)
    b = rng.standard_normal(D); b -= (b @ a) * a; b /= np.linalg.norm(b)

    center = rng.standard_normal(D); center *= R0 / np.linalg.norm(center)
    X = center + S0 * rng.standard_normal((POP, D))

    perm = rng.permutation(POP)
    pa, pb = perm[0::2].copy(), perm[1::2].copy()
    fails = np.zeros(NP_, dtype=int)
    is_new = np.ones(NP_, dtype=bool)
    age = np.zeros(NP_, dtype=int)          # productive generations since formation

    lifetimes = []
    ray_gens = 0
    pair_gens = 0
    traj = []                               # (evals, best_f_true)
    aligns = []
    f0 = None
    evals = 0

    for g in range(gens):
        u = np.cos(omega * g) * a + np.sin(omega * g) * b

        P1, P2 = X[pa], X[pb]
        if variant == "memory":
            ray_mask = is_new.copy()
        else:
            ray_mask = rng.random(NP_) < (p_ray if p_ray is not None else 0.1)
        ray_gens += int(ray_mask.sum()); pair_gens += NP_

        C1 = np.empty_like(P1); C2 = np.empty_like(P2)
        # rays
        if ray_mask.any():
            k = int(ray_mask.sum())
            r2 = rng.integers(0, POP, k); r3 = rng.integers(0, POP, k)
            clash = r2 == r3
            while clash.any():
                r3[clash] = rng.integers(0, POP, int(clash.sum())); clash = r2 == r3
            Dv = X[r2] - X[r3]
            C1[ray_mask] = P1[ray_mask] + SIGMA_RAY * Dv
            C2[ray_mask] = P2[ray_mask] - SIGMA_RAY * Dv
        # geometric crossover
        gm = ~ray_mask
        if gm.any():
            al = rng.random((int(gm.sum()), 1))
            C1[gm] = al * P1[gm] + (1 - al) * P2[gm]
            C2[gm] = (1 - al) * P1[gm] + al * P2[gm]

        cand = np.stack([P1, P2, C1, C2], axis=1)            # copies -> no aliasing
        ft = f_true(cand.reshape(-1, D), u, kappa).reshape(NP_, 4)
        evals += 4 * NP_
        fobs = ft * (1 + gamma * rng.standard_normal(ft.shape)) if gamma > 0 else ft

        idx = np.argsort(fobs, axis=1)[:, :2]                # minimize
        winners = np.take_along_axis(cand, idx[:, :, None], axis=1)
        X[pa] = winners[:, 0]; X[pb] = winners[:, 1]
        surv = (idx >= 2).any(axis=1)

        if f0 is None:
            f0 = float(f_true(X, u, kappa).min())

        if variant == "memory":
            newly = is_new.copy()
            is_new[newly] = False
            fails[newly] = 0
            cont = ~newly
            fails[cont & surv] = 0
            age[surv] += 1
            broke = cont & ~surv
            fails[broke] += 1
            dead = fails >= patience
            if dead.any():
                for k in np.where(dead)[0]:
                    lifetimes.append(int(age[k]))
                freed = np.concatenate([pa[dead], pb[dead]])
                rng.shuffle(freed)
                pa[dead] = freed[0::2]; pb[dead] = freed[1::2]
                is_new[dead] = True; fails[dead] = 0; age[dead] = 0
        else:
            perm = rng.permutation(POP)
            pa, pb = perm[0::2], perm[1::2]

        if g % 5 == 0 or g == gens - 1:
            fb = float(f_true(X, u, kappa).min())
            traj.append((evals, fb))
            S = X[pb] - X[pa]
            ns = np.linalg.norm(S, axis=1); ok = ns > 1e-300
            if ok.any():
                aligns.append(float(np.mean(np.abs((S[ok] @ u) / ns[ok]))))
            if fb < 1e-12 * max(f0, 1e-300):
                break

    # rate: decades of f reduction per 1000 evaluations (least squares on log10 f)
    t = np.array(traj, dtype=float)
    f_end = max(t[-1, 1], 1e-14 * f0)
    lo, hi = f_end * 10, 1e-2 * f0
    m = (t[:, 1] > lo) & (t[:, 1] < hi) & (t[:, 1] > 0)
    if m.sum() < 8:
        lo, hi = f_end * 3, 0.3 * f0
        m = (t[:, 1] > lo) & (t[:, 1] < hi) & (t[:, 1] > 0)
    if m.sum() >= 5:
        sl = np.polyfit(t[m, 0], np.log10(t[m, 1]), 1)[0]
        rate = -sl * 1000.0
    else:
        rate = float("nan")
    return {
        "kappa": kappa, "omega": omega, "gamma": gamma, "patience": patience,
        "variant": variant, "seed": seed,
        "rate": rate, "final_f": float(t[-1, 1] / f0), "evals": int(t[-1, 0]),
        "L": float(np.mean(lifetimes)) if lifetimes else float(len(t) * log_every),
        "ray_frac": ray_gens / max(pair_gens, 1),
        "align_final": float(np.median(aligns[-10:])) if len(aligns) >= 10 else float("nan"),
        "align_start": float(np.median(aligns[:10])) if len(aligns) >= 10 else float("nan"),
    }


def emit(row):
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")


def med(rows, key):
    v = [r[key] for r in rows if np.isfinite(r[key])]
    return float(np.median(v)) if v else float("nan")


def main():
    seeds = list(range(6))
    t0 = time.time()
    open(OUT, "w").close()
    results = []

    def batch(tag, **cfg):
        mem = []
        for s in seeds:
            r = run(variant="memory", seed=s, **cfg); r["tag"] = tag
            emit(r); results.append(r); mem.append(r)
        for s, rm in zip(seeds, mem):
            r = run(variant="forced", seed=s, p_ray=rm["ray_frac"], **cfg); r["tag"] = tag
            emit(r); results.append(r)
        print(f"[{(time.time()-t0)/60:5.1f} min] {tag} done", flush=True)

    # P1/P2/P5: kappa sweep (static valley)
    for kappa in [1, 10, 100, 1000]:
        gens = 1500 if kappa <= 100 else 3000
        batch(f"kappa={kappa}", kappa=float(kappa), gens=gens)

    # P3: drift sweep at kappa=100
    for om in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:
        batch(f"drift w={om}", kappa=100.0, omega=om, gens=1500)

    # P4: patience x noise (memory variant only), kappa=100
    for gamma in [0.0, 0.05, 0.15, 0.3]:
        for m in [1, 2, 4, 8]:
            for s in seeds:
                r = run(variant="memory", seed=s, kappa=100.0, gamma=gamma,
                        patience=m, gens=1500)
                r["tag"] = f"pat g={gamma} m={m}"
                emit(r); results.append(r)
            print(f"[{(time.time()-t0)/60:5.1f} min] pat g={gamma} m={m} done", flush=True)

    # ---------------- summary ----------------
    def sel(**kw):
        out = results
        for k, v in kw.items():
            out = [r for r in out if r.get(k) == v]
        return out

    print("\n=== P1/P2: kappa sweep (rate = decades/1000 evals; speedup = mem/forced) ===")
    print(f"{'kappa':>6} {'rate_mem':>9} {'rate_fcd':>9} {'speedup':>8} {'L_mem':>7} "
          f"{'align0':>7} {'alignF':>7}  (1/sqrt(d)={1/np.sqrt(D):.3f})")
    for kappa in [1, 10, 100, 1000]:
        m_ = sel(kappa=float(kappa), omega=0.0, gamma=0.0, patience=1, variant="memory")
        m_ = [r for r in m_ if r["tag"].startswith("kappa")]
        f_ = sel(kappa=float(kappa), omega=0.0, variant="forced")
        f_ = [r for r in f_ if r["tag"].startswith("kappa")]
        rm, rf = med(m_, "rate"), med(f_, "rate")
        print(f"{kappa:>6} {rm:>9.3f} {rf:>9.3f} {rm/rf if rf else float('nan'):>8.2f} "
              f"{med(m_,'L'):>7.1f} {med(m_,'align_start'):>7.3f} {med(m_,'align_final'):>7.3f}")

    print("\n=== P3: drift sweep (kappa=100) ===")
    print(f"{'omega':>8} {'rate_mem':>9} {'rate_fcd':>9} {'speedup':>8} {'L_mem':>7}")
    for om in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:
        m_ = [r for r in sel(omega=om, variant="memory") if r["tag"].startswith("drift")]
        f_ = [r for r in sel(omega=om, variant="forced") if r["tag"].startswith("drift")]
        rm, rf = med(m_, "rate"), med(f_, "rate")
        print(f"{om:>8} {rm:>9.3f} {rf:>9.3f} {rm/rf if rf else float('nan'):>8.2f} {med(m_,'L'):>7.1f}")

    print("\n=== P4: patience x noise (kappa=100, memory variant, rate) ===")
    hdr = "".join(f"{'m='+str(m):>9}" for m in [1, 2, 4, 8])
    print(f"{'gamma':>7}{hdr}   m*")
    for gamma in [0.0, 0.05, 0.15, 0.3]:
        row, best_m, best_r = f"{gamma:>7}", None, -1
        for m in [1, 2, 4, 8]:
            rr = med([r for r in sel(gamma=gamma, patience=m, variant="memory")
                      if r["tag"].startswith("pat")], "rate")
            row += f"{rr:>9.3f}"
            if np.isfinite(rr) and rr > best_r:
                best_r, best_m = rr, m
        print(row + f"   {best_m}")

    print(f"\ntotal {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
