# pulse11.py  ––  "Pulse Adam"
#
# Self-normalizing adaptive controller inspired by Adam optimizer.
#
# The insight: Adam doesn't need a "baseline" because it normalizes the
# gradient by its own running variance. Our previous controllers failed
# because they compared geo_rate to a HARDCODED baseline (0.80) and used
# a HARDCODED sensitivity (10.0). These are landscape-dependent constants
# that shouldn't exist.
#
# Fix: compare geo_rate to its OWN running mean, normalized by its OWN
# running stddev. This makes the controller self-calibrating:
#   - On a smooth landscape where geo_rate averages 90%, a drop to 80%
#     is a strong signal to explore.
#   - On a rugged landscape where geo_rate averages 40%, a drop to 30%
#     is the same strength signal.
#   - No external calibration needed.
#
# The update rule:
#   z = -(geo_rate - running_mean) / (running_std + ε)
#   ρ += η * z     (geo declining → positive z → increase ρ)
#
# Additionally, there's a DRIFT term: when BOTH operators produce zero
# meaningful improvement (complete stagnation), ρ drifts upward toward
# exploration, providing a stagnation escape hatch.
#
# Components from previous iterations that are KEPT:
#   - Exclusive tournament selection (Pulse7: proven 12-0 vs Pulse4)
#   - Zeno filter (min_improvement threshold: filters trivial gains)
#   - Minimum trial guarantee (both operators get at least 10% of pairs)

from __future__ import annotations
import torch, math
from evox.core import Algorithm


def glued_space(x: torch.Tensor, lb: float, ub: float) -> torch.Tensor:
    return (x - lb) % (ub - lb) + lb


def geometric(p1, p2, _σ):
    α = torch.rand(1, device=p1.device)
    return α * p1 + (1 - α) * p2, (1 - α) * p1 + α * p2


def ray(p1, p2, σ):
    d = p2 - p1
    return p2 + σ * d, p1 - σ * d


OPS = {0: geometric, 1: ray}


class PulseAdam(Algorithm):
    def __init__(
        self,
        pop_size: int,
        dim: int,
        lb: float,
        ub: float,
        σ_min: float = 1e-3,
        σ_max: float = 3.0,
        tournament_size: int = 3,
        minimization: bool = True,
        # Adam-like controller params (analogous to Adam's β₁, β₂, lr)
        η: float = 0.05,           # learning rate for ρ updates
        β: float = 0.9,            # EMA decay for geo_rate stats (≈ Adam's β₁)
        ρ_min: float = 0.05,       # floor: always some ray
        ρ_max: float = 0.95,       # ceiling: always some geo
        ρ_init: float = 0.5,       # start agnostic
        min_frac: float = 0.10,    # minimum fraction of pairs per operator
        min_improvement: float = 1e-4,  # Zeno filter: relative improvement threshold
        stagnation_drift: float = 0.01, # drift toward ray when both operators fail
        device: str | torch.device | None = None,
        debug: bool = False,
    ):
        super().__init__()
        dev = torch.device(device or torch.get_default_device())

        assert pop_size % 2 == 0, "Pop size must be even."

        self.pop_size, self.dim = pop_size, dim
        self.lb, self.ub = lb, ub
        self.tour = tournament_size
        self.minimize = minimization
        self.debug = debug

        self.population = torch.rand(pop_size, dim, device=dev) * (ub - lb) + lb
        self.fitness = torch.empty(pop_size, device=dev)

        self.σ_min, self.σ_max = σ_min, σ_max
        self.sigma = torch.tensor(σ_max, device=dev)

        # Controller state
        self.η = η
        self.β = β
        self.ρ_min, self.ρ_max = ρ_min, ρ_max
        self.ρ = ρ_init
        self.min_frac = min_frac
        self.min_improvement = min_improvement
        self.stagnation_drift = stagnation_drift

        # Adam-like running statistics for geo_rate
        self.geo_mean = 0.5    # EMA of geo_rate (like Adam's m)
        self.geo_var = 0.01    # EMA of (geo_rate - mean)² (like Adam's v)
        self.step_count = 0    # for bias correction

    # ---------- Exclusive tournament ----------------------------------------
    def _choose_parents_exclusive(self, k: int):
        dev = self.population.device
        perm = torch.randperm(self.pop_size, device=dev)

        if self.pop_size >= 2 * k:
            p1_indices = []
            p2_indices = []
            used = set()

            for i in range(k):
                candidates = []
                attempts = 0
                while len(candidates) < self.tour and attempts < self.pop_size:
                    idx = perm[attempts % self.pop_size].item()
                    if idx not in used:
                        candidates.append(idx)
                    attempts += 1
                    if attempts >= self.pop_size:
                        break
                if not candidates:
                    remaining = [j for j in range(self.pop_size) if j not in used]
                    candidates = [remaining[0]] if remaining else [perm[i % self.pop_size].item()]

                cand_t = torch.tensor(candidates, device=dev)
                fn = torch.argmin if self.minimize else torch.argmax
                winner = cand_t[fn(self.fitness[cand_t])].item()
                p1_indices.append(winner)
                used.add(winner)

            for i in range(k):
                candidates = []
                attempts = 0
                rem_perm = perm[torch.randperm(self.pop_size, device=dev)]
                while len(candidates) < self.tour and attempts < self.pop_size:
                    idx = rem_perm[attempts % self.pop_size].item()
                    if idx not in used:
                        candidates.append(idx)
                    attempts += 1
                    if attempts >= self.pop_size:
                        break
                if not candidates:
                    remaining = [j for j in range(self.pop_size) if j not in used]
                    candidates = [remaining[0]] if remaining else [perm[(k + i) % self.pop_size].item()]

                cand_t = torch.tensor(candidates, device=dev)
                fn = torch.argmin if self.minimize else torch.argmax
                winner = cand_t[fn(self.fitness[cand_t])].item()
                p2_indices.append(winner)
                used.add(winner)

            p1 = torch.tensor(p1_indices, device=dev)
            p2 = torch.tensor(p2_indices, device=dev)
        else:
            p1 = perm[:k]
            p2 = perm[k:2*k]

        return p1, p2

    # -------------------------------------------------------------------------
    def init_step(self):
        self.fitness = self.evaluate(self.population)

    def step(self):
        dev = self.population.device
        pairs = self.pop_size // 2
        self.step_count += 1

        # ----- Assign operators with minimum trial guarantee -----
        min_pairs = max(1, int(self.min_frac * pairs))
        n_free = pairs - 2 * min_pairs

        if n_free > 0:
            q_free = torch.tensor([1.0 - self.ρ, self.ρ], device=dev)
            free_ops = torch.multinomial(q_free, n_free, replacement=True)
            op_idx = torch.cat([
                torch.zeros(min_pairs, dtype=torch.long, device=dev),
                torch.ones(min_pairs, dtype=torch.long, device=dev),
                free_ops,
            ])
            op_idx = op_idx[torch.randperm(pairs, device=dev)]
        else:
            op_idx = torch.cat([
                torch.zeros(pairs // 2, dtype=torch.long, device=dev),
                torch.ones(pairs - pairs // 2, dtype=torch.long, device=dev),
            ])
            op_idx = op_idx[torch.randperm(pairs, device=dev)]

        # σ scales with ρ
        self.sigma = self.σ_min * (1.0 - self.ρ) + self.σ_max * self.ρ

        # ----- Variation -----
        p1_idx, p2_idx = self._choose_parents_exclusive(pairs)

        children = []
        for k, i1, i2 in zip(op_idx.tolist(), p1_idx.tolist(), p2_idx.tolist()):
            c1, c2 = OPS[k](self.population[i1], self.population[i2], self.sigma)
            children.extend([c1, c2])
        offspring = glued_space(torch.stack(children), self.lb, self.ub)

        # ----- Evaluate -----
        off_fit = self.evaluate(offspring)

        # ----- Credit assignment: GEOMETRIC ONLY with Zeno filter -----
        geo_succ, geo_total = 0, 0
        ray_succ, ray_total = 0, 0

        for k_idx in range(pairs):
            op = op_idx[k_idx].item()
            i1, i2 = p1_idx[k_idx].item(), p2_idx[k_idx].item()

            par_best = min(self.fitness[i1].item(), self.fitness[i2].item()) if self.minimize \
                       else max(self.fitness[i1].item(), self.fitness[i2].item())

            c1_fit = off_fit[2 * k_idx].item()
            c2_fit = off_fit[2 * k_idx + 1].item()
            off_best = min(c1_fit, c2_fit) if self.minimize else max(c1_fit, c2_fit)

            # Zeno filter: require MEANINGFUL improvement
            rel_improvement = (par_best - off_best) / (abs(par_best) + 1e-12) if self.minimize \
                              else (off_best - par_best) / (abs(par_best) + 1e-12)
            success = rel_improvement > self.min_improvement

            if op == 0:
                geo_total += 1
                geo_succ += int(success)
            else:
                ray_total += 1
                ray_succ += int(success)

        geo_rate = geo_succ / max(geo_total, 1)
        ray_rate = ray_succ / max(ray_total, 1)

        # ----- ADAM-LIKE ρ UPDATE -----
        # Update running statistics (with bias correction like Adam)
        self.geo_mean = self.β * self.geo_mean + (1 - self.β) * geo_rate
        deviation = (geo_rate - self.geo_mean) ** 2
        self.geo_var = self.β * self.geo_var + (1 - self.β) * deviation

        # Bias correction (Adam-style)
        bc = 1.0 - self.β ** self.step_count
        mean_corrected = self.geo_mean / bc
        var_corrected = self.geo_var / bc
        std = math.sqrt(var_corrected + 1e-8)

        # Z-score: how far is current geo_rate from its running mean,
        # in units of its own standard deviation
        z = (geo_rate - mean_corrected) / std

        # Update ρ: geo declining (negative z) → increase ρ (more ray)
        #           geo improving (positive z) → decrease ρ (more geo)
        self.ρ -= self.η * z

        # Stagnation escape: if BOTH operators fail completely, drift toward ray
        if geo_succ == 0 and ray_succ == 0:
            self.ρ += self.stagnation_drift

        self.ρ = max(self.ρ_min, min(self.ρ_max, self.ρ))

        # ----- (μ+λ) selection -----
        comb_pop = torch.cat([self.population, offspring])
        comb_fit = torch.cat([self.fitness, off_fit])
        best_idx = torch.topk(-comb_fit if self.minimize else comb_fit, self.pop_size).indices
        self.population, self.fitness = comb_pop[best_idx], comb_fit[best_idx]

        if self.debug:
            print(
                f"ρ={self.ρ:.3f}  z={z:+.2f}  "
                f"geo={geo_rate:.3f}({geo_succ}/{geo_total})  μ̂={mean_corrected:.3f}  σ̂={std:.3f}  "
                f"ray={ray_rate:.3f}({ray_succ}/{ray_total})  "
                f"σ={float(self.sigma):.3e}  best={float(self.fitness.min()):.2f}"
            )

        return self

    def record_step(self):
        return {"pop": self.population, "fit": self.fitness, "σ": self.sigma, "ρ": self.ρ}
