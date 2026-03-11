# pulse10.py   ––  "Pulse Adaptive v2"
#
# Credit-based adaptive ρ driven by GEOMETRIC SUCCESS ONLY.
#
# Core principle (from José's thesis):
#   - Geometric crossover CONTRACTS the hull. Track its success.
#   - When geo succeeds → hull is productive → keep contracting (low ρ)
#   - When geo fails → hull is exhausted → fire ray to expand (high ρ)
#   - Ray should NEVER be penalized for low success rate. Its role is
#     to inject volume — most injections fail, but the few that succeed
#     open territory for geo to exploit. Ray is the remedy, not the patient.
#
# Controller: ρ = 1 - smoothed_geo_success_rate
#   Smoothed via exponential tracking toward target.
#   Minimum trial guarantee ensures geo always has enough samples.

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


class PulseAdaptive2(Algorithm):
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
        # Adaptive controller
        η: float = 0.2,            # tracking rate: how fast ρ follows the signal
        ρ_min: float = 0.05,       # always keep some ray (exploration floor)
        ρ_max: float = 0.95,       # always keep some geo (exploitation floor)
        ρ_init: float = 0.5,       # start agnostic
        H: int = 5,                # sliding window for geo success rate
        min_frac: float = 0.10,    # minimum fraction of pairs for each operator
        min_improvement: float = 1e-4,  # relative improvement threshold (Zeno filter)
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

        self.η = η
        self.ρ_min, self.ρ_max = ρ_min, ρ_max
        self.ρ = ρ_init
        self.H = H
        self.min_frac = min_frac
        self.min_improvement = min_improvement

        # Geo success rate history (sliding window)
        self.geo_rates: list[float] = []

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

        # ----- Assign operators with minimum trial guarantee -----
        min_pairs = max(1, int(self.min_frac * pairs))
        n_guaranteed_geo = min_pairs
        n_guaranteed_ray = min_pairs
        n_free = pairs - n_guaranteed_geo - n_guaranteed_ray

        if n_free > 0:
            q_free = torch.tensor([1.0 - self.ρ, self.ρ], device=dev)
            free_ops = torch.multinomial(q_free, n_free, replacement=True)
            op_idx = torch.cat([
                torch.zeros(n_guaranteed_geo, dtype=torch.long, device=dev),
                torch.ones(n_guaranteed_ray, dtype=torch.long, device=dev),
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

        # ----- Credit assignment: GEOMETRIC ONLY -----
        # Success = offspring MEANINGFULLY better than best parent.
        # Not just "epsilon better" (Zeno paradox: trivial improvements
        # keep geo_rate high even when the hull is essentially exhausted).
        # We require relative improvement > min_improvement threshold.
        geo_succ, geo_total = 0, 0
        ray_succ, ray_total = 0, 0  # tracked for debug only

        for k_idx in range(pairs):
            op = op_idx[k_idx].item()
            i1, i2 = p1_idx[k_idx].item(), p2_idx[k_idx].item()

            par_best = min(self.fitness[i1].item(), self.fitness[i2].item()) if self.minimize \
                       else max(self.fitness[i1].item(), self.fitness[i2].item())

            c1_fit = off_fit[2 * k_idx].item()
            c2_fit = off_fit[2 * k_idx + 1].item()
            off_best = min(c1_fit, c2_fit) if self.minimize else max(c1_fit, c2_fit)

            # Relative improvement (positive = offspring is better)
            rel_improvement = (par_best - off_best) / (abs(par_best) + 1e-12) if self.minimize \
                              else (off_best - par_best) / (abs(par_best) + 1e-12)

            # Must improve by at least min_improvement fraction to count
            success = rel_improvement > self.min_improvement

            if op == 0:
                geo_total += 1
                geo_succ += int(success)
            else:
                ray_total += 1
                ray_succ += int(success)

        geo_rate = geo_succ / max(geo_total, 1)
        ray_rate = ray_succ / max(ray_total, 1)  # for debug only

        self.geo_rates.append(geo_rate)
        if len(self.geo_rates) > self.H:
            self.geo_rates.pop(0)

        # ----- ρ UPDATE: driven by geo success ONLY -----
        # In a (μ+λ) population of 400, geometric crossover typically
        # has 60-80% success rate even in near-stagnation. So the raw
        # mapping ρ = 1 - geo_rate is too conservative.
        #
        # Instead: use a shifted sigmoid that maps geo_rate to ρ.
        #   geo_rate > baseline (0.80) → ρ → ρ_min (hull is truly productive)
        #   geo_rate ≈ baseline        → ρ ≈ 0.50 (balanced transition)
        #   geo_rate < baseline        → ρ → ρ_max (hull is exhausted)
        #
        # The baseline is the expected "trivial" geo success rate.
        # Deviations from baseline drive ρ.
        avg_geo_rate = sum(self.geo_rates) / len(self.geo_rates)
        baseline = 0.80  # expected geo success in a healthy population
        sensitivity = 10.0  # steepness of the transition
        z = sensitivity * (baseline - avg_geo_rate)  # positive when geo is below baseline
        ρ_target = 1.0 / (1.0 + math.exp(-z))  # sigmoid: 0→1 as geo drops

        # Smooth tracking toward target
        self.ρ = self.ρ + self.η * (ρ_target - self.ρ)
        self.ρ = max(self.ρ_min, min(self.ρ_max, self.ρ))

        # ----- (μ+λ) selection -----
        comb_pop = torch.cat([self.population, offspring])
        comb_fit = torch.cat([self.fitness, off_fit])
        best_idx = torch.topk(-comb_fit if self.minimize else comb_fit, self.pop_size).indices
        self.population, self.fitness = comb_pop[best_idx], comb_fit[best_idx]

        if self.debug:
            print(
                f"ρ={self.ρ:.3f}  ρ_target={ρ_target:.3f}  "
                f"geo={geo_rate:.3f}({geo_succ}/{geo_total})  "
                f"ray={ray_rate:.3f}({ray_succ}/{ray_total})  "
                f"σ={float(self.sigma):.3e}  best={float(self.fitness.min()):.2f}"
            )

        return self

    def record_step(self):
        return {"pop": self.population, "fit": self.fitness, "σ": self.sigma, "ρ": self.ρ}
