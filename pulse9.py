# pulse9.py   ––  "Pulse Adaptive"
#
# Exclusive selection + CREDIT-BASED adaptive controller.
#
# The sigmoid controller failed because it tracked TOTAL progress
# without knowing WHICH operator caused it. It locked at ρ=0.50
# because both operators produce similar aggregate improvement.
#
# This controller tracks per-operator success rates and moves ρ
# toward whichever operator is working better. When BOTH fail
# (stagnation), it pushes toward ray (exploration).
#
# This is the core of the Pulse thesis: adaptive geometricity
# that LISTENS to the topology.

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


class PulseAdaptive(Algorithm):
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
        # Adaptive controller params
        η: float = 0.1,           # learning rate for ρ update
        ρ_min: float = 0.05,      # minimum ray probability (ensures exploration)
        ρ_max: float = 0.95,      # maximum ray probability (ensures exploitation)
        ρ_init: float = 0.5,      # initial operator ratio
        H: int = 10,              # sliding window for success rate tracking
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

        # Population
        self.population = torch.rand(pop_size, dim, device=dev) * (ub - lb) + lb
        self.fitness = torch.empty(pop_size, device=dev)

        # σ bounds
        self.σ_min, self.σ_max = σ_min, σ_max
        self.sigma = torch.tensor(σ_max, device=dev)

        # Adaptive controller state
        self.η = η
        self.ρ_min, self.ρ_max = ρ_min, ρ_max
        self.ρ = ρ_init
        self.H = H

        # Per-operator success history (sliding window)
        # Each entry: (successes, trials)
        self.geo_history: list[tuple[int, int]] = []
        self.ray_history: list[tuple[int, int]] = []

    # ---------- Exclusive tournament selection --------------------------------
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

    # ---------- EvoX lifecycle -----------------------------------------------
    def init_step(self):
        self.fitness = self.evaluate(self.population)

    def step(self):
        dev = self.population.device

        # ----- operator probabilities from current ρ -----
        q = torch.tensor([1.0 - self.ρ, self.ρ], device=dev)

        # σ scales with ρ: more ray → larger σ
        self.sigma = self.σ_min * (1.0 - self.ρ) + self.σ_max * self.ρ

        # ----- variation with exclusive selection & operator tracking -----
        pairs = self.pop_size // 2
        p1_idx, p2_idx = self._choose_parents_exclusive(pairs)
        op_idx = torch.multinomial(q, pairs, replacement=True)

        children = []
        for k, i1, i2 in zip(op_idx.tolist(), p1_idx.tolist(), p2_idx.tolist()):
            c1, c2 = OPS[k](self.population[i1], self.population[i2], self.sigma)
            children.extend([c1, c2])
        offspring = glued_space(torch.stack(children), self.lb, self.ub)

        # ----- evaluate offspring -----
        off_fit = self.evaluate(offspring)

        # ----- CREDIT ASSIGNMENT: track per-operator success -----
        # For each pair, check if at least one offspring beats the best parent
        geo_succ, geo_total = 0, 0
        ray_succ, ray_total = 0, 0

        for k_idx in range(pairs):
            op = op_idx[k_idx].item()
            i1, i2 = p1_idx[k_idx].item(), p2_idx[k_idx].item()

            # Parent fitness
            par_best = min(self.fitness[i1].item(), self.fitness[i2].item()) if self.minimize \
                       else max(self.fitness[i1].item(), self.fitness[i2].item())

            # Offspring fitness
            c1_fit = off_fit[2 * k_idx].item()
            c2_fit = off_fit[2 * k_idx + 1].item()
            off_best = min(c1_fit, c2_fit) if self.minimize else max(c1_fit, c2_fit)

            # Did offspring beat the best parent?
            success = off_best < par_best if self.minimize else off_best > par_best

            if op == 0:  # geometric
                geo_total += 1
                geo_succ += int(success)
            else:  # ray
                ray_total += 1
                ray_succ += int(success)

        # Record in sliding window
        self.geo_history.append((geo_succ, geo_total))
        self.ray_history.append((ray_succ, ray_total))
        if len(self.geo_history) > self.H:
            self.geo_history.pop(0)
        if len(self.ray_history) > self.H:
            self.ray_history.pop(0)

        # Compute windowed success rates
        total_geo_succ = sum(s for s, _ in self.geo_history)
        total_geo_trials = sum(t for _, t in self.geo_history)
        total_ray_succ = sum(s for s, _ in self.ray_history)
        total_ray_trials = sum(t for _, t in self.ray_history)

        geo_rate = total_geo_succ / max(total_geo_trials, 1)
        ray_rate = total_ray_succ / max(total_ray_trials, 1)

        # ----- ADAPTIVE ρ UPDATE -----
        # The logic:
        #   - If ray is more successful than geo → increase ρ (more ray)
        #   - If geo is more successful than ray → decrease ρ (more geo)
        #   - If BOTH are failing (stagnation) → increase ρ (need to explore/expand hull)
        #   - The magnitude of the update scales with the success difference

        if total_geo_trials > 0 and total_ray_trials > 0:
            if geo_rate == 0 and ray_rate == 0:
                # Complete stagnation: push toward exploration
                self.ρ = min(self.ρ + self.η, self.ρ_max)
            else:
                # Move toward whichever operator is performing better
                advantage = ray_rate - geo_rate  # positive = ray is better
                self.ρ = self.ρ + self.η * advantage
                self.ρ = max(self.ρ_min, min(self.ρ_max, self.ρ))
        elif total_geo_trials == 0:
            # No geo was tried → we need some geo to compare
            self.ρ = max(self.ρ - self.η, self.ρ_min)
        elif total_ray_trials == 0:
            # No ray was tried → we need some ray to compare
            self.ρ = min(self.ρ + self.η, self.ρ_max)

        # ----- (μ+λ) selection -----
        comb_pop = torch.cat([self.population, offspring])
        comb_fit = torch.cat([self.fitness, off_fit])
        best_idx = torch.topk(-comb_fit if self.minimize else comb_fit, self.pop_size).indices
        self.population, self.fitness = comb_pop[best_idx], comb_fit[best_idx]

        if self.debug:
            print(
                f"ρ={self.ρ:.3f}  geo_rate={geo_rate:.3f}({geo_succ}/{geo_total})  "
                f"ray_rate={ray_rate:.3f}({ray_succ}/{ray_total})  "
                f"σ={float(self.sigma):.3e}  best={float(self.fitness.min()):.2f}"
            )

        return self

    def record_step(self):
        return {"pop": self.population, "fit": self.fitness, "σ": self.sigma, "ρ": self.ρ}
