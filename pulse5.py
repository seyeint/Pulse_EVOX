# pulse5.py   ––  "Pulse-Cov 1.0"

"""
Covariance-Aware Pulse GA with progress-controlled operator mix.

Core idea
---------
 • Same sigmoid stagnation trigger as Pulse3 — track geometric progress,
   fire non-geometric pulse when interpolation stalls.
 • NEW: The non-geometric ray uses a *random* population difference vector
   for direction (covariance-sampled, valley-aligned) applied from a
   tournament-selected base point. This gives Pulse DE's implicit
   covariance tracking without DE's blindness about *when* to explore.

Operators
---------
 0 - Geometric crossover:  α·p1 + (1-α)·p2          (convex, hull-contracting)
 1 - Covariance-aware ray: p1 ± σ·(x_a - x_b)       (non-convex, hull-expanding,
                           where x_a, x_b are random   direction is covariance-sampled)
"""

from __future__ import annotations
import torch
from evox.core import Algorithm


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def glued_space(x: torch.Tensor, lb: float, ub: float) -> torch.Tensor:
    """Wrap into [lb, ub] (torus)."""
    return (x - lb) % (ub - lb) + lb


def geometric(p1, p2, _σ, _pop):
    """Standard convex combination — geometric crossover."""
    α = torch.rand(1, device=p1.device)
    return α * p1 + (1 - α) * p2, (1 - α) * p1 + α * p2


def covariance_ray(p1, p2, σ, population):
    """
    Covariance-aware extension ray.

    Base points: p1 and p2 (tournament-selected, exploitation)
    Direction:   (x_a - x_b) from two RANDOM population members (covariance-sampled)

    This decouples WHERE we step from (good regions) from WHICH DIRECTION
    we step in (population covariance axes ≈ valley-aligned directions).
    """
    n = population.size(0)
    idx = torch.randint(0, n, (2,), device=population.device)
    d = population[idx[0]] - population[idx[1]]  # covariance-sampled direction
    return p1 + σ * d, p2 - σ * d


OPS = {0: geometric, 1: covariance_ray}


# ---------------------------------------------------------------------
# Covariance-Aware Pulse GA
# ---------------------------------------------------------------------
class PulseCovarianceGA(Algorithm):
    def __init__(
        self,
        pop_size: int,
        dim: int,
        lb: float,
        ub: float,
        sigma_init: float = 1.0,
        tournament_size: int = 3,
        minimization: bool = True,
        H: int = 5,
        gamma: float = 10.0,
        delta_frac: float = 1e-4,
        device: str | torch.device | None = None,
        debug: bool = False,
    ):
        super().__init__()
        dev = torch.device(device or torch.get_default_device())

        # problem settings
        self.pop_size = pop_size
        self.dim = dim
        self.lb, self.ub = lb, ub
        self.tour = tournament_size
        self.minimize = minimization
        self.debug = debug

        # population
        self.population = torch.rand(pop_size, dim, device=dev) * (ub - lb) + lb
        self.fitness = torch.empty(pop_size, device=dev)

        # operator probabilities [geometric, covariance_ray]
        self.q = torch.tensor([1.0, 0.0], device=dev)

        # global step-size
        self.sigma = torch.tensor(sigma_init, device=dev)
        self.max_sigma = 3.0

        # progress controller (sigmoid stagnation trigger)
        self.H = H
        self.gamma = gamma
        self.delta_frac = delta_frac
        self.prev_best = torch.tensor(
            float("inf") if minimization else -float("inf"), device=dev
        )
        self.delta_hist: list[float] = []

    # -----------------------------------------------------------------
    # helper: vectorised tournament selection
    # -----------------------------------------------------------------
    def _choose_parents(self, n_pairs: int):
        dev = self.population.device
        cand1 = torch.randint(0, self.pop_size, (n_pairs, self.tour), device=dev)
        cand2 = torch.randint(0, self.pop_size, (n_pairs, self.tour), device=dev)
        f1, f2 = self.fitness[cand1], self.fitness[cand2]
        fn = torch.argmin if self.minimize else torch.argmax
        p1 = cand1[torch.arange(n_pairs, device=dev), fn(f1, 1)]
        p2 = cand2[torch.arange(n_pairs, device=dev), fn(f2, 1)]
        return p1, p2

    # -----------------------------------------------------------------
    # EvoX hooks
    # -----------------------------------------------------------------
    def init_step(self):
        self.fitness = self.evaluate(self.population)
        self.prev_best = (
            self.fitness.min() if self.minimize else self.fitness.max()
        )

    def step(self):
        # ---------- progress-controlled operator mix ------------------
        # Track how much the best fitness improved recently.
        # When geometric progress stalls → sigmoid flips → pulse fires.
        if self.delta_hist:
            Δ = sum(self.delta_hist) / len(self.delta_hist)
            Δ_target = self.delta_frac * abs(self.prev_best)
            z = self.gamma * (Δ - Δ_target)
            ρ = float(torch.sigmoid(torch.tensor(-z, device=self.q.device)))
            self.q[0] = 1.0 - ρ   # geometric probability
            self.q[1] = ρ         # covariance ray probability
        else:
            ρ = 0.0  # first generation: pure geometric

        # ---------- variation ----------------------------------------
        n_pairs = self.pop_size // 2
        p1_idx, p2_idx = self._choose_parents(n_pairs)
        op_idx = torch.multinomial(self.q, n_pairs, replacement=True)

        offspring = []
        for k, i1, i2 in zip(
            op_idx.tolist(), p1_idx.tolist(), p2_idx.tolist()
        ):
            c1, c2 = OPS[k](
                self.population[i1],
                self.population[i2],
                self.sigma,
                self.population,  # pass full population for covariance sampling
            )
            offspring.extend([c1, c2])
        offspring = glued_space(torch.stack(offspring, 0), self.lb, self.ub)

        # ---------- evaluation & (μ+λ) selection ----------------------
        off_fit = self.evaluate(offspring)
        comb_pop = torch.cat([self.population, offspring])
        comb_fit = torch.cat([self.fitness, off_fit])
        best_idx = torch.topk(
            -comb_fit if self.minimize else comb_fit, self.pop_size
        ).indices
        self.population, self.fitness = comb_pop[best_idx], comb_fit[best_idx]

        # ---------- σ update (1/5 success rule) -----------------------
        off_fit_pairs = off_fit.view(n_pairs, 2)
        parent_fit = torch.stack(
            [self.fitness[p1_idx], self.fitness[p2_idx]], dim=1
        )
        best_parent = (
            parent_fit.min(dim=1)[0]
            if self.minimize
            else parent_fit.max(dim=1)[0]
        )

        # Better than best parent
        better = (
            (off_fit_pairs < best_parent.unsqueeze(1))
            if self.minimize
            else (off_fit_pairs > best_parent.unsqueeze(1))
        )

        # Different from parents (not a clone)
        epsilon = 1e-6
        offspring_pairs = offspring.view(n_pairs, 2, self.dim)
        parent_pairs = torch.stack(
            [self.population[p1_idx], self.population[p2_idx]], dim=1
        )
        is_eq = torch.zeros(
            n_pairs, 2, dtype=torch.bool, device=offspring.device
        )
        for i in range(n_pairs):
            for j in range(2):
                is_eq[i, j] = torch.all(
                    torch.abs(offspring_pairs[i, j] - parent_pairs[i, 0])
                    < epsilon
                ) | torch.all(
                    torch.abs(offspring_pairs[i, j] - parent_pairs[i, 1])
                    < epsilon
                )

        # Success: better AND different
        success = ~is_eq & better
        sr = success.float().mean()
        self.sigma *= torch.exp(0.2 * (sr - 0.2))
        self.sigma = torch.clamp(self.sigma, 1e-6, self.max_sigma)

        # ---------- progress buffer update ----------------------------
        new_best = (
            self.fitness.min() if self.minimize else self.fitness.max()
        )
        delta = abs(self.prev_best - new_best)
        self.prev_best = new_best
        self.delta_hist.append(delta)
        if len(self.delta_hist) > self.H:
            self.delta_hist.pop(0)

        if self.debug:
            print(
                f"ρ={ρ:.2f}  q_geo={self.q[0]:.2f}  q_cov={self.q[1]:.2f}  "
                f"σ={float(self.sigma):.3e}  Δ={delta:.2e}  sr={sr:.2f}"
            )

        return self

    def record_step(self):
        return {
            "pop": self.population,
            "fit": self.fitness,
            "q": self.q,
            "σ": self.sigma,
        }
