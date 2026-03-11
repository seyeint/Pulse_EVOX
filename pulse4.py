# pulse_convex_ray.py      -------- “Pulse-CR 1.0”

from __future__ import annotations
import torch, math
from evox.core import Algorithm


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def glued_space(x: torch.Tensor, lb: float, ub: float) -> torch.Tensor:
    return (x - lb) % (ub - lb) + lb


def geometric(p1, p2, _σ):
    α = torch.rand(1, device=p1.device)
    return α * p1 + (1 - α) * p2, (1 - α) * p1 + α * p2


def ray(p1, p2, σ):
    d = p2 - p1
    return p2 + σ * d, p1 - σ * d


OPS = {0: geometric, 1: ray}


# ---------------------------------------------------------------------
# GA with progress-controlled operator mix and σ = f(ρ)
# ---------------------------------------------------------------------
class PulseConvexRayGA(Algorithm):
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
        H: int = 5,
        γ: float = 5.0,
        δ_frac: float = 1e-3,
        device: str | torch.device | None = None,
        debug: bool = False,
    ):
        super().__init__()
        dev = torch.device(device or torch.get_default_device())

        # problem settings
        self.pop_size, self.dim = pop_size, dim
        self.lb, self.ub = lb, ub
        self.tour = tournament_size
        self.minimize = minimization
        self.debug = debug

        # population
        self.population = torch.rand(pop_size, dim, device=dev) * (ub - lb) + lb
        self.fitness = torch.empty(pop_size, device=dev)

        # σ bounds
        self.σ_min, self.σ_max = σ_min, σ_max
        self.sigma = torch.tensor(σ_max, device=dev)          # start exploratory

        # progress control buffer
        self.H, self.γ, self.δ_frac = H, γ, δ_frac
        self.prev_best = torch.tensor(float("inf") if minimization else -float("inf"), device=dev)
        self.delta_hist: list[float] = []

    # ---------- tournament --------------------------------------------------
    def _choose_parents(self, k: int):
        dev = self.population.device
        cand1 = torch.randint(0, self.pop_size, (k, self.tour), device=dev)
        cand2 = torch.randint(0, self.pop_size, (k, self.tour), device=dev)
        f1, f2 = self.fitness[cand1], self.fitness[cand2]
        fn = torch.argmin if self.minimize else torch.argmax
        p1 = cand1[torch.arange(k, device=dev), fn(f1, 1)]
        p2 = cand2[torch.arange(k, device=dev), fn(f2, 1)]
        return p1, p2

    # ---------- EvoX lifecycle ---------------------------------------------
    def init_step(self):
        self.fitness = self.evaluate(self.population)
        self.prev_best = self.fitness.min() if self.minimize else self.fitness.max()

    def step(self):
        # ----- compute exploration pressure ρ (relative improvement) -------
        if self.delta_hist:
            Δ = sum(self.delta_hist) / len(self.delta_hist)
            Δ_rel = Δ / (abs(self.prev_best) + 1e-12)  # relative progress
            z = self.γ * (Δ_rel - self.δ_frac)         # δ_frac now means percentage
            ρ = float(torch.sigmoid(torch.tensor(-z, device=self.population.device)))
        else:
            ρ = 1.0   # start exploratory

        # operator probabilities
        q = torch.tensor([1.0 - ρ, ρ], device=self.population.device)

        # update σ directly from ρ
        self.sigma = self.σ_min * (1.0 - ρ) + self.σ_max * ρ

        # ----- variation ---------------------------------------------------
        pairs = self.pop_size // 2
        p1_idx, p2_idx = self._choose_parents(pairs)
        op_idx = torch.multinomial(q, pairs, replacement=True)

        children = []
        for k, i1, i2 in zip(op_idx.tolist(), p1_idx.tolist(), p2_idx.tolist()):
            c1, c2 = OPS[k](self.population[i1], self.population[i2], self.sigma)
            children.extend([c1, c2])
        offspring = glued_space(torch.stack(children), self.lb, self.ub)

        # ----- evaluation & (μ+λ) selection -------------------------------
        off_fit = self.evaluate(offspring)
        comb_pop = torch.cat([self.population, offspring])
        comb_fit = torch.cat([self.fitness,   off_fit ])
        best_idx = torch.topk(-comb_fit if self.minimize else comb_fit, self.pop_size).indices
        self.population, self.fitness = comb_pop[best_idx], comb_fit[best_idx]

        # ----- progress bookkeeping ---------------------------------------
        new_best = self.fitness.min() if self.minimize else self.fitness.max()
        δ = abs(self.prev_best - new_best)
        self.prev_best = new_best
        self.delta_hist.append(δ)
        if len(self.delta_hist) > self.H:
            self.delta_hist.pop(0)

        if self.debug:
            print(f"ρ={ρ:.2f}  q_geo={q[0]:.2f}  σ={float(self.sigma):.3e}  Δ={δ:.2e}")

        return self

    def record_step(self):
        return {"pop": self.population, "fit": self.fitness,
                "σ": self.sigma}
