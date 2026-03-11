# pulse7.py   ––  "Pulse-CR Exclusive"
#
# Same as Pulse4 (sigmoid progress trigger + 2-parent extension ray)
# but with EXCLUSIVE SELECTION: each individual can only be selected
# once per generation. This forces directional diversity in the rays
# by preventing elite concentration.

from __future__ import annotations
import torch, math
from evox.core import Algorithm


# helpers
def glued_space(x: torch.Tensor, lb: float, ub: float) -> torch.Tensor:
    return (x - lb) % (ub - lb) + lb


def geometric(p1, p2, _σ):
    α = torch.rand(1, device=p1.device)
    return α * p1 + (1 - α) * p2, (1 - α) * p1 + α * p2


def ray(p1, p2, σ):
    d = p2 - p1
    return p2 + σ * d, p1 - σ * d


OPS = {0: geometric, 1: ray}


class PulseExclusive(Algorithm):
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
        self.sigma = torch.tensor(σ_max, device=dev)

        # progress control
        self.H, self.γ, self.δ_frac = H, γ, δ_frac
        self.prev_best = torch.tensor(
            float("inf") if minimization else -float("inf"), device=dev
        )
        self.delta_hist: list[float] = []

    # ---------- EXCLUSIVE tournament: each individual selected at most once ----
    def _choose_parents_exclusive(self, k: int):
        """
        Tournament selection WITHOUT replacement across the generation.
        Each individual can be a parent at most once, forcing directional
        diversity in the ray operator.

        We need 2*k unique parent indices. If pop_size >= 2*k, we can
        randomly permute all indices, split into 2*k slots of tournament_size
        candidates each, and pick winners. If not enough, we fall back
        to random pairing from a permutation.
        """
        dev = self.population.device
        n_needed = 2 * k  # total unique parent slots needed

        # Simple approach: random permutation gives us unique pairing
        # Each individual appears exactly once as p1 or p2
        perm = torch.randperm(self.pop_size, device=dev)

        if self.pop_size >= n_needed:
            # Enough individuals: assign first k to p1, next k to p2
            # But we still want selection pressure, so do mini-tournaments
            # within shuffled blocks

            # Reshape permutation into blocks for tournament
            # We'll do tournaments but drawing from non-overlapping pools
            p1_indices = []
            p2_indices = []
            used = set()

            for i in range(k):
                # For p1: draw tournament_size candidates, pick best, mark used
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
                    # Fallback: random unused
                    remaining = [j for j in range(self.pop_size) if j not in used]
                    if remaining:
                        candidates = [remaining[0]]
                    else:
                        candidates = [perm[i % self.pop_size].item()]

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
                    if remaining:
                        candidates = [remaining[0]]
                    else:
                        candidates = [perm[(k + i) % self.pop_size].item()]

                cand_t = torch.tensor(candidates, device=dev)
                fn = torch.argmin if self.minimize else torch.argmax
                winner = cand_t[fn(self.fitness[cand_t])].item()
                p2_indices.append(winner)
                used.add(winner)

            p1 = torch.tensor(p1_indices, device=dev)
            p2 = torch.tensor(p2_indices, device=dev)
        else:
            # Not enough individuals, fallback: just pair from permutation
            p1 = perm[:k]
            p2 = perm[k:2*k] if self.pop_size >= 2*k else torch.randperm(self.pop_size, device=dev)[:k]

        return p1, p2

    # ---------- EvoX lifecycle -----------------------------------------------
    def init_step(self):
        self.fitness = self.evaluate(self.population)
        self.prev_best = self.fitness.min() if self.minimize else self.fitness.max()

    def step(self):
        # ----- compute exploration pressure ρ -------
        if self.delta_hist:
            Δ = sum(self.delta_hist) / len(self.delta_hist)
            Δ_rel = Δ / (abs(self.prev_best) + 1e-12)
            z = self.γ * (Δ_rel - self.δ_frac)
            ρ = float(torch.sigmoid(torch.tensor(-z, device=self.population.device)))
        else:
            ρ = 1.0

        q = torch.tensor([1.0 - ρ, ρ], device=self.population.device)
        self.sigma = self.σ_min * (1.0 - ρ) + self.σ_max * ρ

        # ----- variation with EXCLUSIVE selection ----
        pairs = self.pop_size // 2
        p1_idx, p2_idx = self._choose_parents_exclusive(pairs)
        op_idx = torch.multinomial(q, pairs, replacement=True)

        children = []
        for k, i1, i2 in zip(op_idx.tolist(), p1_idx.tolist(), p2_idx.tolist()):
            c1, c2 = OPS[k](self.population[i1], self.population[i2], self.sigma)
            children.extend([c1, c2])
        offspring = glued_space(torch.stack(children), self.lb, self.ub)

        # ----- evaluation & (μ+λ) selection ----------
        off_fit = self.evaluate(offspring)
        comb_pop = torch.cat([self.population, offspring])
        comb_fit = torch.cat([self.fitness, off_fit])
        best_idx = torch.topk(-comb_fit if self.minimize else comb_fit, self.pop_size).indices
        self.population, self.fitness = comb_pop[best_idx], comb_fit[best_idx]

        # ----- progress bookkeeping -------------------
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
        return {"pop": self.population, "fit": self.fitness, "σ": self.sigma}
