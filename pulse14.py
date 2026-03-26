# pulse14.py  ––  "Pulse Greedy Lineage"
#
# Simplest possible pair-level adaptive control.
# No EMA. No threshold. No global ρ. No decay parameter.
#
# Each pair is a (2+2) micro-population:
#   - Pair (A, B) → geometric crossover → offspring (C, D)
#   - Keep best 2 from {A, B, C, D} as the new pair
#   - If offspring made it into the pair → line is productive → continue geo
#   - If both parents survived → line is dead → BREAK
#   - Broken pairs: freed individuals re-pair (only among freed pool)
#   - New pair does extension ray ONCE → then back to geo
#
# This is DE's per-individual greedy selection, but at the pair level.
# 200 independent parallel line searches.

from __future__ import annotations
import torch
from evox.core import Algorithm


def glued_space(x: torch.Tensor, lb: float, ub: float) -> torch.Tensor:
    return (x - lb) % (ub - lb) + lb


class PulseGreedy(Algorithm):
    def __init__(
        self,
        pop_size: int,
        dim: int,
        lb: float,
        ub: float,
        σ_ray: float = 1.0,
        tournament_size: int = 3,
        minimization: bool = True,
        patience: int = 1,    # consecutive failures before breaking pair
        center_init: "torch.Tensor | None" = None,  # optional: start population near this vector
        device: str | torch.device | None = None,
        debug: bool = False,
    ):
        super().__init__()
        dev = torch.device(device or torch.get_default_device())
        assert pop_size % 2 == 0

        self.pop_size, self.dim = pop_size, dim
        self.n_pairs = pop_size // 2
        self.lb, self.ub = lb, ub
        self.σ_ray = σ_ray
        self.tour = tournament_size
        self.minimize = minimization
        self.patience = patience
        self.debug = debug

        self.population = (
            glued_space(
                center_init.unsqueeze(0) + torch.randn(pop_size, dim, device=dev) * 0.05,
                lb, ub
            )
            if center_init is not None
            else torch.rand(pop_size, dim, device=dev) * (ub - lb) + lb
        )
        self.fitness = torch.empty(pop_size, device=dev)

        # Pair state: indices into population
        self.pair_a = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)
        self.pair_b = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)
        self.pair_fails = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)
        self.pair_is_new = torch.ones(self.n_pairs, dtype=torch.bool, device=dev)

    def init_step(self):
        self.fitness = self.evaluate(self.population)
        # Initial pairing: random shuffle
        dev = self.population.device
        perm = torch.randperm(self.pop_size, device=dev)
        for k in range(self.n_pairs):
            self.pair_a[k] = perm[2 * k]
            self.pair_b[k] = perm[2 * k + 1]
        self.pair_is_new[:] = True  # first step = ray to establish directions

    def step(self):
        dev = self.population.device

        # ----- Generate offspring per pair -----
        all_c1 = []
        all_c2 = []
        for k in range(self.n_pairs):
            ia = self.pair_a[k].item()
            ib = self.pair_b[k].item()
            p1 = self.population[ia]
            p2 = self.population[ib]

            if self.pair_is_new[k]:
                # New pair → ray with BORROWED direction from population
                # Pick two random individuals for direction (DE-style)
                r2 = torch.randint(self.pop_size, (1,), device=dev).item()
                r3 = torch.randint(self.pop_size, (1,), device=dev).item()
                while r3 == r2:
                    r3 = torch.randint(self.pop_size, (1,), device=dev).item()
                d = self.population[r2] - self.population[r3]
                # Apply borrowed direction to both pair members
                c1 = p1 + self.σ_ray * d
                c2 = p2 - self.σ_ray * d
            else:
                # Active pair → geometric (inbreed)
                α = torch.rand(1, device=dev)
                c1 = α * p1 + (1 - α) * p2
                c2 = (1 - α) * p1 + α * p2

            all_c1.append(c1)
            all_c2.append(c2)

        offspring_c1 = glued_space(torch.stack(all_c1), self.lb, self.ub)
        offspring_c2 = glued_space(torch.stack(all_c2), self.lb, self.ub)

        # Evaluate all offspring at once
        all_offspring = torch.cat([offspring_c1, offspring_c2], dim=0)  # (2*n_pairs, dim)
        all_off_fit = self.evaluate(all_offspring)
        fit_c1 = all_off_fit[:self.n_pairs]
        fit_c2 = all_off_fit[self.n_pairs:]

        # ----- Per-pair (2+2) greedy selection -----
        pairs_to_break = []
        n_productive = 0

        for k in range(self.n_pairs):
            ia = self.pair_a[k].item()
            ib = self.pair_b[k].item()

            # The 4 candidates: 2 parents + 2 offspring
            candidates_geno = [
                self.population[ia],
                self.population[ib],
                offspring_c1[k],
                offspring_c2[k],
            ]
            candidates_fit = torch.tensor([
                self.fitness[ia].item(),
                self.fitness[ib].item(),
                fit_c1[k].item(),
                fit_c2[k].item(),
            ], device=dev)

            # Select best 2
            if self.minimize:
                best2 = torch.topk(-candidates_fit, 2).indices
            else:
                best2 = torch.topk(candidates_fit, 2).indices

            winner_a = best2[0].item()
            winner_b = best2[1].item()

            # Did any offspring make it?
            offspring_survived = (winner_a >= 2) or (winner_b >= 2)

            # Update population in-place with winners
            self.population[ia] = candidates_geno[winner_a]
            self.fitness[ia] = candidates_fit[winner_a]
            self.population[ib] = candidates_geno[winner_b]
            self.fitness[ib] = candidates_fit[winner_b]

            if self.pair_is_new[k]:
                # Just did ray — transition to geo regardless
                self.pair_is_new[k] = False
                self.pair_fails[k] = 0
                if offspring_survived:
                    n_productive += 1
                continue

            if offspring_survived:
                # Line is still productive → keep inbreeding
                self.pair_fails[k] = 0
                n_productive += 1
            else:
                # Both parents survived → line produced nothing useful
                self.pair_fails[k] += 1
                if self.pair_fails[k] >= self.patience:
                    pairs_to_break.append(k)

        # ----- Re-pair broken pairs -----
        n_broken = len(pairs_to_break)
        if n_broken > 0:
            # Collect individuals freed from broken pairs
            freed = []
            for k in pairs_to_break:
                freed.append(self.pair_a[k].item())
                freed.append(self.pair_b[k].item())

            # Shuffle freed individuals and re-pair
            perm = torch.randperm(len(freed), device=dev)
            shuffled = [freed[perm[i].item()] for i in range(len(freed))]

            for i, k in enumerate(pairs_to_break):
                if 2 * i + 1 < len(shuffled):
                    self.pair_a[k] = shuffled[2 * i]
                    self.pair_b[k] = shuffled[2 * i + 1]
                self.pair_is_new[k] = True   # ray on next step
                self.pair_fails[k] = 0

        if self.debug:
            avg_fails = float(self.pair_fails.float().mean())
            print(
                f"productive={n_productive}/{self.n_pairs}  "
                f"broken={n_broken}  avg_fails={avg_fails:.1f}  "
                f"best={float(self.fitness.min()):.2f}"
            )

        return self

    def record_step(self):
        return {"pop": self.population, "fit": self.fitness}
