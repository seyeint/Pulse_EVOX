# pulse15m.py  ––  "Pulse Mixed": one interpolant + one directed extension,
#                   every generation, no modes.
#
# Exactly pulse14 with a single change: an active pair's two offspring are no
# longer mirrored interpolants. Each generation the pair now tests BOTH
# hypotheses simultaneously:
#
#   C1 = α·A + (1-α)·B                  "the minimum is inside the segment"
#   C2 = Best + (Best - Worst)          "the minimum is beyond the better parent"
#
# (C2 is Wright's heuristic crossover / the Nelder-Mead reflection through the
# better member.) Selection keeps the best 2 of {A, B, C1, C2}; the lineage
# continues while any offspring survives. Because both hypotheses are probed
# every generation, "both parents survived" now genuinely means the line is
# resolved — the eviction rule's semantics are restored by construction,
# with no state machine.
#
# Everything else is pulse14 verbatim: symmetric re-pair rays, cached parent
# fitness (deterministic evaluation), patience, freed-pool re-pairing. All
# operators remain affine combinations (affine span preserved; convex hull
# grows in selected directions at the frontier every generation).

from __future__ import annotations
import torch
from evox.core import Algorithm


def glued_space(x: torch.Tensor, lb: float, ub: float) -> torch.Tensor:
    return (x - lb) % (ub - lb) + lb


class PulseMixed(Algorithm):
    def __init__(
        self,
        pop_size: int,
        dim: int,
        lb: float,
        ub: float,
        σ_ray: float = 1.0,
        minimization: bool = True,
        patience: int = 1,
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
        self.minimize = minimization
        self.patience = patience
        self.debug = debug

        self.population = torch.rand(pop_size, dim, device=dev) * (ub - lb) + lb
        self.fitness = torch.empty(pop_size, device=dev)

        self.pair_a = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)
        self.pair_b = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)
        self.pair_fails = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)
        self.pair_is_new = torch.ones(self.n_pairs, dtype=torch.bool, device=dev)

    def init_step(self):
        self.fitness = self.evaluate(self.population)
        dev = self.population.device
        perm = torch.randperm(self.pop_size, device=dev)
        for k in range(self.n_pairs):
            self.pair_a[k] = perm[2 * k]
            self.pair_b[k] = perm[2 * k + 1]
        self.pair_is_new[:] = True

    def step(self):
        dev = self.population.device

        all_c1, all_c2 = [], []
        for k in range(self.n_pairs):
            ia = self.pair_a[k].item()
            ib = self.pair_b[k].item()
            p1, p2 = self.population[ia], self.population[ib]

            if self.pair_is_new[k]:
                # pulse14's symmetric borrowed-direction ray, unchanged.
                r2 = torch.randint(self.pop_size, (1,), device=dev).item()
                r3 = torch.randint(self.pop_size, (1,), device=dev).item()
                while r3 == r2:
                    r3 = torch.randint(self.pop_size, (1,), device=dev).item()
                d = self.population[r2] - self.population[r3]
                c1 = p1 + self.σ_ray * d
                c2 = p2 - self.σ_ray * d
            else:
                # THE one change: interpolant + directed extension, every gen.
                α = torch.rand(1, device=dev)
                c1 = α * p1 + (1 - α) * p2
                fa, fb = self.fitness[ia].item(), self.fitness[ib].item()
                if (fa <= fb) == self.minimize:
                    best, worst = p1, p2
                else:
                    best, worst = p2, p1
                c2 = best + (best - worst)

            all_c1.append(c1)
            all_c2.append(c2)

        offspring_c1 = glued_space(torch.stack(all_c1), self.lb, self.ub)
        offspring_c2 = glued_space(torch.stack(all_c2), self.lb, self.ub)

        all_off_fit = self.evaluate(torch.cat([offspring_c1, offspring_c2], dim=0))
        fit_c1 = all_off_fit[:self.n_pairs]
        fit_c2 = all_off_fit[self.n_pairs:]

        pairs_to_break = []
        n_productive = 0

        for k in range(self.n_pairs):
            ia = self.pair_a[k].item()
            ib = self.pair_b[k].item()

            # clone(): parent rows are views into self.population; the in-place
            # winner writes below would alias them otherwise.
            candidates_geno = [
                self.population[ia].clone(),
                self.population[ib].clone(),
                offspring_c1[k],
                offspring_c2[k],
            ]
            candidates_fit = torch.tensor([
                self.fitness[ia].item(),
                self.fitness[ib].item(),
                fit_c1[k].item(),
                fit_c2[k].item(),
            ], device=dev)

            if self.minimize:
                best2 = torch.topk(-candidates_fit, 2).indices
            else:
                best2 = torch.topk(candidates_fit, 2).indices

            winner_a = best2[0].item()
            winner_b = best2[1].item()
            offspring_survived = (winner_a >= 2) or (winner_b >= 2)

            self.population[ia] = candidates_geno[winner_a]
            self.fitness[ia] = candidates_fit[winner_a]
            self.population[ib] = candidates_geno[winner_b]
            self.fitness[ib] = candidates_fit[winner_b]

            if self.pair_is_new[k]:
                self.pair_is_new[k] = False
                self.pair_fails[k] = 0
                if offspring_survived:
                    n_productive += 1
                continue

            if offspring_survived:
                self.pair_fails[k] = 0
                n_productive += 1
            else:
                # Interior AND beyond both failed: the line is resolved.
                self.pair_fails[k] += 1
                if self.pair_fails[k] >= self.patience:
                    pairs_to_break.append(k)

        n_broken = len(pairs_to_break)
        if n_broken > 0:
            freed = []
            for k in pairs_to_break:
                freed.append(self.pair_a[k].item())
                freed.append(self.pair_b[k].item())
            perm = torch.randperm(len(freed), device=dev)
            shuffled = [freed[perm[i].item()] for i in range(len(freed))]
            for i, k in enumerate(pairs_to_break):
                if 2 * i + 1 < len(shuffled):
                    self.pair_a[k] = shuffled[2 * i]
                    self.pair_b[k] = shuffled[2 * i + 1]
                self.pair_is_new[k] = True
                self.pair_fails[k] = 0

        if self.debug:
            print(f"productive={n_productive}/{self.n_pairs}  broken={n_broken}  "
                  f"best={float(self.fitness.min()):.2f}")
        return self

    def record_step(self):
        return {"pop": self.population, "fit": self.fitness}
