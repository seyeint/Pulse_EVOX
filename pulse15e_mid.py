# pulse15e_mid.py  ––  KNOCKOUT of P15e: deterministic contraction at fixed quartiles (dither removed)
# (ablation control; single gene removed from the barebones)
# original header:
# contract as the fallback, then evict.
#   new -> ray -> EXTEND (ride the pair's own worse->better direction)
#   extend:   offspring-best => keep riding;
#             parent-best    => switch to CONTRACT (both exterior probes worse
#                               means whatever is left is interior — not a failure)
#   contract: offspring-best => keep contracting;
#             parent-best    => interior worse AND beyond worse: resolved ->
#                               fails += 1; at patience => break
# Justification to test: v1's measured mix (61% extend / 25% contract) says
# monotone chords are the majority pair-lifecycle, so contract-first burns a
# hypothesis test on the minority case. Same automaton, prior flipped.
#
# pulse14 + two changes born from the hull-dynamics diagnosis:
#
#   1. MISCLASSIFICATION REPAIR. pulse14 treats "a parent is best" as evidence
#      the line is dead. But parent-best is the signature of a MONOTONE chord:
#      the 1-D minimum lies BEYOND the better parent, and interior children
#      lose by geometry, not exhaustion. pulse14 grinds or evicts exactly the
#      lines it should ride. pulse15 responds by extrapolating past the better
#      parent along the pair's own secant (Wright's heuristic crossover /
#      Nelder-Mead expansion / Hooke-Jeeves pattern move). Eviction only fires
#      when an extension probe fails: interior worse AND beyond worse means the
#      line is fully resolved.
#
#   2. ORIENTED RAYS. Re-pair rays point from the worse donor to the better
#      donor (the sign information is free), applied to both members, instead
#      of firing symmetric +/- lotteries.
#
# Per-pair automaton:
#   new      -> oriented ray -> contract
#   contract -> interpolation children; offspring-best => stay in contract;
#               parent-best => switch to EXTEND (not a failure!)
#   extend   -> children at  Best + u  and  Best + 2u,  u = Best - Worst;
#               extension child survives => stay in extend (step growth emerges
#               from secant renewal: {C2,B} pairs double the secant per gen);
#               parent-best (probe failed) => fails += 1; at patience => break
#   break    -> re-pair among freed pool -> new
#
# Every operator remains an affine combination: the affine span of the initial
# population is preserved (zero ambient diffusion, cf. the off-manifold drift
# of isotropic ES), while the convex hull now grows in SELECTED directions at
# the frontier instead of only via random re-pair rays.

from __future__ import annotations
import torch
from evox.core import Algorithm


def glued_space(x: torch.Tensor, lb: float, ub: float) -> torch.Tensor:
    return (x - lb) % (ub - lb) + lb


class PulseKmid(Algorithm):
    def __init__(
        self,
        pop_size: int,
        dim: int,
        lb: float,
        ub: float,
        σ_ray: float = 1.0,
        minimization: bool = True,
        patience: int = 1,    # consecutive failed extension probes before breaking
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

        # Pair state
        self.pair_a = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)
        self.pair_b = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)
        self.pair_fails = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)
        self.pair_is_new = torch.ones(self.n_pairs, dtype=torch.bool, device=dev)
        self.pair_extend = torch.zeros(self.n_pairs, dtype=torch.bool, device=dev)

        # diagnostics — tensor-backed so counts survive evox's module
        # wrapping (in-place tensor writes are shared; python ints are not).
        self.op_counts = torch.zeros(3, dtype=torch.long, device=dev)  # ray, contract, extend

    def init_step(self):
        self.fitness = self.evaluate(self.population)
        dev = self.population.device
        perm = torch.randperm(self.pop_size, device=dev)
        for k in range(self.n_pairs):
            self.pair_a[k] = perm[2 * k]
            self.pair_b[k] = perm[2 * k + 1]
        self.pair_is_new[:] = True
        self.pair_extend[:] = False

    def _better_worse(self, i, j):
        """Return (better_idx, worse_idx) under the optimization sense."""
        fi, fj = self.fitness[i].item(), self.fitness[j].item()
        if (fi <= fj) == self.minimize:
            return i, j
        return j, i

    def step(self):
        dev = self.population.device

        all_c1, all_c2 = [], []
        for k in range(self.n_pairs):
            ia = self.pair_a[k].item()
            ib = self.pair_b[k].item()
            p1, p2 = self.population[ia], self.population[ib]

            if self.pair_is_new[k]:
                # Oriented ray: borrowed direction, pointed worse -> better.
                self.op_counts[0] += 1
                r2 = torch.randint(self.pop_size, (1,), device=dev).item()
                r3 = torch.randint(self.pop_size, (1,), device=dev).item()
                while r3 == r2:
                    r3 = torch.randint(self.pop_size, (1,), device=dev).item()
                hi, lo = self._better_worse(r2, r3)
                d = self.population[hi] - self.population[lo]
                c1 = p1 + self.σ_ray * d
                c2 = p2 + self.σ_ray * d
            elif self.pair_extend[k]:
                # Directed extension past the better member along the secant.
                self.op_counts[2] += 1
                bi, wi = self._better_worse(ia, ib)
                u = self.population[bi] - self.population[wi]
                c1 = self.population[bi] + u
                c2 = self.population[bi] + 2.0 * u
            else:
                # Geometric crossover (contraction).
                self.op_counts[1] += 1
                α = torch.tensor([0.25], device=dev)
                c1 = α * p1 + (1 - α) * p2
                c2 = (1 - α) * p1 + α * p2

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

            # clone() parent rows: they are views into self.population and the
            # in-place winner writes below would alias them.
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
                order = torch.argsort(candidates_fit)
            else:
                order = torch.argsort(candidates_fit, descending=True)
            best_of4 = order[0].item()
            winner_a, winner_b = order[0].item(), order[1].item()
            offspring_best = best_of4 >= 2

            self.population[ia] = candidates_geno[winner_a]
            self.fitness[ia] = candidates_fit[winner_a]
            self.population[ib] = candidates_geno[winner_b]
            self.fitness[ib] = candidates_fit[winner_b]

            if offspring_best:
                n_productive += 1

            if self.pair_is_new[k]:
                # Ray fired; move to EXTEND first (monotone chord is the
                # measured majority case).
                self.pair_is_new[k] = False
                self.pair_extend[k] = True
                self.pair_fails[k] = 0
            elif self.pair_extend[k]:
                if offspring_best:
                    self.pair_fails[k] = 0
                else:
                    # Beyond is worse: if anything is left it is interior.
                    self.pair_extend[k] = False
                    self.pair_fails[k] = 0
            else:  # contract mode (the fallback now)
                if offspring_best:
                    self.pair_fails[k] = 0
                else:
                    # Interior worse AND beyond worse: line resolved.
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
                self.pair_extend[k] = False
                self.pair_fails[k] = 0

        if self.debug:
            print(f"productive={n_productive}/{self.n_pairs}  broken={n_broken}  "
                  f"extending={int(self.pair_extend.sum())}  "
                  f"best={float(self.fitness.min()):.2f}")
        return self

    @property
    def n_ray_gens(self):
        return int(self.op_counts[0])

    @property
    def n_contract_gens(self):
        return int(self.op_counts[1])

    @property
    def n_extend_gens(self):
        return int(self.op_counts[2])

    def record_step(self):
        return {"pop": self.population, "fit": self.fitness}
