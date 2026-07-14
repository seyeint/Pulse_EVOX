# pulse15v2.py  ––  "Pulse Directed v2": gated single-shot pattern moves
#
# v1 post-mortem (777-iter CEC race): directed extension produced a strong
# early-phase sprint and two outright function wins (f7, f8), but lost the
# endgame — operator mix showed 62% extension generations. Three diagnosed
# failures, three fixes:
#
#   1. HEDGED ORIENTED RAY. v1 moved both members by +sigma*d, which keeps the
#      offspring secant identical to the parent secant (no direction refresh)
#      and commits both members to one bet. v2: C1 = A + sigma*d (informed bet,
#      d oriented worse-donor -> better-donor), C2 = B - sigma*d (hedge).
#      Secant refresh and hedging restored, orientation edge kept on C1.
#
#   2. EVIDENCE-GATED EXTENSION. v1 extended after a single parent-best event,
#      which fires on alpha-draw flukes and plateau ties. v2 requires TWO
#      consecutive parent-best generations in contract mode before extending.
#
#   3. SINGLE-SHOT EXTENSION (kills direction monogamy). In v1, a pair whose
#      extensions kept succeeding stayed in extend mode indefinitely — never
#      breaking, never re-pairing, freezing the population's direction mixing.
#      v2 makes the extension a one-generation PROBE: success moves the bracket
#      and returns to contraction (re-bracket, refine); failure counts toward
#      eviction. This is exactly Hooke-Jeeves: exploratory moves, one pattern
#      move, re-explore.
#
# Automaton:
#   new      -> hedged oriented ray -> contract (streak=0)
#   contract -> offspring-best => streak=0; parent-best => streak++;
#               streak >= 2 => next generation is an extension probe
#   probe    -> children at Best + u and Best + 2u (u = Best - Worst);
#               ALWAYS return to contract afterward;
#               success => fails=0 (bracket moved); failure => fails++,
#               at patience => break
#   break    -> re-pair among freed pool -> new

from __future__ import annotations
import torch
from evox.core import Algorithm


def glued_space(x: torch.Tensor, lb: float, ub: float) -> torch.Tensor:
    return (x - lb) % (ub - lb) + lb


class PulseDirectedV2(Algorithm):
    def __init__(
        self,
        pop_size: int,
        dim: int,
        lb: float,
        ub: float,
        σ_ray: float = 1.0,
        minimization: bool = True,
        patience: int = 1,    # failed extension probes before breaking
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
        self.pair_extend = torch.zeros(self.n_pairs, dtype=torch.bool, device=dev)
        self.pair_streak = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)

        # tensor-backed so counts survive evox module wrapping: ray, contract, extend
        self.op_counts = torch.zeros(3, dtype=torch.long, device=dev)

    def init_step(self):
        self.fitness = self.evaluate(self.population)
        dev = self.population.device
        perm = torch.randperm(self.pop_size, device=dev)
        for k in range(self.n_pairs):
            self.pair_a[k] = perm[2 * k]
            self.pair_b[k] = perm[2 * k + 1]
        self.pair_is_new[:] = True
        self.pair_extend[:] = False
        self.pair_streak[:] = 0

    def _better_worse(self, i, j):
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
                # Hedged oriented ray: informed bet + contrarian hedge.
                self.op_counts[0] += 1
                r2 = torch.randint(self.pop_size, (1,), device=dev).item()
                r3 = torch.randint(self.pop_size, (1,), device=dev).item()
                while r3 == r2:
                    r3 = torch.randint(self.pop_size, (1,), device=dev).item()
                hi, lo = self._better_worse(r2, r3)
                d = self.population[hi] - self.population[lo]
                c1 = p1 + self.σ_ray * d
                c2 = p2 - self.σ_ray * d
            elif self.pair_extend[k]:
                # Single-shot pattern move past the better member.
                self.op_counts[2] += 1
                bi, wi = self._better_worse(ia, ib)
                u = self.population[bi] - self.population[wi]
                c1 = self.population[bi] + u
                c2 = self.population[bi] + 2.0 * u
            else:
                self.op_counts[1] += 1
                α = torch.rand(1, device=dev)
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

            order = torch.argsort(candidates_fit) if self.minimize \
                else torch.argsort(candidates_fit, descending=True)
            winner_a, winner_b = order[0].item(), order[1].item()
            offspring_best = winner_a >= 2

            self.population[ia] = candidates_geno[winner_a]
            self.fitness[ia] = candidates_fit[winner_a]
            self.population[ib] = candidates_geno[winner_b]
            self.fitness[ib] = candidates_fit[winner_b]

            if offspring_best:
                n_productive += 1

            if self.pair_is_new[k]:
                self.pair_is_new[k] = False
                self.pair_extend[k] = False
                self.pair_streak[k] = 0
                self.pair_fails[k] = 0
            elif self.pair_extend[k]:
                # Probe complete — always back to contraction.
                self.pair_extend[k] = False
                self.pair_streak[k] = 0
                if offspring_best:
                    self.pair_fails[k] = 0
                else:
                    self.pair_fails[k] += 1
                    if self.pair_fails[k] >= self.patience:
                        pairs_to_break.append(k)
            else:  # contract mode
                if offspring_best:
                    self.pair_streak[k] = 0
                else:
                    self.pair_streak[k] += 1
                    if self.pair_streak[k] >= 2:
                        self.pair_extend[k] = True
                        self.pair_streak[k] = 0

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
                self.pair_streak[k] = 0
                self.pair_fails[k] = 0

        if self.debug:
            print(f"productive={n_productive}/{self.n_pairs}  broken={n_broken}  "
                  f"probing={int(self.pair_extend.sum())}  "
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
