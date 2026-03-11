# pulse13.py  ––  "Pulse Lineage v2"
#
# Pair-level adaptive control with EXTENSION RAY on re-pairing.
#
# Architecture (from José):
#   1. Active lineages (EMA > threshold) → GEOMETRIC crossover
#      These pairs keep interpolating — the line is still productive.
#
#   2. Exhausted lineages (EMA ≤ threshold) → BREAK the pair
#      Freed individuals go into an available pool.
#      Tournament selection among the pool (NEVER touching active pairs)
#      forms new pairs.
#
#   3. New pairs → EXTENSION RAY with the new partner
#      THIS is the exploration. Extrapolating along a FRESH direction
#      (not the dead line) genuinely expands the hull.
#
#   4. Next generation, the new pairs start tracking their own EMA
#      and default to geometric crossover.
#
# Key constraints:
#   - Active lineages are NEVER disrupted by tournament/re-pairing
#   - Only freed individuals participate in re-pairing tournament
#   - Extension ray is ONLY done with new partners, never old dead ones

from __future__ import annotations
import torch
from evox.core import Algorithm


def glued_space(x: torch.Tensor, lb: float, ub: float) -> torch.Tensor:
    return (x - lb) % (ub - lb) + lb


class PulseLineage2(Algorithm):
    def __init__(
        self,
        pop_size: int,
        dim: int,
        lb: float,
        ub: float,
        σ_ray: float = 3.0,        # extension ray step size
        tournament_size: int = 3,
        minimization: bool = True,
        # Pair-level EMA params
        β: float = 0.6,            # EMA decay for pair success
        break_threshold: float = 0.15,  # EMA below this → break pair
        min_improvement: float = 1e-4,  # Zeno filter
        device: str | torch.device | None = None,
        debug: bool = False,
    ):
        super().__init__()
        dev = torch.device(device or torch.get_default_device())

        assert pop_size % 2 == 0, "Pop size must be even."

        self.pop_size, self.dim = pop_size, dim
        self.n_pairs = pop_size // 2
        self.lb, self.ub = lb, ub
        self.σ_ray = σ_ray
        self.tour = tournament_size
        self.minimize = minimization
        self.debug = debug

        self.population = torch.rand(pop_size, dim, device=dev) * (ub - lb) + lb
        self.fitness = torch.empty(pop_size, device=dev)

        self.β = β
        self.break_threshold = break_threshold
        self.min_improvement = min_improvement

        # Pair state
        self.pair_a = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)
        self.pair_b = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)
        self.pair_ema = torch.full((self.n_pairs,), 0.5, device=dev)
        self.pair_age = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)
        # Track which pairs are "new" (just re-paired) → these do ray
        self.pair_is_new = torch.zeros(self.n_pairs, dtype=torch.bool, device=dev)

    # ---------- Tournament among a pool (fitness-based) ----------------------
    def _tournament_select(self, pool: list[int]) -> int:
        """Select one individual from pool via tournament."""
        dev = self.population.device
        if len(pool) <= self.tour:
            candidates = pool
        else:
            indices = torch.randperm(len(pool), device=dev)[:self.tour]
            candidates = [pool[i.item()] for i in indices]

        cand_t = torch.tensor(candidates, device=dev)
        fn = torch.argmin if self.minimize else torch.argmax
        return cand_t[fn(self.fitness[cand_t])].item()

    # ---------- Form pairs from available pool via tournament ----------------
    def _form_pairs_tournament(self, available: list[int]):
        """Pair up available individuals using tournament selection.
        Returns lists of (a, b) pairs."""
        pairs_a, pairs_b = [], []
        pool = list(available)

        while len(pool) >= 2:
            # Select first parent
            a = self._tournament_select(pool)
            pool.remove(a)

            # Select second parent
            b = self._tournament_select(pool)
            pool.remove(b)

            pairs_a.append(a)
            pairs_b.append(b)

        return pairs_a, pairs_b

    # -------------------------------------------------------------------------
    def init_step(self):
        self.fitness = self.evaluate(self.population)

        # Form initial pairs: random shuffled pairing
        dev = self.population.device
        perm = torch.randperm(self.pop_size, device=dev)
        for k in range(self.n_pairs):
            self.pair_a[k] = perm[2 * k]
            self.pair_b[k] = perm[2 * k + 1]
        # All initial pairs are "new" → first generation uses ray to establish directions
        self.pair_is_new[:] = True

    def step(self):
        dev = self.population.device

        # ----- Generate offspring per pair -----
        children = []
        for k in range(self.n_pairs):
            ia = self.pair_a[k].item()
            ib = self.pair_b[k].item()
            p1 = self.population[ia]
            p2 = self.population[ib]

            if self.pair_is_new[k]:
                # NEW pair → extension ray (explore fresh direction)
                d = p2 - p1
                c1 = p2 + self.σ_ray * d
                c2 = p1 - self.σ_ray * d
            else:
                # ACTIVE lineage → geometric (interpolate along productive line)
                α = torch.rand(1, device=dev)
                c1 = α * p1 + (1 - α) * p2
                c2 = (1 - α) * p1 + α * p2

            children.extend([c1, c2])

        offspring = glued_space(torch.stack(children), self.lb, self.ub)

        # ----- Evaluate offspring -----
        off_fit = self.evaluate(offspring)

        # ----- Update pair EMAs (only for non-new pairs with geo) -----
        for k in range(self.n_pairs):
            if self.pair_is_new[k]:
                # New pair just did ray — don't update EMA yet, just transition to geo
                self.pair_is_new[k] = False
                continue

            ia = self.pair_a[k].item()
            ib = self.pair_b[k].item()

            par_best = min(self.fitness[ia].item(), self.fitness[ib].item()) if self.minimize \
                       else max(self.fitness[ia].item(), self.fitness[ib].item())

            c1_fit = off_fit[2 * k].item()
            c2_fit = off_fit[2 * k + 1].item()
            off_best = min(c1_fit, c2_fit) if self.minimize else max(c1_fit, c2_fit)

            rel_imp = (par_best - off_best) / (abs(par_best) + 1e-12) if self.minimize \
                      else (off_best - par_best) / (abs(par_best) + 1e-12)
            success = 1.0 if rel_imp > self.min_improvement else 0.0

            self.pair_ema[k] = self.β * self.pair_ema[k] + (1 - self.β) * success
            self.pair_age[k] += 1

        # ----- Global (μ+λ) selection -----
        comb_pop = torch.cat([self.population, offspring])
        comb_fit = torch.cat([self.fitness, off_fit])
        best_idx = torch.topk(-comb_fit if self.minimize else comb_fit, self.pop_size).indices
        self.population = comb_pop[best_idx]
        self.fitness = comb_fit[best_idx]

        # ----- Map old indices → new indices -----
        old_to_new = torch.full((2 * self.pop_size,), -1, dtype=torch.long, device=dev)
        for new_i in range(self.pop_size):
            old_to_new[best_idx[new_i].item()] = new_i

        # ----- Determine which pairs survive vs break -----
        pairs_to_break = []
        surviving_in_pair = set()

        for k in range(self.n_pairs):
            old_a = self.pair_a[k].item()
            old_b = self.pair_b[k].item()

            new_a = old_to_new[old_a].item()
            new_b = old_to_new[old_b].item()

            off_c1 = old_to_new[self.pop_size + 2 * k].item()
            off_c2 = old_to_new[self.pop_size + 2 * k + 1].item()

            # Resolve A: parent survived, or offspring takes its place
            resolved_a = new_a if new_a >= 0 else (off_c1 if off_c1 >= 0 else (off_c2 if off_c2 >= 0 else -1))
            # Resolve B: can't be same as A
            resolved_b = -1
            if new_b >= 0 and new_b != resolved_a:
                resolved_b = new_b
            elif off_c2 >= 0 and off_c2 != resolved_a:
                resolved_b = off_c2
            elif off_c1 >= 0 and off_c1 != resolved_a:
                resolved_b = off_c1

            # Break if:
            # - Members didn't survive selection
            # - EMA dropped below threshold (line is exhausted)
            should_break = (resolved_a < 0 or resolved_b < 0 or
                           self.pair_ema[k] < self.break_threshold)

            if should_break:
                pairs_to_break.append(k)
            else:
                self.pair_a[k] = resolved_a
                self.pair_b[k] = resolved_b
                surviving_in_pair.add(resolved_a)
                surviving_in_pair.add(resolved_b)

        # ----- Re-pair broken pairs -----
        # ONLY freed individuals participate. Active lineages are NEVER touched.
        n_broken = len(pairs_to_break)
        if n_broken > 0:
            available = [i for i in range(self.pop_size) if i not in surviving_in_pair]

            if len(available) >= 2:
                a_list, b_list = self._form_pairs_tournament(available)
                for i, k in enumerate(pairs_to_break):
                    if i < len(a_list):
                        self.pair_a[k] = a_list[i]
                        self.pair_b[k] = b_list[i]
                        surviving_in_pair.add(a_list[i])
                        surviving_in_pair.add(b_list[i])
                    else:
                        remain = [x for x in range(self.pop_size) if x not in surviving_in_pair]
                        if len(remain) >= 2:
                            self.pair_a[k] = remain[0]
                            self.pair_b[k] = remain[1]
                            surviving_in_pair.add(remain[0])
                            surviving_in_pair.add(remain[1])

                    # Mark as NEW → next step will use EXTENSION RAY
                    self.pair_is_new[k] = True
                    self.pair_ema[k] = 0.5   # reset EMA
                    self.pair_age[k] = 0

        if self.debug:
            n_geo = int((~self.pair_is_new).sum().item()) - n_broken  # active geo pairs
            n_ray_next = int(self.pair_is_new.sum().item())  # will do ray next step
            avg_ema = float(self.pair_ema[~self.pair_is_new].mean()) if (~self.pair_is_new).any() else 0
            avg_age = float(self.pair_age.float().mean())
            print(
                f"geo={n_geo}  broke→ray_next={n_broken}  "
                f"ema_active={avg_ema:.3f}  age_avg={avg_age:.1f}  "
                f"best={float(self.fitness.min()):.2f}"
            )

        return self

    def record_step(self):
        return {
            "pop": self.population, "fit": self.fitness,
            "pair_ema": self.pair_ema.clone(), "pair_age": self.pair_age.clone()
        }
