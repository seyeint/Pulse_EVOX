# pulse12.py  ––  "Pulse Lineage"
#
# PAIR-LEVEL adaptive control. No global ρ.
#
# Core idea (from José):
# Instead of one global ρ averaging signal across 200 unrelated pairs,
# each pair is a PERSISTENT LINEAGE with its own improvement EMA.
#
# A pair keeps interpolating (geo) as long as it's productive. When
# its own EMA drops below threshold → ray + re-pair. This is
# intentional inbreeding: exploit each productive line until it dries up.
#
# This maximizes signal-to-noise:
#   - No averaging across unrelated pairs
#   - Each pair independently senses its own local topology
#   - Productive lines are preserved (compounding improvement)
#   - Exhausted lines are broken (exploration via ray + new partner)
#
# Mechanically:
#   - Population of N individuals, organized into N/2 persistent pairs
#   - Each pair has its own EMA of geo improvement rate
#   - High EMA → geo (keep interpolating this line)
#   - Low EMA → ray (escape this line, break the pair next gen)
#   - After global (μ+λ) selection, update pair memberships
#   - Dissolved pairs re-form via tournament selection

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


class PulseLineage(Algorithm):
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
        # Pair-level EMA params
        β: float = 0.7,            # EMA decay for pair success tracking
        break_threshold: float = 0.2,  # EMA below this → break the pair
        min_improvement: float = 1e-4, # Zeno filter
        device: str | torch.device | None = None,
        debug: bool = False,
    ):
        super().__init__()
        dev = torch.device(device or torch.get_default_device())

        assert pop_size % 2 == 0, "Pop size must be even."

        self.pop_size, self.dim = pop_size, dim
        self.n_pairs = pop_size // 2
        self.lb, self.ub = lb, ub
        self.tour = tournament_size
        self.minimize = minimization
        self.debug = debug

        self.population = torch.rand(pop_size, dim, device=dev) * (ub - lb) + lb
        self.fitness = torch.empty(pop_size, device=dev)

        self.σ_min, self.σ_max = σ_min, σ_max

        self.β = β
        self.break_threshold = break_threshold
        self.min_improvement = min_improvement

        # Pair state: each pair is (idx_a, idx_b) with its own EMA
        # Initialized in init_step after first evaluation
        self.pair_a = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)
        self.pair_b = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)
        self.pair_ema = torch.full((self.n_pairs,), 0.5, device=dev)  # start neutral
        self.pair_age = torch.zeros(self.n_pairs, dtype=torch.long, device=dev)

    # ---------- Pair formation via exclusive tournament --------------------
    def _form_pairs(self, indices: torch.Tensor | None = None):
        """Form pairs from the given indices (or entire population).
        Uses exclusive tournament selection: each individual selected at most once."""
        dev = self.population.device

        if indices is None:
            # Form pairs for entire population
            n = self.pop_size
            perm = torch.randperm(n, device=dev)
            n_pairs = n // 2

            a_indices = []
            b_indices = []
            used = set()

            for i in range(n_pairs):
                # Select parent A via tournament
                candidates = []
                for j in range(n):
                    idx = perm[(i * self.tour + j) % n].item()
                    if idx not in used:
                        candidates.append(idx)
                    if len(candidates) >= self.tour:
                        break
                if not candidates:
                    remaining = [j for j in range(n) if j not in used]
                    candidates = [remaining[0]] if remaining else [i]

                cand_t = torch.tensor(candidates, device=dev)
                fn = torch.argmin if self.minimize else torch.argmax
                winner = cand_t[fn(self.fitness[cand_t])].item()
                a_indices.append(winner)
                used.add(winner)

            # Second pass for B partners
            perm2 = torch.randperm(n, device=dev)
            for i in range(n_pairs):
                candidates = []
                for j in range(n):
                    idx = perm2[(i * self.tour + j) % n].item()
                    if idx not in used:
                        candidates.append(idx)
                    if len(candidates) >= self.tour:
                        break
                if not candidates:
                    remaining = [j for j in range(n) if j not in used]
                    candidates = [remaining[0]] if remaining else [i]

                cand_t = torch.tensor(candidates, device=dev)
                fn = torch.argmin if self.minimize else torch.argmax
                winner = cand_t[fn(self.fitness[cand_t])].item()
                b_indices.append(winner)
                used.add(winner)

            return (torch.tensor(a_indices, device=dev),
                    torch.tensor(b_indices, device=dev))
        else:
            # Form pairs from a subset of indices
            n = len(indices)
            if n < 2:
                return torch.tensor([], dtype=torch.long, device=dev), \
                       torch.tensor([], dtype=torch.long, device=dev)
            perm = indices[torch.randperm(n, device=dev)]
            n_pairs = n // 2
            return perm[:n_pairs], perm[n_pairs:2*n_pairs]

    # -------------------------------------------------------------------------
    def init_step(self):
        self.fitness = self.evaluate(self.population)

        # Form initial pairs
        self.pair_a, self.pair_b = self._form_pairs()

    def step(self):
        dev = self.population.device

        # ----- Per-pair operator decision -----
        # High EMA → geometric (interpolate this productive line)
        # Low EMA → ray (this line is exhausted, escape)
        use_ray = self.pair_ema < self.break_threshold

        # σ per pair: exhausted pairs get larger σ for ray escape
        σ_base = self.σ_min + (self.σ_max - self.σ_min) * 0.3  # moderate for geo
        σ_ray = self.σ_max  # full extension for ray

        # ----- Generate offspring -----
        children = []
        for k in range(self.n_pairs):
            ia = self.pair_a[k].item()
            ib = self.pair_b[k].item()
            p1 = self.population[ia]
            p2 = self.population[ib]

            if use_ray[k]:
                σ = σ_ray
                d = p2 - p1
                c1 = p2 + σ * d
                c2 = p1 - σ * d
            else:
                α = torch.rand(1, device=dev)
                c1 = α * p1 + (1 - α) * p2
                c2 = (1 - α) * p1 + α * p2

            children.extend([c1, c2])

        offspring = glued_space(torch.stack(children), self.lb, self.ub)

        # ----- Evaluate offspring -----
        off_fit = self.evaluate(offspring)

        # ----- Update pair EMAs (before global selection) -----
        for k in range(self.n_pairs):
            ia = self.pair_a[k].item()
            ib = self.pair_b[k].item()

            # Best parent fitness
            par_best = min(self.fitness[ia].item(), self.fitness[ib].item()) if self.minimize \
                       else max(self.fitness[ia].item(), self.fitness[ib].item())

            # Best offspring fitness
            c1_fit = off_fit[2 * k].item()
            c2_fit = off_fit[2 * k + 1].item()
            off_best = min(c1_fit, c2_fit) if self.minimize else max(c1_fit, c2_fit)

            # Relative improvement (Zeno-filtered)
            rel_imp = (par_best - off_best) / (abs(par_best) + 1e-12) if self.minimize \
                      else (off_best - par_best) / (abs(par_best) + 1e-12)
            success = 1.0 if rel_imp > self.min_improvement else 0.0

            # Update THIS pair's EMA
            self.pair_ema[k] = self.β * self.pair_ema[k] + (1 - self.β) * success
            self.pair_age[k] += 1

        # ----- Global (μ+λ) selection -----
        comb_pop = torch.cat([self.population, offspring])
        comb_fit = torch.cat([self.fitness, off_fit])
        best_idx = torch.topk(-comb_fit if self.minimize else comb_fit, self.pop_size).indices
        self.population = comb_pop[best_idx]
        self.fitness = comb_fit[best_idx]

        # ----- Update pair memberships -----
        # After selection, individuals have new indices. We need to map
        # old indices → new indices. An individual "survives" if it's in
        # the selected set.
        #
        # Create a mapping: old_index → new_index (or -1 if not selected)
        # Old indices: 0..pop_size-1 = parents, pop_size..2*pop_size-1 = offspring
        old_to_new = torch.full((2 * self.pop_size,), -1, dtype=torch.long, device=dev)
        for new_i in range(self.pop_size):
            old_i = best_idx[new_i].item()
            old_to_new[old_i] = new_i

        # Map pair members through selection
        pairs_to_break = []
        for k in range(self.n_pairs):
            old_a = self.pair_a[k].item()
            old_b = self.pair_b[k].item()

            # Check if parents survived
            new_a = old_to_new[old_a].item()
            new_b = old_to_new[old_b].item()

            # Check if offspring survived (could replace parent in pair)
            off_a = old_to_new[self.pop_size + 2 * k].item()
            off_b = old_to_new[self.pop_size + 2 * k + 1].item()

            # Priority: keep original parent if survived, else use offspring
            if new_a >= 0:
                self.pair_a[k] = new_a
            elif off_a >= 0:
                self.pair_a[k] = off_a  # offspring takes parent's place
            elif off_b >= 0:
                self.pair_a[k] = off_b
            else:
                pairs_to_break.append(k)
                continue

            if new_b >= 0:
                self.pair_b[k] = new_b
            elif off_b >= 0:
                self.pair_b[k] = off_b
            elif off_a >= 0 and self.pair_a[k].item() != off_a:
                self.pair_b[k] = off_a
            else:
                pairs_to_break.append(k)
                continue

            # Also break pairs that have exhausted EMA AND used ray
            if use_ray[k] and self.pair_ema[k] < self.break_threshold:
                pairs_to_break.append(k)

        # ----- Re-pair broken pairs -----
        if pairs_to_break:
            # Collect all individuals that need new partners
            used_in_pairs = set()
            for k in range(self.n_pairs):
                if k not in pairs_to_break:
                    used_in_pairs.add(self.pair_a[k].item())
                    used_in_pairs.add(self.pair_b[k].item())

            available = [i for i in range(self.pop_size) if i not in used_in_pairs]

            if len(available) >= 2:
                # Shuffle and pair up
                perm = torch.tensor(available, device=dev)[torch.randperm(len(available), device=dev)]
                for i, k in enumerate(pairs_to_break):
                    if 2 * i + 1 < len(perm):
                        self.pair_a[k] = perm[2 * i]
                        self.pair_b[k] = perm[2 * i + 1]
                        self.pair_ema[k] = 0.5  # reset EMA for new pair
                        self.pair_age[k] = 0
                    else:
                        # Not enough individuals, pick random
                        self.pair_a[k] = torch.randint(self.pop_size, (1,), device=dev).item()
                        self.pair_b[k] = torch.randint(self.pop_size, (1,), device=dev).item()
                        while self.pair_b[k] == self.pair_a[k]:
                            self.pair_b[k] = torch.randint(self.pop_size, (1,), device=dev).item()
                        self.pair_ema[k] = 0.5
                        self.pair_age[k] = 0

        if self.debug:
            n_geo = int((~use_ray).sum().item())
            n_ray = int(use_ray.sum().item())
            n_broken = len(pairs_to_break) if pairs_to_break else 0
            avg_ema = float(self.pair_ema.mean())
            avg_age = float(self.pair_age.float().mean())
            print(
                f"geo={n_geo}  ray={n_ray}  broken={n_broken}  "
                f"ema_avg={avg_ema:.3f}  age_avg={avg_age:.1f}  "
                f"best={float(self.fitness.min()):.2f}"
            )

        return self

    def record_step(self):
        return {
            "pop": self.population, "fit": self.fitness,
            "pair_ema": self.pair_ema.clone(), "pair_age": self.pair_age.clone()
        }
