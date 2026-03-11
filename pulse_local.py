# pulse_failure_repair.py ────────────────────────────────────────────
# Failure-repair Pulse: Geo on lineages until stall, then defer pulse to re-paired failure pool.
#
# Key: Geo contraction on pairs until no offspring > both parents.
#      Stall → orphan; end-gen re-pair orphans (stalled with stalled), apply non-geo ray.
#      Successful lineages persist with geo; adaptive σ per-lineage.
#
# Dependencies: torch ≥2.1, evox.
# --------------------------------------------------------------------

from __future__ import annotations
import torch
from evox.core import Algorithm

# Helpers
def glued_space(x: torch.Tensor, lb: float, ub: float) -> torch.Tensor:
    return (x - lb) % (ub - lb) + lb

def geometric(p1, p2):
    α = torch.rand(1, device=p1.device)
    return α * p1 + (1 - α) * p2, (1 - α) * p1 + α * p2

def ray(p1, p2, σ):
    d = p2 - p1
    return p2 + σ * d, p1 - σ * d

class PulseFailureRepair(Algorithm):
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
        H: int = 5,  # Progress buffer for per-lineage σ
        γ: float = 5.0,
        δ_frac: float = 1e-3,
        device: str | torch.device | None = None,
        debug: bool = False,
    ):
        super().__init__()
        dev = torch.device(device or torch.get_default_device())

        assert pop_size % 2 == 0, "Pop size must be even."

        # Settings
        self.pop_size, self.dim = pop_size, dim
        self.lb, self.ub = lb, ub
        self.tour = tournament_size
        self.minimize = minimization
        self.debug = debug

        # Population
        self.population = torch.rand(pop_size, dim, device=dev) * (ub - lb) + lb
        self.fitness = torch.empty(pop_size, device=dev)

        # Pairing: partner[i] = j or -1 (unpaired)
        self.partner = torch.full((pop_size,), -1, dtype=torch.long, device=dev)
        self._initial_pairing()

        # Per-pair σ and progress
        self.pair_σ = torch.full((pop_size // 2,), σ_max, device=dev)
        self.σ_min, self.σ_max = σ_min, σ_max
        self.H, self.γ, self.δ_frac = H, γ, δ_frac
        self.delta_hist = [[] for _ in range(pop_size // 2)]  # Per-pair buffer
        self.pair_id = torch.full((pop_size,), -1, dtype=torch.long, device=dev)
        self._assign_pair_ids()

    def _initial_pairing(self):
        idx = torch.randperm(self.pop_size, device=self.population.device)
        for k in range(0, self.pop_size, 2):
            a, b = idx[k].item(), idx[k + 1].item()
            self.partner[a] = b
            self.partner[b] = a

    def _assign_pair_ids(self):
        pid = 0
        for i in range(self.pop_size):
            j = self.partner[i].item()
            if j > i:
                self.pair_id[i] = self.pair_id[j] = pid
                pid += 1

    def _tournament(self, candidates: torch.Tensor):
        cand_fit = self.fitness[candidates]
        fn = torch.argmin if self.minimize else torch.argmax
        return candidates[fn(cand_fit)]

    def init_step(self):
        self.fitness = self.evaluate(self.population)

    def step(self):
        new_pop = self.population.clone()
        new_fit = self.fitness.clone()
        processed = torch.zeros(self.pop_size, dtype=torch.bool, device=self.population.device)
        unpaired = []

        # Process pairs
        for i in range(self.pop_size):
            if processed[i]:
                continue
            j = self.partner[i].item()
            if j < 0:
                unpaired.append(i)
                continue
            processed[i] = processed[j] = True
            pid = self.pair_id[i].item()

            # Geo crossover
            c1, c2 = geometric(self.population[i], self.population[j])
            c1, c2 = glued_space(c1, self.lb, self.ub), glued_space(c2, self.lb, self.ub)
            f1, f2 = self.evaluate(torch.stack([c1, c2]))

            par_fit = torch.stack([self.fitness[i], self.fitness[j]])
            off_fit = torch.stack([f1, f2])

            # Success: At least one offspring better than best parent AND different from both parents
            best_parent = par_fit.min() if self.minimize else par_fit.max()
            better = off_fit < best_parent if self.minimize else off_fit > best_parent
            
            # Check if offspring are different from parents
            epsilon = 1e-6
            is_eq = torch.stack([
                torch.all(torch.abs(c1 - self.population[i]) < epsilon) | 
                torch.all(torch.abs(c1 - self.population[j]) < epsilon),
                torch.all(torch.abs(c2 - self.population[i]) < epsilon) | 
                torch.all(torch.abs(c2 - self.population[j]) < epsilon),
            ], dim=0)
            
            success = torch.any(~is_eq & better)

            # Update local progress and σ
            δ = torch.max(torch.abs(par_fit.mean() - off_fit)) if success else 0.0
            self.delta_hist[pid].append(float(δ))
            if len(self.delta_hist[pid]) > self.H:
                self.delta_hist[pid].pop(0)
            if self.delta_hist[pid]:
                Δ = sum(self.delta_hist[pid]) / len(self.delta_hist[pid])
                Δ_rel = Δ / (abs(par_fit.mean()) + 1e-12)
                z = self.γ * (Δ_rel - self.δ_frac)
                ρ = float(torch.sigmoid(torch.tensor(-z, device=self.population.device)))
                self.pair_σ[pid] = self.σ_min * (1 - ρ) + self.σ_max * ρ

            if success:
                # Continue lineage with offspring
                new_pop[i], new_pop[j] = c1, c2
                new_fit[i], new_fit[j] = f1, f2
            else:
                # Stall: Mark unpaired for re-pairing (defer pulse)
                self.partner[i] = self.partner[j] = -1
                self.pair_id[i] = self.pair_id[j] = -1
                self.delta_hist[pid] = []
                unpaired.extend([i, j])

        # Re-pair unpaired (defer to end-gen)
        if unpaired:
            unpaired_tensor = torch.tensor(unpaired, device=self.population.device)
            perm = torch.randperm(len(unpaired_tensor), device=self.population.device)
            unpaired_tensor = unpaired_tensor[perm]
            unpaired = unpaired_tensor.tolist()
        pid_counter = self.pair_id.max().item() + 1 if self.pair_id.max() >= 0 else 0

        while len(unpaired) >= 2:
            a, b = unpaired.pop(0), unpaired.pop(0)
            self.partner[a] = b
            self.partner[b] = a
            self.pair_id[a] = self.pair_id[b] = pid_counter
            self.delta_hist.append([])
            self.pair_σ = torch.cat([self.pair_σ, torch.tensor([self.σ_max], device=self.pair_σ.device)])
            # Pulse non-geo on new pair
            r1, r2 = ray(new_pop[a], new_pop[b], 1.0)
            r1, r2 = glued_space(r1, self.lb, self.ub), glued_space(r2, self.lb, self.ub)
            fr1, fr2 = self.evaluate(torch.stack([r1, r2]))
            new_pop[a], new_pop[b] = r1, r2
            new_fit[a], new_fit[b] = fr1, fr2
            pid_counter += 1

        # Odd unpaired: Pair with tournament from pop
        if unpaired:
            a = unpaired[0]
            b = self._tournament(torch.arange(self.pop_size, device=self.population.device))
            self.partner[a] = b
            self.partner[b] = a
            self.pair_id[a] = self.pair_id[b] = pid_counter
            self.delta_hist.append([])
            self.pair_σ = torch.cat([self.pair_σ, torch.tensor([self.σ_max], device=self.pair_σ.device)])
            # Pulse on new pair
            r1, r2 = ray(new_pop[a], new_pop[b], 1.0)
            r1, r2 = glued_space(r1, self.lb, self.ub), glued_space(r2, self.lb, self.ub)
            fr1, fr2 = self.evaluate(torch.stack([r1, r2]))
            new_pop[a], new_pop[b] = r1, r2
            new_fit[a], new_fit[b] = fr1, fr2

        self.population = new_pop
        self.fitness = new_fit

        if self.debug:
            print(f"Unpaired re-paired: {len(unpaired)}, Pulsed pairs: {len(unpaired)//2}")

        return self

    def record_step(self):
        return {"pop": self.population, "fit": self.fitness, "partner": self.partner, "pair_id": self.pair_id}