# pulse6.py   ––  "Pulse-CR 2.0" - with actual pulsing controller

"""
Pulse-CR with a controller that ACTUALLY PULSES.

The Problem with the sigmoid controller:
    The sigmoid on relative improvement creates a STABLE EQUILIBRIUM at ρ=0.50.
    The 50/50 operator mix produces just enough improvement to hold the sigmoid
    at its inflection point — the system never commits to either phase.

The Fix: A patience-based binary controller.
    - Start in GEOMETRIC phase (ρ=0, pure exploitation)
    - Track consecutive generations with no improvement ("patience counter")
    - After `patience` generations of stagnation → flip to RAY phase (ρ=1)
    - Stay in RAY phase for exactly `pulse_length` generations
    - Then flip back to GEOMETRIC phase and reset the patience counter

This creates a TRUE breathing cycle:
    [GEO GEO GEO GEO ... stall ... RAY RAY RAY GEO GEO GEO ...]

Also uses δ_frac=1e-2 (higher bar for "meaningful improvement") since that
was the best parameter from the sweep.
"""

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


class PulseCR2(Algorithm):
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
        patience: int = 20,
        pulse_length: int = 5,
        improvement_threshold: float = 1e-2,
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

        # σ
        self.σ_min, self.σ_max = σ_min, σ_max
        self.sigma = torch.tensor(σ_max, device=dev)

        # Pulsing controller state
        self.patience = patience          # how many stall gens before pulsing
        self.pulse_length = pulse_length  # how many gens to stay in ray phase
        self.improvement_threshold = improvement_threshold  # relative improvement threshold

        self.stall_counter = 0            # gens since last meaningful improvement
        self.pulse_counter = 0            # gens remaining in current pulse
        self.in_pulse = False             # are we currently in ray phase?

        self.prev_best = torch.tensor(
            float("inf") if minimization else -float("inf"), device=dev
        )

    # tournament
    def _choose_parents(self, k: int):
        dev = self.population.device
        cand1 = torch.randint(0, self.pop_size, (k, self.tour), device=dev)
        cand2 = torch.randint(0, self.pop_size, (k, self.tour), device=dev)
        f1, f2 = self.fitness[cand1], self.fitness[cand2]
        fn = torch.argmin if self.minimize else torch.argmax
        p1 = cand1[torch.arange(k, device=dev), fn(f1, 1)]
        p2 = cand2[torch.arange(k, device=dev), fn(f2, 1)]
        return p1, p2

    def init_step(self):
        self.fitness = self.evaluate(self.population)
        self.prev_best = self.fitness.min() if self.minimize else self.fitness.max()

    def step(self):
        # ---- BINARY PULSING CONTROLLER ----
        if self.in_pulse:
            # Currently pulsing — count down
            self.pulse_counter -= 1
            if self.pulse_counter <= 0:
                self.in_pulse = False
                self.stall_counter = 0
            ρ = 1.0  # full ray
        else:
            # Geometric phase — check for stagnation
            if self.stall_counter >= self.patience:
                # STAGNATION DETECTED → START PULSE
                self.in_pulse = True
                self.pulse_counter = self.pulse_length
                self.stall_counter = 0
                ρ = 1.0  # full ray
            else:
                ρ = 0.0  # full geometric

        # operator probabilities
        q = torch.tensor([1.0 - ρ, ρ], device=self.population.device)
        # Avoid zero probabilities for multinomial
        q = torch.clamp(q, min=1e-6)
        q /= q.sum()

        # σ: high during pulse, low during geometric
        if self.in_pulse:
            self.sigma = torch.tensor(self.σ_max, device=self.population.device)
        # else: sigma adapts via 1/5 rule below

        # ---- variation ----
        pairs = self.pop_size // 2
        p1_idx, p2_idx = self._choose_parents(pairs)
        op_idx = torch.multinomial(q, pairs, replacement=True)

        children = []
        for k, i1, i2 in zip(op_idx.tolist(), p1_idx.tolist(), p2_idx.tolist()):
            c1, c2 = OPS[k](self.population[i1], self.population[i2], self.sigma)
            children.extend([c1, c2])
        offspring = glued_space(torch.stack(children), self.lb, self.ub)

        # ---- evaluation & (μ+λ) selection ----
        off_fit = self.evaluate(offspring)
        comb_pop = torch.cat([self.population, offspring])
        comb_fit = torch.cat([self.fitness, off_fit])
        best_idx = torch.topk(-comb_fit if self.minimize else comb_fit, self.pop_size).indices
        self.population, self.fitness = comb_pop[best_idx], comb_fit[best_idx]

        # ---- σ update (1/5 rule, only during geometric phase) ----
        if not self.in_pulse:
            off_fit_pairs = off_fit.view(pairs, 2)
            parent_fit = torch.stack(
                [self.fitness[p1_idx], self.fitness[p2_idx]], dim=1
            )
            best_parent = (
                parent_fit.min(dim=1)[0] if self.minimize
                else parent_fit.max(dim=1)[0]
            )
            better = (
                (off_fit_pairs < best_parent.unsqueeze(1)) if self.minimize
                else (off_fit_pairs > best_parent.unsqueeze(1))
            )

            epsilon = 1e-6
            offspring_pairs = offspring.view(pairs, 2, self.dim)
            parent_pairs = torch.stack(
                [self.population[p1_idx], self.population[p2_idx]], dim=1
            )
            is_eq = torch.zeros(pairs, 2, dtype=torch.bool, device=offspring.device)
            for i in range(pairs):
                for j in range(2):
                    is_eq[i, j] = (
                        torch.all(torch.abs(offspring_pairs[i, j] - parent_pairs[i, 0]) < epsilon)
                        | torch.all(torch.abs(offspring_pairs[i, j] - parent_pairs[i, 1]) < epsilon)
                    )

            success = ~is_eq & better
            sr = success.float().mean()
            self.sigma *= torch.exp(0.2 * (sr - 0.2))
            self.sigma = torch.clamp(self.sigma, self.σ_min, self.σ_max)

        # ---- stagnation tracking ----
        new_best = self.fitness.min() if self.minimize else self.fitness.max()
        δ = abs(float(self.prev_best) - float(new_best))
        rel_improvement = δ / (abs(float(self.prev_best)) + 1e-12)

        if rel_improvement > self.improvement_threshold:
            self.stall_counter = 0  # meaningful improvement, reset
        else:
            self.stall_counter += 1  # no meaningful progress

        self.prev_best = new_best

        if self.debug:
            phase = "PULSE" if self.in_pulse else "GEO"
            print(
                f"[{phase:5s}] stall={self.stall_counter:3d}  "
                f"σ={float(self.sigma):.3e}  "
                f"Δ_rel={rel_improvement:.2e}  "
                f"best={float(new_best):.2f}"
            )

        return self

    def record_step(self):
        return {
            "pop": self.population,
            "fit": self.fitness,
            "σ": self.sigma,
            "in_pulse": self.in_pulse,
        }
