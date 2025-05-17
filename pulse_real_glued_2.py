# ridge_aware_ga.py pulse 2.0 - non ugly math + theory update after deep dive on why DL can go full convex search
"""
Purely population-based, ridge-aware evolutionary optimiser that extends the
Pulse idea **without ever using explicit gradients**.

Key ingredients
---------------
* three variation operators
  · geometric crossover  (convex search)
  · non-geometric crossover  (non-convex search)
  · gaussian mutation (isotropic exploration)
* operator probabilities `q` updated by a replicator / bandit rule -> update às contribs e prefs pós fds 
* self-adapting global step-size `sigma` via the classic 1/5 success rule
* tournament + elitist (μ + λ) selection
* glued-space bounds (torus) so variables wrap instead of clip
* vectorised parent selection for GPU throughput
* cached parent fitness so success is judged against the *true* parents
  (aligns with the original Pulse semantics)
"""

import torch
from evox.core import Algorithm

# ---------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------
def glued_space(x, lb, ub):
    rng = ub - lb
    return (x - lb) % rng + lb

# ---------------------------------------------------------------------
# variation operators
# ---------------------------------------------------------------------
def geometric_crossover(p1, p2):
    """One point"""
    alpha = torch.rand(1, device=p1.device)
    return alpha * p1 + (1 - alpha) * p2, (1 - alpha) * p1 + alpha * p2

def non_geometric_crossover(p1, p2, sigma):
    """Extension ray"""
    d = p2 - p1
    return p2 + sigma * d, p1 - sigma * d

def gaussian_mutation(p1, p2, sigma):
    return p1 + sigma * torch.randn_like(p1), p2 + sigma * torch.randn_like(p2)

OPERATORS = {0: geometric_crossover, 1: non_geometric_crossover, 2: gaussian_mutation}

# ---------------------------------------------------------------------
# Pulse
# ---------------------------------------------------------------------
class Pulse(Algorithm):
    def __init__(
        self,
        pop_size: int,
        dim: int,
        lb: float,
        ub: float,
        sigma_init: float = 0.2,
        tournament_size: int = 3,
        minimization: bool = True,
        device: torch.device | None = None,
        debug: bool = False,
    ):
        super().__init__()
        dev = torch.device(device or torch.get_default_device())

        self.pop_size  = pop_size
        self.dim       = dim
        self.lb, self.ub = lb, ub
        self.tour_size = tournament_size
        self.minimize  = minimization
        self.debug     = debug

        # population & fitness
        self.population = torch.rand(pop_size, dim, device=dev) * (ub - lb) + lb
        self.fitness    = torch.empty(pop_size, device=dev)

        # operator probabilities
        self.q = torch.full((3,), 1 / 3, device=dev)
        self._sum_reward = torch.zeros(3, device=dev)
        self._cnt_op     = torch.zeros(3, device=dev)

        # global step-size
        self.sigma = torch.tensor(sigma_init, device=dev)
        # Calculate max_sigma based on problem scale
        self.max_sigma = 3 #0.1 * (ub - lb)  # 10% of range

    # ------------- vectorised tournament ----------------------------------
    def _choose_parents(self, n_pairs):
        dev = self.population.device
        cand1 = torch.randint(0, self.pop_size, (n_pairs, self.tour_size), device=dev)
        cand2 = torch.randint(0, self.pop_size, (n_pairs, self.tour_size), device=dev)

        fit1 = self.fitness[cand1]
        fit2 = self.fitness[cand2]
        fn   = torch.argmin if self.minimize else torch.argmax

        p1 = cand1[torch.arange(n_pairs, device=dev), fn(fit1, 1)]
        p2 = cand2[torch.arange(n_pairs, device=dev), fn(fit2, 1)]
        return p1, p2

    # ------------- evox hooks --------------------------------------------
    def init_step(self):
        self.fitness = self.evaluate(self.population)

    def step(self):
        # ---- guarantee a valid probability simplex BEFORE sampling ----
        eps = 1e-12
        self.q = torch.nan_to_num(self.q, nan=eps, posinf=eps, neginf=eps)  # Replace problematic values
        self.q = torch.clamp(self.q, min=eps)  # Ensure positivity
        self.q /= self.q.sum()  # Normalize to valid simplex
        
        λ2 = self.pop_size // 2  # parent pairs
        pop_old, fit_old = self.population.clone(), self.fitness.clone()

        # variation
        p1_idx, p2_idx = self._choose_parents(λ2)
        op_idx = torch.multinomial(self.q, λ2, replacement=True)

        offspring, op_used = [], []
        for k, i1, i2 in zip(op_idx.tolist(), p1_idx.tolist(), p2_idx.tolist()):
            c1, c2 = OPERATORS[k](pop_old[i1], pop_old[i2], self.sigma)
            offspring.extend([c1, c2]); op_used.extend([k, k])
        offspring = glued_space(torch.stack(offspring), self.lb, self.ub)

        # evaluation
        off_fit = self.evaluate(offspring)

        # (μ+λ) selection
        comb_pop = torch.cat([self.population, offspring])
        comb_fit = torch.cat([self.fitness,   off_fit ])
        sort_idx = torch.topk(-comb_fit if self.minimize else comb_fit, self.pop_size).indices  # mexico
        self.population, self.fitness = comb_pop[sort_idx], comb_fit[sort_idx]

        # operator credit
        off_fit_pair = off_fit.view(λ2, 2)
        op_used_pair = torch.tensor(op_used, device=self.q.device).view(λ2, 2)

        parent_avg = (fit_old[p1_idx] + fit_old[p2_idx]) * 0.5
        reward = (parent_avg.unsqueeze(1) - off_fit_pair) if self.minimize \
                 else (off_fit_pair - parent_avg.unsqueeze(1))

        for k in range(3):
            m = (op_used_pair == k)
            if m.any():
                self._sum_reward[k] += reward[m].sum()
                self._cnt_op[k]     += m.sum()

        eta = 0.2
        avg_rew = torch.where(self._cnt_op > 0, self._sum_reward / self._cnt_op,
                              torch.zeros_like(self._cnt_op))
        
        # Apply replicator update with numerical safeguards
        self.q *= torch.exp(-eta * avg_rew)
        # Apply same sanitization after update which probably needs to be revisited mexico
        self.q = torch.nan_to_num(self.q, nan=eps, posinf=eps, neginf=eps)
        self.q = torch.clamp(self.q, min=eps)
        self.q /= self.q.sum()
        
        self._sum_reward.zero_(); self._cnt_op.zero_()

        # 1/5-success rule
        success = (off_fit_pair < parent_avg.unsqueeze(1)) if self.minimize \
                  else (off_fit_pair > parent_avg.unsqueeze(1))
        sr = success.float().mean()
        self.sigma *= torch.exp(0.2 * (sr - 0.2))
        # Prevent extreme values of sigma, scale based on problem bounds
        self.sigma = torch.clamp(self.sigma, min=1e-6, max=self.max_sigma)

        if self.debug:
            print(f"q={self.q.cpu().numpy()}, sigma={float(self.sigma):.3e}, succ={sr:.2f}")

        return self

    def record_step(self):
        return {"pop": self.population, "fit": self.fitness,
                "q": self.q, "sigma": self.sigma}
