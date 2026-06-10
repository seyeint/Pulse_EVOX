# test_pair_selection.py — invariant test for the (2+2) pair selection step.
#
# Invariant under test: after tell()/step(), each pair's two population slots
# hold exactly the top-2 candidate GENOTYPES from {A, B, C, D} — byte for byte.
#
# The failure mode this guards against: candidates_geno holding *views* into
# self.population while the winner writes mutate population in place. When
# parent A is the runner-up to a non-A winner, the first write clobbers the
# row that candidates_geno[0] still points at, so the second write copies the
# WINNER again — winner duplicated into both slots, runner-up destroyed, with
# the runner-up's fitness label attached to the duplicate. Note the fitness
# vector looks correct afterwards ({winner_fit, runner_up_fit}), which is why
# fitness logs alone can never catch this.
#
# Run:  .venv/bin/python test_pair_selection.py

import sys
import torch

from pulse14 import PulseGreedy
from pulse14_noray import PulseGreedyNoRay


def _check_pair_selection(algo_cls, name, **kwargs):
    """Force the bug-triggering ranking (offspring wins, parent A runner-up)
    on a single pair and assert the survivors are the right genotypes."""
    torch.manual_seed(0)

    algo = algo_cls(pop_size=2, dim=6, lb=-10.0, ub=10.0, patience=1,
                    debug=False, **kwargs)

    # Crafted fitness schedule (minimization): parents [A=0.5, B=10.0],
    # offspring [C=0.1, D=100.0]  =>  top-2 = {C, A}, i.e. winner is an
    # offspring and parent A is the runner-up — the aliasing-sensitive case.
    fit_schedule = [[0.5, 10.0], [0.1, 100.0]]
    calls = {"n": 0}

    def fake_evaluate(pop):
        vals = fit_schedule[min(calls["n"], 1)]
        calls["n"] += 1
        return torch.tensor(vals[: pop.shape[0]], device=pop.device)

    algo.evaluate = fake_evaluate

    algo.init_step()

    # Pin the pair so individual 0 is A (the eventual runner-up).
    algo.pair_a[0] = 0
    algo.pair_b[0] = 1
    algo.pair_fails[0] = 0
    algo.pair_is_new[0] = True

    a_orig = algo.population[0].clone()

    algo.step()

    pop = algo.population
    duplicated = torch.equal(pop[0], pop[1])
    a_survived = any(torch.equal(pop[i], a_orig) for i in range(2))
    fit_sorted = sorted(float(f) for f in algo.fitness)

    print(f"  [{name}] slots identical (winner cloned): {duplicated}")
    print(f"  [{name}] runner-up genotype survived:     {a_survived}")
    print(f"  [{name}] fitness labels after step:       {fit_sorted}"
          f"   <- looks correct even when corrupted")

    assert not duplicated, (
        f"{name}: winner genotype duplicated into both pair slots "
        f"(in-place aliasing in the selection writes)")
    assert a_survived, (
        f"{name}: runner-up parent genotype was destroyed by the winner write")
    return True


def main():
    print("Pair-selection invariant test (forced ranking: offspring wins, "
          "parent A runner-up)\n")
    results = {}
    for name, cls, kw in [
        ("PulseGreedy/pulse14", PulseGreedy, {"σ_ray": 1.0}),
        ("PulseGreedyNoRay/pulse14_noray", PulseGreedyNoRay, {}),
    ]:
        try:
            results[name] = _check_pair_selection(cls, name, **kw)
        except AssertionError as e:
            results[name] = False
            print(f"  [{name}] FAILED: {e}")
        print()

    if all(results.values()):
        print("ALL PAIR-SELECTION TESTS PASSED")
        return 0
    print("PAIR-SELECTION TESTS FAILED:",
          [k for k, v in results.items() if not v])
    return 1


if __name__ == "__main__":
    sys.exit(main())
