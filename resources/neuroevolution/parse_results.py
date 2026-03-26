#!/usr/bin/env python3
"""
Parse run.log to extract best values per algorithm/env/seed,
verify medians, and compute averages.
"""

import re
import statistics
from pathlib import Path
from collections import defaultdict

LOG_PATH = Path(__file__).parent / "run.log"
OUT_PATH = Path(__file__).parent / "results_summary.log"


def parse_log(path: str) -> dict:
    """
    Returns: {env: {algo: [best_seed1, best_seed2, best_seed3]}}
    """
    results = defaultdict(lambda: defaultdict(list))
    current_env = None
    current_algo = None

    with open(path) as f:
        for line in f:
            # Detect environment header
            env_match = re.match(r"^ENV:\s+(\S+)", line.strip())
            if env_match:
                current_env = env_match.group(1)
                continue

            # Detect algorithm + seed header
            algo_match = re.match(
                r"^\s*(OpenES|DE|PSO|LinePulse)\s*\|\s*seed\s+\d+/\d+", line
            )
            if algo_match:
                current_algo = algo_match.group(1)
                continue

            # Detect DONE line
            done_match = re.match(
                r"^\s*DONE:\s+best=([\d.]+)\s+time=([\d.]+)s", line
            )
            if done_match and current_env and current_algo:
                best = float(done_match.group(1))
                results[current_env][current_algo].append(best)
                continue

    return results


def write_summary(results: dict, out_path: str):
    lines = []
    lines.append("=" * 70)
    lines.append("NEUROEVOLUTION BENCHMARK — EXTRACTED RESULTS")
    lines.append("=" * 70)
    lines.append("")

    all_algo_ranks = defaultdict(list)  # algo -> list of ranks across envs
    all_algo_avgs = defaultdict(list)   # algo -> list of averages across envs

    for env, algos in results.items():
        lines.append(f"  {env}:")
        lines.append(f"  {'─' * 60}")

        # Collect (algo, seeds, median, avg) tuples
        entries = []
        for algo, seeds in algos.items():
            if len(seeds) == 0:
                continue
            med = statistics.median(seeds)
            avg = statistics.mean(seeds)
            entries.append((algo, seeds, med, avg))

        # Sort by median descending (best first)
        entries.sort(key=lambda x: x[2], reverse=True)

        # Print per-seed, median, average
        for rank, (algo, seeds, med, avg) in enumerate(entries, 1):
            seeds_str = ", ".join(f"{s:8.2f}" for s in seeds)
            lines.append(
                f"    {algo:<14s}  seeds=[{seeds_str}]"
            )
            lines.append(
                f"    {'':14s}  median={med:8.2f}   avg={avg:8.2f}"
            )
            all_algo_ranks[algo].append(rank)
            all_algo_avgs[algo].append(avg)

        lines.append("")

    # ── Median verification ──────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("MEDIAN VERIFICATION")
    lines.append("=" * 70)

    # Expected medians from the log's FINAL RESULTS section
    expected_medians = {
        "CartpoleSwingup": {
            "LinePulse": 697.57,
            "PSO": 633.35,
            "DE": 402.07,
            "OpenES": 157.72,
        },
        "WalkerWalk": {
            "PSO": 304.29,
            "LinePulse": 298.61,
            "DE": 227.22,
            "OpenES": 53.80,
        },
        "HumanoidWalk": {
            "LinePulse": 151.93,
            "DE": 121.07,
            "PSO": 114.09,
            "OpenES": 40.73,
        },
    }

    all_ok = True
    for env, algos in results.items():
        lines.append(f"\n  {env}:")
        for algo, seeds in algos.items():
            med = statistics.median(seeds)
            exp = expected_medians.get(env, {}).get(algo)
            if exp is not None:
                match = abs(med - exp) < 0.01
                status = "✅ OK" if match else f"❌ MISMATCH (expected {exp})"
                if not match:
                    all_ok = False
            else:
                status = "⚠️  no expected value"
            lines.append(f"    {algo:<14s}  median={med:8.2f}  {status}")

    lines.append("")
    if all_ok:
        lines.append("  ✅ All medians match the log's FINAL RESULTS section.")
    else:
        lines.append("  ❌ Some medians DO NOT match!")

    # ── Averages + ranks ─────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 70)
    lines.append("AVERAGES & RANKINGS")
    lines.append("=" * 70)

    lines.append("")
    lines.append("  Per-environment averages:")
    for env, algos in results.items():
        lines.append(f"\n    {env}:")
        entries = []
        for algo, seeds in algos.items():
            avg = statistics.mean(seeds)
            entries.append((algo, avg))
        entries.sort(key=lambda x: x[1], reverse=True)
        for algo, avg in entries:
            lines.append(f"      {algo:<14s}  avg={avg:8.2f}")

    lines.append("")
    lines.append("  Overall average rank (across envs, by median):")
    rank_entries = []
    for algo, ranks in all_algo_ranks.items():
        avg_rank = statistics.mean(ranks)
        rank_entries.append((algo, avg_rank, ranks))
    rank_entries.sort(key=lambda x: x[1])
    for algo, avg_rank, ranks in rank_entries:
        lines.append(f"    {algo:<14s}  avg_rank={avg_rank:.2f}  {ranks}")

    lines.append("")
    lines.append("  Overall average score (mean of per-env averages):")
    score_entries = []
    for algo, avgs in all_algo_avgs.items():
        overall = statistics.mean(avgs)
        score_entries.append((algo, overall, avgs))
    score_entries.sort(key=lambda x: x[1], reverse=True)
    for algo, overall, avgs in score_entries:
        per_env = ", ".join(f"{a:.2f}" for a in avgs)
        lines.append(f"    {algo:<14s}  overall_avg={overall:8.2f}  [{per_env}]")

    lines.append("")

    output = "\n".join(lines)
    print(output)

    with open(out_path, "w") as f:
        f.write(output + "\n")
    print(f"\n→ Written to {out_path}")


if __name__ == "__main__":
    results = parse_log(LOG_PATH)
    write_summary(results, OUT_PATH)
