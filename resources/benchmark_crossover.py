import torch
import time
from evox.core import vmap


""" SUMMARY: AT DIM 200, TORCH.WHERE STARTS WORTH... 

    Like choosing between assembling two cars and picking one (where),
    vs. building two cars and smashing them together (mult, building final car). At small scale, assembling is fast,
    so picking adds unnecessary steps.
    At large scale, building twice is wasteful and picking the finished one is quicker.

BY THE WAY, TORCH.COND IS PROLLY SLOWER DUE TO THREAD DIVERGENCE
SO NO ONE CARES THAT I AM COMPUTING BOTH CROSSOVERS FOR EVERY PAIR"""

# Print PyTorch version for reference
print(f"PyTorch version: {torch.__version__}")


# Define dimensions for testing
DIM = 2000
N_PAIRS = 10000  # 10,000 pairs

# Create test data
def create_test_data():
    # Random preferences (0-3)
    pref = torch.randint(0, 4, (N_PAIRS,))
    
    # Random parent pairs
    parents = torch.rand((N_PAIRS, 2, DIM))
    
    return pref, parents

# Crossover implementations
def geometric_crossover(parents):
    """Perform geometric crossover in continuous space."""
    p1, p2 = parents
    alpha = torch.rand(size=(1,))
    offspring1 = alpha * p1 + (1-alpha) * p2
    offspring2 = (1-alpha) * p1 + alpha * p2
    return torch.stack([offspring1, offspring2], dim=0)

def non_geometric_crossover(parents):
    """Perform non-geometric (extension ray) crossover in continuous space."""
    p1, p2 = parents
    direction = p2 - p1
    alpha = 1  # Fixed alpha
    offspring1 = p2 + alpha * direction      # Extend beyond p2
    offspring2 = p1 - alpha * direction      # Extend beyond p1
    return torch.stack([offspring1, offspring2], dim=0)

# Approach 1: Using torch.where
def crossover_where(pref, parents):
    """Choose between geometric and non-geometric crossover using torch.where."""
    return torch.where(
        pref == 1,
        non_geometric_crossover(parents),
        geometric_crossover(parents)
    )

def batch_crossover_where(pref, parents):
    """Apply crossover to multiple pairs using torch.where."""
    return vmap(crossover_where, randomness="different")(pref, parents)

# Approach 2: Using multiplication (dev's suggestion)
def crossover_mult(pref, parents):
    """Choose between geometric and non-geometric crossover using multiplication."""
    return (pref == 1) * non_geometric_crossover(parents) + (pref != 1) * geometric_crossover(parents)

def batch_crossover_mult(pref, parents):
    """Apply crossover to multiple pairs using multiplication."""
    return vmap(crossover_mult, randomness="different")(pref, parents)

# Benchmark function
def benchmark(func, pref, parents, name, runs=5):
    """Benchmark a function and return average execution time."""
    _ = func(pref, parents)  # Warmup
    times = []
    for _ in range(runs):
        start = time.time()
        result = func(pref, parents)
        times.append(time.time() - start)
    avg_time = sum(times) / len(times)
    print(f"{name}: {avg_time:.6f} seconds (average of {runs} runs)")
    return result, avg_time

# Main function
def main():
    print(f"Benchmarking crossover methods with {N_PAIRS} pairs, dimension {DIM}")
    print("-" * 60)
    
    # Create test data
    torch.manual_seed(42)
    pref, parents = create_test_data()
    
    print("\nRunning benchmarks...")
    
    # Test torch.where with fresh seed
    torch.manual_seed(42)
    result_where, time_where = benchmark(batch_crossover_where, pref, parents, "torch.where")
    
    # Test multiplication with fresh seed
    torch.manual_seed(42)
    result_mult, time_mult = benchmark(batch_crossover_mult, pref, parents, "multiplication")
    
    # Verify results match
    print("\nVerifying results match:")
    print(f"Shapes match: {result_where.shape == result_mult.shape}")
    print(f"Shape: {result_where.shape}")
    print(f"Values match (atol=1e-6): {torch.allclose(result_where, result_mult, atol=1e-6)}")
    
    # Compare performance
    print("\nPerformance comparison:")
    if time_where < time_mult:
        print(f"torch.where is {time_mult/time_where:.2f}x faster than multiplication")
    else:
        print(f"multiplication is {time_where/time_mult:.2f}x faster than torch.where")
    
    print("\nRecommendation: Use torch.where for better scaling with dimension size")

if __name__ == "__main__":
    main() 