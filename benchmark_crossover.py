import torch
import time
from evox.core import vmap

# Print PyTorch version for reference
print(f"PyTorch version: {torch.__version__}")

# Define dimensions for testing
DIM = 20
N_PAIRS = 10000  # 10,000 pairs as requested

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

# Approach 2: Using torch.cond
def crossover_cond(pref, parents):
    """Choose between geometric and non-geometric crossover using torch.cond."""
    return torch.cond(
        pref == 1,
        lambda: non_geometric_crossover(parents),
        lambda: geometric_crossover(parents),
        None  # This is the operands parameter that was missing
    )

def batch_crossover_cond(pref, parents):
    """Apply crossover to multiple pairs using torch.cond."""
    return vmap(crossover_cond, randomness="different")(pref, parents)

# Approach 3: Using Python if-else
def crossover_if(pref, parents):
    """Choose between geometric and non-geometric crossover using if-else."""
    if pref == 1:
        return non_geometric_crossover(parents)
    else:
        return geometric_crossover(parents)

def batch_crossover_if(pref, parents):
    """Apply crossover to multiple pairs using if-else."""
    return vmap(crossover_if, randomness="different")(pref, parents)

# Benchmark function
def benchmark(func, pref, parents, name, runs=5):
    """Benchmark a function and return average execution time."""
    times = []
    
    # Warmup
    _ = func(pref, parents)
    
    # Timed runs
    for _ in range(runs):
        start = time.time()
        result = func(pref, parents)
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"{name}: {avg_time:.6f} seconds (average of {runs} runs)")
    
    return result, avg_time

# Main function
def main():
    print(f"Benchmarking crossover methods with {N_PAIRS} pairs, dimension {DIM}")
    print("-" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create test data
    pref, parents = create_test_data()
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    
    # Test torch.where
    try:
        result_where, time_where = benchmark(batch_crossover_where, pref, parents, "torch.where")
        run_where = True
    except Exception as e:
        print(f"Error with torch.where: {e}")
        run_where = False
        time_where = float('inf')
    
    # Test torch.cond
    try:
        result_cond, time_cond = benchmark(batch_crossover_cond, pref, parents, "torch.cond")
        run_cond = True
    except Exception as e:
        print(f"Error with torch.cond: {e}")
        run_cond = False
        time_cond = float('inf')
    
    # Test if-else
    try:
        result_if, time_if = benchmark(batch_crossover_if, pref, parents, "if-else")
        run_if = True
    except Exception as e:
        print(f"Error with if-else: {e}")
        run_if = False
        time_if = float('inf')
    
    # Verify results match
    print("\nVerifying results match:")
    if run_where and run_cond:
        where_cond_match = torch.allclose(result_where, result_cond)
        print(f"torch.where and torch.cond match: {where_cond_match}")
    
    if run_where and run_if:
        where_if_match = torch.allclose(result_where, result_if)
        print(f"torch.where and if-else match: {where_if_match}")
    
    if run_cond and run_if:
        cond_if_match = torch.allclose(result_cond, result_if)
        print(f"torch.cond and if-else match: {cond_if_match}")
    
    # Compare performance
    print("\nPerformance comparison:")
    times = []
    names = []
    
    if run_where:
        times.append(time_where)
        names.append("torch.where")
    
    if run_cond:
        times.append(time_cond)
        names.append("torch.cond")
    
    if run_if:
        times.append(time_if)
        names.append("if-else")
    
    if times:
        fastest_idx = times.index(min(times))
        fastest_name = names[fastest_idx]
        fastest_time = times[fastest_idx]
        
        print(f"Fastest method: {fastest_name}")
        
        for i, (name, time) in enumerate(zip(names, times)):
            if i != fastest_idx:
                print(f"{name} is {time/fastest_time:.2f}x the speed of {fastest_name}")
    
    # Print conclusion
    print("\nConclusion:")
    if run_where:
        print("- torch.where works with evox.core.vmap and randomness='different'")
    if run_cond:
        print("- torch.cond works with evox.core.vmap and randomness='different'")
    if run_if:
        print("- if-else works with evox.core.vmap and randomness='different'")
    
    if times:
        print(f"\nRecommendation: Use {fastest_name} for best performance")

if __name__ == "__main__":
    main() 