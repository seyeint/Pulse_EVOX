import torch
import torch.nn as nn
from evox import algorithms
from evox.problems.neuroevolution.brax import BraxProblem
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow
from pulse_real import Pulse_real
from pulse_real_glued import Pulse_real_glued
import time
    
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # For custom Humanoid with 244-dim observation, 17-dim action
        self.features = nn.Sequential(
            nn.Linear(105, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 8)
        )

    def forward(self, x):
        return torch.tanh(self.features(x))
    
def main():
    # Config
    env_name = "ant"  # Change as needed
    algo_name = "Pulse_real"  # Options: "PSO", "DE", "Pulse_real", "Pulse_real_glued"
    pop_size = 128
    generations = 10
    seed = 77

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        device = "cpu"#"mps" doesn't work
    print(f"Using device: {device}")

    # Seed setup
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device == "mps":
        torch.mps.manual_seed(seed)

    # Model and adapter
    model = SimpleMLP().to(device)
    adapter = ParamsAndVector(dummy_model=model)
    pop_center = adapter.to_vector(dict(model.named_parameters()))
    lb = torch.full_like(pop_center, -5)
    ub = torch.full_like(pop_center, 5)

    # Initialize all algorithms (but we'll pick one)
    algorithms_dict = {
        "PSO": algorithms.PSO(pop_size=pop_size, lb=lb, ub=ub, device=device),
        "DE": algorithms.DE(pop_size=pop_size, lb=lb, ub=ub, device=device),
        "Pulse_real": Pulse_real(pop_size=pop_size, dim=len(pop_center), lb=-5, ub=5, p_c=1.0, p_m=0.0, debug=False),
        "Pulse_real_glued": Pulse_real_glued(pop_size=pop_size, dim=len(pop_center), lb=-5, ub=5, p_c=1.0, p_m=0.0, debug=False),
    }

    # Pick the algorithm
    algorithm = algorithms_dict[algo_name]

    # Brax problem
    problem = BraxProblem(
        policy=model,
        env_name=env_name,
        max_episode_length=1000,
        num_episodes=5,
        pop_size=pop_size,
        device=device,
    )

    # Monitor and workflow
    monitor = EvalMonitor(topk=1, device=device)
    workflow = StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        solution_transform=adapter,
        monitor=monitor,
        opt_direction="max",
        device=device,
    )

    # Run
    print(f"\nRunning {algo_name} on {env_name}")
    total_start_time = time.time()

    for i in range(generations):
        iter_start_time = time.time()
        workflow.step()
        iter_end_time = time.time()
        
        # Print progress after every iteration
        print(f"Generation {i} - Best fitness: {monitor.get_best_fitness():.4f} - Time: {iter_end_time - iter_start_time:.2f}s")

    total_end_time = time.time()
    print(f"\nTotal time taken: {total_end_time - total_start_time:.2f} seconds")
    print(f"Final best fitness: {monitor.get_best_fitness()}")
    
    # Save best params
    best_params = adapter.to_params(monitor.get_best_solution())
    torch.save(best_params, f"{env_name}_{algo_name}_best_params.pt")

    # Optional visualization (uncomment in a notebook)
    #from IPython.display import HTML
    #HTML(problem.visualize(best_params))

if __name__ == "__main__":
    main()