import torch
import torch.nn as nn
from IPython.display import HTML
from evox import operators
from evox.problems.neuroevolution.brax import BraxProblem
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow

from jax_version_depra.pulse_real_jax import Pulse_real
from jax_version_depra.pulse_real_glued_jax import Pulse_real_glued

# Neural network policy
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.features = nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 2))

    def forward(self, x):
        x = self.features(x)
        return torch.tanh(x)

def main():
    # Make sure that the model is on the same device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Reset the random seed
    seed = 1234
    torch.manual_seed(seed)
    torch.backends.mps.manual_seed_all(seed)

    # Initialize the MLP model
    model = SimpleMLP().to(device)
    adapter = ParamsAndVector(dummy_model=model)

    # Set the population size
    POP_SIZE = 1024

    # Get the bounds
    model_params = dict(model.named_parameters())
    pop_center = adapter.to_vector(model_params)
    lower_bound = torch.full_like(pop_center, -5)
    upper_bound = torch.full_like(pop_center, 5)

    # Initialize the algorithm (choose one)
    algorithm = Pulse_real_glued(  # Change this to Pulse_real or PSO as needed
        pop_size=POP_SIZE,
        dim=len(pop_center),
        lb=lower_bound, 
        ub=upper_bound,
        mutation=operators.mutation.Gaussian(stdvar=0.0),
        p_c=1.0, p_m=0.0,
        debug=False,
        device=device
    )

    # Initialize the Brax problem
    problem = BraxProblem(
        policy=model,
        env_name="swimmer",
        max_episode_length=1000,
        num_episodes=3,
        pop_size=POP_SIZE,
        device=device,
    )

    # Set a monitor
    monitor = EvalMonitor(
        topk=3,
        device=device,
    )
    monitor.setup()

    # Initialize workflow
    workflow = StdWorkflow(opt_direction="max")
    workflow.setup(
        algorithm=algorithm,
        problem=problem,
        solution_transform=adapter,
        monitor=monitor,
        device=device,
    )

    # Run the workflow
    max_generation = 50

    for i in range(max_generation):
        if i % 10 == 0:
            print(f"Generation {i}")
        workflow.step()

    monitor = workflow.get_submodule("monitor")
    print(f"Top fitness: {monitor.get_best_fitness()}")
    best_params = adapter.to_params(monitor.get_best_solution())
    print(f"Best params: {best_params}")

    # Visualize best policy
    HTML(problem.visualize(best_params))

if __name__ == "__main__":
    main() 

