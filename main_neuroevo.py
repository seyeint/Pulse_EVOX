import torch
import torch.nn as nn
from evox import algorithms
from evox.problems.neuroevolution.mujoco_playground import MujocoProblem
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow
from pulse_real import Pulse_real
from pulse_real_glued import Pulse_real_glued
from pulse_real_glued_2 import Pulse
import time
import os

# Optional: Mujoco rendering backend (if needed later)
os.environ["MUJOCO_GL"] = "osmesa"

class SimpleMLP(nn.Module):
    def __init__(self, in_dim=94, hidden=256, action_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            #nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return torch.tanh(self.net(x))
    
def main():

    """
    Available envs: ('AcrobotSwingup', 'AcrobotSwingupSparse', 'BallInCup', 'CartpoleBalance', 'CartpoleBalanceSparse', 
    'CartpoleSwingup', 'CartpoleSwingupSparse', 'CheetahRun', 'FingerSpin', 'FingerTurnEasy', 'FingerTurnHard', 'FishSwim', 
    'HopperHop', 'HopperStand', 'HumanoidStand', 'HumanoidWalk', 'HumanoidRun', 'PendulumSwingup', 'PointMass', 'ReacherEasy', 
    'ReacherHard', 'SwimmerSwimmer6', 'WalkerRun', 'WalkerStand', 'WalkerWalk', 'BarkourJoystick', 'BerkeleyHumanoidJoystickFlatTerrain', 
    'BerkeleyHumanoidJoystickRoughTerrain', 'G1JoystickFlatTerrain', 'G1JoystickRoughTerrain', 'Go1JoystickFlatTerrain', 'Go1JoystickRoughTerrain',
    'Go1Getup', 'Go1Handstand', 'Go1Footstand', 'H1InplaceGaitTracking', 'H1JoystickGaitTracking', 'Op3Joystick', 'SpotFlatTerrainJoystick',
    'SpotGetup', 'SpotJoystickGaitTracking', 'T1JoystickFlatTerrain', 'T1JoystickRoughTerrain', 'AlohaHandOver', 'AlohaSinglePegInsertion',
    'PandaPickCube', 'PandaPickCubeOrientation', 'PandaPickCubeCartesian', 'PandaOpenCabinet', 'PandaRobotiqPushCube', 'LeapCubeReorient', 'LeapCubeRotateZAxis')

    https://github.com/google-deepmind/mujoco_playground/tree/main/mujoco_playground/_src/dm_control_suite/xmls - for camera names
    """

    # Config
    env_name = "WalkerWalk"  
    algo_name = "RidgeAwareGA"  # Options: "PSO", "DE", "Pulse_real", "Pulse_real_glued"
    pop_size = 256
    generations = 150
    seed = 777

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        device = "cpu"  # "mps" not worth for specific reasons like JAX mujoco backend not supporting mps and more
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
    b = 1
    lb = torch.full_like(pop_center, -b)
    ub = torch.full_like(pop_center, b)

    # Initialize all algorithms (but we'll pick one)
    algorithms_dict = {
        "PSO": algorithms.PSO(pop_size=pop_size, lb=lb, ub=ub, device=device),
        "DE": algorithms.DE(pop_size=pop_size, lb=lb, ub=ub, device=device),
        "Pulse_real": Pulse_real(pop_size=pop_size, dim=len(pop_center), lb=-b, ub=b, p_c=1.0, p_m=0.0, debug=False),
        "Pulse_real_glued": Pulse_real_glued(pop_size=pop_size, dim=len(pop_center), lb=-b, ub=b, p_c=1.0, p_m=0.0, debug=False),
        "RidgeAwareGA": Pulse(pop_size=pop_size, dim=len(pop_center), lb=-b, ub=b, debug=False, device=device)
    }

    # Pick the algorithm
    algorithm = algorithms_dict[algo_name]

    # Mujoco problem (replaced BraxProblem)
    problem = MujocoProblem(
        policy=model,
        env_name=env_name,
        max_episode_length=1000,
        num_episodes=3,
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
    torch.save(best_params, f"resources/model_weights/{env_name}_{algo_name}_best_params.pt")
    #problem.visualize(best_params, output_type='gif', output_path=f"resources/{env_name}_{algo_name}_best_params")
    problem.visualize(best_params, output_type='gif', output_path=f"resources/1_{env_name}_{algo_name}_best_params.gif", camera="side")
    problem.visualize(best_params, output_type='gif', output_path=f"resources/2_{env_name}_{algo_name}_best_params.gif", camera="back")

if __name__ == "__main__":
    main()