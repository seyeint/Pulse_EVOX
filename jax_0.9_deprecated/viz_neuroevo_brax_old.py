import torch
import torch.nn as nn
from evox.problems.neuroevolution.brax import BraxProblem
import os
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Define the model architecture matching main_neuroevo.py
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # For custom Humanoid with 244-dim observation, 17-dim action
        self.features = nn.Sequential(
            nn.Linear(244, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 17)
        )

    def forward(self, x):
        return torch.tanh(self.features(x))

def visualize_and_evaluate():
    # Configuration
    env_name = "humanoidstandup"
    algo_name = "Pulse_real_glued"
    model_path = f"{env_name}_{algo_name}_best_params.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        return
    
    # Device setup
    device = "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = SimpleMLP().to(device)
    
    # Load saved parameters
    print(f"Loading model from {model_path}")
    best_params = torch.load(model_path, map_location=device)
    
    # Apply parameters to model
    model.load_state_dict(best_params)
    print("Model loaded successfully!")
    
    # Create Brax problem for evaluation
    problem = BraxProblem(
        policy=model,
        env_name=env_name,
        max_episode_length=1000,
        num_episodes=1,
        pop_size=1,
        device=device,
    )
    
    # Generate and save visualization
    print("\nGenerating visualization...")
    html = problem.visualize(best_params)
    
    # Save HTML to file
    output_file = f"{env_name}_{algo_name}_visualization.html"
    with open(output_file, "w") as f:
        f.write(html)
    print(f"Visualization saved to {output_file}")
    print(f"Open this file in your browser to view the animation: {os.path.abspath(output_file)}")
    
    # Print basic model info
    print("\nModel Architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    visualize_and_evaluate() 