#!/usr/bin/env python3
"""
Plot Pulse algorithm dynamics from saved data.

Shows the evolution of:
- Operator probabilities (q_geo, q_ray, q_mut for PulseProgressGA; q_geo, q_ray for PulseConvexRayGA)
- Exploration pressure (ρ)
- Step size (σ)
- Progress measure (Δ)

Handles both PulseProgressGA and PulseConvexRayGA variants.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the dynamics data
with open('resources/numerical/pulse_dynamics.pkl', 'rb') as f:
    pulse_dynamics = pickle.load(f)

print(f"Loaded dynamics for {len(pulse_dynamics)} runs")

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Separate data by algorithm type
pulse_progress_data = {k: v for k, v in pulse_dynamics.items() if v.get('algo_type') == 'PulseProgressGA'}
pulse_convex_data = {k: v for k, v in pulse_dynamics.items() if v.get('algo_type') == 'PulseConvexRayGA'}

print(f"PulseProgressGA runs: {len(pulse_progress_data)}")
print(f"PulseConvexRayGA runs: {len(pulse_convex_data)}")

# Extract function numbers and seeds for organization
def organize_by_function(data_dict):
    runs_by_function = {}
    for key in data_dict.keys():
        # Extract function part: 'PulseProgressGA_f1_seed0' -> 'f1'
        parts = key.split('_')
        func_part = parts[1] if len(parts) >= 2 else parts[0]
        if func_part not in runs_by_function:
            runs_by_function[func_part] = []
        runs_by_function[func_part].append(key)
    return runs_by_function

progress_by_func = organize_by_function(pulse_progress_data)
convex_by_func = organize_by_function(pulse_convex_data)

def create_plots_for_algorithm(data_dict, by_func_dict, algo_name, has_mutation=True):
    """Create a 2x3 subplot figure for a single algorithm"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{algo_name} Dynamics Across Functions and Seeds', fontsize=16, fontweight='bold')
    
    # Plot 1: Operator Probabilities
    ax1 = axes[0, 0]
    colors = ['blue', 'red', 'green']
    labels = ['Geometric', 'Ray'] + (['Mutation'] if has_mutation else [])
    
    for i, (func, runs) in enumerate(by_func_dict.items()):
        for run_key in runs:
            data = data_dict[run_key]
            generations = data['generation']
            alpha = 0.3 if len(runs) > 1 else 0.8
            
            ax1.plot(generations, data['q_geo'], color=colors[0], alpha=alpha, linewidth=1)
            ax1.plot(generations, data['q_ray'], color=colors[1], alpha=alpha, linewidth=1)
            if has_mutation and 'q_mut' in data:
                ax1.plot(generations, data['q_mut'], color=colors[2], alpha=alpha, linewidth=1)
    
    # Add legend
    for i, (color, label) in enumerate(zip(colors[:len(labels)], labels)):
        ax1.plot([], [], color=color, label=label, linewidth=2)
    
    ax1.set_title('Operator Probabilities')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Probability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Exploration Pressure (ρ)
    ax2 = axes[0, 1]
    for func, runs in by_func_dict.items():
        for run_key in runs:
            data = data_dict[run_key]
            generations = data['generation']
            alpha = 0.6 if len(runs) > 1 else 0.8
            ax2.plot(generations, data['rho'], alpha=alpha, linewidth=1.5)
    
    ax2.set_title('Exploration Pressure (ρ)')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('ρ = 1 - q_geometric')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Step Size (σ)
    ax3 = axes[0, 2]
    for func, runs in by_func_dict.items():
        for run_key in runs:
            data = data_dict[run_key]
            generations = data['generation']
            alpha = 0.6 if len(runs) > 1 else 0.8
            ax3.plot(generations, data['sigma'], alpha=alpha, linewidth=1.5)
    
    ax3.set_title('Step Size (σ)')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('σ')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Progress Measure (Δ)
    ax4 = axes[1, 0]
    for func, runs in by_func_dict.items():
        for run_key in runs:
            data = data_dict[run_key]
            generations = data['generation']
            alpha = 0.6 if len(runs) > 1 else 0.8
            ax4.plot(generations, data['delta'], alpha=alpha, linewidth=1.5)
    
    ax4.set_title('Progress Measure (Δ)')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Δ (fitness improvement)')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Plot 5: Average ρ across functions
    ax5 = axes[1, 1]
    func_colors = plt.cm.tab10(np.linspace(0, 1, len(by_func_dict)))
    
    for i, (func, runs) in enumerate(by_func_dict.items()):
        all_rho = []
        max_gen = 0
        
        for run_key in runs:
            data = data_dict[run_key]
            all_rho.append(data['rho'])
            max_gen = max(max_gen, len(data['rho']))
        
        # Pad shorter runs with their last value
        padded_rho = []
        for rho_series in all_rho:
            padded = rho_series + [rho_series[-1]] * (max_gen - len(rho_series))
            padded_rho.append(padded)
        
        mean_rho = np.mean(padded_rho, axis=0)
        std_rho = np.std(padded_rho, axis=0)
        generations = list(range(1, len(mean_rho) + 1))
        
        ax5.plot(generations, mean_rho, color=func_colors[i], label=func, linewidth=2)
        ax5.fill_between(generations, 
                         np.maximum(0, mean_rho - std_rho), 
                         np.minimum(1, mean_rho + std_rho), 
                         color=func_colors[i], alpha=0.2)
    
    ax5.set_title('Average Exploration Pressure by Function')
    ax5.set_xlabel('Generation')
    ax5.set_ylabel('ρ (mean ± std)')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)
    
    # Plot 6: Operator transition patterns
    ax6 = axes[1, 2]
    transition_data = {'Early': [], 'Mid': [], 'Late': []}
    
    for func, runs in by_func_dict.items():
        for run_key in runs:
            data = data_dict[run_key]
            n_gens = len(data['rho'])
            
            if n_gens >= 3:
                # Early: first third
                early_rho = np.mean(data['rho'][:n_gens//3])
                # Mid: middle third  
                mid_rho = np.mean(data['rho'][n_gens//3:2*n_gens//3])
                # Late: last third
                late_rho = np.mean(data['rho'][2*n_gens//3:])
                
                transition_data['Early'].append(early_rho)
                transition_data['Mid'].append(mid_rho)
                transition_data['Late'].append(late_rho)
    
    phases = list(transition_data.keys())
    means = [np.mean(transition_data[phase]) for phase in phases]
    stds = [np.std(transition_data[phase]) for phase in phases]
    
    bars = ax6.bar(phases, means, yerr=stds, capsize=5, alpha=0.7, color=['lightblue', 'orange', 'lightgreen'])
    ax6.set_title('Exploration Pressure by Phase')
    ax6.set_ylabel('Average ρ')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = f'resources/numerical/{algo_name.lower()}_dynamics_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {algo_name} dynamics plot to {output_path}")
    
    plt.show()

# Create separate plots for each algorithm
if pulse_progress_data:
    create_plots_for_algorithm(pulse_progress_data, progress_by_func, "PulseProgressGA", has_mutation=True)

if pulse_convex_data:
    create_plots_for_algorithm(pulse_convex_data, convex_by_func, "PulseConvexRayGA", has_mutation=False)



# Print summary statistics
print("\n" + "="*60)
print("PULSE ALGORITHMS DYNAMICS SUMMARY")
print("="*60)

total_runs = len(pulse_dynamics)
print(f"Total runs analyzed: {total_runs}")

def print_algorithm_stats(data_dict, algo_name):
    if not data_dict:
        print(f"\nNo data for {algo_name}")
        return
        
    print(f"\n{algo_name.upper()} STATISTICS:")
    print("-" * 40)
    
    # Calculate overall statistics
    all_final_rho = []
    all_final_sigma = []
    all_mean_delta = []

    for run_key, data in data_dict.items():
        if data['rho']:
            all_final_rho.append(data['rho'][-1])
        if data['sigma']:
            all_final_sigma.append(data['sigma'][-1])
        if data['delta']:
            all_mean_delta.append(np.mean(data['delta']))

    if all_final_rho:
        print(f"Final exploration pressure (ρ):")
        print(f"  Mean: {np.mean(all_final_rho):.3f} ± {np.std(all_final_rho):.3f}")
        print(f"  Range: [{np.min(all_final_rho):.3f}, {np.max(all_final_rho):.3f}]")

    if all_final_sigma:
        print(f"Final step size (σ):")
        print(f"  Mean: {np.mean(all_final_sigma):.3e} ± {np.std(all_final_sigma):.3e}")
        print(f"  Range: [{np.min(all_final_sigma):.3e}, {np.max(all_final_sigma):.3e}]")

    if all_mean_delta:
        print(f"Average progress measure (Δ):")
        print(f"  Mean: {np.mean(all_mean_delta):.3e} ± {np.std(all_mean_delta):.3e}")
        print(f"  Range: [{np.min(all_mean_delta):.3e}, {np.max(all_mean_delta):.3e}]")

# Print stats for both algorithms
print_algorithm_stats(pulse_progress_data, "PulseProgressGA")
print_algorithm_stats(pulse_convex_data, "PulseConvexRayGA")

# Comparison
if pulse_progress_data and pulse_convex_data:
    print(f"\n{'='*60}")
    print("ALGORITHM COMPARISON")
    print("="*60)
    
    progress_final_rho = [data['rho'][-1] for data in pulse_progress_data.values() if data['rho']]
    convex_final_rho = [data['rho'][-1] for data in pulse_convex_data.values() if data['rho']]
    
    if progress_final_rho and convex_final_rho:
        print(f"Final ρ comparison:")
        print(f"  PulseProgressGA:   {np.mean(progress_final_rho):.3f} ± {np.std(progress_final_rho):.3f}")
        print(f"  PulseConvexRayGA:  {np.mean(convex_final_rho):.3f} ± {np.std(convex_final_rho):.3f}")
        
        # Statistical test
        from scipy import stats
        try:
            t_stat, p_value = stats.ttest_ind(progress_final_rho, convex_final_rho)
            print(f"  T-test p-value: {p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")
        except:
            print("  Could not perform statistical test") 