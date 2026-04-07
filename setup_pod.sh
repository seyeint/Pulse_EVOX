#!/bin/bash
# ============================================================================
# RunPod Setup Script for Pulse Neuroevolution
# 
# Template: runpod/pytorch:1.0.2-cu128-torch280-ubuntu2404 (5090)
#      or:  runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 (4090)
# Container Disk: 40GB+
# GPU: RTX 5090 (32GB) or A100 40GB+
#
# Usage:
#   1. rsync code to pod (see bottom of this file)
#   2. ssh into pod
#   3. cd /workspace/Pulse_EVOX && bash setup_pod.sh
#   4. python3 main_neuroevo.py  (dry run first, then flip DRY_RUN=False)
# ============================================================================

set -e
echo "=== Pulse EVOX: RunPod Setup ==="

cd /workspace/Pulse_EVOX || { echo "ERROR: /workspace/Pulse_EVOX not found. rsync first!"; exit 1; }

# 1. System deps (rsync for syncing, osmesa for headless rendering)
echo "[1/5] System dependencies..."
apt-get update -qq && apt-get install -y -qq rsync libosmesa6-dev 2>/dev/null

# 2. JAX with CUDA
echo "[2/5] JAX with CUDA 12..."
pip install -U "jax[cuda12]" --break-system-packages

# 3. MuJoCo Playground (GitHub, force past blinker conflict)
echo "[3/5] MuJoCo Playground..."
pip install git+https://github.com/google-deepmind/mujoco_playground.git --ignore-installed blinker --break-system-packages

# 4. EvoX + utilities
echo "[4/5] EvoX + utilities..."
pip install evox imageio tqdm numpy ml-collections etils --break-system-packages

# 4b. Fix CUDA compatibility — pip install evox may upgrade torch to a cu130
# build that's incompatible with the pod's CUDA driver.
# Reinstall torch from the matching CUDA index + fix JAX cuDNN to match.
# cu124 for RTX 4090 pods, cu128 for RTX 5090 pods.
CUDA_INDEX="cu124"
echo "[4b/6] Fixing CUDA ${CUDA_INDEX} compatibility..."
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/${CUDA_INDEX} --break-system-packages
pip install -U "jax[cuda12]" --break-system-packages

# 5. Patch evox for latest mujoco_playground API (num_vision_envs removed)
echo "[5/6] Patching evox compatibility..."
EVOX_MJX=$(python3 -c "import evox; import os; print(os.path.join(os.path.dirname(evox.__file__), 'problems/neuroevolution/mujoco_playground.py'))")
if grep -q "num_vision_envs" "$EVOX_MJX" 2>/dev/null; then
    sed -i 's/num_vision_envs=pop_size,//' "$EVOX_MJX"
    echo "  Patched: removed num_vision_envs from MujocoProblem"
else
    echo "  Already patched or not needed"
fi

# Create output dirs
mkdir -p resources/neuroevolution/model_weights
mkdir -p resources/neuroevolution/gifs

# Set rendering backend
export MUJOCO_GL=osmesa

# Validation
echo ""
echo "=== Validation ==="
python3 -c "
import torch
print(f'  PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB)')
import jax
print(f'  JAX backend: {jax.default_backend()} | devices: {jax.devices()}')
from evox import algorithms
print(f'  EvoX: OK | OpenES: {hasattr(algorithms, \"OpenES\")}')
from pulse14 import PulseGreedy
print(f'  LinePulse: OK')
from evox.problems.neuroevolution.mujoco_playground import MujocoProblem
print(f'  MujocoProblem: OK')
"

echo ""
echo "=== READY ==="
echo "1. Dry run EA:  MUJOCO_GL=osmesa python3 -u main_mega_ea.py CartpoleSwingup 2>&1 | head -40"
echo "2. Dry run PPO: MUJOCO_GL=osmesa python3 -u main_mega_ppo.py CartpoleSwingup 2>&1 | head -40"
echo ""
echo "3. Flip to paper mode:"
echo "   sed -i 's/DRY_RUN = True/DRY_RUN = False/' main_mega_ea.py main_mega_ppo.py"
echo ""
echo "4. Launch (replace ENV with your environment):"
echo "   mkdir -p resources/mega"
echo "   nohup bash -c 'MUJOCO_GL=osmesa python3 -u main_mega_ea.py ENV > resources/mega/ea_ENV.log 2>&1 && MUJOCO_GL=osmesa python3 -u main_mega_ppo.py ENV > resources/mega/ppo_ENV.log 2>&1' > resources/mega/run_ENV.log 2>&1 &"
