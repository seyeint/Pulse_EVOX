#!/bin/bash
# ============================================================================
# Deploy PPO Baseline to RunPod — one command does everything
#
# Usage: bash deploy_ppo.sh <IP> <PORT>
# Example: bash deploy_ppo.sh 203.57.40.94 10282
#
# This reuses the same pod/setup from deploy_pod.sh.
# If the pod is already set up (setup_pod.sh ran before), this just syncs
# the new script and installs the extra PPO deps.
# ============================================================================

set -e

IP=$1
PORT=$2
KEY=~/.ssh/id_ed25519
LOCAL_DIR=/Users/joseferreira/Documents/shared_projects/Pulse_EVOX
REMOTE_DIR=/workspace/Pulse_EVOX

if [ -z "$IP" ] || [ -z "$PORT" ]; then
    echo "Usage: bash deploy_ppo.sh <IP> <PORT>"
    echo "Example: bash deploy_ppo.sh 203.57.40.94 10282"
    exit 1
fi

SSH_CMD="ssh -p $PORT -i $KEY root@$IP"

echo "=== STEP 1: Install rsync on pod ==="
$SSH_CMD "apt-get update -qq && apt-get install -y -qq rsync" 2>/dev/null

echo ""
echo "=== STEP 2: Rsync code to pod ==="
rsync -avz --no-owner --no-group \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude 'resources/numerical' \
    --exclude '*.pkl' \
    --exclude 'neuroevolution/gifs' \
    --exclude 'neuroevolution/model_weights' \
    --exclude '.DS_Store' \
    -e "ssh -p $PORT -i $KEY" \
    "$LOCAL_DIR/" \
    "root@$IP:$REMOTE_DIR/"

echo ""
echo "=== STEP 3: Setup pod (if needed) ==="
$SSH_CMD "python3 -c 'import brax' 2>/dev/null" && echo "  Already set up, skipping." || {
    echo "  Fresh pod — running setup_pod.sh..."
    $SSH_CMD "cd $REMOTE_DIR && bash setup_pod.sh"
}

echo ""
echo "=== STEP 4: Install extra PPO deps ==="
$SSH_CMD "pip install -q ml-collections etils 2>/dev/null"

echo ""
echo "=== STEP 5: Validate PPO imports ==="
$SSH_CMD "cd $REMOTE_DIR && python3 -c \"
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import dm_control_suite_params
import jax
print(f'  JAX backend: {jax.default_backend()} | devices: {jax.devices()}')
print(f'  Brax PPO: OK')
print(f'  MuJoCo Playground config: OK')
print(f'  Available DM Control envs: CartpoleSwingup, WalkerWalk, HumanoidWalk')
\""

echo ""
echo "=== DONE! ==="
echo ""
echo "SSH in:    ssh root@$IP -p $PORT -i $KEY"
echo ""
echo "Run PPO baseline:"
echo "  cd $REMOTE_DIR"
echo ""
echo "  # Dry run (1M steps, CartpoleSwingup only, ~2 min):"
echo "  MUJOCO_GL=osmesa python3 main_ppo_baseline.py"
echo ""
echo "  # Paper run (60M steps, 3 envs × 3 seeds, ~2-3 hours):"
echo "  sed -i 's/DRY_RUN = True/DRY_RUN = False/' main_ppo_baseline.py"
echo "  MUJOCO_GL=osmesa python3 main_ppo_baseline.py 2>&1 | tee resources/ppo_baseline/run.log"
echo ""
echo "Pull results back:"
echo "  bash pull_ppo_results.sh $IP $PORT"
