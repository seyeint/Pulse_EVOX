#!/bin/bash
# ============================================================================
# Deploy to RunPod — one command does everything
#
# Usage: bash deploy_pod.sh <IP> <PORT>
# Example: bash deploy_pod.sh 203.57.40.94 10282
# ============================================================================

set -e

IP=$1
PORT=$2
KEY=~/.ssh/id_ed25519
LOCAL_DIR=/Users/joseferreira/Documents/shared_projects/Pulse_EVOX
REMOTE_DIR=/workspace/Pulse_EVOX

if [ -z "$IP" ] || [ -z "$PORT" ]; then
    echo "Usage: bash deploy_pod.sh <IP> <PORT>"
    echo "Example: bash deploy_pod.sh 203.57.40.94 10282"
    exit 1
fi

SSH_CMD="ssh -p $PORT -i $KEY root@$IP"

echo "=== STEP 1: Install rsync on pod ==="
$SSH_CMD "apt-get update -qq && apt-get install -y -qq rsync" 2>/dev/null

echo ""
echo "=== STEP 2: Rsync code to pod (excluding heavy files) ==="
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
echo "=== STEP 3: Run setup on pod ==="
$SSH_CMD "cd $REMOTE_DIR && bash setup_pod.sh"

echo ""
echo "=== DONE! ==="
echo ""
echo "SSH in:    ssh root@$IP -p $PORT -i $KEY"
echo "Dry run:   cd $REMOTE_DIR && MUJOCO_GL=osmesa python3 main_neuroevo.py"
echo "Paper run: sed -i 's/DRY_RUN = True/DRY_RUN = False/' main_neuroevo.py && MUJOCO_GL=osmesa python3 main_neuroevo.py"
echo ""
echo "Pull results back:"
echo "  bash pull_results.sh $IP $PORT"
