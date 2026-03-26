#!/bin/bash
# ============================================================================
# Pull PPO baseline results from RunPod back to local Mac
#
# Usage: bash pull_ppo_results.sh <IP> <PORT>
# ============================================================================

IP=$1
PORT=$2
KEY=~/.ssh/id_ed25519
LOCAL_DIR=/Users/joseferreira/Documents/shared_projects/Pulse_EVOX
REMOTE_DIR=/workspace/Pulse_EVOX

if [ -z "$IP" ] || [ -z "$PORT" ]; then
    echo "Usage: bash pull_ppo_results.sh <IP> <PORT>"
    exit 1
fi

echo "Pulling PPO baseline results from pod..."
mkdir -p "$LOCAL_DIR/resources/ppo_baseline"

rsync -avz --no-owner --no-group \
    -e "ssh -p $PORT -i $KEY" \
    "root@$IP:$REMOTE_DIR/resources/ppo_baseline/" \
    "$LOCAL_DIR/resources/ppo_baseline/"

echo ""
echo "Done! Results in $LOCAL_DIR/resources/ppo_baseline/"
echo "  - ppo_results.json   (raw data w/ convergence curves)"
echo "  - run.log            (full console output)"
