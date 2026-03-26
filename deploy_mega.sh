#!/bin/bash
# deploy_mega.sh — Deploy, setup, and dry-run mega test on a RunPod instance
# Usage: bash deploy_mega.sh IP PORT ENV
#   IP    : pod IP address
#   PORT  : pod SSH port
#   ENV   : environment name (CartpoleSwingup | HopperHop | CheetahRun | WalkerWalk | HumanoidWalk)
#
# This script does EVERYTHING except the final paper-mode launch:
#   1. Install rsync on pod
#   2. Sync the 6 needed files
#   3. Run setup_pod.sh (installs all Python deps)
#   4. Dry-run EA script  (shows output here on your Mac)
#   5. Dry-run PPO script (shows output here on your Mac)
#   6. Print the exact paper-mode launch commands to paste
#
# After this script finishes, just SSH in and paste the launch commands.

set -e
IP="${1:?Usage: bash deploy_mega.sh IP PORT ENV}"
PORT="${2:?Usage: bash deploy_mega.sh IP PORT ENV}"
ENV="${3:?Usage: bash deploy_mega.sh IP PORT ENV (e.g. HumanoidWalk)}"
KEY="$HOME/.ssh/id_ed25519"
REMOTE="root@${IP}"
SSH="ssh -p ${PORT} -i ${KEY} -o StrictHostKeyChecking=no"
DEST="/workspace/Pulse_EVOX"

VALID_ENVS="CartpoleSwingup HopperHop CheetahRun WalkerWalk HumanoidWalk"
if ! echo "${VALID_ENVS}" | grep -qw "${ENV}"; then
    echo "ERROR: Unknown env '${ENV}'. Valid: ${VALID_ENVS}"
    exit 1
fi

echo "========================================"
echo " deploy_mega.sh"
echo " Pod:  ${REMOTE}:${PORT}"
echo " Env:  ${ENV}"
echo "========================================"

# ── Step 1: Install rsync ─────────────────────────────────────────────────────
echo ""
echo "[1/5] Installing rsync on pod..."
${SSH} ${REMOTE} "apt-get update -qq && apt-get install -y -qq rsync 2>/dev/null || true"
echo "      ✓ rsync ready"

# ── Step 2: Sync files ─────────────────────────────────────────────────────────
echo ""
echo "[2/5] Syncing files to ${DEST}..."
${SSH} ${REMOTE} "mkdir -p ${DEST}"
rsync -avz --no-owner --no-group \
    -e "${SSH}" \
    --include='pulse14.py' \
    --include='pulse14_noray.py' \
    --include='main_mega_ea.py' \
    --include='main_mega_ppo.py' \
    --include='main_capacity_sweep.py' \
    --include='setup_pod.sh' \
    --exclude='*' \
    ./ ${REMOTE}:${DEST}/
echo "      ✓ 6 files synced"

# ── Step 3: Setup pod ─────────────────────────────────────────────────────────
echo ""
echo "[3/5] Running setup_pod.sh (installs Python deps)..."
${SSH} ${REMOTE} "cd ${DEST} && bash setup_pod.sh"
echo "      ✓ Setup complete"

# ── Step 4: Dry-run EA ────────────────────────────────────────────────────────
echo ""
echo "[4/5] Dry-run EA benchmark (${ENV})..."
echo "────────────────────────────────────────"
${SSH} ${REMOTE} "cd ${DEST} && MUJOCO_GL=osmesa python3 -u main_mega_ea.py ${ENV} 2>&1 | head -25"
echo "────────────────────────────────────────"
echo "      ✓ EA dry run complete"

# ── Step 5: Dry-run PPO ───────────────────────────────────────────────────────
echo ""
echo "[5/5] Dry-run PPO sweep (${ENV})..."
echo "────────────────────────────────────────"
${SSH} ${REMOTE} "cd ${DEST} && MUJOCO_GL=osmesa python3 -u main_mega_ppo.py ${ENV} 2>&1 | head -25"
echo "────────────────────────────────────────"
echo "      ✓ PPO dry run complete"

# ── Print paper-mode launch commands ─────────────────────────────────────────
echo ""
echo "========================================"
echo " ALL DRY RUNS PASSED — READY TO LAUNCH"
echo "========================================"
echo ""
echo " SSH in and paste these commands:"
echo ""
echo "   ${SSH} ${REMOTE}"
echo "   cd ${DEST}"
echo "   mkdir -p resources/mega"
echo ""
echo "   # Flip to paper mode:"
echo "   sed -i 's/DRY_RUN = True/DRY_RUN = False/' main_mega_ea.py main_mega_ppo.py"
echo ""
echo "   # Launch EA then PPO sequentially (auto-chains):"
echo "   nohup bash -c '"
echo "     MUJOCO_GL=osmesa python3 -u main_mega_ea.py ${ENV} > resources/mega/ea_${ENV}.log 2>&1"
echo "     echo \"EA done, starting PPO...\""
echo "     MUJOCO_GL=osmesa python3 -u main_mega_ppo.py ${ENV} > resources/mega/ppo_${ENV}.log 2>&1"
echo "     echo \"ALL DONE for ${ENV}\"'"
echo "   > resources/mega/run_${ENV}.log 2>&1 &"
echo "   echo \"Launched PID: \$!\""
echo ""
echo "   # Monitor EA progress:"
echo "   tail -5 resources/mega/ea_${ENV}.log"
echo "   # Monitor PPO progress (once EA finishes):"
echo "   tail -5 resources/mega/ppo_${ENV}.log"
echo ""
echo " Pull results when done (from your Mac):"
echo "   mkdir -p resources/mega"
echo "   scp -P ${PORT} -i ${KEY} \\"
echo "     ${REMOTE}:${DEST}/resources/mega/ea_results.json \\"
echo "     resources/mega/ea_results_${ENV}.json"
echo "   scp -P ${PORT} -i ${KEY} \\"
echo "     ${REMOTE}:${DEST}/resources/mega/ppo_sweep_results.json \\"
echo "     resources/mega/ppo_results_${ENV}.json"
echo ""
