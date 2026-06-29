#!/usr/bin/env bash
# monitor_training.sh — live health dashboard for a long QLoRA training run.
#
# Run in a separate terminal pane while training is active.
# Usage: bash scripts/monitor_training.sh [log_file]
#
# Monitors: swap (abort trigger), memory pressure, disk, GPU power, iter throughput.
# Abort rule: if swap used > 4 GB and still rising, OR iter time doubles for 3 consecutive
#             iters → stop training (Ctrl-C in the training terminal) and resume from last checkpoint.

LOG_FILE="${1:-}"
INTERVAL=30  # seconds between checks

RED='\033[0;31m'
YLW='\033[1;33m'
GRN='\033[0;32m'
NC='\033[0m'

echo "=== ChemSage Training Monitor ==="
echo "  Checking every ${INTERVAL}s. Abort on: swap >4 GB rising, or iter time doubling."
echo ""

while true; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  $(date '+%H:%M:%S')  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # --- Swap ---
    SWAP=$(sysctl -n vm.swapusage 2>/dev/null)
    SWAP_USED_MB=$(echo "$SWAP" | grep -oE 'used = [0-9.]+[MG]' | grep -oE '[0-9.]+[MG]' | \
        awk '{ if ($0 ~ /G/) printf "%d", $0*1024; else printf "%d", $0 }')
    if [ -n "$SWAP_USED_MB" ]; then
        if [ "$SWAP_USED_MB" -gt 4096 ]; then
            echo -e "  ${RED}SWAP: $SWAP  ← ⛔  ABORT THRESHOLD REACHED${NC}"
        elif [ "$SWAP_USED_MB" -gt 512 ]; then
            echo -e "  ${YLW}SWAP: $SWAP  ← ⚠️  Watch closely${NC}"
        else
            echo -e "  ${GRN}SWAP: $SWAP  ✅${NC}"
        fi
    else
        echo "  SWAP: $SWAP"
    fi

    # --- Memory pressure ---
    MEM_PRESSURE=$(memory_pressure 2>/dev/null | grep "free percentage" | head -1)
    FREE_PCT=$(echo "$MEM_PRESSURE" | grep -oE '[0-9]+%' | head -1 | tr -d '%')
    if [ -n "$FREE_PCT" ]; then
        if [ "$FREE_PCT" -lt 10 ]; then
            echo -e "  ${RED}MEMORY: $MEM_PRESSURE  ← ⛔  CRITICAL${NC}"
        elif [ "$FREE_PCT" -lt 20 ]; then
            echo -e "  ${YLW}MEMORY: $MEM_PRESSURE  ← ⚠️  Warn${NC}"
        else
            echo -e "  ${GRN}MEMORY: $MEM_PRESSURE  ✅${NC}"
        fi
    fi

    # --- Disk ---
    DISK_FREE=$(df -h /Users/dellboy 2>/dev/null | tail -1 | awk '{print $4}')
    DISK_FREE_GB=$(df -BG /Users/dellboy 2>/dev/null | tail -1 | awk '{gsub("G",""); print $4}')
    if [ -n "$DISK_FREE_GB" ] && [ "$DISK_FREE_GB" -lt 5 ] 2>/dev/null; then
        echo -e "  ${RED}DISK:   $DISK_FREE free  ← ⛔  CRITICALLY LOW — training will fail${NC}"
    elif [ -n "$DISK_FREE_GB" ] && [ "$DISK_FREE_GB" -lt 10 ] 2>/dev/null; then
        echo -e "  ${YLW}DISK:   $DISK_FREE free  ← ⚠️  Getting low${NC}"
    else
        echo -e "  ${GRN}DISK:   $DISK_FREE free  ✅${NC}"
    fi

    # --- Checkpoint count ---
    if ls adapters/chem_sage_32b_v5_lora/*_adapters.safetensors 2>/dev/null | head -1 > /dev/null; then
        N_CKPT=$(ls adapters/chem_sage_32b_v5_lora/*_adapters.safetensors 2>/dev/null | wc -l | tr -d ' ')
        CKPT_SIZE_GB=$(du -sh adapters/chem_sage_32b_v5_lora/ 2>/dev/null | awk '{print $1}')
        echo "  CHECKPOINTS: ${N_CKPT} saved, total ${CKPT_SIZE_GB}"
    fi

    # --- Training log tail (iter time + val loss) ---
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        echo "  TRAINING LOG (last 3 iter lines):"
        grep -E "Iter [0-9]+|Val loss" "$LOG_FILE" 2>/dev/null | tail -3 | sed 's/^/    /'
    fi

    echo ""
    sleep "$INTERVAL"
done
