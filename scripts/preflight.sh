#!/usr/bin/env bash
# preflight.sh — flush memory and kill background processes before training.
# Run with: bash scripts/preflight.sh
# Then start training with: python scripts/train_launch.py

set -e

echo "=== ChemSage Pre-Training Memory Flush ==="
echo ""

# 1. Kill orphaned Python processes (stale training runs holding memory)
echo "1. Checking for orphaned Python processes..."
ORPHANS=$(ps aux | grep -E "python.*mlx_lm|python.*train" | grep -v grep | grep -v $$ | awk '{printf "  PID %s — %.0f MB — %s\n", $2, $6/1024, $11}')
if [ -n "$ORPHANS" ]; then
    echo "$ORPHANS"
    read -p "   Kill these? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ps aux | grep -E "python.*mlx_lm|python.*train" | grep -v grep | grep -v $$ | awk '{print $2}' | xargs kill -9 2>/dev/null
        echo "   Killed."
    fi
else
    echo "   None found."
fi

# 2. Kill memory-heavy background apps
echo ""
echo "2. Killing sync/utility apps..."
for app in "Resilio Sync" "Putio Sync" "Grammarly Desktop" "Stats" "OneDrive"; do
    if pgrep -x "$app" > /dev/null 2>&1; then
        killall "$app" 2>/dev/null && echo "   Killed $app" || true
    fi
done

# 3. Stop iCloud sync daemons
echo ""
echo "3. Stopping iCloud sync..."
killall bird cloudd 2>/dev/null && echo "   Stopped bird/cloudd" || echo "   Not running"

# 4. Stop Spotlight indexing
echo ""
echo "4. Stopping Spotlight indexing..."
sudo mdutil -a -i off 2>/dev/null && echo "   Spotlight paused" || echo "   Failed (need sudo)"

# 5. Stop Time Machine
echo ""
echo "5. Stopping Time Machine..."
sudo tmutil disable 2>/dev/null && echo "   Time Machine paused" || echo "   Failed (need sudo)"

# 6. Kill mlx_lm server if running (frees model memory)
echo ""
echo "6. Checking for mlx_lm server..."
if lsof -ti :8080 > /dev/null 2>&1; then
    lsof -ti :8080 | xargs kill 2>/dev/null
    echo "   Killed server on port 8080"
else
    echo "   No server running"
fi

# 7. Flush file cache
echo ""
echo "7. Flushing file cache..."
sudo purge 2>/dev/null && echo "   Cache purged" || echo "   Failed (need sudo)"

# 8. Wait for memory to settle
echo ""
echo "8. Waiting 5 seconds for memory to settle..."
sleep 5

# 9. Report memory state
echo ""
echo "=== Memory Report ==="
echo ""
vm_stat | grep -E "Pages free|Pages active|Pages wired|Swapouts"
echo ""
sysctl -n vm.swapusage
echo ""

# Calculate approximate free memory
FREE_PAGES=$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.')
INACTIVE_PAGES=$(vm_stat | grep "Pages inactive" | awk '{print $3}' | tr -d '.')
PAGE_SIZE=16384  # Apple Silicon uses 16K pages
FREE_GB=$(echo "scale=1; ($FREE_PAGES + $INACTIVE_PAGES) * $PAGE_SIZE / 1073741824" | bc 2>/dev/null || echo "?")
echo "Approximate free + reclaimable: ${FREE_GB} GB"
echo ""

# Check swap
SWAP_USED=$(sysctl -n vm.swapusage | grep -oP 'used = \K[\d.]+')
if [ "$(echo "$SWAP_USED > 100" | bc 2>/dev/null)" = "1" ]; then
    echo "⚠️  WARNING: ${SWAP_USED}M swap in use. Consider rebooting for a clean slate."
else
    echo "✅ Swap is clean."
fi

echo ""
echo "=== Preflight complete ==="
echo "Run training with: python scripts/train_launch.py"
echo ""
echo "After training, restore services with: bash scripts/postflight.sh"
