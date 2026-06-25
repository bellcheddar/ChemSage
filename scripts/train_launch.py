#!/usr/bin/env python3
"""
train_launch.py — launch QLoRA training with MLX cache limits to prevent swap thrashing.

Sets memory/cache limits before invoking mlx_lm.lora, preventing the monotonic
cache growth that causes swap on 64-layer runs.

Usage:
    python scripts/train_launch.py
    python scripts/train_launch.py --config config/train_config.yaml
    python scripts/train_launch.py --cache-fraction 0.50  # cap cache at 50% of working set

Pre-requisite: run scripts/preflight.sh first to flush background processes.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config/train_config.yaml"))
    ap.add_argument("--cache-fraction", type=float, default=0.50,
                    help="Cap MLX cache at this fraction of max working set (default: 0.50)")
    ap.add_argument("--memory-fraction", type=float, default=0.90,
                    help="Cap MLX total memory at this fraction (default: 0.90)")
    args = ap.parse_args()

    try:
        import mlx.core as mx
    except ImportError:
        raise SystemExit("MLX not found. Install with: pip install mlx")

    info = mx.metal.device_info()
    ws = info["max_recommended_working_set_size"]
    ws_gb = ws / (1024**3)

    cache_limit = int(ws * args.cache_fraction)
    memory_limit = int(ws * args.memory_fraction)

    print(f"=== MLX Memory Configuration ===")
    print(f"  Device working set:  {ws_gb:.1f} GB")
    print(f"  Cache limit:         {cache_limit / (1024**3):.1f} GB ({args.cache_fraction:.0%})")
    print(f"  Memory limit:        {memory_limit / (1024**3):.1f} GB ({args.memory_fraction:.0%})")
    print()

    mx.set_cache_limit(cache_limit)
    mx.set_memory_limit(memory_limit)

    print(f"  Cache limit set:     {mx.metal.get_cache_memory() / (1024**3):.1f} GB (current cache)")
    print(f"  Active memory:       {mx.metal.get_active_memory() / (1024**3):.1f} GB")
    print()

    # Check swap before starting
    try:
        import subprocess as sp
        swap = sp.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True)
        print(f"  Swap: {swap.stdout.strip()}")
    except Exception:
        pass

    print()
    print(f"=== Launching training ===")
    print(f"  Config: {args.config}")
    print()

    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(args.config), "--train"]
    try:
        proc = subprocess.run(cmd)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
