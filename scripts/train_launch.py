#!/usr/bin/env python3
"""
train_launch.py — launch QLoRA training with MLX cache limits to prevent swap thrashing.

Sets memory/cache limits before invoking mlx_lm.lora, preventing the monotonic
cache growth that causes swap on high-rank runs.

Usage:
    python scripts/train_launch.py --config config/train_config_r5.yaml
    python scripts/train_launch.py --cache-fraction 0.30  # R5 default (rank=64, seq=3072)
    python scripts/train_launch.py --cache-fraction 0.50  # R4 baseline (rank=32, seq=2048)

Pre-requisite: run scripts/preflight.sh first to flush background processes.

R5 notes (rank=64, seq=3072, iters=3000):
  - cache_fraction 0.30 is correct (0.50 was R4; adding 25.9 GB reclaimable on top
    of a 33 GB live footprint risks swap spikes on the M1 Max 64 GB pool)
  - Checkpoints are ~1 GB each at rank=64; ensure ≥20 GB free disk before launch
  - Peak memory estimate: ~33-36 GB live (model 17.5 + optimizer 3.2 + activations 8 + working 4)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _check_disk(min_gb: float = 20.0) -> None:
    usage = shutil.disk_usage("/Users/dellboy")
    free_gb = usage.free / (1024**3)
    print(f"  Disk free:           {free_gb:.1f} GB", end="")
    if free_gb < min_gb:
        print(f"  ⚠️  WARNING: only {free_gb:.1f} GB free (need ≥{min_gb:.0f} GB for R5 checkpoints)")
        print(f"     rank=64 checkpoints ≈1 GB each × ~15 saves = ~15 GB")
        print(f"     Free up disk space before training or increase save_every in your config.")
        ans = input("     Continue anyway? (y/n) ").strip().lower()
        if ans != "y":
            raise SystemExit("Aborted. Free disk space and re-run.")
    else:
        print("  ✅")


def _check_swap() -> None:
    try:
        r = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True)
        line = r.stdout.strip()
        print(f"  Swap:                {line}")
        # Parse used= field
        for part in line.split(","):
            if "used" in part:
                val_str = part.split("=")[-1].strip().rstrip("M").rstrip("G")
                try:
                    val = float(val_str)
                    unit = "M" if "M" in part else "G"
                    val_mb = val if unit == "M" else val * 1024
                    if val_mb > 500:
                        print(f"  ⚠️  {val_mb:.0f} MB swap in use — consider running scripts/preflight.sh first")
                    else:
                        print("  ✅ Swap clean")
                except ValueError:
                    pass
                break
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config/train_config_r5.yaml"))
    ap.add_argument("--cache-fraction", type=float, default=0.30,
                    help="Cap MLX cache at this fraction of max working set (default: 0.30 for R5/rank=64)")
    ap.add_argument("--memory-fraction", type=float, default=0.90,
                    help="Cap MLX total memory at this fraction (default: 0.90)")
    ap.add_argument("--min-disk-gb", type=float, default=20.0,
                    help="Abort if free disk is below this GB (default: 20)")
    ap.add_argument("--skip-disk-check", action="store_true")
    args = ap.parse_args()

    try:
        import mlx.core as mx
    except ImportError:
        raise SystemExit("MLX not found. Install with: pip install mlx")

    # mx.device_info() in MLX 0.22+; fall back to mx.metal.device_info() for older builds
    try:
        info = mx.device_info()
    except AttributeError:
        info = mx.metal.device_info()
    ws = info["max_recommended_working_set_size"]
    ws_gb = ws / (1024**3)

    cache_limit = int(ws * args.cache_fraction)
    memory_limit = int(ws * args.memory_fraction)

    print("=== MLX Memory Configuration ===")
    print(f"  Device working set:  {ws_gb:.1f} GB")
    print(f"  Cache limit:         {cache_limit / (1024**3):.1f} GB ({args.cache_fraction:.0%})")
    print(f"  Memory limit:        {memory_limit / (1024**3):.1f} GB ({args.memory_fraction:.0%})")

    if not args.skip_disk_check:
        _check_disk(args.min_disk_gb)
    _check_swap()

    print()

    mx.set_cache_limit(cache_limit)
    mx.set_memory_limit(memory_limit)

    try:
        print(f"  Cache limit set:     {mx.metal.get_cache_memory() / (1024**3):.1f} GB (current cache)")
        print(f"  Active memory:       {mx.metal.get_active_memory() / (1024**3):.1f} GB")
    except AttributeError:
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
