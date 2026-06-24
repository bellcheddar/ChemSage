#!/usr/bin/env python3
"""
merge_export.py — fuse the LoRA adapter into the base model (Phase 5), MLX-LM on Apple Silicon.

Fuses the adapter into the base model and saves an MLX model directory ready for
mlx_lm.server or scripts/chat.py.

Usage:
    python scripts/merge_export.py --config config/train_config.yaml
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config/train_config.yaml"))
    ap.add_argument("--save-path", default="chem_sage_32b")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    base    = cfg["model"]
    adapter = cfg.get("adapter_path", "adapters/chem_sage_32b_lora")
    config_dir = args.config.resolve().parent.parent
    adapter = str(config_dir / adapter)
    save_path = str(config_dir / args.save_path) if not Path(args.save_path).is_absolute() else args.save_path

    cmd = [sys.executable, "-m", "mlx_lm", "fuse", "--model", base, "--adapter-path", adapter, "--save-path", save_path]
    print("Fusing:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit("mlx_lm.fuse not found. Install with: pip install mlx-lm")

    print(f"\nFused model saved to ./{args.save_path}")
    print(f"Run chat:  python scripts/chat.py --model {args.save_path}")
    print(f"Or serve:  mlx_lm.server --model {args.save_path} --port 8080")


if __name__ == "__main__":
    main()
