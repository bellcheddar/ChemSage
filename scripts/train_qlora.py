#!/usr/bin/env python3
"""
train_qlora.py — QLoRA fine-tune on Apple Silicon with MLX-LM (Phase 4).

Thin wrapper around mlx_lm.lora. It checks the data directory has the files MLX expects
(train.jsonl + valid.jsonl), then runs the training defined in config/train_config.yaml.

QLoRA is automatic: because config/train_config.yaml points at a 4-bit MLX model, mlx_lm.lora
quantises-and-adapts (QLoRA) rather than full LoRA.

Usage:
    python scripts/train_qlora.py --config config/train_config.yaml

Watch the training and validation loss together (steps_per_eval in the config). Stop early if
validation loss climbs: that is overfitting.

Memory guide (unified memory, QLoRA 4-bit): 16GB Mac fine-tunes ~8B in about an hour,
32GB handles ~14B, 64GB handles ~32B. No VRAM wall, just wall-clock.
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path


def check_data(data_dir: Path):
    """MLX needs train.jsonl and valid.jsonl in the data directory."""
    missing = [f for f in ("train.jsonl", "valid.jsonl") if not (data_dir / f).exists()]
    if missing:
        raise SystemExit(
            f"Missing {missing} in {data_dir}. Run scripts/build_dataset.py first "
            "(it writes train.jsonl, valid.jsonl, test.jsonl)."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config/train_config.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_dir = Path(cfg.get("data", "data/sft"))
    check_data(data_dir)

    lp = cfg.get("lora_parameters", {})
    print(f"MLX-LM QLoRA | model: {cfg['model']} | r={lp.get('rank')} scale={lp.get('scale')} "
          f"| iters={cfg.get('iters')} | data={data_dir}")

    # mlx_lm.lora reads the YAML directly; --train is also set in the config but passed for clarity.
    cmd = ["mlx_lm.lora", "--config", str(args.config), "--train"]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit("mlx_lm.lora not found. Install with: pip install mlx-lm")
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

    print(f"Done. Adapter in {cfg.get('adapter_path')}. Next: scripts/merge_export.py")


if __name__ == "__main__":
    main()
