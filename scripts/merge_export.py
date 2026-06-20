#!/usr/bin/env python3
"""
merge_export.py — fuse the LoRA adapter and serve (Phase 5), MLX-LM on Apple Silicon.

Two serving routes. Route A is the native Mac path and is recommended.

ROUTE A (recommended): serve the fused MLX model directly, no GGUF.
  1. Fuse:
       mlx_lm.fuse --model <base> --adapter-path adapters/chem_sage_lora --save-path fused_model
  2. Serve an OpenAI-compatible endpoint:
       mlx_lm.server --model fused_model --port 8080
  3. Point AnythingLLM / Open WebUI at http://localhost:8080/v1 (OpenAI-compatible).
  No GGUF, no Ollama needed. This is the cleanest route for a Qwen base.

ROUTE B (only if you specifically want Ollama): fuse -> GGUF -> Ollama.
  Caveat: mlx_lm.fuse --export-gguf supports only Llama/Mistral/Mixtral (fp16). A Qwen base will
  NOT export to GGUF directly. For Qwen, fuse to fp16 safetensors then convert with llama.cpp:
       mlx_lm.fuse --model <base> --adapter-path adapters/chem_sage_lora \
                   --save-path fused_model --de-quantize
       python llama.cpp/convert_hf_to_gguf.py fused_model --outfile chem_sage.gguf
       ollama create chem_sage -f Modelfile && ollama run chem_sage
  (For a Llama/Mistral base you can instead just add --export-gguf to the fuse command.)

This script wires Route A and writes a Modelfile for Route B.

Usage:
    python scripts/merge_export.py --config config/train_config.yaml
"""

import argparse
import subprocess
import yaml
from pathlib import Path

MODELFILE_TEMPLATE = """FROM ./{gguf}
SYSTEM \"\"\"{system}\"\"\"
PARAMETER temperature 0.3
PARAMETER top_p 0.9
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config/train_config.yaml"))
    ap.add_argument("--save-path", default="fused_model")
    ap.add_argument("--system", type=Path, default=Path("config/system_prompt.txt"))
    ap.add_argument("--gguf", default="chem_sage.gguf", help="Route B only")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    base = cfg["model"]
    adapter = cfg.get("adapter_path", "adapters/chem_sage_lora")

    # Route A: fuse to an MLX model directory.
    cmd = ["mlx_lm.fuse", "--model", base, "--adapter-path", adapter, "--save-path", args.save_path]
    print("Fusing (Route A):", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit("mlx_lm.fuse not found. Install with: pip install mlx-lm")

    print(f"\nFused model in ./{args.save_path}")
    print("Serve it:  mlx_lm.server --model {} --port 8080".format(args.save_path))
    print("Then point AnythingLLM / Open WebUI at http://localhost:8080/v1")

    # Route B convenience: write a Modelfile for the Ollama/GGUF path.
    system_prompt = args.system.read_text().strip()
    Path("Modelfile").write_text(MODELFILE_TEMPLATE.format(gguf=args.gguf, system=system_prompt))
    print("\n(Modelfile written for the optional Route B / Ollama path; see the docstring.)")


if __name__ == "__main__":
    main()
