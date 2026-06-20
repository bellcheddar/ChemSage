#!/usr/bin/env python3
"""
eval_chem.py — chemistry-aware evaluation harness (Phase 7).

Generic LLM evals miss what matters here. These metrics are cheap, automatic, and brutal:

  - SMILES validity rate : every SMILES the model emits must parse + canonicalise in RDKit.
  - tool-call executability : every emitted code block must run without error.
  - numerical fidelity : recompute stated properties with RDKit; they must match.
  - retrieval faithfulness : answers grounded in corpus, not invented (manual or LLM-judge).
  - win-rate vs base : side-by-side on a frozen prompt set.

Queries an OpenAI-compatible endpoint, so it works against mlx_lm.server (the native Mac serving
route, default http://localhost:8080/v1) or any other OpenAI-compatible server.

Usage:
    mlx_lm.server --model fused_model --port 8080      # in another terminal
    python eval/eval_chem.py --base-url http://localhost:8080/v1 --model fused_model

Claude Code TODO:
  - Implement query_model() with a POST to {base_url}/chat/completions.
  - Run extracted code blocks through rag/tool_exec.py for the executability metric.
  - Emit a one-page HTML scorecard (your vibe-coding stack: validity %, exec %, win-rate),
    brand palette navy #1C244B / accent #467FF7.
"""

import argparse
import json
import re
from pathlib import Path

try:
    from rdkit import Chem
except ImportError:
    raise SystemExit("RDKit is required: pip install rdkit")

SMILES_IN_BLOCK = re.compile(r"MolFromSmiles\('([^']+)'\)")


def smiles_validity(model_output: str):
    """Fraction of emitted SMILES that canonicalise. Returns (valid, total)."""
    smis = SMILES_IN_BLOCK.findall(model_output)
    valid = sum(1 for s in smis if Chem.MolFromSmiles(s) is not None)
    return valid, len(smis)


def query_model(base_url: str, model: str, prompt: str) -> str:
    """TODO: POST to {base_url}/chat/completions (OpenAI-compatible) and return assistant content."""
    raise NotImplementedError("Wire up the OpenAI-compatible chat call (mlx_lm.server).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="fused_model")
    ap.add_argument("--test", type=Path, default=Path("data/sft/test.jsonl"))
    args = ap.parse_args()

    total_valid = total_smiles = 0
    for line in args.test.read_text().splitlines():
        ex = json.loads(line)
        prompt = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        out = query_model(args.base_url, args.model, prompt)   # TODO
        v, t = smiles_validity(out)
        total_valid += v
        total_smiles += t

    rate = (total_valid / total_smiles) if total_smiles else float("nan")
    print(f"SMILES validity: {total_valid}/{total_smiles} = {rate:.1%}")
    # TODO: executability, numerical fidelity, win-rate; write HTML scorecard.


if __name__ == "__main__":
    main()
