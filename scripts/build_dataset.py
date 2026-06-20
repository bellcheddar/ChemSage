#!/usr/bin/env python3
"""
build_dataset.py — generate and validate chemistry SFT pairs (Phase 3).

The dataset is the project. The key trick: generate ground truth by RUNNING RDKit, then template
the question and worked solution around the real values. Never hand-author a numeric property.

Writes MLX-LM's expected layout: a data directory with train.jsonl, valid.jsonl (note: valid, not
val), and test.jsonl, each line a {"messages": [...]} chat record.

Usage:
    python scripts/build_dataset.py --out data/sft --n 1500

Claude Code TODO:
  - Expand generators to cover all six behaviour classes (PROJECT_PLAN.md Phase 3).
  - Pull seed SMILES from real sources (your SAR tables, ChEMBL subsets, the chatRDKit corpus).
  - Add PyMOL command-generation examples and PLIP/docking interpretation examples.
  - Reuse correct tool-call patterns from your existing pipelines (cif_to_plip.py, Boltz-2 scripts).
"""

import argparse
import json
import random
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
except ImportError:
    raise SystemExit("RDKit is required: pip install rdkit")

SYSTEM = "You are ChemSage, a chemically literate research assistant."

# A few seed molecules to start; replace with a real source.
SEED_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",    # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",      # ibuprofen
]


def druglikeness_example(smiles: str):
    """Build one validated drug-likeness SFT pair using RDKit-computed ground truth."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # reject invalid seed
    canon = Chem.MolToSmiles(mol)
    mw = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    answer = (
        "I will compute the descriptors with RDKit rather than estimate them.\n\n"
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print(Chem.MolToSmiles(mol), Descriptors.MolWt(mol), Descriptors.MolLogP(mol),\n"
        "      Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol))\n"
        "```\n\n"
        f"Returned: canonical {canon}, MW {mw}, logP {logp}, HBD {hbd}, HBA {hba}. "
        "Assessed against Lipinski (MW < 500, logP < 5, HBD <= 5, HBA <= 10)."
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"Is {smiles} drug-like under Lipinski? Show your working."},
            {"role": "assistant", "content": answer},
        ]
    }


def validate(example) -> bool:
    """Reject any example whose SMILES does not canonicalise. TODO: also exec code blocks."""
    for m in example["messages"]:
        if m["role"] == "user":
            # crude SMILES sniff; replace with proper extraction
            tok = m["content"].split()[1] if len(m["content"].split()) > 1 else ""
            if tok and Chem.MolFromSmiles(tok) is None:
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/sft"))
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    examples = []
    while len(examples) < args.n:
        ex = druglikeness_example(random.choice(SEED_SMILES))
        if ex and validate(ex):
            examples.append(ex)
        # TODO: round-robin across all six behaviour-class generators here.

    random.shuffle(examples)
    n = len(examples)
    # MLX-LM expects train.jsonl + valid.jsonl in the data dir; test.jsonl is for eval only.
    train, valid, test = examples[:int(.8*n)], examples[int(.8*n):int(.9*n)], examples[int(.9*n):]
    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        with open(args.out / f"{name}.jsonl", "w") as fh:
            for ex in split:
                fh.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(train)} train / {len(valid)} valid / {len(test)} test to {args.out}")


if __name__ == "__main__":
    main()
