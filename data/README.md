# ChemSage dataset layout

MLX-LM expects a data directory containing `train.jsonl` and `valid.jsonl` (note: `valid`, not
`val`). `test.jsonl` is held out and used only by `eval/eval_chem.py`.

```
data/
├── corpus/          # RAG sources: PDFs, SAR CSVs, dossiers, notes (gitignored)
└── sft/             # supervised fine-tuning data (this is the MLX --data directory)
    ├── train.jsonl
    ├── valid.jsonl  # ~10% held out (MLX reads this during training)
    └── test.jsonl   # frozen, eval only
```

## SFT format

One JSON object per line, chat format (`messages`), which MLX-LM accepts directly. Every assistant
turn that touches a molecule does so by emitting a tool (RDKit/PyMOL) code block, never by
asserting a number from memory.

### Worked example (one line of train.jsonl, pretty-printed here for readability)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are ChemSage, a chemically literate research assistant."
    },
    {
      "role": "user",
      "content": "Is aspirin (CC(=O)Oc1ccccc1C(=O)O) drug-like under Lipinski? Show your working."
    },
    {
      "role": "assistant",
      "content": "I will compute the descriptors with RDKit rather than estimate them.\n\n```python\nfrom rdkit import Chem\nfrom rdkit.Chem import Descriptors, Lipinski\nmol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')\nprint('canonical:', Chem.MolToSmiles(mol))\nprint('MW   :', round(Descriptors.MolWt(mol), 1))\nprint('logP :', round(Descriptors.MolLogP(mol), 2))\nprint('HBD  :', Lipinski.NumHDonors(mol))\nprint('HBA  :', Lipinski.NumHAcceptors(mol))\n```\n\nReturned: MW 180.2, logP 1.31, HBD 1, HBA 3, canonical CC(=O)Oc1ccccc1C(=O)O.\n\nAll four Lipinski criteria are satisfied with wide margins (MW < 500, logP < 5, HBD <= 5, HBA <= 10), so aspirin is comfortably drug-like. The carboxylic acid drives the single donor and contributes to its modest, slightly acidic logP."
    }
  ]
}
```

## Construction rules (enforced by scripts/build_dataset.py)

1. **Ground truth first.** Generate the answer by *running RDKit*, then template the question and
   the worked solution around the real values. Never hand-author a numeric property.
2. **Validate on write.** Reject any example whose assistant code block fails to execute, or whose
   SMILES fails to canonicalise.
3. **Balance the behaviour classes** (see PROJECT_PLAN.md Phase 3): SMILES literacy, property
   reasoning, RDKit tool calls, PyMOL tool calls, structural/docking interpretation, medchem SAR.
4. **Keep SMILES inside fenced code blocks** so the model treats them as literals handed to RDKit.
5. **Split** 80/10/10 train/valid/test; the test split is frozen and never seen during training.
