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

## Dataset versions

| Version | Round | Examples | Classes | Command |
|---|---|---|---|---|
| v1 | Rounds 1-2 | 1,500 (1,200 train) | 8 | `build_dataset.py --n 1500` |
| v2 | Round 3 | 5,000 (4,000 train) | 19 | `build_dataset.py --n 5000 --seed 43` |
| v3 | Round 4 | 8,000 (6,400 train) | 67 | `build_dataset.py --n 8000 --seed 51` |

### v2 behaviour classes (Round 3)

Original 8 classes plus 11 new code-heavy generators targeting eval gaps:

| Class | Weight | Purpose |
|---|---|---|
| `smiles_literacy` | 1x | Canonicalise SMILES, identify functional groups |
| `property_reasoning` | 1x | Lipinski/Veber druglikeness assessment |
| `murcko_scaffold` | 1x | Murcko scaffold extraction |
| `tanimoto` | 1x | Tanimoto similarity (Morgan fingerprints) |
| `substructure` | 1x | SMARTS substructure search |
| `pymol_script` | 1x | PyMOL command script generation |
| `plip_interpretation` | 1x | PLIP interaction interpretation |
| `medchem_mmp` | 1x | Matched molecular pair analysis |
| `single_property` | **2x** | One property at a time with RDKit code block |
| `full_profile` | **2x** | Complete property profile via code |
| `qed_score` | 1x | QED drug-likeness score |
| `molecular_formula` | 1x | Formula, atom counts, ring analysis |
| `smiles_validation` | 1x | Valid/invalid SMILES handling |
| `property_comparison` | 1x | Side-by-side two-molecule comparison |
| `corpus_drug` | **2x** | Named drugs from approved_drugs.csv |
| `murcko_sar` | 1x | Scaffold comparison with SAR context |

v2 uses 51 seed SMILES (up from 25) plus ~1,000 approved drugs from the corpus. 243 examples were rejected during validation (95% pass rate).

### v3 additions (Round 4)

v3 adds 32 new generators (51 total) across structural biology, visualization, protein families, and numerical fidelity:

**Structural biology:** `pymol_selection` (2x), `pymol_viz` (2x), `plip_analysis` (2x), `pdb_format` (2x), `pdbtools` (2x), `biopython` (2x), `chimerax` (2x), `gemmi` (1x), `struct_prep` (1x), `docking_interp` (2x)

**Protein families:** `protein_family` (3x) — kinases, dark kinases, antibodies, GPCRs, GTPases, cytokines, receptors with real PDB IDs, ligand SMILES, and family-specific context from corpus CSVs

**Visualization:** `plotly` (2x) — Plotly Python scatter/radar; `r_recipe` (2x) — R/ggplot2/bio3d/Plotly-R

**Numerical fidelity:** `code_then_quote` (3x) — enforces "compute then state" pattern; `rounding_precision` (2x) — MW 1dp, logP 2dp, TPSA 1dp

**Personality:** `personality_drug` (3x) — Lipinski verdicts with ✅/❌ cards, emoji, medchem humor; `personality_valid` (2x) — SMILES validation with flair; `personality_struct` (3x) — structural observations with character

**RAFT (fidelity fix):** `raft_single` (3x) — real RDKit stdout in user turn, assistant quotes from it; `raft_druglike` (3x) — druglikeness with real output; `raft_distractor` (2x) — correct value vs wrong database value

**Robustness:** `refusal` (2x) — out-of-scope and invalid query handling
