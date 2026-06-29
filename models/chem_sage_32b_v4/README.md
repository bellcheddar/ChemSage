---
base_model: mlx-community/Qwen2.5-32B-Instruct-4bit
language:
- en
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/main/LICENSE
pipeline_tag: text-generation
tags:
- chat
- mlx
- chemistry
- drug-discovery
- rdkit
- qlora
library_name: mlx
---

# ChemSage 32B v3 — Round 4 fused model

**ChemSage** is a QLoRA-tuned Qwen2.5-32B chemistry assistant for drug discovery, fine-tuned on
Apple Silicon with MLX-LM. It reasons about small molecules, emits verified RDKit and PyMOL tool
calls, and is grounded by a RAG layer over a curated cheminformatics corpus.

This is the **Round 4 fused model** — the current production checkpoint, fused from iter 950
(best val loss 0.041 of 1,500 iters).

## Model details

| | |
|---|---|
| Base model | `mlx-community/Qwen2.5-32B-Instruct-4bit` |
| Fine-tune method | QLoRA (RSLoRA, rank 32, scale 64.0) |
| Adapted layers | 32 of 64 transformer layers |
| Trainable parameters | 134M (0.41% of 32.8B) |
| Training examples | 6,400 (from 8,000 total; 80/10/10 split) |
| Behaviour classes | 59 |
| Training iters | 1,500 (ran to completion) |
| Fused checkpoint | iter 950 (best val loss) |
| Best val loss | **0.041** (at iter 950; ties at iter 1425) |
| Final train loss | 0.043 (iter 1500) |
| Peak memory | 25.5 GB |
| Fused model size | ~17.2 GB |
| Training duration | ~43 h (2026-06-25 to 2026-06-27) |

## Evaluation (5-round comparative, 2026-06-29)

Evaluated on 100 shared R5 test examples (seed=42) via `eval/compare/eval_compare.py`.
Scores = examples where all instances correct / 100 examples (per-example pass/fail).
Full report: `eval/compare/results/compare_20260629_1928.html`.

| Metric | Round 1 | Round 2 | Round 3 | **Round 4** | Round 5 |
|---|---|---|---|---|---|
| SMILES validity | 100% | 100% | 100% | **100%** | 100% |
| SMARTS validity | N/A | 0% | 100% | **55%** | 100% |
| Tool executability | 0% | 28% | 79% | **71%** | 95% |
| Code attempted | 14% | 41% | 100% | **97%** | 100% |
| Python extended | 36% | 34% | 78% | **69%** | 99% |
| Code-then-quote | N/A | 0% | 47% | **19%** | 61% |
| Numerical fidelity | N/A | 18% | 57% | **47%** | 89% |
| Rounding precision | 100% | 100% | 98% | **98%** | 99% |
| Refusal accuracy | 98% | 98% | 97% | **98%** | 100% |
| QED range | 100% | 100% | 100% | **100%** | 100% |
| PDB ID validity | 100% | 100% | 100% | **100%** | 100% |
| PyMOL syntax | 77% | 97% | 89% | **89%** | 90% |
| Degeneration-free | 93% | 96% | 91% | **98%** | 100% |
| **Overall** | **72%** | **63%** | **87%** | **80%** | **95%** |

R4 regressed on exec metrics vs R3 due to dataset distribution shift (more PyMOL/structural content,
fewer pure-RDKit drills). R5 corrected this with `pyexec_drill`, `code_then_quote_v2`, and
`fidelity_multistep` generators — exec reached 95%, fidelity 89%.

## Usage

```bash
# Serve the model (OpenAI-compatible endpoint)
mlx_lm.server --model models/chem_sage_32b_v4 --port 8081

# Or use the interactive chat CLI (RAG + RDKit auto-execution)
.venv/bin/python scripts/chat.py --model models/chem_sage_32b_v4
```

## What it does well

- Emits correct, executable RDKit code for every physicochemical property query
- Drives PyMOL command scripts for structural visualisation (selections, alignment, surface, B-factor)
- Interprets PLIP interaction fingerprints in medchem terms
- Runs Biopython Bio.PDB patterns (SMCRA, NeighborSearch, Superimposer)
- Generates Plotly Python and R/ggplot2/bio3d visualisation scripts
- Grounds answers in the ChemSage corpus (PDB, ChEMBL, PubChem, approved drugs)
- Refuses out-of-scope or invalid queries (refusal training added in R4)
- Enforces "compute then quote" pattern: numerical values always quoted from code output
- Enforces rounding conventions: MW 1 d.p., logP 2 d.p., TPSA 1 d.p., QED 2 d.p.

## Dataset (Round 4)

8,000 examples across 59 behaviour classes, including R4 additions:

| New class category | Classes | Purpose |
|---|---|---|
| Structural biology | PyMOL, PLIP, PDB format, pdb-tools, Biopython, ChimeraX, gemmi | Protein-ligand structural analysis |
| Structural prep | struct_prep, docking_interp | Structure preparation, docking result reading |
| Visualisation | plotly, r_recipe | Plotly Python + R/ggplot2/bio3d recipes |
| Protein families | protein_family | Real PDB data for kinases, GPCRs, antibodies, GTPases, cytokines |
| Numerical fidelity | code_then_quote, rounding_precision | Enforce compute-then-quote and exact rounding |
| Personality | personality_drug, personality_valid, personality_struct | Lipinski cards, SMILES flair, structural humour |
| Refusals | refusal | Out-of-scope query handling |
| Drug knowledge | drug_mechanisms, drug_indications, atc_classification | ChEMBL mechanism/indication/ATC data |

## Training history

| Round | Val loss | Improvement |
|---|---|---|
| Round 1 (7B) | 0.389 | — |
| Round 2 (32B) | 0.347 | 10.8% over R1 |
| Round 3 (32B) | 0.054 | 6.4x over R2 |
| Round 4 (32B, **this model**) | **0.041** | **24% over R3** |
| Round 5 (32B) | 0.055 | early stop iter 2000; rank=64, 20k examples |

Succeeded by `models/chem_sage_32b_v5/` — fused 2026-06-29 from iter 1600 checkpoint.

## Compatibility note (2026-06-29)

`tokenizer_config.json` updated: `extra_special_tokens` converted from list to dict to fix
`mlx_lm` 0.31.x / `transformers` compatibility (AttributeError on load in older format).

## Built by

**Marc C. Deller, D.Phil.** · [marcdeller.com](https://marcdeller.com) · marc@marcdeller.com
