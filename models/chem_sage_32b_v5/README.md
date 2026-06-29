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

# ChemSage 32B v5 — Round 5 fused model

**ChemSage** is a QLoRA-tuned Qwen2.5-32B chemistry assistant for drug discovery, fine-tuned on
Apple Silicon with MLX-LM. It reasons about small molecules, emits verified RDKit and PyMOL tool
calls, and is grounded by a RAG layer over a curated cheminformatics corpus.

This is the **Round 5 fused model** — the current production checkpoint, fused from iter 1600
(best available checkpoint; val loss 0.055).

## Model details

| | |
|---|---|
| Base model | `mlx-community/Qwen2.5-32B-Instruct-4bit` |
| Fine-tune method | QLoRA (RSLoRA, rank **64**, scale **90.0**) |
| Adapted layers | 32 of 64 transformer layers |
| Trainable parameters | **268M (0.82% of 32.8B)** — double R4 |
| Training examples | **16,000** (from 20,000 total; 80/10/10 split) |
| Behaviour classes | **78** (19 new vs R4) |
| Training iters | 2,000 of 3,000 (early stop — val plateau at 0.055 from iter 1600) |
| Fused checkpoint | iter 1600 (best saved; true best 0.054 at iters 1650/1950 fell between saves) |
| Best val loss | **0.055** |
| Final train loss | 0.054 |
| Peak memory | 30.3 GB |
| Fused model size | ~17.2 GB |
| Training duration | ~37 h (2026-06-27 to 2026-06-29) |

## Evaluation (5-round comparative, 2026-06-29)

Running on 100 shared test examples (seed=42) via `eval/compare/eval_compare.py`.
Scores = examples where all instances correct / 100 examples (per-example pass/fail).
Full report: `eval/compare/results/` (once complete).

| Metric | R1 | R2 | R3 | R4 | **R5 (this)** |
|---|---|---|---|---|---|
| SMILES validity | 2% | 10% | 67% | — | — |
| SMARTS validity | N/A | 0% | 5% | — | — |
| Tool executability | 0% | 5% | 54% | — | — |
| Code attempted | 4% | 12% | 29% | — | — |
| Python extended | 3% | 11% | 59% | — | — |
| Code-then-quote | N/A | 0% | 12% | — | — |
| Numerical fidelity | N/A | 0% | 11% | — | — |
| Rounding precision | 35% | 25% | 36% | — | — |
| Refusal accuracy | 98% | 98% | 97% | — | — |
| QED range | 20% | 14% | 9% | — | — |
| PDB ID validity | 22% | 28% | 23% | — | — |
| PyMOL syntax | 0% | 14% | 6% | — | — |
| Degeneration-free | 93% | 96% | 91% | — | — |

*R4/R5 in progress. Table will be updated on eval completion.*

## Usage

```bash
mlx_lm.server --model models/chem_sage_32b_v5 --port 8081
.venv/bin/python scripts/chat.py --model models/chem_sage_32b_v5
```

## Round 5 additions (78 behaviour classes, 19 new vs R4)

New generators targeting R4 exec/fidelity gaps and breadth:

| Category | New classes |
|---|---|
| Exec drills | `pyexec_drill` (×6), `code_then_quote_v2` (×5), `rounding_explicit` (×5), `fidelity_multistep` (×5) |
| ADMET / SAR | `herg_liability`, `selectivity_profile`, `prodrug_bcs`, `sar_delta` |
| Cheminformatics | `mdanalysis`, `conformer_3d`, `mcs_search`, `reaction_smarts`, `recap_fragmentation` |
| Structural biology | `dssp`, `ppi_interface`, `electron_density`, `biological_assembly` |
| Drug targets | `drug_target_family` (kinases, GPCRs, NHRs) |
| External APIs | `uniprot_api` |

## Training history

| Round | Val loss | Improvement |
|---|---|---|
| Round 1 (7B) | 0.389 | — |
| Round 2 (32B) | 0.347 | 10.8% over R1 |
| Round 3 (32B) | 0.054 | 6.4x over R2 |
| Round 4 (32B) | **0.041** | **24% over R3** |
| Round 5 (32B, **this model**) | 0.055 | early stop; harder task (78 classes, 20k examples) |

R5's higher val floor (0.055 vs R4's 0.041) reflects the harder task: 78 behaviour classes vs
59, 20k examples vs 6.4k, rank=64 adapter. The model is more capable but the task is larger.

## Built by

**Marc C. Deller, D.Phil.** · [marcdeller.com](https://marcdeller.com) · marc@marcdeller.com
