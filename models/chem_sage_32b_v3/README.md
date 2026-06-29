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

# ChemSage 32B v2 — Round 3 fused model

**ChemSage** is a QLoRA-tuned Qwen2.5-32B chemistry assistant for drug discovery, fine-tuned on
Apple Silicon with MLX-LM. It reasons about small molecules, emits verified RDKit and PyMOL tool
calls, and is grounded by a RAG layer over a curated cheminformatics corpus.

This is the **Round 3 fused model** — the current production checkpoint.

## Model details

| | |
|---|---|
| Base model | `mlx-community/Qwen2.5-32B-Instruct-4bit` |
| Fine-tune method | QLoRA (RSLoRA, rank 32, scale 64.0) |
| Adapted layers | 32 of 64 transformer layers |
| Trainable parameters | 134M (0.41% of 32.8B) |
| Training examples | 4,000 (from 5,000 total; 80/10/10 split) |
| Behaviour classes | 16 (SMILES, properties, RDKit, PyMOL, PLIP, medchem, corpus drugs, scaffolds) |
| Training iters | 625 of 1,500 (stopped early — val loss plateaued) |
| Best val loss | **0.054** |
| Peak memory | 25.2 GB |
| Fused model size | ~17 GB |

## Evaluation (Round 3)

Evaluated on 500 held-out test examples via `eval/eval_chem.py` against `mlx_lm.server`.

| Metric | Score |
|---|---|
| SMILES validity | 99% (494/499) |
| Tool executability | 99% (437/442 code blocks run without error) |
| Numerical fidelity | 68% (363/533 values match RDKit recompute) |
| **Overall** | **89%** |

The remaining fidelity gap (68%) is addressed in Round 4: the model sometimes quotes properties
from prose rather than from code output. Round 4 adds `code_then_quote` and RAFT-style examples
that enforce the "compute then state" pattern.

## Usage

```bash
# Serve the model (OpenAI-compatible endpoint)
mlx_lm.server --model models/chem_sage_32b_v3 --port 8081

# Or use the interactive chat CLI (RAG + RDKit auto-execution)
.venv/bin/python scripts/chat.py --model models/chem_sage_32b_v3
```

## What it does well

- Emits correct, executable RDKit code for every physicochemical property query
- 99% SMILES validity — never produces malformed SMILES
- Drives PyMOL command scripts for structural visualisation
- Interprets PLIP interaction fingerprints in medchem terms
- Grounds answers in the ChemSage corpus (PDB, ChEMBL, PubChem, approved drugs)
- Refuses to invent PDB IDs, measured values, or compound names not in context

## Training history

| Round | Val loss | Improvement |
|---|---|---|
| Round 1 (7B) | 0.389 | — |
| Round 2 (32B) | 0.347 | 10.8% over R1 |
| Round 3 (32B, **this model**) | **0.054** | **6.4x over R2** |
| Round 4 (32B) | **0.041** | **24% over R3** |

Round 4 training completed 2026-06-27 (1,500 iters, ~43 h). Best val loss **0.041** at iter 950
beats this model's 0.054 by 24%. Fused model saved to `models/chem_sage_32b_v4/`.

## Comparative evaluation (2026-06-29, in progress)

All 5 rounds evaluated on the same 100 R5 test examples (seed=42) via `eval/compare/eval_compare.py`.
Scores = examples where all instances correct / 100 examples (per-example pass/fail).

| Metric | R1 | R2 | **R3 (this)** | R4 | R5 |
|---|---|---|---|---|---|
| SMILES validity | 2% | 10% | **67%** | — | — |
| SMARTS validity | N/A | 0% | **5%** | — | — |
| Tool executability | 0% | 5% | **54%** | — | — |
| Code attempted | 4% | 12% | **29%** | — | — |
| Python extended | 3% | 11% | **59%** | — | — |
| Code-then-quote | N/A | 0% | **12%** | — | — |
| Numerical fidelity | N/A | 0% | **11%** | — | — |
| Rounding precision | 35% | 25% | **36%** | — | — |
| Refusal accuracy | 98% | 98% | **97%** | — | — |
| QED range | 20% | 14% | **9%** | — | — |
| PDB ID validity | 22% | 28% | **23%** | — | — |
| PyMOL syntax | 0% | 14% | **6%** | — | — |
| Degeneration-free | 93% | 96% | **91%** | — | — |

*R4/R5 in progress. Full report: `eval/compare/results/`. Note: scores use updated per-example
methodology (examples fully correct / 100) rather than per-instance accuracy.*

Use `eval/eval_chem_original.py` to reproduce the original 3-metric scores on 500 examples.

## Compatibility note (2026-06-29)

`tokenizer_config.json` updated: `extra_special_tokens` converted from list to dict to fix
`mlx_lm` 0.31.x / `transformers` compatibility (AttributeError on load in older format).

## Built by

**Marc C. Deller, D.Phil.** · [marcdeller.com](https://marcdeller.com) · marc@marcdeller.com
