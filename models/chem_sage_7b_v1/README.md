---
base_model: mlx-community/Qwen2.5-7B-Instruct-4bit
language:
- en
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE
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

# ChemSage 7B — Round 1 fused model

**ChemSage** is a QLoRA-tuned chemistry assistant for drug discovery, fine-tuned on Apple Silicon
with MLX-LM.

This is the **Round 1 fused model** — the initial proof-of-concept on a 7B base; superseded by
the 32B rounds.

## Model details

| | |
|---|---|
| Base model | `mlx-community/Qwen2.5-7B-Instruct-4bit` |
| Fine-tune method | QLoRA (RSLoRA, rank 32, scale 64.0) |
| Adapted layers | 16 of 28 transformer layers |
| Trainable parameters | ~26M (0.34% of 7.6B) |
| Training examples | 1,200 (from 1,500 total; 80/10/10 split) |
| Behaviour classes | 8 |
| Training iters | 750 |
| Best val loss | **0.389** (at iter 675) |
| Peak memory | ~12 GB |
| Fused model size | ~4.2 GB |

## Evaluation (5-round comparative, 2026-06-29)

Evaluated on 100 shared R5 test examples (seed=42) via `eval/compare/eval_compare.py`.
Scores = examples where all instances correct / 100 examples (per-example pass/fail).
Full report: `eval/compare/results/compare_20260629_1928.html`.

| Metric | R1 (this) | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| SMILES validity | 100% | 100% | 100% | 100% | 100% |
| SMARTS validity | N/A | 0% | 100% | 55% | 100% |
| Tool executability | 0% | 28% | 79% | 71% | 95% |
| Code attempted | 14% | 41% | 100% | 97% | 100% |
| Python extended | 36% | 34% | 78% | 69% | 99% |
| Code-then-quote | N/A | 0% | 47% | 19% | 61% |
| Numerical fidelity | N/A | 18% | 57% | 47% | 89% |
| Rounding precision | 100% | 100% | 98% | 98% | 99% |
| Refusal accuracy | 98% | 98% | 97% | 98% | 100% |
| QED range | 100% | 100% | 100% | 100% | 100% |
| PDB ID validity | 100% | 100% | 100% | 100% | 100% |
| PyMOL syntax | 77% | 97% | 89% | 89% | 90% |
| Degeneration-free | 93% | 96% | 91% | 98% | 100% |
| **Overall** | **72%** | **63%** | **87%** | **80%** | **95%** |

## Usage

```bash
# Download from HuggingFace
huggingface-cli download Dellboy/chem_sage_7b_v1 --local-dir models/chem_sage_7b_v1

mlx_lm.server --model models/chem_sage_7b_v1 --port 8081
.venv/bin/python scripts/chat.py --model models/chem_sage_7b_v1
```

## Training history

| Round | Val loss | Improvement |
|---|---|---|
| Round 1 (7B, **this model**) | **0.389** | — |
| Round 2 (32B) | 0.347 | 10.8% over R1 |
| Round 3 (32B) | 0.054 | 6.4x over R2 |
| Round 4 (32B) | 0.041 | 24% over R3 |
| Round 5 (32B) | 0.055 | early stop; rank=64, 20k examples |

## Built by

**Marc C. Deller, D.Phil.** · [marcdeller.com](https://marcdeller.com) · marc@marcdeller.com
