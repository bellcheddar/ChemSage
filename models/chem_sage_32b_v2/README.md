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

# ChemSage 32B — Round 2 fused model

**ChemSage** is a QLoRA-tuned Qwen2.5-32B chemistry assistant for drug discovery, fine-tuned on
Apple Silicon with MLX-LM.

This is the **Round 2 fused model** — the first 32B checkpoint; superseded by Round 3.

## Model details

| | |
|---|---|
| Base model | `mlx-community/Qwen2.5-32B-Instruct-4bit` |
| Fine-tune method | QLoRA (RSLoRA, rank 32, scale 64.0) |
| Adapted layers | 32 of 64 transformer layers |
| Trainable parameters | 134M (0.41% of 32.8B) |
| Training examples | 1,200 (from 1,500 total; 80/10/10 split) |
| Behaviour classes | 8 |
| Training iters | 750 |
| Best val loss | **0.347** (at iter 675) |
| Peak memory | 26.1 GB |
| Fused model size | ~17 GB |

## Evaluation (Round 2)

Evaluated on 150 held-out test examples via `eval/eval_chem_original.py`.

| Metric | Score |
|---|---|
| SMILES validity | 100% (25/25) |
| Tool executability | 41% (72/174 code blocks) |
| Numerical fidelity | 10% (7/73 values) |
| **Overall** | **51%** |

Superseded by Round 3 (val loss 0.054, exec 99%, fidelity 68%) and Round 4.

## Training history

| Round | Val loss | Improvement |
|---|---|---|
| Round 1 (7B) | 0.389 | — |
| Round 2 (32B, **this model**) | **0.347** | 10.8% over R1 |
| Round 3 (32B) | 0.054 | 6.4x over R2 |
| Round 4 (32B) | **0.041** | **24% over R3** |
| Round 5 (32B) | 0.055 | early stop iter 2000; rank=64, 20k examples |

## Compatibility note (2026-06-29)

`tokenizer_config.json` updated: `extra_special_tokens` converted from list to dict to fix
`mlx_lm` 0.31.x / `transformers` compatibility (AttributeError on load in older format).

## Built by

**Marc C. Deller, D.Phil.** · [marcdeller.com](https://marcdeller.com) · marc@marcdeller.com
