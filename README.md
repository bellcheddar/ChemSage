# 🧪 ChemSage

> **A hybrid, chemically aware LLM for drug discovery: RAG for facts, a QLoRA-tuned model for behaviour, and live RDKit/PyMOL tool calls for chemical truth. Fine-tuned and served locally on Apple Silicon with MLX-LM.**

ChemSage is a locally hosted, open-source assistant that reasons about small molecules (SMILES,
scaffolds, properties), drives the cheminformatics and structural tools a medicinal chemist
actually uses (RDKit, PyMOL, docking, PLIP), and grounds every factual claim in your own corpus.

**Why it matters:** off-the-shelf models invent chemistry (wrong logP, malformed SMILES) and
cannot see your private SAR data. ChemSage fixes both by separating concerns: volatile knowledge
lives in retrieval, stable behaviour lives in the weights, and deterministic chemical facts are
computed live by trusted tools rather than guessed. It is useful for: interrogating your own
assay and SAR data, triaging compounds, generating correct PyMOL/RDKit scripts, and interpreting
docking and interaction output, all on local hardware.

## The hybrid in one line

RAG keeps it truthful today; the fine-tune makes it consistent tomorrow; RDKit makes it correct always.

## Quick start

See **[`PROJECT_PLAN.md`](PROJECT_PLAN.md)** for the full step-by-step build (phases 0 to 8). Short version:

| Step | What | Script / command |
|---|---|---|
| 0 | Env: MLX-LM + RDKit + GUI | (`requirements.txt`) |
| 1 | Base model (Qwen 2.5 32B, MLX 4-bit) | `config/train_config.yaml` |
| 2 | **RAG first** (ship this alone) | `scripts/ingest_rag.py` |
| 2a | Download ChEMBL corpus | `scripts/download_chembl.py` + `scripts/download_chembl_extended.py` |
| 2b | Download PubChem corpus | `scripts/download_pubchem.py` |
| 2c | Download PDB corpus | `scripts/download_pdb.py` + `scripts/download_pdb_supplement.py` |
| 3 | Build + validate SFT dataset | `scripts/build_dataset.py` |
| 4 | QLoRA fine-tune | `mlx_lm.lora --config config/train_config.yaml --train` |
| 5 | Fuse + serve | `mlx_lm.fuse` → `mlx_lm.server --port 8081` |
| 6 | Close the hybrid loop | `rag/tool_exec.py` |
| 7 | Chemistry-aware evaluation | `eval/eval_chem.py` |

## Training

### Training parameters

| Parameter | 7B (v1) | 32B (v2) |
|---|---|---|
| Base model | `mlx-community/Qwen2.5-7B-Instruct-4bit` | `mlx-community/Qwen2.5-32B-Instruct-4bit` |
| Fine-tune type | LoRA (QLoRA via 4-bit base) | LoRA (QLoRA via 4-bit base) |
| `use_rslora` | `true` | `true` |
| `use_dora` | `false` | `false` |
| LoRA rank | 32 | 32 |
| LoRA scale | 64.0 | 64.0 |
| Effective RSLoRA scale | 64/sqrt(32) = 11.3x | 64/sqrt(32) = 11.3x |
| LoRA dropout | 0.05 | 0.05 |
| LoRA keys | q/k/v/o_proj + gate/up/down_proj | q/k/v/o_proj + gate/up/down_proj |
| `num_layers` | 16 (of 28 total) | 32 (of 64 total) |
| `iters` | 750 | 750 |
| `batch_size` | 4 | 4 |
| `learning_rate` | 2e-5 | 2e-5 |
| LR schedule | Cosine decay, 50-step warmup | Cosine decay, 50-step warmup |
| `weight_decay` | 0.01 | 0.01 |
| `max_seq_length` | 2048 | 2048 |
| `grad_checkpoint` | `true` | `true` |
| `seed` | 42 | 42 |
| `steps_per_eval` | 25 | 25 |
| `save_every` | 25 | 25 |
| `val_batches` | 25 | 25 |
| Trainable params | ~26M (0.34%) | 134M (0.41%) |
| Peak memory | ~12 GB | 24.9 GB |
| Training data | 2,400 examples | 2,400 examples |
| Fused model size | 4.0 GB | 17 GB |
| Fused model path | `models/chem_sage_fused/` | `chem_sage_32b/` |

### Training loss curves

Validation loss measured every 25 steps over 25 batches. Only val-loss checkpoints shown for brevity; train loss sampled at matching iters where available.

| Iter | 7B Val | 7B Train | 32B Val | 32B Train | Notes |
|------|--------|----------|---------|-----------|-------|
| 1 | 3.800 | — | 3.191 | — | Baseline (untrained) |
| 25 | 1.020 | — | 0.580 | — | First checkpoint |
| 50 | 0.740 | — | 0.533 | 0.553 | |
| 75 | 0.630 | — | 0.527 | — | |
| 100 | 0.580 | — | 0.495 | 0.531 | |
| 150 | 0.530 | — | 0.496 | 0.464 | Slight wobble |
| 200 | 0.490 | — | 0.449 | 0.465 | |
| 250 | 0.460 | — | 0.436 | 0.436 | |
| 300 | 0.440 | — | 0.416 | 0.424 | |
| 350 | 0.430 | — | 0.407 | 0.443 | |
| 400 | 0.420 | — | 0.400 | 0.411 | |
| 450 | 0.410 | — | 0.393 | — | |
| 500 | 0.405 | — | 0.375 | 0.349 | |
| 550 | 0.400 | — | 0.366 | 0.366 | |
| 600 | 0.395 | — | 0.359 | 0.370 | |
| 625 | — | — | 0.355 | — | |
| 650 | 0.392 | — | 0.380 | 0.266 | Slight wobble |
| 675 | **0.389** | — | **0.347** | — | **Best checkpoint (both models)** |
| 700 | 0.390 | — | 0.386 | 0.287 | Wobble (near-zero LR, batch variance) |
| 725 | — | — | 0.351 | — | Recovered |
| 750 | 0.390 | — | 0.372 | 0.278 | Final |

Both models achieved their best val loss at **iter 675**. The 32B converged faster (0.580 at iter 25 vs 1.020 for 7B) and reached a lower floor (0.347 vs 0.389).

### Evaluation results

Evaluated on 300 held-out test examples via `eval/eval_chem.py` against `mlx_lm.server`. System prompt injected to instruct RDKit code generation.

| Metric | 7B | 32B | Delta |
|--------|-----|------|-------|
| SMILES validity | 0/0 (n/a) | 25/25 (100%) | 32B generates valid SMILES; 7B never attempts |
| Tool executability | 0/8 (0%) | 72/174 (41%) | 32B generates 22x more code blocks, 41% run |
| Numerical fidelity | 0/0 (n/a) | 7/73 (10%) | 32B attempts property computation; 7B does not |
| Code blocks generated | 8 | 174 | 32B is 22x more likely to emit code |
| **Overall** | **0%** | **51%** | Weighted: (100 + 41 + 10) / 3 |

Scorecard files: `eval/scorecard_7b.html`, `eval/scorecard_32b.html`

The 32B model represents a step change: it engages with chemistry (generating SMILES, writing RDKit code, computing properties) where the 7B produces only prose. The main gap is tool executability (59% of code blocks fail) and numerical fidelity (model computes values in prose rather than emitting RDKit code).

## RAG corpus

The `data/corpus/` directory contains the full retrieval knowledge base: tool reference docs plus two databases of experimental chemistry data:

### ChEMBL (`data/corpus/chembl/`, ~70 MB)

| File | Records | Contents |
|---|---|---|
| `approved_drugs.csv` | 3,257 | All FDA/EMA approved small molecules with SMILES + properties |
| `bioactivity_key_targets.csv` | 112,357 | IC50/Ki/Kd for EGFR, CDK2, HIV protease, SARS-CoV-2 3CLpro, VEGFR2, Bcr-Abl, COX-2, BRAF, ALK, JAK1/2, hERG |
| `bioactivity_extended_targets.csv` | ~200k | IC50/Ki for 36 additional targets: BTK, PARP1, CDK4/6, FLT3, mTOR, AKT1, PI3Ka, RET, MET, SRC, HDAC1, BRD4, EZH2, AR, ERa, PPARg, GLP-1R, Factor Xa, Thrombin, ACE, HMG-CoA, PDE5, DHFR, IDH1/2, KRAS + more |
| `admet_data.csv` | ~50k | Caco-2 permeability, CLint, t1/2, fu,plasma, solubility, logD, CYP3A4/2D6/2C9/1A2/2C19 inhibition |
| `clinical_candidates_phase2_3.csv` | ~10k | Phase 2-3 small molecules with properties |
| `fragment_actives.csv` | 10,000 | MW < 300 fragments with pIC50 >= 4 |
| `natural_products.csv` | ~5k | Natural product-derived compounds |
| `drug_mechanisms.csv` | 7,561 | Mechanism of action + target for all drugs |
| `drug_indications.csv` | 60,055 | Disease indications for all drugs |
| `atc_classification.csv` | 5,579 | Full ATC therapeutic class hierarchy |

### PubChem (`data/corpus/pubchem/`, ~6.6 MB)

| File | Records | Contents |
|---|---|---|
| `fda_drugs_with_pubchem_cids.csv` | 3,257 | ChEMBL approved drugs cross-referenced to PubChem CIDs |
| `tox21_panel.csv` | ~90k | 9 Tox21 nuclear receptor / stress-pathway assays (AhR, AR, ERa, NFkB, Nrf2, MMP, p53, ATAD5, TR) |
| `pkis2_kinase_panel.csv` | ~50k | Published Kinase Inhibitor Set 2: 367-kinase selectivity panel |
| `drug_repurposing_hub.csv` | ~6k | Broad Institute Drug Repurposing Hub clinical annotations |
| `adme_cyp_hts.csv` | 19,803 | NCATS Caco-2 permeability, P-gp efflux, CYP2C9 inhibition HTS |
| `bindingdb_affinities.csv` | — | BindingDB Ki/IC50/Kd (literature-curated) |
| `nci60_gi50.csv` | — | NCI-60 cancer cell line growth inhibition |

### PDB (`data/corpus/pdb/`, ~111 MB)

| File | Records | Contents |
|---|---|---|
| `pdb_all_entries.csv` | 255,239 | All PDB structures: ID, method, resolution, compound, organism, deposit date |
| `pdb_ligands_all.csv` | 16,171 | All unique PDB ligands with SMILES, InChI, InChIKey, formula, MW, type, ChEMBL/DrugBank/PubChem cross-references |
| `pdb_ligands_druglike.csv` | 382 | Drug-like CCD subset (MW 100-800 Da, non-polymer, with SMILES) |
| `pdb_ligand_structure_pairs.csv` | 121,541 | Protein-ligand pairs: which ligands are bound in which structures (with resolution + method) |
| `pdb_druglike_structure_pairs.csv` | 1,671 | Drug-like protein-ligand pairs only |
| `pdb_binding_affinities.csv` | 46,885 | Measured Kd/Ki/IC50 from PDBbind-annotated structures (9,209 entries) |
| `sifts_pdb_uniprot.csv` | 965,937 | SIFTS: PDB chain -> UniProt accession (canonical target identity bridge) |
| `sifts_pdb_pfam.csv` | 1,022,861 | SIFTS: PDB chain -> Pfam domain family (fold-level annotation) |

## Recent changes

### v2: Qwen 2.5 32B upgrade (2026-06-24)

- Upgraded base model from Qwen 2.5 7B to 32B (4-bit, MLX); trained with 32 adapted layers (vs 16 for 7B)
- Best val loss 0.347 at step 675 (vs 0.389 for 7B): 10.8% improvement
- Fused model saved to `chem_sage_32b/` (17 GB); serves on 64 GB Apple Silicon at 24.9 GB peak memory
- Eval shows 100% SMILES validity, 41% tool executability, 10% numerical fidelity (7B scored 0% across the board)
- Added tqdm progress bar and live metrics to `eval/eval_chem.py`
- System prompt injection in eval harness (test examples lack system messages)
- Fixed `merge_export.py`: uses `sys.executable -m mlx_lm fuse` (corrected deprecated syntax), resolves adapter path relative to config directory

### v1: Corpus fast-path and chat CLI

- All 17 corpus CSVs searchable directly from the chat CLI via keyword fast-path (`rag/corpus_lookup.py`) that bypasses the LLM for enumeration queries
- Up to 500 matching rows loaded per source; pageable in 50-row pages with CSV export
- PDB binding affinity cross-reference: resolves target names to PDB IDs via `pdb_all_entries.csv`
- Deduplication by `comp_id` sorted by `comp_mw` descending (highest-MW representative per compound)
- Intent filtering: SMILES/affinity/sequence queries drop irrelevant sources from the menu
- `_CAPS_STOPWORDS` prevents format words ("SMILES") from winning keyword extraction over target names ("EGFR")
- RAG hybrid loop: top-5 corpus chunks per query via ChromaDB (BAAI/bge-base-en-v1.5, 55k chunks)
- RDKit auto-execution of code blocks in model responses
- ASCII art banner (ChemSage, `small_slant` font), Rich Markdown rendering, tabulate `rounded_outline` tables
- SMILES/InChI/IUPAC columns truncated to 48 chars; `maxcolwidths=52` prevents tabulate re-wrapping

## Stack

Everything runs locally on **Apple Silicon** via **MLX-LM** (`mlx_lm.lora` to train, `mlx_lm.fuse`
to merge, `mlx_lm.server` for an OpenAI-compatible endpoint the GUI talks to). Unified memory means
no VRAM wall: 16 GB handles ~8B, 32 GB ~14B, 64 GB ~32B. QLoRA is automatic because the base is a
4-bit MLX model. No GGUF or Ollama is required (a GGUF/Ollama route is documented as optional in
`scripts/merge_export.py`).

## To Do

### Corpus improvements
- [ ] Add more diverse Q&A types to SFT dataset: multi-step reasoning ("selectivity profile of X vs Y"), comparative questions, failure-mode examples ("this SMILES is invalid because...")
- [ ] Increase SFT dataset from 2,400 to 5,000+ examples with more RDKit code-generation demonstrations
- [ ] Add examples that explicitly teach "emit RDKit code, don't compute manually" for MW/logP/HBD/HBA/TPSA
- [ ] Include negative examples (invalid SMILES, unanswerable questions) to improve refusal quality

### Training improvements (round 2)
- [ ] Increase `iters` from 750 to 1,500; update `lr_schedule` arguments to `[2e-5, 1500]`
- [ ] Increase `batch_size` from 4 to 8 (peak memory 24.9 GB, 64 GB available: halves wall-clock time)
- [ ] Try `num_layers: 64` (all layers) since memory headroom is ~40 GB
- [ ] Consider `rank: 64` for more adapter expressivity on the 32B model
- [ ] Save adapter to `adapters/chem_sage_32b_v2_lora` to keep runs distinct

### Evaluation and model quality
- [ ] Improve tool executability from 41% to >70%: root cause is import errors and wrong RDKit function names in generated code; fix by adding more diverse RDKit code examples to SFT data
- [ ] Improve numerical fidelity from 10% to >50%: model computes values in prose instead of code; fix by adding system-prompt-following examples to SFT data
- [ ] Add perplexity benchmark against base (unfine-tuned) 32B to measure domain adaptation gain
- [ ] Add hallucination rate metric: count invented PDB IDs, ChEMBL IDs, SMILES not in corpus

### Infrastructure
- [ ] Set up HuggingFace token for authenticated downloads (avoid rate limits during training)
- [ ] Create a `scripts/run_eval.sh` that starts server, waits, runs eval, and stops server in one command

---

*Built by **Marc C. Deller, D.Phil.** · [marcdeller.com](https://marcdeller.com) · marc@marcdeller.com*
