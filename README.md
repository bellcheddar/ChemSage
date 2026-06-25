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

### Training data

| | Round 1 (7B) | Round 2 (32B) | Round 3 (32B) | Round 4 (32B, planned) |
|---|---|---|---|---|
| Total examples | 1,500 | 1,500 | 5,000 | 8,000 |
| Train / valid / test | 1,200 / 150 / 150 | 1,200 / 150 / 150 | 4,000 / 500 / 500 | 6,400 / 800 / 800 |
| Behaviour classes | 8 | 8 | 19 | 59 |
| Seed SMILES | 25 | 25 | 51 + corpus drugs | 51 + corpus drugs |
| Corpus drug examples | — | — | ~1,000 | ~1,000 + protein family PDBs |
| Code-block density | Low (~50%) | Low (~50%) | High (~80%) | High (~85%) |
| Generator script | `--n 1500` | `--n 1500` | `--n 5000 --seed 43` | `--n 8000 --seed 47` |
| Validation | RDKit + SMILES | RDKit + SMILES | RDKit + SMILES | RDKit + SMILES |

**Behaviour classes by round:**

*Round 3 additions (targeting R2 eval gaps):*

| Class | Weight | Purpose |
|---|---|---|
| `single_property` | 2x | One property at a time with code block |
| `full_profile` | 2x | Complete property profile via code |
| `qed_score` | 1x | QED drug-likeness score |
| `molecular_formula` | 1x | Formula, atom counts, ring analysis |
| `smiles_validation` | 1x | Valid/invalid SMILES handling |
| `property_comparison` | 1x | Side-by-side two-molecule comparison |
| `corpus_drug` | 2x | Named drugs from approved_drugs.csv |
| `murcko_sar` | 1x | Scaffold comparison with SAR context |

*Round 4 additions (structural biology + fidelity):*

| Class | Weight | Purpose |
|---|---|---|
| `pymol_selection` | 2x | PyMOL selection algebra (within, byres, chain, resi) |
| `pymol_viz` | 2x | PyMOL visualization (surface, alignment, symmetry, B-factor, figures) |
| `plip_analysis` | 2x | PLIP interaction types (H-bond, hydrophobic, pi-stack, salt bridge, halogen) |
| `pdb_format` | 2x | PDB file format (ATOM/HETATM, altloc, B-factor interpretation) |
| `pdbtools` | 2x | pdb-tools CLI recipes (selchain, selres, tidy, splitmodel) |
| `biopython` | 2x | Bio.PDB patterns (SMCRA, NeighborSearch, Superimposer) |
| `chimerax` | 2x | ChimeraX commands (hbonds, clashes, matchmaker) |
| `gemmi` | 1x | gemmi mmCIF/PDB parsing and neighbour search |
| `struct_prep` | 1x | Structure preparation (PDBFixer, protonation, 3D conformers) |
| `docking_interp` | 2x | Docking result interpretation (Vina scores, RMSD, Kd estimation) |
| `plotly` | 2x | Plotly Python visualization (MW vs logP, radar charts) |
| `r_recipe` | 2x | R/ggplot2/bio3d/Plotly-R (SAR plots, RMSD, RMSF) |
| `protein_family` | 3x | Kinases, GPCRs, antibodies, GTPases, cytokines, receptors (real PDB data) |
| `code_then_quote` | 3x | "Compute then quote" pattern for numerical fidelity |
| `rounding_precision` | 2x | Exact rounding conventions (MW 1dp, logP 2dp, TPSA 1dp) |
| `personality_drug` | 3x | Lipinski verdicts with ✅/❌ cards, emoji, and medchem humor |
| `personality_valid` | 2x | SMILES validation with flair and emoji |
| `personality_struct` | 3x | Structural observations with character and humor |
| `refusal` | 2x | Out-of-scope and invalid query handling |

### Training parameters

| Parameter | Round 1 (7B) | Round 2 (32B) | Round 3 (32B) | Round 4 (32B, planned) |
|---|---|---|---|---|
| Base model | `Qwen2.5-7B-Instruct-4bit` | `Qwen2.5-32B-Instruct-4bit` | `Qwen2.5-32B-Instruct-4bit` | `Qwen2.5-32B-Instruct-4bit` |
| Fine-tune type | QLoRA (4-bit base) | QLoRA (4-bit base) | QLoRA (4-bit base) | QLoRA (4-bit base) |
| `use_rslora` | `true` | `true` | `true` | `true` |
| LoRA rank | 32 | 32 | 32 | 32 |
| LoRA scale | 64.0 | 64.0 | 64.0 | 64.0 |
| LoRA keys | q/k/v/o_proj + gate/up/down_proj | q/k/v/o_proj + gate/up/down_proj | q/k/v/o_proj + gate/up/down_proj | q/k/v/o_proj + gate/up/down_proj |
| `num_layers` | 16 (of 28) | 32 (of 64) | 32 (of 64) | 48 (of 64) |
| `iters` | 750 | 750 | 1,500 (stopped at 625) | 1,500 |
| `batch_size` | 4 | 4 | 4 | 4 |
| `learning_rate` | 2e-5 | 2e-5 | 2e-5 | 2e-5 |
| LR schedule | Cosine, 50-step warmup | Cosine, 50-step warmup | Cosine, 50-step warmup | Cosine, 50-step warmup |
| `steps_per_report` | 10 | 10 | 10 | 25 (aligned with eval) |
| `seed` | 42 | 42 | 43 | 47 |
| Trainable params | ~26M (0.34%) | 134M (0.41%) | 134M (0.41%) | ~200M (~0.61%) |
| Peak memory | ~12 GB | 24.9 GB | 25.2 GB | ~28 GB (est.) |
| Training data | 1,200 examples | 1,200 examples | 4,000 examples | 6,400 examples |
| Adapter path | `adapters/chem_sage_lora/` | `adapters/chem_sage_32b_lora/` | `adapters/chem_sage_32b_v2_lora/` | `adapters/chem_sage_32b_v3_lora/` |
| Fused model path | `models/chem_sage_fused/` | `chem_sage_32b/` | `chem_sage_32b_v2/` | `chem_sage_32b_v3/` |

Notes: Round 3 was configured for 1,500 iters but stopped early at actual iter 625 after val loss plateaued at ~0.060. Training was resumed from iter 100 checkpoint after a battery shutdown. 64 layers was attempted first but caused swap thrashing at 31.5 GB peak; reverted to 32 layers. batch_size=8 was also attempted but ran 4.5x slower due to memory pressure; reverted to 4. Round 4 will try 48 layers as a compromise (~28 GB estimated peak).

### Training loss curves

Validation loss measured every 25 steps over 25 batches. Round 3 was resumed from iter 100 checkpoint after a battery shutdown; actual iter = logged iter + 100 for the resumed portion.

| Iter | R1 Val | R2 Val | R3 Val | Notes |
|------|--------|--------|--------|-------|
| 1 | 3.800 | 3.191 | 2.110 | Baseline (untrained) |
| 25 | 1.020 | 0.580 | 0.238 | |
| 50 | 0.740 | 0.533 | 0.145 | R3 already below R2's best |
| 75 | 0.630 | 0.527 | 0.122 | |
| 100 | 0.580 | 0.495 | 0.110 | R3 battery shutdown; resumed from here |
| 150 | 0.530 | 0.496 | 0.141 | Post-resume wobble |
| 200 | 0.490 | 0.449 | 0.116 | |
| 250 | 0.460 | 0.436 | 0.087 | |
| 300 | 0.440 | 0.416 | 0.077 | |
| 350 | 0.430 | 0.407 | 0.068 | |
| 400 | 0.420 | 0.400 | 0.077 | Slight wobble |
| 450 | 0.410 | 0.393 | 0.068 | |
| 500 | 0.405 | 0.375 | **0.059** | |
| 550 | 0.400 | 0.366 | 0.062 | Plateau |
| 600 | 0.395 | 0.359 | 0.060 | |
| 625 | — | — | **0.054** | **Best R3 checkpoint (fused)** |
| 675 | **0.389** | **0.347** | — | Best R1/R2 checkpoint |
| 750 | 0.390 | 0.372 | — | R1/R2 final |

Round 3 val loss (**0.054**) is **6.4x better** than Round 2 (0.347) and **7.2x better** than Round 1 (0.389). The improvement is driven by dataset quality (19 behaviour classes with code-heavy examples) and 3.3x more training data (4,000 vs 1,200 examples). Round 3 was stopped early at iter 625 of 1,500 after val loss plateaued.

### Evaluation results

Evaluated on held-out test examples via `eval/eval_chem.py` against `mlx_lm.server`. System prompt injected to instruct RDKit code generation. Round 3 eval pending.

| Metric | Round 1 (7B) | Round 2 (32B) | Round 3 (32B) |
|--------|------|-------|-------|
| SMILES validity | 0/0 (n/a) | 25/25 (100%) | **494/499 (99%)** |
| Tool executability | 0/8 (0%) | 72/174 (41%) | **437/442 (99%)** |
| Numerical fidelity | 0/0 (n/a) | 7/73 (10%) | **363/533 (68%)** |
| Code blocks generated | 8 | 174 | 442 |
| **Overall** | **0%** | **51%** | **89%** |

Scorecard files: `eval/scorecard_7b.html` (R1), `eval/scorecard_32b.html` (R2), `eval/scorecard_32b_v2.html` (R3)

Round 3 represents a massive improvement: tool executability jumped from 41% to **99%** (every code block runs), SMILES validity held at **99%**, and numerical fidelity improved from 10% to **68%**. The remaining fidelity gap (model computes values in prose rather than quoting code output) is targeted by RAFT-style records in the Round 4 dataset.

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

### Round 3: Code-heavy dataset and extended training (2026-06-25)

- Expanded SFT dataset from 1,500 to 5,000 examples (4,000 train) across 19 behaviour classes
- 8 new generator classes targeting eval gaps: `single_property`, `full_profile`, `qed_score`, `molecular_formula`, `smiles_validation`, `property_comparison`, `corpus_drug`, `murcko_sar`
- 51 seed SMILES + ~1,000 corpus drug examples from `approved_drugs.csv`
- Configured for 1,500 iters; stopped early at iter 625 after val loss plateaued at ~0.060
- Best val loss **0.054** at iter 625 (vs 0.347 for Round 2): **6.4x improvement**
- Attempted 64 layers (swap thrashing at 31.5 GB) and batch_size=8 (4.5x slower); reverted to 32 layers / batch 4
- Resumed from iter 100 checkpoint after battery shutdown; no data loss
- Eval: SMILES 99%, Exec 99%, Fidelity 68% (vs R2: 100%, 41%, 10%)
- Fused model saved to `chem_sage_32b_v2/` (17 GB)

### Round 2: Qwen 2.5 32B upgrade (2026-06-24)

- Upgraded base model from Qwen 2.5 7B to 32B (4-bit, MLX); trained with 32 adapted layers (vs 16 for 7B)
- Best val loss 0.347 at step 675 (vs 0.389 for 7B): 10.8% improvement
- Fused model saved to `chem_sage_32b/` (17 GB); serves on 64 GB Apple Silicon at 24.9 GB peak memory
- Eval shows 100% SMILES validity, 41% tool executability, 10% numerical fidelity (7B scored 0% across the board)
- Added tqdm progress bar and live metrics to `eval/eval_chem.py`
- System prompt injection in eval harness (test examples lack system messages)
- Fixed `merge_export.py`: uses `sys.executable -m mlx_lm fuse` (corrected deprecated syntax), resolves adapter path relative to config directory

### Round 1: Corpus fast-path and chat CLI

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

### Completed (Rounds 1-3)
- [x] Increase SFT dataset from 1,500 to 5,000 examples with code-heavy generators (Round 3)
- [x] Add examples that teach "emit RDKit code, don't compute manually" (Round 3: `single_property`, `full_profile`)
- [x] Include SMILES validation examples (Round 3: `smiles_validation`)
- [x] Increase `iters` from 750 to 1,500 (Round 3; stopped early at 625 — val loss plateaued)
- [x] Try `num_layers: 64` — caused swap thrashing at 31.5 GB; reverted to 32
- [x] Try `batch_size: 8` — 4.5x slower due to memory pressure; reverted to 4
- [x] Add structural biology generators: PyMOL selections/viz, PLIP, PDB format, pdb-tools, Biopython, ChimeraX, gemmi
- [x] Add Plotly and R/ggplot2/bio3d recipe generators
- [x] Add protein family generators: kinases, GPCRs, antibodies, GTPases, cytokines, receptors (real PDB data)
- [x] Add numerical fidelity drills: `code_then_quote` (3x), `rounding_precision` (2x)
- [x] Add refusal/out-of-scope examples
- [x] Expand dataset to 8,000 examples across 51 behaviour classes

### Round 4 dataset (ready)
- [x] 8,000 examples across 51 behaviour classes (`build_dataset.py --n 8000 --seed 47`)
- [x] Protein family coverage: kinases (7,687 PDB), antibodies (3,322), GPCRs (103), GTPases (359), cytokines (769), receptors (6,891) — real PDB IDs and ligand SMILES from corpus
- [x] Numerical fidelity drills: `code_then_quote` enforces "compute then quote" pattern; `rounding_precision` teaches MW 1dp, logP 2dp, TPSA 1dp
- [x] Structural biology: PyMOL selection algebra, visualization recipes, PLIP interaction analysis, PDB format, pdb-tools, Biopython Bio.PDB, ChimeraX, gemmi, structure preparation, docking interpretation
- [x] Visualization: Plotly Python (scatter, radar), R/ggplot2 (SAR plots), bio3d (RMSD, RMSF)
- [ ] Set `steps_per_report: 25` to align train/val reporting at same iters

### Round 4 training (planned)
- [ ] Try `num_layers: 48` (compromise: est. ~28 GB peak, within 64 GB)
- [ ] Adapter path: `adapters/chem_sage_32b_v3_lora`
- [ ] Fused model path: `chem_sage_32b_v3/`
- [ ] Consider `rank: 64` for more adapter expressivity

### Evaluation and model quality
- [x] Complete Round 3 eval: SMILES 99%, Exec 99%, Fidelity 68%
- [ ] Improve numerical fidelity from 68% to >85% via RAFT records + `code_then_quote` drills (Round 4)
- [ ] Add perplexity benchmark against base (unfine-tuned) 32B
- [ ] Add hallucination rate metric: count invented PDB IDs, ChEMBL IDs, SMILES not in corpus

### Infrastructure
- [ ] Set up HuggingFace token for authenticated downloads
- [ ] Create `scripts/run_eval.sh` (start server, wait, run eval, stop server)

---

*Built by **Marc C. Deller, D.Phil.** · [marcdeller.com](https://marcdeller.com) · marc@marcdeller.com*
