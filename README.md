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

| | Round 1 (7B) | Round 2 (32B) | Round 3 (32B) | Round 4 (32B) | Round 5 (32B) |
|---|---|---|---|---|---|
| Total examples | 1,500 | 1,500 | 5,000 | 8,000 | 20,000 |
| Train / valid / test | 1,200 / 150 / 150 | 1,200 / 150 / 150 | 4,000 / 500 / 500 | 6,400 / 800 / 800 | 16,000 / 2,000 / 2,000 |
| Behaviour classes | 8 | 8 | 19 | 59 | 78 |
| Seed SMILES | 25 | 25 | 51 + corpus drugs | 51 + corpus drugs | 51 + corpus drugs |
| Corpus drug examples | — | — | ~1,000 | ~1,000 + protein family PDBs | ~1,200 + protein family PDBs |
| Code-block density | Low (~50%) | Low (~50%) | High (~80%) | High (~85%) | High (~90%) |
| Generator script | `--n 1500` | `--n 1500` | `--n 5000 --seed 43` | `--n 8000 --seed 47` | `--n 20000 --seed 51` |
| Validation | RDKit + SMILES | RDKit + SMILES | RDKit + SMILES | RDKit + SMILES | RDKit + SMILES |
| Token lengths (p50/p99/max) | — | — | — | — | 322 / 488 / 558 |
| Rejected examples | — | — | 243 (95%) | — | 707 (96.5%) |

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

| Parameter | Round 1 (7B) | Round 2 (32B) | Round 3 (32B) | Round 4 (32B) | Round 5 (32B, in progress) |
|---|---|---|---|---|---|
| Base model | `Qwen2.5-7B-Instruct-4bit` | `Qwen2.5-32B-Instruct-4bit` | `Qwen2.5-32B-Instruct-4bit` | `Qwen2.5-32B-Instruct-4bit` | `Qwen2.5-32B-Instruct-4bit` |
| Fine-tune type | QLoRA (4-bit base) | QLoRA (4-bit base) | QLoRA (4-bit base) | QLoRA (4-bit base) | QLoRA (4-bit base) |
| `use_rslora` | `true` | `true` | `true` | `true` | `true` |
| LoRA rank | 32 | 32 | 32 | 32 | **64** |
| LoRA scale | 64.0 | 64.0 | 64.0 | 64.0 | **90.0** |
| LoRA keys | q/k/v/o_proj + gate/up/down_proj | q/k/v/o_proj + gate/up/down_proj | q/k/v/o_proj + gate/up/down_proj | q/k/v/o_proj + gate/up/down_proj | q/k/v/o_proj + gate/up/down_proj |
| `num_layers` | 16 (of 28) | 32 (of 64) | 32 (of 64) | 32 (of 64) | 32 (of 64) |
| `iters` | 750 | 750 | 1,500 (stopped at 625) | 1,500 (complete) | **3,000** |
| `batch_size` | 4 | 4 | 4 | 4 | 4 |
| `learning_rate` | 2e-5 | 2e-5 | 2e-5 | 2e-5 | 2e-5 |
| LR schedule | Cosine, 50-step warmup | Cosine, 50-step warmup | Cosine, 50-step warmup | Cosine, 50-step warmup | Cosine, **100**-step warmup |
| `max_seq_length` | 2048 | 2048 | 2048 | 2048 | **3072** |
| `steps_per_report` | 10 | 10 | 10 | 25 (aligned with eval) | **50** (aligned with eval) |
| `val_batches` | — | — | — | 25 | **50** |
| `seed` | 42 | 42 | 43 | 51 | 51 |
| Trainable params | ~26M (0.34%) | 134M (0.41%) | 134M (0.41%) | 134M (0.41%) | **268M (0.82%)** |
| Peak memory | ~12 GB | 24.9 GB | 25.2 GB | 24.9 GB | ~33 GB est. |
| Training data | 1,200 examples | 1,200 examples | 4,000 examples | 6,400 examples | **16,000 examples** |
| Adapter path | `adapters/chem_sage_7b_v1_lora/` | `adapters/chem_sage_32b_v2_lora/` | `adapters/chem_sage_32b_v3_lora/` | `adapters/chem_sage_32b_v4_lora/` | `adapters/chem_sage_32b_v5_lora/` |
| Fused model path | `models/chem_sage_7b_v1/` | `models/chem_sage_32b_v2/` | `models/chem_sage_32b_v3/` | `models/chem_sage_32b_v4/` | `models/chem_sage_32b_v5/` |

Notes: Round 3 was configured for 1,500 iters but stopped early at actual iter 625 after val loss plateaued at ~0.060. Training was resumed from iter 100 checkpoint after a battery shutdown. 64 layers was attempted first but caused swap thrashing at 31.5 GB peak; reverted to 32 layers. batch_size=8 was also attempted but ran 4.5x slower due to memory pressure; reverted to 4. Round 4 uses 32 layers (same as R3) — 48 and 64 layers were found to be too slow (~2–2.5 min/iter) on M1 Max; 32 layers is the throughput sweet spot at ~40 sec/iter. Round 5 doubles the LoRA rank to 64 with RSLoRA scale 90.0 (effective alpha ≈ 11.25, same order as R4's 11.31), triples the iters to 3,000, extends max_seq_length to 3,072, and doubles val_batches to 50 for a more stable eval signal. Trainable parameters doubled to 268M (0.82%).

### Training loss curves

Validation loss measured every 25 steps over 25 batches. Round 3 was resumed from iter 100 checkpoint after a battery shutdown; actual iter = logged iter + 100 for the resumed portion.

| Iter | R1 Val | R2 Val | R3 Val | R4 Val | R5 Val | Notes |
|------|--------|--------|--------|--------|--------|-------|
| 1 | 3.800 | 3.191 | 2.110 | 2.397 | **2.210** | Baseline; R5 lower than R4 despite 2.5× data — rank=64 adapter fits better |
| 25 | 1.020 | 0.580 | 0.238 | 0.480 | — | |
| 50 | 0.740 | 0.533 | 0.145 | 0.277 | — | R3 already below R2's best |
| 75 | 0.630 | 0.527 | 0.122 | 0.200 | — | |
| 100 | 0.580 | 0.495 | 0.110 | 0.146 | — | R3 battery shutdown; resumed from here |
| 200 | 0.490 | 0.449 | 0.116 | 0.109 | — | |
| 300 | 0.440 | 0.416 | 0.076 | 0.082 | — | |
| 400 | 0.420 | 0.400 | 0.059 | 0.074 | — | |
| 500 | 0.405 | 0.375 | 0.060 | 0.071 | — | |
| 600 | 0.395 | 0.359 | 0.060 | 0.059 | — | R4 matches R3's best |
| 625 | — | — | **0.054** | 0.055 | — | **Best R3 checkpoint (fused)** |
| 650 | — | — | — | 0.052 | — | R4 beats R3 best |
| 675 | **0.389** | **0.347** | — | 0.069 | — | Best R1/R2 checkpoint |
| 700 | 0.390 | 0.372 | — | 0.058 | — | |
| 750 | — | — | — | 0.056 | — | |
| 775 | — | — | — | 0.049 | — | |
| 800 | — | — | — | 0.051 | — | |
| 950 | — | — | — | **0.041** | — | **Best R4 checkpoint (fused)** |
| 1000 | — | — | — | 0.043 | — | |
| 1100 | — | — | — | 0.043 | — | |
| 1425 | — | — | — | **0.041** | — | Ties R4 best at iter 950 |
| 1500 | — | — | — | 0.050 | — | R4 final iter |
| 1600 | — | — | — | — | **0.055** | **Best R5 checkpoint (fused)** |
| 2000 | — | — | — | — | 0.061 | R5 early stop |

Round 3 val loss (**0.054**) is **6.4x better** than Round 2 (0.347) and **7.2x better** than Round 1 (0.389). The improvement is driven by dataset quality (19 behaviour classes with code-heavy examples) and 3.3x more training data (4,000 vs 1,200 examples). Round 3 was stopped early at iter 625 of 1,500 after val loss plateaued.

Round 4 **completed 2026-06-27** (1,500 iters, ~43 h). Best val loss: **0.041** at iter 950, beating R3's best of 0.054 by **24%**. The best was also tied at iter 1,425. Fused from iter 950 checkpoint into `models/chem_sage_32b_v4/` (~17.2 GB). Peak memory 25.5 GB, 32 adapted layers, 134M trainable parameters (0.41%).

Round 5 **early-stopped 2026-06-29** at iter 2000 of 3,000. Val loss plateaued at 0.055 from iter 1600 (true best 0.054 at iters 1650 and 1950, but fell between checkpoint saves). Fused from iter 1600 checkpoint into `models/chem_sage_32b_v5/` (~17.2 GB). Higher floor than R4 (0.055 vs 0.041) reflects the harder task: 78 behaviour classes vs 59, 20k examples vs 6.4k.

### Evaluation results

#### Original scorecard eval (per-round, `eval/eval_chem.py`)

| Metric | Round 1 (7B) | Round 2 (32B) | Round 3 (32B) |
|--------|------|-------|-------|
| SMILES validity | 0/0 (n/a) | 25/25 (100%) | **494/499 (99%)** |
| Tool executability | 0/8 (0%) | 72/174 (41%) | **437/442 (99%)** |
| Numerical fidelity | 0/0 (n/a) | 7/73 (10%) | **363/533 (68%)** |
| Code blocks generated | 8 | 174 | 442 |
| **Overall** | **0%** | **51%** | **89%** |

Scorecard files: `eval/scorecards/scorecard_r1_7b.html`, `scorecard_r2_32b.html`, `scorecard_r3_32b_v2.html`.

#### Comparative eval — all 4 rounds (2026-06-27, `eval/compare/eval_compare.py`)

100 shared R4 test examples (seed=42). Full report: `eval/compare/results/compare_20260627_1753.html`.

Scores use the corrected eval harness (v4, `eval_rescore.py`): PyMOL command scripts excluded
from PyExec; Biopython/Plotly installed; FileNotFoundError forgiven; QED regex fixed; QED in
numerical fidelity; degeneration + code-attempted + PDB ID validity + PyMOL syntax + SMARTS
validity metrics added (13 total).

| Metric | Round 1 | Round 2 | Round 3 ★ | Round 4 | Δ vs R3 |
|---|---|---|---|---|---|
| SMILES validity | 97% | 98% | 98% | **98%** | +0% |
| Tool executability (RDKit) | 78% | 81% | **87%** | 74% | −13% |
| Numerical fidelity | 65% | 62% | 57% | **61%** | +4% |
| Rounding precision *(v3)* | 98% | 98% | 97% | **95%** | −2% |
| Code→Quote fidelity *(v3)* | 57% | 52% | 50% | **59%** | +9% |
| Refusal accuracy *(v3)* | 100% | 100% | 99% | **100%** | +1% |
| QED range check *(v3)* | 100% | 100% | 100% | **100%** | +0% |
| Python exec (extended) *(v3)* | 79% | 86% | **92%** | 75% | −17% |
| Degeneration-free *(v3)* | 87% | 86% | 88% | **92%** | +4% |
| Code Attempted *(v3)* | 65% | 70% | 75% | **80%** | +5% |
| PDB ID Validity *(v4)* | 100% | 100% | 100% | **100%** | +0% |
| PyMOL Syntax Check *(v4)* | 100% | 98% | 99% | **100%** | +1% |
| SMARTS Validity *(v4)* | 86% | 100% | 100% | **100%** | +0% |

R4 wins on its trained targets: **Code→Quote +9%**, **Code Attempted +5%**, **Degeneration-free +4%**,
**Refusal 100%**, **Fidelity +4%**, **PDB ID Validity 100%**, **PyMOL Syntax 100%**, **SMARTS 100%**.

The Exec and PyExec gaps vs R3 are genuine model errors — wrong RDKit API names for targeted
single-property questions (QED, TPSA, fingerprint similarity) that are underrepresented as
standalone question types in the training data. These are R5 training data targets.

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

### Eval harness fixes and 5-round comparative eval (2026-06-29)

**Tokenizer fix (all models):**
- `tokenizer_config.json` in `chem_sage_7b_v1`, `chem_sage_32b_v2`, `chem_sage_32b_v3`, `chem_sage_32b_v4` had `extra_special_tokens` as a JSON list of strings. Newer `transformers` calls `.keys()` on this field — causing an `AttributeError` on load. The server silently swallowed the error, loaded a broken tokenizer, and then hung indefinitely on every inference request (all queries timed out). Fixed by converting to an empty dict `{}` in all four files. `chem_sage_32b_v5` was already in the correct format.

**Eval harness model-path fix (`eval/compare/eval_compare.py`):**
- `query_model()` previously sent `"model": "chemsage"` in every HTTP POST to the local `mlx_lm.server`. Since `mlx_lm` 0.31.3, an unknown model name triggers a 120-second HuggingFace lazy-load timeout, causing all R1–R4 evals to time out. Fixed by passing the full local model directory path (`model_path`) as the `"model"` field.

**Per-example pass/fail scoring:**
- Changed the eval metric methodology throughout `eval_compare.py` from per-instance accuracy (`ok / total_attempts`) to per-example pass/fail (`passed_examples / 100`). An example "passes" a metric if every instance in that response is correct (`ok == tot`, `tot > 0`). This makes round-over-round comparisons meaningful: R1 calling 2 SMILES correctly is no longer the same score as R3 calling 100 SMILES correctly. The `_per_example_agg()` helper re-derives this from `results[].scores` in `raw_results.json`.

**5-round comparative eval (in progress, 2026-06-29):**
- Running all 5 rounds on 100 shared R5 test examples (seed=42) via `eval/compare/eval_compare.py`. R1 and R2 complete; R3–R5 in progress. Full report in `eval/compare/results/` on completion. All README eval tables will be updated with final numbers.

### Round 5: Higher rank + 2.5× corpus (2026-06-28, training in progress)

- Expanded SFT dataset from 8,000 to **20,000 examples** (16,000 train / 2,000 val / 2,000 test) across **78 behaviour classes** (19 new vs R4)
- Raised LoRA rank to **64** (from 32); scale set to **90.0** (RSLoRA effective alpha ≈ 11.25, same order as R4); trainable parameters **doubled to 268M (0.82%)**
- Extended `max_seq_length` to **3,072** (from 2,048); all corpus examples fit (p99=488, max=558 tokens)
- Tripled training iters to **3,000**; doubled `val_batches` to 50 for a more stable eval signal
- New generators targeting R4 exec/PyExec gap and numerical fidelity: `pyexec_drill` (×6), `code_then_quote_v2` (×5), `rounding_explicit` (×5), `fidelity_multistep` (×5)
- New content generators: `herg_liability`, `selectivity_profile`, `prodrug_bcs`, `sar_delta`, `mdanalysis`, `dssp`, `ppi_interface`, `uniprot_api`, `conformer_3d`, `mcs_search`, `reaction_smarts`, `recap_fragmentation`, `electron_density`, `biological_assembly`, `drug_target_family` (×3-4 each)
- 707 examples rejected during validation (96.5% pass rate); token lengths all under max_seq_length (p99=488)
- Training launched 2026-06-28 with `caffeinate -i` under single-process MLX launch (`/tmp/r5_launch.py`) — memory limits set in-process before importing mlx_lm; cache 16.7 GB / memory limit 50.1 GB
- **Iter 1 val loss: 2.210** (lower than R4's 2.397 — rank=64 adapter has more capacity from the start)
- Config: `config/train_config_r5.yaml`; adapter path: `adapters/chem_sage_32b_v5_lora/`
- **Early stop at iter 2000** — val loss plateaued at 0.055 from iter 1600; fused from iter 1600 checkpoint into `models/chem_sage_32b_v5/` (~17.2 GB)

### Round 4: Expanded dataset training (2026-06-25 to 2026-06-27, complete)

- Expanded SFT dataset from 5,000 to 8,000 examples (6,400 train / 800 val / 800 test) across 59 behaviour classes
- New behaviour classes: drug mechanisms/indications/ATC, multi-step reasoning, refusals, personality, structural biology, docking, visualisation (Plotly, R/ggplot2), numerical fidelity drills (`code_then_quote`, `rounding_precision`)
- `steps_per_report` aligned with `steps_per_eval` (both 25) so train/val loss appear at the same iters
- Training ran to completion: 1,500 iters, ~43 h (2026-06-25 15:39 to 2026-06-27 10:59 BST)
- **Best val loss: 0.041 at iter 950** — beats R3's all-time best of 0.054 by **24%** (tied at iter 1425)
- Peak memory 25.5 GB, 32 adapted layers, 134M trainable parameters (0.41%)
- Fused from iter 950 checkpoint into `models/chem_sage_32b_v4/` (~17.2 GB, 4 safetensors shards)

### Comparative eval harness (2026-06-26)

New script `eval/compare/eval_compare.py` runs all 4 model rounds side-by-side on the same 100 R4 test examples and produces a rich HTML + Markdown comparison report.

**Features:**
- **Option A auto-server**: spawns and kills `mlx_lm.server` per model automatically (no manual terminal juggling); health-checks the OpenAI-compatible endpoint before starting eval; sends SIGTERM to the process group on completion
- **Shared examples**: 100 examples sampled from the R4 test set with a fixed seed (reproducible)
- **All 13 metrics**: 3 core + 7 new metrics, all reporting `n/a` gracefully on older models
- **R3 as baseline**: metric table includes a Δ column (±%) vs Round 3; colour-coded green/red
- **Model profile panel**: full spec comparison across architecture, training config, training outcomes, corpus, and eval runtime
- **Runtime resource stats**: server process CPU mean/peak %, RSS mean/peak GB (via `psutil`), mean latency (s), throughput (tok/s)
- **Validation loss curves**: SVG comparison chart of all 4 models' val loss curves (clipped at 0.60; dashed line for estimated curves)
- **Per-example table**: 100 collapsible rows; click "responses" to see all model outputs side-by-side
- **Resume flag**: `--resume` loads `raw_results.json` cache and skips already-evaluated models
- **Branding**: ChemSage ASCII art (pyfiglet), clickable [marcdeller.com](https://marcdeller.com) links throughout

**Usage** (run after R4 training completes and fuse is done):
```bash
cd chem_sage/
# Full run — all 4 models
python eval/compare/eval_compare.py

# Subset and resume
python eval/compare/eval_compare.py --models round3 round4 --resume

# Quick smoke-test (10 examples)
python eval/compare/eval_compare.py --limit 10
```

**Config:** `eval/compare/models.yaml` — all model metadata, loss curves, and training stats in one place; edit to update R4 final values after training.

### ChromaDB 1.x compatibility fix (2026-06-25)

- ChromaDB 1.0+ switched to a Rust connection backend; `PersistentClient` timed out on databases created with the old Python backend
- Fix: pass `Settings(allow_reset=True, anonymized_telemetry=False)` to `PersistentClient` in both `rag/retrieve.py` and `scripts/ingest_rag.py`
- No data loss, no re-ingest — all 55,742 indexed chunks preserved; confirmed connected and serving queries
- Downgrade to `chromadb<1.0.0` is not possible on Python 3.14 (no pre-built `chroma-hnswlib` wheel)

### Chat CLI: Streaming output and UX polish (2026-06-25)

**Streaming generation:**
- Response tokens now stream live to the terminal in real-time using `mlx_lm.stream_generate` + Rich `Live`; raw text streams token-by-token during generation, then the final response re-renders as formatted Markdown — no more waiting behind a "Thinking…" spinner
- Falls back to blocking `generate` + spinner if `stream_generate` is unavailable

**New slash commands:**
- `/save` — exports the full conversation to a timestamped Markdown file on the Desktop (includes marcdeller.com attribution header)
- `/info` — shows a panel with loaded model name/size, training config (iters, RSLoRA rank, lr, ctx), corpus stats (chunks, CSV files, MB), and embed model; clickable hyperlinks to [marcdeller.com](https://marcdeller.com) and [marc@marcdeller.com](mailto:marc@marcdeller.com)
- `/retry` — re-generates the previous response using the same RAG context; replaces the last assistant turn in history

**UX refinements:**
- **Tab completion**: type `/` and press Tab for a dropdown of all slash commands; completion menu styled to match the ChemSage colour scheme
- **RAG retrieval indicator**: `🔍 Retrieving context…` spinner during retrieval, then `📚 N chunks retrieved` confirmation line before generation begins
- **Context window indicator in toolbar**: bottom toolbar now shows `ctx N/3` alongside the turn counter so you can see how much of the 3-turn context window is filled
- **`/reset` confirmation**: asks `Clear history? [y/N]` before wiping — no accidental data loss
- **Generation stats**: now includes elapsed seconds — `⚡ N tok · X.X tok/s · X.Xs`
- **Styled goodbye**: exit via `quit`/`q`/Ctrl-C shows a full-screen Panel with the ChemSage ASCII banner, "Thank you for using ChemSage", and clickable links to [marcdeller.com](https://marcdeller.com) and [marc@marcdeller.com](mailto:marc@marcdeller.com)
- **Hyperlinks throughout**: banner, `/help`, `/info`, `/save` output, and goodbye Panel all include clickable `[link=...]` Rich markup pointing to marcdeller.com and marc@marcdeller.com

**Earlier polish (same date):**
- `prompt_toolkit` input with arrow-key history and styled `You:` prompt
- Persistent bottom toolbar: `⬡ ChemSage · model · QLoRA · RAG · turn N · ctx N/3`
- Corpus tables: Rich `Table` with rounded cyan borders, single-line rows (ellipsis overflow), abbreviated headers, `✓`/`✗` booleans, method abbreviations
- RDKit output in yellow-bordered `Panel`; retrieval errors in red-bordered `Panel`
- Terminal title set to `ChemSage` on launch and `/clear`
- Added `rich`, `prompt_toolkit`, `pyfiglet` to `requirements.txt`

### Eval harness: R4 extended metrics (2026-06-25)

Extended `eval/eval_chem.py` with new metrics that degrade gracefully to `n/a` when run against R1-R3 models/datasets, enabling a clean side-by-side comparison across all rounds:

| New metric | What it measures | Why added |
|---|---|---|
| **Rounding precision** | MW 1 d.p. · logP 2 d.p. · TPSA 1 d.p. · QED 2 d.p. | R4 drills exact decimal-place conventions |
| **Code-then-quote** | prose values match what code blocks actually printed to stdout | Distinguishes quoting own code vs. recalling from memory |
| **Extended executability** | all `python` blocks (RDKit + Plotly + Biopython) | R4 adds Plotly and Biopython code generation |
| **Refusal accuracy** | out-of-scope prompts correctly declined | R4 adds explicit refusal training |
| **QED range** | all stated QED values in [0, 1] | R4 adds QED computation training |

Backwards compatibility: R1-R3 test sets have no refusal examples (refusal shows n/a), no QED mentions (QED shows n/a), code blocks may not print (code-then-quote shows n/a). All five new metrics return `(0, 0)` when inapplicable. The core 3-metric overall score formula is unchanged for direct cross-round comparison.

New scorecard features: per-class breakdown table (if test examples carry a `class` field), overall score card, collapsible response viewer in per-example table, extended exec breakdown by library category, mean latency in header, cleaner terminal summary with n/a annotation.

### Round 3: Code-heavy dataset and extended training (2026-06-25)

- Expanded SFT dataset from 1,500 to 5,000 examples (4,000 train) across 19 behaviour classes
- 8 new generator classes targeting eval gaps: `single_property`, `full_profile`, `qed_score`, `molecular_formula`, `smiles_validation`, `property_comparison`, `corpus_drug`, `murcko_sar`
- 51 seed SMILES + ~1,000 corpus drug examples from `approved_drugs.csv`
- Configured for 1,500 iters; stopped early at iter 625 after val loss plateaued at ~0.060
- Best val loss **0.054** at iter 625 (vs 0.347 for Round 2): **6.4x improvement**
- Attempted 64 layers (swap thrashing at 31.5 GB) and batch_size=8 (4.5x slower); reverted to 32 layers / batch 4
- Resumed from iter 100 checkpoint after battery shutdown; no data loss
- Eval: SMILES 99%, Exec 99%, Fidelity 68% (vs R2: 100%, 41%, 10%)
- Fused model saved to `models/chem_sage_32b_v3/` (17 GB)

### Round 2: Qwen 2.5 32B upgrade (2026-06-24)

- Upgraded base model from Qwen 2.5 7B to 32B (4-bit, MLX); trained with 32 adapted layers (vs 16 for 7B)
- Best val loss 0.347 at step 675 (vs 0.389 for 7B): 10.8% improvement
- Fused model saved to `models/chem_sage_32b_v2/` (17 GB); serves on 64 GB Apple Silicon at 24.9 GB peak memory
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
- ASCII art banner (ChemSage, `small_slant` font), Rich Markdown rendering, Rich `Table` corpus tables with rounded cyan borders
- Column display config (`_COL_CFG`) maps all corpus column names to abbreviated headers and per-column max widths; ellipsis overflow keeps all rows single-line

## Stack

Everything runs locally on **Apple Silicon** via **MLX-LM** (`mlx_lm.lora` to train, `mlx_lm.fuse`
to merge, `mlx_lm.server` for an OpenAI-compatible endpoint the GUI talks to). Unified memory means
no VRAM wall: 16 GB handles ~8B, 32 GB ~14B, 64 GB ~32B. QLoRA is automatic because the base is a
4-bit MLX model. No GGUF or Ollama is required (a GGUF/Ollama route is documented as optional in
`scripts/merge_export.py`).

The chat CLI (`scripts/chat.py`) uses **Rich** for streaming Markdown output (`Live`), corpus tables
(rounded borders, ellipsis overflow, abbreviated headers), panels, progress spinners, and a themed
console; **prompt_toolkit** for the interactive input session (arrow-key history, slash-command tab
completion, styled prompt, persistent bottom toolbar with context window indicator); **ChromaDB** +
**sentence-transformers** for the RAG retrieval layer; and **RDKit** for live chemical property
computation. Responses stream token-by-token via `mlx_lm.stream_generate` and re-render as Markdown
on completion. New commands: `/save` (export conversation to Markdown), `/info` (session/model
details), `/retry` (regenerate last response).

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

### Round 5 dataset (complete)
- [x] 20,000 examples across 78 behaviour classes (`build_dataset.py --n 20000 --seed 51`)
- [x] 19 new generator classes targeting R4 exec/PyExec gap: `pyexec_drill` (×6), `code_then_quote_v2` (×5), `rounding_explicit` (×5), `fidelity_multistep` (×5), `herg_liability`, `selectivity_profile`, `prodrug_bcs`, `sar_delta`, `mdanalysis`, `dssp`, `ppi_interface`, `uniprot_api`, `conformer_3d`, `mcs_search`, `reaction_smarts`, `recap_fragmentation`, `electron_density`, `biological_assembly`, `drug_target_family`
- [x] Token lengths p50=322, p99=488, max=558 — all under max_seq_length=3072
- [x] 707 examples rejected (96.5% pass rate); 78 classes, class balance max:median < 3:1
- [x] `data/sft_r5/` — 16,000 train / 2,000 val / 2,000 test

### Round 5 training (complete — early stop)
- [x] Config: `config/train_config_r5.yaml` — rank=64, scale=90, iters=3000, max_seq=3072, val_batches=50
- [x] Launch: single-process `caffeinate -i .venv/bin/python /tmp/r5_launch.py` (cache 16.7 GB, memory limit 50.1 GB)
- [x] **Training started 2026-06-28** — Iter 1 val loss 2.210
- [x] No rank=64 instability — converged normally
- [x] **Early stop at iter 2000** — val loss plateaued at 0.055 from iter 1600 (best: 0.054 at iters 1650 and 1950, unsaved)
- [x] Best available checkpoint: iter 1600, val loss **0.055** → copied to `adapters.safetensors`
- [x] Fused into `models/chem_sage_32b_v5/` (~17.2 GB)
- [ ] Run `eval/compare/eval_compare.py` after fuse

### Round 4 dataset (complete)
- [x] 8,000 examples across 59 behaviour classes (`build_dataset.py --n 8000 --seed 47`)
- [x] Protein family coverage: kinases (7,687 PDB), antibodies (3,322), GPCRs (103), GTPases (359), cytokines (769), receptors (6,891) — real PDB IDs and ligand SMILES from corpus
- [x] Numerical fidelity drills: `code_then_quote` enforces "compute then quote" pattern; `rounding_precision` teaches MW 1dp, logP 2dp, TPSA 1dp
- [x] Structural biology: PyMOL selection algebra, visualization recipes, PLIP interaction analysis, PDB format, pdb-tools, Biopython Bio.PDB, ChimeraX, gemmi, structure preparation, docking interpretation
- [x] Visualization: Plotly Python (scatter, radar), R/ggplot2 (SAR plots), bio3d (RMSD, RMSF)
- [x] Set `steps_per_report: 25` to align train/val reporting at same iters

### Round 4 training (complete)
- [x] `num_layers: 32` — 48 and 64 layers tested and found too slow (~2 min/iter); 32 is the M1 Max sweet spot
- [x] Adapter path: `adapters/chem_sage_32b_v4_lora` (checkpoints saved every 200 iters)
- [x] Training completed 2026-06-27 10:59 BST — 1,500 iters, ~43 h
- [x] Best checkpoint: iter 950, val loss **0.041** (also tied at iter 1425)
- [x] Copied `0000950_adapters.safetensors` → `adapters.safetensors`
- [x] Fused into `models/chem_sage_32b_v4/` (~17.2 GB)
- [x] Updated `eval/compare/models.yaml` with final stats and complete val loss curve
- [x] Raised rank to 64 in Round 5 for more adapter expressivity

### Evaluation and model quality
- [x] Complete Round 3 eval: SMILES 99%, Exec 99%, Fidelity 68%
- [x] Comparative eval harness: `eval/compare/eval_compare.py` with 13 metrics, loss curves, resource stats
- [x] Comparative eval run after R4 fuse: full 4-round report `eval/compare/results/compare_20260627_1753.html`
- [ ] Run R5 comparative eval after fuse: add Round 5 to `eval/compare/models.yaml` and run `eval_compare.py`
- [ ] Improve exec/PyExec from R4's 74%/75% to >85% (R5 training target: `pyexec_drill`, `code_then_quote_v2`)
- [ ] Add perplexity benchmark against base (unfine-tuned) 32B
- [ ] Add hallucination rate metric: count invented PDB IDs, ChEMBL IDs, SMILES not in corpus

### Infrastructure
- [x] ChromaDB 1.x fix: `allow_reset=True` in `rag/retrieve.py` and `scripts/ingest_rag.py` — 55,742 chunks preserved, no re-ingest required
- [x] `psutil` added to `requirements.txt` for server process monitoring in comparative eval
- [ ] Set up HuggingFace token for authenticated downloads
- [ ] Create `scripts/run_eval.sh` (start server, wait, run eval, stop server)

---

*Built by **Marc C. Deller, D.Phil.** · [marcdeller.com](https://marcdeller.com) · marc@marcdeller.com*
