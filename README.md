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
| 1 | Base model (Qwen 2.5 7B, MLX 4-bit) | `config/train_config.yaml` |
| 2 | **RAG first** (ship this alone) | `scripts/ingest_rag.py` |
| 2a | Download ChEMBL corpus | `scripts/download_chembl.py` + `scripts/download_chembl_extended.py` |
| 2b | Download PubChem corpus | `scripts/download_pubchem.py` |
| 2c | Download PDB corpus | `scripts/download_pdb.py` + `scripts/download_pdb_supplement.py` |
| 3 | Build + validate SFT dataset | `scripts/build_dataset.py` |
| 4 | QLoRA fine-tune | `mlx_lm.lora --config config/train_config.yaml --train` |
| 5 | Fuse + serve | `mlx_lm.fuse` → `mlx_lm.server --port 8081` |
| 6 | Close the hybrid loop | `rag/tool_exec.py` |
| 7 | Chemistry-aware evaluation | `eval/eval_chem.py` |

## RAG corpus

The `data/corpus/` directory contains the full retrieval knowledge base — tool reference docs plus two databases of experimental chemistry data:

### ChEMBL (`data/corpus/chembl/`, ~70 MB)

| File | Records | Contents |
|---|---|---|
| `approved_drugs.csv` | 3,257 | All FDA/EMA approved small molecules with SMILES + properties |
| `bioactivity_key_targets.csv` | 112,357 | IC50/Ki/Kd for EGFR, CDK2, HIV protease, SARS-CoV-2 3CLpro, VEGFR2, Bcr-Abl, COX-2, BRAF, ALK, JAK1/2, hERG |
| `bioactivity_extended_targets.csv` | ~200k | IC50/Ki for 36 additional targets: BTK, PARP1, CDK4/6, FLT3, mTOR, AKT1, PI3Kα, RET, MET, SRC, HDAC1, BRD4, EZH2, AR, ERα, PPARγ, GLP-1R, Factor Xa, Thrombin, ACE, HMG-CoA, PDE5, DHFR, IDH1/2, KRAS + more |
| `admet_data.csv` | ~50k | Caco-2 permeability, CLint, t½, fu,plasma, solubility, logD, CYP3A4/2D6/2C9/1A2/2C19 inhibition |
| `clinical_candidates_phase2_3.csv` | ~10k | Phase 2–3 small molecules with properties |
| `fragment_actives.csv` | 10,000 | MW < 300 fragments with pIC50 ≥ 4 |
| `natural_products.csv` | ~5k | Natural product-derived compounds |
| `drug_mechanisms.csv` | 7,561 | Mechanism of action + target for all drugs |
| `drug_indications.csv` | 60,055 | Disease indications for all drugs |
| `atc_classification.csv` | 5,579 | Full ATC therapeutic class hierarchy |

### PubChem (`data/corpus/pubchem/`, ~6.6 MB)

| File | Records | Contents |
|---|---|---|
| `fda_drugs_with_pubchem_cids.csv` | 3,257 | ChEMBL approved drugs cross-referenced to PubChem CIDs |
| `tox21_panel.csv` | ~90k | 9 Tox21 nuclear receptor / stress-pathway assays (AhR, AR, ERα, NFkB, Nrf2, MMP, p53, ATAD5, TR) |
| `pkis2_kinase_panel.csv` | ~50k | Published Kinase Inhibitor Set 2 — 367-kinase selectivity panel |
| `drug_repurposing_hub.csv` | ~6k | Broad Institute Drug Repurposing Hub clinical annotations |
| `adme_cyp_hts.csv` | 19,803 | NCATS Caco-2 permeability, P-gp efflux, CYP2C9 inhibition HTS |
| `bindingdb_affinities.csv` | — | BindingDB Ki/IC50/Kd (literature-curated) |
| `nci60_gi50.csv` | — | NCI-60 cancer cell line growth inhibition |

### PDB (`data/corpus/pdb/`, ~111 MB)

| File | Records | Contents |
|---|---|---|
| `pdb_all_entries.csv` | 255,239 | All PDB structures: ID, method, resolution, compound, organism, deposit date |
| `pdb_ligands_all.csv` | 16,171 | All unique PDB ligands with SMILES, InChI, InChIKey, formula, MW, type, ChEMBL/DrugBank/PubChem cross-references |
| `pdb_ligands_druglike.csv` | 382 | Drug-like CCD subset (MW 100–800 Da, non-polymer, with SMILES) |
| `pdb_ligand_structure_pairs.csv` | 121,541 | Protein-ligand pairs: which ligands are bound in which structures (with resolution + method) |
| `pdb_druglike_structure_pairs.csv` | 1,671 | Drug-like protein-ligand pairs only |
| `pdb_binding_affinities.csv` | 46,885 | Measured Kd/Ki/IC50 from PDBbind-annotated structures (9,209 entries) |
| `sifts_pdb_uniprot.csv` | 965,937 | SIFTS: PDB chain → UniProt accession (canonical target identity bridge) |
| `sifts_pdb_pfam.csv` | 1,022,861 | SIFTS: PDB chain → Pfam domain family (fold-level annotation) |

## Recent changes

### Corpus fast-path (grounded lookup without LLM)

All 17 corpus CSVs are now searchable directly from the chat CLI via a keyword fast-path that bypasses the LLM entirely. Enumeration queries ("list all EGFR structures", "find erlotinib", "show imatinib") are routed to a registry-based lookup (`rag/corpus_lookup.py`) that word-boundary searches each file and returns verified data — zero hallucination risk on factual retrieval.

Key behaviours:
- Up to 500 matching rows loaded per source; pageable in 50-row pages
- Interactive source menu: results from multiple files are presented as a numbered list; the user picks which table to view
- CSV export from any table to a chosen path (default: Desktop)
- PDB binding affinity cross-reference: queries against `pdb_all_entries.csv` to resolve target names to PDB IDs, then joins `pdb_binding_affinities.csv` — surfaces measured Kd/Ki/IC50 for targets with no direct compound column
- Word-boundary regex throughout (`\bEGFR\b`) prevents substring false-positives (VEGFR matching EGFR)

### QLoRA retraining

The model was retrained from scratch after the initial run collapsed at step 50 (val loss 45.4). Root cause: RSLoRA scales updates by `scale/√rank`, not `scale/rank`. At rank 32, scale 64, the effective multiplier is ≈11.3×, making the expert-recommended `lr=1e-4` equivalent to ~37× the previously stable learning rate.

Final config: `lr=2e-5`, `rank=32`, `scale=64`, `use_rslora=true`, cosine decay with 50-step warmup, 750 iterations. Best checkpoint: step 675, val loss **0.389**. Fused model saved to `models/chem_sage_fused/` (4.0 GB).

### Chat CLI (`scripts/chat.py`)

- **RAG hybrid loop**: top-5 corpus chunks retrieved per query via ChromaDB (BAAI/bge-base-en-v1.5 embeddings, 55k chunks), injected into the system prompt as grounded context
- **RDKit auto-execution**: any `python` code block in the model response is executed live; computed values printed below the response
- **Identity grounding**: system prompt attributes ChemSage to Marc Deller and instructs the model not to claim another origin
- **Hallucination guard**: system prompt prohibits inventing PDB IDs, SMILES, ChEMBL IDs, or measured values not present in the retrieved context

### UI and terminal output

- ASCII art banner (`MARCDELLER.COM`, `small_slant` font) in bold cyan on startup
- Coloured `bouncingBar` progress spinner (with elapsed timer) for initialisation, model load, and retriever load — all third-party library output suppressed during loading (`stdout`/`stderr` redirect + `logging.disable`)
- Rich Markdown rendering for LLM responses: syntax-highlighted code blocks (`monokai`), formatted tables, bold/italic
- Tabulate `rounded_outline` tables with left-aligned columns for all corpus results
- SMILES and InChI columns truncated to 48 characters to prevent terminal overflow

## Stack

Everything runs locally on **Apple Silicon** via **MLX-LM** (`mlx_lm.lora` to train, `mlx_lm.fuse`
to merge, `mlx_lm.server` for an OpenAI-compatible endpoint the GUI talks to). Unified memory means
no VRAM wall: 16 GB handles ~8B, 32 GB ~14B, 64 GB ~32B. QLoRA is automatic because the base is a
4-bit MLX model. No GGUF or Ollama is required (a GGUF/Ollama route is documented as optional in
`scripts/merge_export.py`).

## Conventions

British English. No em dashes (colons or parentheses). Brand palette navy `#1C244B`,
accent `#467FF7`. Chemical correctness routes through RDKit, never through model confidence.

---

*Built by **Marc C. Deller, D.Phil.** · [marcdeller.com](https://marcdeller.com) · marc@marcdeller.com*
