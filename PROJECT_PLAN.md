# ChemSage: a hybrid, chemically aware LLM for drug discovery

**Mission:** build a locally hosted, open-source assistant that reasons fluently about small
molecules (SMILES, scaffolds, properties), drives the cheminformatics and structural tools a
medicinal chemist actually uses (RDKit, PyMOL, docking, PLIP), and grounds every factual claim
in the user's own corpus (papers, assay tables, target dossiers, SAR series).

**Author:** Marc C. Deller, D.Phil. ([marcdeller.com](https://marcdeller.com))
**Status:** planning / scaffold (v1)
**Fine-tune stack:** MLX-LM on Apple Silicon (committed; see section 3).
**Hand-off:** this document is the build brief for Claude Code. Phases are ordered so each one
is independently testable and delivers value before the next begins.

---

## 1. Design thesis (read this before writing any code)

Two failure modes kill most "custom chemistry LLM" projects. We design around both.

**(a) The model invents chemistry.** A 7B parameter model cannot reliably compute a logP, count
H-bond donors, or canonicalise a SMILES string in its head, and it will confidently produce wrong
numbers if asked to. The fix is not "fine-tune harder": it is to make the model **emit tool calls**
(RDKit, PyMOL, a docking wrapper) and reason over the returned values, rather than hallucinating
them. Chemical correctness is delegated to deterministic tools that already exist and are trusted.

**(b) People conflate behaviour with facts.** The current consensus (2026) is: put **volatile
knowledge in retrieval** and **stable behaviour in the weights**. So:

| Concern | Where it lives | Mechanism |
|---|---|---|
| Your SAR tables, assay readouts, target dossiers, internal papers | external corpus | **RAG** |
| Public literature you cite | external corpus | **RAG** |
| House output format, correct SMILES handling conventions, *when and how to call RDKit/PyMOL*, medchem reasoning style | model weights | **QLoRA fine-tune** |
| Deterministic property/geometry facts (logP, HBD/HBA, TPSA, RMSD, contacts) | neither: computed live | **tool calls the model emits** |

ChemSage is therefore a **hybrid**: a QLoRA-tuned base model (tuned for tool-emitting, chemically
literate *behaviour*) served behind a RAG layer (for *facts*), with a small tool-execution shim that
runs the RDKit/PyMOL code the model produces and feeds results back. "Chemically aware" = tuned to
reach for the right tool and to respect chemical validity, not tuned to memorise chemistry.

---

## 2. Architecture

```
                         ┌──────────────────────────────────────────┐
   user query  ──────▶   │   GUI front-end (AnythingLLM / Open WebUI) │
                         └───────────────┬──────────────────────────┘
                                         │
                         ┌───────────────▼──────────────┐
                         │   RAG retriever (vector store) │  ◀── ingested corpus
                         │   Chroma / LanceDB + embeddings │      (papers, SAR, dossiers)
                         └───────────────┬──────────────┘
                                         │  query + retrieved context
                         ┌───────────────▼──────────────┐
                         │   mlx_lm.server  →  fused model │  (QLoRA-tuned, MLX, OpenAI-compatible)
                         └───────────────┬──────────────┘
                                         │  may emit a tool call (RDKit / PyMOL / dock)
                         ┌───────────────▼──────────────┐
                         │   tool-exec shim (sandboxed)   │  RDKit, PyMOL, docking wrapper
                         └───────────────┬──────────────┘
                                         │  validated results (SMILES canonicalised, props, RMSD)
                                         ▼
                                   grounded answer
```

The tool-exec shim is the chemistry-aware heart of the system: it guarantees that any SMILES the
model touches is parsed and canonicalised by RDKit, and that any computed property is real. The
fine-tuned model is served by `mlx_lm.server`, which exposes an OpenAI-compatible endpoint the GUI
talks to directly.

---

## 3. Hardware and stack (committed)

Everything runs locally on Apple Silicon. Fine-tuning uses **MLX-LM** (`mlx_lm.lora`), Apple's
framework built from scratch for the unified-memory architecture: no discrete GPU, no VRAM wall,
zero CPU/GPU memory copies. The whole pipeline (RAG, training, serving) stays on the Mac.

Memory guide for QLoRA (4-bit base + LoRA adapter), unified memory:

| Mac unified memory | Comfortable model size | Rough fine-tune time |
|---|---|---|
| 16 GB | ~8B | about an hour |
| 32 GB | ~14B | a few hours |
| 64 GB | ~32B | longer, but feasible |

The tradeoff versus NVIDIA is wall-clock speed, not capability: unified memory lets a Mac fine-tune
models that would not fit a 24 GB discrete card. Supported architectures in MLX-LM include Llama,
Mistral, Qwen2, Phi, Gemma, Mixtral. MLX requires HuggingFace-format (safetensors) weights, not
GGUF.

**Record here:** Mac memory ____ GB, base model ____ (see Phase 1).

---

## 4. Phase-by-phase build plan

### Phase 0 — Environment (0.5 day)

1. Create a venv/conda env (Python 3.11+) and install from `requirements.txt`
   (`mlx`, `mlx-lm`, `rdkit`, plus the RAG stack).
2. Confirm MLX-LM works: `mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-4bit --prompt "hello"`.
   (You should be on mlx-lm 0.30+.)
3. Install **RDKit** (`pip install rdkit`) and confirm a SMILES round-trip in Python.
4. Install the GUI: **AnythingLLM** or **Open WebUI** (Docker is simplest). It will point at an
   OpenAI-compatible endpoint later (Phase 6).

**Exit test:** `mlx_lm.generate` returns text, and RDKit canonicalises `OC(=O)c1ccccc1C(=O)O`
without error.

### Phase 1 — Base model selection (0.5 day)

Choose a permissively licensed base in the 7B to 8B tier, in MLX 4-bit form (so training is QLoRA
automatically):

| Model | Licence | Notes |
|---|---|---|
| **mlx-community/Qwen2.5-7B-Instruct-4bit** (recommended) | Apache 2.0 | Strong code/tool-use behaviour, no licence friction |
| mlx-community/Mistral-7B-Instruct-v0.3-4bit | Apache 2.0 | Solid generalist; also the cleanest GGUF export if you ever want Route B |
| Llama 3.1 8B (4bit) | Meta (gated) | Accept terms on Hugging Face first |

Tool-use and code fluency matter more than raw size here, because the model's job is to emit
correct RDKit/PyMOL code, not to be an encyclopaedia. Qwen 2.5 7B Instruct (4-bit) is the default.
If a model is not already in MLX 4-bit form, quantise it: `mlx_lm.convert --hf-path <model> -q`.

Serving caveat to note now (it shapes Phase 5): MLX direct GGUF export covers only Llama/Mistral/
Mixtral. With a Qwen base we serve the fused MLX model directly via `mlx_lm.server` and never touch
GGUF, which is the cleaner path anyway.

**Exit test:** base model selected and pullable; recorded in section 3.

### Phase 2 — RAG pipeline first (1 to 2 days)

Build retrieval before any training. It delivers most of the value immediately and is reversible.

1. **Corpus staging** (`data/corpus/`): drop PDFs of papers, exported SAR tables (CSV),
   target dossiers, internal notes.
2. **Ingest** (`scripts/ingest_rag.py`): chunk, embed, write to the vector store.
   - Embeddings: a local sentence-transformer (e.g. `BAAI/bge-base-en-v1.5`).
   - Store: **Chroma** or **LanceDB** (both are what AnythingLLM/Open WebUI use under the hood, so
     you can let the GUI own this, or run your own pipeline for control).
   - Chemistry note: when chunking SAR tables, keep each compound row intact and prepend the
     assay/target header to every chunk so retrieval stays coherent.
3. **Wire to GUI:** create an AnythingLLM/Open WebUI workspace, attach the corpus, and for now set
   the chat model to a stock base served by `mlx_lm.server` (we swap in the tuned model in Phase 6).

**Exit test:** ask a question whose answer is only in your corpus and get a grounded answer with
the source chunk shown. This is a usable product on its own.

### Phase 3 — Chemistry-aware fine-tuning dataset (the real work, 3 to 5 days)

The dataset is where the project succeeds or fails. Aim for **a few hundred to a few thousand
high-quality instruction pairs** (200 to 500 is enough to start in MLX), not a giant pile of raw
text (continual pretraining on raw text gives only marginal gains; supervised instruction tuning is
what moves the needle).

Target these behaviour classes (`data/sft/` as JSONL, schema in `data/README.md`):

1. **SMILES literacy:** interpret, canonicalise, identify functional groups, spot invalid strings.
   Every assistant answer that produces or transforms a SMILES does so by *emitting RDKit code*.
2. **Property reasoning via tools:** "Is this drug-like?" → emit RDKit computing MW, logP, HBD,
   HBA, TPSA, rotatable bonds, then reason over the printed values against Lipinski/Veber.
3. **Tool invocation, RDKit:** substructure search, Murcko scaffold, Tanimoto similarity, PAINS
   flags, bioisostere suggestions, reaction SMARTS.
4. **Tool invocation, PyMOL:** generate correct PyMOL command scripts (load, select, show, colour
   by B-factor, measure distances, ray-trace a figure).
5. **Structural / docking interpretation:** read PLIP or docking output and explain the
   interaction fingerprint in medchem terms.
6. **Medchem reasoning:** SAR rationalisation, matched molecular pairs, metabolic soft-spot calls,
   selectivity narratives, framed in *your* house style.

**Construction strategy (let Claude Code help here):**
- Seed from real, validated examples (your own pipelines: `cif_to_plip.py`, chatRDKit, Boltz-2
  workflows are gold sources of correct tool-call patterns).
- Programmatically generate SMILES-task pairs by *running RDKit first to get the ground truth*,
  then templating the Q and the worked-solution A around it. This guarantees label correctness and
  scales cheaply. (Auto-generated truth is the trick: never hand-write a logP.)
- Output the MLX layout: `train.jsonl` + `valid.jsonl` in `data/sft/`, plus a frozen `test.jsonl`
  for Phase 7. Chat format (`{"messages": [...]}`), one record per line.

**Tokenisation note:** standard BPE tokenisers shred SMILES into chemically meaningless fragments.
Two mitigations, in order of effort: (i) keep SMILES inside fenced code blocks so the model treats
them as literals it passes to RDKit rather than tokens it reasons over character by character;
(ii) optionally add **SELFIES** representations alongside SMILES in the dataset (every SELFIES
string is a valid molecule, which is a useful inductive bias). Start with (i).

**Exit test:** dataset validates (every assistant code block runs; every SMILES parses); the three
splits are written; a spot-read of 20 examples reads like something you would have written.

### Phase 4 — QLoRA fine-tune with MLX-LM (0.5 to 1 day of compute)

Train only small adapter matrices; freeze the base. Because the base is a 4-bit MLX model,
`mlx_lm.lora` runs QLoRA automatically.

Run it (wrapped by `scripts/train_qlora.py`, which checks the data dir first):

```
mlx_lm.lora --config config/train_config.yaml --train
```

Key settings live in `config/train_config.yaml` (a native MLX config):

| Setting | Start value | Note |
|---|---|---|
| `model` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | 4-bit base ⇒ QLoRA automatically |
| `fine_tune_type` | `lora` | or `dora` for a small quality bump |
| `num_layers` | 16 | layers to fine-tune (-1 = all) |
| `lora_parameters.rank` | 16 | r. 8 lighter, 32 more expressive |
| `lora_parameters.scale` | 16.0 | MLX scale (NOT identical to HF alpha; example default ~20, tune) |
| `iters` | 1000 | MLX counts steps, not epochs: iters ≈ (N / batch_size) × epochs |
| `learning_rate` | 2e-4 | |
| `max_seq_length` | 2048 | |

Before the first run, reconcile field names against your installed mlx-lm version's example
(`mlx_lm/examples/lora_config.yaml`): key names have shifted across versions (e.g. `lora_layers`
vs `num_layers`).

**Critical:** watch training and validation loss together (`steps_per_eval`). Climbing validation
loss means overfitting: stop early. A small clean dataset that slightly underfits beats a large
noisy one that overfits your house quirks into nonsense.

**Exit test:** adapter written to `adapters/chem_sage_lora`, validation loss sane, and a quick
`mlx_lm.generate --adapter-path adapters/chem_sage_lora` side-by-side versus the base shows the
tuned model reaching for tools and respecting SMILES validity.

### Phase 5 — Fuse and serve (0.5 day)

**Route A (recommended, Mac-native, no GGUF):**
1. Fuse the adapter into the model:
   `mlx_lm.fuse --model <base> --adapter-path adapters/chem_sage_lora --save-path fused_model`
2. Serve an OpenAI-compatible endpoint:
   `mlx_lm.server --model fused_model --port 8080`
3. The GUI (and `eval/eval_chem.py`) talk to `http://localhost:8080/v1`. Done: no GGUF, no Ollama.

**Route B (optional, only if you specifically want Ollama):** fuse with `--de-quantize` to fp16
safetensors, then convert with llama.cpp's `convert_hf_to_gguf.py` (which supports Qwen2), then
`ollama create chem_sage -f <Modelfile>`. Direct `--export-gguf` from `mlx_lm.fuse` only works
for Llama/Mistral/Mixtral, so a Qwen base needs the llama.cpp step.

**Exit test:** `mlx_lm.server` is up and answers a chemistry prompt (via curl to `/v1/chat/
completions`) with a tool-emitting response.

### Phase 6 — Close the hybrid loop (0.5 day)

1. In the GUI workspace from Phase 2, switch the chat endpoint to the served fused model
   (`http://localhost:8080/v1`).
2. Stand up the **tool-exec shim** (`rag/tool_exec.py`): a sandboxed runner that detects RDKit/
   PyMOL code blocks in the model output, executes them, and returns results. Start with RDKit-only
   in a restricted subprocess (no filesystem/network), add PyMOL once stable.
3. Now: retrieval supplies the facts, the tuned model supplies the behaviour and tool calls, the
   shim supplies deterministic chemical truth. That is the hybrid.

**Exit test:** a single query ("here is a SMILES from my SAR table, is it more drug-like than the
series lead, and which residues would it contact in PDB X?") triggers retrieval, a tuned response,
and an executed RDKit/PLIP call, returning a grounded, numerically correct answer.

### Phase 7 — Chemistry-specific evaluation (1 day, then ongoing)

Generic LLM evals miss what matters here. Build a chemistry harness (`eval/eval_chem.py`, which
calls the OpenAI-compatible `mlx_lm.server` endpoint):

- **SMILES validity rate:** every SMILES the model emits must parse and canonicalise in RDKit.
  This is a free, automatic, brutal grader. Target ~100%.
- **Tool-call executability:** every emitted RDKit/PyMOL block must run without error.
- **Numerical fidelity:** where the model states a property, recompute with RDKit and check it
  matches (it should, if it called the tool rather than guessing).
- **Retrieval faithfulness:** answers grounded in corpus claims, not invented.
- **Side-by-side vs base** on the frozen `test.jsonl` set, scored qualitatively for medchem reasoning.

**Exit test:** harness runs on every model build and emits a one-page scorecard (an HTML dashboard
is a natural fit for your vibe-coding stack: validity %, executability %, win-rate vs base).

### Phase 8 — Iterate (ongoing)

Treat it as a design-build-test-learn loop: failures from Phase 7 become new Phase 3 training
examples; new papers/SAR get ingested into RAG continuously. Re-tune only when you see a
*behavioural* gap retrieval cannot fix. Keep facts in RAG, keep behaviour in the weights.

---

## 5. Repository structure

```
chem_sage/
├── PROJECT_PLAN.md          # this file
├── README.md                # repo overview (house standard)
├── requirements.txt         # Python deps (MLX-LM + RAG + RDKit)
├── .gitignore
├── config/
│   ├── train_config.yaml    # native MLX-LM QLoRA config
│   └── system_prompt.txt    # the ChemSage house system prompt
├── data/
│   ├── README.md            # dataset schema + worked JSONL example
│   ├── corpus/              # RAG source documents (gitignored)
│   └── sft/                 # train.jsonl + valid.jsonl + test.jsonl
├── scripts/
│   ├── ingest_rag.py        # chunk + embed + store corpus
│   ├── build_dataset.py     # generate/validate SFT pairs (RDKit ground truth)
│   ├── train_qlora.py       # MLX-LM QLoRA wrapper (mlx_lm.lora)
│   └── merge_export.py      # fuse (mlx_lm.fuse) + serve (mlx_lm.server); optional GGUF route
├── rag/
│   └── tool_exec.py         # sandboxed RDKit/PyMOL execution shim
└── eval/
    └── eval_chem.py         # chemistry-aware evaluation harness
```

---

## 6. Hand-off notes for Claude Code

Build in this order; do not skip ahead:

1. **Phase 2 RAG first.** Get `ingest_rag.py` working and a GUI workspace answering corpus
   questions on a *base* model served by `mlx_lm.server`. Ship this before touching training.
2. **`build_dataset.py` next**, with the RDKit-ground-truth generator, emitting MLX's
   train.jsonl/valid.jsonl/test.jsonl. The dataset is the project. Wire up validation (every code
   block executes, every SMILES canonicalises) from day one.
3. **`train_qlora.py`** wraps `mlx_lm.lora --config config/train_config.yaml --train`.
4. **`merge_export.py`** fuses with `mlx_lm.fuse` and serves with `mlx_lm.server` (Route A).
5. **`tool_exec.py`** sandbox (RDKit subprocess, no net/fs) before enabling any PyMOL execution.
6. **`eval_chem.py`** scorecard against the OpenAI-compatible endpoint; then iterate.

House conventions: British English (colour, behaviour, optimise, licence), no em dashes (use
colons or parentheses), peer-level comments, no marketing language. Brand palette for any UI:
navy `#1C244B`, accent `#467FF7`.

Hard rules:
- Never let the model's stated chemistry stand unverified: route through RDKit.
- Never train on unvalidated SFT examples: a bad label is worse than a missing one.
- The tool-exec shim is sandboxed (restricted subprocess) before PyMOL/docking are enabled.
- Fine-tune stack is MLX-LM only; do not reintroduce the NVIDIA/Unsloth path.

---

## 7. Key references

- MLX-LM LoRA guide: `mlx_lm.lora`, `mlx_lm.fuse`, `mlx_lm.server`
  (github.com/ml-explore/mlx-lm, see `mlx_lm/LORA.md` and `mlx_lm/examples/lora_config.yaml`).
- MLX quantisation: `mlx_lm.convert --hf-path <model> -q`.
- Ollama + llama.cpp (`convert_hf_to_gguf.py`): only for the optional Route B / GGUF path.
- AnythingLLM / Open WebUI (RAG GUI front-ends; both speak OpenAI-compatible endpoints).
- RDKit (cheminformatics), SELFIES (robust molecular string representation), PLIP (interactions).

---

*Built by Marc C. Deller, D.Phil. · [marcdeller.com](https://marcdeller.com) · marc@marcdeller.com*
