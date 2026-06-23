#!/usr/bin/env python3
"""
chat.py — interactive ChemSage CLI with hybrid RAG + fine-tuned model.

Each turn:
  1. Retrieves the top-N most relevant corpus chunks for the question.
  2. Injects them into the system prompt as grounded context.
  3. Generates a response with the fused Qwen2.5-7B model.
  4. Auto-executes any RDKit code blocks and prints the computed values.

Usage:
    python scripts/chat.py
    python scripts/chat.py --model models/chem_sage_fused
    python scripts/chat.py --no-rag              # model-only, no retrieval
    python scripts/chat.py --n-chunks 8          # retrieve more context
    python scripts/chat.py --max-tokens 1024     # longer responses
"""

import argparse
import contextlib
import os
import re
import sys
from pathlib import Path

# Allow imports from project root regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from tabulate import tabulate

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.theme import Theme

_MAX_LOAD   = 500   # must match rag/corpus_lookup._MAX_LOAD
_PAGE_SIZE  = 50
_TRUNC_COLS = {
    "smiles", "inchi", "inchikey",
    "comp_name", "name", "header", "compound",   # long chemical / entry names
    "assay_description", "mechanism_of_action",   # free-text annotation columns
    "who_name", "efo_term", "mesh_heading",
}
_TRUNC_LEN  = 48

_THEME = Theme({
    "chemsage.header":  "bold cyan",
    "chemsage.section": "bold yellow",
    "chemsage.label":   "dim",
    "chemsage.ok":      "bold green",
    "chemsage.warn":    "bold yellow",
})
_console = Console(theme=_THEME, highlight=False)


def _progress() -> Progress:
    """Coloured indeterminate spinner for loading steps."""
    return Progress(
        SpinnerColumn(spinner_name="bouncingBar", style="bold cyan"),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        TimeElapsedColumn(),
        console=_console,
        transient=True,
    )


# ---------------------------------------------------------------------------
# Silence third-party chatter during loading
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _quiet():
    """Suppress stdout, stderr, and Python logging ≤ WARNING for the block.

    The transformers LOAD REPORT uses Python logging (not stdout/stderr), so
    logging.disable() is required alongside the fd redirect.
    """
    import logging
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull, devnull
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)
        sys.stdout, sys.stderr = old_out, old_err
        devnull.close()


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    try:
        import pyfiglet
        art = pyfiglet.figlet_format("ChemSage", font="small_slant")
    except Exception:
        art = "  MARCDELLER.COM"

    _console.print()
    for line in art.splitlines():
        _console.print(f"[chemsage.header]{escape(line)}[/chemsage.header]")

    _console.print()
    _console.print(
        "  [chemsage.label]ChemSage[/chemsage.label]"
        "  [dim]·[/dim]"
        "  [dim]Drug Discovery AI[/dim]"
        "  [dim]·[/dim]"
        "  [dim]Qwen2.5-7B · QLoRA · RAG[/dim]"
    )
    _console.print(
        "  [dim]Designed by Marc C. Deller  ·  marc@marcdeller.com  ·  "
        "[link=https://marcdeller.com]https://marcdeller.com[/link][/dim]"
    )
    _console.print()
    _console.rule(style="dim")
    _console.print()


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_WITH_RAG = """\
You are ChemSage, a chemistry-specialised AI assistant for drug discovery running locally on Apple Silicon. \
You were built by Marc Deller using a fine-tuned Qwen2.5-7B model with QLoRA training on curated chemistry \
and structural-biology data. If asked who created you or who you are, say this — do not attribute yourself \
to any other organisation.

You have access to a curated knowledge base. Relevant excerpts have been retrieved and are shown below \
in the "Retrieved context" section. Use them as your primary source of truth for factual claims.

Knowledge base contents:
- 255,239 PDB structures (method, resolution, compound, organism)
- 46,885 measured binding affinities (Kd, Ki, IC50, EC50) from PDBbind/RCSB
- 112,357 ChEMBL bioactivity measurements across 12 primary targets
- ~200,000 extended bioactivity records across 36 additional targets
- 3,257 approved drugs with physicochemical properties and QED scores
- 7,561 drug mechanism-of-action annotations
- 16,171 CCD ligand structures with SMILES, InChI, cross-references
- SIFTS UniProt and Pfam mappings for all PDB chains

Rules:
- For exact physicochemical properties (MW, logP, HBD, HBA, TPSA, QED), always compute with RDKit — \
emit a ```python``` code block using the input SMILES; do not recall numbers from memory.
- NEVER invent PDB IDs, SMILES strings, compound names, ChEMBL IDs, measured values (IC50, Ki, Kd), \
or any other specific identifiers. Only report values that appear verbatim in the retrieved context above.
- When the retrieved context contains the answer, quote the source file and the exact values shown.
- When the context does not contain enough information, say so explicitly — do not fill gaps from memory.\
"""

_SYSTEM_NO_RAG = """\
You are ChemSage, a chemistry-specialised AI assistant for drug discovery, built by Marc Deller \
using a fine-tuned Qwen2.5-7B model with QLoRA training on curated chemistry and structural-biology data. \
If asked who created you, say this — do not attribute yourself to any other organisation.
For exact physicochemical properties always compute with RDKit — emit a ```python``` code block \
using the input SMILES. When uncertain about a specific measured value, say so.\
"""


def _build_system(context: str | None) -> str:
    if not context:
        return _SYSTEM_NO_RAG
    return _SYSTEM_WITH_RAG + f"\n\n=== Retrieved context ===\n\n{context}\n\n=== End context ==="


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

_DESKTOP          = Path.home() / "Desktop"
_CTX_SMILES_LEN   = 40   # truncate SMILES when injecting into LLM context (save tokens)
_CTX_ROWS_PER_SRC = 8    # max rows per corpus source injected as LLM context


def _format_corpus_context(results: list) -> str:
    """Convert corpus lookup results into a concise, LLM-readable context block."""
    parts = [f"Grounded corpus data (keyword: '{results[0].keyword}'):"]
    for r in results:
        parts.append(f"\n[{r.title}] — {r.total} total matches")
        for _, row in r.df.head(_CTX_ROWS_PER_SRC).iterrows():
            items = []
            for k, v in row.items():
                v = str(v).strip()
                if not v:
                    continue
                if k.lower() == "smiles" and len(v) > _CTX_SMILES_LEN:
                    v = v[:_CTX_SMILES_LEN] + "…"
                items.append(f"{k}: {v}")
            if items:
                parts.append("  • " + "  |  ".join(items))
        if r.total > _CTX_ROWS_PER_SRC:
            parts.append(f"  (… {r.total - _CTX_ROWS_PER_SRC} more rows not shown)")
    return "\n".join(parts)


def _offer_corpus_tables(results: list) -> None:
    """After the LLM response, offer the user an optional deep-dive into the raw tables."""
    _console.print()
    summary = ", ".join(f"{r.total} {r.title.lower()}" for r in results)
    _console.print(
        f"  [dim]Full corpus data available ({summary})"
        f" — [v] view tables   [Enter] continue[/dim]"
    )
    try:
        choice = input("  → ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        _console.print()
        return
    if choice == "v":
        _print_lookup(results)


def _export_result(result, keyword: str) -> None:
    """Prompt for a save location (default: Desktop) and write the CSV."""
    safe     = re.sub(r"[^\w\-]", "_", f"{keyword}_{result.title}")
    default  = _DESKTOP / f"{safe}.csv"
    try:
        raw = input(f"  Save to [{default}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        _console.print("\n  [dim]Export cancelled.[/dim]")
        return
    fpath = Path(raw).expanduser() if raw else default
    fpath.parent.mkdir(parents=True, exist_ok=True)
    result.df.to_csv(fpath, index=False)
    _console.print(
        f"  [chemsage.ok]✓[/chemsage.ok]  {len(result.df)} rows → [dim]{fpath}[/dim]"
    )


def _fmt_table(df: "pd.DataFrame") -> str:
    display = df.copy()
    for col in display.columns:
        if col.lower() in _TRUNC_COLS:
            display[col] = display[col].apply(
                lambda v: str(v)[:_TRUNC_LEN] + "…" if len(str(v)) > _TRUNC_LEN else str(v)
            )
    return tabulate(
        display.to_dict("records"),
        headers="keys",
        tablefmt="rounded_outline",
        numalign="left",
        stralign="left",
        floatfmt=".3g",
        maxcolwidths=52,
    )


def _view_result(result, keyword: str) -> None:
    """Page through a single LookupResult, offering export at each step."""
    plural = "es" if result.total != 1 else ""
    cap_note = f"  ({_MAX_LOAD} loaded)" if result.total > _MAX_LOAD else ""
    _console.print()
    _console.print(
        f"[chemsage.section]── {result.title}"
        f"  ({result.total} match{plural}{cap_note}) ──[/chemsage.section]"
    )

    df     = result.df
    offset = 0

    while True:
        page = df.iloc[offset:offset + _PAGE_SIZE]
        _console.print(escape(_fmt_table(page)))
        offset += _PAGE_SIZE

        has_more_pages  = offset < len(df)
        has_capped_rows = result.total > len(df)

        if not has_more_pages:
            if has_capped_rows:
                _console.print(
                    f"\n  [dim]{result.total - len(df)} rows beyond the"
                    f" {_MAX_LOAD}-row load cap.[/dim]"
                )
                try:
                    choice = input("  [e] export CSV   [Enter] back → ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    _console.print()
                    break
                if choice == "e":
                    _export_result(result, keyword)
            break

        remaining = len(df) - offset
        try:
            choice = input(
                f"\n  {remaining} more rows"
                f"  ·  [Enter] next page   [e] export CSV   [b] back → "
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            _console.print()
            break

        if choice == "e":
            _export_result(result, keyword)
            break
        elif choice == "b":
            break
        # Enter → next page


def _print_lookup(results: list) -> None:
    """Show a source menu, let the user pick a table, page through it, repeat."""
    keyword = results[0].keyword

    while True:
        _console.print()
        _console.print(
            f"[chemsage.header]Corpus search — '{keyword}'"
            f"  ({len(results)} source{'s' if len(results) != 1 else ''} matched)[/chemsage.header]"
        )
        _console.print()

        for i, r in enumerate(results, 1):
            cap = f"  [dim](load cap — {_MAX_LOAD} shown)[/dim]" if r.total > _MAX_LOAD else ""
            _console.print(
                f"  [chemsage.section]{i}[/chemsage.section]"
                f"  {r.title}"
                f"  [dim]({r.total:,} match{'es' if r.total != 1 else ''})[/dim]"
                f"{cap}"
            )

        _console.print()
        try:
            raw = input(f"  Choose [1-{len(results)}] or [q] quit → ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            _console.print()
            break

        if raw == "q" or raw == "":
            break

        if raw.isdigit() and 1 <= int(raw) <= len(results):
            _view_result(results[int(raw) - 1], keyword)
        else:
            _console.print(f"  [dim]Enter a number between 1 and {len(results)}, or q to quit.[/dim]")


def _print_response(text: str) -> None:
    """Render LLM response as rich Markdown."""
    _console.print()
    _console.print("[chemsage.header]ChemSage:[/chemsage.header]")
    _console.print(Markdown(text, code_theme="monokai"))


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def chat_loop(
    model_path:   str,
    use_rag:      bool,
    n_chunks:     int,
    chroma_store: str,
    max_tokens:   int,
) -> None:
    # ── Initialise ───────────────────────────────────────────────────────────
    with _progress() as bar:
        bar.add_task("Initialising…", total=None)
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        from rag.corpus_lookup import lookup as corpus_lookup
        from rag.retrieve import Retriever, format_context
        from rag.tool_exec import execute

    _print_banner()

    # ── Model ────────────────────────────────────────────────────────────────
    with _progress() as bar:
        bar.add_task("Loading model…", total=None)
        with _quiet():
            model, tokenizer = load(model_path)
    _console.print(f"  [chemsage.ok]✓[/chemsage.ok]  [chemsage.label]Model[/chemsage.label]   {escape(Path(model_path).name)}")
    _model_gb = sum(p.stat().st_size for p in Path(model_path).rglob("*") if p.is_file()) / 1e9
    _console.print(
        f"  [chemsage.ok]✓[/chemsage.ok]  [chemsage.label]Disk    [/chemsage.label]  "
        f"{_model_gb:.1f} GB  ·  7B parameters  ·  4-bit quantised"
    )

    # ── Retriever ────────────────────────────────────────────────────────────
    retriever: Retriever | None = None
    chunk_count = 0
    if use_rag:
        with _progress() as bar:
            bar.add_task("Loading retriever…", total=None)
            with _quiet():
                try:
                    retriever = Retriever(store=chroma_store)
                    chunk_count = retriever._collection.count()
                except Exception:
                    pass
        if retriever:
            _console.print(f"  [chemsage.ok]✓[/chemsage.ok]  [chemsage.label]Corpus  [/chemsage.label]  {chunk_count:,} indexed chunks")
            _corpus_dir = Path("data/corpus")
            if _corpus_dir.exists():
                _csv_n   = len(list(_corpus_dir.rglob("*.csv")))
                _md_n    = len(list(_corpus_dir.rglob("*.md")))
                _corp_mb = sum(p.stat().st_size for p in _corpus_dir.rglob("*") if p.is_file()) / 1e6
                _console.print(
                    f"  [chemsage.ok]✓[/chemsage.ok]  [chemsage.label]Sources [/chemsage.label]  "
                    f"{_csv_n} CSV files  ·  {_md_n} reference docs  ·  {_corp_mb:.0f} MB"
                )
        else:
            _console.print(f"  [chemsage.warn]⚠[/chemsage.warn]  [chemsage.label]Retriever unavailable — model only[/chemsage.label]")

    # ── Training config stats ────────────────────────────────────────────────
    try:
        import yaml as _yaml
        _cfg_path = Path(__file__).parent.parent / "config" / "train_config.yaml"
        _cfg      = _yaml.safe_load(_cfg_path.read_text())
        _iters    = _cfg.get("iters", 750)
        _rank     = _cfg.get("lora_parameters", {}).get("rank", 32)
        _lr       = _cfg.get("learning_rate", 2e-5)
        _ctx      = _cfg.get("max_seq_length", 2048)
        _ltype    = "RSLoRA" if _cfg.get("use_rslora", False) else "LoRA"
    except Exception:
        _iters, _rank, _lr, _ctx, _ltype = 750, 32, 2e-5, 2048, "RSLoRA"
    _lr_str = f"{_lr:.0e}" if isinstance(_lr, float) else str(_lr)
    _console.print(
        f"  [chemsage.ok]✓[/chemsage.ok]  [chemsage.label]Training[/chemsage.label]  "
        f"{_iters:,} iters  ·  {_ltype} rank {_rank}  ·  lr {_lr_str}  ·  ctx {_ctx:,}"
    )

    # ── Ready ─────────────────────────────────────────────────────────────────
    _console.print()
    _console.rule(style="dim")
    _console.print()
    _console.print("  [dim]Type your question and press Enter.  'quit' to exit.[/dim]")
    _console.print()

    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            _console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            _console.print("[dim]Goodbye.[/dim]")
            break

        # Easter egg
        if re.search(r"\badam\b.*\basshat\b", user_input, re.IGNORECASE):
            _print_response("Yes! Absolutely! A very big one.")
            history.append({"role": "user",      "content": user_input})
            history.append({"role": "assistant", "content": "Yes! Absolutely! A very big one."})
            continue

        # ── Corpus fast-path: bulk enumeration → table, never LLM ───────────────
        # LLM cannot enumerate hundreds of SMILES/IDs accurately; grounded tables
        # are the correct output.  Analysis and explanation queries reach the LLM.
        fast = corpus_lookup(user_input)
        if fast:
            _print_lookup(fast)
            summary = "Corpus lookup: " + ", ".join(
                f"{r.total} {r.title.lower()}" for r in fast
            ) + f" for '{fast[0].keyword}'."
            history.append({"role": "user",      "content": user_input})
            history.append({"role": "assistant", "content": summary})
            continue

        # ── RAG retrieval + LLM for everything else ───────────────────────────
        context: str | None = None
        if retriever:
            try:
                chunks  = retriever.retrieve(user_input, n=n_chunks)
                context = format_context(chunks)
            except Exception as exc:
                _console.print(f"[chemsage.warn][retrieval error: {escape(str(exc))}][/chemsage.warn]")

        messages = [{"role": "system", "content": _build_system(context)}]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        with _console.status("[bold cyan]  Thinking…[/bold cyan]",
                             spinner="dots", spinner_style="bold cyan"):
            response = generate(
                model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                sampler=make_sampler(temp=0.15),
                logits_processors=make_logits_processors(repetition_penalty=1.15),
                verbose=False,
            )
        _print_response(response)

        # Execute RDKit code blocks
        exec_result = execute(response)
        if not exec_result.startswith("[no executable"):
            _console.print()
            _console.print("[chemsage.section][ RDKit ][/chemsage.section]")
            _console.print(Markdown(f"```\n{exec_result}\n```", code_theme="monokai"))

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": response})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="ChemSage — RAG-augmented chemistry assistant")
    ap.add_argument("--model",      default="models/chem_sage_fused",
                    help="Path to fused model directory (default: models/chem_sage_fused)")
    ap.add_argument("--chroma",     default=".chroma",
                    help="ChromaDB store path (default: .chroma)")
    ap.add_argument("--no-rag",     action="store_true",
                    help="Disable RAG retrieval — use fine-tuned model only")
    ap.add_argument("--n-chunks",   type=int, default=5,
                    help="Number of corpus chunks to retrieve per query (default: 5)")
    ap.add_argument("--max-tokens", type=int, default=600,
                    help="Maximum tokens to generate per response (default: 600)")
    args = ap.parse_args()

    chat_loop(
        model_path=args.model,
        use_rag=not args.no_rag,
        n_chunks=args.n_chunks,
        chroma_store=args.chroma,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
