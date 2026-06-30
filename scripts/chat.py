#!/usr/bin/env python3
"""
chat.py — interactive ChemSage CLI with hybrid RAG + fine-tuned model.

Each turn:
  1. Retrieves the top-N most relevant corpus chunks for the question.
  2. Injects them into the system prompt as grounded context.
  3. Streams the response live from the fused Qwen2.5-32B model.
  4. Auto-executes any RDKit code blocks and prints the computed values.

Usage:
    python scripts/chat.py
    python scripts/chat.py --model models/chem_sage_32b_v5
    python scripts/chat.py --no-rag              # model-only, no retrieval
    python scripts/chat.py --n-chunks 8          # retrieve more context
    python scripts/chat.py --max-tokens 1024     # longer responses

Slash commands:
    /help           show available commands
    /clear          clear screen and reprint banner
    /reset          clear conversation history (asks for confirmation)
    /history [n]    show last n conversation turns (default: all)
    /save           save conversation to a Markdown file
    /info           show model, corpus, and training configuration
    /retry          regenerate the previous response
"""

import argparse
import contextlib
import os
import re
import sys
import time
from pathlib import Path

# Allow imports from project root regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_toolkit import PromptSession
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from rich import box as rich_box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

_MAX_LOAD  = 500   # must match rag/corpus_lookup._MAX_LOAD
_PAGE_SIZE = 50

# (max_width, display header) for every known corpus column.
# Unknown columns fall back to _DEFAULT_COL_WIDTH with their original name.
_COL_CFG: dict[str, tuple[int, str]] = {
    # PDB / structural
    "pdb_id":               (7,  "PDB"),
    "comp_id":              (8,  "Ligand"),
    "comp_name":            (26, "Name"),
    "comp_mw":              (7,  "MW"),
    "comp_type":            (10, "Type"),
    "method":               (8,  "Method"),
    "resolution_a":         (6,  "Res Å"),
    "is_druglike":          (5,  "Drug?"),
    "smiles":               (22, "SMILES"),
    "canonical_smiles":     (22, "SMILES"),
    "inchi":                (22, "InChI"),
    "inchikey":             (14, "InChIKey"),
    "formula":              (12, "Formula"),
    # Binding affinities
    "affinity_type":        (6,  "Type"),
    "affinity_value":       (8,  "Value"),
    "affinity_unit":        (6,  "Unit"),
    "provenance":           (12, "Source"),
    "seq_identity":         (5,  "SeqID"),
    "symbol":               (8,  "Symbol"),
    # ChEMBL bioactivity
    "molecule_chembl_id":   (14, "ChEMBL ID"),
    "target_name":          (22, "Target"),
    "target_chembl_id":     (14, "Target ID"),
    "standard_type":        (6,  "Type"),
    "standard_value":       (8,  "Value"),
    "standard_units":       (6,  "Units"),
    "pchembl_value":        (7,  "pChEMBL"),
    "assay_chembl_id":      (14, "Assay ID"),
    "document_chembl_id":   (14, "Doc ID"),
    "activity_comment":     (18, "Comment"),
    # Approved drugs
    "name":                 (22, "Name"),
    "mw":                   (7,  "MW"),
    "alogp":                (6,  "logP"),
    "hbd":                  (4,  "HBD"),
    "hba":                  (4,  "HBA"),
    "tpsa":                 (6,  "TPSA"),
    "rotatable_bonds":      (4,  "RB"),
    "qed":                  (5,  "QED"),
    "max_phase":            (5,  "Phase"),
    "oral":                 (5,  "Oral?"),
    "black_box_warning":    (4,  "BBW"),
    # Drug mechanisms / indications
    "mechanism_of_action":  (30, "Mechanism"),
    "action_type":          (12, "Action"),
    "direct_interaction":   (6,  "Direct"),
    "disease_efficacy":     (6,  "Efficacy"),
    "chembl_id":            (14, "ChEMBL ID"),
    "assay_description":    (26, "Assay"),
    # General / annotation
    "who_name":             (22, "WHO Name"),
    "efo_term":             (22, "EFO Term"),
    "mesh_heading":         (22, "MeSH"),
    "header":               (22, "Header"),
    "compound":             (22, "Compound"),
}

_DEFAULT_COL_WIDTH = 14

_METHOD_SHORT: dict[str, str] = {
    "x-ray diffraction":        "X-RAY",
    "electron microscopy":      "CRYO-EM",
    "solution nmr":             "NMR",
    "solid-state nmr":          "ssNMR",
    "neutron diffraction":      "NEUTRON",
    "fiber diffraction":        "FIBER",
    "electron crystallography": "e-CRYST",
    "powder diffraction":       "POWDER",
}

_THEME = Theme({
    "chemsage.header":  "bold cyan",
    "chemsage.section": "bold yellow",
    "chemsage.label":   "dim",
    "chemsage.ok":      "bold green",
    "chemsage.warn":    "bold yellow",
})
_console = Console(theme=_THEME, highlight=False)

_PT_STYLE = Style.from_dict({
    "bottom-toolbar":      "bg:#0d1f2d #4db8ff",
    "bottom-toolbar.text": "bg:#0d1f2d #4db8ff",
    "prompt":              "bold #4db8ff",
    "completion-menu.completion":          "bg:#1a3a5c #c0e0ff",
    "completion-menu.completion.current":  "bg:#4db8ff #0d1f2d bold",
})

_SLASH_COMPLETER = WordCompleter(
    ["/help", "/clear", "/reset", "/history", "/save", "/info", "/retry"],
    sentence=True,
)

_DESKTOP = Path.home() / "Desktop"


# ---------------------------------------------------------------------------
# Toolbar
# ---------------------------------------------------------------------------

def _make_toolbar(history: list, model_name: str = "ChemSage", rag_active: bool = True):
    def _toolbar():
        turns = len(history) // 2
        ctx   = min(turns, 3)
        state = (
            f"  ·  turn {turns}  ·  ctx {ctx}/3"
            if turns else ""
        )
        rag_tag = " · RAG" if rag_active else ""
        return HTML(
            f'<b fg="#4db8ff"> ⬡ ChemSage </b>'
            f'<style fg="#3a6a9a"> · {escape(model_name)} · QLoRA{rag_tag}{state}</style>'
            f'<style fg="#2a4a6a">    /help · /clear · /reset · /history · /save · /info · /retry</style>'
        )
    return _toolbar


# ---------------------------------------------------------------------------
# Progress spinner
# ---------------------------------------------------------------------------

def _progress() -> Progress:
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
    """Suppress stdout, stderr, and Python logging ≤ WARNING for the block."""
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
# Banner / Goodbye
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    sys.stdout.write("\x1b]0;ChemSage\x07")
    sys.stdout.flush()

    try:
        import pyfiglet
        art = pyfiglet.figlet_format("ChemSage", font="small_slant")
    except Exception:
        art = "  CHEMSAGE"

    _console.print()
    for line in art.splitlines():
        _console.print(f"[chemsage.header]{escape(line)}[/chemsage.header]")

    _console.print()
    _console.print(
        "  [chemsage.label]ChemSage[/chemsage.label]"
        "  [dim]·[/dim]"
        "  [dim]Drug Discovery AI[/dim]"
        "  [dim]·[/dim]"
        "  [dim]Qwen2.5-32B · QLoRA · RAG[/dim]"
    )
    _console.print(
        "  [dim]Built by "
        "[link=https://marcdeller.com]Marc C. Deller[/link]"
        "  ·  "
        "[link=mailto:marc@marcdeller.com]marc@marcdeller.com[/link]"
        "[/dim]"
    )
    _console.print()
    _console.rule(style="dim")
    _console.print()


def _print_goodbye() -> None:
    try:
        import pyfiglet
        art = pyfiglet.figlet_format("ChemSage", font="small_slant")
    except Exception:
        art = "  CHEMSAGE"

    art_markup = "\n".join(
        f"[chemsage.header]{escape(line)}[/chemsage.header]"
        for line in art.splitlines()
    )
    content = (
        art_markup
        + "\n\n"
        "  [dim]Thank you for using ChemSage.[/dim]\n\n"
        "  [dim]Built by "
        "[link=https://marcdeller.com]Marc C. Deller[/link]"
        "  ·  "
        "[link=mailto:marc@marcdeller.com]marc@marcdeller.com[/link]"
        "[/dim]"
    )
    _console.print()
    _console.print(Panel(content, border_style="dim cyan", padding=(1, 2)))
    _console.print()


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_WITH_RAG = """\
You are ChemSage, a chemistry-specialised AI assistant for drug discovery running locally on Apple Silicon. \
You were built by Marc Deller using a fine-tuned Qwen2.5-32B model with QLoRA training on curated chemistry \
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
- When the context does not contain enough information, say so explicitly — do not fill gaps from memory.
- When the retrieved context contains structured data (PDB structures, bioactivity measurements, \
compounds, binding affinities), present the data directly as a formatted markdown table — do NOT \
generate code to search for the data externally. The corpus has already been queried; render what was found.\
"""

_SYSTEM_NO_RAG = """\
You are ChemSage, a chemistry-specialised AI assistant for drug discovery, built by Marc Deller \
using a fine-tuned Qwen2.5-32B model with QLoRA training on curated chemistry and structural-biology data. \
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

_CTX_SMILES_LEN   = 40
_CTX_ROWS_PER_SRC = 8


def _format_corpus_context(results: list) -> str:
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
    summary = "  " + "\n  ".join(
        f"[chemsage.ok]⬡[/chemsage.ok]  {r.total:,} {r.title.lower()}"
        for r in results
    )
    _console.print()
    _console.print(Panel(
        summary + "\n\n  [b][v][/b] view tables   [b][Enter][/b] continue",
        title="[chemsage.header]Corpus Data Available[/chemsage.header]",
        border_style="dim cyan",
        padding=(0, 1),
    ))
    try:
        choice = pt_prompt("  → ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        _console.print()
        return
    if choice == "v":
        _print_lookup(results)


def _export_result(result, keyword: str) -> None:
    safe    = re.sub(r"[^\w\-]", "_", f"{keyword}_{result.title}")
    default = _DESKTOP / f"{safe}.csv"
    try:
        raw = pt_prompt(f"  Save to [{default}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        _console.print("\n  [dim]Export cancelled.[/dim]")
        return
    fpath = Path(raw).expanduser() if raw else default
    fpath.parent.mkdir(parents=True, exist_ok=True)
    result.df.to_csv(fpath, index=False)
    _console.print(
        f"  [chemsage.ok]✓[/chemsage.ok]  {len(result.df)} rows → [dim]{fpath}[/dim]"
    )


def _make_rich_table(df: "pd.DataFrame") -> Table:
    tbl = Table(
        box=rich_box.ROUNDED,
        show_edge=True,
        header_style="bold cyan",
        border_style="dim cyan",
        padding=(0, 1),
    )

    for col in df.columns:
        max_w, header = _COL_CFG.get(col.lower(), (_DEFAULT_COL_WIDTH, col))
        tbl.add_column(header, max_width=max_w, no_wrap=True, overflow="ellipsis")

    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            key = col.lower()
            if isinstance(val, bool):
                cell = "✓" if val else "✗"
            elif isinstance(val, float):
                cell = f"{val:.3g}"
            else:
                cell = str(val)
                if cell.lower() in ("true",  "yes", "1"): cell = "✓"
                elif cell.lower() in ("false", "no",  "0"): cell = "✗"
            if key == "method":
                cell = _METHOD_SHORT.get(cell.lower(), cell.split()[0] if cell else cell)
            cells.append(escape(cell))
        tbl.add_row(*cells)

    return tbl


def _view_result(result, keyword: str) -> None:
    plural   = "es" if result.total != 1 else ""
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
        _console.print(_make_rich_table(page))
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
                    choice = pt_prompt("  [e] export CSV   [Enter] back → ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    _console.print()
                    break
                if choice == "e":
                    _export_result(result, keyword)
            break

        remaining = len(df) - offset
        try:
            choice = pt_prompt(
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


def _print_lookup(results: list) -> None:
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
            raw = pt_prompt(f"  Choose [1-{len(results)}] or [q] quit → ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            _console.print()
            break

        if raw in ("q", ""):
            break

        if raw.isdigit() and 1 <= int(raw) <= len(results):
            _view_result(results[int(raw) - 1], keyword)
        else:
            _console.print(f"  [dim]Enter a number between 1 and {len(results)}, or q to quit.[/dim]")


def _print_response(text: str) -> None:
    _console.print()
    _console.rule("[chemsage.header] ChemSage [/chemsage.header]", style="dim cyan")
    _console.print()
    _console.print(Markdown(text, code_theme="monokai"))


def _print_stats(n_tokens: int, elapsed: float) -> None:
    tok_per_s = n_tokens / elapsed if elapsed > 0 else 0
    _console.print(
        f"\n  [dim]⚡ {n_tokens} tok · {tok_per_s:.1f} tok/s · {elapsed:.1f}s[/dim]"
    )


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

def _handle_slash_command(cmd: str, history: list, info: dict | None = None) -> None:
    parts = cmd.strip().split(None, 1)
    name  = parts[0].lower()
    arg   = parts[1].strip() if len(parts) > 1 else ""

    if name == "/help":
        _console.print(Panel(
            "  [b]/help[/b]            show this message\n"
            "  [b]/clear[/b]           clear screen and reprint banner\n"
            "  [b]/reset[/b]           clear conversation history\n"
            "  [b]/history [n][/b]     show last n turns (default: all)\n"
            "  [b]/save[/b]            save conversation to Markdown\n"
            "  [b]/info[/b]            show model, corpus, and training details\n"
            "  [b]/retry[/b]           regenerate the previous response\n\n"
            "  [b]quit / exit / q[/b]  exit ChemSage\n\n"
            "  [dim]Built by [link=https://marcdeller.com]Marc C. Deller[/link]"
            "  ·  [link=mailto:marc@marcdeller.com]marc@marcdeller.com[/link][/dim]",
            title="[chemsage.header] Commands [/chemsage.header]",
            border_style="cyan",
            padding=(1, 2),
        ))
        return

    if name == "/clear":
        os.system("clear")
        _print_banner()
        return

    if name == "/reset":
        try:
            confirm = pt_prompt("  Clear conversation history? [y/N] → ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            _console.print()
            return
        if confirm == "y":
            history.clear()
            _console.print("\n  [chemsage.ok]✓[/chemsage.ok]  [dim]Conversation history cleared.[/dim]\n")
        else:
            _console.print("\n  [dim]Cancelled.[/dim]\n")
        return

    if name == "/history":
        pairs = list(zip(history[::2], history[1::2]))
        n     = int(arg) if arg.isdigit() else len(pairs)
        shown = pairs[-n:]
        if not shown:
            _console.print("\n  [dim]No conversation history yet.[/dim]\n")
            return
        _console.print()
        offset = len(pairs) - len(shown) + 1
        for i, (u, a) in enumerate(shown, offset):
            _console.print(f"  [chemsage.label][{i}] You:[/chemsage.label]     {escape(u['content'][:120])}")
            _console.print(f"      [chemsage.label]ChemSage:[/chemsage.label] {escape(a['content'][:120])}")
            _console.print()
        return

    if name == "/save":
        _save_conversation(history)
        return

    if name == "/info":
        _print_info(info)
        return

    _console.print(
        f"\n  [chemsage.warn]Unknown command:[/chemsage.warn] [dim]{escape(cmd)}[/dim]"
        "  (type [b]/help[/b] for commands)\n"
    )


def _save_conversation(history: list) -> None:
    if not history:
        _console.print("\n  [dim]Nothing to save yet.[/dim]\n")
        return
    ts      = time.strftime("%Y%m%d_%H%M%S")
    default = _DESKTOP / f"chemsage_{ts}.md"
    try:
        raw = pt_prompt(f"  Save to [{default}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        _console.print("\n  [dim]Cancelled.[/dim]\n")
        return
    fpath = Path(raw).expanduser() if raw else default
    fpath.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# ChemSage Conversation\n",
        f"*{time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n",
        "Built by [Marc C. Deller](https://marcdeller.com) · marc@marcdeller.com\n\n",
        "---\n\n",
    ]
    for i, (u, a) in enumerate(zip(history[::2], history[1::2]), 1):
        lines.append(f"## Turn {i}\n\n")
        lines.append(f"**You:** {u['content']}\n\n")
        lines.append(f"**ChemSage:** {a['content']}\n\n")

    fpath.write_text("".join(lines))
    turns = len(history) // 2
    _console.print(
        f"\n  [chemsage.ok]✓[/chemsage.ok]  {turns} turn{'s' if turns != 1 else ''}"
        f" saved → [dim]{fpath}[/dim]\n"
    )


def _print_info(info: dict | None) -> None:
    if not info:
        _console.print("\n  [dim]Session info not available.[/dim]\n")
        return

    rag_status = (
        f"{info.get('chunk_count', 0):,} indexed chunks  ·  "
        f"{info.get('corpus_csv', '?')} CSV files  ·  "
        f"{info.get('corpus_md', '?')} docs  ·  "
        f"{info.get('corpus_mb', 0):.0f} MB"
        if info.get("rag")
        else "disabled (--no-rag)"
    )

    content = (
        f"  [chemsage.label]Model   [/chemsage.label]  {escape(info.get('model_name', '?'))}\n"
        f"  [chemsage.label]Disk    [/chemsage.label]  {info.get('model_gb', 0):.1f} GB"
        f"  ·  32B parameters  ·  4-bit quantised\n\n"
        f"  [chemsage.label]Training[/chemsage.label]  "
        f"{info.get('iters', '?'):,} iters  ·  "
        f"{info.get('ltype', 'RSLoRA')} rank {info.get('rank', 32)}"
        f"  ·  lr {info.get('lr', '?')}  ·  ctx {info.get('ctx', 2048):,}\n\n"
        f"  [chemsage.label]Corpus  [/chemsage.label]  {rag_status}\n"
        f"  [chemsage.label]Embed   [/chemsage.label]  BAAI/bge-base-en-v1.5  ·  ChromaDB\n\n"
        f"  [chemsage.label]Built by[/chemsage.label]  "
        "[link=https://marcdeller.com]Marc C. Deller[/link]"
        "  ·  "
        "[link=mailto:marc@marcdeller.com]marc@marcdeller.com[/link]"
    )
    _console.print()
    _console.print(Panel(
        content,
        title="[chemsage.header] ChemSage — Session Info [/chemsage.header]",
        border_style="cyan",
        padding=(1, 2),
    ))
    _console.print()


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
        try:
            from mlx_lm import stream_generate
        except ImportError:
            stream_generate = None
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
    model_name = Path(model_path).name
    _console.print(
        f"  [chemsage.ok]✓[/chemsage.ok]  [chemsage.label]Model   [/chemsage.label]  {escape(model_name)}"
    )
    model_gb = sum(p.stat().st_size for p in Path(model_path).rglob("*") if p.is_file()) / 1e9
    _console.print(
        f"  [chemsage.ok]✓[/chemsage.ok]  [chemsage.label]Disk    [/chemsage.label]  "
        f"{model_gb:.1f} GB  ·  32B parameters  ·  4-bit quantised"
    )

    # ── Retriever ────────────────────────────────────────────────────────────
    retriever: Retriever | None = None
    chunk_count = 0
    corpus_csv = corpus_md = 0
    corpus_mb  = 0.0
    if use_rag:
        with _progress() as bar:
            bar.add_task("Loading retriever…", total=None)
            with _quiet():
                try:
                    retriever   = Retriever(store=chroma_store)
                    chunk_count = retriever._collection.count()
                except Exception:
                    pass
        if retriever:
            _console.print(
                f"  [chemsage.ok]✓[/chemsage.ok]  [chemsage.label]Corpus  [/chemsage.label]  {chunk_count:,} indexed chunks"
            )
            _corpus_dir = Path("data/corpus")
            if _corpus_dir.exists():
                corpus_csv = len(list(_corpus_dir.rglob("*.csv")))
                corpus_md  = len(list(_corpus_dir.rglob("*.md")))
                corpus_mb  = sum(p.stat().st_size for p in _corpus_dir.rglob("*") if p.is_file()) / 1e6
                _console.print(
                    f"  [chemsage.ok]✓[/chemsage.ok]  [chemsage.label]Sources [/chemsage.label]  "
                    f"{corpus_csv} CSV files  ·  {corpus_md} reference docs  ·  {corpus_mb:.0f} MB"
                )
        else:
            _console.print(
                "  [chemsage.warn]⚠[/chemsage.warn]  "
                "[chemsage.label]Retriever unavailable — model only[/chemsage.label]"
            )

    # ── Training config ───────────────────────────────────────────────────────
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

    # ── Session info dict (passed to /info) ───────────────────────────────────
    _session_info = {
        "model_name":  model_name,
        "model_gb":    model_gb,
        "chunk_count": chunk_count,
        "corpus_csv":  corpus_csv,
        "corpus_md":   corpus_md,
        "corpus_mb":   corpus_mb,
        "rag":         retriever is not None,
        "iters":       _iters,
        "rank":        _rank,
        "lr":          _lr_str,
        "ctx":         _ctx,
        "ltype":       _ltype,
    }

    # ── Ready ─────────────────────────────────────────────────────────────────
    _console.print()
    _console.rule(style="dim")
    _console.print()
    _console.print(
        "  [dim]Type your question and press Enter.  "
        "/help for commands.  Tab to complete.  'quit' to exit.[/dim]"
    )
    _console.print()

    history: list[dict] = []
    session = PromptSession(
        history=InMemoryHistory(),
        style=_PT_STYLE,
        bottom_toolbar=_make_toolbar(history, model_name, rag_active=retriever is not None),
        completer=_SLASH_COMPLETER,
        complete_while_typing=False,
    )

    # ── Generation helper (closure over model / tokenizer / imports) ──────────
    def _do_generate(messages: list, _max_tokens: int) -> tuple[str, float]:
        """Stream-generate a response; return (full_text, elapsed_seconds)."""
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        _console.print()
        _console.rule("[chemsage.header] ChemSage [/chemsage.header]", style="dim cyan")
        _console.print()

        full_text = ""
        t0 = time.time()

        if stream_generate is not None:
            with Live(
                Text(""),
                console=_console,
                refresh_per_second=12,
                vertical_overflow="visible",
            ) as live:
                for chunk in stream_generate(
                    model, tokenizer, prompt=prompt_str,
                    max_tokens=_max_tokens,
                    sampler=make_sampler(temp=0.15),
                    logits_processors=make_logits_processors(repetition_penalty=1.15),
                ):
                    tok = chunk.text if hasattr(chunk, "text") else str(chunk)
                    if tok:
                        full_text += tok
                        live.update(Text(full_text))
                if full_text:
                    live.update(Markdown(full_text, code_theme="monokai"))
        else:
            # Fallback: blocking generate with spinner
            with _console.status(
                "[bold cyan]  Thinking…[/bold cyan]",
                spinner="dots", spinner_style="bold cyan",
            ):
                full_text = generate(
                    model, tokenizer, prompt=prompt_str,
                    max_tokens=_max_tokens,
                    sampler=make_sampler(temp=0.15),
                    logits_processors=make_logits_processors(repetition_penalty=1.15),
                )
            _console.print(Markdown(full_text, code_theme="monokai"))

        return full_text, time.time() - t0

    # State for /retry
    _last_messages:  list | None = None
    _last_context:   str  | None = None
    _last_user:      str  | None = None

    while True:
        try:
            user_input = session.prompt(HTML("<b>You</b>: ")).strip()
        except (KeyboardInterrupt, EOFError):
            _print_goodbye()
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            _print_goodbye()
            break

        # ── /retry — handled here so it can access model internals ────────────
        if user_input.lower() == "/retry":
            if _last_messages is None:
                _console.print("\n  [dim]Nothing to retry yet.[/dim]\n")
                continue
            response, elapsed = _do_generate(_last_messages, max_tokens)
            try:
                n_tokens = len(tokenizer.encode(response))
            except Exception:
                n_tokens = len(response.split())
            _print_stats(n_tokens, elapsed)
            exec_result = execute(response)
            if not exec_result.startswith("[no executable"):
                _console.print()
                _console.print(Panel(
                    Markdown(f"```\n{exec_result}\n```", code_theme="monokai"),
                    title="[chemsage.section] RDKit Output [/chemsage.section]",
                    border_style="yellow",
                    padding=(0, 1),
                ))
            # Replace last assistant turn in history
            if len(history) >= 2:
                history[-1] = {"role": "assistant", "content": response}
            continue

        if user_input.startswith("/"):
            _handle_slash_command(user_input, history, info=_session_info)
            continue

        # Easter egg
        if re.search(r"\badam\b.*\basshat\b", user_input, re.IGNORECASE):
            _print_response("Yes! Absolutely! A very big one.")
            history.append({"role": "user",      "content": user_input})
            history.append({"role": "assistant", "content": "Yes! Absolutely! A very big one."})
            continue

        # ── Corpus fast-path: bulk enumeration → table, never LLM ────────────
        fast = corpus_lookup(user_input)
        if fast:
            _print_lookup(fast)
            summary = "Corpus lookup: " + ", ".join(
                f"{r.total} {r.title.lower()}" for r in fast
            ) + f" for '{fast[0].keyword}'."
            history.append({"role": "user",      "content": user_input})
            history.append({"role": "assistant", "content": summary})
            continue

        # ── RAG retrieval ─────────────────────────────────────────────────────
        context: str | None = None
        if retriever:
            try:
                with _console.status(
                    "[dim]  🔍 Retrieving context…[/dim]",
                    spinner="dots", spinner_style="dim cyan",
                ):
                    chunks  = retriever.retrieve(user_input, n=n_chunks)
                    context = format_context(chunks)
                _console.print(
                    f"  [dim]📚 {len(chunks)} chunk{'s' if len(chunks) != 1 else ''} retrieved[/dim]"
                )
            except Exception as exc:
                _console.print(Panel(
                    escape(str(exc)),
                    title="[bold red] Retrieval Error [/bold red]",
                    border_style="red",
                    padding=(0, 1),
                ))

        messages = [{"role": "system", "content": _build_system(context)}]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": user_input})

        # Stash for /retry
        _last_messages = messages
        _last_context  = context
        _last_user     = user_input

        # ── Generate ──────────────────────────────────────────────────────────
        response, elapsed = _do_generate(messages, max_tokens)

        try:
            n_tokens = len(tokenizer.encode(response))
        except Exception:
            n_tokens = len(response.split())
        _print_stats(n_tokens, elapsed)

        # ── Execute RDKit code blocks ──────────────────────────────────────────
        exec_result = execute(response)
        if not exec_result.startswith("[no executable"):
            _console.print()
            _console.print(Panel(
                Markdown(f"```\n{exec_result}\n```", code_theme="monokai"),
                title="[chemsage.section] RDKit Output [/chemsage.section]",
                border_style="yellow",
                padding=(0, 1),
            ))

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": response})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="ChemSage — RAG-augmented chemistry assistant")
    ap.add_argument("--model",      default="models/chem_sage_32b_v5",
                    help="Path to fused model directory (default: models/chem_sage_32b_v5)")
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
