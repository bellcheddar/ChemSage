#!/usr/bin/env python3
"""
Monitor ChemSage comparative eval progress.

Usage (from chem_sage/ project root):
    .venv/bin/python scripts/eval_progress.py
"""

import json
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.live import Live
    from rich import box
except ImportError:
    sys.exit("pip install rich")

PROJECT_ROOT = Path(__file__).parent.parent
LOG_FILE     = PROJECT_ROOT / "logs" / "eval_compare_run.log"
CACHE_FILE   = PROJECT_ROOT / "eval" / "compare" / "results" / "raw_results.json"
PARTIAL_FILE = PROJECT_ROOT / "eval" / "compare" / "results" / "partial_results.json"

MODEL_ORDER = ["round1", "round2", "round3", "round4"]
MODEL_META  = {
    "round1": ("Round 1", "#94a3b8"),
    "round2": ("Round 2", "#f59e0b"),
    "round3": ("Round 3", "#10b981"),
    "round4": ("Round 4", "#467FF7"),
}
N_EXAMPLES = 100
N_MODELS   = 4
TOTAL      = N_MODELS * N_EXAMPLES
REFRESH    = 10   # seconds between refreshes

# U+2500 BOX DRAWINGS LIGHT HORIZONTAL (the ─ char used as separator)
DASH = "─"


def read_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    except Exception:
        return {}


def read_partial() -> dict:
    try:
        return json.loads(PARTIAL_FILE.read_text()) if PARTIAL_FILE.exists() else {}
    except Exception:
        return {}


def parse_log(log_text: str) -> tuple[str | None, int, str]:
    """
    Return (current_model_id | None, examples_done_in_current_model, status_hint).
    Parses the raw log (which contains \\r from tqdm) to find the active model
    and how many examples have been processed so far.
    """
    # Model section headers are printed as:
    #   \n──────...──────\n  Round X  ·  path\n──────...──────\n
    # The ─ char is U+2500.
    header_re = re.compile(
        rf"{DASH}{{20,}}\n\s+(Round \d+)\s+\xb7\s+\S+",
        re.UNICODE,
    )

    current_model = None
    last_pos      = -1
    for m in header_re.finditer(log_text):
        rname = m.group(1).lower().replace(" ", "")   # "round1" … "round4"
        current_model = rname
        last_pos      = m.end()

    if current_model is None or last_pos < 0:
        # Nothing parsed yet — server may still be starting
        if "Waiting for server" in log_text:
            return "round1", 0, "loading server…"
        return None, 0, "starting…"

    section = log_text[last_pos:]

    # Detect server state
    if "Waiting for server" in section and "ready." not in section:
        return current_model, 0, "loading server…"

    # Parse tqdm progress lines (\\r causes many overwrites in the captured file)
    tqdm_re = re.compile(r"(\d+)/100")
    ex_done = 0
    for line in reversed(section.replace("\r", "\n").split("\n")):
        mm = tqdm_re.search(line)
        if mm:
            ex_done = int(mm.group(1))
            break

    # Detect whether the current section has already finished (all 100 done)
    # If so, the model is still being saved to cache — treat as 100
    if "100/100" in section:
        ex_done = N_EXAMPLES

    return current_model, ex_done, f"{ex_done}/100 examples"


def pct(ok: int, tot: int) -> str:
    return f"{ok / tot:.0%}" if tot else "n/a"


def fmt_td(seconds: float) -> str:
    td = timedelta(seconds=int(max(0, seconds)))
    h, rem = divmod(td.seconds, 3600)
    m, s   = divmod(rem, 60)
    if td.days or h:
        return f"{td.days*24+h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def bar_str(done: int, total: int, width: int = 36, color: str = "#467FF7") -> Text:
    filled  = int(done / total * width) if total else 0
    unfill  = width - filled
    t = Text()
    t.append("█" * filled, style=f"bold {color}")
    t.append("░" * unfill, style="dim")
    return t


def build_display(
    cache:         dict,
    current_model: str | None,
    current_ex:    int,
    status_hint:   str,
    elapsed:       float,
    log_mtime:     float,
    partial:       dict | None = None,
) -> Panel:
    completed_count = sum(
        1 for mid in MODEL_ORDER
        if mid in cache and not cache[mid].get("skipped")
    )
    in_flight   = current_ex if (current_model and current_model not in cache) else 0
    total_done  = completed_count * N_EXAMPLES + in_flight

    rate     = total_done / elapsed if elapsed > 2 else 0
    eta_secs = (TOTAL - total_done) / rate if rate > 0 else 0
    eta_str  = fmt_td(eta_secs) if rate > 0 else "…"
    ela_str  = fmt_td(elapsed)

    # ── model table ──────────────────────────────────────────────────────
    # 8 metrics: 3 core + 5 R4 (marked *)
    METRIC_COLS = [
        ("smiles_validity",    "SMILES"),
        ("tool_executability", "Exec"),
        ("numerical_fidelity", "Fidelity"),
        ("rounding_precision", "Round*"),
        ("code_then_quote",    "C→Q*"),
        ("refusal_accuracy",   "Refusal*"),
        ("qed_range",          "QED*"),
        ("python_extended",    "PyExec*"),
    ]

    t = Table(
        box=box.ROUNDED,
        expand=True,
        header_style="bold #467FF7",
        show_header=True,
        padding=(0, 1),
    )
    t.add_column("Model",    style="bold", min_width=9,  no_wrap=True)
    t.add_column("Progress", min_width=20, no_wrap=True)
    for _, label in METRIC_COLS:
        t.add_column(label, justify="right", min_width=7, no_wrap=True)
    t.add_column("Time", justify="right", min_width=6, no_wrap=True)

    BLANK_METRICS = ["—"] * len(METRIC_COLS)

    for mid in MODEL_ORDER:
        name, color = MODEL_META[mid]
        res = cache.get(mid)

        if res and not res.get("skipped"):
            # ── completed ──
            agg  = res.get("aggregate", {})
            vals = [pct(*agg.get(key, (0, 0))) for key, _ in METRIC_COLS]
            mins = res.get("total_min", "?")
            t.add_row(
                Text(name, style=f"bold {color}"),
                Text("✓  done", style="bold green"),
                *vals,
                f"{mins}m",
            )

        elif mid == current_model and mid not in cache:
            # ── in progress ──
            bar_w  = 12
            filled = int(current_ex / N_EXAMPLES * bar_w)
            b = Text()
            b.append("▶ ", style=f"bold {color}")
            b.append("█" * filled, style=f"bold {color}")
            b.append("░" * (bar_w - filled), style="dim")
            b.append(f" {current_ex}/100", style=f"bold {color}")
            if status_hint and "server" in status_hint:
                b = Text(f"▶ {status_hint}", style=f"bold {color}")
            # Show live partial scores if available
            if partial and partial.get("model_id") == mid and partial.get("aggregate"):
            	live_agg = partial["aggregate"]
            	vals = [pct(*live_agg.get(key, (0, 0))) for key, _ in METRIC_COLS]
            else:
                vals = BLANK_METRICS
            t.add_row(
                Text(name, style=f"bold {color}"),
                b, *vals, "—",
            )

        elif res and res.get("skipped"):
            t.add_row(
                Text(name, style="dim"),
                Text("⚠  skipped", style="yellow"),
                *BLANK_METRICS, "—",
            )

        else:
            t.add_row(
                Text(name, style="dim"),
                Text("·  waiting", style="dim"),
                *BLANK_METRICS, "—",
            )

    # ── overall progress ─────────────────────────────────────────────────
    pct_val   = total_done / TOTAL
    overall_b = bar_str(total_done, TOTAL)
    last_upd  = time.strftime("%H:%M:%S", time.localtime(log_mtime))

    summary_line = Text()
    summary_line.append("\n  Progress  ")
    summary_line.append_text(overall_b)
    summary_line.append(f"  {total_done}/{TOTAL} ({pct_val:.0%})")
    summary_line.append(f"   elapsed ")
    summary_line.append(ela_str, style="bold")
    summary_line.append("   ETA ")
    summary_line.append(eta_str, style="bold yellow")
    summary_line.append(f"\n\n  log updated {last_upd}  ·  refresh {REFRESH}s  ·  Ctrl-C to exit", style="dim")

    return Panel(
        Group(t, summary_line),
        title="[bold]⬡  ChemSage  Eval Monitor[/bold]",
        border_style="#467FF7",
        padding=(0, 1),
    )


def main() -> None:
    console = Console()

    if not LOG_FILE.exists():
        console.print(f"[red]Log not found:[/red] {LOG_FILE}")
        console.print("Start the eval:  [bold].venv/bin/python eval/compare/eval_compare.py[/bold]")
        return

    stat       = LOG_FILE.stat()
    start_time = getattr(stat, "st_birthtime", stat.st_ctime)   # macOS birth time

    console.print(f"\n[bold #467FF7]ChemSage Eval Monitor[/bold #467FF7]  "
                  f"started {time.strftime('%H:%M:%S', time.localtime(start_time))}\n")

    with Live(console=console, refresh_per_second=0.2, auto_refresh=False) as live:
        while True:
            try:
                cache    = read_cache()
                log_text = LOG_FILE.read_bytes().decode("utf-8", errors="replace")
                mtime    = LOG_FILE.stat().st_mtime
                elapsed  = time.time() - start_time

                cur_model, cur_ex, hint = parse_log(log_text)
                partial = read_partial()
                live.update(build_display(cache, cur_model, cur_ex, hint, elapsed, mtime, partial))
                live.refresh()

                done = sum(1 for m in MODEL_ORDER if m in cache and not cache[m].get("skipped"))
                if done == N_MODELS:
                    time.sleep(0.3)
                    break

                time.sleep(REFRESH)

            except KeyboardInterrupt:
                break

    # Final summary
    console.print("\n[bold green]✅  All models evaluated![/bold green]\n")
    results_dir = PROJECT_ROOT / "eval" / "compare" / "results"
    for p in sorted(results_dir.glob("compare_*.html")):
        console.print(f"  [bold]HTML →[/bold] {p}")
    for p in sorted(results_dir.glob("compare_*.md")):
        console.print(f"  [bold]MD   →[/bold] {p}")
    console.print()


if __name__ == "__main__":
    main()
