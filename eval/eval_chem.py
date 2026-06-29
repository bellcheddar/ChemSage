#!/usr/bin/env python3
"""
eval_chem.py — chemistry-aware evaluation harness (Phase 7, R4 extended).

Queries an OpenAI-compatible endpoint (mlx_lm.server) against the frozen test split.

ORIGINAL METRICS (unchanged, comparable across all rounds):
  - SMILES validity        every MolFromSmiles() call must parse in RDKit
  - Tool executability     every RDKit code block must run without error
  - Numerical fidelity     stated MW/logP/HBD/HBA/TPSA match RDKit recomputed truth

R4 EXTENSION METRICS (degrade gracefully to n/a for R1-R3 datasets):
  - Rounding precision     MW 1 d.p. · logP 2 d.p. · TPSA 1 d.p. · QED 2 d.p.
  - Code-then-quote        prose values match what the code blocks actually printed
  - Extended executability all ```python``` blocks (RDKit + Plotly + Biopython + other)
  - Refusal accuracy       out-of-scope prompts correctly declined (requires 'class' field)
  - QED range              all stated QED values lie in [0, 1]

CROSS-ROUND AGGREGATES:
  - Overall score          (SMILES + Exec + Fidelity) / 3  (same formula as R3, comparable)
  - Per-class breakdown    if test examples carry a 'class' field

Usage:
    mlx_lm.server --model chem_sage_32b_v3 --port 8080   # separate terminal
    python eval/eval_chem.py --model chem_sage_32b_v3 --out eval/scorecard_32b_v3.html
"""

import argparse
import json
import re
import sys
import textwrap
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
except ImportError:
    raise SystemExit("RDKit is required: pip install rdkit")

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.tool_exec import extract_code, run_sandboxed

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

SMILES_RE     = re.compile(r"MolFromSmiles\(['\"]([^'\"]+)['\"]\)")
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
FLOAT_RE      = re.compile(r"\b(\d+\.\d+)\b")

# Prose property claims: "MW 342.4", "logP: -0.12", "TPSA = 67.8 Å²"
PROP_RE = re.compile(
    r"(?P<prop>MW|logP|HBD|HBA|TPSA)\s*[=:\s]\s*(?P<val>[\d.]+)",
    re.IGNORECASE,
)

# Same as PROP_RE but also captures QED, and groups integer vs decimal parts
PREC_RE = re.compile(
    r"(?P<prop>MW|logP|TPSA|QED)\s*[=:\s]\s*\d+(?:\.(?P<dec>\d+))?",
    re.IGNORECASE,
)

QED_RE = re.compile(r"\bQED\s*[=:\s]\s*(?P<val>\d+\.\d+)", re.IGNORECASE)

# Expected decimal places per property
_PREC_DP: dict[str, int] = {"mw": 1, "logp": 2, "tpsa": 1, "qed": 2}

# Phrases that signal a refusal response
_REFUSAL_PHRASES = (
    "i cannot", "i can't", "i'm not able", "i am not able",
    "i don't have", "i do not have", "outside my scope",
    "outside the scope", "i'm not designed", "i'm unable",
    "i am unable", "i won't ", "can't help with",
    "cannot help with", "not able to help", "i'd recommend consulting",
    "please consult", "beyond my", "not something i can",
    "that falls outside", "i'm not equipped", "i should not",
    "i'm not in a position",
)

# Library markers for block categorisation (must be in block.lower())
_LIB_MARKERS = (
    ("rdkit",     ("from rdkit", "import rdkit")),
    ("plotly",    ("import plotly", "from plotly")),
    ("biopython", ("from bio", "import bio")),
)

# ---------------------------------------------------------------------------
# Injected system prompt (used when test examples lack a system message)
# ---------------------------------------------------------------------------

_EVAL_SYSTEM = (
    "You are ChemSage, a chemistry-specialised AI assistant for drug discovery. "
    "For exact physicochemical properties (MW, logP, HBD, HBA, TPSA, QED), always compute with RDKit — "
    "emit a ```python``` code block using the input SMILES; do not recall numbers from memory. "
    "NEVER invent SMILES strings, PDB IDs, ChEMBL IDs, or measured values."
)


# ---------------------------------------------------------------------------
# Model query
# ---------------------------------------------------------------------------

def query_model(base_url: str, model: str, messages: list[dict]) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    try:
        resp = requests.post(
            url,
            json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 1024},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return f"[error] cannot connect to {url} — is mlx_lm.server running?"
    except Exception as e:
        return f"[error] {e}"


# ---------------------------------------------------------------------------
# Original metrics (unchanged — all rounds comparable)
# ---------------------------------------------------------------------------

def smiles_validity(output: str) -> tuple[int, int]:
    """Count MolFromSmiles() calls that produce a valid molecule."""
    smis  = SMILES_RE.findall(output)
    valid = sum(1 for s in smis if Chem.MolFromSmiles(s) is not None)
    return valid, len(smis)


def tool_executability(output: str) -> tuple[int, int]:
    """Count RDKit ```python``` blocks that run without error."""
    rdkit_blocks = [b for b in extract_code(output) if "rdkit" in b.lower()]
    if not rdkit_blocks:
        return 0, 0
    ok = sum(1 for b in rdkit_blocks if not run_sandboxed(b).startswith("[error]"))
    return ok, len(rdkit_blocks)


def numerical_fidelity(output: str) -> tuple[int, int]:
    """Check stated MW/logP/HBD/HBA/TPSA values against RDKit recomputed truth.

    Accepts a match if the stated value is within tolerance of ANY molecule in the output
    (handles two-molecule comparison examples). Checks prose only (code blocks stripped).
    """
    smis = SMILES_RE.findall(output)
    if not smis:
        return 0, 0

    molecules = []
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            molecules.append({
                "mw":   round(Descriptors.MolWt(mol), 1),
                "logp": round(Descriptors.MolLogP(mol), 2),
                "hbd":  Lipinski.NumHDonors(mol),
                "hba":  Lipinski.NumHAcceptors(mol),
                "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 1),
            })
    if not molecules:
        return 0, 0

    tol   = {"mw": 0.5, "logp": 0.1, "hbd": 0, "hba": 0, "tpsa": 0.5}
    prose = CODE_BLOCK_RE.sub("", output)

    checked = ok = 0
    for match in PROP_RE.finditer(prose):
        key = match.group("prop").lower()
        try:
            stated = float(match.group("val").rstrip("."))
        except ValueError:
            continue
        if key not in molecules[0]:
            continue
        checked += 1
        if any(abs(stated - m[key]) <= tol[key] for m in molecules):
            ok += 1
    return ok, checked


# ---------------------------------------------------------------------------
# R4 extension metrics (all return (0, 0) when nothing to measure → n/a)
# ---------------------------------------------------------------------------

def rounding_precision(output: str) -> tuple[int, int]:
    """Check stated property values follow expected decimal-place conventions.

    MW → 1 d.p.  logP → 2 d.p.  TPSA → 1 d.p.  QED → 2 d.p.
    Values stated without a decimal point (e.g. "MW: 342") count as wrong.
    Returns (0, 0) if no property claims found → n/a for models that never state props.
    """
    prose   = CODE_BLOCK_RE.sub("", output)
    checked = ok = 0
    for m in PREC_RE.finditer(prose):
        key = m.group("prop").lower()
        if key not in _PREC_DP:
            continue
        dec_str = m.group("dec") or ""
        n_dp    = len(dec_str)
        checked += 1
        if n_dp == _PREC_DP[key]:
            ok += 1
    return ok, checked


def code_then_quote(output: str) -> tuple[int, int]:
    """Strict fidelity: prose values must match what the code blocks actually printed.

    Executes all RDKit blocks, collects every float from stdout, then checks that
    prose property claims are within ±0.05 of a printed value.

    Returns (0, 0) when code blocks print nothing (e.g. older models that don't use
    print() in their blocks), so the metric is naturally n/a for R1-R3 if they don't print.
    The delta between this and numerical_fidelity reveals how often the model is quoting
    its own code output vs. recalling from memory.
    """
    rdkit_blocks = [b for b in extract_code(output) if "rdkit" in b.lower()]
    if not rdkit_blocks:
        return 0, 0

    printed: set[float] = set()
    for block in rdkit_blocks:
        result = run_sandboxed(block)
        # Only parse from successful runs that actually printed something
        if result.startswith("[error]") or result == "[ok, no output]":
            continue
        for fm in FLOAT_RE.finditer(result):
            try:
                printed.add(float(fm.group(1)))
            except ValueError:
                pass

    if not printed:
        return 0, 0   # code ran but printed nothing → n/a

    prose   = CODE_BLOCK_RE.sub("", output)
    checked = ok = 0
    for m in PROP_RE.finditer(prose):
        try:
            stated = float(m.group("val").rstrip("."))
        except ValueError:
            continue
        checked += 1
        if any(abs(stated - p) < 0.05 for p in printed):
            ok += 1
    return ok, checked


def _block_category(block: str) -> str:
    """Return the library category of a ```python``` block."""
    for cat, markers in _LIB_MARKERS:
        if any(m in block.lower() for m in markers):
            return cat
    return "python"


def python_executability_extended(output: str) -> dict[str, tuple[int, int]]:
    """Run ALL ```python``` blocks; categorise by library; skip unavailable dependencies.

    Categories: rdkit · plotly · biopython · python (other)
    Blocks that fail with ModuleNotFoundError / ImportError are skipped (not penalised)
    since the eval environment may not have every library installed.
    Blocked code (network/file patterns) is also skipped.
    Returns empty dict → all categories show n/a when there are no Python blocks.
    """
    cats: dict[str, list[int]] = {}
    for block in extract_code(output):
        cat    = _block_category(block)
        result = run_sandboxed(block)

        # Skip: blocked by static check or dependency unavailable
        if result.startswith("[blocked]"):
            continue
        if any(x in result for x in ("ModuleNotFoundError", "ImportError", "No module named")):
            continue

        cats.setdefault(cat, [0, 0])
        cats[cat][1] += 1
        if not result.startswith("[error]"):
            cats[cat][0] += 1

    return {k: (v[0], v[1]) for k, v in cats.items()}


def _is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


def refusal_accuracy(output: str, expected: str) -> tuple[int, int]:
    """For examples whose ground-truth response is a refusal, check the model also refused.

    Returns (0, 0) when the ground truth is NOT a refusal — so this is automatically n/a
    for R1-R3 test sets that contain no refusal examples.
    """
    if not _is_refusal(expected):
        return 0, 0
    return (1, 1) if _is_refusal(output) else (0, 1)


def qed_range_check(output: str) -> tuple[int, int]:
    """Verify every stated QED value lies in [0, 1].

    Returns (0, 0) when the output contains no QED mentions → n/a for rounds without
    QED training examples.
    """
    prose = CODE_BLOCK_RE.sub("", output)
    vals  = []
    for m in QED_RE.finditer(prose):
        try:
            vals.append(float(m.group("val")))
        except ValueError:
            pass
    if not vals:
        return 0, 0
    return sum(1 for v in vals if 0.0 <= v <= 1.0), len(vals)


# ---------------------------------------------------------------------------
# HTML scorecard
# ---------------------------------------------------------------------------

def _pct(num: int, denom: int) -> str:
    return f"{num / denom:.0%}" if denom else "n/a"


def _pct_f(num: int, denom: int) -> float | None:
    return (num / denom) if denom else None


def _card(val: str, label: str, sub: str = "", variant: str = "") -> str:
    cls   = f"card {variant}" if variant else "card"
    body  = f"<div class='val'>{val}</div><div class='lbl'>{label}"
    body += f"<br><small>{sub}</small>" if sub else ""
    body += "</div>"
    return f"<div class='{cls}'>{body}</div>"


def _html_scorecard(results: list[dict], model: str, base_url: str, n_test: int) -> str:

    # ── Core metric aggregates ────────────────────────────────────────────────
    smi_ok  = sum(r["smiles_valid"]    for r in results)
    smi_tot = sum(r["smiles_total"]    for r in results)
    exe_ok  = sum(r["exec_ok"]         for r in results)
    exe_tot = sum(r["exec_total"]      for r in results)
    fid_ok  = sum(r["fidelity_ok"]     for r in results)
    fid_tot = sum(r["fidelity_total"]  for r in results)

    # Overall score (same 3-metric formula as R3 for cross-round comparability)
    core_pcts = [p for p in (
        _pct_f(smi_ok, smi_tot),
        _pct_f(exe_ok, exe_tot),
        _pct_f(fid_ok, fid_tot),
    ) if p is not None]
    overall = f"{sum(core_pcts)/len(core_pcts):.0%}" if core_pcts else "n/a"

    # ── R4 extension aggregates ───────────────────────────────────────────────
    prc_ok  = sum(r["precision_ok"]    for r in results)
    prc_tot = sum(r["precision_total"] for r in results)
    ctq_ok  = sum(r["ctq_ok"]          for r in results)
    ctq_tot = sum(r["ctq_total"]       for r in results)
    ref_ok  = sum(r["refusal_ok"]      for r in results)
    ref_tot = sum(r["refusal_total"]   for r in results)
    qed_ok  = sum(r["qed_ok"]          for r in results)
    qed_tot = sum(r["qed_total"]       for r in results)

    # Extended exec: aggregate across all library categories
    ext_ok = ext_tot = 0
    for r in results:
        for ok_v, tot_v in r.get("exec_cats", {}).values():
            ext_ok  += ok_v
            ext_tot += tot_v

    # Extended exec breakdown string (e.g. "rdkit 98% · plotly 85%")
    cat_totals: dict[str, list[int]] = {}
    for r in results:
        for cat, (ok_v, tot_v) in r.get("exec_cats", {}).items():
            cat_totals.setdefault(cat, [0, 0])
            cat_totals[cat][0] += ok_v
            cat_totals[cat][1] += tot_v
    ext_breakdown = "  ·  ".join(
        f"{cat} {_pct(v[0], v[1])}" for cat, v in sorted(cat_totals.items())
    ) or "no Python blocks"

    # ── Mean latency ─────────────────────────────────────────────────────────
    lats     = [r["latency"] for r in results if r.get("latency", 0) > 0]
    mean_lat = f"{sum(lats)/len(lats):.1f}s" if lats else "n/a"

    # ── Per-class breakdown ───────────────────────────────────────────────────
    has_classes = any(r.get("class") for r in results)
    class_stats: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {k: [0, 0] for k in ("n", "smiles", "exec", "fidelity", "precision", "ctq", "refusal")}
    )
    if has_classes:
        for r in results:
            cls = r.get("class") or "unknown"
            class_stats[cls]["n"][0] += 1
            for key, ok_v, tot_v in (
                ("smiles",    r["smiles_valid"],   r["smiles_total"]),
                ("exec",      r["exec_ok"],         r["exec_total"]),
                ("fidelity",  r["fidelity_ok"],     r["fidelity_total"]),
                ("precision", r["precision_ok"],    r["precision_total"]),
                ("ctq",       r["ctq_ok"],           r["ctq_total"]),
                ("refusal",   r["refusal_ok"],       r["refusal_total"]),
            ):
                class_stats[cls][key][0] += ok_v
                class_stats[cls][key][1] += tot_v

    class_section = ""
    if has_classes:
        cls_rows = ""
        for cls in sorted(class_stats):
            s = class_stats[cls]
            cls_rows += (
                f"<tr>"
                f"<td><b>{cls}</b></td>"
                f"<td class='num'>{s['n'][0]}</td>"
                f"<td class='num'>{_pct(s['smiles'][0],    s['smiles'][1])}</td>"
                f"<td class='num'>{_pct(s['exec'][0],      s['exec'][1])}</td>"
                f"<td class='num'>{_pct(s['fidelity'][0],  s['fidelity'][1])}</td>"
                f"<td class='num'>{_pct(s['precision'][0], s['precision'][1])}</td>"
                f"<td class='num'>{_pct(s['ctq'][0],       s['ctq'][1])}</td>"
                f"<td class='num'>{_pct(s['refusal'][0],   s['refusal'][1])}</td>"
                f"</tr>\n"
            )
        class_section = f"""
        <h2 class="section-h2">Per-class breakdown</h2>
        <table>
          <thead>
            <tr>
              <th>Class</th><th>N</th>
              <th>SMILES</th><th>Exec</th><th>Fidelity</th>
              <th>Precision</th><th>Ctq</th><th>Refusal</th>
            </tr>
          </thead>
          <tbody>{cls_rows}</tbody>
        </table>"""

    # ── Per-example rows (first 100) ─────────────────────────────────────────
    rows = ""
    for i, r in enumerate(results[:100], 1):
        q         = r["prompt"]
        q_display = (q[:80] + "…") if len(q) > 80 else q
        cls_badge = (
            f"<span class='badge'>{r['class']}</span>" if r.get("class") else ""
        )
        resp_raw  = r.get("output", "")
        resp_safe = resp_raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        resp_prev = (resp_safe[:400] + "…") if len(resp_safe) > 400 else resp_safe
        rows += (
            f"<tr>"
            f"<td class='num'>{i}</td>"
            f"<td class='q'>{q_display}{cls_badge}</td>"
            f"<td class='num'>{r['smiles_valid']}/{r['smiles_total']}</td>"
            f"<td class='num'>{r['exec_ok']}/{r['exec_total']}</td>"
            f"<td class='num'>{r['fidelity_ok']}/{r['fidelity_total']}</td>"
            f"<td class='num'>{r['precision_ok']}/{r['precision_total']}</td>"
            f"<td class='num'>{r['ctq_ok']}/{r['ctq_total']}</td>"
            f"<td class='num'>{r['refusal_ok']}/{r['refusal_total']}</td>"
            f"<td><details><summary>view</summary>"
            f"<div class='resp'>{resp_prev}</div></details></td>"
            f"</tr>\n"
        )

    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>ChemSage Eval — {datetime.now():%Y-%m-%d %H:%M}</title>
        <style>
          :root {{
            --navy:   #1C244B;
            --accent: #467FF7;
            --green:  #2ecc88;
            --bg:     #f5f7fb;
            --text:   #1a1a2e;
            --card:   #ffffff;
            --border: #e4e8f0;
          }}
          * {{ box-sizing: border-box; margin: 0; padding: 0; }}
          body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg); color: var(--text);
            padding: 2.5rem 2rem; max-width: 1300px; margin: auto;
          }}
          h1 {{ color: var(--navy); font-size: 1.7rem; margin-bottom: .25rem; }}
          .sub {{ color: #666; font-size: .82rem; margin-bottom: 2rem; }}
          .section-lbl {{
            font-size: .68rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: .07em; color: #999; margin: 1.8rem 0 .5rem;
          }}
          .section-h2 {{
            color: var(--navy); font-size: 1.1rem; margin: 2rem 0 .8rem;
          }}
          .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: .5rem; }}
          .card {{
            background: var(--card); border-radius: 12px;
            padding: 1.1rem 1.4rem; box-shadow: 0 2px 10px rgba(0,0,0,.06);
            min-width: 130px; text-align: center; flex: 1;
          }}
          .card.overall {{
            border-top: 3px solid var(--accent);
            padding: 1.3rem 1.8rem;
          }}
          .card.r4 {{ border-top: 3px solid var(--green); }}
          .card .val {{ font-size: 2rem; font-weight: 700; color: var(--accent); }}
          .card.overall .val {{ font-size: 2.6rem; }}
          .card.r4 .val {{ color: var(--green); }}
          .card .lbl {{ font-size: .76rem; color: #666; margin-top: .3rem; line-height: 1.5; }}
          .card .lbl small {{ font-size: .68rem; color: #999; }}
          table {{
            width: 100%; border-collapse: collapse;
            background: var(--card); border-radius: 12px;
            overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,.06);
            margin-bottom: 1.5rem; font-size: .77rem;
          }}
          th {{
            background: var(--navy); color: #fff;
            text-align: left; padding: .6rem .9rem; font-size: .75rem; font-weight: 600;
          }}
          td {{ padding: .45rem .9rem; border-bottom: 1px solid var(--border); }}
          td.q {{ max-width: 280px; word-break: break-word; }}
          td.num {{ text-align: center; white-space: nowrap; }}
          tr:last-child td {{ border-bottom: none; }}
          tr:nth-child(even) {{ background: #f9fafd; }}
          .badge {{
            display: inline-block; margin-left: .35rem;
            background: #e8effe; color: var(--accent);
            border-radius: 4px; padding: .08rem .35rem;
            font-size: .63rem; white-space: nowrap; vertical-align: middle;
          }}
          details summary {{
            cursor: pointer; color: var(--accent); font-size: .72rem; user-select: none;
          }}
          .resp {{
            background: #f0f4ff; border-radius: 6px; padding: .5rem .7rem;
            font-size: .7rem; max-height: 220px; overflow-y: auto;
            margin-top: .35rem; white-space: pre-wrap; word-break: break-word;
            font-family: monospace;
          }}
          .footer {{
            margin-top: 2rem; font-size: .72rem; color: #aaa;
            border-top: 1px solid var(--border); padding-top: 1rem;
          }}
          a {{ color: var(--accent); text-decoration: none; }}
          a:hover {{ text-decoration: underline; }}
        </style>
        </head>
        <body>

        <h1>ChemSage Evaluation Scorecard</h1>
        <div class="sub">
          Model: <b>{model}</b> &nbsp;·&nbsp; {base_url}
          &nbsp;·&nbsp; {datetime.now():%Y-%m-%d %H:%M}
          &nbsp;·&nbsp; {n_test} test examples
          &nbsp;·&nbsp; mean latency {mean_lat}
        </div>

        <div class="section-lbl">Core metrics — same formula as R1–R3, directly comparable</div>
        <div class="cards">
          {_card(overall,              "Overall score",       "(SMILES + Exec + Fidelity) / 3", "overall")}
          {_card(_pct(smi_ok,smi_tot), "SMILES validity",    f"{smi_ok}/{smi_tot} SMILES parsed")}
          {_card(_pct(exe_ok,exe_tot), "Tool executability", f"{exe_ok}/{exe_tot} RDKit blocks ran")}
          {_card(_pct(fid_ok,fid_tot), "Numerical fidelity", f"{fid_ok}/{fid_tot} props matched RDKit")}
          {_card(str(n_test),          "Test examples",      "evaluated")}
        </div>

        <div class="section-lbl">Round 4 extensions — n/a = metric not applicable to this model or dataset</div>
        <div class="cards">
          {_card(_pct(prc_ok,prc_tot), "Rounding precision",   f"{prc_ok}/{prc_tot} correct d.p.", "r4")}
          {_card(_pct(ctq_ok,ctq_tot), "Code-then-quote",      f"{ctq_ok}/{ctq_tot} prose = printed", "r4")}
          {_card(_pct(ext_ok,ext_tot), "Extended exec",        ext_breakdown, "r4")}
          {_card(_pct(ref_ok,ref_tot), "Refusal accuracy",     f"{ref_ok}/{ref_tot} refused correctly", "r4")}
          {_card(_pct(qed_ok,qed_tot), "QED in [0, 1]",        f"{qed_ok}/{qed_tot} values in range", "r4")}
        </div>

        {class_section}

        <h2 class="section-h2">Per-example results <span style="font-weight:400;font-size:.85rem;color:#999;">(first 100 shown)</span></h2>
        <table>
          <thead>
            <tr>
              <th>#</th><th>Prompt</th>
              <th>SMILES</th><th>Exec</th><th>Fidelity</th>
              <th>Precision</th><th>Ctq</th><th>Refusal</th>
              <th>Response</th>
            </tr>
          </thead>
          <tbody>
        {rows}  </tbody>
        </table>

        <div class="footer">
          Generated by ChemSage eval harness &nbsp;·&nbsp;
          <a href="https://marcdeller.com">Marc C. Deller, D.Phil.</a> &nbsp;·&nbsp;
          <a href="mailto:marc@marcdeller.com">marc@marcdeller.com</a>
        </div>
        </body>
        </html>
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="ChemSage evaluation harness — runs against mlx_lm.server"
    )
    ap.add_argument("--base-url", default="http://localhost:8080/v1",
                    help="OpenAI-compatible endpoint (default: http://localhost:8080/v1)")
    ap.add_argument("--model",    default="fused_model",
                    help="Model name passed to the API (default: fused_model)")
    ap.add_argument("--test",     type=Path, default=Path("data/sft/test.jsonl"),
                    help="Frozen test split (default: data/sft/test.jsonl)")
    ap.add_argument("--out",      type=Path, default=Path("eval/scorecard.html"),
                    help="HTML scorecard output path (default: eval/scorecard.html)")
    ap.add_argument("--limit",    type=int,  default=None,
                    help="Evaluate only the first N examples (smoke-test mode)")
    args = ap.parse_args()

    if not args.test.exists():
        raise SystemExit(
            f"Test set not found: {args.test}. Run scripts/build_dataset.py first."
        )

    examples = [json.loads(l) for l in args.test.read_text().splitlines() if l.strip()]
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit("Test file is empty.")

    from tqdm import tqdm

    has_classes = any(ex.get("class") or ex.get("behaviour_class") for ex in examples)
    print(f"Evaluating {len(examples)} examples against {args.base_url} ({args.model})")
    if has_classes:
        print("  Per-class breakdown will be included in the scorecard.")
    else:
        print("  No 'class' field found — per-class breakdown skipped.")

    results: list[dict] = []
    errors   = 0
    running: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    pbar = tqdm(examples, unit="ex", dynamic_ncols=True)
    for ex in pbar:
        # Build message list, stripping the ground-truth assistant turn
        messages = [m for m in ex["messages"] if m["role"] in ("system", "user")]
        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": _EVAL_SYSTEM})

        prompt   = next((m["content"] for m in messages if m["role"] == "user"), "")
        expected = next(
            (m["content"] for m in ex["messages"] if m["role"] == "assistant"), ""
        )
        cls = ex.get("class") or ex.get("behaviour_class") or ""

        t0      = time.time()
        output  = query_model(args.base_url, args.model, messages)
        latency = time.time() - t0

        if output.startswith("[error]"):
            errors += 1

        # ── Run all metrics ───────────────────────────────────────────────────
        sv, st = smiles_validity(output)
        ev, et = tool_executability(output)
        fv, ft = numerical_fidelity(output)
        pv, pt = rounding_precision(output)
        cv, ct = code_then_quote(output)
        rv, rt = refusal_accuracy(output, expected)
        qv, qt = qed_range_check(output)
        ec     = python_executability_extended(output)

        for key, ok_v, tot_v in (
            ("smiles",    sv, st), ("exec",    ev, et), ("fidelity", fv, ft),
            ("precision", pv, pt), ("ctq",     cv, ct),
            ("refusal",   rv, rt), ("qed",     qv, qt),
        ):
            running[key][0] += ok_v
            running[key][1] += tot_v

        results.append({
            "prompt":         prompt,
            "output":         output,
            "expected":       expected,
            "class":          cls,
            "latency":        latency,
            "smiles_valid":   sv, "smiles_total":    st,
            "exec_ok":        ev, "exec_total":       et,
            "fidelity_ok":    fv, "fidelity_total":   ft,
            "precision_ok":   pv, "precision_total":  pt,
            "ctq_ok":         cv, "ctq_total":        ct,
            "refusal_ok":     rv, "refusal_total":    rt,
            "qed_ok":         qv, "qed_total":        qt,
            "exec_cats":      ec,
        })

        pbar.set_postfix_str(
            f"SMILES {running['smiles'][0]}/{running['smiles'][1]}  "
            f"Exec {running['exec'][0]}/{running['exec'][1]}  "
            f"Fid {running['fidelity'][0]}/{running['fidelity'][1]}  "
            f"Prec {running['precision'][0]}/{running['precision'][1]}"
            + (f"  err={errors}" if errors else ""),
            refresh=True,
        )

    # ── Write scorecard ───────────────────────────────────────────────────────
    html = _html_scorecard(results, args.model, args.base_url, len(examples))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)

    # ── Terminal summary ──────────────────────────────────────────────────────
    W = 62
    print(f"\n{'─' * W}")
    print(f"  {'Metric':<28} {'Score':>8}   Detail")
    print(f"{'─' * W}")
    for label, key in (
        ("SMILES validity",    "smiles"),
        ("Tool executability", "exec"),
        ("Numerical fidelity", "fidelity"),
    ):
        ok, tot = running[key]
        print(f"  {label:<28} {_pct(ok, tot):>8}   ({ok}/{tot})")
    print(f"  {'─' * (W - 4)}")
    for label, key in (
        ("Rounding precision",  "precision"),
        ("Code-then-quote",     "ctq"),
        ("Refusal accuracy",    "refusal"),
        ("QED in [0, 1]",       "qed"),
    ):
        ok, tot = running[key]
        score   = _pct(ok, tot)
        suffix  = "" if tot else "  [n/a — not applicable to this dataset]"
        print(f"  {label:<28} {score:>8}   ({ok}/{tot}){suffix}")
    print(f"{'─' * W}")
    if errors:
        print(f"  ⚠  {errors} query error{'s' if errors != 1 else ''} — check server connection.")
    print(f"\nScorecard → {args.out}\n")


if __name__ == "__main__":
    main()
