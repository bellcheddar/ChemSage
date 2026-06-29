#!/usr/bin/env python3
"""
eval_chem_original.py — original chemistry-aware evaluation harness (Phase 7, R1-R3).

Preserved as-is from before the R4 extension. Use this to rerun R1-R3 scorecards
with the original 3-metric formula for an exact apples-to-apples comparison.

Queries an OpenAI-compatible endpoint (mlx_lm.server at http://localhost:8080/v1 by default)
against the frozen test split and measures:

  - SMILES validity rate   : every MolFromSmiles() call in emitted code must parse in RDKit.
  - Tool executability     : every RDKit code block must run without error (via tool_exec).
  - Numerical fidelity     : stated MW/logP/HBD/HBA/TPSA values must match RDKit recomputation.

Writes an HTML scorecard (eval/scorecard.html) using the marcdeller.com brand palette.

Usage:
    mlx_lm.server --model fused_model --port 8080   # in a separate terminal
    python eval/eval_chem_original.py --base-url http://localhost:8080/v1 --model fused_model
"""

import argparse
import json
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import requests

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
except ImportError:
    raise SystemExit("RDKit is required: pip install rdkit")

# Resolve the rag/ sibling directory so tool_exec is importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.tool_exec import extract_code, run_sandboxed

SMILES_RE = re.compile(r"MolFromSmiles\(['\"]([^'\"]+)['\"]\)")
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
# Match stated property values in prose: "MW 342.4", "MW: 342.4", "MW = 342.4 Da", "MW=342.4"
PROP_RE   = re.compile(
    r"(?P<prop>MW|logP|HBD|HBA|TPSA)\s*[=:\s]\s*(?P<val>[\d.]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# System prompt (injected if absent from test examples)
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
# Metrics
# ---------------------------------------------------------------------------

def smiles_validity(output: str) -> tuple[int, int]:
    """Count MolFromSmiles() calls in the output that produce a valid molecule."""
    smis  = SMILES_RE.findall(output)
    valid = sum(1 for s in smis if Chem.MolFromSmiles(s) is not None)
    return valid, len(smis)


def tool_executability(output: str) -> tuple[int, int]:
    """Count RDKit code blocks that run without error."""
    rdkit_blocks = [b for b in extract_code(output) if "rdkit" in b.lower()]
    if not rdkit_blocks:
        return 0, 0
    ok = sum(1 for b in rdkit_blocks if not run_sandboxed(b).startswith("[error]"))
    return ok, len(rdkit_blocks)


def numerical_fidelity(output: str) -> tuple[int, int]:
    """Check stated MW/logP/HBD/HBA/TPSA values against RDKit recomputation.

    Requires a MolFromSmiles() call in the same output so we know which molecule to check.
    Tolerates a small floating-point delta (0.5 for MW/logP, 0 for integer descriptors).
    Only checks prose (strips code blocks to avoid matching code expressions).
    For multi-molecule outputs, accepts a value if it matches ANY molecule present.
    """
    smis = SMILES_RE.findall(output)
    if not smis:
        return 0, 0

    # Build truth tables for ALL valid molecules (handles comparison examples)
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

    tol = {"mw": 0.5, "logp": 0.1, "hbd": 0, "hba": 0, "tpsa": 0.5}

    # Strip code blocks — only check prose claims
    prose = CODE_BLOCK_RE.sub("", output)

    checked = ok = 0
    for match in PROP_RE.finditer(prose):
        key   = match.group("prop").lower()
        try:
            stated = float(match.group("val").rstrip("."))
        except ValueError:
            continue
        if key not in molecules[0]:
            continue
        checked += 1
        if any(abs(stated - mol_truth[key]) <= tol[key] for mol_truth in molecules):
            ok += 1

    return ok, checked


# ---------------------------------------------------------------------------
# HTML scorecard
# ---------------------------------------------------------------------------

def _pct(num: int, denom: int) -> str:
    return f"{num / denom:.0%}" if denom else "n/a"


def _html_scorecard(results: list[dict], model: str, base_url: str, n_test: int) -> str:
    smi_ok  = sum(r["smiles_valid"]  for r in results)
    smi_tot = sum(r["smiles_total"]  for r in results)
    exe_ok  = sum(r["exec_ok"]       for r in results)
    exe_tot = sum(r["exec_total"]    for r in results)
    fid_ok  = sum(r["fidelity_ok"]   for r in results)
    fid_tot = sum(r["fidelity_total"] for r in results)

    rows = ""
    for i, r in enumerate(results[:100], 1):
        q = r["prompt"]
        q_display = (q[:90] + "…") if len(q) > 90 else q
        rows += (
            f"<tr>"
            f"<td>{i}</td>"
            f"<td class='q'>{q_display}</td>"
            f"<td class='num'>{r['smiles_valid']}/{r['smiles_total']}</td>"
            f"<td class='num'>{r['exec_ok']}/{r['exec_total']}</td>"
            f"<td class='num'>{r['fidelity_ok']}/{r['fidelity_total']}</td>"
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
            --bg:     #f5f7fb;
            --text:   #1a1a2e;
            --card:   #ffffff;
            --border: #e4e8f0;
          }}
          * {{ box-sizing: border-box; margin: 0; padding: 0; }}
          body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg); color: var(--text);
            padding: 2.5rem 2rem; max-width: 1100px; margin: auto;
          }}
          h1   {{ color: var(--navy); font-size: 1.7rem; margin-bottom: .2rem; }}
          .sub {{ color: #666; font-size: .85rem; margin-bottom: 2rem; }}
          .cards {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 2.5rem; }}
          .card {{
            background: var(--card); border-radius: 14px;
            padding: 1.4rem 2rem; box-shadow: 0 2px 10px rgba(0,0,0,.06);
            min-width: 160px; text-align: center; flex: 1;
          }}
          .card .val {{ font-size: 2.5rem; font-weight: 700; color: var(--accent); }}
          .card .lbl {{ font-size: .8rem; color: #666; margin-top: .3rem; line-height: 1.4; }}
          table {{
            width: 100%; border-collapse: collapse;
            background: var(--card); border-radius: 14px;
            overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,.06);
          }}
          th {{
            background: var(--navy); color: #fff;
            text-align: left; padding: .7rem 1rem; font-size: .82rem; font-weight: 600;
          }}
          td {{ padding: .55rem 1rem; border-bottom: 1px solid var(--border); font-size: .8rem; }}
          td.q   {{ max-width: 380px; word-break: break-word; }}
          td.num {{ text-align: center; }}
          tr:last-child td {{ border-bottom: none; }}
          tr:nth-child(even) {{ background: #f9fafd; }}
          .footer {{ margin-top: 2rem; font-size: .72rem; color: #aaa; }}
          a {{ color: var(--accent); text-decoration: none; }}
          a:hover {{ text-decoration: underline; }}
        </style>
        </head>
        <body>
        <h1>ChemSage Evaluation Scorecard</h1>
        <div class="sub">
          Model: <b>{model}</b> &nbsp;|&nbsp; Endpoint: {base_url}
          &nbsp;|&nbsp; {datetime.now():%Y-%m-%d %H:%M}
          &nbsp;|&nbsp; {n_test} test examples
        </div>
        <div class="cards">
          <div class="card">
            <div class="val">{_pct(smi_ok, smi_tot)}</div>
            <div class="lbl">SMILES validity<br>({smi_ok}/{smi_tot} SMILES parsed)</div>
          </div>
          <div class="card">
            <div class="val">{_pct(exe_ok, exe_tot)}</div>
            <div class="lbl">Tool executability<br>({exe_ok}/{exe_tot} RDKit blocks ran)</div>
          </div>
          <div class="card">
            <div class="val">{_pct(fid_ok, fid_tot)}</div>
            <div class="lbl">Numerical fidelity<br>({fid_ok}/{fid_tot} props matched RDKit)</div>
          </div>
          <div class="card">
            <div class="val">{n_test}</div>
            <div class="lbl">Test examples<br>evaluated</div>
          </div>
        </div>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Prompt</th>
              <th>SMILES ok/total</th>
              <th>Exec ok/total</th>
              <th>Fidelity ok/total</th>
            </tr>
          </thead>
          <tbody>
        {rows}  </tbody>
        </table>
        <div class="footer">
          Generated by ChemSage eval harness &nbsp;·&nbsp;
          Marc C. Deller, D.Phil. &nbsp;·&nbsp;
          <a href="https://marcdeller.com">marcdeller.com</a>
        </div>
        </body>
        </html>
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model",    default="fused_model")
    ap.add_argument("--test",     type=Path, default=Path("data/sft/test.jsonl"))
    ap.add_argument("--out",      type=Path, default=Path("eval/scorecard.html"))
    ap.add_argument("--limit",    type=int,  default=None,
                    help="Evaluate only the first N examples (useful for a quick smoke-test)")
    args = ap.parse_args()

    if not args.test.exists():
        raise SystemExit(
            f"Test set not found: {args.test}. Run scripts/build_dataset.py first."
        )

    examples = [
        json.loads(line) for line in args.test.read_text().splitlines() if line.strip()
    ]
    if args.limit:
        examples = examples[:args.limit]
    if not examples:
        raise SystemExit("Test file is empty.")

    from tqdm import tqdm

    print(f"Evaluating {len(examples)} examples against {args.base_url} ({args.model})")
    results = []
    errors = 0
    smi_running = [0, 0]
    exe_running = [0, 0]
    fid_running = [0, 0]

    pbar = tqdm(examples, unit="ex", dynamic_ncols=True)
    for ex in pbar:
        messages = [m for m in ex["messages"] if m["role"] in ("system", "user")]
        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": _EVAL_SYSTEM})
        prompt   = next((m["content"] for m in messages if m["role"] == "user"), "")
        output   = query_model(args.base_url, args.model, messages)

        if output.startswith("[error]"):
            errors += 1

        sv,  st  = smiles_validity(output)
        ev,  et  = tool_executability(output)
        fv,  ft  = numerical_fidelity(output)
        smi_running[0] += sv; smi_running[1] += st
        exe_running[0] += ev; exe_running[1] += et
        fid_running[0] += fv; fid_running[1] += ft

        results.append({
            "prompt":         prompt,
            "output":         output,
            "smiles_valid":   sv, "smiles_total":   st,
            "exec_ok":        ev, "exec_total":      et,
            "fidelity_ok":    fv, "fidelity_total":  ft,
        })
        pbar.set_postfix_str(
            f"SMILES {smi_running[0]}/{smi_running[1]}  "
            f"Exec {exe_running[0]}/{exe_running[1]}  "
            f"Fid {fid_running[0]}/{fid_running[1]}"
            + (f"  err={errors}" if errors else ""),
            refresh=True,
        )

    html = _html_scorecard(results, args.model, args.base_url, len(examples))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)

    smi_ok  = sum(r["smiles_valid"]  for r in results)
    smi_tot = sum(r["smiles_total"]  for r in results)
    exe_ok  = sum(r["exec_ok"]       for r in results)
    exe_tot = sum(r["exec_total"]    for r in results)
    fid_ok  = sum(r["fidelity_ok"]   for r in results)
    fid_tot = sum(r["fidelity_total"] for r in results)

    print(f"\nSMILES validity   : {smi_ok}/{smi_tot} = {_pct(smi_ok, smi_tot)}")
    print(f"Tool executability: {exe_ok}/{exe_tot} = {_pct(exe_ok, exe_tot)}")
    print(f"Numerical fidelity: {fid_ok}/{fid_tot} = {_pct(fid_ok, fid_tot)}")
    print(f"\nScorecard → {args.out}")


if __name__ == "__main__":
    main()
