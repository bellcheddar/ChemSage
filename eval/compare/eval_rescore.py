#!/usr/bin/env python3
"""
eval_rescore.py — re-score existing raw_results.json with updated metric functions.

Loads the cached model outputs and applies the current (fixed) metric functions
without re-running any model servers.  Writes a new timestamped report.

Usage:
    cd chem_sage/
    python eval/compare/eval_rescore.py
    python eval/compare/eval_rescore.py --input results/raw_results.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import everything from eval_compare (metrics, reporters, etc.)
from eval.compare.eval_compare import (
    METRICS,
    score_all,
    html_report,
    md_report,
)


def rescore(raw: dict) -> dict:
    """Re-run metric functions over cached outputs; return updated raw dict."""
    for mid, model_data in raw.items():
        if model_data.get("skipped"):
            continue
        results = model_data.get("results", [])
        for res in results:
            output   = res.get("output", "")
            expected = res.get("expected", "")
            prompt   = res.get("prompt", "")
            res["scores"] = score_all(output, expected, prompt)

        # Re-aggregate
        agg: dict[str, tuple[int, int]] = {}
        for key, *_ in METRICS:
            agg[key] = (
                sum(r["scores"][key][0] for r in results),
                sum(r["scores"][key][1] for r in results),
            )
        model_data["aggregate"] = agg
    return raw


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-score ChemSage eval from cached outputs")
    ap.add_argument(
        "--input", type=Path,
        default=Path(__file__).parent / "results" / "raw_results.json",
        help="Path to existing raw_results.json",
    )
    ap.add_argument(
        "--config", type=Path,
        default=Path(__file__).parent / "models.yaml",
        help="Path to models.yaml",
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).parent / "results",
    )
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"raw_results.json not found: {args.input}")

    cfg_raw  = yaml.safe_load(args.config.read_text())
    settings = cfg_raw["settings"]
    all_cfgs = cfg_raw["models"]

    raw = json.loads(args.input.read_text())
    print(f"Loaded cached results for: {[k for k in raw if not raw[k].get('skipped')]}")
    print("Re-scoring with updated metric functions…")

    rescored = rescore(raw)

    # Print quick table
    print(f"\n{'Model':<14} {'SMILES':>8} {'Exec':>8} {'Fidelity':>10} {'PyExec':>8}")
    print("─" * 52)
    for cfg in all_cfgs:
        mid = cfg["id"]
        er  = rescored.get(mid)
        if not er or er.get("skipped"):
            continue
        agg = er["aggregate"]
        def pct(k):
            ok, tot = agg[k]
            return f"{ok/tot:.0%}" if tot else "n/a"
        print(f"  {cfg['display_name']:<12} {pct('smiles_validity'):>8} "
              f"{pct('tool_executability'):>8} {pct('numerical_fidelity'):>10} "
              f"{pct('python_extended'):>8}")
    print()

    # Save rescored JSON
    rescored_path = args.out_dir / "raw_results_rescored.json"
    rescored_path.write_text(json.dumps(rescored, indent=2, default=str))
    print(f"Rescored JSON → {rescored_path}")

    # Generate reports
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M")
    html_path    = args.out_dir / f"compare_{timestamp}.html"
    md_path      = args.out_dir / f"compare_{timestamp}.md"

    html_path.write_text(html_report(all_cfgs, rescored, settings, generated_at))
    md_path.write_text(md_report(all_cfgs, rescored, settings, generated_at))
    print(f"HTML  → {html_path}")
    print(f"MD    → {md_path}")


if __name__ == "__main__":
    main()
