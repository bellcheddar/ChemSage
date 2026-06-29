#!/usr/bin/env python3
"""
eval_compare.py — side-by-side multi-round comparison eval for ChemSage.

Spawns one mlx_lm.server per model (Option A), runs N shared examples from the
R4 test set, aggregates all 13 metrics, and writes .html + .md reports.

Usage:
    cd chem_sage/
    python eval/compare/eval_compare.py
    python eval/compare/eval_compare.py --resume          # skip already-run models
    python eval/compare/eval_compare.py --models r3 r4   # subset only
    python eval/compare/eval_compare.py --limit 10        # quick smoke-test
"""

import argparse
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import yaml

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem import QED as _QED
except ImportError:
    raise SystemExit("RDKit is required: pip install rdkit")

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import pyfiglet
    _ART = pyfiglet.figlet_format("ChemSage", font="small_slant")
except Exception:
    _ART = "  ChemSage"

# Project root is two levels up from this file (eval/compare/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from rag.tool_exec import extract_code, run_sandboxed

# ─── system prompt ─────────────────────────────────────────────────────────────

EVAL_SYSTEM = (
    "You are ChemSage, a chemistry-specialised AI assistant for drug discovery. "
    "For exact physicochemical properties (MW, logP, HBD, HBA, TPSA, QED), always compute with RDKit — "
    "emit a ```python``` code block using the input SMILES; do not recall numbers from memory. "
    "NEVER invent SMILES strings, PDB IDs, ChEMBL IDs, or measured values."
)

# ─── regex ─────────────────────────────────────────────────────────────────────

SMILES_RE = re.compile(r"MolFromSmiles\(['\"]([^'\"]+)['\"]\)")
CODE_RE   = re.compile(r"```.*?```", re.DOTALL)
PROP_RE   = re.compile(
    r"(?P<prop>MW|logP|HBD|HBA|TPSA|QED)\s*[=:\s]\s*(?P<val>[\d.]+)", re.IGNORECASE
)
# Matches "QED = 0.70", "QED: 0.70", "QED 0.57", "QED score: 0.33", "QED is 0.45" etc.
# Allows up to 30 non-digit non-newline chars between the keyword and the value.
QED_RE    = re.compile(
    r"QED(?:\s+(?:score|value|drug.?like(?:ness)?(?:\s+score)?))?[^0-9\n]{0,30}"
    r"(0?\.\d+|1(?:\.0+)?)(?!\d)",
    re.IGNORECASE,
)
REFUSAL_PHRASES = (
    "cannot", "unable", "outside", "harmful", "illegal", "unethical",
    "i'm not able", "i cannot", "beyond the scope", "not appropriate",
    "not designed to", "i won't", "i will not",
)

# Training-log parsing regexes
_TRAIN_LINE_RE = re.compile(
    r"^Iter (\d+): Train loss ([\d.]+), "
    r"Learning Rate ([\d.eE+-]+), "
    r"It/sec ([\d.]+), Tokens/sec ([\d.]+), "
    r"Trained Tokens (\d+), Peak mem ([\d.]+) GB"
)
_VAL_TOOK_RE = re.compile(
    r"^Iter (\d+): Val loss [\d.]+, Val took ([\d.]+)s"
)

# Ordered list of (key, label, unit) for training metrics chart
_TRAIN_METRIC_INFO = [
    ("train_loss",     "Training Loss",   ""),
    ("learning_rate",  "Learning Rate",   ""),
    ("itps",           "It / sec",        "it/s"),
    ("tokens_per_sec", "Tokens / sec",    "tok/s"),
    ("trained_tokens", "Trained Tokens",  ""),
    ("peak_mem",       "Peak Mem",        "GB"),
    ("val_took",       "Val Duration",    "s"),
]

# PyMOL command-line syntax detector.
# Blocks like `fetch 1HSG, async=0` are PyMOL scripts accidentally placed in
# ```python``` fences by the model.  They are not Python and should not count
# as Python execution failures.
_PYMOL_CMD_RE = re.compile(
    r"^(?:fetch|select|show|hide|color|orient|zoom|align|center|remove|ray|png|save)\s+\S",
    re.MULTILINE,
)
_PYTHON_STRUCTURE_RE = re.compile(
    r"^(?:import |from .+ import |def |class |\w+ ?=)",
    re.MULTILINE,
)

def _is_pymol_command_block(code: str) -> bool:
    """Return True when a code block is PyMOL command-line syntax (not Python API)."""
    return bool(_PYMOL_CMD_RE.search(code)) and not bool(_PYTHON_STRUCTURE_RE.search(code))


# Extracts ALL fenced code blocks regardless of language tag (python, pymol, bash, bare, etc.)
_ALL_CODE_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)


def _extract_all_blocks(text: str) -> list[str]:
    """Return content of every fenced code block in text, regardless of language tag."""
    return [m.group(1).strip() for m in _ALL_CODE_RE.finditer(text)]

# ─── resource monitor ──────────────────────────────────────────────────────────

class ResourceMonitor:
    """Samples CPU % and RSS of a server PID every 3 seconds in a daemon thread."""

    def __init__(self, pid: int):
        self._pid = pid
        self.cpu_samples: list[float] = []
        self.rss_samples: list[int]   = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if _HAS_PSUTIL:
            self._thread.start()

    def _run(self):
        try:
            proc = psutil.Process(self._pid)
            proc.cpu_percent(interval=None)   # prime the counter
            while not self._stop.wait(3.0):
                try:
                    children = proc.children(recursive=True)
                    cpu = proc.cpu_percent(interval=None) + sum(
                        c.cpu_percent(interval=None) for c in children
                    )
                    rss = proc.memory_info().rss + sum(
                        c.memory_info().rss for c in children
                    )
                    self.cpu_samples.append(cpu)
                    self.rss_samples.append(rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        except Exception:
            pass

    def stop(self) -> dict:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        _m = lambda xs: round(statistics.mean(xs), 1) if xs else None
        _x = lambda xs: round(max(xs), 1) if xs else None
        return {
            "cpu_mean_pct": _m(self.cpu_samples),
            "cpu_peak_pct": _x(self.cpu_samples),
            "rss_mean_gb":  round(statistics.mean(self.rss_samples) / 1e9, 2) if self.rss_samples else None,
            "rss_peak_gb":  round(max(self.rss_samples) / 1e9, 2) if self.rss_samples else None,
            "n_samples":    len(self.cpu_samples),
        }


# ─── server management ─────────────────────────────────────────────────────────

def _start_server(model_path: str, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "mlx_lm.server", "--model", model_path, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )


def _wait_for_server(port: int, timeout: int) -> bool:
    # Probe /v1/models — returns 200 as soon as the server is up regardless of model name.
    url      = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception:
            pass
        time.sleep(3)
    return False


def _stop_server(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()


# ─── test set sampling ─────────────────────────────────────────────────────────

def sample_examples(test_file: Path, n: int, seed: int) -> list[dict]:
    lines = [json.loads(l) for l in test_file.read_text().splitlines() if l.strip()]
    if len(lines) <= n:
        return lines
    return random.Random(seed).sample(lines, n)


# ─── model query ───────────────────────────────────────────────────────────────

def query_model(port: int, messages: list[dict], timeout: int = 120, model_path: str = "chemsage") -> tuple[str, float]:
    url = f"http://localhost:{port}/v1/chat/completions"
    t0  = time.time()
    try:
        resp = requests.post(url, json={
            "model": model_path,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024,
        }, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"], time.time() - t0
    except Exception as e:
        return f"[error] {e}", time.time() - t0


# ─── metrics ───────────────────────────────────────────────────────────────────

def smiles_validity(output: str) -> tuple[int, int]:
    smis = SMILES_RE.findall(output)
    ok   = sum(1 for s in smis if Chem.MolFromSmiles(s) is not None)
    return ok, len(smis)


def tool_executability(output: str) -> tuple[int, int]:
    blocks = [b for b in extract_code(output) if "rdkit" in b.lower()]
    if not blocks:
        return 0, 0
    ok = sum(1 for b in blocks if not run_sandboxed(b).startswith("[error]"))
    return ok, len(blocks)


def numerical_fidelity(output: str) -> tuple[int, int]:
    smis = SMILES_RE.findall(output)
    if not smis:
        return 0, 0
    molecules = []
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        if mol:
            molecules.append({
                "mw":   round(Descriptors.MolWt(mol), 1),
                "logp": round(Descriptors.MolLogP(mol), 2),
                "hbd":  Lipinski.NumHDonors(mol),
                "hba":  Lipinski.NumHAcceptors(mol),
                "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 1),
                "qed":  round(_QED.qed(mol), 3),
            })
    if not molecules:
        return 0, 0
    tol   = {"mw": 0.5, "logp": 0.1, "hbd": 0, "hba": 0, "tpsa": 0.5, "qed": 0.005}
    prose = CODE_RE.sub("", output)
    ok = checked = 0
    for m in PROP_RE.finditer(prose):
        key = m.group("prop").lower()
        try:
            stated = float(m.group("val").rstrip("."))
        except ValueError:
            continue
        if key not in molecules[0]:
            continue
        checked += 1
        if any(abs(stated - mol[key]) <= tol[key] for mol in molecules):
            ok += 1
    return ok, checked


def rounding_precision(output: str) -> tuple[int, int]:
    """MW ≥1 dp, logP ≥2 dp, TPSA ≥1 dp, QED ≥2 dp."""
    DP_RULES = [
        (re.compile(r"(?:MW|molecular weight)[^\d\n]*?(\d+\.\d+)", re.IGNORECASE), 1),
        (re.compile(r"(?:logP|log P)[^\d\n]*?(-?\d+\.\d+)", re.IGNORECASE), 2),
        (re.compile(r"TPSA[^\d\n]*?(\d+\.\d+)", re.IGNORECASE), 1),
        (re.compile(r"QED[^\d\n]*?(0?\.\d+)", re.IGNORECASE), 2),
    ]
    prose   = CODE_RE.sub("", output)
    ok = checked = 0
    for pattern, req_dp in DP_RULES:
        for m in pattern.finditer(prose):
            val_str = m.group(1)
            if "." in val_str:
                dp = len(val_str.split(".")[1])
                checked += 1
                if dp >= req_dp:
                    ok += 1
    return ok, checked


def code_then_quote(output: str) -> tuple[int, int]:
    """Numeric values from code stdout should appear in prose within ±0.05."""
    blocks = extract_code(output)
    if not blocks:
        return 0, 0
    prose    = CODE_RE.sub("", output)
    prose_vs = {float(x) for x in re.findall(r"-?\d+\.\d+", prose)}
    ok = checked = 0
    for block in blocks:
        if _is_pymol_command_block(block):
            continue  # PyMOL scripts produce no numeric stdout
        result = run_sandboxed(block)
        if result.startswith("[error]") or result == "[ok, no output]":
            continue
        code_vs = {float(x) for x in re.findall(r"-?\d+\.\d+", result)}
        for cv in code_vs:
            checked += 1
            if any(abs(cv - pv) <= 0.05 for pv in prose_vs):
                ok += 1
    return ok, checked


def python_extended(output: str) -> tuple[int, int]:
    """All Python blocks should run without error.

    Forgiven (environment gaps, not model errors):
    - ModuleNotFoundError / ImportError: library not in eval sandbox
    - FileNotFoundError: PDB / data file not present in sandbox
    - PyMOL command-line syntax in ```python``` fence: training-data artefact
      (model correctly wrote PyMOL script but used wrong code fence language);
      excluded from denominator entirely rather than counted as a failure.
    """
    blocks = extract_code(output)
    if not blocks:
        return 0, 0
    ok = total = 0
    for block in blocks:
        if _is_pymol_command_block(block):
            continue  # exclude PyMOL command scripts from denominator
        total += 1
        result = run_sandboxed(block)
        if result.startswith("[error]"):
            if (
                "ModuleNotFoundError" in result
                or "ImportError" in result
                or "FileNotFoundError" in result
                or "No such file" in result
            ):
                ok += 1  # environment gap, not model error
        else:
            ok += 1
    return ok, total


def degeneration_score(output: str) -> tuple[int, int]:
    """1=clean output, 0=repetition collapse (any 3-token sequence repeated >5×)."""
    tokens = output.split()
    if len(tokens) < 12:
        return 1, 1  # too short to degenerate
    counts: dict[tuple, int] = {}
    for i in range(len(tokens) - 2):
        tg = (tokens[i], tokens[i + 1], tokens[i + 2])
        counts[tg] = counts.get(tg, 0) + 1
    max_repeat = max(counts.values()) if counts else 0
    return (0 if max_repeat > 5 else 1), 1


_COMPUTE_TRIGGERS = (
    "compute ", "calculate ", "use rdkit", "rdkit to",
    "qed score", "qed of ", "what is the qed",
    "topological polar surface area",
    "tanimoto", "fingerprint simil", "same scaffold",
)


def code_attempted(output: str, prompt: str = "") -> tuple[int, int]:
    """Penalise prose-dodging on targeted compute questions.

    0/0 if the prompt doesn't require a computation (not penalised).
    1/1 if computation required and model emitted at least one code block.
    0/1 if computation required but model answered with prose only.
    """
    if not prompt:
        return 0, 0
    pl = prompt.lower()
    if not any(kw in pl for kw in _COMPUTE_TRIGGERS):
        return 0, 0
    return (1 if extract_code(output) else 0), 1


def _is_refusal(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in REFUSAL_PHRASES)


def refusal_accuracy(output: str, expected: str) -> tuple[int, int]:
    """Model should refuse iff ground truth refuses."""
    if not expected:
        return 0, 0
    gt_refuses    = _is_refusal(expected)
    model_refuses = _is_refusal(output)
    return (1 if gt_refuses == model_refuses else 0), 1


def qed_range_check(output: str) -> tuple[int, int]:
    """Stated QED values must lie in [0, 1]."""
    matches = QED_RE.findall(output)
    if not matches:
        return 0, 0
    ok = sum(1 for v in matches if 0.0 <= float(v) <= 1.0)
    return ok, len(matches)


# PDB ID pattern: 4 characters, first is a digit, rest alphanumeric (uppercase)
_PDB_ID_RE = re.compile(r"\b([0-9][A-Z0-9]{3})\b")

# Known-bad PDB-ID-like strings that are not PDB IDs (common false positives)
_PDB_ID_BLOCKLIST = frozenset({
    "1000", "2000", "3000", "4000", "5000",  # round numbers
    "1A1A", "2B2B",                            # obvious placeholders
})

# Recognised PyMOL command names (first word of a command line)
_PYMOL_VALID_CMDS = frozenset({
    "fetch", "load", "save", "delete", "reinitialize", "remove",
    "select", "deselect", "show", "hide", "enable", "disable",
    "color", "colour", "spectrum", "label", "util",
    "zoom", "center", "orient", "origin", "clip", "view",
    "align", "super", "fit", "rms", "rms_cur", "intra_fit", "pair_fit", "matchmaker",
    "bg_color", "bg_colour", "set", "get", "unset",
    "distance", "angle", "dihedral", "h_add", "h_bond",
    "ray", "png", "draw", "refresh", "rebuild", "recolor",
    "cartoon", "surface", "stick", "ribbon", "sphere", "dots",
    "symexp", "create", "extract", "split_states",
    "rotate", "translate", "move",
    "flag", "iterate", "alter", "iterate_state",
    "cmd", "pymol", "python", "as",
    "group", "ungroup", "order",
    "export_coords", "import_coords",
    "#",   # comment
})

# SMARTS pattern detector: backtick or single-quoted strings that look like SMARTS
_SMARTS_INLINE_RE = re.compile(r"`([^`]{3,80})`|'([A-Za-z0-9@#\[\]()=+\-\.%:/$\\*;!,~&|]{4,80})'")


def pdb_id_validity(output: str) -> tuple[int, int]:
    """PDB IDs mentioned in output are well-formed (4-char, digit-first, alphanumeric).

    Counts every PDB-ID-shaped token (not in blocklist); checks it matches the PDB format.
    False positives on numbers like '1ABC' are rare in chemistry responses.
    0/0 if no PDB-ID-shaped tokens are found.
    """
    candidates = [m.group(1) for m in _PDB_ID_RE.finditer(output.upper())
                  if m.group(1) not in _PDB_ID_BLOCKLIST]
    if not candidates:
        return 0, 0
    ok = sum(1 for c in candidates if c[0].isdigit() and c[1:].isalnum())
    return ok, len(candidates)


def pymol_syntax_check(output: str) -> tuple[int, int]:
    """PyMOL command blocks should use only recognised PyMOL commands.

    Returns (valid_command_lines, total_command_lines).
    0/0 if no PyMOL command blocks found.
    Uses _extract_all_blocks so bare ``` fences (typical for PyMOL scripts) are captured.
    """
    blocks = _extract_all_blocks(output)
    pymol_blocks = [b for b in blocks if _is_pymol_command_block(b)]
    if not pymol_blocks:
        return 0, 0
    ok = total = 0
    for block in pymol_blocks:
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            first_word = stripped.split()[0].lower().rstrip(",;")
            total += 1
            if first_word in _PYMOL_VALID_CMDS:
                ok += 1
    return ok, total


def smarts_validity(output: str) -> tuple[int, int]:
    """SMARTS patterns embedded in output (backtick or single-quoted) must parse with RDKit."""
    candidates = []
    for m in _SMARTS_INLINE_RE.finditer(output):
        s = m.group(1) or m.group(2)
        # Heuristic: only treat as SMARTS if it contains typical SMARTS syntax chars
        if any(c in s for c in ("[", "#", "@", "$", "~", "!")):
            candidates.append(s)
    if not candidates:
        return 0, 0
    ok = sum(1 for s in candidates if Chem.MolFromSmarts(s) is not None)
    return ok, len(candidates)


# Ordered metric registry: (id, label, fn, is_new)
METRICS = [
    ("smiles_validity",    "SMILES Validity",         smiles_validity,      False),
    ("tool_executability", "Tool Executability",       tool_executability,   False),
    ("numerical_fidelity", "Numerical Fidelity",       numerical_fidelity,   False),
    ("rounding_precision", "Rounding Precision",       rounding_precision,   True),
    ("code_then_quote",    "Code→Quote Fidelity",      code_then_quote,      True),
    ("refusal_accuracy",   "Refusal Accuracy",         refusal_accuracy,     True),
    ("qed_range",          "QED Range Check",          qed_range_check,      True),
    ("python_extended",    "Python Exec (extended)",   python_extended,      True),
    ("degeneration",       "Degeneration-free",        degeneration_score,   True),
    ("code_attempted",     "Code Attempted",           code_attempted,       True),
    # v4 harness: structural biology + medchem metrics
    ("pdb_id_validity",    "PDB ID Validity",          pdb_id_validity,      True),
    ("pymol_syntax",       "PyMOL Syntax Check",       pymol_syntax_check,   True),
    ("smarts_validity",    "SMARTS Validity",          smarts_validity,      True),
]


def score_all(output: str, expected: str = "", prompt: str = "") -> dict[str, tuple[int, int]]:
    scores: dict[str, tuple[int, int]] = {}
    for key, _, fn, _ in METRICS:
        if key == "refusal_accuracy":
            scores[key] = fn(output, expected)   # type: ignore[call-arg]
        elif key == "code_attempted":
            scores[key] = fn(output, prompt)     # type: ignore[call-arg]
        else:
            scores[key] = fn(output)             # type: ignore[call-arg]
    return scores


# ─── per-model eval loop ───────────────────────────────────────────────────────

def run_model_eval(
    model_cfg: dict,
    examples: list[dict],
    settings: dict,
    partial_file: Optional[Path] = None,
) -> dict:
    from tqdm import tqdm

    port      = settings["port"]
    timeout   = settings.get("query_timeout", 120)
    su_timeout = settings.get("server_startup_timeout", 200)

    model_path = str(PROJECT_ROOT / model_cfg["path"])
    print(f"\n{'─'*62}")
    print(f"  {model_cfg['display_name']}  ·  {model_cfg['path']}")
    print(f"{'─'*62}")

    if not Path(model_path).exists():
        print(f"  ⚠  Model path not found: {model_path}  — skipping.")
        return {"model_id": model_cfg["id"], "skipped": True, "reason": "path not found"}

    proc = _start_server(model_path, port)
    print(f"  ⏳ Waiting for server on :{port} (up to {su_timeout}s)…", end="", flush=True)
    ready = _wait_for_server(port, timeout=su_timeout)
    if not ready:
        _stop_server(proc)
        return {"model_id": model_cfg["id"], "skipped": True, "reason": "server startup timeout"}
    print(" ready.")

    monitor = ResourceMonitor(proc.pid)
    monitor.start()

    results  = []
    t_start  = time.time()
    errors   = 0

    pbar = tqdm(examples, desc=f"  {model_cfg['display_name'][:22]:<22}", unit="ex")
    for ex in pbar:
        messages = [m for m in ex.get("messages", []) if m["role"] in ("system", "user")]
        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": EVAL_SYSTEM})

        prompt   = next((m["content"] for m in messages if m["role"] == "user"), "")
        expected = next(
            (m["content"] for m in reversed(ex.get("messages", [])) if m["role"] == "assistant"),
            "",
        )

        output, latency = query_model(port, messages, timeout=timeout, model_path=model_path)
        if output.startswith("[error]"):
            errors += 1

        scores = score_all(output, expected, prompt)
        results.append({
            "prompt":   prompt,
            "expected": expected,
            "output":   output,
            "latency":  round(latency, 2),
            "class":    ex.get("class", ""),
            "scores":   scores,
        })

        # Live tqdm suffix
        smi_ok  = sum(r["scores"]["smiles_validity"][0]    for r in results)
        smi_tot = sum(r["scores"]["smiles_validity"][1]    for r in results)
        exe_ok  = sum(r["scores"]["tool_executability"][0] for r in results)
        exe_tot = sum(r["scores"]["tool_executability"][1] for r in results)
        pbar.set_postfix_str(
            f"SMILES {smi_ok}/{smi_tot}  Exec {exe_ok}/{exe_tot}"
            + (f"  err={errors}" if errors else ""),
            refresh=True,
        )

        # Write partial aggregate so the progress monitor can show live scores
        if partial_file is not None:
            try:
                partial_agg = {
                    key: (
                        sum(r["scores"][key][0] for r in results),
                        sum(r["scores"][key][1] for r in results),
                    )
                    for key, *_ in METRICS
                }
                partial_file.write_text(json.dumps({
                    "model_id": model_cfg["id"],
                    "n":        len(results),
                    "aggregate": partial_agg,
                }, default=str))
            except Exception:
                pass

    total_secs = time.time() - t_start

    # Aggregate first — so results are ready even if monitor.stop() raises
    agg: dict[str, tuple[int, int]] = {}
    for key, *_ in METRICS:
        agg[key] = (
            sum(r["scores"][key][0] for r in results),
            sum(r["scores"][key][1] for r in results),
        )

    latencies    = [r["latency"] for r in results if not r["output"].startswith("[error]")]
    word_count   = sum(len(r["output"].split()) for r in results)
    mean_latency = round(statistics.mean(latencies), 1) if latencies else 0
    tok_per_sec  = round(word_count / total_secs, 1) if total_secs else 0

    resource_stats = monitor.stop()
    _stop_server(proc)

    return {
        "model_id":      model_cfg["id"],
        "skipped":       False,
        "results":       results,
        "aggregate":     agg,
        "resource":      resource_stats,
        "mean_latency":  mean_latency,
        "tok_per_sec":   tok_per_sec,
        "total_min":     round(total_secs / 60, 1),
        "n":             len(results),
        "errors":        errors,
    }


# ─── SVG loss curve chart ──────────────────────────────────────────────────────

def _loss_curve_svg(model_configs: list[dict]) -> str:
    W, H, PL, PR, PT, PB = 560, 200, 55, 20, 20, 30
    CW = W - PL - PR
    CH = H - PT - PB
    MAX_ITER = 1500
    YMAX     = 0.60   # clip above this for readability

    axes = ""
    # Gridlines + Y labels
    for val in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        y = PT + CH * (1 - val / YMAX)
        axes += (
            f'<line x1="{PL}" y1="{y:.1f}" x2="{PL+CW}" y2="{y:.1f}" '
            f'stroke="#e8edf5" stroke-width="1"/>'
            f'<text x="{PL-6}" y="{y+4:.1f}" text-anchor="end" '
            f'fill="#9ca3af" font-size="9" font-family="monospace">{val:.1f}</text>'
        )
    # X labels
    for it in [0, 250, 500, 750, 1000, 1250, 1500]:
        x = PL + CW * (it / MAX_ITER)
        axes += (
            f'<text x="{x:.1f}" y="{H-PB+13:.1f}" text-anchor="middle" '
            f'fill="#9ca3af" font-size="9" font-family="monospace">{it}</text>'
        )
    # Border box
    axes += (
        f'<rect x="{PL}" y="{PT}" width="{CW}" height="{CH}" '
        f'fill="none" stroke="#d1d5db" stroke-width="1"/>'
    )

    lines = ""
    legend_items = []
    for cfg in model_configs:
        curve  = cfg.get("val_loss_curve", [])
        color  = cfg.get("color", "#888")
        dname  = cfg.get("display_name", "?")
        pts    = []
        for pair in curve:
            if pair[1] is None:
                continue
            it, loss = pair[0], pair[1]
            x = PL + CW * (it / MAX_ITER)
            y = PT + CH * (1 - min(float(loss), YMAX) / YMAX)
            pts.append(f"{x:.1f},{y:.1f}")
        if pts:
            dash = "4,3" if cfg.get("val_loss_curve_note", "").startswith("estimated") else "none"
            lines += (
                f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
                f'stroke-width="2" stroke-dasharray="{dash}" '
                f'stroke-linejoin="round" opacity="0.9"/>'
            )
        legend_items.append((color, dname))

    # Legend (bottom)
    legend = ""
    lx = PL
    for color, name in legend_items:
        legend += (
            f'<rect x="{lx}" y="{H+5}" width="12" height="10" rx="2" fill="{color}"/>'
            f'<text x="{lx+16}" y="{H+14}" fill="#374151" font-size="10" '
            f'font-family="sans-serif">{name}</text>'
        )
        lx += 110

    label = (
        f'<text x="{PL//2}" y="{PT + CH//2}" text-anchor="middle" '
        f'fill="#9ca3af" font-size="9" font-family="sans-serif" '
        f'transform="rotate(-90,{PL//2},{PT + CH//2})">Val Loss</text>'
        f'<text x="{PL + CW//2}" y="{H}" text-anchor="middle" '
        f'fill="#9ca3af" font-size="9" font-family="sans-serif">Iteration</text>'
    )

    total_h = H + 25
    return (
        f'<svg width="100%" viewBox="0 0 {W} {total_h}" '
        f'style="display:block;max-width:{W}px">'
        f'{axes}{lines}{legend}{label}'
        f'</svg>'
    )


def _loss_curve_plotly(model_configs: list[dict]) -> str:
    """Interactive Plotly loss curve; falls back to SVG if plotly unavailable."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return _loss_curve_svg(model_configs)

    fig = go.Figure()
    for cfg in model_configs:
        curve = cfg.get("val_loss_curve", [])
        pts   = [(p[0], p[1]) for p in curve if p[1] is not None]
        if not pts:
            continue
        iters, losses = zip(*pts)
        estimated = cfg.get("val_loss_curve_note", "").startswith("estimated")
        fig.add_trace(go.Scatter(
            x=list(iters),
            y=list(losses),
            name=cfg.get("display_name", "?"),
            mode="lines",
            line=dict(
                color=cfg.get("color", "#888"),
                dash="dot" if estimated else "solid",
                width=2.5,
            ),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Iter: %{x}<br>"
                "Val Loss: %{y:.4f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=60, r=20, t=30, b=50),
        xaxis=dict(title="Iteration", gridcolor="#e8edf5", linecolor="#d1d5db"),
        yaxis=dict(title="Val Loss", range=[0, 0.62], gridcolor="#e8edf5", linecolor="#d1d5db"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1,
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="'Segoe UI', system-ui, sans-serif", size=12, color="#374151"),
        hoverlabel=dict(bgcolor="white", bordercolor="#e2e8f0", font_size=12),
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={
            "displayModeBar": True,
            "responsive": True,
            "scrollZoom": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


# ─── training log parser ───────────────────────────────────────────────────────

def _parse_training_log(log_path: Path) -> dict[str, list]:
    """Extract per-iter metric curves from a training log file.

    Returns a dict with keys matching _TRAIN_METRIC_INFO keys, each a list of
    [iter, value] pairs.  Empty lists for metrics not found in the log.
    """
    curves: dict[str, list] = {k: [] for k, *_ in _TRAIN_METRIC_INFO}
    try:
        with log_path.open(errors="replace") as f:
            for raw in f:
                line = raw.strip()
                m = _TRAIN_LINE_RE.match(line)
                if m:
                    it = int(m.group(1))
                    curves["train_loss"].append([it, float(m.group(2))])
                    curves["learning_rate"].append([it, float(m.group(3))])
                    curves["itps"].append([it, float(m.group(4))])
                    curves["tokens_per_sec"].append([it, float(m.group(5))])
                    curves["trained_tokens"].append([it, int(m.group(6))])
                    curves["peak_mem"].append([it, float(m.group(7))])
                    continue
                m = _VAL_TOOK_RE.match(line)
                if m:
                    curves["val_took"].append([int(m.group(1)), float(m.group(2))])
    except (OSError, IOError):
        pass
    return curves


def _training_metrics_plotly(model_configs: list[dict]) -> str:
    """Interactive chart of training-log metrics with a native Plotly dropdown.

    Uses include_plotlyjs=False — the first chart's CDN script must already be
    in the page before this div is rendered.  Returns empty string if no log
    data is available for any model.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ""

    # Parse log files for every model that declares one
    log_curves: dict[str, dict] = {}
    for cfg in model_configs:
        lf = cfg.get("log_file")
        log_curves[cfg["id"]] = _parse_training_log(PROJECT_ROOT / lf) if lf else {}

    # Filter to metrics that have data for at least one model
    available = [
        (mkey, mlabel, munit)
        for mkey, mlabel, munit in _TRAIN_METRIC_INFO
        if any(log_curves[cfg["id"]].get(mkey) for cfg in model_configs)
    ]
    if not available:
        return ""

    fig = go.Figure()
    n_models  = len(model_configs)
    n_metrics = len(available)

    for mi, (mkey, mlabel, munit) in enumerate(available):
        for cfg in model_configs:
            pts = log_curves[cfg["id"]].get(mkey, [])
            x   = [p[0] for p in pts]
            y   = [p[1] for p in pts]
            hover_y = "%{y:.4g}" + (f" {munit}" if munit else "")
            fig.add_trace(go.Scatter(
                x=x, y=y,
                name=cfg.get("display_name", "?"),
                mode="lines",
                line=dict(color=cfg.get("color", "#888"), width=2),
                visible=(mi == 0),
                showlegend=(mi == 0),
                legendgroup=cfg["id"],
                hovertemplate=(
                    f"<b>%{{fullData.name}}</b><br>"
                    f"Iter: %{{x}}<br>"
                    f"{mlabel}: {hover_y}"
                    "<extra></extra>"
                ),
            ))

    def _vmask(sel: int) -> list:
        return [mi == sel for mi in range(n_metrics) for _ in range(n_models)]

    def _lgmask(sel: int) -> list:
        return [mi == sel for mi in range(n_metrics) for _ in range(n_models)]

    first_title = f"{available[0][1]} ({available[0][2]})" if available[0][2] else available[0][1]

    buttons = [
        dict(
            label=mlabel,
            method="update",
            args=[
                {"visible": _vmask(mi), "showlegend": _lgmask(mi)},
                {"yaxis.title.text": f"{mlabel} ({munit})" if munit else mlabel},
            ],
        )
        for mi, (mkey, mlabel, munit) in enumerate(available)
    ]

    fig.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=60, r=20, t=70, b=50),
        xaxis=dict(title="Iteration", gridcolor="#e8edf5", linecolor="#d1d5db"),
        yaxis=dict(title=first_title, gridcolor="#e8edf5", linecolor="#d1d5db"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1,
        ),
        updatemenus=[dict(
            type="dropdown",
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0, xanchor="left",
            y=1.22, yanchor="top",
            bgcolor="#ffffff",
            bordercolor="#d1d5db",
            borderwidth=1,
            font=dict(size=12, color="#374151"),
            pad=dict(r=10, t=2),
        )],
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="'Segoe UI', system-ui, sans-serif", size=12, color="#374151"),
        hoverlabel=dict(bgcolor="white", bordercolor="#e2e8f0", font_size=12),
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,   # already loaded by the val-loss chart
        config={
            "displayModeBar": True,
            "responsive": True,
            "scrollZoom": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


# ─── HTML report ───────────────────────────────────────────────────────────────

def _pct(ok: int, tot: int, na: str = "n/a") -> str:
    return f"{ok/tot:.0%}" if tot else na


def _per_example_agg(eval_result: dict) -> dict:
    """Recompute aggregate as per-example pass/fail scored over n_examples.

    Returns {metric_key: [passed, n_examples]} where:
      passed     = examples where ALL instances were correct (ok == tot, tot > 0)
      n_examples = total examples evaluated
      [0, 0]     = metric never triggered in any response (renders as N/A)

    Falls back to raw aggregate when per-example data is unavailable.
    """
    results = eval_result.get("results", [])
    n = eval_result.get("n", len(results))
    if not results:
        return eval_result.get("aggregate", {})

    attempted: dict[str, int] = {}
    passed: dict[str, int] = {}
    for r in results:
        for key, v in r.get("scores", {}).items():
            ok, tot = v[0], v[1]
            if tot > 0:
                attempted[key] = attempted.get(key, 0) + 1
                if ok == tot:
                    passed[key] = passed.get(key, 0) + 1

    out: dict[str, list] = {}
    for key in (eval_result.get("aggregate") or {}):
        att = attempted.get(key, 0)
        out[key] = [passed.get(key, 0), n] if att > 0 else [0, 0]
    return out


def _delta_html(r4_ok: int, r4_tot: int, r3_ok: int, r3_tot: int) -> str:
    if not r4_tot or not r3_tot:
        return '<span style="color:#9ca3af">—</span>'
    d = (r4_ok / r4_tot) - (r3_ok / r3_tot)
    if abs(d) < 0.005:
        return '<span style="color:#9ca3af">±0%</span>'
    color = "#10b981" if d > 0 else "#ef4444"
    sign  = "+" if d > 0 else ""
    return f'<span style="color:{color};font-weight:600">{sign}{d:.0%}</span>'


def _model_card_html(cfg: dict, eval_result: Optional[dict]) -> str:
    color   = cfg.get("color", "#888")
    is_base = cfg.get("is_baseline", False)
    badge   = ' ★ baseline' if is_base else ''

    if eval_result and not eval_result.get("skipped"):
        agg = _per_example_agg(eval_result)
        # Mean score across applicable metrics (per-example pass rate)
        scores = [ok/tot for key, _, _, _ in METRICS
                  for ok, tot in [agg[key]] if tot > 0]
        overall = f"{statistics.mean(scores):.0%}" if scores else "n/a"
    else:
        overall = "pending" if not eval_result else "skipped"

    return f"""
    <div class="model-card" style="border-top:4px solid {color}">
      <div class="card-name" style="color:{color}">{cfg['display_name']}{badge}</div>
      <div class="card-sub">{cfg.get('subtitle','')}</div>
      <div class="card-score">{overall}</div>
      <div class="card-label">overall score</div>
      <div class="card-meta">
        val loss <b>{cfg.get('best_val_loss','?')}</b>
        &nbsp;·&nbsp;
        {cfg.get('iters_actual') or '?'}/{cfg.get('iters_configured','?')} iters
        &nbsp;·&nbsp;
        {cfg.get('training_examples','?')} examples
      </div>
    </div>"""


def _profile_row(label: str, values: list[str]) -> str:
    cells = "".join(f"<td>{v}</td>" for v in values)
    return f"<tr><th>{label}</th>{cells}</tr>"


def _model_profile_table(model_configs: list[dict], eval_results: dict) -> str:
    ids = [c["id"] for c in model_configs]

    def vals(key: str, default: str = "—") -> list[str]:
        return [str(c.get(key, default)) for c in model_configs]

    def res_vals(fn) -> list[str]:
        out = []
        for c in model_configs:
            er = eval_results.get(c["id"])
            if er and not er.get("skipped"):
                out.append(fn(er))
            else:
                out.append("—")
        return out

    header = "<tr><th>Attribute</th>" + "".join(
        f'<th style="color:{c.get("color","#888")}">{c["display_name"]}</th>'
        for c in model_configs
    ) + "</tr>"

    rows = "".join([
        "<tr class='section-row'><td colspan='5'>Architecture</td></tr>",
        _profile_row("Base model",          vals("base_model")),
        _profile_row("Parameters (total)",  vals("params_total")),
        _profile_row("Parameters (trained)",vals("params_trainable")),
        _profile_row("Trainable %",         vals("trainable_pct")),
        _profile_row("LoRA type",           vals("lora_type")),
        _profile_row("Rank / scale",        [f"{c.get('lora_rank','?')} / {c.get('lora_scale','?')}" for c in model_configs]),
        _profile_row("Adapted layers",      [f"{c.get('adapted_layers','?')} / {c.get('total_layers','?')}" for c in model_configs]),

        "<tr class='section-row'><td colspan='5'>Training</td></tr>",
        _profile_row("Learning rate",       vals("learning_rate")),
        _profile_row("LR schedule",         vals("lr_schedule")),
        _profile_row("Batch size",          vals("batch_size")),
        _profile_row("Seed",                vals("seed")),
        _profile_row("Iters configured",    vals("iters_configured")),
        _profile_row("Iters actual",        [f"{c.get('iters_actual') or '(ongoing)'}" for c in model_configs]),
        _profile_row("Training examples",   vals("training_examples")),
        _profile_row("Behaviour classes",   vals("behaviour_classes")),
        _profile_row("Duration",            vals("training_duration")),

        "<tr class='section-row'><td colspan='5'>Training outcomes</td></tr>",
        _profile_row("Best val loss",       [f"<b>{c.get('best_val_loss','?')}</b> @ iter {c.get('best_val_iter','?')}" for c in model_configs]),
        _profile_row("Final train loss",    [str(c.get("final_train_loss") or "(pending)") for c in model_configs]),
        _profile_row("Peak train mem (GB)", vals("peak_memory_gb")),
        _profile_row("Fused model size (GB)", vals("fused_size_gb")),

        "<tr class='section-row'><td colspan='5'>Corpus</td></tr>",
        _profile_row("Corpus chunks",       vals("corpus_chunks")),
        _profile_row("Corpus notes",        vals("corpus_note")),

        "<tr class='section-row'><td colspan='5'>Eval runtime</td></tr>",
        _profile_row("Mean latency (s)",    res_vals(lambda e: str(e["mean_latency"]))),
        _profile_row("Throughput (tok/s)",  res_vals(lambda e: str(e["tok_per_sec"]))),
        _profile_row("Eval duration (min)", res_vals(lambda e: str(e["total_min"]))),
        _profile_row("Server CPU mean %",   res_vals(lambda e: str(e["resource"].get("cpu_mean_pct", "—")))),
        _profile_row("Server CPU peak %",   res_vals(lambda e: str(e["resource"].get("cpu_peak_pct", "—")))),
        _profile_row("Server RSS mean (GB)",res_vals(lambda e: str(e["resource"].get("rss_mean_gb", "—")))),
        _profile_row("Server RSS peak (GB)",res_vals(lambda e: str(e["resource"].get("rss_peak_gb", "—")))),
        _profile_row("Eval errors",         res_vals(lambda e: str(e.get("errors", "—")))),
    ])

    return f"<table class='profile-table'><thead>{header}</thead><tbody>{rows}</tbody></table>"


def _metric_table(model_configs: list[dict], eval_results: dict, baseline_id: str) -> str:
    """Metric table with pre-computed delta matrix and N/A awareness.

    Returns a <script> data block followed by the <table>.  The script block
    sets window._cseDeltaData / _cseDefaultCmp / _cseDefaultBase so the JS
    controls in html_report() can drive the Δ column and N/A toggle without a
    server round-trip.
    """
    def _round_num(cfg: dict) -> int:
        try:
            return int(cfg.get("id", "").replace("round", ""))
        except ValueError:
            return 0

    model_round = {cfg["id"]: _round_num(cfg) for cfg in model_configs}

    # Per-example scores for models that have results
    model_scores: dict[str, dict] = {
        cfg["id"]: _per_example_agg(eval_results[cfg["id"]])
        for cfg in model_configs
        if cfg["id"] in eval_results and not eval_results[cfg["id"]].get("skipped")
    }

    # Pre-compute delta for every (compare, base) pair
    delta_data: dict[str, dict] = {}
    for cmp_cfg in model_configs:
        for base_cfg in model_configs:
            if cmp_cfg["id"] == base_cfg["id"]:
                continue
            pair_key = f"{cmp_cfg['id']}|{base_cfg['id']}"
            deltas: dict[str, Optional[float]] = {}
            for mkey, *_ in METRICS:
                ca = model_scores.get(cmp_cfg["id"])
                ba = model_scores.get(base_cfg["id"])
                if ca and ba:
                    ok_c, tot_c = ca[mkey]
                    ok_b, tot_b = ba[mkey]
                    deltas[mkey] = round((ok_c / tot_c) - (ok_b / tot_b), 4) if (tot_c and tot_b) else None
                else:
                    deltas[mkey] = None
            delta_data[pair_key] = deltas

    # Default compare = last model with eval results; default base = baseline_id
    evaled_ids = [cfg["id"] for cfg in model_configs if cfg["id"] in model_scores]
    default_cmp  = evaled_ids[-1] if evaled_ids else baseline_id
    default_base = baseline_id

    # ── Header ──────────────────────────────────────────────────────────────────
    header = (
        "<tr><th>Metric</th>"
        + "".join(
            f'<th style="color:{c.get("color","#888")}" data-model-id="{c["id"]}"'
            f' data-model-round="{model_round.get(c["id"], 0)}">'
            + c["display_name"]
            + (" ★" if c.get("is_baseline") else "")
            + "</th>"
            for c in model_configs
        )
        + '<th id="delta-col-header">Δ vs baseline</th></tr>'
    )

    # ── Rows ────────────────────────────────────────────────────────────────────
    R4_ROUND = 4   # round that introduced the new metrics
    rows = ""
    for mkey, label, _, is_new in METRICS:
        introduced = R4_ROUND if is_new else 0
        tag = " <span class='r4-tag'>R4</span>" if is_new else ""

        # Score cells (one per model)
        cells = ""
        for cfg in model_configs:
            mid    = cfg["id"]
            mround = model_round.get(mid, 0)
            er     = eval_results.get(mid)
            if er and not er.get("skipped"):
                ok, tot = _per_example_agg(er)[mkey]
                score_str = _pct(ok, tot)
            else:
                score_str = "—"  # —

            # Add N/A toggle spans for pre-R4 models on R4 metrics
            if is_new and 0 < mround < R4_ROUND:
                cell_content = (
                    f'<span class="score-val">{score_str}</span>'
                    f'<span class="na-badge" style="display:none"'
                    f' title="Model was trained before this metric was introduced">N/A</span>'
                )
            else:
                cell_content = score_str

            cells += (
                f'<td data-model-id="{mid}" data-model-round="{mround}"'
                f' data-metric="{mkey}" data-introduced="{introduced}">{cell_content}</td>'
            )

        # Delta cell — one span per (cmp, base) pair; JS shows/hides
        delta_spans = ""
        for cmp_cfg in model_configs:
            for base_cfg in model_configs:
                if cmp_cfg["id"] == base_cfg["id"]:
                    continue
                pair_key = f"{cmp_cfg['id']}|{base_cfg['id']}"
                d = delta_data.get(pair_key, {}).get(mkey)
                is_default = (cmp_cfg["id"] == default_cmp and base_cfg["id"] == default_base)
                vis = "" if is_default else "display:none"
                if d is None:
                    inner = '<span style="color:#9ca3af">—</span>'
                elif abs(d) < 0.005:
                    inner = '<span style="color:#9ca3af">±0%</span>'
                else:
                    color = "#10b981" if d > 0 else "#ef4444"
                    sign  = "+" if d > 0 else ""
                    inner = f'<span style="color:{color};font-weight:600">{sign}{d:.0%}</span>'
                delta_spans += (
                    f'<span class="delta-span" data-cmp="{cmp_cfg["id"]}"'
                    f' data-base="{base_cfg["id"]}" style="{vis}">{inner}</span>'
                )

        rows += (
            f'<tr data-metric="{mkey}" data-introduced="{introduced}">'
            f'<td>{label}{tag}</td>{cells}'
            f'<td class="delta-cell">{delta_spans}</td></tr>'
        )

    # Embed delta data for JS (plain <script> so braces are safe)
    data_script = (
        "<script>"
        f"window._cseDeltaData={json.dumps(delta_data)};"
        f'window._cseDefaultCmp="{default_cmp}";'
        f'window._cseDefaultBase="{default_base}";'
        "</script>"
    )

    return (
        data_script
        + f"<table class='metric-table' id='metric-table'>"
        f"<thead>{header}</thead><tbody>{rows}</tbody></table>"
    )


def _per_example_rows(model_configs: list[dict], eval_results: dict, n: int) -> str:
    # Collect all examples by index
    per_model: dict[str, list] = {}
    n_examples = 0
    for cfg in model_configs:
        er = eval_results.get(cfg["id"])
        if er and not er.get("skipped"):
            per_model[cfg["id"]] = er["results"]
            n_examples = max(n_examples, len(er["results"]))

    if not per_model:
        return "<p>No results available.</p>"

    first_model_results = next(iter(per_model.values()))
    rows = ""
    for i in range(min(n, n_examples)):
        prompt = first_model_results[i]["prompt"] if i < len(first_model_results) else "—"
        prompt_trunc = (prompt[:90] + "…") if len(prompt) > 90 else prompt
        prompt_trunc = prompt_trunc.replace("<", "&lt;").replace(">", "&gt;")

        score_cells = ""
        responses   = ""
        for cfg in model_configs:
            mid = cfg["id"]
            color = cfg.get("color", "#888")
            if mid not in per_model or i >= len(per_model[mid]):
                score_cells += "<td>—</td>"
                continue
            r = per_model[mid][i]
            ok_vals = []
            for key, _, _, _ in METRICS:
                ok, tot = r["scores"][key]
                if tot:
                    ok_vals.append((key, ok, tot))
            mini = " ".join(
                f'<span title="{k}" style="color:{"#10b981" if ok==tot else "#ef4444"}">'
                f'{"✓" if ok==tot else "✗"}</span>'
                for k, ok, tot in ok_vals[:4]
            )
            score_cells += f"<td>{mini}</td>"

            out_esc = r["output"].replace("<", "&lt;").replace(">", "&gt;")
            responses += (
                f'<div style="margin:.5rem 0">'
                f'<span style="color:{color};font-weight:600">{cfg["display_name"]}</span>'
                f' <span style="color:#9ca3af;font-size:.75rem">{r["latency"]}s</span>'
                f'<pre style="white-space:pre-wrap;background:#f8fafc;border:1px solid #e2e8f0;'
                f'border-radius:6px;padding:.5rem;margin:.3rem 0;font-size:.75rem">'
                f'{out_esc[:800]}{"…" if len(out_esc)>800 else ""}</pre>'
                f'</div>'
            )

        rows += (
            f"<tr>"
            f"<td>{i+1}</td>"
            f'<td class="q">{prompt_trunc}</td>'
            f"{score_cells}"
            f"<td>"
            f'<details><summary style="cursor:pointer;color:#467FF7;font-size:.8rem">responses</summary>'
            f'<div style="min-width:400px">{responses}</div>'
            f"</details>"
            f"</td>"
            f"</tr>"
        )

    return rows


def html_report(
    model_configs: list[dict],
    eval_results: dict,
    settings: dict,
    generated_at: str,
) -> str:
    baseline_id = settings.get("baseline", "round3")
    n           = settings.get("n_examples", 100)

    art_html   = _ART.replace("<", "&lt;").replace(">", "&gt;")
    cards      = "".join(
        _model_card_html(c, eval_results.get(c["id"])) for c in model_configs
    )
    loss_chart      = _loss_curve_plotly(model_configs)
    training_chart  = _training_metrics_plotly(model_configs)
    profile         = _model_profile_table(model_configs, eval_results)
    metric_t   = _metric_table(model_configs, eval_results, baseline_id)
    ex_cols    = "".join(
        f'<th style="color:{c.get("color","#888")}">{c["display_name"][:8]}</th>'
        for c in model_configs
    )
    ex_rows    = _per_example_rows(model_configs, eval_results, n)

    # Metric comparison controls (compare/baseline selectors + N/A toggle)
    _evaled = [
        (c["id"], c["display_name"]) for c in model_configs
        if c["id"] in eval_results and not eval_results[c["id"]].get("skipped")
    ]
    _all = [(c["id"], c["display_name"]) for c in model_configs]
    _default_cmp  = _evaled[-1][0] if _evaled else baseline_id
    _default_base = baseline_id

    def _opt(val: str, label: str, sel_val: str) -> str:
        s = " selected" if val == sel_val else ""
        return f'<option value="{val}"{s}>{label}</option>'

    cmp_opts  = "".join(_opt(v, l, _default_cmp)  for v, l in _evaled)
    base_opts = "".join(_opt(v, l, _default_base) for v, l in _all)
    metric_controls = (
        '<div class="metric-controls">'
        f'<label>Compare: <select id="cmp-select">{cmp_opts}</select></label>'
        f'<label>vs.&nbsp;Baseline: <select id="base-select">{base_opts}</select></label>'
        '<label class="toggle-label">'
        '<input type="checkbox" id="na-toggle">&nbsp;'
        "Show N/A for metrics models weren't trained on"
        "</label>"
        "</div>"
    )

    # JS block — separate string so curly braces don't clash with the f-string below
    if training_chart:
        training_metrics_section = (
            "<h2>Training Metrics</h2>"
            '<div class="chart-box">'
            + training_chart
            + '<div class="chart-note">Parsed from training log files. '
            "Use the dropdown to switch between metrics. "
            "Models without a <code>log_file</code> in models.yaml are not shown."
            "</div></div>"
        )
    else:
        training_metrics_section = ""

    js_block = """<script>
(function () {
  function updateDelta() {
    var cmpId  = document.getElementById('cmp-select').value;
    var baseId = document.getElementById('base-select').value;
    document.querySelectorAll('.delta-cell .delta-span').forEach(function (s) {
      s.style.display = 'none';
    });
    document.querySelectorAll(
      '.delta-cell .delta-span[data-cmp="' + cmpId + '"][data-base="' + baseId + '"]'
    ).forEach(function (s) { s.style.display = ''; });
    var cmpOpt  = document.querySelector('#cmp-select option[value="'  + cmpId  + '"]');
    var baseOpt = document.querySelector('#base-select option[value="' + baseId + '"]');
    var dh = document.getElementById('delta-col-header');
    if (dh && cmpOpt && baseOpt)
      dh.textContent = 'Δ ' + cmpOpt.text + ' vs ' + baseOpt.text;
  }

  function updateNA() {
    var show = document.getElementById('na-toggle').checked;
    document.querySelectorAll('#metric-table td[data-introduced="4"]').forEach(function (td) {
      var mround = parseInt(td.getAttribute('data-model-round'), 10);
      if (mround > 0 && mround < 4) {
        var sv = td.querySelector('.score-val');
        var na = td.querySelector('.na-badge');
        if (sv) sv.style.display = show ? 'none' : '';
        if (na) na.style.display = show ? ''     : 'none';
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    var cmpSel  = document.getElementById('cmp-select');
    var baseSel = document.getElementById('base-select');
    var naTog   = document.getElementById('na-toggle');
    if (window._cseDefaultCmp  && cmpSel)  cmpSel.value  = window._cseDefaultCmp;
    if (window._cseDefaultBase && baseSel) baseSel.value = window._cseDefaultBase;
    if (cmpSel)  cmpSel.addEventListener('change',  updateDelta);
    if (baseSel) baseSel.addEventListener('change', updateDelta);
    if (naTog)   naTog.addEventListener('change',   updateNA);
    updateDelta();
  });
})();
</script>"""

    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>ChemSage Comparative Eval — {generated_at}</title>
        <style>
          :root {{
            --navy:#1C244B; --accent:#467FF7; --bg:#f0f4fb;
            --card:#fff; --border:#e2e8f0; --text:#1a1a2e;
          }}
          *{{box-sizing:border-box;margin:0;padding:0}}
          body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);
                color:var(--text);padding:2rem 1.5rem;max-width:1280px;margin:auto}}
          a{{color:var(--accent);text-decoration:none}}
          a:hover{{text-decoration:underline}}
          /* header */
          .header{{background:var(--navy);color:#fff;border-radius:16px;padding:2rem 2.5rem;
                   margin-bottom:2rem}}
          .art{{font-family:monospace;font-size:.65rem;line-height:1.2;color:#93c5fd;
                white-space:pre;margin-bottom:1rem}}
          .header h1{{font-size:1.4rem;font-weight:700;margin-bottom:.3rem}}
          .header .meta{{font-size:.8rem;color:#94a3b8}}
          /* model cards */
          .cards-row{{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:2rem}}
          .model-card{{background:var(--card);border-radius:14px;padding:1.2rem 1.5rem;
                       flex:1;min-width:200px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
          .card-name{{font-size:.9rem;font-weight:700;margin-bottom:.15rem}}
          .card-sub{{font-size:.72rem;color:#6b7280;margin-bottom:.8rem}}
          .card-score{{font-size:2.2rem;font-weight:800;color:var(--accent)}}
          .card-label{{font-size:.7rem;color:#9ca3af;margin-bottom:.5rem}}
          .card-meta{{font-size:.72rem;color:#6b7280}}
          /* sections */
          h2{{font-size:1.1rem;color:var(--navy);margin:2rem 0 .8rem}}
          .section-note{{font-size:.75rem;color:#9ca3af;margin-top:.4rem}}
          /* tables */
          table{{width:100%;border-collapse:collapse;background:var(--card);
                 border-radius:12px;overflow:hidden;
                 box-shadow:0 2px 8px rgba(0,0,0,.05);margin-bottom:1.5rem}}
          th{{background:var(--navy);color:#fff;text-align:left;
              padding:.6rem 1rem;font-size:.8rem;font-weight:600}}
          td{{padding:.5rem 1rem;border-bottom:1px solid var(--border);font-size:.8rem}}
          tr:last-child td{{border-bottom:none}}
          tr:nth-child(even){{background:#f8fafc}}
          td.q{{max-width:320px;word-break:break-word}}
          .profile-table th:first-child{{width:200px}}
          .profile-table .section-row td{{
            background:#eef2f9;color:var(--navy);font-weight:600;font-size:.75rem;
            padding:.35rem 1rem;border-bottom:2px solid var(--border)
          }}
          .metric-table td:first-child{{font-weight:500}}
          .r4-tag{{background:#dcfce7;color:#166534;font-size:.65rem;
                   padding:.1rem .35rem;border-radius:4px;font-weight:600;vertical-align:middle}}
          /* loss chart */
          .chart-box{{background:var(--card);border-radius:12px;padding:1.5rem;
                      box-shadow:0 2px 8px rgba(0,0,0,.05);margin-bottom:1.5rem}}
          .chart-note{{font-size:.7rem;color:#9ca3af;margin-top:.5rem;text-align:center}}
          /* metric controls */
          .metric-controls{{display:flex;gap:1.2rem;align-items:center;
                             margin-bottom:.8rem;flex-wrap:wrap;
                             background:var(--card);border-radius:10px;padding:.8rem 1rem;
                             box-shadow:0 1px 4px rgba(0,0,0,.04)}}
          .metric-controls label{{font-size:.8rem;color:#374151;
                                   display:flex;align-items:center;gap:.4rem}}
          .metric-controls select{{font-size:.8rem;border:1px solid #d1d5db;
                                    border-radius:6px;padding:.25rem .5rem;
                                    background:#fff;color:#374151;cursor:pointer}}
          .metric-controls select:focus{{outline:2px solid var(--accent);outline-offset:1px}}
          .toggle-label{{cursor:pointer;user-select:none}}
          .na-badge{{background:#fef3c7;color:#92400e;font-size:.65rem;
                     padding:.1rem .4rem;border-radius:4px;font-weight:700;
                     letter-spacing:.02em}}
          .delta-cell{{min-width:90px}}
          /* footer */
          .footer{{margin-top:2.5rem;font-size:.72rem;color:#9ca3af;text-align:center;
                   border-top:1px solid var(--border);padding-top:1rem}}
        </style>
        </head>
        <body>

        <!-- ── Header ── -->
        <div class="header">
          <div class="art">{art_html}</div>
          <h1>ChemSage — Multi-Round Comparative Evaluation</h1>
          <div class="meta">
            {generated_at}
            &nbsp;·&nbsp; {n} shared test examples
            &nbsp;·&nbsp; 13 metrics
            &nbsp;·&nbsp;
            <a href="https://github.com/bellcheddar/ChemSage" style="color:#93c5fd">GitHub</a>
            &nbsp;·&nbsp;
            <a href="https://marcdeller.com" style="color:#93c5fd">marcdeller.com</a>
            &nbsp;·&nbsp;
            <a href="https://marcdeller.com/chemsage" style="color:#93c5fd">marcdeller.com/chemsage</a>
            &nbsp;·&nbsp;
            <a href="mailto:marc@marcdeller.com" style="color:#93c5fd">marc@marcdeller.com</a>
          </div>
        </div>

        <!-- ── Summary cards ── -->
        <h2>Model Overview</h2>
        <div class="cards-row">{cards}</div>

        <!-- ── Loss curves ── -->
        <h2>Validation Loss Curves</h2>
        <div class="chart-box">
          {loss_chart}
          <div class="chart-note">
            Y-axis clipped at 0.62 for readability — initial values can reach 2.4–3.2.
            Dotted line = estimated (no log file). Hover for exact values.
          </div>
        </div>

        <!-- ── Training metrics ── -->
        {training_metrics_section}

        <!-- ── Model profiles ── -->
        <h2>Model Profiles</h2>
        {profile}

        <!-- ── Metric comparison ── -->
        <h2>Metric Comparison</h2>
        <p class="section-note" style="margin-bottom:.6rem">
          <span class="r4-tag">R4</span> = metric introduced in Round 4 eval harness.
          Use the controls below to change the Δ comparison pair, or toggle N/A for
          metrics models weren't trained on.
        </p>
        {metric_controls}
        {metric_t}

        <!-- ── Per-example table ── -->
        <h2>Per-Example Results ({n} examples)</h2>
        <p class="section-note" style="margin-bottom:.8rem">
          Mini scores show first 4 metrics: ✓ = full credit, ✗ = partial or zero.
          Click <em>responses</em> to expand all model outputs for that example.
        </p>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Prompt</th>
              {ex_cols}
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>{ex_rows}</tbody>
        </table>

        <div class="footer">
          Generated by ChemSage comparative eval harness &nbsp;·&nbsp;
          Marc C. Deller, D.Phil. &nbsp;·&nbsp;
          <a href="https://github.com/bellcheddar/ChemSage">GitHub</a>
          &nbsp;·&nbsp;
          <a href="https://marcdeller.com">marcdeller.com</a>
          &nbsp;·&nbsp;
          <a href="https://marcdeller.com/chemsage">marcdeller.com/chemsage</a>
          &nbsp;·&nbsp;
          <a href="mailto:marc@marcdeller.com">marc@marcdeller.com</a>
        </div>
        {js_block}
        </body>
        </html>
    """)


# ─── Markdown report ───────────────────────────────────────────────────────────

def md_report(
    model_configs: list[dict],
    eval_results: dict,
    settings: dict,
    generated_at: str,
) -> str:
    baseline_id = settings.get("baseline", "round3")
    n           = settings.get("n_examples", 100)

    lines = [
        "# ChemSage — Multi-Round Comparative Evaluation",
        "",
        f"*{generated_at} · {n} shared test examples · 13 metrics*  ",
        "*[marcdeller.com](https://marcdeller.com) · [marc@marcdeller.com](mailto:marc@marcdeller.com)*",
        "",
        "---",
        "",
        "## Overview",
        "",
    ]

    # Summary table
    header = "| Model | Overall | SMILES | Exec | Fidelity | Best Val Loss |"
    sep    = "|---|---|---|---|---|---|"
    lines += [header, sep]
    for cfg in model_configs:
        er  = eval_results.get(cfg["id"])
        mid = cfg["id"]
        if er and not er.get("skipped"):
            agg     = _per_example_agg(er)
            scores  = [ok/tot for key, _, _, _ in METRICS for ok, tot in [agg[key]] if tot]
            overall = f"{statistics.mean(scores):.0%}" if scores else "n/a"
            smi     = _pct(*agg["smiles_validity"])
            exe     = _pct(*agg["tool_executability"])
            fid     = _pct(*agg["numerical_fidelity"])
        else:
            overall = "—"
            smi = exe = fid = "—"
        star = " ★" if cfg.get("is_baseline") else ""
        bvl  = cfg.get("best_val_loss", "?")
        name = f"**{cfg['display_name']}{star}**" if cfg.get("is_baseline") else cfg["display_name"]
        lines.append(f"| {name} | {overall} | {smi} | {exe} | {fid} | {bvl} |")

    lines += ["", "---", "", "## All Metrics", ""]
    metric_header = "| Metric | " + " | ".join(c["display_name"] for c in model_configs) + " | Δ vs baseline |"
    metric_sep    = "|---|" + "---|" * (len(model_configs) + 1)
    lines += [metric_header, metric_sep]

    for key, label, _, r4_new in METRICS:
        tag  = " *(R4)*" if r4_new else ""
        row  = f"| {label}{tag} |"
        r3_agg = None
        for cfg in model_configs:
            er = eval_results.get(cfg["id"])
            if er and not er.get("skipped"):
                ok, tot = _per_example_agg(er)[key]
                row += f" {_pct(ok, tot)} |"
                if cfg["id"] == baseline_id:
                    r3_agg = (ok, tot)
            else:
                row += " — |"
        r4_er = eval_results.get("round4")
        if r4_er and not r4_er.get("skipped") and r3_agg:
            ok4, tot4 = _per_example_agg(r4_er)[key]
            if tot4 and r3_agg[1]:
                d = (ok4 / tot4) - (r3_agg[0] / r3_agg[1])
                row += f' {"+" if d >= 0 else ""}{d:.0%} |'
            else:
                row += " — |"
        else:
            row += " — |"
        lines.append(row)

    lines += ["", "---", "", "## Model Profiles", ""]

    PROFILE_FIELDS = [
        ("Base model",           "base_model"),
        ("Parameters (total)",   "params_total"),
        ("Trainable",            "params_trainable"),
        ("LoRA rank / scale",    None),
        ("Adapted layers",       None),
        ("Training examples",    "training_examples"),
        ("Behaviour classes",    "behaviour_classes"),
        ("Iters (actual)",       "iters_actual"),
        ("Best val loss",        "best_val_loss"),
        ("Final train loss",     "final_train_loss"),
        ("Peak mem (train, GB)", "peak_memory_gb"),
        ("Training duration",    "training_duration"),
    ]

    p_header = "| | " + " | ".join(c["display_name"] for c in model_configs) + " |"
    p_sep    = "|---|" + "---|" * len(model_configs)
    lines   += [p_header, p_sep]

    for label, field in PROFILE_FIELDS:
        if field is None:
            if label == "LoRA rank / scale":
                vals = [f'{c.get("lora_rank","?")} / {c.get("lora_scale","?")}' for c in model_configs]
            else:
                vals = [f'{c.get("adapted_layers","?")} / {c.get("total_layers","?")}' for c in model_configs]
        else:
            vals = [str(c.get(field, "—")) for c in model_configs]
        lines.append("| " + label + " | " + " | ".join(vals) + " |")

    lines += ["", "---", "", "## Eval Runtime Stats", ""]
    rt_header = "| | " + " | ".join(c["display_name"] for c in model_configs) + " |"
    rt_sep    = "|---|" + "---|" * len(model_configs)
    lines    += [rt_header, rt_sep]

    RT_FIELDS = [
        ("Mean latency (s)",   lambda e: str(e["mean_latency"])),
        ("Throughput (tok/s)", lambda e: str(e["tok_per_sec"])),
        ("Eval time (min)",    lambda e: str(e["total_min"])),
        ("Server CPU mean %",  lambda e: str(e["resource"].get("cpu_mean_pct", "—"))),
        ("Server RSS peak GB", lambda e: str(e["resource"].get("rss_peak_gb", "—"))),
    ]
    for label, fn in RT_FIELDS:
        row = f"| {label} |"
        for cfg in model_configs:
            er = eval_results.get(cfg["id"])
            if er and not er.get("skipped"):
                row += f" {fn(er)} |"
            else:
                row += " — |"
        lines.append(row)

    lines += [
        "",
        "---",
        "",
        "*Generated by ChemSage comparative eval harness — "
        "[marcdeller.com](https://marcdeller.com)*",
    ]

    return "\n".join(lines)


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ChemSage multi-round comparative eval")
    ap.add_argument("--config",  type=Path,
                    default=Path(__file__).parent / "models.yaml",
                    help="Path to models.yaml")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "results",
                    help="Output directory")
    ap.add_argument("--resume",  action="store_true",
                    help="Load cached raw_results.json and skip already-run models")
    ap.add_argument("--models",  nargs="*",
                    help="Subset of model ids to run (default: all)")
    ap.add_argument("--limit",   type=int,
                    help="Evaluate only the first N examples (smoke-test)")
    args = ap.parse_args()

    # Load config
    cfg_raw  = yaml.safe_load(args.config.read_text())
    settings = cfg_raw["settings"]
    all_cfgs = cfg_raw["models"]

    # Filter to requested subset
    if args.models:
        all_cfgs = [c for c in all_cfgs if c["id"] in args.models]
        if not all_cfgs:
            raise SystemExit(f"No models matched: {args.models}")

    # Resolve test file from project root
    test_file = PROJECT_ROOT / settings["test_file"]
    if not test_file.exists():
        raise SystemExit(f"Test file not found: {test_file}")

    n = args.limit if args.limit else settings["n_examples"]
    examples = sample_examples(test_file, n, settings["sample_seed"])
    print(f"Sampled {len(examples)} examples (seed={settings['sample_seed']})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_file   = args.out_dir / "raw_results.json"
    partial_file = args.out_dir / "partial_results.json"

    # Load cached results if resuming
    eval_results: dict = {}
    if args.resume and cache_file.exists():
        eval_results = json.loads(cache_file.read_text())
        print(f"Loaded cached results for: {list(eval_results.keys())}")

    # Run each model
    for cfg in all_cfgs:
        mid = cfg["id"]
        if mid in eval_results:
            print(f"  ↩  Skipping {cfg['display_name']} (cached)")
            continue
        result = run_model_eval(cfg, examples, settings, partial_file=partial_file)
        eval_results[mid] = result
        # Save after each model so resume works on failure
        cache_file.write_text(json.dumps(eval_results, indent=2, default=str))
        partial_file.unlink(missing_ok=True)   # clean up partial now that full result is saved

    # Generate reports
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M")

    html_path = args.out_dir / f"compare_{timestamp}.html"
    md_path   = args.out_dir / f"compare_{timestamp}.md"

    html_str = html_report(all_cfgs, eval_results, settings, generated_at)
    md_str   = md_report(all_cfgs, eval_results, settings, generated_at)

    html_path.write_text(html_str)
    md_path.write_text(md_str)

    print(f"\n{'═'*62}")
    print(f"  HTML report → {html_path}")
    print(f"  MD report   → {md_path}")
    print(f"  Raw results → {cache_file}")
    print(f"{'═'*62}\n")

    # Terminal summary
    baseline_id = settings.get("baseline", "round3")
    print(f"\n{'─'*62}")
    print(f"  {'Model':<20} {'SMILES':>8} {'Exec':>8} {'Fidelity':>10} {'Val Loss':>10}")
    print(f"{'─'*62}")
    for cfg in all_cfgs:
        er  = eval_results.get(cfg["id"])
        star = " ★" if cfg.get("is_baseline") else "  "
        if er and not er.get("skipped"):
            agg = _per_example_agg(er)
            smi = _pct(*agg["smiles_validity"])
            exe = _pct(*agg["tool_executability"])
            fid = _pct(*agg["numerical_fidelity"])
        else:
            smi = exe = fid = "—"
        bvl = cfg.get("best_val_loss", "?")
        print(f"  {cfg['display_name'] + star:<20} {smi:>8} {exe:>8} {fid:>10} {str(bvl):>10}")
    print(f"{'─'*62}\n")


if __name__ == "__main__":
    main()
