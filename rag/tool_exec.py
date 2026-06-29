#!/usr/bin/env python3
"""
tool_exec.py — sandboxed execution shim for the chemistry tools the model emits (Phase 6).

Detects RDKit Python code blocks in a model response, runs each in a restricted subprocess,
and returns the stdout so the assistant can reason over real values rather than guessed ones.

Security posture (best-effort on macOS, no root required):
  - Static blocklist: obvious network/filesystem import patterns rejected before execution.
  - Clean environment: subprocess inherits only PATH and PYTHONPATH.
  - Isolated working directory: a fresh temp dir per run, discarded afterwards.
  - Hard timeout: 20 s per block; process killed on expiry.
  - PyMOL blocks are NOT auto-executed (they require a display and file access): they are
    returned as-is for the user to run manually in the GUI.

Enable PyMOL/docking execution only after additional isolation is in place.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CODE_BLOCK = re.compile(r"```python\s*(.*?)```", re.DOTALL)
TIMEOUT_S  = 20

# Patterns that signal network or destructive filesystem access.
# Rejection is a static text check; it is not a substitute for OS-level isolation.
_BLOCKED = frozenset([
    "import socket",
    "import http",
    "import urllib",
    "import requests",
    "import ftplib",
    "import smtplib",
    "import subprocess",
    "os.system(",
    "os.popen(",
    "shutil.rmtree",
    "shutil.copy",
    "__import__(",
])

# RDKit imports that are permitted (used for the "is this a chemistry block?" heuristic)
_RDKIT_MARKERS = frozenset(["from rdkit", "import rdkit"])


def extract_code(model_output: str) -> list[str]:
    """Return all ```python ... ``` blocks from the model response."""
    return [m.strip() for m in CODE_BLOCK.findall(model_output)]


def _is_rdkit_block(code: str) -> bool:
    return any(marker in code for marker in _RDKIT_MARKERS)


def _static_check(code: str) -> str | None:
    """Return a block reason string if the code contains a disallowed pattern, else None."""
    for pattern in _BLOCKED:
        if pattern in code:
            return f"[blocked] disallowed pattern in code: '{pattern}'"
    return None


def run_sandboxed(code: str) -> str:
    """Execute one RDKit code block in an isolated subprocess. Returns stdout or an error string."""
    blocked = _static_check(code)
    if blocked:
        return blocked

    env = {
        "PATH":       os.environ.get("PATH", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        # Keep CONDA_PREFIX / virtual env vars so RDKit is importable
        **{k: v for k, v in os.environ.items()
           if k.startswith(("CONDA_", "VIRTUAL_ENV", "HOME"))},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "chem_block.py"
        script.write_text(code)
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
                cwd=tmpdir,
                env=env,
            )
            if proc.returncode == 0:
                return proc.stdout.strip() or "[ok, no output]"
            return f"[error]\n{proc.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return f"[error] execution timed out after {TIMEOUT_S}s"
        except Exception as e:
            return f"[error] {e}"


def execute(model_output: str) -> str:
    """Run every RDKit block in the model output; return concatenated results.

    PyMOL blocks (no rdkit import) are flagged but not executed.
    """
    blocks = extract_code(model_output)
    if not blocks:
        return "[no executable tool calls found]"

    parts = []
    for i, block in enumerate(blocks, 1):
        if not _is_rdkit_block(block):
            parts.append(f"[block {i}] [skipped: PyMOL/non-RDKit — run manually in the GUI]")
            continue
        result = run_sandboxed(block)
        parts.append(f"[block {i}]\n{result}")

    return "\n\n".join(parts)


if __name__ == "__main__":
    _demo = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors\n"
        "mol = Chem.MolFromSmiles('OC(=O)c1ccccc1')\n"
        "print(Chem.MolToSmiles(mol), round(Descriptors.MolWt(mol), 1))\n"
        "```"
    )
    print(execute(_demo))
