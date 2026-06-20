#!/usr/bin/env python3
"""
tool_exec.py — sandboxed execution shim for the chemistry tools the model emits (Phase 6).

This is the chemistry-aware heart of the hybrid: it detects RDKit/PyMOL code blocks in a model
response, runs them in a restricted subprocess, and returns the results so the assistant reasons
over real values, not hallucinated ones.

Security: start RDKit-only, in a subprocess with NO network and NO filesystem write access.
Enable PyMOL/docking execution only once the sandbox is proven (PROJECT_PLAN.md hard rules).

Claude Code TODO:
  - Replace the naive subprocess with a properly isolated runner (resource limits, timeout,
    restricted builtins, temp working dir, no network namespace).
  - Add a SMILES round-trip guard: any SMILES in the model output is canonicalised by RDKit
    before anything else runs; invalid strings are flagged back to the model.
"""

import re
import subprocess
import sys
import tempfile

CODE_BLOCK = re.compile(r"```python\s*(.*?)```", re.DOTALL)
TIMEOUT_S = 20


def extract_code(model_output: str):
    """Pull python code blocks from the assistant message."""
    return [m.strip() for m in CODE_BLOCK.findall(model_output)]


def run_sandboxed(code: str) -> str:
    """Execute one code block in a restricted subprocess. TODO: harden isolation."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
        path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=TIMEOUT_S,
            # TODO: drop network, restrict cwd, set rlimits
        )
        return proc.stdout if proc.returncode == 0 else f"[error]\n{proc.stderr}"
    except subprocess.TimeoutExpired:
        return "[error] execution timed out"


def execute(model_output: str) -> str:
    """Run every code block in the model output and return concatenated results."""
    results = [run_sandboxed(block) for block in extract_code(model_output)]
    return "\n".join(results) if results else "[no executable tool calls found]"


if __name__ == "__main__":
    demo = "```python\nfrom rdkit import Chem\nprint(Chem.MolToSmiles(Chem.MolFromSmiles('OC(=O)c1ccccc1')))\n```"
    print(execute(demo))
