"""
chat_remote.py — drop-in launcher for web sessions.

Patches stream_generate / generate in mlx_lm to call the remote HF Space API,
then imports and runs the real chat.py unchanged.
All Rich output, corpus tables, slash commands, and CLI behaviour are identical.
"""
from __future__ import annotations

import json
import runpy
import os
import sys
import types
from pathlib import Path

# ── locate chat.py — works in both repo layout and droplet layout ──────────
# Repo:    chem_sage/web/flask_app/chat_remote.py → chem_sage/scripts/chat.py
# Droplet: /opt/chemsage/chat_remote.py           → /opt/chem_sage_scripts/chat.py
_this = Path(__file__).resolve()
_candidates = [
    _this.parent.parent.parent / "scripts" / "chat.py",  # repo layout
    Path("/opt/chem_sage_scripts/chat.py"),               # droplet default
    Path(os.environ.get("CHEMSAGE_CHAT_PY", "__none__")),
]
chat_path = next((p for p in _candidates if p.exists()), None)
if chat_path is None:
    raise FileNotFoundError(
        "chat.py not found. Set CHEMSAGE_CHAT_PY env var or check repo layout."
    )
# Add repo root and scripts dir so chat.py's imports resolve
sys.path.insert(0, str(chat_path.parent.parent))
sys.path.insert(0, str(chat_path.parent))

HF_SPACE_URL = os.environ.get("HF_SPACE_URL", "").rstrip("/")

# ---------------------------------------------------------------------------
# Fake mlx_lm that calls the remote API instead of a local GPU
# ---------------------------------------------------------------------------

def _remote_stream_generate(model, tokenizer, *, prompt: str, max_tokens: int = 512,
                             sampler=None, logits_processors=None, **kwargs):
    """Yield text chunks from the HF Space streaming endpoint."""
    import urllib.request
    body = json.dumps({
        "prompt":         prompt,
        "max_tokens":     max_tokens,
        "temperature":    0.15,
        "repeat_penalty": 1.15,
    }).encode()
    req = urllib.request.Request(
        f"{HF_SPACE_URL}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                token = json.loads(payload).get("token", "")
            except json.JSONDecodeError:
                continue
            if token:
                # Yield a fake chunk object that chat.py expects
                yield types.SimpleNamespace(text=token)


def _remote_generate(model, tokenizer, *, prompt: str, max_tokens: int = 512, **kwargs) -> str:
    """Blocking version — collect all tokens and return."""
    return "".join(chunk.text for chunk in _remote_stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens))


def _fake_load(model_path, *args, **kwargs):
    """Return dummy objects; the real model lives on the HF Space."""
    return types.SimpleNamespace(_is_remote=True), None


# Inject a fake mlx_lm module before chat.py imports it
_mlx_lm_fake = types.ModuleType("mlx_lm")
_mlx_lm_fake.load            = _fake_load
_mlx_lm_fake.stream_generate = _remote_stream_generate
_mlx_lm_fake.generate        = _remote_generate

# Also fake out sub-modules chat.py imports from mlx_lm
for _sub in ("utils", "generate", "sample_utils"):
    _m = types.ModuleType(f"mlx_lm.{_sub}")
    sys.modules[f"mlx_lm.{_sub}"] = _m

# make_sampler / make_logits_processors are passed as kwargs to our remote
# stream_generate which ignores them — just return None so the import works.
sys.modules["mlx_lm.sample_utils"].make_sampler           = lambda *a, **k: None
sys.modules["mlx_lm.sample_utils"].make_logits_processors = lambda *a, **k: None

_mlx_lm_fake.utils         = sys.modules["mlx_lm.utils"]
_mlx_lm_fake.generate_mod  = sys.modules["mlx_lm.generate"]
sys.modules["mlx_lm"] = _mlx_lm_fake

# Fake mlx.core (used for mx.set_wired_limit etc.) — silently ignore
import types as _t
_mlx = _t.ModuleType("mlx")
_mlx.core = _t.ModuleType("mlx.core")
_mlx.core.set_wired_limit = lambda *a, **k: None
_mlx.core.metal           = _t.ModuleType("mlx.core.metal")
_mlx.core.metal.device_info = lambda: {}
sys.modules.update({"mlx": _mlx, "mlx.core": _mlx.core, "mlx.core.metal": _mlx.core.metal})

# ---------------------------------------------------------------------------
# Run the real chat.py
# ---------------------------------------------------------------------------

# Inject flags chat.py requires; --no-rag avoids ChromaDB/sentence-transformers on the droplet
if "--model" not in sys.argv:
    sys.argv += ["--model", "chem_sage_32b_v5"]
if "--no-rag" not in sys.argv:
    sys.argv.append("--no-rag")

import runpy
runpy.run_path(str(chat_path), run_name="__main__")
