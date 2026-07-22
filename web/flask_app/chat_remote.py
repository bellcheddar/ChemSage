"""
chat_remote.py — drop-in launcher for web sessions.

Patches stream_generate / generate in mlx_lm to call the remote HF Space API,
then imports and runs the real chat.py unchanged.
All Rich output, corpus tables, slash commands, and CLI behaviour are identical.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path

# ── locate the chem_sage repo root (two levels up from this file) ──
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

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

chat_path = REPO_ROOT / "scripts" / "chat.py"

# Inject --model flag (chat.py requires it even though we don't load locally)
if "--model" not in sys.argv:
    sys.argv += ["--model", "chem_sage_32b_v5"]

spec   = importlib.util.spec_from_file_location("chat", str(chat_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
