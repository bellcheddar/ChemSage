---
title: ChemSage API
emoji: ⚗️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
hardware: zero-gpu
---

# ChemSage Inference API

ZeroGPU-backed inference endpoint for ChemSage 32B v5 (Q4\_K\_M GGUF, 19.9 GB).

Consumed by the Flask PTY app at [chemsage.mdeller.com](https://chemsage.mdeller.com).

**API (Gradio 5 call protocol):**

```
POST /gradio_api/call/generate
Body: {"data": ["<prompt>", <max_tokens>, <temperature>, <repeat_penalty>]}
→ {"event_id": "..."}

GET /gradio_api/call/generate/{event_id}
→ SSE stream; look for:
    event: complete
    data: ["<generated text>"]
```

**Cold start:** first call after idle loads the model into the A10G (~20-30 s). The ZeroGPU worker
caches the model between calls (worker lifetime ~48 h), so subsequent requests in the same session
are fast. `@spaces.GPU(duration=120)` — 120 s wall-clock limit per call.

**CUDA:** uses the native cu130 llama-cpp-python wheel (CUDA 13.0); ZeroGPU Blackwell hardware.
This is a portfolio demo; availability is best-effort.
