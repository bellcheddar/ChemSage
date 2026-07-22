---
title: ChemSage API
emoji: ⚗️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# ChemSage Inference API

ZeroGPU-backed inference endpoint for ChemSage 32B v5 (Q4\_K\_M GGUF).

Consumed by the Flask PTY app at [chemsage.mdeller.com](https://chemsage.mdeller.com).

**Endpoint:** `POST /generate` — returns `text/event-stream` of token chunks.

**Cold start:** first request after idle downloads the 18 GB GGUF and allocates the A10G GPU
(~60-120 s). This is a portfolio demo; availability is best-effort.
