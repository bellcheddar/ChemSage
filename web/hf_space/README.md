---
title: ChemSage API
emoji: ⚗️
colorFrom: cyan
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ChemSage Inference API

ZeroGPU-backed streaming inference endpoint for ChemSage 32B v5 (Q4_K_M GGUF).

Endpoint: `POST /generate` — returns `text/event-stream` of token chunks.
