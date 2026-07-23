"""
ChemSage inference API — HuggingFace Space (Gradio SDK, ZeroGPU)

Custom routes (/generate, /health) live on a top-level FastAPI app;
Gradio is mounted on that app via gr.mount_gradio_app so ZeroGPU works.

Cold-start note: first request after idle downloads the 18 GB GGUF and
allocates the GPU (~60-120 s). Subsequent requests within the same GPU
lease are fast.
"""
from __future__ import annotations

import json

import gradio as gr
import spaces
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from huggingface_hub import hf_hub_download

REPO_ID      = "Dellboy/chem_sage_32b_v5-GGUF"
FILENAME     = "chem_sage_32b_v5_q4km.gguf"
N_CTX        = 3072
N_GPU_LAYERS = -1

_model_path: str | None = None


def _get_model_path() -> str:
    global _model_path
    if _model_path is None:
        _model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    return _model_path


@spaces.GPU(duration=180)
def _generate_tokens(
    prompt: str,
    max_tokens: int,
    temperature: float,
    repeat_penalty: float,
) -> list[str]:
    """Run inside the ZeroGPU lease; collect all tokens and return."""
    from llama_cpp import Llama

    llm = Llama(
        model_path=_get_model_path(),
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    tokens: list[str] = []
    for chunk in llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        repeat_penalty=repeat_penalty,
        stream=True,
    ):
        tok = chunk["choices"][0]["text"]
        if tok:
            tokens.append(tok)
    return tokens


# ── Top-level FastAPI app — custom routes registered here survive launch ──────

app = FastAPI()


@app.post("/generate")
async def generate(request: Request):
    body           = await request.json()
    prompt         = body.get("prompt", "")
    max_tokens     = int(body.get("max_tokens", 512))
    temperature    = float(body.get("temperature", 0.15))
    repeat_penalty = float(body.get("repeat_penalty", 1.15))

    tokens = _generate_tokens(prompt, max_tokens, temperature, repeat_penalty)

    def event_stream():
        for tok in tokens:
            yield f"data: {json.dumps({'token': tok})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Gradio UI (minimal — required for ZeroGPU) ────────────────────────────────

with gr.Blocks(title="ChemSage API") as demo:
    gr.Markdown(
        "## ⚗️ ChemSage Inference API\n\n"
        "Internal endpoint for [chemsage.mdeller.com](https://chemsage.mdeller.com). "
        "Use `POST /generate` — returns `text/event-stream` of token chunks.\n\n"
        "**Cold start:** first request after idle takes ~60-120 s (GGUF download + GPU alloc)."
    )

# Mount Gradio at root — routes above take priority over Gradio's catch-all
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
