"""
ChemSage inference API — HuggingFace Space (Gradio SDK, ZeroGPU)

Exposes a named Gradio endpoint `generate` reachable at:
  POST  /gradio_api/call/generate          → {"event_id": "..."}
  GET   /gradio_api/call/generate/{id}     → SSE stream with process_completed event

All custom-FastAPI-route approaches fail in Gradio 6.x because the SvelteKit SPA
intercepts unknown paths and Gradio mounts its API as a sub-app at /gradio_api/.
Using Gradio's own call API sidesteps both problems.

Cold-start: first request downloads the 18 GB GGUF and allocates the GPU (~60-120 s).
"""
from __future__ import annotations

import gradio as gr
import spaces
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
def generate(
    prompt: str,
    max_tokens: int   = 512,
    temperature: float = 0.15,
    repeat_penalty: float = 1.15,
) -> str:
    """Generate a response. Called through Gradio's queue so ZeroGPU works."""
    from llama_cpp import Llama

    llm = Llama(
        model_path=_get_model_path(),
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    return "".join(
        chunk["choices"][0]["text"]
        for chunk in llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            stream=True,
        )
        if chunk["choices"][0]["text"]
    )


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt (ChatML format)", lines=6),
        gr.Number(label="max_tokens", value=512),
        gr.Number(label="temperature", value=0.15),
        gr.Number(label="repeat_penalty", value=1.15),
    ],
    outputs=gr.Textbox(label="Response", lines=12),
    title="⚗️ ChemSage API",
    description=(
        "Internal inference API for "
        "[chemsage.mdeller.com](https://chemsage.mdeller.com).  "
        "**Cold start:** first request takes ~60-120 s (GGUF download + GPU alloc)."
    ),
    api_name="generate",
)

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
