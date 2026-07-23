"""
ChemSage inference API — HuggingFace Space (Gradio SDK, ZeroGPU)

Uses the cu130 llama-cpp-python wheel (built against CUDA 13.0) which matches
the ZeroGPU Blackwell environment exactly — no CUDA ABI shim required.

Model is lazy-loaded and cached in the ZeroGPU worker (_llm global).

API client calls:
  POST /gradio_api/call/generate          → {"event_id": "..."}
  GET  /gradio_api/call/generate/{id}     → SSE, wait for data: ["<text>"]
"""
from __future__ import annotations
import os

import gradio as gr
import spaces
from huggingface_hub import hf_hub_download

REPO_ID   = "Dellboy/chem_sage_32b_v5-GGUF"
FILENAME  = "chem_sage_32b_v5_q4km.gguf"
N_CTX     = 2048

print(f"Downloading {FILENAME} …")
_model_path: str = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
print(f"Model cached at {_model_path}")

_llm = None


@spaces.GPU(duration=120)
def generate(prompt: str, max_tokens: float, temperature: float, repeat_penalty: float) -> str:
    global _llm

    if _llm is None:
        from llama_cpp import Llama
        print("Loading model into A10G VRAM …")
        _llm = Llama(
            model_path=_model_path,
            n_ctx=N_CTX,
            n_gpu_layers=-1,
            n_threads=os.cpu_count() or 8,
            verbose=False,
        )
        print("Model ready.")

    return "".join(
        chunk["choices"][0]["text"]
        for chunk in _llm(
            prompt,
            max_tokens=int(max_tokens),
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            stream=True,
        )
        if chunk["choices"][0]["text"]
    )


with gr.Blocks(title="ChemSage API") as demo:
    gr.Markdown(
        "## ⚗️ ChemSage Inference API\n\n"
        "Internal endpoint for [chemsage.mdeller.com](https://chemsage.mdeller.com).\n\n"
        "**Cold start:** first request loads the 32 B model into the A10G (~20-30 s). "
        "Subsequent requests in the same session are fast."
    )
    with gr.Row():
        with gr.Column():
            prompt_in      = gr.Textbox(label="Prompt (ChatML format)", lines=6)
            max_tokens_in  = gr.Number(label="max_tokens",     value=256)
            temperature_in = gr.Number(label="temperature",    value=0.15)
            penalty_in     = gr.Number(label="repeat_penalty", value=1.15)
            btn            = gr.Button("Generate")
        with gr.Column():
            output_out = gr.Textbox(label="Response", lines=12)

    btn.click(
        fn=generate,
        inputs=[prompt_in, max_tokens_in, temperature_in, penalty_in],
        outputs=output_out,
        api_name="generate",
    )

demo.queue()
demo.launch()
