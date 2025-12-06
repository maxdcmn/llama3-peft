import subprocess

import modal

MINUTES = 60
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_ADAPTER = "maxdcmn/llama-3.2-finetome"
APP = "llama-3.2-3B-finetome"
GPU = "L40S:1"
VLLM_PORT = 8000


vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App(APP)


@app.function(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    num_gpus = int(GPU.split(":")[1]) if ":" in GPU else 1
    
    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        BASE_MODEL,
        "--served-model-name",
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--enable-lora",
        "--lora-modules",
        f"default={LORA_ADAPTER}",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "llama3_json",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.9",
        "--enable-chunked-prefill",
        "--no-enforce-eager",
        "--tensor-parallel-size",
        str(num_gpus),
    ]

    subprocess.Popen(cmd)
