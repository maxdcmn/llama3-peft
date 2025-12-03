import sys
from pathlib import Path

import modal

APP_DIR = Path("/app")
if APP_DIR.exists():
    sys.path.insert(0, str(APP_DIR))

app = modal.App(name="llama3-peft-training")

vol = modal.Volume.from_name("model-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir(
        ".",
        "/app",
        ignore=[
            ".venv",
            ".git",
            "__pycache__",
            "model-checkpoints",
            ".modal_cache",
            "**/.venv/**",
            "**/.git/**",
            "**/__pycache__/**",
        ],
    )
)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10")
    .pip_install("vllm==0.5.1", "torch==2.3.0")
    .entrypoint([])
)

MINUTES = 60
HOURS = 60 * MINUTES

VOLUME_CONFIG = {"/outputs": vol}


class Colors:
    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"
