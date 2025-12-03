import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import modal
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

APP_DIR = Path("/app")
if APP_DIR.exists():
    sys.path.insert(0, str(APP_DIR))

try:
    from .common import HOURS, MINUTES, VOLUME_CONFIG, app, image
except ImportError:
    from src.common import HOURS, MINUTES, VOLUME_CONFIG, app, image


@dataclass
class TrainingConfig:
    """
    Training configuration for the model.
    """

    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    dataset_id: str = "mlabonne/FineTome-100k"
    dataset_split: str = "train"
    sequence_length: int = 1024
    batch_size_per_device: int = 1
    gradient_accumulation: int = 8
    warmup_steps: int = 50
    lr: float = 2e-4
    weight_decay: float = 0.01
    epochs: int = 1
    max_training_steps: int = 200
    log_interval: int = 5
    checkpoint_interval: int = 100
    results_dir: str = "/outputs"
    model_cache_dir: str | None = None
    random_seed: int = 9024309
    enable_gradient_checkpointing: bool = False
    gradient_checkpointing_opts: dict | None = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    ddp_unused_params: bool | None = False
    precision: Any = field(
        default_factory=lambda: (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
    )
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout_rate: float = 0.05
    lora_targets: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


def prepare_dataset(cfg: TrainingConfig) -> Dataset:
    """
    Load and prepare the training dataset.
    """
    if cfg.dataset_id.startswith("local:"):
        local_path = cfg.dataset_id.replace("local:", "")
        if not os.path.isabs(local_path):
            base_dir = Path(__file__).parent.parent
            local_path = base_dir / local_path
        ds = load_dataset("json", data_files=str(local_path), split="train")
    else:
        ds = load_dataset(cfg.dataset_id, split=cfg.dataset_split)

    def format_conversation(example):
        conv = example.get("conversations", [])
        parts = []
        for turn in conv:
            content = turn.get("value")
            if not content:
                continue
            role = turn.get("from", "").lower()
            prefix = "User" if role in ("human", "user") else "Assistant"
            parts.append(f"{prefix}: {content}")
        example["text"] = "\n\n".join(parts)
        return example

    ds = ds.map(format_conversation, remove_columns=["conversations"])
    ds = ds.filter(
        lambda x: isinstance(x.get("text"), str) and len(x["text"].strip()) > 0
    )
    return ds


def load_tokenizer(cfg: TrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_cache_dir,
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = cfg.sequence_length
    return tokenizer


def load_base_model(cfg: TrainingConfig):
    return AutoModelForCausalLM.from_pretrained(
        cfg.model_cache_dir,
        dtype=cfg.precision,
        device_map=None,
        trust_remote_code=True,
    )


def create_trainer(
    cfg: TrainingConfig, model, tokenizer, dataset: Dataset
) -> SFTTrainer:
    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout_rate,
        target_modules=cfg.lora_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        dataset_text_field="text",
        max_length=cfg.sequence_length,
        per_device_train_batch_size=cfg.batch_size_per_device,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        warmup_steps=cfg.warmup_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_training_steps,
        logging_steps=cfg.log_interval,
        save_steps=cfg.checkpoint_interval,
        save_total_limit=3,
        output_dir=cfg.results_dir,
        bf16=cfg.precision == torch.bfloat16,
        fp16=cfg.precision == torch.float16,
        seed=cfg.random_seed,
        packing=False,
        report_to="wandb",
        dataset_num_proc=2,
        gradient_checkpointing=cfg.enable_gradient_checkpointing,
        gradient_checkpointing_kwargs=cfg.gradient_checkpointing_opts,
        ddp_find_unused_parameters=cfg.ddp_unused_params,
    )

    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )


def run_training(cfg: TrainingConfig) -> None:
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    cache_folder_name = cfg.model_cache_dir or cfg.model_id.replace("/", "__")
    cache_path = Path(cfg.results_dir) / cache_folder_name
    token = os.environ.get("HF_TOKEN")

    if not cache_path.exists() or not any(cache_path.iterdir()):
        snapshot_download(
            repo_id=cfg.model_id,
            local_dir=str(cache_path),
            token=token,
        )

    cfg.model_cache_dir = str(cache_path)
    cfg.results_dir = str(cache_path)

    dataset = prepare_dataset(cfg)
    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg)
    trainer = create_trainer(cfg, model, tokenizer, dataset)

    trainer.train()

    trainer.save_model(cfg.results_dir)
    tokenizer.save_pretrained(cfg.results_dir)


@app.function(
    image=image,
    gpu="A100:2",
    timeout=3 * HOURS,
    volumes=VOLUME_CONFIG,
    env={
        "NCCL_DEBUG": "WARN",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    },
    secrets=[
        modal.Secret.from_name("my-huggingface-secret"),
        modal.Secret.from_name("wandb"),
    ],
)
def train_distributed():
    num_gpus = max(1, torch.cuda.device_count())

    subprocess.run(
        [
            "accelerate",
            "launch",
            "--num_processes",
            str(num_gpus),
            "--num_machines",
            "1",
            "--mixed_precision",
            (
                "bf16"
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else "fp16"
            ),
            "/app/src/train.py",
        ],
        check=True,
        cwd="/app",
    )


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("my-huggingface-secret"),
        modal.Secret.from_name("wandb"),
    ],
    volumes=VOLUME_CONFIG,
)
def upload_model(checkpoint_path: str, repo_name: str | None = None):
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable must be set")

    api = HfApi(token=token)
    username = api.whoami(token=token)["name"]

    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_dir}")

    if not repo_name:
        repo_name = f"{username}/{checkpoint_dir.name}-fine-tuned"

    api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_name,
        repo_type="model",
        folder_path=str(checkpoint_dir),
        path_in_repo=".",
        commit_message="Upload fine-tuned LoRA adapter weights",
    )
    return repo_name


@app.local_entrypoint()
def run():
    """
    Local entrypoint to start training.
    """
    train_distributed.remote()


@app.local_entrypoint()
def push(checkpoint_path: str, repo_name: str = ""):
    """
    Local entrypoint to push model to HF.
    """
    repo_id = upload_model.remote(checkpoint_path, repo_name if repo_name else None)
    print(f"Uploaded to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    config = TrainingConfig(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        dataset_id="mlabonne/FineTome-100k",
        dataset_split="train",
        sequence_length=1024,
        batch_size_per_device=1,
        gradient_accumulation=8,
        epochs=1,
        warmup_steps=50,
        max_training_steps=200,
        lr=2e-4,
        log_interval=5,
        checkpoint_interval=100,
        results_dir="/outputs",
    )
    run_training(config)
