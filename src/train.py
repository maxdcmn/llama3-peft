import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, List

import modal
import torch
import yaml
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
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
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_torch"
    epochs: int = 1
    max_training_steps: int = 200
    log_interval: int = 5
    checkpoint_interval: int = 100
    save_total_limit: int = 3
    results_dir: str = "/outputs"
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
    resume_from_checkpoint: str = None
    config_name: str = None


def load_config(config_path: str) -> TrainingConfig:
    """
    Load training configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        TrainingConfig: Training configuration
    """
    path = Path(config_path)
    if not path.is_absolute() and (modal_path := APP_DIR / path).exists():
        path = modal_path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        data = yaml.safe_load(f) or {}

    defaults = TrainingConfig()
    kwargs = {}
    for f in fields(TrainingConfig):
        value = data.get(f.name, getattr(defaults, f.name))
        if isinstance(value, str) and f.type in (float, int):
            try:
                value = float(value) if f.type == float else int(value)
            except ValueError:
                pass
        kwargs[f.name] = value

    if kwargs.get("config_name") is None:
        raise ValueError("config_name must be specified in config file")

    return TrainingConfig(**kwargs)


def prepare_dataset(cfg: TrainingConfig, tokenizer) -> Dataset:
    """
    Prepare the dataset for training.

    Args:
        cfg: Training configuration
        tokenizer: Tokenizer

    Returns:
        Dataset: Prepared dataset
    """
    # Load dataset from local JSONL file or HuggingFace Hub
    if cfg.dataset_id.startswith("local:"):
        path = Path(cfg.dataset_id.replace("local:", ""))
        if not path.is_absolute():
            path = Path(__file__).parent.parent / path
        ds = load_dataset("json", data_files=str(path), split="train")
    else:
        ds = load_dataset(cfg.dataset_id, split=cfg.dataset_split)

    # FineTome uses "human"/"gpt", but Llama expects "user"/"assistant"
    ROLE_MAP = {"human": "user", "gpt": "assistant"}

    def convert_chat_format(example):
        """
        Convert different chat formats to Llamas expected format, then apply
        the tokenizers chat template to get the final training text.
        """
        conv = example.get("conversations", [])
        if not conv:
            return {"text": ""}

        messages = []
        for turn in conv:
            # FineTome format: {"from": "human", "value": "..."}
            if "from" in turn:
                msg = {
                    "role": ROLE_MAP.get(turn.get("from", "user"), "user"),
                    "content": turn.get("value", ""),
                }
            # Tool response format (for DDG dataset)
            elif turn.get("role") == "tool":
                msg = {
                    "role": "tool",
                    "content": turn.get("content", ""),
                    "tool_call_id": turn.get("tool_call_id"),
                    "name": turn.get("name"),
                }
            # Standard format: {"role": "user", "content": "..."}
            else:
                msg = {
                    "role": turn.get("role", "assistant"),
                    "content": turn.get("content", ""),
                }
                # Preserve tool_calls if present (for training tool-calling behavior)
                tool_calls = turn.get("tool_calls")
                if tool_calls and len(tool_calls) > 0:
                    msg["tool_calls"] = tool_calls
            messages.append(msg)

        # Apply the models chat template to get properly formatted training text (adds special tokens)
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return {"text": text if text else ""}
        except Exception as e:
            print(f"Error applying chat template: {e}")
            return {"text": ""}

    ds = ds.map(
        convert_chat_format,
        remove_columns=[c for c in ds.column_names if c != "conversations"],
    )
    # Filter out empty examples
    return ds.filter(lambda x: x.get("text", "").strip())


def create_trainer(
    cfg: TrainingConfig, model, tokenizer, dataset: Dataset, run_name: str
) -> SFTTrainer:
    """
    Create the trainer.

    Args:
        cfg (TrainingConfig): Training configuration
        model: Model
        tokenizer: Tokenizer
        dataset (Dataset): Dataset
        run_name (str): Name of the run

    Returns:
        SFTTrainer: Trainer
    """
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
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_training_steps,
        logging_steps=cfg.log_interval,
        save_steps=cfg.checkpoint_interval,
        save_total_limit=cfg.save_total_limit,
        output_dir=cfg.results_dir,
        bf16=cfg.precision == torch.bfloat16,
        fp16=cfg.precision == torch.float16,
        seed=cfg.random_seed,
        packing=False,
        report_to="wandb",
        run_name=run_name,
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


def run_training(cfg: TrainingConfig):
    """
    Run the training.

    Args:
        cfg: Training configuration
    """
    base_dir = Path(cfg.results_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if cfg.resume_from_checkpoint:
        checkpoint_path = Path(cfg.resume_from_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        output_dir = (
            checkpoint_path.parent
            if checkpoint_path.name.startswith("checkpoint-")
            else checkpoint_path
        )
        cfg.results_dir = str(output_dir)
        print(f"Resuming from: {checkpoint_path}")
        run_name = output_dir.name
    else:
        output_dir = base_dir / cfg.config_name
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg.results_dir = str(output_dir)
        run_name = cfg.config_name
        config_dict = asdict(cfg)
        config_dict["wandb_run_name"] = run_name
        with (output_dir / "config.yaml").open("w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id, token=token, use_fast=False, trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.model_max_length = cfg.sequence_length
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        token=token,
        dtype=cfg.precision,
        device_map=None,
        trust_remote_code=True,
    )
    dataset = prepare_dataset(cfg, tokenizer)

    trainer = create_trainer(cfg, model, tokenizer, dataset, run_name)
    train_output = trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    trainer.save_model(cfg.results_dir)
    tokenizer.save_pretrained(cfg.results_dir)

    if train_output.metrics:
        with (output_dir / "training_metrics.json").open("w") as f:
            json.dump(train_output.metrics, f, indent=2)

    if cfg.config_name:
        push_to_hub(cfg.results_dir, cfg.config_name)


@app.function(
    image=image,
    gpu="A100:2",
    timeout=12 * HOURS,
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
def train_distributed(config_path: str, resume_from_checkpoint: str = None):
    """
    Launch distributed training via accelerate.

    Args:
        config_path: Path to the YAML configuration file
        resume_from_checkpoint: Path to the checkpoint directory to resume training from
    """
    num_gpus = max(1, torch.cuda.device_count())
    precision = (
        "bf16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "fp16"
    )
    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        str(num_gpus),
        "--num_machines",
        "1",
        "--mixed_precision",
        precision,
        "--dynamo_backend",
        "no",
        "/app/src/train.py",
        "--config",
        config_path,
    ]
    if resume_from_checkpoint:
        cmd.extend(["--resume-from-checkpoint", resume_from_checkpoint])
    subprocess.run(cmd, check=True, cwd="/app")


@app.local_entrypoint()
def train(config_path: str, resume_from_checkpoint: str = None):
    """
    Local entrypoint to start training.

    Args:
        config_path (str): Path to the YAML configuration file
        resume_from_checkpoint (str): Path to the checkpoint directory to resume training from
    """
    train_distributed.remote(config_path, resume_from_checkpoint)


def push_to_hub(output_dir: str, repo_name: str):
    """
    Push checkpoint to HF.

    Args:
        output_dir (str): Path to experiment output directory ("/outputs/llama-3.2-3B-finetome")
        repo_name (str): Repo name ("username/llama-3.2-3B-finetome")
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN required")
    api = HfApi(token=token)
    username = api.whoami(token=token)["name"]
    output_path = Path(output_dir)

    with (output_path / "config.yaml").open() as f:
        model_id = yaml.safe_load(f).get("model_id", "meta-llama/Llama-3.2-3B-Instruct")

    adapter_path = output_path / "adapter_config.json"
    if adapter_path.exists():
        with adapter_path.open() as f:
            adapter_config = json.load(f)
        adapter_config["base_model_name_or_path"] = model_id
        with adapter_path.open("w") as f:
            json.dump(adapter_config, f, indent=2)

    readme_path = output_path / "README.md"
    readme_path.unlink(missing_ok=True)
    readme_path.write_text(
        f"""---
base_model: {model_id}
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:{model_id}
- lora
- sft
- transformers
- trl
---
"""
    )

    repo_id = f"{username}/{repo_name}"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(output_path),
        path_in_repo=".",
        commit_message="Upload finetuned LoRA adapter",
        ignore_patterns=["checkpoint-*", "runs", "*.log"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with LoRA fine-tuning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.resume_from_checkpoint:
        cfg.resume_from_checkpoint = args.resume_from_checkpoint
    run_training(cfg)
