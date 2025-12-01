# Llama 3 PEFT

This repo contains a parameter efficient fine-tuning pipeline for Llama 3. The goal is to understand fine-tuning on limited GPU resources.

The code is built on top of the [Modal Labs LLM Finetuning guide](https://github.com/modal-labs/llm-finetuning).

## Improving Model Performance

<details>
<summary><b>Model-Centric Approach</b></summary>

### LoRA Hyperparameter Tuning

| Parameter | Current | Adjusted | Why |
|-----------|---------|----------|-----|
| `lora_r` | 16 | 32-128 | Higher rank increases adaptation capacity, enabling the model to capture more complex patterns. |
| `lora_alpha` | 32 | 16-64 | The `alpha/r` ratio controls adaptation strength. A ratio of 1/2 provides balanced updates without overwhelming the base model weights. |
| `lora_dropout` | 0.05 | 0.0-0.2 | Increased dropout on small datasets prevents overfitting, improving generalization to unseen data. |

[Hu et al. (2021) LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---

### Learning Rate and Optimizer

| Parameter | Current | Adjusted | Why |
|-----------|---------|----------|-----|
| `learning_rate` | 3e-4 | 1e-4-1e-3 | Grid or random search helps find learning rates that prevent training instability and enable faster convergence. |
| `lr_scheduler` | linear | cosine, cosine_with_restarts | Cosine scheduling with restarts helps escape local minima by increasing the learning rate periodically. |
| `warmup_steps` | 10 | 1-10% of total steps | Adequate warmup prevents early training divergence by gradually ramping up the learning rate. |
| `optimizer` | adamw_bnb_8bit | adamw_torch | Full-precision optimizers maintain numerical accuracy compared to lower bit variants, reducing quantization errors that accumulate during training. |

---

### Training Configuration

| Parameter | Current | Adjusted | Why |
|-----------|---------|----------|-----|
| `micro_batch_size` | 4 | 8-16 | Larger batch sizes provide more stable gradient estimates, reducing training variance and enabling faster convergence. |
| `sequence_len` | 1024 | 2048-4096 | Longer sequences allow the model to process more context, improving performance on tasks requiring larger context. |
| `max_steps` | 6250 | 12500-18750 | Training for 2-3 epochs may improve the model, but requires monitoring validation loss to prevent overfitting after epoch 2. |
| `gradient_checkpointing` | true | true | Gradient checkpointing trades computation for memory by recomputing activations during backward pass (prevents out-of-memory issues). |

---

### Model Architecture

| Parameter | Current | Adjusted | Why |
|-----------|---------|----------|-----|
| Quantization | 4-bit | 8-bit, full precision | Higher precision quantization (8-bit or full) reduces information loss. Full precision requires 2-4x more VRAM but provides the original quality. |
| Quantization Methods | QLoRA | GPTQ, AWQ | Alternative quantization methods may provide better quality-memory trade-offs depending on the use case. GPTQ and AWQ use different quantization strategies that might preserve more information. |

---

</details>

<details>
<summary><b>Data-Centric Approach</b></summary>

### Data Filtering

Filter for relevant examples and balance distribution when targeting specific applications. Remove short or repetitive outputs that don't contain useful patterns.

---

### Data Mixing

Mix datasets.

---

</details>
