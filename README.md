# LLM Fine-Tuning & Inference Pipeline

> Parameter-efficient fine-tuning of large language models using **LoRA** and **QLoRA**, with quantised local inference for production deployment.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black?style=flat-square)](https://black.readthedocs.io)

---

## Overview

This project implements a **production-grade** pipeline for fine-tuning large language models with parameter-efficient methods, then deploying them with quantised inference. It covers the full lifecycle — from raw dataset preparation to serving responses through a batched inference engine.

```
Raw Data → Preprocessing → Tokenisation → LoRA/QLoRA Fine-Tuning
    → Checkpoint Management → Adapter Merging → GGUF Quantisation
        → llama-cpp Inference Engine → Evaluation & Benchmarking
```

### Key Features

| Feature | Description |
|---|---|
| **LoRA / QLoRA** | Rank-decomposed adapter training with 4-bit NF4 quantisation |
| **Gradient Checkpointing** | Train 7B+ models on a single consumer GPU |
| **Flash Attention 2** | Memory-efficient attention for long contexts |
| **Dynamic Batching** | Throughput-optimised inference with adaptive batch sizing |
| **Streaming Generation** | Token-by-token streaming for low-latency applications |
| **GGUF Export** | llama-cpp compatible quantised model export |
| **W&B Integration** | Full experiment tracking, gradient norms, loss curves |
| **Evaluation Suite** | Perplexity, ROUGE, BERTScore, custom task metrics |

---

## Architecture

```
llm-finetune/
├── src/
│   ├── data/
│   │   ├── dataset.py          # Dataset abstractions (chat, completion, instruct)
│   │   ├── preprocessing.py    # Cleaning, deduplication, quality filtering
│   │   ├── tokenisation.py     # Prompt templates, packing, masking
│   │   └── collator.py         # Dynamic padding, DataCollator implementations
│   ├── training/
│   │   ├── trainer.py          # Custom HuggingFace Trainer subclass
│   │   ├── lora_config.py      # LoRA/QLoRA configuration builders
│   │   ├── callbacks.py        # Checkpoint, early stopping, LR logging
│   │   └── scheduler.py        # Cosine/linear warmup schedulers
│   ├── inference/
│   │   ├── engine.py           # Core inference engine with dynamic batching
│   │   ├── quantise.py         # GGUF export and llama-cpp integration
│   │   ├── streaming.py        # Streaming token generator
│   │   └── sampler.py          # Sampling strategies (top-p, top-k, mirostat)
│   ├── evaluation/
│   │   ├── benchmarks.py       # Perplexity, ROUGE, BERTScore
│   │   ├── harness.py          # lm-evaluation-harness integration
│   │   └── compare.py          # Multi-run comparison and statistical tests
│   └── utils/
│       ├── logging.py          # Structured logging with rich
│       ├── memory.py           # VRAM profiling and OOM guards
│       ├── reproducibility.py  # Seed management, deterministic ops
│       └── checkpoint.py       # Safe checkpoint save/resume
├── configs/
│   ├── base.yaml               # Shared defaults
│   ├── lora_7b.yaml            # Llama-3 7B LoRA recipe
│   ├── qlora_13b.yaml          # Llama-3 13B QLoRA (4-bit) recipe
│   └── inference.yaml          # Inference engine settings
├── scripts/
│   ├── train.py                # Training entrypoint
│   ├── merge_adapter.py        # Merge LoRA adapters into base model
│   ├── quantise.py             # GGUF quantisation entrypoint
│   ├── evaluate.py             # Evaluation entrypoint
│   └── benchmark_inference.py  # Throughput/latency benchmarking
├── tests/
│   ├── test_dataset.py
│   ├── test_tokenisation.py
│   ├── test_lora_config.py
│   └── test_inference.py
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_training_analysis.ipynb
    └── 03_inference_profiling.ipynb
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/yourname/llm-finetune
cd llm-finetune

# CPU/CUDA (pick one)
pip install -e ".[train]"
pip install -e ".[train,flash-attn]"   # + Flash Attention 2
pip install -e ".[inference]"          # llama-cpp only
pip install -e ".[all]"                # everything
```

### 2. Prepare Data

```bash
# From HuggingFace Hub
python scripts/train.py \
  --config configs/lora_7b.yaml \
  --dataset "HuggingFaceH4/ultrachat_200k" \
  --split "train_sft" \
  --dry-run  # validate config without training
```

Or use a local JSONL file. Every record must have `"messages"` in OpenAI chat format:

```jsonc
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user",   "content": "Explain LoRA in one paragraph."},
  {"role": "assistant", "content": "LoRA (Low-Rank Adaptation) ..."}
]}
```

### 3. Train

```bash
# LoRA on a 7B model (needs ~20 GB VRAM)
python scripts/train.py --config configs/lora_7b.yaml

# QLoRA on a 13B model (needs ~14 GB VRAM with 4-bit)
python scripts/train.py --config configs/qlora_13b.yaml

# Resume from checkpoint
python scripts/train.py --config configs/lora_7b.yaml \
  --resume-from outputs/lora_7b/checkpoint-500
```

### 4. Merge & Export

```bash
# Merge LoRA adapter weights into the base model
python scripts/merge_adapter.py \
  --base-model meta-llama/Meta-Llama-3-8B \
  --adapter-path outputs/lora_7b/final_adapter \
  --output-path outputs/merged_model

# Export to GGUF (Q4_K_M quantisation)
python scripts/quantise.py \
  --model-path outputs/merged_model \
  --quant-type Q4_K_M \
  --output-path outputs/model_q4km.gguf
```

### 5. Inference

```python
from src.inference.engine import InferenceEngine
from src.inference.sampler import SamplingConfig

engine = InferenceEngine.from_gguf(
    model_path="outputs/model_q4km.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,   # offload all layers to GPU
)

sampling = SamplingConfig(temperature=0.7, top_p=0.9, max_tokens=512)

# Single call
response = engine.generate("Explain gradient checkpointing.", sampling)

# Streaming
for token in engine.stream("Write a haiku about backpropagation.", sampling):
    print(token, end="", flush=True)

# Batch inference
responses = engine.generate_batch(
    prompts=["Prompt A", "Prompt B", "Prompt C"],
    sampling=sampling,
    max_batch_size=4,
)
```

### 6. Evaluate

```bash
python scripts/evaluate.py \
  --model-path outputs/model_q4km.gguf \
  --tasks perplexity,rouge,bertscore \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split test_sft \
  --output-dir outputs/eval_results
```

---

## Configuration

Configs are YAML files merged at runtime (base → task-specific → CLI overrides).

**`configs/lora_7b.yaml`** (excerpt):
```yaml
model:
  name_or_path: meta-llama/Meta-Llama-3-8B
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  bias: none
  task_type: CAUSAL_LM

training:
  num_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8   # effective batch = 16
  learning_rate: 2.0e-4
  lr_scheduler: cosine_with_warmup
  warmup_ratio: 0.03
  max_grad_norm: 1.0
  fp16: false
  bf16: true
  gradient_checkpointing: true
  dataloader_num_workers: 4
  save_strategy: steps
  save_steps: 250
  logging_steps: 10
  report_to: [wandb, tensorboard]

data:
  max_seq_length: 2048
  packing: true                    # sequence packing for efficiency
  pad_to_multiple_of: 8
```

---

## Memory & Performance Guide

| Model | Method | VRAM | Throughput |
|---|---|---|---|
| Llama 3 8B | Full fine-tune (BF16) | ~60 GB | baseline |
| Llama 3 8B | LoRA r=64 (BF16) | ~20 GB | ~0.85× |
| Llama 3 8B | QLoRA r=64 (NF4) | ~10 GB | ~0.60× |
| Llama 3 70B | QLoRA r=16 (NF4) | ~48 GB | ~0.40× |
| — | GGUF Q4_K_M (CPU) | ~5 GB RAM | ~15 tok/s |
| — | GGUF Q4_K_M (GPU) | ~6 GB VRAM | ~80 tok/s |

> Throughput measured on RTX 3090, batch size 1, 512-token sequences.

---

## Concepts

### LoRA (Low-Rank Adaptation)

LoRA freezes the pre-trained weights `W₀ ∈ ℝᵐˣⁿ` and injects trainable rank decomposition matrices:

```
W = W₀ + ΔW = W₀ + BA
where B ∈ ℝᵐˣʳ, A ∈ ℝʳˣⁿ, r ≪ min(m, n)
```

Only `B` and `A` are trained, reducing trainable parameters from `m×n` to `r×(m+n)`. At `r=64` on Llama-3-8B, this is ~1% of total parameters.

### QLoRA

QLoRA combines LoRA with 4-bit NF4 (NormalFloat4) quantisation of the frozen base model, double quantisation of the quantisation constants, and paged optimisers to handle VRAM spikes. This enables fine-tuning 65B models on a single A100-80GB.

### Sequence Packing

Instead of padding short sequences to `max_seq_length`, packing bins multiple examples into each context window, achieving near-100% token utilisation and significant training speedups.

---

## Development

```bash
# Run tests
pytest tests/ -v --tb=short

# Type check
mypy src/ --ignore-missing-imports

# Lint + format
ruff check src/ && black src/

# Profile VRAM during training (dry run)
python scripts/train.py --config configs/lora_7b.yaml --profile-memory
```

---

## License

MIT — see [LICENSE](LICENSE).
