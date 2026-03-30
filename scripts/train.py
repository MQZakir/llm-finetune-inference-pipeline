"""
Training entrypoint — fine-tune an LLM with LoRA or QLoRA.

Usage
-----
  python scripts/train.py --config configs/lora_7b.yaml
  python scripts/train.py --config configs/qlora_13b.yaml --dry-run
  python scripts/train.py --config configs/lora_7b.yaml \\
      --resume-from outputs/lora_7b/checkpoint-500
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from omegaconf import OmegaConf

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collator import DynamicPaddingCollator, SequencePackingCollator
from src.data.dataset import DatasetConfig, DatasetFormat, FineTuneDataset, PackedDataset
from src.training.lora_config import QuantisationConfig, get_preset
from src.training.trainer import (
    CheckpointCleanupCallback,
    EarlyStoppingOnPlateau,
    FinetuneTrainer,
    RichProgressCallback,
)
from src.utils.memory import VRAMProfiler, estimate_model_vram
from src.utils.reproducibility import capture_environment, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    resume_from: str | None = typer.Option(None, "--resume-from", help="Checkpoint path to resume from"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and data without training"),
    profile_memory: bool = typer.Option(False, "--profile-memory", help="Log VRAM before first step"),
    override: list[str] = typer.Option([], "--set", help="Override config values, e.g. training.learning_rate=1e-4"),
) -> None:
    """Fine-tune an LLM with LoRA or QLoRA."""

    # ------------------------------------------------------------------ #
    # 1. Load & merge configuration
    # ------------------------------------------------------------------ #
    base_cfg = OmegaConf.load(Path(__file__).parent.parent / "configs" / "base.yaml")
    task_cfg = OmegaConf.load(config)
    cli_cfg  = OmegaConf.from_dotlist(override)
    cfg = OmegaConf.merge(base_cfg, task_cfg, cli_cfg)

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    if dry_run:
        logger.info("Dry run — validating config and data only.")

    # ------------------------------------------------------------------ #
    # 2. Reproducibility
    # ------------------------------------------------------------------ #
    seed_everything(cfg.get("seed", 42))
    env = capture_environment()
    logger.info("Environment fingerprint: %s", env.get("torch", "N/A"))

    # ------------------------------------------------------------------ #
    # 3. Load tokenizer
    # ------------------------------------------------------------------ #
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------ #
    # 4. Build dataset
    # ------------------------------------------------------------------ #
    ds_cfg = DatasetConfig(
        name_or_path=cfg.data.name_or_path,
        split=cfg.data.get("split", "train"),
        format=DatasetFormat(cfg.data.get("format", "chat")),
        max_samples=cfg.data.get("max_samples"),
        system_prompt=cfg.data.get("system_prompt"),
    )

    train_ds = FineTuneDataset.from_config(
        ds_cfg, tokenizer, max_seq_length=cfg.data.max_seq_length
    )
    logger.info("Dataset stats: %s", train_ds.token_stats())

    if cfg.data.get("packing", True):
        logger.info("Using sequence packing (chunk_size=%d)", cfg.data.max_seq_length)
        train_dataset = PackedDataset(train_ds, chunk_size=cfg.data.max_seq_length)
        collator = SequencePackingCollator()
    else:
        train_dataset = train_ds
        collator = DynamicPaddingCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=cfg.data.get("pad_to_multiple_of", 8),
        )

    if dry_run:
        logger.info("Dry run complete. Dataset OK (%d samples).", len(train_dataset))
        raise typer.Exit(0)

    # ------------------------------------------------------------------ #
    # 5. Load model
    # ------------------------------------------------------------------ #
    import torch
    from transformers import AutoModelForCausalLM

    lora_cfg, quant_cfg = get_preset(cfg.get("preset", "qlora-7b"))
    bnb_config = quant_cfg.to_bnb_config()

    model_kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": getattr(torch, cfg.model.get("torch_dtype", "bfloat16")),
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    attn = cfg.model.get("attn_implementation")
    if attn:
        model_kwargs["attn_implementation"] = attn

    logger.info("Loading model: %s", cfg.model.name_or_path)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **model_kwargs)

    if bnb_config:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.training.get("gradient_checkpointing", True),
        )

    # ------------------------------------------------------------------ #
    # 6. Apply LoRA
    # ------------------------------------------------------------------ #
    from peft import get_peft_model
    peft_config = lora_cfg.to_peft_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if profile_memory:
        vram_profiler = VRAMProfiler()
        vram_profiler.log_snapshot("post-model-load")
    else:
        vram_profiler = None

    # ------------------------------------------------------------------ #
    # 7. Training arguments
    # ------------------------------------------------------------------ #
    from transformers import TrainingArguments

    output_dir = cfg.training.get("output_dir", f"outputs/{config.stem}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.get("lr_scheduler", "cosine"),
        warmup_ratio=cfg.training.get("warmup_ratio", 0.03),
        max_grad_norm=cfg.training.get("max_grad_norm", 1.0),
        fp16=cfg.training.get("fp16", False),
        bf16=cfg.training.get("bf16", True),
        gradient_checkpointing=cfg.training.get("gradient_checkpointing", True),
        dataloader_num_workers=cfg.training.get("dataloader_num_workers", 2),
        save_strategy=cfg.training.get("save_strategy", "steps"),
        save_steps=cfg.training.get("save_steps", 250),
        logging_steps=cfg.training.get("logging_steps", 10),
        report_to=cfg.training.get("report_to", ["none"]),
        remove_unused_columns=False,
        label_names=["labels"],
        ddp_find_unused_parameters=False,
    )

    # ------------------------------------------------------------------ #
    # 8. Build trainer
    # ------------------------------------------------------------------ #
    callbacks = [
        RichProgressCallback(),
        CheckpointCleanupCallback(keep_best=3),
    ]
    if cfg.training.get("early_stopping_patience"):
        callbacks.append(
            EarlyStoppingOnPlateau(
                patience=cfg.training.early_stopping_patience
            )
        )

    trainer = FinetuneTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
        vram_profiler=vram_profiler,
    )

    # ------------------------------------------------------------------ #
    # 9. Train
    # ------------------------------------------------------------------ #
    logger.info("Starting training (output_dir=%s)", output_dir)

    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # ------------------------------------------------------------------ #
    # 10. Save final adapter
    # ------------------------------------------------------------------ #
    adapter_dir = Path(output_dir) / "final_adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Adapter saved to %s", adapter_dir)

    # Save environment snapshot alongside the adapter
    import json
    (adapter_dir / "environment.json").write_text(json.dumps(env, indent=2, default=str))

    if vram_profiler:
        vram_profiler.log_snapshot("training-complete")


if __name__ == "__main__":
    app()
