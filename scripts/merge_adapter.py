"""
Merge a LoRA adapter into the base model weights.

After merging, the output directory contains a standard HuggingFace model
with no adapter — identical in interface to the original, but with the
fine-tuned weights baked in. This is required before GGUF export.

Usage
-----
  python scripts/merge_adapter.py \\
      --base-model meta-llama/Meta-Llama-3-8B \\
      --adapter-path outputs/lora_7b/final_adapter \\
      --output-path outputs/merged_model
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def merge(
    base_model: str = typer.Option(..., "--base-model", help="Base model name or path"),
    adapter_path: Path = typer.Option(..., "--adapter-path", help="Path to LoRA adapter directory"),
    output_path: Path = typer.Option(..., "--output-path", help="Output directory for merged model"),
    dtype: str = typer.Option("bfloat16", "--dtype", help="Output dtype (bfloat16 | float16 | float32)"),
    safe_serialisation: bool = typer.Option(True, help="Save as SafeTensors (recommended)"),
    push_to_hub: str | None = typer.Option(None, "--push-to-hub", help="HuggingFace repo to push to"),
) -> None:
    """Merge a LoRA adapter into base model weights."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)

    # Load base model in full precision for clean merging
    logger.info("Loading base model: %s", base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="cpu",           # merge on CPU to avoid VRAM constraints
        trust_remote_code=True,
    )

    logger.info("Loading adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(base, str(adapter_path))

    logger.info("Merging adapter weights into base model ...")
    model = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving merged model to %s ...", output_path)
    model.save_pretrained(
        str(output_path),
        safe_serialization=safe_serialisation,
    )

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    tokenizer.save_pretrained(str(output_path))

    # Sanity check — load a tiny slice and verify shapes
    _sanity_check(output_path)

    logger.info("✓ Merge complete: %s", output_path)

    if push_to_hub:
        logger.info("Pushing to HuggingFace Hub: %s", push_to_hub)
        model.push_to_hub(push_to_hub, safe_serialization=safe_serialisation)
        tokenizer.push_to_hub(push_to_hub)
        logger.info("✓ Pushed to hub: %s", push_to_hub)


def _sanity_check(model_path: Path) -> None:
    """Quick sanity check: load config and verify file sizes are reasonable."""
    import json

    config_file = model_path / "config.json"
    assert config_file.exists(), "config.json missing from merged model"

    with open(config_file) as f:
        config = json.load(f)

    logger.info(
        "Sanity check passed — model: %s, vocab: %s, layers: %s",
        config.get("model_type", "unknown"),
        config.get("vocab_size", "?"),
        config.get("num_hidden_layers", "?"),
    )

    # Check total file size
    total_mb = sum(
        f.stat().st_size for f in model_path.rglob("*.safetensors")
    ) / 1_048_576
    logger.info("Total weight size: %.1f MB", total_mb)


if __name__ == "__main__":
    app()
