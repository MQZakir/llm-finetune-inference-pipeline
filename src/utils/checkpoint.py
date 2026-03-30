"""
Safe checkpoint save / resume utilities.

HuggingFace Trainer handles most of this, but this module provides
additional safety for multi-process saves and checkpoint validation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    path: Path
    step: int
    epoch: float
    eval_loss: float | None
    train_loss: float | None
    timestamp: str | None

    @classmethod
    def from_directory(cls, path: Path) -> "CheckpointInfo | None":
        """Parse a HuggingFace Trainer checkpoint directory."""
        try:
            step = int(path.name.split("-")[-1])
        except (ValueError, IndexError):
            return None

        trainer_state = path / "trainer_state.json"
        if not trainer_state.exists():
            return cls(path=path, step=step, epoch=0.0, eval_loss=None, train_loss=None, timestamp=None)

        state = json.loads(trainer_state.read_text())
        log_history = state.get("log_history", [])

        eval_loss = None
        train_loss = None
        for entry in reversed(log_history):
            if "eval_loss" in entry and eval_loss is None:
                eval_loss = entry["eval_loss"]
            if "loss" in entry and train_loss is None:
                train_loss = entry["loss"]
            if eval_loss is not None and train_loss is not None:
                break

        return cls(
            path=path,
            step=step,
            epoch=state.get("epoch", 0.0),
            eval_loss=eval_loss,
            train_loss=train_loss,
            timestamp=None,
        )


def list_checkpoints(output_dir: str | Path) -> list[CheckpointInfo]:
    """List all valid checkpoints in an output directory, sorted by step."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []

    checkpoints = []
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            info = CheckpointInfo.from_directory(d)
            if info is not None:
                checkpoints.append(info)

    return sorted(checkpoints, key=lambda c: c.step)


def best_checkpoint(
    output_dir: str | Path,
    metric: str = "eval_loss",
    mode: str = "min",
) -> CheckpointInfo | None:
    """Return the checkpoint with the best score on a given metric."""
    checkpoints = list_checkpoints(output_dir)
    valid = [c for c in checkpoints if getattr(c, metric.replace("eval_", "eval_"), None) is not None]

    if not valid:
        # Fallback to last checkpoint
        return checkpoints[-1] if checkpoints else None

    return min(valid, key=lambda c: c.eval_loss or float("inf")) if mode == "min" \
        else max(valid, key=lambda c: c.eval_loss or float("-inf"))


def safe_save_model(
    model,
    output_dir: str | Path,
    tokenizer=None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Save model and tokenizer with an atomic write pattern.

    Writes to a temp directory first, then renames — prevents partial
    writes from corrupting the checkpoint if the process is killed mid-save.
    """
    output_dir = Path(output_dir)
    tmp_dir = output_dir.with_suffix(".tmp")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    logger.info("Saving model to %s ...", output_dir)
    model.save_pretrained(str(tmp_dir), safe_serialization=True)

    if tokenizer is not None:
        tokenizer.save_pretrained(str(tmp_dir))

    if metadata:
        (tmp_dir / "training_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )

    # Atomic rename
    if output_dir.exists():
        shutil.rmtree(output_dir)
    tmp_dir.rename(output_dir)

    logger.info("✓ Saved to %s", output_dir)
    return output_dir


def verify_checkpoint(path: str | Path) -> bool:
    """
    Verify a checkpoint's integrity by checking required files exist
    and SafeTensors files are non-empty.
    """
    path = Path(path)
    required = ["config.json"]
    weight_patterns = ["*.safetensors", "pytorch_model*.bin", "adapter_model*.safetensors"]

    for req in required:
        if not (path / req).exists():
            logger.warning("Missing required file: %s/%s", path, req)
            return False

    has_weights = any(
        list(path.glob(pat)) for pat in weight_patterns
    )
    if not has_weights:
        logger.warning("No weight files found in %s", path)
        return False

    # Check no file is suspiciously small (< 1 KB — likely a truncated write)
    for pattern in weight_patterns:
        for f in path.glob(pattern):
            if f.stat().st_size < 1024:
                logger.warning("Suspiciously small file: %s (%d bytes)", f, f.stat().st_size)
                return False

    return True
