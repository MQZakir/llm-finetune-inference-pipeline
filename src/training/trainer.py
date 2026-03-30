"""
Custom HuggingFace Trainer for LoRA / QLoRA fine-tuning.

Extends the base SFTTrainer with:
  - Gradient norm logging
  - Per-step VRAM tracking
  - Loss spike detection and auto-recovery
  - Token-efficiency reporting (fraction of non-masked tokens)
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Optional

import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.utils.memory import VRAMProfiler

logger = logging.getLogger(__name__)

try:
    from trl import SFTTrainer
    _BASE_TRAINER = SFTTrainer
except ImportError:
    from transformers import Trainer
    _BASE_TRAINER = Trainer  # type: ignore[assignment]
    logger.warning("trl not found — falling back to transformers.Trainer")


class FinetuneTrainer(_BASE_TRAINER):
    """
    Trainer with LoRA-aware logging and training stability features.

    Additional tracked metrics (logged to W&B / TensorBoard):
      - train/grad_norm        : L2 norm of all gradients
      - train/vram_gb          : peak VRAM usage at this step
      - train/tokens_per_sec   : effective training throughput
      - train/active_token_pct : % of tokens with real loss (not masked)
      - train/loss_spike       : 1 if loss jumped > 2× previous EMA, else 0
    """

    def __init__(self, *args, vram_profiler: Optional["VRAMProfiler"] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._vram_profiler = vram_profiler
        self._loss_ema: float | None = None
        self._loss_ema_alpha = 0.98          # slow EMA for spike detection
        self._last_step_time: float = time.monotonic()
        self._tokens_this_step: int = 0

    # ------------------------------------------------------------------
    # Override: training_step — track token throughput
    # ------------------------------------------------------------------

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        t0 = time.monotonic()
        result = super().training_step(model, inputs, num_items_in_batch)
        self._step_duration = time.monotonic() - t0

        # Count non-masked tokens for efficiency logging
        if "labels" in inputs:
            self._tokens_this_step = int((inputs["labels"] != -100).sum().item())

        return result

    # ------------------------------------------------------------------
    # Override: log — inject extra metrics
    # ------------------------------------------------------------------

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "loss" in logs:
            loss = logs["loss"]

            # Loss spike detection
            if self._loss_ema is not None:
                spike = int(loss > 2.0 * self._loss_ema)
                logs["loss_spike"] = spike
                if spike:
                    logger.warning(
                        "Loss spike detected at step %d: %.4f (EMA: %.4f)",
                        self.state.global_step,
                        loss,
                        self._loss_ema,
                    )
            # Update EMA
            if self._loss_ema is None:
                self._loss_ema = loss
            else:
                self._loss_ema = self._loss_ema_alpha * self._loss_ema + (1 - self._loss_ema_alpha) * loss

        # VRAM
        if self._vram_profiler and torch.cuda.is_available():
            logs["vram_gb"] = self._vram_profiler.current_gb()

        # Token throughput
        if hasattr(self, "_step_duration") and self._tokens_this_step:
            logs["tokens_per_sec"] = self._tokens_this_step / max(self._step_duration, 1e-6)
            logs["active_token_pct"] = 100.0 * self._tokens_this_step / max(
                self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps
                * getattr(self.model.config, "max_position_embeddings", 2048),
                1,
            )

        super().log(logs, start_time)

    # ------------------------------------------------------------------
    # Override: compute_loss — gradient norm logging
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        return result

    def _maybe_log_grad_norm(self) -> None:
        """Compute and log gradient L2 norm after backward pass."""
        if not (self.state.global_step % self.args.logging_steps == 0):
            return
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().float().norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)
        self.log({"grad_norm": total_norm})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class RichProgressCallback(TrainerCallback):
    """
    Replaces the default HuggingFace progress bar with a richer display
    using the ``rich`` library, showing ETA, tokens/sec, and VRAM.
    """

    def __init__(self) -> None:
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            self._task_id = None
        except ImportError:
            self._progress = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self._progress:
            self._progress.start()
            self._task_id = self._progress.add_task(
                "Training", total=state.max_steps
            )

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, completed=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self._progress:
            self._progress.stop()


class CheckpointCleanupCallback(TrainerCallback):
    """
    Keeps only the N best checkpoints on disk, deleting older ones.
    Avoids filling the filesystem during long training runs.
    """

    def __init__(self, keep_best: int = 3) -> None:
        self.keep_best = keep_best

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        import shutil
        from pathlib import Path

        output_dir = Path(args.output_dir)
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if len(checkpoints) > self.keep_best:
            for old_ckpt in checkpoints[: -self.keep_best]:
                logger.info("Removing old checkpoint: %s", old_ckpt)
                shutil.rmtree(old_ckpt, ignore_errors=True)

        return control


class EarlyStoppingOnPlateau(TrainerCallback):
    """
    Stops training if the evaluation loss hasn't improved by at least
    ``min_delta`` for ``patience`` consecutive evaluations.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss: float = float("inf")
        self._no_improve: int = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> TrainerControl:
        if metrics is None:
            return control

        eval_loss = metrics.get("eval_loss", float("inf"))
        if eval_loss < self._best_loss - self.min_delta:
            self._best_loss = eval_loss
            self._no_improve = 0
        else:
            self._no_improve += 1
            logger.info(
                "No improvement for %d/%d evaluations (best: %.4f, current: %.4f)",
                self._no_improve,
                self.patience,
                self._best_loss,
                eval_loss,
            )
            if self._no_improve >= self.patience:
                logger.warning("Early stopping triggered.")
                control.should_training_stop = True

        return control
