"""
VRAM and system memory profiling utilities.

Provides tools to measure peak GPU memory usage during training and inference,
detect OOM risks before they occur, and log memory stats to experiment trackers.
"""

from __future__ import annotations

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """A point-in-time memory measurement."""
    timestamp: float
    allocated_gb: float
    reserved_gb: float
    peak_allocated_gb: float
    free_gb: float
    total_gb: float
    system_ram_gb: float | None = None

    @property
    def utilisation_pct(self) -> float:
        return 100.0 * self.allocated_gb / max(self.total_gb, 1e-6)


class VRAMProfiler:
    """
    Tracks GPU memory usage over the course of a training run.

    Usage
    -----
    >>> profiler = VRAMProfiler(device=0)
    >>> profiler.reset_peak()
    >>> # ... training step ...
    >>> snap = profiler.snapshot()
    >>> print(f"Peak: {snap.peak_allocated_gb:.2f} GB")
    """

    def __init__(self, device: int | str = 0) -> None:
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            self._device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
            self._torch = torch
            self._available = True
        except (ImportError, RuntimeError):
            self._available = False

    def reset_peak(self) -> None:
        if self._available:
            self._torch.cuda.reset_peak_memory_stats(self._device)

    def current_gb(self) -> float:
        if not self._available:
            return 0.0
        return self._torch.cuda.memory_allocated(self._device) / 1e9

    def peak_gb(self) -> float:
        if not self._available:
            return 0.0
        return self._torch.cuda.max_memory_allocated(self._device) / 1e9

    def snapshot(self) -> MemorySnapshot:
        if not self._available:
            return MemorySnapshot(
                timestamp=time.time(),
                allocated_gb=0, reserved_gb=0, peak_allocated_gb=0,
                free_gb=0, total_gb=0,
            )

        allocated = self._torch.cuda.memory_allocated(self._device)
        reserved  = self._torch.cuda.memory_reserved(self._device)
        peak      = self._torch.cuda.max_memory_allocated(self._device)
        total     = self._torch.cuda.get_device_properties(self._device).total_memory
        free      = total - allocated

        system_ram = _get_system_ram_gb()

        return MemorySnapshot(
            timestamp=time.time(),
            allocated_gb=allocated / 1e9,
            reserved_gb=reserved / 1e9,
            peak_allocated_gb=peak / 1e9,
            free_gb=free / 1e9,
            total_gb=total / 1e9,
            system_ram_gb=system_ram,
        )

    def log_snapshot(self, tag: str = "") -> MemorySnapshot:
        snap = self.snapshot()
        label = f"[{tag}] " if tag else ""
        logger.info(
            "%sVRAM: %.2f GB allocated / %.2f GB total (%.1f%% | peak %.2f GB)",
            label,
            snap.allocated_gb,
            snap.total_gb,
            snap.utilisation_pct,
            snap.peak_allocated_gb,
        )
        return snap

    @contextmanager
    def track(self, tag: str = "") -> Generator[MemorySnapshot, None, None]:
        """Context manager that resets peak stats and logs before/after."""
        self.reset_peak()
        before = self.log_snapshot(f"{tag}:before")
        try:
            yield before
        finally:
            self.log_snapshot(f"{tag}:after")


def estimate_model_vram(
    num_params: int,
    dtype_bytes: int = 2,        # 2 for BF16/FP16, 4 for FP32
    optimizer: bool = True,      # include Adam optimizer states
    gradients: bool = True,
    activations_gb: float = 2.0, # rough activation memory estimate
) -> dict[str, float]:
    """
    Estimate VRAM requirements for training a model.

    Adam optimizer stores two extra FP32 states per parameter
    (first and second moment), so the optimizer overhead is 8 bytes/param.

    Returns a dict with keys: model_gb, optimizer_gb, gradients_gb,
    activations_gb, total_gb.
    """
    model_gb = num_params * dtype_bytes / 1e9
    optimizer_gb = (num_params * 8 / 1e9) if optimizer else 0.0
    grad_gb = (num_params * dtype_bytes / 1e9) if gradients else 0.0

    return {
        "model_gb":      round(model_gb, 2),
        "optimizer_gb":  round(optimizer_gb, 2),
        "gradients_gb":  round(grad_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "total_gb":      round(model_gb + optimizer_gb + grad_gb + activations_gb, 2),
    }


def clear_gpu_cache() -> None:
    """Aggressively free all unused GPU memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def _get_system_ram_gb() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().used / 1e9
    except ImportError:
        return None


@contextmanager
def oom_guard(fallback_fn=None) -> Generator[None, None, None]:
    """
    Catches CUDA OOM errors and optionally calls a fallback.

    Usage
    -----
    >>> with oom_guard(fallback_fn=lambda: reduce_batch_size()):
    ...     outputs = model(**inputs)
    """
    try:
        yield
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("CUDA OOM! Clearing cache and retrying via fallback.")
            clear_gpu_cache()
            if fallback_fn is not None:
                fallback_fn()
            else:
                raise
        else:
            raise
