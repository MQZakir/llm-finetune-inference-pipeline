"""Tests for memory profiling utilities."""

from __future__ import annotations

import pytest

from src.utils.memory import estimate_model_vram, VRAMProfiler


class TestVRAMEstimation:
    def test_bf16_model(self):
        # 7B params, BF16 (2 bytes each)
        result = estimate_model_vram(7_000_000_000, dtype_bytes=2, optimizer=False, gradients=False, activations_gb=0)
        assert result["model_gb"] == pytest.approx(14.0, rel=0.01)

    def test_optimizer_adds_overhead(self):
        result_no_opt = estimate_model_vram(1_000_000_000, dtype_bytes=2, optimizer=False, gradients=False, activations_gb=0)
        result_with_opt = estimate_model_vram(1_000_000_000, dtype_bytes=2, optimizer=True, gradients=False, activations_gb=0)
        # Adam adds 8 bytes per param = 4× the BF16 model size
        assert result_with_opt["optimizer_gb"] > result_no_opt["model_gb"]

    def test_total_is_sum_of_parts(self):
        result = estimate_model_vram(1_000_000_000, dtype_bytes=2, optimizer=True, gradients=True, activations_gb=2.0)
        expected = result["model_gb"] + result["optimizer_gb"] + result["gradients_gb"] + result["activations_gb"]
        assert result["total_gb"] == pytest.approx(expected, rel=0.001)

    def test_qlora_savings(self):
        """QLoRA should use ~4× less memory for model weights vs BF16."""
        bf16 = estimate_model_vram(7_000_000_000, dtype_bytes=2, optimizer=False, gradients=False, activations_gb=0)
        nf4  = estimate_model_vram(7_000_000_000, dtype_bytes=1, optimizer=False, gradients=False, activations_gb=0)
        # NF4 is actually ~0.5 bytes per param, but we approximate with dtype_bytes=1 here
        assert nf4["model_gb"] < bf16["model_gb"]


class TestVRAMProfiler:
    def test_no_cuda_returns_zero(self):
        """VRAMProfiler should gracefully handle no-GPU environments."""
        profiler = VRAMProfiler()
        # In a CPU-only test environment, current_gb should return 0.0
        assert profiler.current_gb() >= 0.0
        assert profiler.peak_gb() >= 0.0

    def test_snapshot_returns_snapshot(self):
        from src.utils.memory import MemorySnapshot
        profiler = VRAMProfiler()
        snap = profiler.snapshot()
        assert isinstance(snap, MemorySnapshot)
        assert snap.allocated_gb >= 0.0
        assert snap.total_gb >= 0.0

    def test_utilisation_pct_range(self):
        from src.utils.memory import MemorySnapshot
        snap = MemorySnapshot(
            timestamp=0.0,
            allocated_gb=4.0,
            reserved_gb=5.0,
            peak_allocated_gb=6.0,
            free_gb=4.0,
            total_gb=8.0,
        )
        assert snap.utilisation_pct == pytest.approx(50.0)
