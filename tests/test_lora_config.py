"""Tests for LoRA / QLoRA configuration builders."""

from __future__ import annotations

import pytest

from src.training.lora_config import (
    PRESETS,
    LoRAConfig,
    QuantisationConfig,
    QuantisationType,
    get_preset,
    lora_7b,
    qlora_7b,
    qlora_13b,
    qlora_70b,
)


class TestLoRAConfig:
    def test_effective_scale(self):
        cfg = LoRAConfig(r=64, alpha=128)
        assert cfg.effective_scale == 2.0

    def test_effective_scale_unity(self):
        cfg = LoRAConfig(r=16, alpha=16)
        assert cfg.effective_scale == 1.0

    def test_trainable_param_estimate(self):
        cfg = LoRAConfig(r=64, alpha=128)
        n = cfg.trainable_param_estimate(hidden_size=4096, num_layers=32)
        # Should be in the millions for a 7B model
        assert n > 1_000_000
        assert n < 1_000_000_000  # but much less than total params

    def test_default_target_modules(self):
        cfg = LoRAConfig()
        assert "q_proj" in cfg.target_modules
        assert "v_proj" in cfg.target_modules
        assert "down_proj" in cfg.target_modules

    def test_to_peft_config_import_error(self):
        """to_peft_config should raise ImportError if peft not installed."""
        import sys
        import importlib
        peft_backup = sys.modules.get("peft")
        sys.modules["peft"] = None  # type: ignore
        cfg = LoRAConfig()
        with pytest.raises((ImportError, TypeError)):
            cfg.to_peft_config()
        # Restore
        if peft_backup is not None:
            sys.modules["peft"] = peft_backup
        else:
            del sys.modules["peft"]


class TestQuantisationConfig:
    def test_nf4_bits(self):
        cfg = QuantisationConfig(quant_type=QuantisationType.NF4)
        assert cfg.bits == 4

    def test_int8_bits(self):
        cfg = QuantisationConfig(quant_type=QuantisationType.INT8)
        assert cfg.bits == 8

    def test_none_bits(self):
        cfg = QuantisationConfig(quant_type=QuantisationType.NONE)
        assert cfg.bits == 16

    def test_vram_multiplier_nf4(self):
        cfg = QuantisationConfig(quant_type=QuantisationType.NF4)
        assert cfg.vram_multiplier() == pytest.approx(0.25)

    def test_vram_multiplier_none(self):
        cfg = QuantisationConfig(quant_type=QuantisationType.NONE)
        assert cfg.vram_multiplier() == pytest.approx(1.0)

    def test_to_bnb_config_returns_none_for_no_quant(self):
        cfg = QuantisationConfig(quant_type=QuantisationType.NONE)
        assert cfg.to_bnb_config() is None


class TestPresets:
    def test_all_presets_return_tuples(self):
        for name in PRESETS:
            lora, quant = get_preset(name)
            assert isinstance(lora, LoRAConfig)
            assert isinstance(quant, QuantisationConfig)

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent-preset")

    def test_lora_7b_no_quantisation(self):
        _, quant = lora_7b()
        assert quant.quant_type == QuantisationType.NONE

    def test_qlora_7b_uses_nf4(self):
        _, quant = qlora_7b()
        assert quant.quant_type == QuantisationType.NF4

    def test_qlora_70b_reduced_rank(self):
        lora, _ = qlora_70b()
        assert lora.r <= 16

    def test_qlora_70b_reduced_targets(self):
        lora, _ = qlora_70b()
        # 70B preset reduces target modules to fit VRAM
        assert len(lora.target_modules) < len(LoRAConfig().target_modules)
