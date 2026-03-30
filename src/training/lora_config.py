"""
LoRA and QLoRA configuration builders.

Provides opinionated factory functions for common model families with
sensible defaults based on published recipes, plus a fully-custom path.

Theory
------
LoRA parameterises the weight update as  ΔW = B · A  where:
  B ∈ ℝ^{d × r},  A ∈ ℝ^{r × k},  r ≪ min(d, k)

The adapter is scaled by  α / r  before being added to W₀.  A higher
α/r ratio means the adapter has more influence; most recipes keep α = 2r.

QLoRA adds:
  - 4-bit NF4 quantisation of the frozen base model weights
  - Double quantisation (quantise the quantisation constants themselves)
  - Paged AdamW to handle occasional VRAM spikes without OOM crashes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class QuantisationType(str, Enum):
    NF4 = "nf4"          # NormalFloat4 — best quality for normally-distributed weights
    FP4 = "fp4"          # Standard 4-bit float
    INT8 = "int8"        # 8-bit integer quantisation
    NONE = "none"        # No quantisation (full LoRA)


@dataclass
class LoRAConfig:
    """
    Full specification for a LoRA adapter.

    Parameters
    ----------
    r : int
        Rank of the decomposition.  Typical values: 8, 16, 32, 64, 128.
        Higher r → more parameters, more expressive, but slower training.
    alpha : float
        LoRA scaling factor.  The effective scale applied to ΔW is α/r.
        Convention: alpha = 2 * r (doubles the effective learning rate of
        the adapter versus the scale at r=alpha).
    dropout : float
        Dropout applied to the low-rank matrices during training.
        0.0 for most cases; 0.05–0.1 for small datasets.
    target_modules : list[str]
        Which weight matrices to inject adapters into.  Defaults cover the
        full attention mechanism + FFN gate/up/down projections.
    bias : str
        'none'  — do not train any bias terms (recommended)
        'all'   — train all bias terms
        'lora_only' — train only adapter bias terms
    task_type : str
        PEFT task type.  Almost always 'CAUSAL_LM' for decoder models.
    modules_to_save : list[str]
        Modules to unfreeze and train in full (not LoRA).  Typically the
        embedding layer and LM head when the vocabulary is extended.
    """

    r: int = 64
    alpha: float = 128.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: list[str] = field(default_factory=list)

    def to_peft_config(self):
        """Convert to a ``peft.LoraConfig`` object."""
        try:
            from peft import LoraConfig, TaskType
        except ImportError as e:
            raise ImportError("Install peft: pip install peft") from e

        return LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=self.modules_to_save or None,
        )

    @property
    def effective_scale(self) -> float:
        """The scaling factor applied to ΔW = α / r."""
        return self.alpha / self.r

    def trainable_param_estimate(self, hidden_size: int, num_layers: int) -> int:
        """
        Rough estimate of trainable parameter count.

        Counts only q/k/v/o projections (hidden_size × hidden_size each)
        and gate/up/down projections (hidden_size × 4*hidden_size each).
        """
        attn_mods = sum(1 for m in self.target_modules if m in {"q_proj", "k_proj", "v_proj", "o_proj"})
        ffn_mods  = sum(1 for m in self.target_modules if m in {"gate_proj", "up_proj", "down_proj"})

        attn_params = attn_mods * 2 * self.r * hidden_size
        ffn_params  = ffn_mods  * 2 * self.r * hidden_size * 4  # FFN is 4× hidden
        return (attn_params + ffn_params) * num_layers


@dataclass
class QuantisationConfig:
    """
    BitsAndBytes quantisation settings for QLoRA.

    The NF4 data type is specifically designed for normally-distributed
    weights (like those in transformer models trained with Adam), and
    provides better quality than FP4 at the same bit-width.
    """

    quant_type: QuantisationType = QuantisationType.NF4
    compute_dtype: str = "bfloat16"      # dtype for forward pass computation
    double_quant: bool = True            # quantise quantisation constants → saves ~0.5 bits
    quant_storage: str = "uint8"

    def to_bnb_config(self):
        """Convert to a ``transformers.BitsAndBytesConfig`` object."""
        if self.quant_type == QuantisationType.NONE:
            return None
        try:
            import torch
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError("Install bitsandbytes: pip install bitsandbytes") from e

        compute_dtype = getattr(torch, self.compute_dtype)

        if self.quant_type == QuantisationType.INT8:
            return BitsAndBytesConfig(load_in_8bit=True)

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.quant_type.value,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.double_quant,
            bnb_4bit_quant_storage=getattr(torch, self.quant_storage),
        )

    @property
    def bits(self) -> int:
        return {
            QuantisationType.NF4: 4,
            QuantisationType.FP4: 4,
            QuantisationType.INT8: 8,
            QuantisationType.NONE: 16,
        }[self.quant_type]

    def vram_multiplier(self) -> float:
        """Approximate VRAM reduction vs BF16 baseline."""
        return self.bits / 16.0


# ---------------------------------------------------------------------------
# Opinionated factory presets
# ---------------------------------------------------------------------------

def lora_7b() -> tuple[LoRAConfig, QuantisationConfig]:
    """
    LoRA recipe for 7–8B models (e.g. Llama-3-8B, Mistral-7B).
    Requires ~20 GB VRAM in BF16.
    """
    return (
        LoRAConfig(r=64, alpha=128, dropout=0.05),
        QuantisationConfig(quant_type=QuantisationType.NONE),
    )


def qlora_7b() -> tuple[LoRAConfig, QuantisationConfig]:
    """
    QLoRA recipe for 7–8B models.
    Requires ~10 GB VRAM with 4-bit NF4.
    """
    return (
        LoRAConfig(r=64, alpha=128, dropout=0.05),
        QuantisationConfig(quant_type=QuantisationType.NF4),
    )


def qlora_13b() -> tuple[LoRAConfig, QuantisationConfig]:
    """
    QLoRA recipe for 13B models (e.g. Llama-2-13B).
    Requires ~14 GB VRAM with 4-bit NF4.
    """
    return (
        LoRAConfig(r=32, alpha=64, dropout=0.05),
        QuantisationConfig(quant_type=QuantisationType.NF4),
    )


def qlora_70b() -> tuple[LoRAConfig, QuantisationConfig]:
    """
    QLoRA recipe for 70B models.
    Requires ~48 GB VRAM (2× A100-40GB or 1× A100-80GB) with 4-bit NF4.
    """
    return (
        LoRAConfig(
            r=16,
            alpha=32,
            dropout=0.0,
            target_modules=["q_proj", "v_proj"],   # reduced targets to fit VRAM
        ),
        QuantisationConfig(quant_type=QuantisationType.NF4),
    )


def lora_code() -> tuple[LoRAConfig, QuantisationConfig]:
    """
    LoRA recipe optimised for code generation tasks.
    Higher rank to capture syntactic patterns.
    """
    return (
        LoRAConfig(
            r=128,
            alpha=256,
            dropout=0.0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "embed_tokens", "lm_head",
            ],
            modules_to_save=["embed_tokens", "lm_head"],
        ),
        QuantisationConfig(quant_type=QuantisationType.NONE),
    )


PRESETS = {
    "lora-7b":   lora_7b,
    "qlora-7b":  qlora_7b,
    "qlora-13b": qlora_13b,
    "qlora-70b": qlora_70b,
    "lora-code": lora_code,
}


def get_preset(name: str) -> tuple[LoRAConfig, QuantisationConfig]:
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}"
        )
    return PRESETS[name]()


def log_trainable_params(model) -> None:
    """Print a summary of trainable vs frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(total, 1),
    )
