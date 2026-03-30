"""
GGUF quantisation — export a HuggingFace model to llama.cpp format.

Supported quantisation types (ordered best → most compressed):
  Q8_0   — 8-bit, minimal quality loss
  Q6_K   — 6-bit K-quants
  Q5_K_M — 5-bit K-quants (medium)
  Q4_K_M — 4-bit K-quants (medium) ← recommended default
  Q4_K_S — 4-bit K-quants (small)
  Q3_K_M — 3-bit K-quants (medium)
  Q2_K   — 2-bit, significant quality loss

K-quants use a block-wise approach where a shared scale factor is computed
per 256-weight block, producing better quality than naive per-weight quantisation.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class GGUFQuantType(str, Enum):
    Q8_0   = "Q8_0"
    Q6_K   = "Q6_K"
    Q5_K_M = "Q5_K_M"
    Q4_K_M = "Q4_K_M"
    Q4_K_S = "Q4_K_S"
    Q3_K_M = "Q3_K_M"
    Q2_K   = "Q2_K"
    F16    = "f16"      # 16-bit float (lossless, large)
    F32    = "f32"      # 32-bit float (lossless, very large)


# Approximate size multipliers vs FP32 (rough guide, varies by model)
QUANT_SIZE_RATIO: dict[GGUFQuantType, float] = {
    GGUFQuantType.Q8_0:   0.500,
    GGUFQuantType.Q6_K:   0.375,
    GGUFQuantType.Q5_K_M: 0.312,
    GGUFQuantType.Q4_K_M: 0.250,
    GGUFQuantType.Q4_K_S: 0.230,
    GGUFQuantType.Q3_K_M: 0.188,
    GGUFQuantType.Q2_K:   0.125,
    GGUFQuantType.F16:    0.500,
    GGUFQuantType.F32:    1.000,
}


def export_to_gguf(
    model_path: str | Path,
    output_path: str | Path,
    quant_type: GGUFQuantType = GGUFQuantType.Q4_K_M,
    llama_cpp_dir: str | Path | None = None,
    keep_f16: bool = False,
) -> Path:
    """
    Convert a HuggingFace model directory to a quantised GGUF file.

    This is a two-step process:
      1. Convert the HF SafeTensors to a float16 GGUF (lossless)
      2. Quantise the F16 GGUF to the target format

    Parameters
    ----------
    model_path : str | Path
        Path to the HuggingFace model directory (with config.json, *.safetensors).
    output_path : str | Path
        Destination path for the quantised .gguf file.
    quant_type : GGUFQuantType
        Target quantisation format.
    llama_cpp_dir : str | Path | None
        Directory of a compiled llama.cpp installation.
        If None, attempts to find ``convert_hf_to_gguf.py`` and ``llama-quantize``
        on the PATH or via the ``llama-cpp-python`` package.
    keep_f16 : bool
        If True, keep the intermediate F16 GGUF file.

    Returns
    -------
    Path
        Path to the final quantised GGUF file.
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assert model_path.exists(), f"Model directory not found: {model_path}"

    # Locate llama.cpp tools
    convert_script, quantize_bin = _find_llama_cpp_tools(llama_cpp_dir)

    # Step 1: Convert to F16 GGUF
    f16_path = output_path.with_suffix("").with_name(output_path.stem + "_f16.gguf")
    logger.info("Converting %s → F16 GGUF ...", model_path)
    _run(
        [sys.executable, str(convert_script), str(model_path),
         "--outfile", str(f16_path), "--outtype", "f16"],
        desc="HF → F16 GGUF conversion",
    )

    # Step 2: Quantise to target format
    if quant_type in (GGUFQuantType.F16, GGUFQuantType.F32):
        shutil.copy(f16_path, output_path)
    else:
        logger.info("Quantising %s → %s ...", f16_path.name, quant_type.value)
        _run(
            [str(quantize_bin), str(f16_path), str(output_path), quant_type.value],
            desc=f"GGUF quantisation ({quant_type.value})",
        )

    if not keep_f16 and f16_path.exists() and f16_path != output_path:
        f16_path.unlink()

    size_mb = output_path.stat().st_size / 1_048_576
    logger.info("✓ GGUF export complete: %s (%.1f MB)", output_path, size_mb)
    return output_path


def estimate_gguf_size(
    model_path: str | Path,
    quant_type: GGUFQuantType = GGUFQuantType.Q4_K_M,
) -> float:
    """
    Estimate the output GGUF file size in GB based on the source model.

    Returns
    -------
    float
        Estimated size in gigabytes.
    """
    model_path = Path(model_path)
    total_bytes = sum(
        f.stat().st_size
        for f in model_path.rglob("*.safetensors")
    )
    total_bytes += sum(
        f.stat().st_size
        for f in model_path.rglob("pytorch_model*.bin")
    )
    fp32_gb = total_bytes / 1e9
    ratio = QUANT_SIZE_RATIO.get(quant_type, 0.25)
    return fp32_gb * ratio


def quantisation_comparison(model_path: str | Path) -> list[dict]:
    """
    Return a table comparing estimated sizes for all quantisation types.

    Example
    -------
    >>> rows = quantisation_comparison("outputs/merged_model")
    >>> for r in rows:
    ...     print(f"{r['type']:12s}  ~{r['size_gb']:.1f} GB  ({r['quality']})")
    """
    quality = {
        GGUFQuantType.F32:    "lossless (huge)",
        GGUFQuantType.F16:    "lossless (large)",
        GGUFQuantType.Q8_0:   "near-lossless",
        GGUFQuantType.Q6_K:   "excellent",
        GGUFQuantType.Q5_K_M: "very good",
        GGUFQuantType.Q4_K_M: "good (recommended)",
        GGUFQuantType.Q4_K_S: "good (smaller)",
        GGUFQuantType.Q3_K_M: "moderate",
        GGUFQuantType.Q2_K:   "low",
    }
    return [
        {
            "type":    qt.value,
            "size_gb": estimate_gguf_size(model_path, qt),
            "quality": quality.get(qt, ""),
        }
        for qt in GGUFQuantType
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_llama_cpp_tools(
    llama_cpp_dir: str | Path | None,
) -> tuple[Path, Path]:
    """
    Locate ``convert_hf_to_gguf.py`` and ``llama-quantize`` (or ``quantize``).
    """
    if llama_cpp_dir:
        base = Path(llama_cpp_dir)
        convert = base / "convert_hf_to_gguf.py"
        quantize = base / "llama-quantize"
        if not quantize.exists():
            quantize = base / "quantize"
        return convert, quantize

    # Try llama-cpp-python installed package
    try:
        import llama_cpp
        pkg_dir = Path(llama_cpp.__file__).parent
        convert = pkg_dir / "convert_hf_to_gguf.py"
        quantize = pkg_dir / "llama-quantize"
        if convert.exists() and quantize.exists():
            return convert, quantize
    except ImportError:
        pass

    # Try PATH
    quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")
    convert_script = shutil.which("convert_hf_to_gguf.py")

    if quantize_bin and convert_script:
        return Path(convert_script), Path(quantize_bin)

    raise FileNotFoundError(
        "Could not find llama.cpp tools. Options:\n"
        "  1. Install llama-cpp-python: pip install llama-cpp-python\n"
        "  2. Build llama.cpp from source and pass llama_cpp_dir=...\n"
        "  3. Add llama-quantize and convert_hf_to_gguf.py to your PATH"
    )


def _run(cmd: list[str], desc: str) -> None:
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{desc} failed (exit {result.returncode}):\n{result.stderr}"
        )
