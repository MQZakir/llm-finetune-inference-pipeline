"""
Reproducibility — seed management, deterministic ops, environment capture.

Setting a seed alone is not sufficient for full reproducibility in PyTorch.
This module handles all sources of non-determinism, including CUDA kernels
and cuDNN algorithm selection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import sys

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all random seeds and optionally enable deterministic CUDA ops.

    Parameters
    ----------
    seed : int
        The seed value. Default 42.
    deterministic : bool
        If True, enables ``torch.use_deterministic_algorithms(True)`` and
        sets ``CUBLAS_WORKSPACE_CONFIG``. This may reduce training speed by
        5–15% but ensures bit-exact reproducibility across runs.

    Notes
    -----
    ``torch.backends.cudnn.benchmark = False`` disables cuDNN autotuning,
    which selects different kernels depending on input shape. Without this,
    two runs with different batch sizes may produce different results even
    with the same seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic

        if deterministic:
            # Required for deterministic CUDA operations
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except AttributeError:
                pass  # older PyTorch
    except ImportError:
        pass

    logger.info("Seeds set to %d (deterministic=%s)", seed, deterministic)


def capture_environment() -> dict:
    """
    Capture a snapshot of the full software environment for reproducibility.

    Returns a dict suitable for saving alongside model checkpoints, enabling
    exact environment reconstruction after the fact.
    """
    env: dict = {
        "platform": platform.platform(),
        "python": sys.version,
        "packages": _get_relevant_packages(),
    }

    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["cudnn_version"] = str(torch.backends.cudnn.version())
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass

    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        pass

    try:
        import peft
        env["peft"] = peft.__version__
    except ImportError:
        pass

    return env


def env_fingerprint(env: dict | None = None) -> str:
    """Return a short hash of the environment for change detection."""
    env = env or capture_environment()
    serialised = json.dumps(env, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:12]


def _get_relevant_packages() -> dict[str, str]:
    """Get versions of ML-relevant installed packages."""
    packages = [
        "torch", "transformers", "peft", "trl", "datasets",
        "accelerate", "bitsandbytes", "safetensors", "tokenizers",
        "llama-cpp-python", "numpy", "scipy",
    ]
    installed = {}
    try:
        from importlib.metadata import PackageNotFoundError, version
        for pkg in packages:
            try:
                installed[pkg] = version(pkg)
            except PackageNotFoundError:
                pass
    except ImportError:
        pass
    return installed
