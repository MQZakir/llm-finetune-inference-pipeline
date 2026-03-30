"""
Integration with EleutherAI's lm-evaluation-harness.

Provides a thin wrapper that runs standard LLM benchmarks
(HellaSwag, ARC, MMLU, TruthfulQA, etc.) against a GGUF or HF model,
and returns structured results.

Usage
-----
>>> from src.evaluation.harness import run_harness_eval
>>> results = run_harness_eval(
...     model_path="outputs/model_q4km.gguf",
...     tasks=["hellaswag", "arc_easy", "mmlu"],
...     num_fewshot=0,
...     limit=500,
... )
>>> print(results["hellaswag"]["acc"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_harness_eval(
    model_path: str,
    tasks: list[str],
    backend: str = "llama-cpp",
    num_fewshot: int = 0,
    limit: int | None = None,
    batch_size: int = 1,
    output_path: str | None = None,
    device: str = "auto",
) -> dict[str, dict[str, float]]:
    """
    Run lm-evaluation-harness against a model.

    Parameters
    ----------
    model_path : str
        Path to a GGUF file (backend='llama-cpp') or HF model dir (backend='hf').
    tasks : list[str]
        Task names from lm-evaluation-harness.
        Common: hellaswag, arc_easy, arc_challenge, mmlu, truthfulqa_mc1,
                winogrande, gsm8k, humaneval
    num_fewshot : int
        Number of few-shot examples. 0 for zero-shot.
    limit : int | None
        Limit evaluation to this many examples per task (for quick checks).
    output_path : str | None
        Save raw results JSON to this path.

    Returns
    -------
    dict mapping task_name → {metric_name → score}
    """
    try:
        import lm_eval
    except ImportError as e:
        raise ImportError(
            "Install lm-evaluation-harness:\n"
            "  pip install lm-eval\n"
            "or: pip install lm-eval[all]"
        ) from e

    logger.info(
        "Running harness eval: tasks=%s, fewshot=%d, limit=%s",
        tasks, num_fewshot, limit,
    )

    if backend == "llama-cpp":
        model_type = "gguf"
        model_args = f"pretrained={model_path}"
    elif backend == "hf":
        model_type = "hf"
        model_args = f"pretrained={model_path},dtype=bfloat16,device_map={device}"
    else:
        raise ValueError(f"Unknown backend: {backend}")

    results = lm_eval.simple_evaluate(
        model=model_type,
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
        log_samples=False,
    )

    structured = _parse_results(results)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(structured, indent=2))
        logger.info("Harness results saved to %s", output_path)

    return structured


def _parse_results(raw: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract metric scores from raw lm-eval output."""
    out: dict[str, dict[str, float]] = {}
    for task, metrics in raw.get("results", {}).items():
        out[task] = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not k.endswith("_stderr")
        }
    return out


# ---------------------------------------------------------------------------
# Convenience: standard benchmark suites
# ---------------------------------------------------------------------------

BENCHMARK_SUITES: dict[str, list[str]] = {
    "quick": [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "winogrande",
    ],
    "standard": [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "mmlu",
        "truthfulqa_mc1",
    ],
    "code": [
        "humaneval",
        "mbpp",
    ],
    "reasoning": [
        "gsm8k",
        "math",
        "bbh",
    ],
}


def run_suite(
    model_path: str,
    suite: str = "quick",
    **kwargs,
) -> dict[str, dict[str, float]]:
    """Run a named benchmark suite."""
    if suite not in BENCHMARK_SUITES:
        raise ValueError(f"Unknown suite '{suite}'. Available: {list(BENCHMARK_SUITES.keys())}")
    return run_harness_eval(model_path, BENCHMARK_SUITES[suite], **kwargs)
