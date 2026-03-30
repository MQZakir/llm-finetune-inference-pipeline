"""
Multi-run comparison, statistical significance testing, and result visualisation.

Provides tools to compare multiple model checkpoints or configurations,
determine whether improvements are statistically significant, and generate
comparison tables for reporting.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Results from a single evaluation run."""
    name: str
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute a bootstrap confidence interval for the mean of `scores`.

    Parameters
    ----------
    scores : list[float]
        Sample of per-example scores.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g. 0.95 for 95% CI).

    Returns
    -------
    (lower, upper) confidence interval bounds.
    """
    import random
    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_bootstrap):
        resample = [rng.choice(scores) for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    alpha = 1.0 - ci
    lo = means[int(n_bootstrap * alpha / 2)]
    hi = means[int(n_bootstrap * (1 - alpha / 2))]
    return lo, hi


def paired_t_test(
    scores_a: list[float],
    scores_b: list[float],
) -> tuple[float, float]:
    """
    Paired two-sided t-test between two sets of per-example scores.

    Returns
    -------
    (t_statistic, p_value)
    """
    assert len(scores_a) == len(scores_b), "Score lists must be the same length"
    n = len(scores_a)
    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    mean_d = sum(diffs) / n
    if n < 2:
        return 0.0, 1.0

    std_d = statistics.stdev(diffs)
    if std_d == 0:
        return 0.0, 1.0

    t_stat = mean_d / (std_d / math.sqrt(n))

    # Approximate p-value using Student's t CDF (two-sided)
    # For n > 30, t ≈ z, so we use the normal approximation
    try:
        from scipy import stats
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    except ImportError:
        # Fallback: normal approximation
        z = abs(t_stat)
        p_value = 2 * (1 - _normal_cdf(z))

    return t_stat, p_value


def compare_multiple_runs(runs: list[RunResult]) -> dict[str, list[dict]]:
    """
    Produce a structured comparison table for multiple runs.

    Returns
    -------
    dict mapping metric → list of {run_name, score, rank}
    """
    all_metrics = set()
    for run in runs:
        all_metrics.update(run.metrics.keys())

    comparison: dict[str, list[dict]] = {}
    for metric in sorted(all_metrics):
        entries = []
        for run in runs:
            score = run.metrics.get(metric)
            if score is not None:
                entries.append({"run": run.name, "score": score})

        # Rank by score (higher is better for most metrics, lower for perplexity)
        reverse = "ppl" not in metric.lower()
        entries.sort(key=lambda x: x["score"], reverse=reverse)
        for rank, entry in enumerate(entries, 1):
            entry["rank"] = rank

        comparison[metric] = entries

    return comparison


def print_comparison_table(runs: list[RunResult]) -> None:
    """Print a rich comparison table to stdout."""
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
    except ImportError:
        _print_plain_table(runs)
        return

    comparison = compare_multiple_runs(runs)

    for metric, entries in comparison.items():
        table = Table(title=f"Metric: {metric}")
        table.add_column("Rank", justify="right")
        table.add_column("Model / Run", style="cyan")
        table.add_column("Score", justify="right", style="bold")

        best_score = entries[0]["score"] if entries else 0
        for entry in entries:
            is_best = entry["rank"] == 1
            score_str = f"[green]{entry['score']:.4f}[/green]" if is_best else f"{entry['score']:.4f}"
            table.add_row(str(entry["rank"]), entry["run"], score_str)

        console.print(table)


def _print_plain_table(runs: list[RunResult]) -> None:
    comparison = compare_multiple_runs(runs)
    for metric, entries in comparison.items():
        print(f"\n{'='*50}")
        print(f"Metric: {metric}")
        print(f"{'─'*50}")
        for e in entries:
            print(f"  {e['rank']}. {e['run']:<30} {e['score']:.4f}")


def save_comparison(runs: list[RunResult], output_path: str | Path) -> None:
    """Save full comparison results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "runs": [
            {"name": r.name, "metrics": r.metrics, "metadata": r.metadata}
            for r in runs
        ],
        "comparison": compare_multiple_runs(runs),
    }
    output_path.write_text(json.dumps(data, indent=2))
    logger.info("Comparison saved to %s", output_path)


def _normal_cdf(z: float) -> float:
    """Approximate normal CDF for scipy-free environments."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
