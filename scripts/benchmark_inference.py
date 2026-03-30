"""
Inference throughput and latency benchmarking.

Measures:
  - Time to first token (TTFT) — p50, p95, p99
  - Tokens per second (generation throughput)
  - Memory usage during inference
  - Batch scaling curve

Usage
-----
  python scripts/benchmark_inference.py \\
      --model-path outputs/model_q4km.gguf \\
      --backend llama-cpp \\
      --n-runs 20

  python scripts/benchmark_inference.py \\
      --model-path meta-llama/Meta-Llama-3-8B-Instruct \\
      --backend hf \\
      --batch-sizes 1,2,4,8
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.engine import LlamaCppEngine, HFInferenceEngine, SamplingConfig

logging.basicConfig(level=logging.WARNING)
console = Console()
app = typer.Typer()

TEST_PROMPTS = [
    "Explain the difference between LoRA and full fine-tuning in three paragraphs.",
    "Write a Python function that computes the softmax of a vector.",
    "What are the main advantages of quantisation for LLM inference?",
    "Describe the attention mechanism in transformer models.",
    "How does gradient checkpointing reduce memory usage during training?",
]


@app.command()
def benchmark(
    model_path: str = typer.Option(..., "--model-path"),
    backend: str = typer.Option("llama-cpp", "--backend", help="llama-cpp | hf"),
    n_runs: int = typer.Option(10, "--n-runs"),
    n_gpu_layers: int = typer.Option(-1, "--n-gpu-layers", help="llama-cpp GPU layers (-1=all)"),
    max_tokens: int = typer.Option(200, "--max-tokens"),
    batch_sizes: str = typer.Option("1", "--batch-sizes", help="Comma-separated batch sizes to test"),
    output_json: Path | None = typer.Option(None, "--output-json"),
) -> None:
    """Benchmark inference throughput and latency."""

    console.print(f"[bold]Benchmarking:[/bold] {model_path} ({backend})")

    # Build engine
    if backend == "llama-cpp":
        engine = LlamaCppEngine(model_path, n_gpu_layers=n_gpu_layers, verbose=False)
    elif backend == "hf":
        engine = HFInferenceEngine.from_pretrained(model_path)
    else:
        raise typer.BadParameter(f"Unknown backend: {backend}")

    sampling = SamplingConfig(temperature=0.0, max_tokens=max_tokens, seed=42)
    sizes = [int(x) for x in batch_sizes.split(",")]

    all_results = []

    # Single-prompt latency benchmark
    console.print("\n[dim]Running single-prompt latency benchmark ...[/dim]")
    ttft_ms_list: list[float] = []
    tps_list: list[float] = []

    prompt = TEST_PROMPTS[0]
    for run in range(n_runs):
        t_start = time.perf_counter()
        t_first: float | None = None
        n_tokens = 0

        for token in engine.stream(prompt, sampling):
            if t_first is None:
                t_first = time.perf_counter()
            n_tokens += 1

        t_end = time.perf_counter()
        if t_first:
            ttft_ms_list.append((t_first - t_start) * 1000)
        decode_time = t_end - (t_first or t_start)
        if decode_time > 0:
            tps_list.append(n_tokens / decode_time)

        console.print(
            f"  Run {run+1:2d}/{n_runs}: "
            f"TTFT={ttft_ms_list[-1]:.0f}ms  "
            f"TPS={tps_list[-1]:.1f}",
            end="\r",
        )

    console.print()

    latency_results = {
        "ttft_p50_ms":  statistics.median(ttft_ms_list),
        "ttft_p95_ms":  _percentile(ttft_ms_list, 95),
        "ttft_p99_ms":  _percentile(ttft_ms_list, 99),
        "tps_mean":     statistics.mean(tps_list),
        "tps_p50":      statistics.median(tps_list),
        "tps_p95":      _percentile(tps_list, 95),
        "n_runs":       n_runs,
    }
    all_results.append({"benchmark": "latency", **latency_results})

    # Print latency table
    table = Table(title="Latency Results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in latency_results.items():
        table.add_row(k, f"{v:.2f}" if isinstance(v, float) else str(v))
    console.print(table)

    # Batch throughput benchmark
    if len(sizes) > 1 or sizes[0] > 1:
        console.print("\n[dim]Running batch throughput benchmark ...[/dim]")
        batch_table = Table(title="Batch Throughput")
        batch_table.add_column("Batch Size", justify="right")
        batch_table.add_column("Total TPS", justify="right")
        batch_table.add_column("Per-Request TPS", justify="right")

        for bs in sizes:
            prompts = (TEST_PROMPTS * ((bs // len(TEST_PROMPTS)) + 1))[:bs]
            t0 = time.perf_counter()
            responses = engine.generate_batch(prompts, sampling, max_batch_size=bs)
            elapsed = time.perf_counter() - t0

            total_tokens = sum(len(r.split()) for r in responses)  # approx
            total_tps = total_tokens / elapsed
            per_req_tps = total_tps / bs

            batch_table.add_row(str(bs), f"{total_tps:.1f}", f"{per_req_tps:.1f}")
            all_results.append({
                "benchmark": "batch",
                "batch_size": bs,
                "total_tps": total_tps,
                "per_request_tps": per_req_tps,
            })

        console.print(batch_table)

    if output_json:
        output_json.write_text(json.dumps(all_results, indent=2))
        console.print(f"\n[dim]Results saved to {output_json}[/dim]")


def _percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


if __name__ == "__main__":
    app()
