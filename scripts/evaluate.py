"""
Evaluation entrypoint — run all configured metrics against a model.

Usage
-----
  # Perplexity on a HF model
  python scripts/evaluate.py \\
      --model-path outputs/merged_model \\
      --backend hf \\
      --tasks perplexity \\
      --dataset HuggingFaceH4/ultrachat_200k \\
      --split test_sft

  # ROUGE + BERTScore with a GGUF model
  python scripts/evaluate.py \\
      --model-path outputs/model_q4km.gguf \\
      --backend llama-cpp \\
      --tasks rouge,bertscore \\
      --dataset HuggingFaceH4/ultrachat_200k \\
      --split test_sft \\
      --max-samples 500

  # Compare fine-tuned vs baseline
  python scripts/evaluate.py \\
      --model-path outputs/model_q4km.gguf \\
      --baseline-path baseline_q4km.gguf \\
      --tasks rouge,bertscore \\
      --dataset HuggingFaceH4/ultrachat_200k \\
      --split test_sft
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.benchmarks import (
    EvaluationConfig,
    Evaluator,
    compare_runs,
    compute_perplexity,
)
from src.inference.engine import LlamaCppEngine, HFInferenceEngine, SamplingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer()


@app.command()
def evaluate(
    model_path: str = typer.Option(..., "--model-path", help="Model path (HF dir or GGUF file)"),
    backend: str = typer.Option("llama-cpp", "--backend", help="llama-cpp | hf"),
    tasks: str = typer.Option("rouge", "--tasks", help="Comma-separated: perplexity,rouge,bertscore,exact_match"),
    dataset: str = typer.Option(..., "--dataset", help="HF dataset name or local JSONL path"),
    split: str = typer.Option("test", "--split"),
    prompt_column: str = typer.Option("prompt", "--prompt-column"),
    reference_column: str = typer.Option("response", "--reference-column"),
    text_column: str = typer.Option("text", "--text-column", help="For perplexity eval"),
    messages_column: str = typer.Option("messages", "--messages-column"),
    max_samples: int = typer.Option(500, "--max-samples"),
    max_tokens: int = typer.Option(256, "--max-tokens"),
    temperature: float = typer.Option(0.0, "--temperature", help="0.0 for deterministic eval"),
    n_gpu_layers: int = typer.Option(-1, "--n-gpu-layers"),
    output_dir: Path = typer.Option(Path("outputs/eval_results"), "--output-dir"),
    baseline_path: str | None = typer.Option(None, "--baseline-path", help="Optional baseline model for comparison"),
) -> None:
    """Evaluate a fine-tuned LLM across multiple metrics."""

    task_list = [t.strip() for t in tasks.split(",")]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load dataset
    # ------------------------------------------------------------------ #
    console.print(f"[dim]Loading dataset: {dataset} ({split}) ...[/dim]")
    raw_ds = _load_dataset(dataset, split, max_samples)

    # ------------------------------------------------------------------ #
    # Build engine(s)
    # ------------------------------------------------------------------ #
    engine = _build_engine(model_path, backend, n_gpu_layers)
    baseline_engine = _build_engine(baseline_path, backend, n_gpu_layers) if baseline_path else None

    # ------------------------------------------------------------------ #
    # Perplexity (HF backend only)
    # ------------------------------------------------------------------ #
    results: dict[str, float] = {}

    if "perplexity" in task_list:
        if backend != "hf":
            console.print("[yellow]Warning:[/yellow] Perplexity requires --backend hf. Skipping.")
        else:
            console.print("[dim]Computing perplexity ...[/dim]")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            hf_model = engine._model
            tokenizer = engine._tokenizer
            texts = raw_ds[text_column] if text_column in raw_ds.column_names else None
            if texts is None:
                console.print(f"[yellow]Column '{text_column}' not found — skipping perplexity.[/yellow]")
            else:
                results.update(compute_perplexity(hf_model, tokenizer, texts[:max_samples]))

    # ------------------------------------------------------------------ #
    # Generation metrics (rouge, bertscore, exact_match)
    # ------------------------------------------------------------------ #
    gen_tasks = [t for t in task_list if t in ("rouge", "bertscore", "exact_match")]
    if gen_tasks:
        prompts, references = _extract_prompts_references(
            raw_ds, prompt_column, reference_column, messages_column
        )
        sampling = SamplingConfig(temperature=temperature, max_tokens=max_tokens)
        eval_cfg = EvaluationConfig(
            metrics=gen_tasks,
            output_dir=str(output_dir),
        )
        evaluator = Evaluator(eval_cfg)
        console.print(f"[dim]Running generation metrics: {gen_tasks} ...[/dim]")
        gen_results = evaluator.run_generation_metrics(engine, prompts, references, sampling)
        results.update(gen_results)

    # ------------------------------------------------------------------ #
    # Baseline comparison
    # ------------------------------------------------------------------ #
    comparison: dict | None = None
    if baseline_engine and gen_tasks:
        console.print("[dim]Running baseline generation ...[/dim]")
        baseline_eval_cfg = EvaluationConfig(metrics=gen_tasks)
        baseline_evaluator = Evaluator(baseline_eval_cfg)
        baseline_results = baseline_evaluator.run_generation_metrics(
            baseline_engine, prompts, references, sampling
        )
        comparison = compare_runs(baseline_results, results, name_a="baseline", name_b="finetuned")

    # ------------------------------------------------------------------ #
    # Display results
    # ------------------------------------------------------------------ #
    _print_results(results, comparison)

    # Save
    output = {"model": model_path, "dataset": dataset, "split": split, "metrics": results}
    if comparison:
        output["comparison"] = comparison

    results_path = output_dir / "metrics.json"
    results_path.write_text(json.dumps(output, indent=2))
    console.print(f"\n[dim]Results saved → {results_path}[/dim]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_engine(path: str | None, backend: str, n_gpu_layers: int):
    if path is None:
        return None
    if backend == "llama-cpp":
        return LlamaCppEngine(path, n_gpu_layers=n_gpu_layers)
    elif backend == "hf":
        return HFInferenceEngine.from_pretrained(path)
    else:
        raise typer.BadParameter(f"Unknown backend: {backend}")


def _load_dataset(name_or_path: str, split: str, max_samples: int):
    from datasets import load_dataset
    from pathlib import Path as P

    p = P(name_or_path)
    if p.exists() and p.suffix in (".jsonl", ".json"):
        import json as _json
        rows = [_json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        from datasets import Dataset
        ds = Dataset.from_list(rows[:max_samples])
    else:
        ds = load_dataset(name_or_path, split=split)
        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))
    return ds


def _extract_prompts_references(
    ds,
    prompt_col: str,
    response_col: str,
    messages_col: str,
) -> tuple[list[str], list[str]]:
    """Extract prompt/reference pairs from a dataset."""
    prompts: list[str] = []
    references: list[str] = []

    for record in ds:
        if prompt_col in record and response_col in record:
            prompts.append(record[prompt_col])
            references.append(record[response_col])
        elif messages_col in record:
            msgs = record[messages_col]
            user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
            asst_msgs = [m["content"] for m in msgs if m["role"] == "assistant"]
            if user_msgs and asst_msgs:
                prompts.append(user_msgs[-1])
                references.append(asst_msgs[-1])

    return prompts, references


def _print_results(results: dict[str, float], comparison: dict | None) -> None:
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right", style="bold")

    for k, v in sorted(results.items()):
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)

    if comparison:
        cmp_table = Table(title="Comparison: Baseline → Fine-tuned")
        cmp_table.add_column("Metric", style="cyan")
        cmp_table.add_column("Baseline", justify="right")
        cmp_table.add_column("Fine-tuned", justify="right")
        cmp_table.add_column("Δ", justify="right")
        cmp_table.add_column("% Change", justify="right")

        for metric, row in comparison.items():
            delta = row["delta"]
            pct = row["pct_change"]
            delta_str = f"[green]+{delta:.4f}[/green]" if delta > 0 else f"[red]{delta:.4f}[/red]"
            pct_str = f"[green]+{pct:.1f}%[/green]" if pct > 0 else f"[red]{pct:.1f}%[/red]"
            cmp_table.add_row(
                metric,
                str(row.get("baseline", "")),
                str(row.get("finetuned", "")),
                delta_str,
                pct_str,
            )
        console.print(cmp_table)


if __name__ == "__main__":
    app()
