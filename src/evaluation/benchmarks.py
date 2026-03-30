"""
Evaluation suite — perplexity, ROUGE, BERTScore, and task benchmarks.

Provides a unified ``Evaluator`` class that runs all configured metrics
against a model and returns a structured results dict suitable for
W&B logging, CSV export, or console display.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    metrics: list[str] = field(default_factory=lambda: ["perplexity"])
    # Perplexity
    ppl_batch_size: int = 4
    ppl_stride: int = 512       # sliding window stride for long documents
    # ROUGE
    rouge_types: list[str] = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])
    # BERTScore
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    bertscore_batch_size: int = 64
    # Output
    output_dir: str | None = None


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    batch_size: int = 4,
    stride: int = 512,
    max_length: int | None = None,
) -> dict[str, float]:
    """
    Compute per-token perplexity using a sliding-window approach.

    For texts longer than the model's context window, the window slides
    by ``stride`` tokens, with previously-seen tokens masked from the loss.
    This avoids artificially low perplexity from teacher-forced prefixes.

    Returns
    -------
    dict with keys: mean_ppl, median_ppl, min_ppl, max_ppl, std_ppl
    """
    model.eval()
    max_length = max_length or getattr(model.config, "max_position_embeddings", 2048)
    all_ppls: list[float] = []

    for text in tqdm(texts, desc="Computing perplexity"):
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings["input_ids"]
        seq_len = input_ids.size(1)

        nlls: list[torch.Tensor] = []
        prev_end = 0

        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_len = end - max(begin, prev_end)

            chunk_ids = input_ids[:, begin:end].to(model.device)
            labels = chunk_ids.clone()
            # Mask tokens that were part of the previous window
            labels[:, : chunk_ids.size(1) - target_len] = IGNORE_INDEX

            with torch.inference_mode():
                output = model(chunk_ids, labels=labels)
            nlls.append(output.loss * target_len)
            prev_end = end

            if end == seq_len:
                break

        total_nll = torch.stack(nlls).sum()
        ppl = math.exp(total_nll.item() / seq_len)
        all_ppls.append(ppl)

    import statistics
    return {
        "mean_ppl":   sum(all_ppls) / len(all_ppls),
        "median_ppl": statistics.median(all_ppls),
        "min_ppl":    min(all_ppls),
        "max_ppl":    max(all_ppls),
        "std_ppl":    statistics.stdev(all_ppls) if len(all_ppls) > 1 else 0.0,
    }


def compute_rouge(
    predictions: list[str],
    references: list[str],
    rouge_types: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Returns
    -------
    dict mapping metric name (e.g. 'rouge1_f1') to mean score over dataset.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise ImportError("pip install rouge-score") from e

    rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    aggregated: dict[str, list[float]] = {f"{t}_f1": [] for t in rouge_types}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for t in rouge_types:
            aggregated[f"{t}_f1"].append(scores[t].fmeasure)

    return {k: sum(v) / len(v) for k, v in aggregated.items()}


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 64,
    device: str | None = None,
) -> dict[str, float]:
    """
    Compute BERTScore precision, recall, and F1.

    BERTScore matches tokens in the prediction to the most similar tokens
    in the reference using contextualised embeddings, providing a semantic
    similarity metric that correlates better with human judgement than ROUGE.
    """
    try:
        from bert_score import score as bert_score
    except ImportError as e:
        raise ImportError("pip install bert-score") from e

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    P, R, F = bert_score(
        predictions,
        references,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )

    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall":    R.mean().item(),
        "bertscore_f1":        F.mean().item(),
    }


def compute_exact_match(
    predictions: list[str],
    references: list[str],
    normalise: bool = True,
) -> dict[str, float]:
    """Exact match accuracy, with optional whitespace/case normalisation."""
    def norm(s: str) -> str:
        return " ".join(s.strip().lower().split()) if normalise else s

    matches = sum(norm(p) == norm(r) for p, r in zip(predictions, references))
    return {"exact_match": matches / len(predictions) if predictions else 0.0}


# ---------------------------------------------------------------------------
# Unified evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Runs all configured evaluation metrics and aggregates results.

    Example
    -------
    >>> from src.inference.engine import LlamaCppEngine, SamplingConfig
    >>> engine = LlamaCppEngine("model_q4km.gguf")
    >>> evaluator = Evaluator(EvaluationConfig(metrics=["rouge", "bertscore"]))
    >>> results = evaluator.run_generation_metrics(
    ...     engine=engine,
    ...     prompts=test_prompts,
    ...     references=test_references,
    ...     sampling=SamplingConfig(temperature=0.0),
    ... )
    >>> print(results)
    """

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def run_perplexity(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
        text_column: str = "text",
    ) -> dict[str, float]:
        texts = dataset[text_column]
        return compute_perplexity(
            model,
            tokenizer,
            texts,
            batch_size=self.config.ppl_batch_size,
            stride=self.config.ppl_stride,
        )

    def run_generation_metrics(
        self,
        engine: Any,           # InferenceEngine
        prompts: list[str],
        references: list[str],
        sampling: Any,         # SamplingConfig
    ) -> dict[str, float]:
        from src.inference.engine import SamplingConfig

        logger.info("Generating %d responses for evaluation ...", len(prompts))
        predictions = engine.generate_batch(prompts, sampling)

        results: dict[str, float] = {}

        if "rouge" in self.config.metrics:
            logger.info("Computing ROUGE ...")
            results.update(compute_rouge(predictions, references, self.config.rouge_types))

        if "bertscore" in self.config.metrics:
            logger.info("Computing BERTScore ...")
            results.update(
                compute_bertscore(
                    predictions,
                    references,
                    model_type=self.config.bertscore_model,
                    batch_size=self.config.bertscore_batch_size,
                )
            )

        if "exact_match" in self.config.metrics:
            results.update(compute_exact_match(predictions, references))

        if self.config.output_dir:
            self._save_results(results, predictions, prompts, references)

        return results

    def _save_results(
        self,
        metrics: dict[str, float],
        predictions: list[str],
        prompts: list[str],
        references: list[str],
    ) -> None:
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

        rows = [
            {"prompt": p, "reference": r, "prediction": pred}
            for p, r, pred in zip(prompts, references, predictions)
        ]
        (out / "predictions.json").write_text(json.dumps(rows, indent=2))
        logger.info("Evaluation results saved to %s", out)


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def compare_runs(
    results_a: dict[str, float],
    results_b: dict[str, float],
    name_a: str = "baseline",
    name_b: str = "finetuned",
) -> dict[str, dict]:
    """
    Compare two sets of evaluation results, computing relative improvement.

    Returns a dict mapping metric → {name_a, name_b, delta, pct_change}.
    """
    comparison: dict[str, dict] = {}
    all_metrics = set(results_a) | set(results_b)

    for metric in sorted(all_metrics):
        a = results_a.get(metric)
        b = results_b.get(metric)
        if a is not None and b is not None:
            delta = b - a
            pct = 100.0 * delta / abs(a) if a != 0 else float("inf")
            comparison[metric] = {
                name_a:      round(a, 4),
                name_b:      round(b, 4),
                "delta":     round(delta, 4),
                "pct_change": round(pct, 2),
            }

    return comparison
