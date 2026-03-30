"""
GGUF quantisation entrypoint.

Usage
-----
  python scripts/quantise.py \\
      --model-path outputs/merged_model \\
      --quant-type Q4_K_M \\
      --output-path outputs/model_q4km.gguf

  # Show size estimates before committing
  python scripts/quantise.py \\
      --model-path outputs/merged_model \\
      --estimate-only
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.quantise import GGUFQuantType, export_to_gguf, quantisation_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer()


@app.command()
def quantise(
    model_path: Path = typer.Option(..., "--model-path", help="Path to merged HF model"),
    output_path: Path = typer.Option(None, "--output-path", help="Output .gguf path"),
    quant_type: GGUFQuantType = typer.Option(GGUFQuantType.Q4_K_M, "--quant-type"),
    llama_cpp_dir: str | None = typer.Option(None, "--llama-cpp-dir"),
    keep_f16: bool = typer.Option(False, "--keep-f16"),
    estimate_only: bool = typer.Option(False, "--estimate-only", help="Print size estimates and exit"),
) -> None:
    """Export a HuggingFace model to GGUF format with quantisation."""

    if estimate_only or output_path is None:
        rows = quantisation_comparison(model_path)
        table = Table(title=f"GGUF Size Estimates — {model_path.name}")
        table.add_column("Quant Type", style="cyan")
        table.add_column("Est. Size (GB)", justify="right")
        table.add_column("Quality", style="dim")
        for r in rows:
            table.add_row(r["type"], f"{r['size_gb']:.2f}", r["quality"])
        console.print(table)

        if estimate_only:
            raise typer.Exit(0)

    assert output_path is not None, "Provide --output-path"

    result = export_to_gguf(
        model_path=model_path,
        output_path=output_path,
        quant_type=quant_type,
        llama_cpp_dir=llama_cpp_dir,
        keep_f16=keep_f16,
    )
    console.print(f"[green]✓[/green] GGUF export complete: [bold]{result}[/bold]")


if __name__ == "__main__":
    app()
