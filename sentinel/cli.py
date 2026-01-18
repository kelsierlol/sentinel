"""Sentinel CLI - ML Data Quality Gate."""

import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.progress import track

app = typer.Typer(
    name="sentinel",
    help="ML data quality gate - detects corrupted images before they hurt training",
    add_completion=False,
)
console = Console()


@app.command()
def check(
    input_dir: Path = typer.Argument(..., help="Directory containing images to check"),
    output: Path = typer.Option("results.json", help="Output JSON file"),
    weights: Optional[Path] = typer.Option(None, help="Weights directory"),
    threshold: float = typer.Option(0.05, help="Score threshold for flagging (0-1)"),
    batch_size: int = typer.Option(32, help="Batch size for inference"),
    device: str = typer.Option("auto", help="Device: cuda, cpu, mps, or auto"),
    save_masks: bool = typer.Option(False, help="Save alpha map masks"),
):
    """Check a batch of images for corruption."""
    console.print(f"[bold blue]Sentinel Check[/bold blue]")
    console.print(f"Input: {input_dir}")
    console.print(f"Threshold: {threshold}")

    # TODO: Implement actual checking logic
    console.print("[yellow]⚠ CLI skeleton created - implementation coming in Phase 3[/yellow]")
    console.print("Next steps:")
    console.print("  1. Test the fixed alpha controller on CIFAR-10")
    console.print("  2. Verify PR-AUC improves from 0.04 → 0.70+")
    console.print("  3. Then build full CLI functionality")


@app.command()
def train(
    data_dir: Path = typer.Argument(..., help="Directory with clean training images"),
    output_dir: Path = typer.Option("weights", help="Output directory for weights"),
    epochs_unet: int = typer.Option(50, help="Epochs to train UNet"),
    epochs_alpha: int = typer.Option(30, help="Epochs to train alpha controller"),
    batch_size: int = typer.Option(32, help="Batch size"),
    lr: float = typer.Option(1e-4, help="Learning rate"),
    device: str = typer.Option("auto", help="Device: cuda, cpu, mps, or auto"),
):
    """Train sentinel models on clean data."""
    console.print(f"[bold green]Sentinel Train[/bold green]")
    console.print(f"Data: {data_dir}")
    console.print(f"Epochs: UNet={epochs_unet}, Alpha={epochs_alpha}")

    # TODO: Implement training logic
    console.print("[yellow]⚠ Training CLI coming in Phase 3[/yellow]")


@app.command()
def benchmark(
    imagenet_c: Optional[Path] = typer.Option(None, help="ImageNet-C dataset path"),
    imagenet_val: Optional[Path] = typer.Option(None, help="ImageNet validation path"),
    output: Path = typer.Option("benchmark_results.json", help="Output JSON file"),
):
    """Run ImageNet-C benchmarks."""
    console.print(f"[bold magenta]Sentinel Benchmark[/bold magenta]")

    # TODO: Implement benchmark logic
    console.print("[yellow]⚠ Benchmarking coming in Phase 2[/yellow]")


@app.command()
def version():
    """Show sentinel version."""
    import sentinel
    console.print(f"Sentinel version: [bold]{sentinel.__version__}[/bold]")
    console.print("https://github.com/kelsierlol/sentinel")


if __name__ == "__main__":
    app()
