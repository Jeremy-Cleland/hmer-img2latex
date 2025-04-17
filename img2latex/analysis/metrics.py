# Path: img2latex/analysis/metrics.py
"""
Metrics analysis for the image-to-LaTeX model.
"""

import json
import os
from typing import Dict, List, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from img2latex.utils.logging import get_logger
from img2latex.utils.path_utils import path_manager
from img2latex.utils.registry import experiment_registry
from img2latex.utils.visualize_metrics import (
    plot_metrics_over_time,
    print_prediction_samples,
    print_token_distribution,
)

app = typer.Typer(help="Analyze metrics")
console = Console()
logger = get_logger(__name__)


def load_experiment_metrics(experiment_name: str) -> List[Dict]:
    """
    Load all metrics files for an experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        List of metrics dictionaries
    """
    # Get metrics directory
    metrics_dir = path_manager.get_metrics_dir(experiment_name)

    if not os.path.exists(metrics_dir):
        console.print(f"[red]Metrics directory not found: {metrics_dir}[/red]")
        return []

    # Find all metric files - look for any prefix followed by enhanced_metrics_epoch pattern
    metric_files = list(metrics_dir.glob("*_enhanced_metrics_epoch_*.json"))

    if not metric_files:
        console.print(
            f"[red]No metrics files found for experiment: {experiment_name}[/red]"
        )
        return []

    # Load metrics files
    metrics_list = []
    for file_path in sorted(metric_files, key=lambda p: int(p.stem.split("_")[-1])):
        try:
            with open(file_path, "r") as f:
                metrics = json.load(f)
                metrics_list.append(metrics)
        except Exception as e:
            console.print(
                f"[yellow]Error loading metrics file {file_path}: {e}[/yellow]"
            )

    return metrics_list


@app.command("visualize")
def visualize_metrics(
    experiment: str = typer.Option(..., help="Name of the experiment"),
    epoch: Optional[int] = typer.Option(
        None, help="Specific epoch to visualize (default: latest)"
    ),
    show_samples: bool = typer.Option(True, help="Show prediction samples"),
    show_token_dist: bool = typer.Option(True, help="Show token distribution"),
    plot_history: bool = typer.Option(True, help="Plot metrics history"),
    metrics_file: Optional[str] = typer.Option(
        None,
        help="Path to specific metrics file (deprecated, use experiment name instead)",
    ),
):
    """
    Visualize metrics for a specific experiment.
    """
    # Load metrics files
    metrics_list = load_experiment_metrics(experiment)

    if not metrics_list:
        typer.echo("No metrics found for the experiment.")
        raise typer.Exit(code=1)

    # Get specific epoch or latest
    if epoch is not None:
        metrics = next((m for m in metrics_list if m.get("epoch") == epoch), None)
        if not metrics:
            console.print(f"[red]Metrics for epoch {epoch} not found[/red]")
            raise typer.Exit(code=1)
    else:
        # Get latest epoch
        metrics = max(metrics_list, key=lambda m: m.get("epoch", 0))
        epoch = metrics.get("epoch", 0)

    # Print basic metrics
    table = Table(title=f"Metrics for {experiment} (Epoch {epoch})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    basic_metrics = [
        ("BLEU Score", metrics.get("bleu", 0)),
        ("Levenshtein Similarity", metrics.get("levenshtein", 0)),
        ("Accuracy", metrics.get("accuracy", 0)),
        ("Batch Size", metrics.get("batch_size", 0)),
        ("Number of Tokens", metrics.get("num_tokens", 0)),
    ]

    for name, value in basic_metrics:
        if isinstance(value, float):
            table.add_row(name, f"{value:.4f}")
        else:
            table.add_row(name, str(value))

    console.print(table)

    # Show token distribution
    if show_token_dist and "token_distribution" in metrics:
        print_token_distribution(metrics["token_distribution"])

    # Show prediction samples
    if show_samples and "samples" in metrics:
        print_prediction_samples(metrics["samples"])

    # Plot metrics history
    if plot_history and len(metrics_list) > 1:
        metrics_dir = path_manager.get_metrics_dir(experiment)
        output_file = metrics_dir / f"{experiment}_metrics_plot.png"
        plot_metrics_over_time(metrics_list, output_file)
        console.print(f"[green]Metrics history plot saved to: {output_file}[/green]")


@app.command("latest")
def show_latest_metrics(
    experiment: str = typer.Option(..., help="Name of the experiment"),
    metrics_file: Optional[str] = typer.Option(
        None,
        help="Path to specific metrics file (deprecated, use experiment name instead)",
    ),
):
    """
    Show latest metrics for an experiment.
    """
    # Load metrics files
    metrics_list = load_experiment_metrics(experiment)

    if not metrics_list:
        typer.echo("No metrics found for the experiment.")
        raise typer.Exit(code=1)

    # Get latest epoch metrics
    latest_metrics = max(metrics_list, key=lambda m: m.get("epoch", 0))
    epoch = latest_metrics.get("epoch", 0)

    # Print a simplified view
    console.print(
        Panel(f"[bold]Latest Metrics for {experiment} (Epoch {epoch})[/bold]")
    )

    console.print(f"BLEU Score: [green]{latest_metrics.get('bleu', 0):.4f}[/green]")
    console.print(
        f"Levenshtein: [green]{latest_metrics.get('levenshtein', 0):.4f}[/green]"
    )
    console.print(f"Accuracy: [green]{latest_metrics.get('accuracy', 0):.4f}[/green]")

    # Print token distribution summary
    if "token_distribution" in latest_metrics:
        token_dist = latest_metrics["token_distribution"]
        pred_info = token_dist.get("predictions", {})

        if "diversity" in pred_info:
            console.print(
                f"Token Diversity: [green]{pred_info['diversity']:.4f}[/green]"
            )

        if "repetition_factor" in pred_info:
            rep_factor = pred_info["repetition_factor"]
            color = "red" if rep_factor > 0.5 else "green"
            console.print(f"Repetition Factor: [{color}]{rep_factor:.4f}[/{color}]")

    # Print a sample prediction
    if "samples" in latest_metrics and latest_metrics["samples"]:
        sample = latest_metrics["samples"][0]
        console.print("\n[bold]Sample Prediction:[/bold]")
        console.print(f"Target: [cyan]{sample.get('target', 'N/A')}[/cyan]")
        console.print(f"Prediction: [yellow]{sample.get('prediction', 'N/A')}[/yellow]")


@app.command("compare")
def compare_experiments(
    experiments: List[str] = typer.Option(
        None, help="List of experiments to compare (if none, all are compared)"
    ),
    metric: str = typer.Option(
        "bleu", help="Metric to compare (bleu, levenshtein, accuracy)"
    ),
):
    """
    Compare metrics across multiple experiments.
    """
    if not experiments:
        # If no experiments provided, list all from registry
        try:
            experiments = experiment_registry.list_experiments()
        except Exception as e:
            console.print(f"[red]Error loading experiment registry: {e}[/red]")
            raise typer.Exit(code=1)

        if not experiments:
            console.print("[red]No experiments found in registry.[/red]")
            raise typer.Exit(code=1)

    # Load metrics for each experiment
    experiments_data = []
    for exp_name in experiments:
        metrics_list = load_experiment_metrics(exp_name)
        if metrics_list:
            latest_metrics = max(metrics_list, key=lambda m: m.get("epoch", 0))
            epoch = latest_metrics.get("epoch", 0)
            metric_value = latest_metrics.get(metric, 0)
            experiments_data.append(
                {
                    "experiment": exp_name,
                    "epoch": epoch,
                    "value": metric_value,
                }
            )

    if not experiments_data:
        console.print("[red]No metrics data found for any experiment.[/red]")
        raise typer.Exit(code=1)

    # Sort by metric value, descending
    experiments_data.sort(key=lambda x: x["value"], reverse=True)

    # Print comparison table
    table = Table(title=f"Experiment Comparison - {metric.upper()}")
    table.add_column("Rank", style="dim")
    table.add_column("Experiment", style="cyan")
    table.add_column("Epoch", style="blue")
    table.add_column(metric.upper(), style="green")

    for i, exp_data in enumerate(experiments_data, 1):
        table.add_row(
            str(i),
            exp_data["experiment"],
            str(exp_data["epoch"]),
            f"{exp_data['value']:.4f}",
        )

    console.print(table)


@app.command("export")
def export_metrics(
    experiment: str = typer.Option(..., help="Name of the experiment"),
    format: str = typer.Option("csv", help="Export format (csv, json)"),
    output: Optional[str] = typer.Option(
        None, help="Output file path (default: based on experiment name)"
    ),
):
    """
    Export metrics for an experiment to CSV or JSON.
    """
    # Load metrics files
    metrics_list = load_experiment_metrics(experiment)

    if not metrics_list:
        console.print(f"[red]No metrics found for experiment: {experiment}[/red]")
        raise typer.Exit(code=1)

    # Create DataFrame from metrics
    data = []
    for m in metrics_list:
        row = {
            "epoch": m.get("epoch", 0),
            "bleu": m.get("bleu", 0),
            "levenshtein": m.get("levenshtein", 0),
            "accuracy": m.get("accuracy", 0),
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Determine output path
    if output is None:
        metrics_dir = path_manager.get_metrics_dir(experiment)
        output = metrics_dir / f"{experiment}_metrics.{format}"

    # Export data
    if format.lower() == "csv":
        df.to_csv(output, index=False)
    elif format.lower() == "json":
        df.to_json(output, orient="records", indent=2)
    else:
        console.print(f"[red]Unsupported format: {format}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Metrics exported to: {output}[/green]")


if __name__ == "__main__":
    app()
