"""
Script to visualize enhanced metrics from training.
"""

import glob
import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

app = typer.Typer(help="Visualize enhanced metrics from training")


def load_metrics_files(metrics_dir: str, experiment_name: str) -> List[Dict[str, Any]]:
    """
    Load all enhanced metrics files for an experiment.

    Args:
        metrics_dir: Path to metrics directory
        experiment_name: Name of experiment

    Returns:
        List of metrics dictionaries
    """
    pattern = os.path.join(
        metrics_dir, f"{experiment_name}_enhanced_metrics_epoch_*.json"
    )
    files = sorted(glob.glob(pattern))

    metrics_list = []

    for file in files:
        try:
            with open(file, "r") as f:
                metrics = json.load(f)
                metrics_list.append(metrics)
        except Exception as e:
            console.print(f"[red]Error loading file {file}: {e}[/red]")

    # Sort by epoch
    metrics_list.sort(key=lambda x: x.get("epoch", 0))

    return metrics_list


def print_prediction_samples(metrics: Dict[str, Any]) -> None:
    """
    Print samples of predictions vs targets.

    Args:
        metrics: Metrics dictionary
    """
    samples = metrics.get("samples", {}).get("samples", [])

    if not samples:
        console.print("[yellow]No sample predictions available[/yellow]")
        return

    console.print(Panel.fit("[bold]Sample Predictions[/bold]", border_style="cyan"))

    for i, sample in enumerate(samples):
        prediction = sample.get("prediction", "")
        target = sample.get("target", "")
        low_confidence = sample.get("low_confidence_tokens", [])

        # Create prediction display
        pred_display = Text(f"Prediction: {prediction}")

        # Create target display
        target_display = Text(f"Target: {target}")

        # Create details panel content
        details = []

        if low_confidence:
            details.append(Text("Low confidence tokens:"))
            for token, conf in low_confidence:
                details.append(Text(f"  {token}: {conf:.2f}", style="yellow"))

        # Create panel for this sample
        panel_content = [pred_display, "", target_display]
        if details:
            panel_content.extend(["", *details])

        console.print(
            Panel(
                "\n".join(str(item) for item in panel_content),
                title=f"Sample {i + 1}",
                border_style="blue",
            )
        )


def print_token_distribution(metrics: Dict[str, Any]) -> None:
    """
    Print token distribution information.

    Args:
        metrics: Metrics dictionary
    """
    token_dist = metrics.get("token_distribution", {})
    predictions = token_dist.get("predictions", {})
    targets = token_dist.get("targets", {})

    if not predictions or not targets:
        console.print("[yellow]No token distribution information available[/yellow]")
        return

    console.print(
        Panel.fit("[bold]Token Distribution Analysis[/bold]", border_style="cyan")
    )

    # Create table for top tokens
    table = Table(title="Top Tokens")
    table.add_column("Rank", style="dim")
    table.add_column("Prediction Token", style="green")
    table.add_column("Count", style="green")
    table.add_column("Target Token", style="blue")
    table.add_column("Count", style="blue")

    pred_tokens = predictions.get("top_tokens", [])
    target_tokens = targets.get("top_tokens", [])

    # Ensure both lists have same length for display
    max_len = max(len(pred_tokens), len(target_tokens))
    pred_tokens = pred_tokens + [("-", 0)] * (max_len - len(pred_tokens))
    target_tokens = target_tokens + [("-", 0)] * (max_len - len(target_tokens))

    for i, ((pred_token, pred_count), (target_token, target_count)) in enumerate(
        zip(pred_tokens, target_tokens)
    ):
        table.add_row(
            str(i + 1), pred_token, str(pred_count), target_token, str(target_count)
        )

    console.print(table)

    # Print diversity and entropy metrics
    pred_diversity = predictions.get("diversity", 0)
    target_diversity = targets.get("diversity", 0)
    pred_entropy = predictions.get("entropy", 0)
    target_entropy = targets.get("entropy", 0)
    repetition_factor = predictions.get("repetition_factor", 0)

    metrics_table = Table(title="Distribution Metrics")
    metrics_table.add_column("Metric")
    metrics_table.add_column("Predictions", style="green")
    metrics_table.add_column("Targets", style="blue")

    metrics_table.add_row(
        "Diversity (unique/total)", f"{pred_diversity:.4f}", f"{target_diversity:.4f}"
    )
    metrics_table.add_row("Entropy", f"{pred_entropy:.4f}", f"{target_entropy:.4f}")
    metrics_table.add_row("Top token repetition", f"{repetition_factor:.4f}", "-")

    console.print(metrics_table)


def plot_metrics_over_time(
    metrics_list: List[Dict[str, Any]], output_dir: Optional[str] = None
) -> None:
    """
    Plot metrics trends over epochs.

    Args:
        metrics_list: List of metrics dictionaries
        output_dir: Directory to save plots (optional)
    """
    if not metrics_list:
        console.print("[yellow]No metrics available for plotting[/yellow]")
        return

    epochs = [m.get("epoch", i) for i, m in enumerate(metrics_list)]

    # Diversity plot
    diversities = []
    target_diversities = []

    for m in metrics_list:
        token_dist = m.get("token_distribution", {})
        pred_info = token_dist.get("predictions", {})
        target_info = token_dist.get("targets", {})

        diversities.append(pred_info.get("diversity", 0))
        target_diversities.append(target_info.get("diversity", 0))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, diversities, "g-", label="Predictions")
    plt.plot(epochs, target_diversities, "b--", label="Targets")
    plt.xlabel("Epoch")
    plt.ylabel("Diversity")
    plt.title("Token Diversity Over Training")
    plt.legend()
    plt.grid(True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "diversity.png"))

    plt.show()

    # Repetition factor plot
    repetition_factors = []

    for m in metrics_list:
        token_dist = m.get("token_distribution", {})
        pred_info = token_dist.get("predictions", {})

        repetition_factors.append(pred_info.get("repetition_factor", 0))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, repetition_factors, "r-")
    plt.xlabel("Epoch")
    plt.ylabel("Repetition Factor")
    plt.title("Token Repetition Factor Over Training")
    plt.grid(True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "repetition.png"))

    plt.show()


@app.command()
def visualize_metrics(
    metrics_dir: str = typer.Argument(..., help="Path to metrics directory"),
    experiment: str = typer.Argument(..., help="Experiment name"),
    epoch: Optional[int] = typer.Option(
        None, help="Specific epoch to visualize (default: latest)"
    ),
    output_dir: Optional[str] = typer.Option(None, help="Directory to save plots"),
    plot_trends: bool = typer.Option(False, help="Plot metrics trends over epochs"),
) -> None:
    """
    Visualize enhanced metrics from training.
    """
    # Load metrics files
    metrics_list = load_metrics_files(metrics_dir, experiment)

    if not metrics_list:
        console.print(f"[red]No metrics found for experiment {experiment}[/red]")
        return

    # If epoch is specified, find metrics for that epoch
    if epoch is not None:
        metrics = None
        for m in metrics_list:
            if m.get("epoch") == epoch:
                metrics = m
                break

        if metrics is None:
            console.print(f"[red]No metrics found for epoch {epoch}[/red]")
            console.print(
                f"[yellow]Available epochs: {[m.get('epoch') for m in metrics_list]}[/yellow]"
            )
            return
    else:
        # Use latest epoch
        metrics = metrics_list[-1]
        console.print(
            f"[green]Showing metrics for epoch {metrics.get('epoch')}[/green]"
        )

    # Display metrics
    print_prediction_samples(metrics)
    print_token_distribution(metrics)

    # Plot trends if requested
    if plot_trends:
        console.print("[green]Plotting metrics trends...[/green]")
        plot_metrics_over_time(metrics_list, output_dir)


if __name__ == "__main__":
    app()
