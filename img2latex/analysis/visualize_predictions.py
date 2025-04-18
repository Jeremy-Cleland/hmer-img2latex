#!/usr/bin/env python
"""
Visualize sample predictions alongside ground truth formulas.
Creates comparison figures for the report.
"""

import json
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import typer
from rich.console import Console

from img2latex.utils.visualization import (
    DEFAULT_THEME,
    apply_dark_theme,
    ensure_plots_dir,
)

app = typer.Typer(help="Generate prediction visualization figures")
console = Console()


def load_predictions(predictions_file: Path) -> List[Dict]:
    """Load predictions from a JSON file."""
    with open(predictions_file, "r") as f:
        return json.load(f)


def get_sample_predictions(
    predictions: List[Dict], num_samples: int = 5, seed: int = 42
) -> List[Dict]:
    """
    Get a sample of predictions for visualization.

    Args:
        predictions: List of prediction dictionaries
        num_samples: Number of samples to select
        seed: Random seed for reproducibility

    Returns:
        List of sample predictions
    """
    random.seed(seed)

    # Get random samples
    samples = random.sample(predictions, min(num_samples, len(predictions)))

    return samples


def render_latex_comparison(
    output_dir: Path,
    samples: List[Dict],
    output_filename: str = "latex_comparison.png",
    theme: Dict = None,
):
    """
    Create a figure comparing latex predictions and references.

    Args:
        output_dir: Output directory path
        samples: List of sample prediction dictionaries
        output_filename: Filename for the output figure
        theme: Theme settings (uses DEFAULT_THEME if None)
    """
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    num_samples = len(samples)

    # Set up the figure
    fig_height = 2.5 * num_samples
    fig, axes = plt.subplots(
        num_samples, 1, figsize=(12, fig_height), facecolor=theme["background_color"]
    )
    if num_samples == 1:
        axes = [axes]  # Ensure axes is a list

    # Set background color for all axes
    for ax in axes:
        ax.set_facecolor(theme["background_color"])

    plt.suptitle(
        "LaTeX Prediction vs. Ground Truth",
        fontsize=18,
        y=0.98,
        color=theme["text_color"],
    )

    # Add colorful backgrounds to cells - adjusted for dark theme
    colors = {
        "reference": "#1e3a5f",  # Dark blue
        "prediction": "#3b2e3a",  # Dark purple
    }

    for i, sample in enumerate(samples):
        reference = sample.get("reference", "")
        prediction = sample.get("prediction", "")

        text = (
            f"Sample {i + 1}:\n\n"
            f"Ground Truth:\n{reference}\n\n"
            f"Prediction:\n{prediction}\n"
        )

        # Split text into lines
        lines = text.split("\n")

        # Create a table-like display
        table_data = []
        colors_data = []

        current_section = None

        for line in lines:
            if "Ground Truth:" in line:
                current_section = "reference"
            elif "Prediction:" in line:
                current_section = "prediction"

            if current_section and (
                line
                and not any(
                    x in line for x in ["Ground Truth:", "Prediction:", "Sample"]
                )
            ):
                colors_data.append(
                    colors.get(current_section, theme["background_color"])
                )
            else:
                colors_data.append(theme["background_color"])

            table_data.append(line)

        # Create a table with dark theme
        table = axes[i].table(
            cellText=[[line] for line in table_data],
            cellLoc="left",
            cellColours=[[color] for color in colors_data],
            colWidths=[0.9],
            loc="center",
        )

        # Style the table for dark theme
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.8)  # Adjust row heights

        # Set text color for all cells
        for (j, _), cell in table._cells.items():
            cell.get_text().set_color(theme["text_color"])
            # Add borders with grid color
            cell.set_edgecolor(theme["grid_color"])

        # Remove axis elements
        axes[i].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=150, facecolor=theme["background_color"])
    plt.close()
    console.print(f"[green]Saved LaTeX comparison to {output_path}[/green]")


def calculate_metrics(predictions: List[Dict]) -> Dict:
    """
    Calculate basic metrics from predictions.

    Args:
        predictions: List of prediction dictionaries

    Returns:
        Dictionary of metrics
    """
    # Calculate exact match percentage (if same prediction across all samples)
    unique_predictions = set()
    for item in predictions:
        unique_predictions.add(item.get("prediction", ""))

    metrics = {
        "total_samples": len(predictions),
        "unique_predictions": len(unique_predictions),
        "repetition_rate": len(unique_predictions) / max(1, len(predictions)),
    }

    return metrics


def create_metrics_figure(
    metrics: Dict,
    output_dir: Path,
    output_filename: str = "prediction_metrics.png",
    theme: Dict = None,
):
    """
    Create a figure showing prediction metrics.

    Args:
        metrics: Dictionary of metrics
        output_dir: Output directory path
        output_filename: Filename for the output figure
        theme: Theme settings (uses DEFAULT_THEME if None)
    """
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=theme["background_color"])
    ax.set_facecolor(theme["background_color"])

    # Create bar chart with theme colors
    data = [
        metrics["total_samples"],
        metrics["unique_predictions"],
        metrics["repetition_rate"] * 100,  # Convert to percentage
    ]

    labels = ["Total Samples", "Unique Predictions", "Repetition Rate (%)"]

    bars = ax.bar(labels, data, color=theme["bar_colors"][:3])

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            color=theme["text_color"],
        )

    # Add title and labels
    ax.set_title("Prediction Metrics Overview", fontsize=16, color=theme["text_color"])
    ax.set_ylabel("Value", fontsize=14, color=theme["text_color"])

    # Customize y-axis
    ax.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])
    ax.tick_params(colors=theme["text_color"])

    # Calculate appropriate y limit
    y_max = max(data) * 1.2
    ax.set_ylim(0, y_max)

    plt.tight_layout()
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=150, facecolor=theme["background_color"])
    plt.close()
    console.print(f"[green]Saved metrics figure to {output_path}[/green]")


@app.command()
def visualize(
    predictions_file: str = typer.Option(
        "outputs/img2latex_v1/predictions/predictions.json",
        help="Path to the predictions JSON file",
    ),
    output_dir: str = typer.Option(
        "outputs/img2latex_v1/plots/predictions",
        help="Directory to save the visualization figures",
    ),
    num_samples: int = typer.Option(5, help="Number of samples to visualize"),
    theme_key: str = typer.Option("dark", help="Theme to use: 'dark' for dark mode"),
):
    """Visualize sample predictions alongside ground truth formulas."""
    predictions_path = Path(predictions_file)
    output_path = Path(output_dir)

    # Ensure output directory exists
    ensure_plots_dir(output_path)

    # Use dark theme
    theme = DEFAULT_THEME

    console.print(
        f"[bold blue]Loading predictions from {predictions_path}...[/bold blue]"
    )
    if not predictions_path.exists():
        console.print(
            f"[bold red]Predictions file not found: {predictions_path}[/bold red]"
        )
        return

    try:
        predictions = load_predictions(predictions_path)

        console.print("[bold blue]Calculating prediction metrics...[/bold blue]")
        metrics = calculate_metrics(predictions)
        create_metrics_figure(metrics, output_path, theme=theme)

        console.print(
            f"[bold blue]Selecting {num_samples} sample predictions...[/bold blue]"
        )
        samples = get_sample_predictions(predictions, num_samples)

        if not samples:
            console.print("[bold yellow]No predictions found.[/bold yellow]")
            return

        console.print(
            f"[bold blue]Generating visualization figures in {output_path} with dark theme...[/bold blue]"
        )
        render_latex_comparison(output_path, samples, theme=theme)

        console.print(
            "[bold green]Visualization figures generated successfully with dark theme![/bold green]"
        )

    except Exception as e:
        console.print(
            f"[bold red]Error generating visualization figures: {e}[/bold red]"
        )


if __name__ == "__main__":
    app()
