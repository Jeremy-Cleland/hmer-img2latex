#!/usr/bin/env python
"""
Script for plotting learning curves from training metrics.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
import yaml
from rich.console import Console

from img2latex.analysis.utils import ensure_output_dir

app = typer.Typer(help="Plot learning curves from training metrics")
console = Console()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_metrics_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load metrics data from file.

    Args:
        file_path: Path to the metrics file (CSV or JSON)

    Returns:
        DataFrame containing metrics data

    Raises:
        ValueError: If the file format is not supported
    """
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()

    if file_extension == ".csv":
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load CSV metrics file: {e}")

    elif file_extension == ".json":
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Parse based on structure
            if isinstance(data, dict) and "metrics" in data:
                # Handle structured metrics format
                metrics_dict = {}
                for split, metrics in data["metrics"].items():
                    for metric_name, values in metrics.items():
                        if isinstance(values, list):
                            for i, value in enumerate(values):
                                key = f"{split}_{metric_name}"
                                if key not in metrics_dict:
                                    metrics_dict[key] = []
                                metrics_dict[key].append(
                                    {"step": i, "value": value, "split": split}
                                )

                # Convert to DataFrame
                rows = []
                for metric_name, values in metrics_dict.items():
                    for entry in values:
                        rows.append(
                            {
                                "metric": metric_name,
                                "step": entry["step"],
                                "value": entry["value"],
                                "split": entry["split"],
                            }
                        )
                return pd.DataFrame(rows)

            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                # Handle list of metric records
                return pd.DataFrame(data)

            elif isinstance(data, dict):
                # Handle flat metrics structure
                rows = []
                for metric_name, values in data.items():
                    if isinstance(values, list):
                        for i, value in enumerate(values):
                            rows.append(
                                {"metric": metric_name, "step": i, "value": value}
                            )
                return pd.DataFrame(rows)

            else:
                raise ValueError("Unsupported JSON metrics format")

        except Exception as e:
            raise ValueError(f"Failed to load JSON metrics file: {e}")

    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def plot_learning_curves(
    metrics_data: pd.DataFrame,
    output_dir: Path,
    metrics: Optional[List[str]] = None,
    smoothing_factor: float = None,
    figure_size: tuple = None,
):
    """Plot learning curves for specified metrics.

    Args:
        metrics_data: DataFrame containing metrics data
        output_dir: Directory to save plots
        metrics: List of metrics to plot (if None, plot all)
        smoothing_factor: Smoothing factor for the curves (0-1)
        figure_size: Figure size as (width, height) tuple
    """
    # Use defaults if not provided
    if smoothing_factor is None:
        smoothing_factor = 0.0

    if figure_size is None:
        figure_size = (10, 6)

    # Set up plot style
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    # Determine metrics to plot
    if metrics is None:
        if "metric" in metrics_data.columns:
            available_metrics = metrics_data["metric"].unique()
        else:
            # Assume each column is a metric except special columns
            special_cols = ["step", "epoch", "iteration", "split"]
            available_metrics = [
                col for col in metrics_data.columns if col not in special_cols
            ]
        metrics = available_metrics

    # Apply smoothing if needed
    if smoothing_factor > 0:
        for metric in metrics:
            if metric in metrics_data.columns:
                metrics_data[f"{metric}_smooth"] = (
                    metrics_data[metric].ewm(alpha=(1 - smoothing_factor)).mean()
                )

    # Determine x-axis
    if "step" in metrics_data.columns:
        x_col = "step"
    elif "epoch" in metrics_data.columns:
        x_col = "epoch"
    elif "iteration" in metrics_data.columns:
        x_col = "iteration"
    else:
        # Create a step column if not present
        metrics_data["step"] = range(len(metrics_data))
        x_col = "step"

    # Plot each metric
    for metric in metrics:
        plt.figure(figsize=figure_size)

        # Check if metric exists in various formats
        if metric in metrics_data.columns:
            # Direct column
            if "split" in metrics_data.columns:
                for split in metrics_data["split"].unique():
                    split_data = metrics_data[metrics_data["split"] == split]
                    plt.plot(
                        split_data[x_col], split_data[metric], label=f"{split} {metric}"
                    )
                    if smoothing_factor > 0:
                        plt.plot(
                            split_data[x_col],
                            split_data[f"{metric}_smooth"],
                            label=f"{split} {metric} (smoothed)",
                            linestyle="--",
                        )
            else:
                plt.plot(metrics_data[x_col], metrics_data[metric], label=metric)
                if smoothing_factor > 0:
                    plt.plot(
                        metrics_data[x_col],
                        metrics_data[f"{metric}_smooth"],
                        label=f"{metric} (smoothed)",
                        linestyle="--",
                    )

        elif "metric" in metrics_data.columns:
            # Filter for this metric
            metric_data = metrics_data[metrics_data["metric"] == metric]
            if not metric_data.empty:
                if "split" in metric_data.columns:
                    for split in metric_data["split"].unique():
                        split_data = metric_data[metric_data["split"] == split]
                        plt.plot(
                            split_data[x_col],
                            split_data["value"],
                            label=f"{split} {metric}",
                        )
                else:
                    plt.plot(metric_data[x_col], metric_data["value"], label=metric)

        # Set labels and title
        plt.xlabel(x_col.capitalize())
        plt.ylabel("Value")
        plt.title(f"Learning Curve: {metric}")
        plt.legend()
        plt.tight_layout()

        # Save the figure
        fig_path = output_dir / f"{metric}_curve.png"
        plt.savefig(fig_path)
        plt.close()

        console.print(f"[green]Saved {metric} curve to {fig_path}[/green]")


@app.command()
def plot(
    config_path: str = typer.Option(
        "img2latex/configs/config.yaml", help="Path to the configuration file"
    ),
    metrics_file: str = typer.Argument(..., help="Path to the metrics file"),
    output_dir: str = typer.Option(
        "outputs/learning_curves", help="Directory to save the learning curves"
    ),
    metrics: Optional[List[str]] = typer.Option(
        None, help="Specific metrics to plot (comma-separated)"
    ),
    smoothing: float = typer.Option(
        None, help="Smoothing factor for the curves (0-1), overrides config"
    ),
):
    """Plot learning curves from training metrics."""
    # Load configuration
    cfg = load_config(config_path)

    # Get config values with defaults
    smoothing_factor = (
        smoothing
        if smoothing is not None
        else cfg["analysis"].get("curve_smoothing", 0.0)
    )
    figure_size = tuple(cfg["visualization"].get("curve_figure_size", (10, 6)))

    # Ensure output directory exists
    output_path = ensure_output_dir(output_dir, "curves")

    console.print(f"[bold green]Loading metrics from {metrics_file}[/bold green]")

    # Load metrics data
    try:
        metrics_data = load_metrics_data(metrics_file)
    except Exception as e:
        console.print(f"[bold red]Error loading metrics file: {e}[/bold red]")
        return

    console.print(
        f"[green]Loaded metrics data with {len(metrics_data)} entries[/green]"
    )

    # Plot learning curves
    console.print("[green]Plotting learning curves...[/green]")
    plot_learning_curves(
        metrics_data=metrics_data,
        output_dir=output_path,
        metrics=metrics,
        smoothing_factor=smoothing_factor,
        figure_size=figure_size,
    )

    console.print(
        f"[bold green]Learning curves generated in {output_path}[/bold green]"
    )


if __name__ == "__main__":
    app()
