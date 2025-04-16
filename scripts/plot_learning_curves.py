#!/usr/bin/env python
"""
Script to plot learning curves for the img2latex model training.

Features:
- Read metrics data from CSV or JSON
- Plot each metric vs epoch
- Save figures to output directory
"""

import json
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import typer
from utils import ensure_output_dir

# Create Typer app
app = typer.Typer(help="Plot learning curves from training metrics")


def load_metrics_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load metrics data from a CSV or JSON file.

    Args:
        file_path: Path to the metrics file (CSV or JSON)

    Returns:
        DataFrame containing the metrics data

    Raises:
        ValueError: If the file format is not supported
    """
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()

    if file_extension == ".csv":
        return pd.read_csv(file_path)
    elif file_extension == ".json":
        # Read JSON file
        with open(file_path, "r") as f:
            data = json.load(f)

        # Check if data is a dict with epoch-based records or a list of records
        if isinstance(data, dict):
            # If data is stored as {epoch1: {metric1: value, ...}, epoch2: {...}}
            records = []
            for epoch, metrics in data.items():
                record = metrics.copy()
                # Try to convert epoch to int, otherwise use as is
                try:
                    record["epoch"] = int(epoch)
                except ValueError:
                    record["epoch"] = epoch
                records.append(record)
            return pd.DataFrame(records)
        elif isinstance(data, list):
            # If data is stored as [{epoch: 1, metric1: value, ...}, {epoch: 2, ...}]
            return pd.DataFrame(data)
        else:
            raise ValueError("Unsupported JSON data structure")
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def plot_learning_curves(
    metrics_data: pd.DataFrame, output_dir: Path, metrics: Optional[List[str]] = None
) -> None:
    """Plot learning curves for various metrics.

    Args:
        metrics_data: DataFrame containing the metrics data
        output_dir: Directory to save the output plots
        metrics: List of metrics to plot. If None, plots all available metrics except 'epoch'
    """
    # Ensure 'epoch' column exists
    if "epoch" not in metrics_data.columns:
        raise ValueError("Metrics data must contain an 'epoch' column")

    # If no metrics specified, use all columns except 'epoch'
    if metrics is None:
        metrics = [col for col in metrics_data.columns if col != "epoch"]

    # Sort by epoch to ensure proper plotting
    metrics_data = metrics_data.sort_values("epoch")

    # Create individual plots for each metric
    for metric in metrics:
        if metric not in metrics_data.columns:
            print(f"Warning: Metric '{metric}' not found in data")
            continue

        # Create figure
        plt.figure(figsize=(10, 6))

        # Plot the metric
        plt.plot(metrics_data["epoch"], metrics_data[metric])

        # Add labels and title
        plt.xlabel("Epoch")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} vs Epoch")

        # Add grid for readability
        plt.grid(True, alpha=0.3)

        # Save figure
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_curve.png")
        plt.close()

        print(f"Saved {metric} curve to {output_dir / f'{metric}_curve.png'}")

    # Create a combined plot for train/val pairs
    # Common metric pairs to look for
    metric_pairs = [
        ("train_loss", "val_loss"),
        ("train_acc", "val_acc"),
        ("train_accuracy", "val_accuracy"),
    ]

    for train_metric, val_metric in metric_pairs:
        if train_metric in metrics_data.columns and val_metric in metrics_data.columns:
            plt.figure(figsize=(10, 6))

            plt.plot(metrics_data["epoch"], metrics_data[train_metric], label="Train")
            plt.plot(
                metrics_data["epoch"], metrics_data[val_metric], label="Validation"
            )

            # Use the common name for the metric (without train/val prefix)
            metric_name = train_metric.replace("train_", "")

            plt.xlabel("Epoch")
            plt.ylabel(metric_name.replace("_", " ").title())
            plt.title(f"{metric_name.replace('_', ' ').title()} vs Epoch")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f"{metric_name}_comparison.png")
            plt.close()

            print(
                f"Saved {metric_name} comparison to {output_dir / f'{metric_name}_comparison.png'}"
            )


@app.command()
def plot_learning_curves_from_file(
    metrics_file: str = typer.Argument(
        ..., help="Path to the metrics file (CSV or JSON)"
    ),
    output_dir: str = typer.Option(
        "outputs/learning_curves", help="Directory to save the output plots"
    ),
    metrics: Optional[List[str]] = typer.Option(
        None, help="List of metrics to plot (default: all metrics in the file)"
    ),
) -> None:
    """Plot learning curves from a metrics file."""
    # Ensure output directory exists
    output_path = ensure_output_dir(output_dir, "curves")

    # Load metrics data
    print(f"Loading metrics from {metrics_file}...")
    try:
        metrics_data = load_metrics_data(metrics_file)

        # Plot learning curves
        print("Plotting learning curves...")
        plot_learning_curves(metrics_data, output_path, metrics)

        print(f"Learning curves plotted successfully to {output_path}")

    except Exception as e:
        print(f"Error plotting learning curves: {e}")


if __name__ == "__main__":
    app()
