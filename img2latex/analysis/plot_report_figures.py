#!/usr/bin/env python
"""
Generate report-ready figures from training metrics.
Creates various plots for model analysis and performance evaluation.
"""

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from rich.console import Console

from img2latex.utils.visualization import (
    DEFAULT_THEME,
    apply_dark_theme,
    ensure_plots_dir,
)

app = typer.Typer(help="Generate report-ready plots from training metrics")
console = Console()


def load_metrics(metrics_file: Path) -> pd.DataFrame:
    """Load metrics from JSON file and convert to DataFrame."""
    with open(metrics_file, "r") as f:
        metrics_data = json.load(f)

    # Check if the metrics are in the 'steps' format
    if "steps" in metrics_data:
        # Extract data from each step
        data = []
        for step, metrics in metrics_data["steps"].items():
            metrics["step"] = int(step)
            data.append(metrics)

        return pd.DataFrame(data)
    else:
        # Handle other formats if needed
        raise ValueError("Unsupported metrics format")


def plot_training_curves(
    metrics_df: pd.DataFrame, output_dir: Path, theme: Dict = None
):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        metrics_df: DataFrame with metrics data
        output_dir: Directory to save plots
        theme: Theme settings (uses DEFAULT_THEME if None)
    """
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    plt.figure(figsize=(10, 6), facecolor=theme["background_color"])

    # Plot loss curves
    plt.plot(metrics_df["epoch"], metrics_df["train_loss"], "b-", label="Training Loss")
    plt.plot(metrics_df["epoch"], metrics_df["val_loss"], "r-", label="Validation Loss")

    plt.title("Training and Validation Loss", fontsize=16, color=theme["text_color"])
    plt.xlabel("Epoch", fontsize=14, color=theme["text_color"])
    plt.ylabel("Loss", fontsize=14, color=theme["text_color"])
    plt.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])

    # Create a more visible legend
    legend = plt.legend(
        fontsize=12,
        frameon=True,
        facecolor=theme["background_color"],
        edgecolor=theme["grid_color"],
    )
    for text in legend.get_texts():
        text.set_color(theme["text_color"])

    plt.tight_layout()
    loss_path = output_dir / "loss_curves.png"
    plt.savefig(loss_path, dpi=300, facecolor=theme["background_color"])
    plt.close()
    console.print(f"[green]Saved loss curves to {loss_path}[/green]")

    # Plot accuracy curves
    plt.figure(figsize=(10, 6), facecolor=theme["background_color"])
    plt.plot(
        metrics_df["epoch"], metrics_df["train_acc"], "b-", label="Training Accuracy"
    )
    plt.plot(
        metrics_df["epoch"], metrics_df["val_acc"], "r-", label="Validation Accuracy"
    )

    plt.title(
        "Training and Validation Accuracy", fontsize=16, color=theme["text_color"]
    )
    plt.xlabel("Epoch", fontsize=14, color=theme["text_color"])
    plt.ylabel("Accuracy", fontsize=14, color=theme["text_color"])
    plt.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])

    # Create a more visible legend
    legend = plt.legend(
        fontsize=12,
        frameon=True,
        facecolor=theme["background_color"],
        edgecolor=theme["grid_color"],
    )
    for text in legend.get_texts():
        text.set_color(theme["text_color"])

    plt.tight_layout()
    acc_path = output_dir / "accuracy_curves.png"
    plt.savefig(acc_path, dpi=300, facecolor=theme["background_color"])
    plt.close()
    console.print(f"[green]Saved accuracy curves to {acc_path}[/green]")


def plot_bleu_levenshtein(
    metrics_df: pd.DataFrame, output_dir: Path, theme: Dict = None
):
    """
    Plot BLEU and Levenshtein scores over epochs.

    Args:
        metrics_df: DataFrame with metrics data
        output_dir: Directory to save plots
        theme: Theme settings (uses DEFAULT_THEME if None)
    """
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Plot BLEU score progression
    plt.figure(figsize=(10, 6), facecolor=theme["background_color"])
    plt.plot(
        metrics_df["epoch"],
        metrics_df["val_bleu"],
        "-",
        marker="o",
        markersize=4,
        color=theme["main_color"],
    )

    plt.title("BLEU Score Progression", fontsize=16, color=theme["text_color"])
    plt.xlabel("Epoch", fontsize=14, color=theme["text_color"])
    plt.ylabel("BLEU Score", fontsize=14, color=theme["text_color"])
    plt.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])

    plt.tight_layout()
    bleu_path = output_dir / "bleu_score.png"
    plt.savefig(bleu_path, dpi=300, facecolor=theme["background_color"])
    plt.close()
    console.print(f"[green]Saved BLEU score plot to {bleu_path}[/green]")

    # Plot Levenshtein distance
    plt.figure(figsize=(10, 6), facecolor=theme["background_color"])
    plt.plot(
        metrics_df["epoch"],
        metrics_df["val_levenshtein"],
        "-",
        marker="o",
        markersize=4,
        color=theme["bar_colors"][0],
    )

    plt.title(
        "Levenshtein Distance Progression", fontsize=16, color=theme["text_color"]
    )
    plt.xlabel("Epoch", fontsize=14, color=theme["text_color"])
    plt.ylabel("Levenshtein Distance", fontsize=14, color=theme["text_color"])
    plt.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])

    plt.tight_layout()
    lev_path = output_dir / "levenshtein_distance.png"
    plt.savefig(lev_path, dpi=300, facecolor=theme["background_color"])
    plt.close()
    console.print(f"[green]Saved Levenshtein distance plot to {lev_path}[/green]")


def plot_metrics_correlation(
    metrics_df: pd.DataFrame, output_dir: Path, theme: Dict = None
):
    """
    Create a correlation matrix of different metrics.

    Args:
        metrics_df: DataFrame with metrics data
        output_dir: Directory to save plots
        theme: Theme settings (uses DEFAULT_THEME if None)
    """
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Select relevant columns for correlation
    corr_cols = [
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "val_bleu",
        "val_levenshtein",
    ]

    corr_df = metrics_df[corr_cols]
    corr_matrix = corr_df.corr()

    plt.figure(figsize=(10, 8), facecolor=theme["background_color"])

    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        cmap=theme["cmap"],
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"color": "white"},
        cbar_kws={"shrink": 0.8},
    )

    # Set title with theme color
    plt.title("Metrics Correlation Matrix", fontsize=16, color=theme["text_color"])

    # Adjust colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(colors=theme["text_color"])

    # Adjust axis labels
    plt.xlabel("Metrics", fontsize=14, color=theme["text_color"])
    plt.ylabel("Metrics", fontsize=14, color=theme["text_color"])

    # Adjust tick parameters
    plt.xticks(color=theme["text_color"])
    plt.yticks(color=theme["text_color"])

    plt.tight_layout()
    corr_path = output_dir / "metrics_correlation.png"
    plt.savefig(corr_path, dpi=300, facecolor=theme["background_color"])
    plt.close()
    console.print(f"[green]Saved metrics correlation matrix to {corr_path}[/green]")


def plot_metrics_radar(metrics_df: pd.DataFrame, output_dir: Path, theme: Dict = None):
    """
    Create a radar chart showing model progress across metrics.

    Args:
        metrics_df: DataFrame with metrics data
        output_dir: Directory to save plots
        theme: Theme settings (uses DEFAULT_THEME if None)
    """
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Get metrics for first, middle, and last epoch
    epochs = sorted(metrics_df["epoch"].unique())

    first_epoch = epochs[0]
    middle_epoch = epochs[len(epochs) // 2]
    last_epoch = epochs[-1]

    selected_epochs = [first_epoch, middle_epoch, last_epoch]
    selected_data = metrics_df[metrics_df["epoch"].isin(selected_epochs)]

    # Set up the radar chart
    metrics = [
        "val_acc",
        "val_bleu",
        "1-val_loss/3",
        "1-val_levenshtein",
    ]  # Normalize loss

    # Number of metrics
    N = len(metrics)

    # Create angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw=dict(polar=True),
        facecolor=theme["background_color"],
    )
    ax.set_facecolor(theme["background_color"])

    # Add metric labels
    label_names = ["Accuracy", "BLEU Score", "Loss (inv)", "Levenshtein (inv)"]
    plt.xticks(angles[:-1], label_names, fontsize=12, color=theme["text_color"])
    plt.yticks(color=theme["text_color"])

    for i, epoch in enumerate(selected_epochs):
        values = []
        epoch_data = selected_data[selected_data["epoch"] == epoch]

        for metric in metrics:
            if metric == "1-val_loss/3":
                # Invert and normalize loss (lower is better)
                values.append(1 - epoch_data["val_loss"].values[0] / 3)
            elif metric == "1-val_levenshtein":
                # Invert Levenshtein (lower is better)
                values.append(1 - epoch_data["val_levenshtein"].values[0])
            else:
                values.append(epoch_data[metric].values[0])

        # Close the loop
        values += values[:1]

        # Plot the values with theme colors
        ax.plot(
            angles,
            values,
            linewidth=2,
            label=f"Epoch {epoch}",
            color=theme["bar_colors"][i % len(theme["bar_colors"])],
        )
        ax.fill(
            angles,
            values,
            alpha=0.25,
            color=theme["bar_colors"][i % len(theme["bar_colors"])],
        )

    # Create a more visible legend
    legend = plt.legend(
        loc="upper right",
        bbox_to_anchor=(0.1, 0.1),
        frameon=True,
        facecolor=theme["background_color"],
        edgecolor=theme["grid_color"],
    )
    for text in legend.get_texts():
        text.set_color(theme["text_color"])

    plt.title("Model Progress Across Metrics", fontsize=16, color=theme["text_color"])

    # Add grid
    ax.grid(True, color=theme["grid_color"], alpha=0.3)

    radar_path = output_dir / "metrics_radar.png"
    plt.savefig(radar_path, dpi=300, facecolor=theme["background_color"])
    plt.close()
    console.print(f"[green]Saved metrics radar chart to {radar_path}[/green]")


def create_composite_plot(
    metrics_df: pd.DataFrame, output_dir: Path, theme: Dict = None
):
    """
    Create a composite plot showing multiple metrics in a grid.

    Args:
        metrics_df: DataFrame with metrics data
        output_dir: Directory to save plots
        theme: Theme settings (uses DEFAULT_THEME if None)
    """
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    fig, axes = plt.subplots(
        2, 2, figsize=(16, 12), facecolor=theme["background_color"]
    )

    # Apply background color to all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor(theme["background_color"])

    # Plot loss curves
    axes[0, 0].plot(
        metrics_df["epoch"],
        metrics_df["train_loss"],
        "-",
        label="Training",
        color=theme["bar_colors"][0],
    )
    axes[0, 0].plot(
        metrics_df["epoch"],
        metrics_df["val_loss"],
        "-",
        label="Validation",
        color=theme["bar_colors"][1],
    )
    axes[0, 0].set_title("Loss Curves", fontsize=14, color=theme["text_color"])
    axes[0, 0].set_xlabel("Epoch", fontsize=12, color=theme["text_color"])
    axes[0, 0].set_ylabel("Loss", fontsize=12, color=theme["text_color"])
    axes[0, 0].grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])
    legend = axes[0, 0].legend(
        frameon=True, facecolor=theme["background_color"], edgecolor=theme["grid_color"]
    )
    for text in legend.get_texts():
        text.set_color(theme["text_color"])
    axes[0, 0].tick_params(colors=theme["text_color"])

    # Plot accuracy curves
    axes[0, 1].plot(
        metrics_df["epoch"],
        metrics_df["train_acc"],
        "-",
        label="Training",
        color=theme["bar_colors"][0],
    )
    axes[0, 1].plot(
        metrics_df["epoch"],
        metrics_df["val_acc"],
        "-",
        label="Validation",
        color=theme["bar_colors"][1],
    )
    axes[0, 1].set_title("Accuracy Curves", fontsize=14, color=theme["text_color"])
    axes[0, 1].set_xlabel("Epoch", fontsize=12, color=theme["text_color"])
    axes[0, 1].set_ylabel("Accuracy", fontsize=12, color=theme["text_color"])
    axes[0, 1].grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])
    legend = axes[0, 1].legend(
        frameon=True, facecolor=theme["background_color"], edgecolor=theme["grid_color"]
    )
    for text in legend.get_texts():
        text.set_color(theme["text_color"])
    axes[0, 1].tick_params(colors=theme["text_color"])

    # Plot BLEU score
    axes[1, 0].plot(
        metrics_df["epoch"],
        metrics_df["val_bleu"],
        "-",
        marker="o",
        markersize=4,
        color=theme["main_color"],
    )
    axes[1, 0].set_title("BLEU Score", fontsize=14, color=theme["text_color"])
    axes[1, 0].set_xlabel("Epoch", fontsize=12, color=theme["text_color"])
    axes[1, 0].set_ylabel("BLEU Score", fontsize=12, color=theme["text_color"])
    axes[1, 0].grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])
    axes[1, 0].tick_params(colors=theme["text_color"])

    # Plot Levenshtein distance
    axes[1, 1].plot(
        metrics_df["epoch"],
        metrics_df["val_levenshtein"],
        "-",
        marker="o",
        markersize=4,
        color=theme["bar_colors"][0],
    )
    axes[1, 1].set_title("Levenshtein Distance", fontsize=14, color=theme["text_color"])
    axes[1, 1].set_xlabel("Epoch", fontsize=12, color=theme["text_color"])
    axes[1, 1].set_ylabel(
        "Levenshtein Distance", fontsize=12, color=theme["text_color"]
    )
    axes[1, 1].grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])
    axes[1, 1].tick_params(colors=theme["text_color"])

    plt.suptitle(
        "Model Training Metrics", fontsize=18, color=theme["text_color"], y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle

    composite_path = output_dir / "composite_metrics.png"
    plt.savefig(composite_path, dpi=300, facecolor=theme["background_color"])
    plt.close()
    console.print(f"[green]Saved composite metrics plot to {composite_path}[/green]")


@app.command()
def generate(
    metrics_file: str = typer.Option(
        "outputs/img2latex_v1/metrics/metrics.json",
        help="Path to the metrics JSON file",
    ),
    output_dir: str = typer.Option(
        "outputs/img2latex_v1/plots", help="Directory to save the plots"
    ),
    theme_key: str = typer.Option("dark", help="Theme to use: 'dark' for dark mode"),
):
    """Generate report-ready figures from training metrics."""
    metrics_path = Path(metrics_file)
    output_path = Path(output_dir)

    # Ensure output directory exists
    ensure_plots_dir(output_path)

    # Use the dark theme
    theme = DEFAULT_THEME

    console.print(f"[bold blue]Loading metrics from {metrics_path}...[/bold blue]")
    metrics_df = load_metrics(metrics_path)

    console.print(
        f"[bold blue]Generating plots in {output_path} with dark theme...[/bold blue]"
    )

    # Create all plots with the theme
    plot_training_curves(metrics_df, output_path, theme)
    plot_bleu_levenshtein(metrics_df, output_path, theme)
    plot_metrics_correlation(metrics_df, output_path, theme)
    plot_metrics_radar(metrics_df, output_path, theme)
    create_composite_plot(metrics_df, output_path, theme)

    console.print(
        "[bold green]All plots generated successfully with dark theme![/bold green]"
    )


if __name__ == "__main__":
    app()
