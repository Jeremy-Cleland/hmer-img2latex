#!/usr/bin/env python
"""
Main entry point for the img2latex analysis toolkit.

Provides a unified CLI for all analysis tools:
- Image analysis
- Project analysis
- Learning curve plotting
- Token distribution analysis
- Error analysis
- Preprocessing visualization
"""

from typing import List, Optional

import typer

# Import analysis functions from individual scripts
from scripts.analyze_images import analyze_images
from scripts.analyze_project import analyze_project
from scripts.analyze_token_distribution import analyze_tokens
from scripts.error_analysis import analyze_errors
from scripts.plot_learning_curves import plot_learning_curves_from_file
from scripts.visualize_preprocessing import visualize_preprocessing

# Create Typer app
app = typer.Typer(help="img2latex analysis toolkit")


@app.command()
def images(
    image_folder: str = typer.Argument(..., help="Path to folder containing images"),
    output_dir: str = typer.Option(
        "outputs/image_analysis", help="Directory to save analysis results"
    ),
    max_images: Optional[int] = typer.Option(
        None, help="Maximum number of images to analyze (None for all)"
    ),
    rows: int = typer.Option(5, help="Number of rows in the sample image grid"),
    cols: int = typer.Option(6, help="Number of columns in the sample image grid"),
    bg_color: str = typer.Option(
        "#FFFFFF", help="Background color for visualizations (hex code)"
    ),
):
    """Analyze image characteristics in a dataset."""
    analyze_images(
        image_folder=image_folder,
        output_dir=output_dir,
        max_images=max_images,
        sample_grid_rows=rows,
        sample_grid_cols=cols,
        bg_color=bg_color,
    )


@app.command()
def project(
    config_path: str = typer.Argument(..., help="Path to the YAML config file"),
    base_dir: str = typer.Option(".", help="Base directory for the project"),
    output_dir: str = typer.Option(
        "outputs/project_analysis", help="Directory to save analysis results"
    ),
    detailed: bool = typer.Option(
        False, help="Perform detailed analysis (Git comparison, hyperparameter sweep)"
    ),
):
    """Analyze the img2latex project configuration and status."""
    analyze_project(
        config_path=config_path,
        base_dir=base_dir,
        output_dir=output_dir,
        detailed=detailed,
    )


@app.command()
def curves(
    metrics_file: str = typer.Argument(
        ..., help="Path to the metrics file (CSV or JSON)"
    ),
    output_dir: str = typer.Option(
        "outputs/learning_curves", help="Directory to save the output plots"
    ),
    metrics: Optional[List[str]] = typer.Option(
        None, help="List of metrics to plot (default: all metrics in the file)"
    ),
):
    """Plot learning curves from training metrics."""
    plot_learning_curves_from_file(
        metrics_file=metrics_file, output_dir=output_dir, metrics=metrics
    )


@app.command()
def tokens(
    predictions_file: str = typer.Argument(
        ..., help="Path to file containing predictions and ground truths"
    ),
    output_dir: str = typer.Option(
        "outputs/token_analysis", help="Directory to save analysis results"
    ),
    token_delimiter: str = typer.Option(" ", help="Delimiter to use for tokenization"),
    top_k: int = typer.Option(20, help="Number of top divergent tokens to report"),
):
    """Analyze token distributions in predictions and ground truths."""
    analyze_tokens(
        predictions_file=predictions_file,
        output_dir=output_dir,
        token_delimiter=token_delimiter,
        top_k=top_k,
    )


@app.command()
def errors(
    predictions_file: str = typer.Argument(
        ..., help="Path to file containing predictions and references with scores"
    ),
    output_dir: str = typer.Option(
        "outputs/error_analysis", help="Directory to save analysis results"
    ),
    samples_per_bucket: int = typer.Option(
        5, help="Number of examples to include per bucket in the report"
    ),
    random_seed: Optional[int] = typer.Option(
        None, help="Random seed for reproducible sampling"
    ),
):
    """Analyze errors in model predictions and generate a report."""
    analyze_errors(
        predictions_file=predictions_file,
        output_dir=output_dir,
        samples_per_bucket=samples_per_bucket,
        random_seed=random_seed,
    )


@app.command()
def preprocess(
    image_path: str = typer.Argument(..., help="Path to the input image"),
    output_dir: str = typer.Option(
        "outputs/preprocessing", help="Directory to save visualization"
    ),
    image_folder: Optional[str] = typer.Option(
        None, help="Path to folder with similar images (for stats)"
    ),
    bg_color: str = typer.Option("white", help="Background color for the plot"),
    cnn_mode: bool = typer.Option(
        True, help="Visualize CNN preprocessing (will show both pipelines anyway)"
    ),
):
    """Visualize preprocessing steps for an image in the latex recognition pipeline."""
    visualize_preprocessing(
        image_path=image_path,
        output_dir=output_dir,
        image_folder=image_folder,
        bg_color=bg_color,
        cnn_mode=cnn_mode,
    )


if __name__ == "__main__":
    app()
