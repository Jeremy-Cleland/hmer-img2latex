# Table of Contents
- __init__.py
- cli.py
- __main__.py
- analysis/metrics.py
- analysis/preprocess.py
- analysis/__init__.py
- analysis/tokens.py
- analysis/utils.py
- analysis/images.py
- analysis/curves.py
- analysis/errors.py
- analysis/project.py
- training/predictor.py
- training/metrics.py
- training/__init__.py
- training/trainer.py
- utils/logging.py
- utils/visualize_metrics.py
- utils/registry.py
- utils/__init__.py
- utils/mps_utils.py
- utils/path_utils.py
- configs/config.yaml
- configs/__init__.py
- model/decoder.py
- model/__init__.py
- model/encoder.py
- model/seq2seq.py
- data/transforms.py
- data/__init__.py
- data/dataset.py
- data/tokenizer.py
- data/utils.py

## File: __init__.py

- Extension: .py
- Language: python
- Size: 191 bytes
- Created: 2025-04-16 03:29:56
- Modified: 2025-04-16 03:29:56

### Code

```python
"""
img2latex - Image to LaTeX conversion using PyTorch.

This package provides tools and models for converting images of mathematical
expressions into LaTeX code.
"""

__version__ = "0.1.0"

```

## File: cli.py

- Extension: .py
- Language: python
- Size: 19438 bytes
- Created: 2025-04-16 22:16:16
- Modified: 2025-04-16 22:16:16

### Code

```python
"""
Command-line interface for the image-to-LaTeX model.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy
import torch
import torch.serialization
import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from img2latex.analysis import (
    curves as curves_app,
)
from img2latex.analysis import (
    errors as errors_app,
)
from img2latex.analysis import (
    images as images_app,
)
from img2latex.analysis import (
    metrics as metrics_app,
)
from img2latex.analysis import (
    preprocess as preprocess_app,
)
from img2latex.analysis import (
    project as project_app,
)
from img2latex.analysis import (
    tokens as tokens_app,
)
from img2latex.data.dataset import create_data_loaders
from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.model.seq2seq import Seq2SeqModel
from img2latex.training.predictor import Predictor
from img2latex.training.trainer import Trainer
from img2latex.utils.logging import configure_logging, get_logger
from img2latex.utils.mps_utils import set_device, set_seed
from img2latex.utils.path_utils import path_manager
from img2latex.utils.registry import experiment_registry
from img2latex.utils.visualize_metrics import (
    load_metrics_files,
    plot_metrics_over_time,
    print_prediction_samples,
    print_token_distribution,
)

app = typer.Typer(name="img2latex", help="Image to LaTeX conversion tool")
console = Console()
logger = get_logger(__name__)

# Create analysis sub-app
analysis_app = typer.Typer(help="Analysis tools for img2latex")
app.add_typer(analysis_app, name="analyze")
# Register analysis sub-commands
analysis_app.add_typer(images_app.app, name="images", help="Analyze images")
analysis_app.add_typer(project_app.app, name="project", help="Analyze project setup")
analysis_app.add_typer(curves_app.app, name="curves", help="Plot learning curves")
analysis_app.add_typer(
    tokens_app.app, name="tokens", help="Analyze token distributions"
)
analysis_app.add_typer(errors_app.app, name="errors", help="Perform error analysis")
analysis_app.add_typer(
    preprocess_app.app, name="preprocess", help="Visualize preprocessing steps"
)
analysis_app.add_typer(metrics_app.app, name="metrics", help="Analyze model metrics")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Loaded configuration as a dictionary
    """
    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


@app.command()
def train(
    config_path: str = typer.Option(
        "img2latex/configs/config.yaml", help="Path to configuration file"
    ),
    experiment_name: str = typer.Option("img2latex_v1", help="Name of the experiment"),
    checkpoint_path: Optional[str] = typer.Option(
        None, help="Path to checkpoint to resume training from"
    ),
    data_dir: Optional[str] = typer.Option(
        None, help="Path to data directory (overrides config)"
    ),
    device: Optional[str] = typer.Option(
        None, help="Device to use for training (cpu, cuda, mps)"
    ),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
):
    """Train the image-to-LaTeX model."""
    # Start a spinner
    with console.status("[bold green]Setting up training...", spinner="dots"):
        # Set random seed
        set_seed(seed)

        # Load configuration
        config = load_config(config_path)

        # Override data directory if provided
        if data_dir:
            config["data"]["data_dir"] = data_dir

        # Override device if provided
        if device:
            config["training"]["device"] = device

        # Override experiment name in config if it doesn't match the CLI argument
        if "training" in config and "experiment_name" in config["training"]:
            if config["training"]["experiment_name"] != experiment_name:
                console.print(
                    f"[yellow]Warning: Overriding experiment name in config ({config['training']['experiment_name']}) with CLI argument ({experiment_name})[/yellow]"
                )
                config["training"]["experiment_name"] = experiment_name
        else:
            if "training" not in config:
                config["training"] = {}
            config["training"]["experiment_name"] = experiment_name

        # Create directories first
        experiment_registry.path_manager.create_experiment_structure(experiment_name)

        # Set log directory in config
        if "logging" not in config:
            config["logging"] = {}
        log_dir = str(path_manager.get_log_dir(experiment_name))
        config["logging"]["log_dir"] = log_dir

        # Configure logging (now that we know the experiment directory)
        configure_logging(config)

        logger.info(f"Starting setup for experiment: {experiment_name}")
        logger.info(f"Log directory: {log_dir}")

        # Set device
        device_obj = set_device(config["training"]["device"])

        # Save config to experiment directory
        config_save_path = path_manager.get_config_path(experiment_name)
        with open(config_save_path, "w") as f:
            yaml.dump(config, f)

        logger.info(f"Saved configuration to {config_save_path}")

        # Create tokenizer
        tokenizer = LaTeXTokenizer(max_sequence_length=config["data"]["max_seq_length"])

        # Fit tokenizer on formulas file
        formulas_path = os.path.join(
            config["data"]["data_dir"], config["data"]["formulas_file"]
        )
        tokenizer.fit_on_formulas_file(formulas_path)

        # Create data loaders
        data_loaders = create_data_loaders(
            data_dir=config["data"]["data_dir"],
            tokenizer=tokenizer,
            model_type=config["model"]["name"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
        )

        # Create model
        encoder_params = config["model"]["encoder"]
        decoder_params = config["model"]["decoder"]
        embedding_dim = config["model"]["embedding_dim"]

        # Get the correct encoder params based on model type
        if config["model"]["name"] == "cnn_lstm":
            encoder_params = encoder_params["cnn"]
        else:
            encoder_params = encoder_params["resnet"]

        # Set embedding dimension
        encoder_params["embedding_dim"] = embedding_dim

        # Create model
        model = Seq2SeqModel(
            model_type=config["model"]["name"],
            vocab_size=tokenizer.vocab_size,
            encoder_params=encoder_params,
            decoder_params=decoder_params,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_loader=data_loaders["train"],
            val_loader=data_loaders["val"],
            config=config,
            experiment_name=experiment_name,
            device=device_obj,
        )

        # Load checkpoint if provided
        if checkpoint_path:
            trainer.load_checkpoint(checkpoint_path)

    # Train the model
    console.print(
        f"[bold green]Starting training for experiment '{experiment_name}'[/bold green]"
    )
    try:
        best_metrics = trainer.train()
        console.print("[bold green]Training completed successfully![/bold green]")
        console.print("Best validation metrics:")
        for key, value in best_metrics.items():
            if isinstance(value, float):
                console.print(f"  {key}: {value:.4f}")
            else:
                console.print(f"  {key}: {value}")
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        logger.exception("Training failed")
        raise typer.Exit(code=1)


@app.command()
def predict(
    checkpoint_path: str = typer.Argument(..., help="Path to trained model checkpoint"),
    image_path: str = typer.Argument(..., help="Path to image file"),
    beam_size: int = typer.Option(
        0, help="Beam size for beam search (0 for greedy search)"
    ),
    max_length: int = typer.Option(
        141, help="Maximum length of the generated sequence"
    ),
    temperature: float = typer.Option(1.0, help="Temperature for sampling"),
    top_k: int = typer.Option(0, help="Top-k sampling parameter"),
    top_p: float = typer.Option(0.0, help="Top-p (nucleus) sampling parameter"),
    device: Optional[str] = typer.Option(
        None, help="Device to use for inference (cpu, cuda, mps)"
    ),
):
    """Predict LaTeX for an image."""
    # Start a spinner
    with console.status("[bold green]Loading model...", spinner="dots"):
        # Set device
        device_obj = set_device(device)

        # Create predictor from checkpoint
        predictor = Predictor.from_checkpoint(
            checkpoint_path=checkpoint_path, device=device_obj
        )

    # Predict
    with console.status("[bold green]Generating LaTeX...", spinner="dots"):
        latex = predictor.predict(
            image=image_path,
            beam_size=beam_size,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    # Print result
    console.print("[bold green]Generated LaTeX:[/bold green]")
    console.print(f"[cyan]{latex}[/cyan]")


@app.command()
def evaluate(
    checkpoint_path: str = typer.Argument(..., help="Path to trained model checkpoint"),
    data_dir: str = typer.Argument(..., help="Path to data directory"),
    split: str = typer.Option(
        "test", help="Data split to evaluate on (train, val, test)"
    ),
    batch_size: int = typer.Option(32, help="Batch size for evaluation"),
    num_samples: Optional[int] = typer.Option(
        None, help="Number of samples to evaluate (None for all)"
    ),
    beam_size: int = typer.Option(
        0, help="Beam size for beam search (0 for greedy search)"
    ),
    device: Optional[str] = typer.Option(
        None, help="Device to use for evaluation (cpu, cuda, mps)"
    ),
):
    """Evaluate the model on a dataset."""
    # Start a spinner
    with console.status("[bold green]Loading model and data...", spinner="dots"):
        # Set device
        device_obj = set_device(device)

        # --- Infer experiment name from checkpoint path ---
        try:
            ckpt_path = Path(checkpoint_path)
            # Assumes path like outputs/experiment_name/checkpoints/...
            experiment_name = ckpt_path.parent.parent.name
            console.print(f"Inferred experiment name: {experiment_name}")
        except IndexError:
            console.print(
                "[yellow]Warning: Could not automatically determine experiment name from checkpoint path.[/yellow]"
            )
            experiment_name = None
        # --------------------------------------------------

        # Load checkpoint
        # Explicitly set weights_only=False and allow numpy scalars using add_safe_globals
        try:
            # Allow the specific scalar type and dtype reported in the errors
            torch.serialization.add_safe_globals(
                [
                    numpy._core.multiarray.scalar,  # Allow scalar types
                    numpy.dtype,  # Allow generic dtype objects
                    numpy.dtypes.Float64DType,  # Allow specific Float64DType object
                ]
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=device_obj, weights_only=False
            )
        except AttributeError as e:
            # Fallback if the specific numpy path doesn't exist
            console.print(
                f"[yellow]Warning: Could not find numpy type for add_safe_globals, trying without: {e}[/yellow]"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=device_obj, weights_only=False
            )
        except ImportError as e:
            # Fallback if numpy isn't available
            console.print(
                f"[yellow]Warning: NumPy import failed, trying torch.load without add_safe_globals: {e}[/yellow]"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=device_obj, weights_only=False
            )

        config = checkpoint.get("config", {})

        # Create predictor from checkpoint
        predictor = Predictor.from_checkpoint(
            checkpoint_path=checkpoint_path, device=device_obj
        )

        # Create data loader for evaluation
        max_samples = {"train": None, "val": None, "test": None}
        if num_samples:
            max_samples[split] = num_samples

        data_loaders = create_data_loaders(
            data_dir=data_dir,
            tokenizer=predictor.tokenizer,
            model_type=predictor.model_type,
            batch_size=batch_size,
            num_workers=4,
            max_samples=max_samples,
        )

        # Get the data loader for the specified split
        if split not in data_loaders:
            console.print(f"[bold red]Invalid split: {split}[/bold red]")
            raise typer.Exit(code=1)

        data_loader = data_loaders[split]

    # Evaluate
    console.print(
        f"[bold green]Evaluating on {split} split with {len(data_loader.dataset)} samples[/bold green]"
    )

    all_predictions = []
    all_targets = []
    all_results_for_saving = []  # List to store results for JSON export

    with Progress(
        SpinnerColumn(), TextColumn("[bold green]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("[bold green]Evaluating...", total=len(data_loader))

        for batch_idx, batch in enumerate(data_loader):
            # Get images and targets
            images = batch["images"].to(device_obj)
            targets = batch["formulas"].to(device_obj)

            # Get raw formulas for reference
            raw_formulas = batch["raw_formulas"]

            # Process the entire batch at once using batch prediction
            # This is much more efficient than processing images one by one
            latex_predictions = predictor.predict_batch(
                images=images,
                beam_size=beam_size,
                max_length=predictor.tokenizer.max_sequence_length,
                batch_size=len(images),  # Process the whole batch at once
            )

            # Process predictions and targets
            for i, latex in enumerate(latex_predictions):
                # Get the target formula
                target_ids = targets[i].cpu().numpy().tolist()
                # Get the raw target string
                raw_target = raw_formulas[i]

                # Filter out padding
                target_ids = [
                    idx for idx in target_ids if idx != predictor.tokenizer.pad_token_id
                ]

                # Convert prediction to token IDs
                pred_ids = predictor.tokenizer.encode(latex)

                # Add to lists
                all_predictions.append(pred_ids)
                all_targets.append(target_ids)
                # Add prediction and raw target to results list for saving
                all_results_for_saving.append(
                    {"prediction": latex, "reference": raw_target}
                )

            # Update progress
            progress.update(task, advance=1)

    # Calculate metrics
    from img2latex.training.metrics import calculate_metrics

    metrics = calculate_metrics(all_predictions, all_targets)

    # Print results
    console.print("[bold green]Evaluation Results:[/bold green]")
    console.print(f"BLEU-4 Score: {metrics['bleu']:.4f}")
    console.print(f"Levenshtein Similarity: {metrics['levenshtein']:.4f}")
    console.print(f"Number of Samples: {metrics['batch_size']}")

    # --- Save predictions if experiment name was found ---
    if experiment_name:
        try:
            predictions_dir = (
                path_manager.get_experiment_dir(experiment_name) / "predictions"
            )
            predictions_dir.mkdir(parents=True, exist_ok=True)
            save_path = predictions_dir / "predictions.json"

            with open(save_path, "w") as f:
                json.dump(all_results_for_saving, f, indent=2)

            console.print(f"[green]Predictions saved to: {save_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving predictions: {e}[/red]")
    # -----------------------------------------------------


@app.command("visualize")
def visualize_metrics(
    experiment_name: str = typer.Argument(..., help="Name of the experiment"),
    epoch: int = typer.Option(
        None, help="Specific epoch to visualize (default: latest)"
    ),
    output_dir: str = typer.Option(None, help="Directory to save plots"),
    plot_trends: bool = typer.Option(False, help="Plot metrics trends over epochs"),
):
    """Visualize enhanced metrics from training."""
    console.print(
        f"[bold green]Visualizing metrics for experiment: {experiment_name}[/bold green]"
    )

    # Get metrics directory
    metrics_dir = path_manager.get_metrics_dir(experiment_name)

    if not os.path.exists(metrics_dir):
        console.print(
            "[bold red]Error:[/bold red] Metrics directory not found for experiment."
        )
        return

    # Load metrics files
    metrics_list = load_metrics_files(metrics_dir, experiment_name)

    if not metrics_list:
        console.print("[bold red]Error:[/bold red] No metrics files found.")
        return

    console.print(f"[green]Loaded {len(metrics_list)} metrics files[/green]")

    # If epoch is specified, show details for that epoch
    if epoch is not None:
        metrics = next((m for m in metrics_list if m.get("epoch") == epoch), None)
        if not metrics:
            console.print(
                f"[bold red]Error:[/bold red] No metrics found for epoch {epoch}"
            )
            return
    else:
        # Use the latest epoch
        metrics = metrics_list[-1]
        console.print(f"[green]Using latest epoch: {metrics.get('epoch')}[/green]")

    # Print sample predictions
    print_prediction_samples(metrics)

    # Print token distribution
    print_token_distribution(metrics)

    # Plot trends if requested
    if plot_trends:
        plot_metrics_over_time(metrics_list, output_dir)
        console.print("[green]Plots generated successfully[/green]")


# Register analysis subcommands under analysis_app
analysis_app.add_typer(images_app.app, name="images", help="Analyze raw images")
analysis_app.add_typer(
    project_app.app, name="project", help="Inspect project configuration and data"
)
analysis_app.add_typer(curves_app.app, name="curves", help="Plot learning curves")
analysis_app.add_typer(
    tokens_app.app, name="tokens", help="Analyze token distributions"
)
analysis_app.add_typer(
    errors_app.app, name="errors", help="Generate error analysis report"
)
analysis_app.add_typer(
    preprocess_app.app, name="preprocess", help="Visualize preprocessing pipeline"
)

if __name__ == "__main__":
    app()

```

## File: __main__.py

- Extension: .py
- Language: python
- Size: 121 bytes
- Created: 2025-04-16 16:23:24
- Modified: 2025-04-16 16:23:24

### Code

```python
"""
Main entry point for the img2latex package.
"""

from img2latex.cli import app

if __name__ == "__main__":
    app()

```

## File: analysis/metrics.py

- Extension: .py
- Language: python
- Size: 10340 bytes
- Created: 2025-04-16 22:03:14
- Modified: 2025-04-16 22:03:14

### Code

```python
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

```

## File: analysis/preprocess.py

- Extension: .py
- Language: python
- Size: 13083 bytes
- Created: 2025-04-16 17:06:09
- Modified: 2025-04-16 17:06:09

### Code

```python
#!/usr/bin/env python
"""
Script to visualize the preprocessing steps for images in the latex recognition pipeline.

Features:
- Pad image to fixed width
- Display tensor visualization
- Show preprocessing steps with annotations
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
import yaml
from PIL import Image

from img2latex.analysis.utils import ensure_output_dir

# Create Typer app
app = typer.Typer(help="Visualize preprocessing steps for images")


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def pad_image(img: np.ndarray, target_width: int, pad_value: int = 255) -> np.ndarray:
    """Pad image to target width.

    Args:
        img: Input image array
        target_width: Desired width after padding
        pad_value: Value to use for padding (default: 255 for white)

    Returns:
        Padded image array
    """
    height, width = img.shape[:2]

    # No padding needed if image is already wider than target
    if width >= target_width:
        return img

    # Calculate padding
    pad_width = target_width - width

    # Create padded image
    if len(img.shape) == 3:  # Color image
        padded_img = (
            np.ones((height, target_width, img.shape[2]), dtype=img.dtype) * pad_value
        )
        padded_img[:, :width, :] = img
    else:  # Grayscale image
        padded_img = np.ones((height, target_width), dtype=img.dtype) * pad_value
        padded_img[:, :width] = img

    return padded_img


def show_image_tensor(
    ax: plt.Axes,
    tensor: Union[torch.Tensor, np.ndarray],
    title: str = "",
    cmap: Optional[str] = None,
) -> None:
    """Display image tensor on a matplotlib axes.

    Args:
        ax: Matplotlib axes to plot on
        tensor: Image tensor or array to display
        title: Title for the plot
        cmap: Colormap to use (default: None, which uses grayscale for 1-channel, RGB for 3-channel)
    """
    # Convert tensor to numpy if needed
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if tensor.requires_grad:
            tensor = tensor.detach()
        img_array = tensor.numpy()
    else:
        img_array = tensor

    # Squeeze singleton dimensions (e.g., batch size of 1)
    img_array = np.squeeze(img_array)

    # Handle channel dimension for plotting
    if len(img_array.shape) == 3:
        # Check if channels-first format (C,H,W)
        if img_array.shape[0] in [1, 3]:
            # Convert from (C,H,W) to (H,W,C)
            img_array = np.transpose(img_array, (1, 2, 0))

        # If single channel in last dimension, squeeze it
        if img_array.shape[2] == 1:
            img_array = np.squeeze(img_array, axis=2)

    # Display image
    ax.imshow(img_array, cmap=cmap)

    # Add title if provided
    if title:
        ax.set_title(title)

    # Remove axes
    ax.axis("off")


def get_image_stats(
    image_folder: Union[str, Path], num_samples: int = 1000
) -> Tuple[float, float, float, float]:
    """Get average statistics for images in a folder.

    Args:
        image_folder: Path to folder with images
        num_samples: Maximum number of images to sample

    Returns:
        Tuple of (mean_width, mean_height, mean_aspect_ratio, std_aspect_ratio)
    """
    image_folder = Path(image_folder)
    image_files = list(image_folder.glob("*.png"))

    # Sample image files if more than num_samples
    if len(image_files) > num_samples:
        import random

        random.shuffle(image_files)
        image_files = image_files[:num_samples]

    widths = []
    heights = []
    aspect_ratios = []

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if not widths:
        return 0, 0, 0, 0

    mean_width = sum(widths) / len(widths)
    mean_height = sum(heights) / len(heights)
    mean_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    std_aspect_ratio = (
        sum((r - mean_aspect_ratio) ** 2 for r in aspect_ratios) / len(aspect_ratios)
    ) ** 0.5

    return mean_width, mean_height, mean_aspect_ratio, std_aspect_ratio


def create_preprocessing_visualization(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    image_folder: Optional[Union[str, Path]] = None,
    bg_color: str = "white",
    cnn_mode: bool = True,
    model_config: dict = None,
) -> None:
    """Create visualization of preprocessing steps.

    Args:
        image_path: Path to input image
        output_path: Path to save visualization
        image_folder: Path to folder with similar images (for stats)
        bg_color: Background color for the plot
        cnn_mode: Whether to use CNN mode (grayscale) or ResNet mode (RGB)
        model_config: Model configuration for image dimensions
    """
    # Load the image
    img = Image.open(image_path)

    # Use default dimensions if model_config not provided
    if model_config is None:
        model_config = {
            "cnn": {"height": 64, "width": 800, "channels": 1},
            "resnet": {"height": 64, "width": 800, "channels": 3},
        }

    # Get dataset stats if folder provided
    stats_text = ""
    if image_folder:
        mean_width, mean_height, mean_aspect_ratio, std_aspect_ratio = get_image_stats(
            image_folder
        )
        stats_text = (
            f"Dataset stats: Mean size: {mean_width:.1f}x{mean_height:.1f}, "
            f"Mean aspect ratio: {mean_aspect_ratio:.2f}±{std_aspect_ratio:.2f}"
        )

    # Create a figure with 2 rows: CNN and ResNet preprocessing
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=bg_color)

    # Add a subtitle for each row
    axes[0, 0].text(
        -0.1,
        0.5,
        "CNN Pipeline",
        va="center",
        ha="right",
        transform=axes[0, 0].transAxes,
        fontsize=14,
        fontweight="bold",
    )

    axes[1, 0].text(
        -0.1,
        0.5,
        "ResNet Pipeline",
        va="center",
        ha="right",
        transform=axes[1, 0].transAxes,
        fontsize=14,
        fontweight="bold",
    )

    # Add overall title
    plt.suptitle(
        f"Image Preprocessing Visualization: {Path(image_path).name}",
        fontsize=16,
        fontweight="bold",
    )

    # Process image for each pipeline (CNN and ResNet)
    for row, mode in enumerate(["cnn", "resnet"]):
        # Get model-specific config
        mode_config = model_config[mode]
        target_height = mode_config["height"]
        target_width = mode_config["width"]

        # Step 1: Original image
        orig_img = np.array(img)
        show_image_tensor(axes[row, 0], orig_img, f"1. Original: {orig_img.shape}")

        # Step 2: Resize to fixed height
        width, height = img.size
        aspect_ratio = width / height
        resize_width = int(aspect_ratio * target_height)

        resized_img = img.resize((resize_width, target_height), Image.LANCZOS)
        resized_arr = np.array(resized_img)
        show_image_tensor(axes[row, 1], resized_arr, f"2. Resize: {resized_arr.shape}")

        # Step 3: Pad to fixed width
        if mode == "cnn":
            # For CNN, convert to grayscale first
            if len(resized_arr.shape) == 3:
                grayscale_img = resized_img.convert("L")
                resized_arr = np.array(grayscale_img)

            padded_arr = pad_image(resized_arr, target_width)
            show_image_tensor(
                axes[row, 2],
                padded_arr,
                f"3. Pad+Gray: {padded_arr.shape}",
                cmap="gray",
            )
        else:
            # For ResNet, keep RGB
            if len(resized_arr.shape) == 2:
                # If already grayscale, convert to RGB
                rgb_img = Image.fromarray(resized_arr).convert("RGB")
                resized_arr = np.array(rgb_img)

            padded_arr = pad_image(resized_arr, target_width)
            show_image_tensor(
                axes[row, 2], padded_arr, f"3. Pad+RGB: {padded_arr.shape}"
            )

        # Step 4: Convert to tensor and normalize
        if mode == "cnn":
            # Convert to tensor with shape [1, H, W]
            tensor = torch.from_numpy(padded_arr).float()
            tensor = tensor.unsqueeze(0) if len(tensor.shape) == 2 else tensor

            # Normalize to [0, 1]
            tensor = tensor / 255.0

            # Display normalized tensor
            show_image_tensor(
                axes[row, 3],
                tensor,
                f"4. Normalize: {tuple(tensor.shape)}",
                cmap="gray",
            )
        else:
            # Convert to tensor with shape [3, H, W]
            tensor = torch.from_numpy(padded_arr).float()
            tensor = tensor.permute(2, 0, 1)  # From HWC to CHW

            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            normalized = (tensor / 255.0 - mean) / std

            # For display, denormalize
            display_tensor = normalized * std + mean

            # Display normalized tensor
            show_image_tensor(
                axes[row, 3], display_tensor, f"4. Normalize: {tuple(normalized.shape)}"
            )

    # Add dataset stats if available
    if stats_text:
        plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=12)

    # Add config info to title
    cnn_dims = f"CNN: {model_config['cnn']['height']}x{model_config['cnn']['width']}"
    resnet_dims = (
        f"ResNet: {model_config['resnet']['height']}x{model_config['resnet']['width']}"
    )
    plt.figtext(
        0.5,
        0.97,
        f"Model dimensions: {cnn_dims}, {resnet_dims}",
        ha="center",
        fontsize=10,
    )

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close()

    print(f"Preprocessing visualization saved to {output_path}")


@app.command()
def visualize(
    config_path: str = typer.Option(
        "img2latex/configs/config.yaml", help="Path to the configuration file"
    ),
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
) -> None:
    """Visualize the preprocessing steps for an image."""
    # Load configuration
    cfg = load_config(config_path)

    # Use config for image_folder if not specified and image_path doesn't provide a folder
    if image_folder is None:
        if os.path.isdir(os.path.dirname(image_path)):
            image_folder = os.path.dirname(image_path)
        else:
            data_dir = cfg["data"]["data_dir"]
            img_dir = cfg["data"]["img_dir"]
            image_folder = os.path.join(data_dir, img_dir)

    # Ensure output directory exists
    output_path = ensure_output_dir(output_dir, "preprocess")

    # Create output filename from input filename
    input_filename = os.path.basename(image_path)
    output_filename = f"preprocessing_{input_filename}"
    full_output_path = output_path / output_filename

    # Extract model dimensions from config
    model_name = cfg["model"]["name"]

    # Get CNN and ResNet config
    cnn_config = cfg["model"]["encoder"]["cnn"]
    resnet_config = cfg["model"]["encoder"]["resnet"]

    # Create visualization
    print(f"Creating preprocessing visualization for {image_path}...")
    create_preprocessing_visualization(
        image_path=image_path,
        output_path=full_output_path,
        image_folder=image_folder,
        bg_color=bg_color,
        cnn_mode=cnn_mode,
        model_config={
            "cnn": {
                "height": cnn_config["img_height"],
                "width": cnn_config["img_width"],
                "channels": cnn_config["channels"],
            },
            "resnet": {
                "height": resnet_config["img_height"],
                "width": resnet_config["img_width"],
                "channels": resnet_config["channels"],
            },
        },
    )

    print(f"Visualization saved to {full_output_path}")


if __name__ == "__main__":
    app()

```

## File: analysis/__init__.py

- Extension: .py
- Language: python
- Size: 1096 bytes
- Created: 2025-04-16 21:51:44
- Modified: 2025-04-16 21:51:44

### Code

```python
# This file marks img2latex.analysis as a Python package
"""
Analysis tools for the img2latex project.

This package contains various tools for analyzing model performance,
data distributions, and visualizing results.

Important implementation notes:
-------------------------------

1. All analysis modules should load and use values from the config file
   to ensure consistency with the actual training and inference code:

   - preprocess.py: Uses image dimensions (height, width, channels) from config
   - tokens.py: Uses max_seq_length from config for consistent tokenization
   - errors.py: Can use max_seq_length from config for error analysis

2. When adding new analysis modules, ensure they use the standard pattern:
   - Import config loading helper
   - Take config_path as first parameter with default to "img2latex/configs/config.yaml"
   - Extract and use relevant parameters from the config

This ensures that when training parameters change, analysis modules will automatically
use the updated values, preventing inconsistencies between analysis and actual model behavior.
"""

```

## File: analysis/tokens.py

- Extension: .py
- Language: python
- Size: 19073 bytes
- Created: 2025-04-16 22:11:28
- Modified: 2025-04-16 22:11:28

### Code

```python
#!/usr/bin/env python
"""
Script to analyze token distributions in predictions vs references.

Features:
- Load predictions and references
- Calculate token frequencies
- Compute KL divergence between distributions
- Identify the most divergent tokens
- Visualize token distributions
"""

import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import typer
import yaml
from rich.console import Console
from scipy.stats import entropy

from img2latex.analysis.utils import (
    ensure_output_dir,
    load_csv_file,
    load_json_file,
    save_json_file,
)

# Import PathManager
from img2latex.utils.path_utils import path_manager

# Create Typer app
app = typer.Typer(help="Analyze token distributions in img2latex predictions")

# Create console for rich output
console = Console()


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_predictions_data(file_path: Union[str, Path]) -> Tuple[List[str], List[str]]:
    """Load predictions and ground truth data from a file.

    Args:
        file_path: Path to the predictions file (CSV or JSON)

    Returns:
        Tuple of (predictions, ground_truths) as lists of strings

    Raises:
        ValueError: If the file format is not supported or data is missing
    """
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()

    predictions = []
    ground_truths = []

    if file_extension == ".csv":
        data = load_csv_file(file_path)

        # Check for required columns
        required_columns = ["prediction", "reference"]
        missing_columns = [col for col in required_columns if col not in data[0].keys()]

        if missing_columns:
            # Try alternate column names
            alternates = {
                "prediction": ["hypothesis", "pred", "output", "decoded"],
                "reference": ["ground_truth", "target", "label", "gold", "true"],
            }

            # For each missing column, check for alternates
            for missing in missing_columns:
                for alt in alternates[missing]:
                    if alt in data[0].keys():
                        # Replace missing column name with alternate
                        missing_columns.remove(missing)
                        if missing == "prediction":
                            predictions = [row[alt] for row in data]
                        else:
                            ground_truths = [row[alt] for row in data]
                        break

        # If still missing columns, raise error
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")

        # If columns were found directly
        if not predictions:
            predictions = [row["prediction"] for row in data]
        if not ground_truths:
            ground_truths = [row["reference"] for row in data]

    elif file_extension == ".json":
        data = load_json_file(file_path)

        # Check data structure
        if isinstance(data, list):
            # If data is a list of dictionaries
            if all("prediction" in item and "reference" in item for item in data):
                predictions = [item["prediction"] for item in data]
                ground_truths = [item["reference"] for item in data]
            elif all("hypothesis" in item and "reference" in item for item in data):
                predictions = [item["hypothesis"] for item in data]
                ground_truths = [item["reference"] for item in data]
            else:
                raise ValueError(
                    "JSON list items must contain 'prediction'/'hypothesis' and 'reference' fields"
                )
        elif isinstance(data, dict):
            # If data is a dictionary
            if "predictions" in data and "references" in data:
                predictions = data["predictions"]
                ground_truths = data["references"]
            elif "hypotheses" in data and "references" in data:
                predictions = data["hypotheses"]
                ground_truths = data["references"]
            else:
                raise ValueError(
                    "JSON dict must contain 'predictions'/'hypotheses' and 'references' fields"
                )
        else:
            raise ValueError("Unsupported JSON data structure")
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    # Ensure same length
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Mismatch in number of predictions ({len(predictions)}) and ground truths ({len(ground_truths)})"
        )

    return predictions, ground_truths


def tokenize_sequences(
    sequences: List[str], delimiter: str = " ", max_length: Optional[int] = None
) -> List[List[str]]:
    """Tokenize sequences using the specified delimiter.

    Args:
        sequences: List of text sequences
        delimiter: Token delimiter (default: space)
        max_length: Maximum sequence length for truncation (default: None, no truncation)

    Returns:
        List of tokenized sequences
    """
    tokenized = [seq.split(delimiter) for seq in sequences]

    # Truncate if max_length is specified
    if max_length is not None:
        tokenized = [seq[:max_length] for seq in tokenized]

    return tokenized


def compute_token_frequencies(tokenized_sequences: List[List[str]]) -> Counter:
    """Compute token frequencies from tokenized sequences.

    Args:
        tokenized_sequences: List of tokenized sequences

    Returns:
        Counter with token frequencies
    """
    # Flatten the list of token lists and count occurrences
    all_tokens = [token for seq in tokenized_sequences for token in seq]
    return Counter(all_tokens)


def calculate_kl_divergence(
    pred_freqs: Counter, truth_freqs: Counter
) -> Tuple[float, Dict[str, float]]:
    """Calculate KL-divergence between prediction and ground truth token distributions.

    Args:
        pred_freqs: Counter with prediction token frequencies
        truth_freqs: Counter with ground truth token frequencies

    Returns:
        Tuple of (overall_kl_divergence, token_kl_contributions)
    """
    # Get all unique tokens
    all_tokens = set(list(pred_freqs.keys()) + list(truth_freqs.keys()))

    # Create probability vectors (normalized frequencies)
    total_pred = sum(pred_freqs.values())
    total_truth = sum(truth_freqs.values())

    # Smoothing to avoid zeros (add small epsilon to all counts)
    epsilon = 1e-10

    # Calculate probabilities for each token
    p_truth = np.array(
        [
            (truth_freqs.get(token, 0) + epsilon)
            / (total_truth + len(all_tokens) * epsilon)
            for token in all_tokens
        ]
    )

    p_pred = np.array(
        [
            (pred_freqs.get(token, 0) + epsilon)
            / (total_pred + len(all_tokens) * epsilon)
            for token in all_tokens
        ]
    )

    # Calculate KL-divergence
    kl = entropy(p_truth, p_pred)

    # Calculate per-token contributions to KL-divergence
    token_kl = {}
    for i, token in enumerate(all_tokens):
        token_contribution = p_truth[i] * np.log(p_truth[i] / p_pred[i])
        token_kl[token] = token_contribution

    return kl, token_kl


def find_divergent_tokens(
    token_kl: Dict[str, float], top_k: int = 20
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Find the most over- and under-represented tokens.

    Args:
        token_kl: Dictionary of token KL-divergence contributions
        top_k: Number of top tokens to return

    Returns:
        Tuple of (under_represented_tokens, over_represented_tokens)
    """
    # Sort tokens by KL contribution
    sorted_tokens = sorted(token_kl.items(), key=lambda x: x[1], reverse=True)

    # Get top-K under-represented (highest KL contribution)
    under_represented = sorted_tokens[:top_k]

    # Get top-K over-represented (lowest or negative KL contribution)
    over_represented = sorted(token_kl.items(), key=lambda x: x[1])[:top_k]

    return under_represented, over_represented


def plot_token_distributions(
    pred_freqs: Counter, truth_freqs: Counter, output_dir: Path, top_k: int = 30
) -> None:
    """Plot token frequency histograms for predictions and ground truth.

    Args:
        pred_freqs: Counter with prediction token frequencies
        truth_freqs: Counter with ground truth token frequencies
        output_dir: Directory to save output plots
        top_k: Number of top tokens to display
    """
    # Get top tokens by frequency in ground truth
    top_truth_tokens = [token for token, _ in truth_freqs.most_common(top_k)]

    # Create frequency lists for top tokens
    truth_values = [truth_freqs.get(token, 0) for token in top_truth_tokens]
    pred_values = [pred_freqs.get(token, 0) for token in top_truth_tokens]

    # Create figure
    plt.figure(figsize=(15, 8))

    # Create bar positions
    x = np.arange(len(top_truth_tokens))
    width = 0.35

    # Create bars
    plt.bar(x - width / 2, truth_values, width, label="Ground Truth")
    plt.bar(x + width / 2, pred_values, width, label="Predictions")

    # Add labels and title
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.title("Token Frequency Comparison")
    plt.xticks(x, top_truth_tokens, rotation=45, ha="right")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "token_frequency_comparison.png")
    plt.close()

    # Plot distribution of token frequencies (log scale)
    plt.figure(figsize=(10, 6))

    # Get frequencies
    truth_counts = list(truth_freqs.values())
    pred_counts = list(pred_freqs.values())

    # Create histograms (log scale)
    plt.hist(truth_counts, bins=50, alpha=0.5, label="Ground Truth")
    plt.hist(pred_counts, bins=50, alpha=0.5, label="Predictions")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Token Frequency (log scale)")
    plt.ylabel("Count (log scale)")
    plt.title("Distribution of Token Frequencies")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "token_frequency_distribution.png")
    plt.close()


def print_divergence_report(
    under_represented: List[Tuple[str, float]],
    over_represented: List[Tuple[str, float]],
    kl_divergence: float,
    output_dir: Path,
) -> None:
    """Print and save report of token divergence analysis.

    Args:
        under_represented: List of (token, kl_value) tuples for under-represented tokens
        over_represented: List of (token, kl_value) tuples for over-represented tokens
        kl_divergence: Overall KL-divergence value
        output_dir: Directory to save the report
    """
    # Build report string
    report = [
        "# Token Distribution Analysis Report\n",
        f"## Overall KL-Divergence: {kl_divergence:.6f}\n",
        "\n## Under-represented Tokens\n",
        "Tokens that appear less frequently in predictions than in ground truth:\n",
        "| Token | KL Contribution |\n",
        "|-------|----------------|\n",
    ]

    for token, kl in under_represented:
        report.append(f"| `{token}` | {kl:.6f} |\n")

    report.extend(
        [
            "\n## Over-represented Tokens\n",
            "Tokens that appear more frequently in predictions than in ground truth:\n",
            "| Token | KL Contribution |\n",
            "|-------|----------------|\n",
        ]
    )

    for token, kl in over_represented:
        report.append(f"| `{token}` | {kl:.6f} |\n")

    # Write to file
    with open(output_dir / "token_divergence_report.md", "w") as f:
        f.writelines(report)

    # Print summary to console
    print(f"Overall KL-divergence: {kl_divergence:.6f}")
    print("\nTop 5 under-represented tokens:")
    for token, kl in under_represented[:5]:
        print(f"  {token}: {kl:.6f}")

    print("\nTop 5 over-represented tokens:")
    for token, kl in over_represented[:5]:
        print(f"  {token}: {kl:.6f}")

    print(f"\nFull report saved to {output_dir / 'token_divergence_report.md'}")


@app.callback(invoke_without_command=True)
def analyze(
    ctx: typer.Context,
    config_path: str = typer.Option(
        "img2latex/configs/config.yaml", help="Path to the configuration file"
    ),
    predictions_file: Optional[str] = typer.Option(
        None,
        "--predictions-file",
        "-p",
        help="Path to the specific predictions JSON/CSV file (use this or --experiment)",
    ),
    experiment: Optional[str] = typer.Option(
        None,
        "--experiment",
        "-e",
        help="Name of the experiment to analyze predictions from (use this or --predictions-file)",
    ),
    output_dir: str = typer.Option(
        "outputs/token_analysis", help="Directory to save the token analysis results"
    ),
    top_k: int = typer.Option(20, help="Number of most divergent tokens to display"),
    token_delimiter: str = typer.Option(
        " ", help="Delimiter used to tokenize the sequences"
    ),
    apply_max_length: bool = typer.Option(
        True, help="Apply max sequence length from config to tokenization"
    ),
) -> None:
    """Analyze token distributions in predictions versus references."""
    # Prevent running if a subcommand was invoked (though none are defined)
    if ctx.invoked_subcommand is not None:
        return

    # Validate arguments
    if not predictions_file and not experiment:
        console.print(
            "[red]Error: Please provide either --predictions-file or --experiment.[/red]"
        )
        raise typer.Exit(code=1)
    if predictions_file and experiment:
        console.print(
            "[red]Error: Please provide either --predictions-file or --experiment, not both.[/red]"
        )
        raise typer.Exit(code=1)

    # Determine the effective predictions file path
    effective_predictions_file: Path
    if experiment:
        try:
            # Assume predictions are stored in <experiment_dir>/predictions/predictions.json
            experiment_dir = path_manager.get_experiment_dir(experiment)
            predictions_subdir = experiment_dir / "predictions"
            # Look for .json or .csv
            potential_files = list(predictions_subdir.glob("predictions.*"))
            json_file = predictions_subdir / "predictions.json"
            csv_file = predictions_subdir / "predictions.csv"

            if json_file.exists():
                effective_predictions_file = json_file
            elif csv_file.exists():
                effective_predictions_file = csv_file
            else:
                console.print(
                    f"[red]Error: Could not find 'predictions.json' or 'predictions.csv' in {predictions_subdir}[/red]"
                )
                raise typer.Exit(code=1)

            console.print(
                f"Using predictions file for experiment '{experiment}': {effective_predictions_file}"
            )
        except Exception as e:
            console.print(
                f"[red]Error finding predictions file for experiment '{experiment}': {e}[/red]"
            )
            raise typer.Exit(code=1)
    else:
        # Use the provided predictions_file path
        effective_predictions_file = Path(predictions_file)
        if not effective_predictions_file.exists():
            console.print(
                f"[red]Error: Predictions file not found: {effective_predictions_file}[/red]"
            )
            raise typer.Exit(code=1)
        console.print(f"Using provided predictions file: {effective_predictions_file}")

    # Load configuration (optional, might be needed for max_length)
    cfg = {}
    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        console.print(
            f"[yellow]Warning: Config file {config_path} not found. Max length will not be applied from config.[/yellow]"
        )

    # Ensure output directory exists
    output_path = ensure_output_dir(
        output_dir, f"token_analysis_{experiment or 'custom'}"
    )

    console.print(f"Loading predictions from {effective_predictions_file}...")
    try:
        # Load prediction data using the effective path
        predictions, ground_truths = load_predictions_data(effective_predictions_file)

        # Tokenization
        max_len = None
        if apply_max_length and cfg:
            max_len = cfg.get("data", {}).get("max_seq_length")
            if max_len:
                console.print(f"Applying max sequence length from config: {max_len}")

        console.print("Tokenizing sequences...")
        pred_tokens = tokenize_sequences(predictions, token_delimiter, max_len)
        truth_tokens = tokenize_sequences(ground_truths, token_delimiter, max_len)

        # Compute token frequencies
        reference_freqs = compute_token_frequencies(truth_tokens)
        prediction_freqs = compute_token_frequencies(pred_tokens)

        # Calculate KL divergence
        kl_div, token_divergences = calculate_kl_divergence(
            reference_freqs, prediction_freqs
        )

        console.print(f"\n[bold]Overall KL divergence: {kl_div:.4f}[/bold]")

        # Find most divergent tokens
        divergent_tokens = find_divergent_tokens(
            reference_freqs, prediction_freqs, token_divergences, top_k
        )

        # Print divergence report
        print_divergence_report(divergent_tokens, reference_freqs, prediction_freqs)

        # Visualize token distributions
        visualize_file = output_path / "token_distributions.png"
        plot_token_distributions(
            divergent_tokens, reference_freqs, prediction_freqs, visualize_file
        )

        # Save detailed results to JSON
        results = {
            "overall_kl_divergence": kl_div,
            "divergent_tokens": [
                {
                    "token": token,
                    "reference_freq": reference_freqs[token],
                    "prediction_freq": prediction_freqs.get(token, 0),
                    "divergence": divergence,
                }
                for token, divergence in divergent_tokens
            ],
            "token_stats": {
                "unique_reference_tokens": len(reference_freqs),
                "unique_prediction_tokens": len(prediction_freqs),
                "reference_token_count": sum(reference_freqs.values()),
                "prediction_token_count": sum(prediction_freqs.values()),
            },
        }

        results_file = output_path / "token_analysis.json"
        save_json_file(results, results_file)

        console.print(
            f"[bold green]Token analysis complete. Results saved to {output_path}[/bold green]"
        )
    except Exception as e:
        console.print(f"[bold red]Error processing predictions: {e}[/bold red]")


if __name__ == "__main__":
    app()

```

## File: analysis/utils.py

- Extension: .py
- Language: python
- Size: 3924 bytes
- Created: 2025-04-16 17:11:28
- Modified: 2025-04-16 17:11:28

### Code

```python
#!/usr/bin/env python
"""
Common utility functions for analysis scripts in the img2latex project.
"""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class NumPyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy data types.

    This encoder converts NumPy arrays, ints, and floats to their Python equivalents
    for proper JSON serialization.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return super().default(obj)


def ensure_output_dir(base_dir: str, analysis_type: str) -> Path:
    """Ensure output directory exists, creating it if necessary.

    Args:
        base_dir: Base directory path (can be relative or absolute)
        analysis_type: Type of analysis (subdirectory name)

    Returns:
        Path object to the full output directory
    """
    if os.path.isabs(base_dir):
        # If absolute path is provided, use it directly
        output_dir = Path(base_dir)
    else:
        # If relative path, create it under the project root
        output_dir = Path(os.getcwd()) / base_dir

    # Add analysis type subdirectory
    if analysis_type:
        output_dir = output_dir / analysis_type

    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory set to: {output_dir}")

    return output_dir


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file and return its contents as a dictionary.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing the JSON data

    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file cannot be decoded as JSON
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save dictionary to a JSON file.

    Args:
        data: Dictionary to save
        file_path: Path where to save the JSON file
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumPyJSONEncoder)


def save_csv_file(
    data: List[Dict[str, Any]],
    file_path: Union[str, Path],
    fieldnames: Optional[List[str]] = None,
) -> None:
    """Save list of dictionaries to a CSV file.

    Args:
        data: List of dictionaries to save
        file_path: Path where to save the CSV file
        fieldnames: List of field names for the CSV header. If None, uses keys from the first dictionary.
    """
    if not data:
        print(f"Warning: No data to save to {file_path}")
        return

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def load_csv_file(file_path: Union[str, Path]) -> List[Dict[str, str]]:
    """Load CSV file and return its contents as a list of dictionaries.

    Args:
        file_path: Path to the CSV file

    Returns:
        List of dictionaries, where each dictionary represents a row

    Raises:
        FileNotFoundError: If the file does not exist
    """
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Returns:
        Path object pointing to the project root
    """
    # Assuming this script is in the scripts/ directory
    return Path(__file__).parent.parent.absolute()

```

## File: analysis/images.py

- Extension: .py
- Language: python
- Size: 19114 bytes
- Created: 2025-04-16 21:51:39
- Modified: 2025-04-16 21:51:39

### Code

```python
#!/usr/bin/env python
"""
Script to analyze image characteristics including sizes, color channels,
and pixel value distributions of LaTeX formula images.
"""

import glob
import os
import random
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer
import yaml
from matplotlib.gridspec import GridSpec
from PIL import Image

from img2latex.analysis.utils import ensure_output_dir, save_json_file

# Create Typer app
app = typer.Typer(help="Analyze image characteristics for the img2latex model")


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def analyze_images(image_folder, num_samples=500, detailed_samples=100):
    """
    Analyze images to determine size distribution, color channels, and pixel values.

    Args:
        image_folder: Path to folder containing images
        num_samples: Number of images to sample for basic analysis
        detailed_samples: Number of images to use for detailed pixel analysis

    Returns:
        Dictionary with image statistics
    """
    # Get all image files
    image_files = glob.glob(os.path.join(image_folder, "*.png"))

    # Sample if there are more images than num_samples
    if len(image_files) > num_samples:
        image_files = random.sample(image_files, num_samples)

    # Select a smaller subset for detailed pixel analysis
    detailed_files = image_files[: min(detailed_samples, len(image_files))]

    # Collect information
    widths = []
    heights = []
    aspect_ratios = []
    size_counts = defaultdict(int)

    # Color and pixel value analysis
    color_modes = []
    channel_counts = []
    dtypes = []
    min_values = []
    max_values = []
    mean_values = []
    std_values = []

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                # Size analysis
                width, height = img.size
                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height)
                size_counts[(width, height)] += 1

                # Color analysis
                color_modes.append(img.mode)

                # Convert to numpy for detailed analysis
                if img_path in detailed_files:
                    img_array = np.array(img)

                    # Determine number of channels
                    if len(img_array.shape) == 2:
                        channels = 1
                    else:
                        channels = img_array.shape[2]
                    channel_counts.append(channels)

                    # Pixel value analysis
                    dtypes.append(img_array.dtype)
                    min_values.append(float(img_array.min()))
                    max_values.append(float(img_array.max()))
                    mean_values.append(float(np.mean(img_array)))
                    std_values.append(float(np.std(img_array)))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Calculate size statistics
    stats = {
        "num_images": len(widths),
        "width_min": min(widths),
        "width_max": max(widths),
        "width_mean": np.mean(widths),
        "width_median": np.median(widths),
        "height_min": min(heights),
        "height_max": max(heights),
        "height_mean": np.mean(heights),
        "height_median": np.median(heights),
        "aspect_ratio_min": min(aspect_ratios),
        "aspect_ratio_max": max(aspect_ratios),
        "aspect_ratio_mean": np.mean(aspect_ratios),
        "aspect_ratio_median": np.median(aspect_ratios),
        "most_common_sizes": sorted(
            size_counts.items(), key=lambda x: x[1], reverse=True
        )[:10],
        "widths": widths,
        "heights": heights,
        "aspect_ratios": aspect_ratios,
        # Color and pixel statistics
        "color_modes": dict(Counter(color_modes)),
        "channel_counts": dict(Counter(channel_counts)) if channel_counts else {},
        "dtypes": dict(Counter(str(dt) for dt in dtypes)) if dtypes else {},
        "min_pixel_value": min(min_values) if min_values else None,
        "max_pixel_value": max(max_values) if max_values else None,
        "mean_pixel_value": np.mean(mean_values) if mean_values else None,
        "std_pixel_value": np.mean(std_values) if std_values else None,
        # Normalization check
        "is_normalized_0_1": (
            min(min_values) >= 0 and max(max_values) <= 1
            if min_values and max_values
            else None
        ),
        "is_normalized_neg1_1": (
            min(min_values) >= -1 and max(max_values) <= 1 and min(min_values) < 0
            if min_values and max_values
            else None
        ),
        "is_uint8": (
            min(min_values) >= 0
            and max(max_values) <= 255
            and any(str(dt) == "uint8" for dt in dtypes)
            if min_values and max_values and dtypes
            else None
        ),
    }

    return stats


def create_image_grid(
    image_folder, output_path, rows=5, cols=6, figsize=(15, 12), bg_color="#121212"
):
    """
    Create a grid of images with formulas.

    Args:
        image_folder: Path to folder containing images
        output_path: Path to save the output visualization
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        figsize: Figure size (width, height) in inches
        bg_color: Background color for the plot
    """
    # Get all image files
    image_files = glob.glob(os.path.join(image_folder, "*.png"))

    # Choose a random sample
    num_images = rows * cols
    if len(image_files) > num_images:
        selected_images = random.sample(image_files, num_images)
    else:
        selected_images = image_files[:num_images]

    # Create figure with dark background
    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    gs = GridSpec(rows, cols, figure=fig)

    # Load and display images
    for i, img_path in enumerate(selected_images):
        if i >= rows * cols:
            break

        row = i // cols
        col = i % cols

        # Create subplot
        ax = fig.add_subplot(gs[row, col])

        # Load image
        img = Image.open(img_path)
        img_array = np.array(img)

        # Invert colors for dark background if image is grayscale
        if len(img_array.shape) == 2 or (
            len(img_array.shape) == 3 and img_array.shape[2] == 1
        ):
            # For grayscale images, invert so formula appears white on dark background
            if np.mean(img_array) < 128:  # If image is mostly dark
                img_array = 255 - img_array  # Invert

        # Display image
        cmap = "gray" if len(img_array.shape) == 2 or img_array.shape[-1] == 1 else None
        ax.imshow(img_array, cmap=cmap, vmin=0, vmax=255)

        # Remove axes and set background color
        ax.axis("off")
        ax.set_facecolor(bg_color)

        # Add image filename as title
        filename = os.path.basename(img_path)
        ax.set_title(filename, color="white", fontsize=8)

    # Adjust layout and set overall background color
    plt.tight_layout()
    fig.patch.set_facecolor(bg_color)

    # Save the figure
    plt.savefig(output_path, facecolor=bg_color, bbox_inches="tight", dpi=300)
    print(f"Image grid saved to {output_path}")

    return fig


def visualize_size_distribution(stats, output_path, bg_color="#121212"):
    """
    Create visualizations of image size distributions.

    Args:
        stats: Dictionary with image statistics
        output_path: Path to save the output visualization
        bg_color: Background color for the plot
    """
    # Set style for plots
    sns.set(style="darkgrid")
    plt.rcParams["axes.facecolor"] = bg_color
    plt.rcParams["figure.facecolor"] = bg_color
    plt.rcParams["text.color"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["ytick.color"] = "white"

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 12), facecolor=bg_color)
    gs = GridSpec(2, 2, figure=fig)

    # Width distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(stats["widths"], ax=ax1, color="skyblue", kde=True)
    ax1.set_title("Width Distribution", color="white")
    ax1.set_xlabel("Width (pixels)")
    ax1.set_ylabel("Count")

    # Height distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(stats["heights"], ax=ax2, color="salmon", kde=True)
    ax2.set_title("Height Distribution", color="white")
    ax2.set_xlabel("Height (pixels)")
    ax2.set_ylabel("Count")

    # Aspect ratio distribution
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(stats["aspect_ratios"], ax=ax3, color="lightgreen", kde=True)
    ax3.set_title("Aspect Ratio Distribution", color="white")
    ax3.set_xlabel("Aspect Ratio (width/height)")
    ax3.set_ylabel("Count")

    # Width vs Height scatter plot
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(stats["widths"], stats["heights"], alpha=0.5, color="orchid")
    ax4.set_title("Width vs Height", color="white")
    ax4.set_xlabel("Width (pixels)")
    ax4.set_ylabel("Height (pixels)")

    # Add table with statistics
    table_text = (
        f"Number of images: {stats['num_images']}\n"
        f"Width range: {stats['width_min']} - {stats['width_max']} px\n"
        f"Width mean: {stats['width_mean']:.1f} px\n"
        f"Height range: {stats['height_min']} - {stats['height_max']} px\n"
        f"Height mean: {stats['height_mean']:.1f} px\n"
        f"Aspect ratio range: {stats['aspect_ratio_min']:.2f} - {stats['aspect_ratio_max']:.2f}\n"
        f"Aspect ratio mean: {stats['aspect_ratio_mean']:.2f}\n"
        f"Most common size: {stats['most_common_sizes'][0][0]} px ({stats['most_common_sizes'][0][1]} images)"
    )

    fig.text(
        0.5,
        0.01,
        table_text,
        fontsize=12,
        color="white",
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle("Image Size Analysis", fontsize=16, color="white")

    # Save the figure
    plt.savefig(output_path, facecolor=bg_color, bbox_inches="tight", dpi=300)
    print(f"Size distribution visualization saved to {output_path}")

    return fig


def visualize_pixel_values(stats, output_path, bg_color="#121212"):
    """
    Create visualizations of pixel value distributions and color modes.

    Args:
        stats: Dictionary with image statistics
        output_path: Path to save the output visualization
        bg_color: Background color for the plot
    """
    # Set style for plots
    sns.set(style="darkgrid")
    plt.rcParams["axes.facecolor"] = bg_color
    plt.rcParams["figure.facecolor"] = bg_color
    plt.rcParams["text.color"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["ytick.color"] = "white"

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 12), facecolor=bg_color)
    gs = GridSpec(2, 2, figure=fig)

    # Color modes pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    color_modes = stats["color_modes"]
    labels = color_modes.keys()
    sizes = color_modes.values()
    explode = [0.1] * len(labels)  # explode all slices
    ax1.pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
        textprops={"color": "white"},
    )
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.set_title("Image Color Modes", color="white")

    # Channel counts pie chart
    ax2 = fig.add_subplot(gs[0, 1])
    if stats["channel_counts"]:
        channel_counts = stats["channel_counts"]
        labels = [f"{k} channel{'s' if k > 1 else ''}" for k in channel_counts.keys()]
        sizes = channel_counts.values()
        explode = [0.1] * len(labels)  # explode all slices
        ax2.pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
            textprops={"color": "white"},
        )
        ax2.axis("equal")
    ax2.set_title("Number of Channels", color="white")

    # Pixel value range
    ax3 = fig.add_subplot(gs[1, 0])
    if stats["min_pixel_value"] is not None and stats["max_pixel_value"] is not None:
        pixel_range = [stats["min_pixel_value"], stats["max_pixel_value"]]
        bar_labels = ["Min", "Max"]
        ax3.bar(bar_labels, pixel_range, color=["blue", "red"])
        ax3.set_title("Pixel Value Range", color="white")
        ax3.set_ylabel("Pixel Value")

        # Annotate the exact values
        for i, v in enumerate(pixel_range):
            ax3.text(i, v / 2, f"{v:.2f}", ha="center", color="white")

    # Data types
    ax4 = fig.add_subplot(gs[1, 1])
    if stats["dtypes"]:
        dtypes = stats["dtypes"]
        labels = list(dtypes.keys())
        sizes = list(dtypes.values())
        ax4.bar(labels, sizes, color="purple")
        ax4.set_title("Image Data Types", color="white")
        ax4.set_ylabel("Count")
        ax4.tick_params(axis="x", rotation=45)

        # Annotate the bars with counts
        for i, v in enumerate(sizes):
            ax4.text(i, v / 2, str(v), ha="center", color="white")

    # Add normalization information
    normalization_text = "Pixel Value Analysis:\n"

    if stats["is_normalized_0_1"]:
        normalization_text += "✓ Images appear to be normalized to [0, 1] range\n"
    elif stats["is_normalized_neg1_1"]:
        normalization_text += "✓ Images appear to be normalized to [-1, 1] range\n"
    elif stats["is_uint8"]:
        normalization_text += "✓ Images appear to be in standard uint8 format (0-255)\n"
    else:
        normalization_text += "? Images have non-standard normalization\n"

    normalization_text += (
        f"Min: {stats['min_pixel_value']:.2f}, Max: {stats['max_pixel_value']:.2f}\n"
    )
    normalization_text += (
        f"Mean: {stats['mean_pixel_value']:.2f}, Std: {stats['std_pixel_value']:.2f}\n"
    )

    most_common_mode = max(stats["color_modes"].items(), key=lambda x: x[1])[0]
    normalization_text += (
        f"Most common color mode: {most_common_mode} "
        + f"({stats['color_modes'][most_common_mode]} images, "
        + f"{stats['color_modes'][most_common_mode] / stats['num_images'] * 100:.1f}%)"
    )

    fig.text(
        0.5,
        0.01,
        normalization_text,
        fontsize=12,
        color="white",
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle("Color and Pixel Value Analysis", fontsize=16, color="white")

    # Save the figure
    plt.savefig(output_path, facecolor=bg_color, bbox_inches="tight", dpi=300)
    print(f"Pixel value distribution visualization saved to {output_path}")

    return fig


@app.command()
def analyze(
    config_path: str = typer.Option(
        "img2latex/configs/config.yaml", help="Path to the configuration file"
    ),
    image_folder: str = typer.Option(
        None,
        help="Path to folder containing images (defaults to data_dir/img_dir from config)",
    ),
    output_dir: str = typer.Option(
        "outputs/image_analysis", help="Directory to save the output analysis"
    ),
    num_samples: int = typer.Option(
        500, help="Number of images to sample for basic analysis"
    ),
    detailed_samples: int = typer.Option(
        100, help="Number of images to use for detailed pixel analysis"
    ),
) -> None:
    """Analyze images to determine size distribution, color channels, and pixel values."""
    # Load configuration
    cfg = load_config(config_path)

    # Use config for image_folder if not specified

    data_dir = cfg["data"]["data_dir"]
    img_dir = cfg["data"]["img_dir"]
    image_folder = os.path.join(data_dir, img_dir)

    # Setup output directory
    output_path = ensure_output_dir(output_dir, "images")

    # Analyze images
    print(f"Analyzing images from {image_folder}...")
    stats = analyze_images(
        image_folder, num_samples=num_samples, detailed_samples=detailed_samples
    )

    # Save stats to JSON
    stats_path = output_path / "image_stats.json"
    # Use save_json_file that handles NumPy types
    save_json_file(stats, stats_path)

    # Print summary statistics
    print("\nImage Size Statistics:")
    print(f"Number of images analyzed: {stats['num_images']}")
    print(f"Width range: {stats['width_min']} - {stats['width_max']} pixels")
    print(f"Width mean: {stats['width_mean']:.1f} pixels")
    print(f"Height range: {stats['height_min']} - {stats['height_max']} pixels")
    print(f"Height mean: {stats['height_mean']:.1f} pixels")
    print(
        f"Aspect ratio range: {stats['aspect_ratio_min']:.2f} - {stats['aspect_ratio_max']:.2f}"
    )
    print(f"Aspect ratio mean: {stats['aspect_ratio_mean']:.2f}")

    print("\nMost common image sizes:")
    for (width, height), count in stats["most_common_sizes"]:
        print(f"  {width}x{height} pixels: {count} images")

    print("\nColor and Pixel Value Analysis:")
    print(f"Color modes: {stats['color_modes']}")
    print(f"Channel counts: {stats['channel_counts']}")
    print(f"Data types: {stats['dtypes']}")
    print(f"Pixel value range: {stats['min_pixel_value']} - {stats['max_pixel_value']}")
    print(f"Mean pixel value: {stats['mean_pixel_value']:.2f}")
    print(f"Std dev of pixel values: {stats['std_pixel_value']:.2f}")

    # Determine normalization
    if stats["is_normalized_0_1"]:
        print("Images appear to be normalized to [0, 1] range")
    elif stats["is_normalized_neg1_1"]:
        print("Images appear to be normalized to [-1, 1] range")
    elif stats["is_uint8"]:
        print("Images appear to be in standard uint8 format (0-255)")
    else:
        print("Images have non-standard normalization")

    # Create image grid
    print("\nCreating image grid...")
    grid_output_path = output_path / "formula_image_grid.png"
    create_image_grid(
        image_folder, grid_output_path, rows=5, cols=6, bg_color="#121212"
    )

    # Create size distribution visualization
    print("\nCreating size distribution visualization...")
    dist_output_path = output_path / "size_distribution.png"
    visualize_size_distribution(stats, dist_output_path, bg_color="#121212")

    # Create pixel value distribution visualization
    print("\nCreating pixel value distribution visualization...")
    pixel_output_path = output_path / "pixel_distribution.png"
    visualize_pixel_values(stats, pixel_output_path, bg_color="#121212")

    print(f"\nAnalysis complete. Results saved to {output_path}")


if __name__ == "__main__":
    app()

```

## File: analysis/curves.py

- Extension: .py
- Language: python
- Size: 6393 bytes
- Created: 2025-04-16 21:58:08
- Modified: 2025-04-16 21:58:08

### Code

```python
#!/usr/bin/env python
"""
Script to plot learning curves for the img2latex model training.

Features:
- Read metrics data from CSV or JSON
- Plot each metric vs epoch
- Save figures to output directory
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import typer
import yaml

from img2latex.analysis.utils import ensure_output_dir

# Create Typer app
app = typer.Typer(help="Plot learning curves from training metrics")


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
    config_path: str = typer.Option(
        "img2latex/configs/config.yaml", help="Path to the configuration file"
    ),
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
    # Load configuration
    cfg = load_config(config_path)

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

```

## File: analysis/errors.py

- Extension: .py
- Language: python
- Size: 18771 bytes
- Created: 2025-04-16 17:06:10
- Modified: 2025-04-16 17:06:10

### Code

```python
#!/usr/bin/env python
"""
Script for error analysis of the img2latex model predictions.

This script analyzes prediction errors, categorizes them by type,
and generates a comprehensive error analysis report.
"""

import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

import typer
import yaml
from Levenshtein import distance
from rich.console import Console

from img2latex.analysis.utils import (
    ensure_output_dir,
    load_csv_file,
    load_json_file,
    save_json_file,
)

# Create Typer app
app = typer.Typer(help="Error analysis for img2latex model predictions")

# Create console for rich output
console = Console()


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_predictions(file_path: Union[str, Path]) -> List[Dict[str, Union[str, float]]]:
    """Load predictions with scores from a file.

    Args:
        file_path: Path to the predictions file (CSV or JSON)

    Returns:
        List of prediction records with reference, hypothesis, and scores

    Raises:
        ValueError: If the file format is not supported or data is missing
    """
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()

    records = []

    if file_extension == ".csv":
        data = load_csv_file(file_path)

        # Check for required columns
        required_columns = ["reference", "hypothesis"]
        score_columns = ["edit_distance", "bleu", "levenshtein"]

        # Check if at least one score column exists
        has_scores = any(col in data[0] for col in score_columns)

        # For each record, create a standardized entry
        for i, row in enumerate(data):
            record = {"id": i}

            # Extract reference and hypothesis
            reference = None
            hypothesis = None

            if "reference" in row:
                reference = row["reference"]
            elif "ground_truth" in row:
                reference = row["ground_truth"]
            elif "target" in row:
                reference = row["target"]

            if "hypothesis" in row:
                hypothesis = row["hypothesis"]
            elif "prediction" in row:
                hypothesis = row["prediction"]
            elif "output" in row:
                hypothesis = row["output"]

            if reference is None or hypothesis is None:
                raise ValueError("Missing reference or hypothesis in data")

            record["reference"] = reference
            record["hypothesis"] = hypothesis

            # Extract scores
            for score_col in score_columns:
                if score_col in row:
                    try:
                        record[score_col] = float(row[score_col])
                    except ValueError:
                        # If not a valid float, skip
                        pass

            # If no scores present, calculate edit distance
            if "edit_distance" not in record and "levenshtein" not in record:
                record["edit_distance"] = distance(reference, hypothesis)

            records.append(record)

    elif file_extension == ".json":
        data = load_json_file(file_path)

        # Check data structure
        if isinstance(data, list):
            # Ensure each item has reference and hypothesis
            for i, item in enumerate(data):
                record = {"id": i}

                # Extract reference and hypothesis
                reference = (
                    item.get("reference")
                    or item.get("ground_truth")
                    or item.get("target")
                )
                hypothesis = (
                    item.get("hypothesis")
                    or item.get("prediction")
                    or item.get("output")
                )

                if reference is None or hypothesis is None:
                    raise ValueError(f"Missing reference or hypothesis in record {i}")

                record["reference"] = reference
                record["hypothesis"] = hypothesis

                # Extract scores
                for score_field in ["edit_distance", "bleu", "levenshtein"]:
                    if score_field in item:
                        try:
                            record[score_field] = float(item[score_field])
                        except ValueError:
                            # If not a valid float, skip
                            pass

                # If no scores present, calculate edit distance
                if "edit_distance" not in record and "levenshtein" not in record:
                    record["edit_distance"] = distance(reference, hypothesis)

                records.append(record)

        elif isinstance(data, dict):
            # Check if this is a structured results object
            if "results" in data and isinstance(data["results"], list):
                return load_predictions(data["results"])

            # Otherwise, try to extract parallel lists
            references = (
                data.get("references")
                or data.get("ground_truths")
                or data.get("targets")
            )
            hypotheses = (
                data.get("hypotheses") or data.get("predictions") or data.get("outputs")
            )

            if not references or not hypotheses:
                raise ValueError("Could not find reference/hypothesis lists in data")

            if len(references) != len(hypotheses):
                raise ValueError(
                    f"Mismatch in list lengths: {len(references)} references vs {len(hypotheses)} hypotheses"
                )

            # Extract scores if available
            bleu_scores = data.get("bleu_scores", [None] * len(references))
            edit_distances = data.get("edit_distances", [None] * len(references))
            levenshtein_scores = data.get(
                "levenshtein_scores", [None] * len(references)
            )

            # Create records
            for i in range(len(references)):
                record = {
                    "id": i,
                    "reference": references[i],
                    "hypothesis": hypotheses[i],
                }

                # Add scores if available
                if i < len(bleu_scores) and bleu_scores[i] is not None:
                    record["bleu"] = float(bleu_scores[i])

                if i < len(edit_distances) and edit_distances[i] is not None:
                    record["edit_distance"] = float(edit_distances[i])

                if i < len(levenshtein_scores) and levenshtein_scores[i] is not None:
                    record["levenshtein"] = float(levenshtein_scores[i])

                # If no distance/score available, calculate edit distance
                if "edit_distance" not in record and "levenshtein" not in record:
                    record["edit_distance"] = distance(references[i], hypotheses[i])

                records.append(record)
        else:
            raise ValueError("Unsupported JSON data structure")
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    return records


def bucket_by_edit_distance(
    records: List[Dict[str, Union[str, float]]],
    ranges: List[Tuple[int, int]] = [(0, 0), (1, 1), (2, 3), (4, float("inf"))],
) -> Dict[str, List[Dict[str, Union[str, float]]]]:
    """Group records into buckets based on edit distance ranges.

    Args:
        records: List of prediction records
        ranges: List of (min_dist, max_dist) tuples defining the ranges

    Returns:
        Dictionary mapping range names to lists of records
    """
    buckets = {
        f"{low}-{high if high != float('inf') else 'inf'}": [] for low, high in ranges
    }

    for record in records:
        # Use either edit_distance or levenshtein score
        distance = record.get("edit_distance", record.get("levenshtein", None))

        if distance is None:
            continue

        # Find the appropriate bucket
        for (low, high), bucket_name in zip(ranges, buckets.keys()):
            if low <= distance <= high:
                buckets[bucket_name].append(record)
                break

    return buckets


def identify_error_patterns(
    records: List[Dict[str, Union[str, float]]],
) -> List[Dict[str, Union[str, int]]]:
    """Identify common error patterns in the predictions.

    Args:
        records: List of prediction records

    Returns:
        List of error pattern dictionaries with pattern and count
    """
    # Initialize error patterns
    latex_error_patterns = {
        "missing_brace": r"(?<!\{)\}|(?<!\\)\{(?!\})",  # Unbalanced braces
        "missing_command": r"(?<!\\)(?:alpha|beta|gamma|delta|theta|lambda|pi|sigma)",  # Missing \before command
        "missing_backslash": r"(?<!\\)(?:frac|sqrt|sum|int|lim)",  # Missing \ before command
        "extra_space": r"\\[a-zA-Z]+\s+\{",  # Space between command and brace
        "missing_subscript": r"_(?!\{|[0-9a-zA-Z])",  # Underscore not followed by brace or character
        "missing_superscript": r"\^(?!\{|[0-9a-zA-Z])",  # Caret not followed by brace or character
        "incorrect_fraction": r"\\frac\{[^{}]*\}(?!\{)",  # \frac with only one argument
        "mismatched_braces": r"\{[^{}]*$|^[^{}]*\}",  # Unclosed or unopened brace
    }

    # Count pattern occurrences
    pattern_counter = Counter()

    for record in records:
        hyp = record["hypothesis"]
        ref = record["reference"]

        # Check for each error pattern
        for pattern_name, regex in latex_error_patterns.items():
            # Count in hypothesis
            hyp_matches = len(re.findall(regex, hyp))
            # Count in reference
            ref_matches = len(re.findall(regex, ref))

            # If more errors in hypothesis than reference, count the difference
            if hyp_matches > ref_matches:
                pattern_counter[pattern_name] += hyp_matches - ref_matches

    # Format results
    error_patterns = [
        {
            "pattern": pattern,
            "count": count,
            "description": get_pattern_description(pattern),
        }
        for pattern, count in pattern_counter.most_common()
        if count > 0
    ]

    return error_patterns


def get_pattern_description(pattern_name: str) -> str:
    """Return a human-readable description of an error pattern.

    Args:
        pattern_name: Name of the error pattern

    Returns:
        Description of the error pattern
    """
    descriptions = {
        "missing_brace": "Missing or unbalanced braces",
        "missing_command": "Missing backslash before command (like alpha, beta)",
        "missing_backslash": "Missing backslash before LaTeX command",
        "extra_space": "Extra space between command and opening brace",
        "missing_subscript": "Subscript (_) not followed by content",
        "missing_superscript": "Superscript (^) not followed by content",
        "incorrect_fraction": "Fraction with missing second argument",
        "mismatched_braces": "Mismatched opening/closing braces",
    }

    return descriptions.get(pattern_name, "Unknown error pattern")


def generate_error_report(
    buckets: Dict[str, List[Dict[str, Union[str, float]]]],
    error_patterns: List[Dict[str, Union[str, int]]],
    samples_per_bucket: int = 5,
    output_path: Path = None,
) -> str:
    """Generate a Markdown report of error analysis.

    Args:
        buckets: Dictionary of edit distance buckets
        error_patterns: List of error pattern dictionaries
        samples_per_bucket: Number of examples to include per bucket
        output_path: Path to save the report (if provided)

    Returns:
        Markdown report as a string
    """
    # Initialize report
    report = ["# Error Analysis Report\n\n", "## Error Buckets by Edit Distance\n\n"]

    # Add bucket statistics
    for bucket_name, records in buckets.items():
        report.append(f"### Bucket: Edit Distance {bucket_name}\n\n")
        report.append(f"- **Count**: {len(records)} examples\n")

        if records:
            # Calculate average BLEU score if available
            bleu_scores = [r.get("bleu", None) for r in records if "bleu" in r]
            if bleu_scores:
                avg_bleu = sum(bleu_scores) / len(bleu_scores)
                report.append(f"- **Average BLEU Score**: {avg_bleu:.4f}\n")

            # Sample examples from this bucket
            samples = random.sample(records, min(samples_per_bucket, len(records)))

            report.append("\n**Sample Examples:**\n\n")

            for i, sample in enumerate(samples):
                report.append(f"**Example {i + 1}:**\n\n")
                report.append(f"- **Reference**: `{sample['reference']}`\n")
                report.append(f"- **Hypothesis**: `{sample['hypothesis']}`\n")

                # Add scores
                if "bleu" in sample:
                    report.append(f"- **BLEU Score**: {sample['bleu']:.4f}\n")
                if "edit_distance" in sample:
                    report.append(f"- **Edit Distance**: {sample['edit_distance']}\n")
                elif "levenshtein" in sample:
                    report.append(
                        f"- **Levenshtein Distance**: {sample['levenshtein']}\n"
                    )

                report.append("\n")

        report.append("\n")

    # Add error pattern section
    report.append("## Common Error Patterns\n\n")

    if error_patterns:
        report.append("| Error Pattern | Count | Description |\n")
        report.append("|--------------|-------|-------------|\n")

        for pattern in error_patterns:
            report.append(
                f"| {pattern['pattern']} | {pattern['count']} | {pattern['description']} |\n"
            )
    else:
        report.append("No common error patterns identified.\n")

    # Join report into a single string
    report_text = "".join(report)

    # Save report if output path provided
    if output_path:
        with open(output_path, "w") as f:
            f.write(report_text)

    return report_text


@app.command()
def analyze(
    config_path: str = typer.Option(
        "img2latex/configs/config.yaml", help="Path to the configuration file"
    ),
    predictions_file: str = typer.Argument(
        ..., help="Path to the predictions JSON file"
    ),
    output_dir: str = typer.Option(
        "outputs/error_analysis", help="Directory to save the error analysis results"
    ),
    min_edit_distance: int = typer.Option(
        1, help="Minimum edit distance to consider as an error"
    ),
    max_samples: int = typer.Option(
        50, help="Maximum number of samples to include in each error category"
    ),
    truncate_sequences: bool = typer.Option(
        False, help="Truncate sequences to max_seq_length from config"
    ),
) -> None:
    """Analyze prediction errors and generate a comprehensive error report."""
    # Load configuration
    cfg = load_config(config_path)
    max_seq_length = cfg["data"].get("max_seq_length") if truncate_sequences else None

    # Ensure output directory exists
    output_path = ensure_output_dir(output_dir, "errors")

    console.print(
        f"[bold green]Loading predictions from {predictions_file}[/bold green]"
    )

    # Load predictions
    try:
        predictions = load_predictions(predictions_file)

        # Truncate sequences if needed
        if max_seq_length:
            console.print(
                f"[blue]Truncating sequences to {max_seq_length} tokens[/blue]"
            )
            for pred in predictions:
                if len(pred["reference"].split()) > max_seq_length:
                    pred["reference"] = " ".join(
                        pred["reference"].split()[:max_seq_length]
                    )
                if len(pred["hypothesis"].split()) > max_seq_length:
                    pred["hypothesis"] = " ".join(
                        pred["hypothesis"].split()[:max_seq_length]
                    )
                # Recalculate edit distance after truncation
                pred["edit_distance"] = distance(pred["reference"], pred["hypothesis"])
    except Exception as e:
        console.print(f"[bold red]Error loading predictions file: {e}[/bold red]")
        return

    console.print(f"[green]Loaded {len(predictions)} prediction samples[/green]")

    # Group records by edit distance
    console.print("[green]Grouping records by edit distance...[/green]")
    distance_ranges = [(0, 0), (1, 1), (2, 3), (4, float("inf"))]
    buckets = bucket_by_edit_distance(predictions, distance_ranges)

    # Print bucket statistics
    console.print("\n[bold]Edit distance buckets:[/bold]")
    for bucket_name, bucket_records in buckets.items():
        console.print(f"  {bucket_name}: {len(bucket_records)} examples")

    # Identify error patterns
    console.print("\n[green]Identifying error patterns...[/green]")
    error_patterns = identify_error_patterns(predictions)

    # Generate error report
    console.print("\n[green]Generating error report...[/green]")
    report_path = output_path / "error_analysis_report.md"
    report = generate_error_report(buckets, error_patterns, max_samples, report_path)

    # Save bucketed examples to JSON for further analysis
    bucketed_data = {}
    for bucket_name, bucket_records in buckets.items():
        bucketed_data[bucket_name] = [
            {
                "id": record.get("id", i),
                "reference": record["reference"],
                "hypothesis": record["hypothesis"],
                "edit_distance": record.get(
                    "edit_distance", record.get("levenshtein", 0)
                ),
                "bleu": record.get("bleu", 0),
            }
            for i, record in enumerate(
                bucket_records[:100]
            )  # Limit to 100 examples per bucket
        ]

    # Save report to JSON
    save_json_file(bucketed_data, output_path / "error_buckets.json")
    console.print(
        f"[green]Saved error buckets to {output_path / 'error_buckets.json'}[/green]"
    )

    # Print top error patterns
    if error_patterns:
        console.print("\n[bold]Top error patterns:[/bold]")
        for pattern in error_patterns[:5]:  # Show top 5
            console.print(
                f"  {pattern['pattern']}: {pattern['count']} occurrences - {pattern['description']}"
            )

    console.print(
        f"[bold green]Error analysis complete. Report saved to {report_path}[/bold green]"
    )


if __name__ == "__main__":
    app()

```

## File: analysis/project.py

- Extension: .py
- Language: python
- Size: 26126 bytes
- Created: 2025-04-16 17:11:34
- Modified: 2025-04-16 17:11:34

### Code

```python
#!/usr/bin/env python
"""
Script to analyze various aspects of the img2latex project setup,
including configuration, paths, data characteristics, and model parameters.

Features:
- Load and validate config.yaml
- Check for missing files
- Compare current config.yaml to last Git commit
- Summarize hyperparameter sweep
- Snapshot environment
- Check model config consistency
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd
import typer
import yaml
from rich.console import Console

from img2latex.analysis.utils import ensure_output_dir, save_json_file

# Suppress specific warnings if needed (e.g., from libraries)
# warnings.filterwarnings("ignore", category=SomeWarningCategory)

# Check if required project modules are available
if not importlib.util.find_spec("img2latex"):
    print("Error: Could not import project modules.", file=sys.stderr)
    print(
        "Please ensure this script is run from the project root directory",
        file=sys.stderr,
    )
    print("or that the 'img2latex' package is in the Python path.", file=sys.stderr)
    sys.exit(1)

# Import modules after availability check
from img2latex.utils.logging import get_logger

# Setup basic logging for the script itself
logger = get_logger(__name__.split(".")[-1], log_level="INFO")

# Create console for rich output
console = Console()

# Create Typer app
app = typer.Typer(help="Analyze img2latex project configuration and data.")


# --- Helper Functions ---


def load_config(config_path: Union[str, Path]) -> Dict:
    """Load and parse a YAML config file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary containing the parsed config

    Raises:
        FileNotFoundError: If the config file is not found
        yaml.YAMLError: If the config file cannot be parsed
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Successfully loaded config from: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Error: Config file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        raise


def validate_config(config: Dict) -> List[str]:
    """Validate the configuration to ensure required fields are present and valid.

    Args:
        config: Dictionary containing the configuration

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required top-level sections
    required_sections = ["data", "model", "training", "evaluation", "logging"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")

    # If missing sections, return early
    if errors:
        return errors

    # Check data section
    data_config = config.get("data", {})
    required_data_fields = [
        "data_dir",
        "train_file",
        "validate_file",
        "test_file",
        "formulas_file",
        "img_dir",
        "batch_size",
    ]
    for field in required_data_fields:
        if field not in data_config:
            errors.append(f"Missing required field in data section: {field}")

    # Check model section
    model_config = config.get("model", {})
    if "name" not in model_config:
        errors.append("Missing required field: model.name")
    else:
        model_name = model_config["name"]
        valid_models = ["cnn_lstm", "resnet_lstm"]
        if model_name not in valid_models:
            errors.append(
                f"Invalid model name: {model_name}. Must be one of {valid_models}"
            )

        # Check encoder and decoder settings based on model name
        encoder_config = model_config.get("encoder", {})
        if model_name == "cnn_lstm":
            if "cnn" not in encoder_config:
                errors.append(
                    "Missing required section: model.encoder.cnn for cnn_lstm model"
                )
            else:
                cnn_config = encoder_config.get("cnn", {})
                required_cnn_fields = [
                    "img_height",
                    "img_width",
                    "channels",
                    "conv_filters",
                ]
                for field in required_cnn_fields:
                    if field not in cnn_config:
                        errors.append(
                            f"Missing required field in model.encoder.cnn: {field}"
                        )
        elif model_name == "resnet_lstm":
            if "resnet" not in encoder_config:
                errors.append(
                    "Missing required section: model.encoder.resnet for resnet_lstm model"
                )
            else:
                resnet_config = encoder_config.get("resnet", {})
                required_resnet_fields = [
                    "img_height",
                    "img_width",
                    "channels",
                    "model_name",
                ]
                for field in required_resnet_fields:
                    if field not in resnet_config:
                        errors.append(
                            f"Missing required field in model.encoder.resnet: {field}"
                        )

        # Check decoder settings
        decoder_config = model_config.get("decoder", {})
        required_decoder_fields = ["hidden_dim", "lstm_layers", "dropout"]
        for field in required_decoder_fields:
            if field not in decoder_config:
                errors.append(f"Missing required field in model.decoder: {field}")

    # Check training section
    training_config = config.get("training", {})
    required_training_fields = ["optimizer", "learning_rate", "epochs", "device"]
    for field in required_training_fields:
        if field not in training_config:
            errors.append(f"Missing required field in training section: {field}")

    # Check evaluation section
    eval_config = config.get("evaluation", {})
    if "metrics" not in eval_config:
        errors.append("Missing required field: evaluation.metrics")

    return errors


def check_missing_files(
    config: Dict, base_dir: Union[str, Path]
) -> Dict[str, List[str]]:
    """Check for missing files referenced in the config.

    Args:
        config: Dictionary containing the configuration
        base_dir: Base directory for relative paths

    Returns:
        Dictionary with lists of missing files by category
    """
    base_dir = Path(base_dir)
    missing_files = {"data_files": [], "image_directory": []}

    # Check data files
    data_config = config.get("data", {})
    data_dir = Path(data_config.get("data_dir", ""))

    # If data_dir is not absolute, make it relative to base_dir
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    # Check train/val/test files
    for file_key in ["train_file", "validate_file", "test_file", "formulas_file"]:
        file_path = data_dir / data_config.get(file_key, "")
        if not file_path.exists():
            missing_files["data_files"].append(str(file_path))

    # Check image directory
    img_dir = data_dir / data_config.get("img_dir", "")
    if not img_dir.exists():
        missing_files["image_directory"].append(str(img_dir))

    return missing_files


def compare_config_with_git(config_path: Union[str, Path]) -> Dict[str, List[str]]:
    """Compare current config with the last committed version in Git.

    Args:
        config_path: Path to the current config file

    Returns:
        Dictionary with lists of added, modified, and deleted fields
    """
    # Initialize result dictionary
    changes = {"added": [], "modified": [], "deleted": []}

    try:
        # Get the last committed version from Git
        result = subprocess.run(
            ["git", "show", "HEAD:" + str(config_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        # If the command failed, the file might not be tracked in Git
        if result.returncode != 0:
            return {
                "error": f"Could not retrieve previous version: {result.stderr.strip()}"
            }

        # Parse the last committed version
        prev_config = yaml.safe_load(result.stdout)

        # Parse the current version
        with open(config_path, "r") as file:
            current_config = yaml.safe_load(file)

        # Compare configurations using a recursive function
        def compare_dicts(prev, curr, path=""):
            # Check for added or modified fields
            for key in curr:
                curr_path = f"{path}.{key}" if path else key

                if key not in prev:
                    changes["added"].append(curr_path)
                elif isinstance(curr[key], dict) and isinstance(prev[key], dict):
                    # Recursively compare nested dictionaries
                    compare_dicts(prev[key], curr[key], curr_path)
                elif curr[key] != prev[key]:
                    changes["modified"].append(
                        f"{curr_path}: {prev[key]} -> {curr[key]}"
                    )

            # Check for deleted fields
            for key in prev:
                curr_path = f"{path}.{key}" if path else key
                if key not in curr:
                    changes["deleted"].append(curr_path)

        # Perform comparison
        compare_dicts(prev_config, current_config)

        return changes

    except Exception as e:
        return {"error": f"Error comparing configs: {str(e)}"}


def summarize_hyperparameter_sweep(output_dir: Union[str, Path]) -> pd.DataFrame:
    """Read metrics from multiple output directories and summarize results.

    Args:
        output_dir: Base directory containing output subdirectories

    Returns:
        DataFrame summarizing hyperparameter sweep results
    """
    output_dir = Path(output_dir)

    # Find all subdirectories with metrics.json
    results = []

    for subdir in output_dir.glob("*"):
        if not subdir.is_dir():
            continue

        config_path = subdir / "config.yaml"
        metrics_path = subdir / "metrics.json"

        if not config_path.exists() or not metrics_path.exists():
            continue

        try:
            # Load config and metrics
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            # Extract hyperparameters of interest
            experiment_name = subdir.name
            model_name = config.get("model", {}).get("name", "unknown")
            batch_size = config.get("data", {}).get("batch_size", 0)
            learning_rate = config.get("training", {}).get("learning_rate", 0)
            optimizer = config.get("training", {}).get("optimizer", "unknown")

            if "embedding_dim" in config.get("model", {}):
                embedding_dim = config["model"]["embedding_dim"]
            else:
                embedding_dim = 0

            if "hidden_dim" in config.get("model", {}).get("decoder", {}):
                hidden_dim = config["model"]["decoder"]["hidden_dim"]
            else:
                hidden_dim = 0

            # Extract metrics of interest
            best_val_loss = metrics.get("best_val_loss", float("inf"))
            best_val_bleu = metrics.get("best_val_bleu", 0)
            best_val_accuracy = metrics.get("best_val_accuracy", 0)
            best_epoch = metrics.get("best_epoch", 0)

            # Combine into a single result
            result = {
                "experiment_name": experiment_name,
                "model_name": model_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "best_val_loss": best_val_loss,
                "best_val_bleu": best_val_bleu,
                "best_val_accuracy": best_val_accuracy,
                "best_epoch": best_epoch,
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing {subdir}: {e}")

    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        return df
    else:
        return pd.DataFrame(
            columns=[
                "experiment_name",
                "model_name",
                "batch_size",
                "learning_rate",
                "optimizer",
                "embedding_dim",
                "hidden_dim",
                "best_val_loss",
                "best_val_bleu",
                "best_val_accuracy",
                "best_epoch",
            ]
        )


def snapshot_environment() -> str:
    """Capture the current Python environment using pip freeze.

    Returns:
        String containing the output of pip freeze
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error capturing environment: {e}")
        return f"Error: {e.stderr}"


def check_model_consistency(config: Dict) -> List[str]:
    """Check model configuration for consistency.

    Args:
        config: Dictionary containing the configuration

    Returns:
        List of consistency warnings
    """
    warnings = []

    # Get model configuration
    model_config = config.get("model", {})
    model_name = model_config.get("name")

    if model_name == "cnn_lstm":
        # Check CNN encoder configuration
        cnn_config = model_config.get("encoder", {}).get("cnn", {})
        if cnn_config:
            # Check channel consistency
            channels = cnn_config.get("channels")
            if channels != 1:
                warnings.append(
                    f"CNN model typically uses grayscale images (channels=1), but config has channels={channels}"
                )

            # Check kernel size consistency
            kernel_size = cnn_config.get("kernel_size")
            if kernel_size and (kernel_size < 3 or kernel_size > 5):
                warnings.append(f"Unusual kernel size for CNN model: {kernel_size}")

    elif model_name == "resnet_lstm":
        # Check ResNet encoder configuration
        resnet_config = model_config.get("encoder", {}).get("resnet", {})
        if resnet_config:
            # Check channel consistency
            channels = resnet_config.get("channels")
            if channels != 3:
                warnings.append(
                    f"ResNet model typically uses RGB images (channels=3), but config has channels={channels}"
                )

            # Check model name
            resnet_model = resnet_config.get("model_name")
            valid_models = [
                "resnet18",
                "resnet34",
                "resnet50",
                "resnet101",
                "resnet152",
            ]
            if resnet_model not in valid_models:
                warnings.append(
                    f"Invalid ResNet model name: {resnet_model}. Should be one of {valid_models}"
                )

    # Check decoder configuration
    decoder_config = model_config.get("decoder", {})
    if decoder_config:
        # Check embedding dimension and hidden dimension consistency
        embedding_dim = model_config.get("embedding_dim")
        hidden_dim = decoder_config.get("hidden_dim")

        if embedding_dim and hidden_dim and embedding_dim != hidden_dim:
            warnings.append(
                f"Embedding dimension ({embedding_dim}) != hidden dimension ({hidden_dim}). This is unusual."
            )

        # Check dropout value
        dropout = decoder_config.get("dropout")
        if dropout is not None:
            if dropout <= 0 or dropout >= 0.5:
                warnings.append(
                    f"Unusual dropout value: {dropout}. Typical values are between 0.1 and 0.3."
                )

    # Check training configuration
    training_config = config.get("training", {})
    if training_config:
        # Check learning rate
        lr = training_config.get("learning_rate")
        if lr and (lr > 0.1 or lr < 1e-5):
            warnings.append(
                f"Unusual learning rate: {lr}. Typical values are between 1e-5 and 1e-2."
            )

        # Check optimizer and weight decay consistency
        optimizer = training_config.get("optimizer")
        weight_decay = training_config.get("weight_decay")

        if optimizer == "adam" and weight_decay and weight_decay > 0.01:
            warnings.append(
                f"High weight decay ({weight_decay}) for Adam optimizer. Consider reducing."
            )

        # Check device
        device = training_config.get("device")
        if device not in ["cuda", "cpu", "mps"]:
            warnings.append(
                f"Unknown device: {device}. Should be 'cuda', 'cpu', or 'mps'."
            )

    return warnings


def plot_hyperparameter_comparison(
    sweep_data: pd.DataFrame, output_dir: Path, metric: str = "best_val_bleu"
) -> None:
    """Create bar charts comparing hyperparameter performance.

    Args:
        sweep_data: DataFrame containing hyperparameter sweep results
        output_dir: Directory to save the output figures
        metric: Metric to plot (e.g., 'best_val_bleu', 'best_val_loss')
    """
    if sweep_data.empty:
        print("No data to plot")
        return

    # Create plots for different hyperparameters
    for param in [
        "model_name",
        "batch_size",
        "learning_rate",
        "optimizer",
        "embedding_dim",
        "hidden_dim",
    ]:
        if param in sweep_data.columns:
            # Create figure
            plt.figure(figsize=(10, 6))

            # Group by the parameter and get mean of the metric
            grouped = sweep_data.groupby(param)[metric].mean().reset_index()

            # Sort for better visualization
            grouped = grouped.sort_values(metric, ascending=False)

            # Create bar chart
            plt.bar(grouped[param].astype(str), grouped[metric])

            # Add labels and title
            plt.xlabel(param)
            plt.ylabel(metric)
            plt.title(f"Effect of {param} on {metric}")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            # Save figure
            plt.tight_layout()
            plt.savefig(output_dir / f"{param}_comparison.png")
            plt.close()


@app.command()
def analyze(
    config_path: str = typer.Option(
        "img2latex/configs/config.yaml", help="Path to the configuration file"
    ),
    base_dir: str = typer.Option(".", help="Base directory for the project"),
    output_dir: str = typer.Option(
        "outputs/project_analysis", help="Directory to save analysis results"
    ),
    detailed: bool = typer.Option(
        False, help="Perform detailed analysis (Git comparison, hyperparameter sweep)"
    ),
) -> None:
    """Analyze project configuration, paths, and code structure."""
    # Ensure output directory exists
    output_path = ensure_output_dir(output_dir, "project")

    # Record analysis results
    results = {}

    # Load and validate configuration
    print(f"Loading configuration from {config_path}...")
    try:
        config = load_config(config_path)
        results["config"] = {"status": "loaded", "file_path": str(config_path)}

        # Validate configuration
        validation_errors = validate_config(config)
        if validation_errors:
            results["validation"] = {"status": "failed", "errors": validation_errors}
            print(
                f"Configuration validation failed with {len(validation_errors)} errors"
            )
        else:
            results["validation"] = {"status": "passed"}
            print("Configuration validation passed")

    except Exception as e:
        results["config"] = {"status": "failed", "error": str(e)}
        print(f"Failed to load configuration: {e}")

        # Save results and exit early
        save_json_file(results, output_path / "project_analysis.json")
        return

    # Check for missing files
    try:
        missing_files = check_missing_files(config, base_dir)
        has_missing = any(len(files) > 0 for files in missing_files.values())

        results["missing_files"] = {
            "status": "found" if has_missing else "none",
            "files": missing_files,
        }

        if has_missing:
            print("Missing files found:")
            for category, files in missing_files.items():
                if files:
                    print(f"  {category}:")
                    for file in files:
                        print(f"    - {file}")
        else:
            print("No missing files found")

    except Exception as e:
        results["missing_files"] = {"status": "error", "error": str(e)}
        print(f"Error checking for missing files: {e}")

    # Compare config with Git version
    if detailed:
        try:
            config_changes = compare_config_with_git(config_path)

            if "error" in config_changes:
                results["git_comparison"] = {
                    "status": "error",
                    "error": config_changes["error"],
                }
                print(f"Error comparing with Git: {config_changes['error']}")
            else:
                has_changes = any(
                    len(changes) > 0 for changes in config_changes.values()
                )

                results["git_comparison"] = {
                    "status": "changed" if has_changes else "unchanged",
                    "changes": config_changes,
                }

                if has_changes:
                    print("Configuration changes since last commit:")
                    for change_type, changes in config_changes.items():
                        if changes:
                            print(f"  {change_type.capitalize()}:")
                            for change in changes:
                                print(f"    - {change}")
                else:
                    print("No configuration changes since last commit")

        except Exception as e:
            results["git_comparison"] = {"status": "error", "error": str(e)}
            print(f"Error comparing with Git: {e}")

    # Summarize hyperparameter sweep
    if detailed:
        print("Analyzing hyperparameter sweep results...")
        try:
            # Default location for output directories
            outputs_dir = Path(base_dir) / "outputs"

            # Summarize results
            sweep_results = summarize_hyperparameter_sweep(outputs_dir)

            if not sweep_results.empty:
                # Save to CSV
                sweep_csv_path = output_path / "hyperparameter_sweep.csv"
                sweep_results.to_csv(sweep_csv_path, index=False)

                # Create plots
                plot_hyperparameter_comparison(
                    sweep_results, output_path, "best_val_bleu"
                )
                plot_hyperparameter_comparison(
                    sweep_results, output_path, "best_val_loss"
                )

                results["hyperparameter_sweep"] = {
                    "status": "analyzed",
                    "num_experiments": len(sweep_results),
                    "summary_path": str(sweep_csv_path),
                }

                print(
                    f"Hyperparameter sweep analyzed ({len(sweep_results)} experiments)"
                )
                print(f"Summary saved to {sweep_csv_path}")
            else:
                results["hyperparameter_sweep"] = {
                    "status": "empty",
                    "message": "No experiment results found",
                }
                print("No hyperparameter sweep results found")

        except Exception as e:
            results["hyperparameter_sweep"] = {"status": "error", "error": str(e)}
            print(f"Error analyzing hyperparameter sweep: {e}")

    # Snapshot environment
    print("Capturing environment snapshot...")
    try:
        env_snapshot = snapshot_environment()
        env_path = output_path / "environment_snapshot.txt"

        with open(env_path, "w") as f:
            f.write(env_snapshot)

        results["environment"] = {"status": "captured", "snapshot_path": str(env_path)}

        print(f"Environment snapshot saved to {env_path}")

    except Exception as e:
        results["environment"] = {"status": "error", "error": str(e)}
        print(f"Error capturing environment: {e}")

    # Check model consistency
    if "config" in results and results["config"]["status"] == "loaded":
        try:
            consistency_warnings = check_model_consistency(config)

            results["model_consistency"] = {
                "status": "warnings" if consistency_warnings else "consistent",
                "warnings": consistency_warnings,
            }

            if consistency_warnings:
                print("Model consistency warnings:")
                for warning in consistency_warnings:
                    print(f"  - {warning}")
            else:
                print("Model configuration is consistent")

        except Exception as e:
            results["model_consistency"] = {"status": "error", "error": str(e)}
            print(f"Error checking model consistency: {e}")

    # Record analysis timestamp
    from datetime import datetime

    results["timestamp"] = datetime.now().isoformat()

    # Save results
    save_json_file(results, output_path / "project_analysis.json")

    print(f"Project analysis complete. Results saved to {output_path}")


if __name__ == "__main__":
    app()

```

## File: training/predictor.py

- Extension: .py
- Language: python
- Size: 19082 bytes
- Created: 2025-04-16 22:24:51
- Modified: 2025-04-16 22:24:51

### Code

```python
"""
Prediction logic for the image-to-LaTeX model.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.data.utils import load_image
from img2latex.model.seq2seq import Seq2SeqModel
from img2latex.utils.logging import get_logger
from img2latex.utils.mps_utils import set_device

logger = get_logger(__name__)


class Predictor:
    """
    Predictor for the image-to-LaTeX model.

    This class handles inference with the trained model.
    """

    def __init__(
        self,
        model: Seq2SeqModel,
        tokenizer: LaTeXTokenizer,
        device: Optional[torch.device] = None,
        model_type: str = "cnn_lstm",
    ):
        """
        Initialize the predictor.

        Args:
            model: Trained model
            tokenizer: Tokenizer for LaTeX formulas
            device: Device to use for inference
            model_type: Type of model ("cnn_lstm" or "resnet_lstm")
        """
        # Set device
        if device is None:
            self.device = set_device()
        else:
            self.device = device

        # Set model and tokenizer
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model_type = model_type

        # Set model to evaluation mode
        self.model.eval()

        logger.info(
            f"Initialized predictor for model type {model_type} on device {self.device}"
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, device: Optional[torch.device] = None
    ) -> "Predictor":
        """
        Create a predictor from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint
            device: Device to use for inference

        Returns:
            Initialized predictor
        """
        # Set device
        if device is None:
            device = set_device()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract configuration
        config = checkpoint.get("config", {})
        model_config = config.get("model", {})
        model_type = model_config.get("name", "cnn_lstm")

        # Create tokenizer
        tokenizer_config = checkpoint.get("tokenizer_config", {})
        tokenizer = LaTeXTokenizer(
            special_tokens=tokenizer_config.get("special_tokens"),
            max_sequence_length=tokenizer_config.get("max_sequence_length", 141),
        )
        tokenizer.token_to_id = tokenizer_config.get("token_to_id", {})
        tokenizer.id_to_token = {
            idx: token for token, idx in tokenizer.token_to_id.items()
        }
        tokenizer.vocab_size = len(tokenizer.token_to_id)

        # Set special token IDs
        tokenizer.pad_token_id = tokenizer.token_to_id[tokenizer.special_tokens["PAD"]]
        tokenizer.start_token_id = tokenizer.token_to_id[
            tokenizer.special_tokens["START"]
        ]
        tokenizer.end_token_id = tokenizer.token_to_id[tokenizer.special_tokens["END"]]
        tokenizer.unk_token_id = tokenizer.token_to_id[tokenizer.special_tokens["UNK"]]

        # Create model
        encoder_params = model_config.get("encoder", {})
        if model_type == "cnn_lstm":
            encoder_params = encoder_params.get("cnn", {})
        else:
            encoder_params = encoder_params.get("resnet", {})

        decoder_params = model_config.get("decoder", {})

        # Set embedding dimension
        embedding_dim = model_config.get("embedding_dim", 256)
        encoder_params["embedding_dim"] = embedding_dim

        # Create model
        model = Seq2SeqModel(
            model_type=model_type,
            vocab_size=tokenizer.vocab_size,
            encoder_params=encoder_params,
            decoder_params=decoder_params,
        )

        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Create predictor
        predictor = cls(
            model=model, tokenizer=tokenizer, device=device, model_type=model_type
        )

        logger.info(f"Created predictor from checkpoint {checkpoint_path}")
        return predictor

    def predict(
        self,
        image: Union[str, torch.Tensor, np.ndarray, Image.Image],
        beam_size: int = 0,
        max_length: int = 141,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> str:
        """
        Predict LaTeX formula for an image.

        Args:
            image: Image to predict (path, tensor, array, or PIL image)
            beam_size: Beam size for beam search (0 for greedy search)
            max_length: Maximum length of the generated sequence
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Predicted LaTeX formula
        """
        # Prepare image tensor
        img_tensor = self._prepare_image(image)

        # Move to device
        img_tensor = img_tensor.to(self.device)

        # Get special token IDs
        start_token_id = self.tokenizer.start_token_id
        end_token_id = self.tokenizer.end_token_id

        # Generate sequence
        with torch.no_grad():
            sequence = self.model.inference(
                image=img_tensor,
                start_token_id=start_token_id,
                end_token_id=end_token_id,
                max_length=max_length,
                beam_size=beam_size,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        # Convert sequence to LaTeX
        # Skip start token if present at the beginning
        if sequence[0] == start_token_id:
            sequence = sequence[1:]
        # Remove end token if present at the end
        if sequence and sequence[-1] == end_token_id:
            sequence = sequence[:-1]

        # Decode sequence
        latex = self.tokenizer.decode(sequence)

        return latex

    def predict_batch(
        self,
        images: List[Union[str, torch.Tensor, np.ndarray, Image.Image]],
        beam_size: int = 0,
        max_length: int = 141,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        batch_size: int = 16,
    ) -> List[str]:
        """
        Predict LaTeX formulas for a batch of images.

        Args:
            images: List of images to predict
            beam_size: Beam size for beam search (0 for greedy search)
            max_length: Maximum length of the generated sequence
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            batch_size: Batch size for prediction

        Returns:
            List of predicted LaTeX formulas
        """
        results = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]

            # Prepare image tensors
            img_tensors = [self._prepare_image(img) for img in batch_images]
            batch_tensor = torch.stack(img_tensors).to(self.device)

            # Get special token IDs
            start_token_id = self.tokenizer.start_token_id
            end_token_id = self.tokenizer.end_token_id

            # For greedy search (non-beam search), we can process the whole batch at once
            if beam_size == 0:
                # --- Start of Restored Batched Greedy Search Logic ---
                with torch.no_grad():
                    # Encode the entire batch
                    if batch_tensor.ndim == 5 and batch_tensor.shape[1] == 1:
                        squeezed_batch_tensor = batch_tensor.squeeze(1)
                    else:
                        squeezed_batch_tensor = batch_tensor
                    encoder_outputs = self.model.encoder(
                        squeezed_batch_tensor
                    )  # [B, EmbDim] or [B, SeqLen, EmbDim]

                    # Initialize batch sequences with start tokens
                    current_batch_size = squeezed_batch_tensor.size(0)
                    device = squeezed_batch_tensor.device
                    # Keep track of sequences for each item in the batch
                    sequences = torch.full(
                        (current_batch_size, 1),
                        start_token_id,
                        dtype=torch.long,
                        device=device,
                    )  # [B, 1]
                    # Flag for sequences that have finished (found <END>)
                    finished = torch.zeros(
                        current_batch_size, dtype=torch.bool, device=device
                    )

                    # Initialize hidden state for the whole batch
                    hidden = None

                    # Generate tokens step-by-step for the batch
                    for _ in range(max_length):
                        # Get the last token for each sequence in the batch
                        input_tokens = sequences[:, -1].unsqueeze(1)  # [B, 1]

                        # Decode one step for the whole batch
                        # Pass appropriate part of encoder_outputs (might depend on attention)
                        output, hidden = self.model.decoder.decode_step(
                            encoder_outputs, input_tokens, hidden
                        )
                        logits = output.squeeze(1)  # [B, VocabSize]

                        # Apply temperature, top-k, top-p sampling
                        if temperature != 1.0:
                            logits = logits / temperature
                        probs = torch.softmax(logits, dim=-1)

                        if top_k > 0:
                            top_k = min(top_k, probs.size(-1))
                            kth_prob, _ = torch.topk(probs, top_k, dim=-1)
                            kth_prob = kth_prob[
                                :, -1, None
                            ]  # Use [:, -1, None] for batch
                            indices_to_remove = probs < kth_prob
                            probs[indices_to_remove] = 0.0
                            probs_sum = probs.sum(dim=-1, keepdim=True)
                            if torch.any(probs_sum > 0):
                                probs = probs / probs_sum

                        if top_p > 0.0:
                            sorted_probs, sorted_indices = torch.sort(
                                probs, descending=True
                            )
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                                :, :-1
                            ].clone()
                            sorted_indices_to_remove[:, 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(
                                -1, sorted_indices, sorted_indices_to_remove
                            )
                            probs[indices_to_remove] = 0.0
                            probs_sum = probs.sum(dim=-1, keepdim=True)
                            if torch.any(probs_sum > 0):
                                probs = probs / probs_sum

                        # Sample or take argmax
                        if temperature > 0 and (top_k > 0 or top_p > 0.0):
                            next_tokens = torch.multinomial(probs, 1)  # [B, 1]
                        else:
                            next_tokens = torch.argmax(
                                probs, dim=-1, keepdim=True
                            )  # [B, 1]

                        # Update sequences only for those not finished
                        sequences = torch.cat(
                            [sequences, next_tokens], dim=1
                        )  # [B, SeqLen+1]

                        # Update finished flags
                        finished = finished | (next_tokens.squeeze(1) == end_token_id)

                        # Break if all sequences are finished
                        if torch.all(finished):
                            break

                # Convert final tensor sequences to lists of lists
                final_sequences = []
                for seq_tensor in sequences:
                    seq_list = seq_tensor.cpu().tolist()
                    # Trim sequence after first <END> token
                    try:
                        end_idx = seq_list.index(end_token_id)
                        final_sequences.append(seq_list[:end_idx])  # Exclude <END>
                    except ValueError:
                        final_sequences.append(seq_list)  # No <END> found
                # Assign to the variable name expected by the later code
                sequences = final_sequences
                # --- End of Restored Batched Greedy Search Logic ---
            else:
                # For beam search, process each image individually
                with torch.no_grad():
                    sequences = []
                    for j in range(batch_tensor.size(0)):
                        img_tensor = batch_tensor[j : j + 1]  # Keep batch dimension

                        sequence = self.model.inference(
                            image=img_tensor,
                            start_token_id=start_token_id,
                            end_token_id=end_token_id,
                            max_length=max_length,
                            beam_size=beam_size,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                        )
                        sequences.append(sequence)

            # Convert sequences to LaTeX
            for sequence in sequences:
                # Skip start token if present at the beginning
                if sequence[0] == start_token_id:
                    sequence = sequence[1:]
                # Remove end token if present at the end
                if sequence and sequence[-1] == end_token_id:
                    sequence = sequence[:-1]

                # Decode sequence
                latex = self.tokenizer.decode(sequence)
                results.append(latex)

        return results

    def _prepare_image(
        self, image: Union[str, torch.Tensor, np.ndarray, Image.Image]
    ) -> torch.Tensor:
        """
        Prepare an image for prediction.

        Args:
            image: Image to prepare (path, tensor, array, or PIL image)

        Returns:
            Preprocessed image tensor
        """
        # Determine image size based on model type
        if self.model_type == "cnn_lstm":
            img_size = (64, 800)
            channels = 1
        else:  # resnet_lstm
            img_size = (64, 800)
            channels = 3

        # Handle different input types
        if isinstance(image, str):
            # Load image from path using the utility function
            img_tensor = load_image(image, img_size, channels)
        elif isinstance(image, torch.Tensor):
            # For tensor input, process it appropriately
            img_tensor = self._preprocess_tensor(image, img_size, channels)
        elif isinstance(image, np.ndarray):
            # Convert numpy array to tensor and then process
            tensor_image = self._numpy_to_tensor(image, channels)
            img_tensor = self._preprocess_tensor(tensor_image, img_size, channels)
        elif isinstance(image, Image.Image):
            # Convert PIL image to tensor using the utility function logic
            # First convert and resize
            if channels == 1 and image.mode != "L":
                image = image.convert("L")
            elif channels == 3 and image.mode != "RGB":
                image = image.convert("RGB")

            # Resize the image
            image = image.resize(img_size[::-1])  # PIL uses (width, height)

            # Convert to tensor using NumPy as intermediate
            img_array = np.array(image)
            if channels == 1:
                img_array = np.expand_dims(img_array, axis=0)
            else:
                img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW

            img_tensor = torch.from_numpy(img_array).float() / 255.0
            # Normalize to [-1, 1]
            img_tensor = img_tensor * 2.0 - 1.0
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. "
                "Expected str, torch.Tensor, numpy.ndarray, or PIL.Image.Image."
            )

        # Convert grayscale to RGB for ResNet if needed
        if self.model_type == "resnet_lstm" and img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)

        # Add batch dimension if not present
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def _preprocess_tensor(
        self, tensor: torch.Tensor, img_size: Tuple[int, int], channels: int
    ) -> torch.Tensor:
        """
        Preprocess a tensor image.

        Args:
            tensor: Image tensor
            img_size: Target image size (height, width)
            channels: Number of channels (1 for grayscale, 3 for RGB)

        Returns:
            Preprocessed tensor
        """
        # Add channel dimension for grayscale if needed
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        # Resize if needed
        if tensor.shape[-2:] != img_size:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0) if tensor.dim() == 3 else tensor,
                size=img_size,
                mode="bilinear",
                align_corners=False,
            )
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)

        # Normalize if needed
        if tensor.min() < 0 or tensor.max() > 1:
            tensor = tensor / 255.0
            # Normalize to [-1, 1]
            tensor = tensor * 2.0 - 1.0

        return tensor

    def _numpy_to_tensor(self, array: np.ndarray, channels: int) -> torch.Tensor:
        """
        Convert a numpy array to a tensor with proper channel format.

        Args:
            array: Numpy array
            channels: Number of channels (1 for grayscale, 3 for RGB)

        Returns:
            Tensor with proper channel format
        """
        # Handle different array formats
        if array.ndim == 2:
            # Grayscale image without channel dimension
            array = np.expand_dims(array, axis=0)
        elif array.ndim == 3 and array.shape[0] not in [1, 3]:
            # HWC format, convert to CHW
            array = np.transpose(array, (2, 0, 1))

        # Convert to torch tensor
        return torch.from_numpy(array).float()

```

## File: training/metrics.py

- Extension: .py
- Language: python
- Size: 20635 bytes
- Created: 2025-04-16 21:23:26
- Modified: 2025-04-16 21:23:26

### Code

```python
# Path: img2latex/training/metrics.py
"""
Evaluation metrics for the image-to-LaTeX model.
"""

import collections
import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import entropy

from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


def _detach_to_cpu(
    tensor_or_list: Union[torch.Tensor, List, Any],
) -> Union[torch.Tensor, List, Any]:
    """
    Move a tensor to CPU or process a list of tensors.

    Args:
        tensor_or_list: Either a tensor, a list (possibly containing tensors), or other data types

    Returns:
        The same tensor or list with all tensors detached and moved to CPU
    """
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list.detach().cpu()
    elif isinstance(tensor_or_list, list):
        return [
            t.detach().cpu().tolist()
            if isinstance(t, torch.Tensor)
            else _detach_to_cpu(t)
            if isinstance(t, list)
            else t
            for t in tensor_or_list
        ]
    return tensor_or_list


def levenshtein_distance(sequence_one: List[int], sequence_two: List[int]) -> float:
    """
    Calculate the Levenshtein distance between two sequences.

    Args:
        sequence_one: First sequence of tokens
        sequence_two: Second sequence of tokens

    Returns:
        Normalized Levenshtein similarity (1 - distance/max_length)
    """
    rows = len(sequence_one)
    cols = len(sequence_two)

    # Create a distance table
    dist_tab = np.zeros((rows + 1, cols + 1), dtype=int)

    # Initialize first row and column
    for i in range(1, rows + 1):
        dist_tab[i][0] = i
    for i in range(1, cols + 1):
        dist_tab[0][i] = i

    # Fill the table
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            # If tokens match
            if sequence_one[r - 1] == sequence_two[c - 1]:
                # Same cost as min cost from prev tokens
                dist_tab[r][c] = dist_tab[r - 1][c - 1]
            else:
                # Min of deletion, insertion, or substitution respectively
                dist_tab[r][c] = 1 + min(
                    dist_tab[r - 1][c], dist_tab[r][c - 1], dist_tab[r - 1][c - 1]
                )

    # Return normalized similarity (1 - distance/max_length)
    # Higher values indicate more similar sequences
    raw_distance = dist_tab[rows][cols]
    max_length = max(rows, cols)

    # Avoid division by zero
    if max_length == 0:
        return 1.0

    return 1.0 - (raw_distance / max_length)


def bleu_n_score(
    generated_sequence: List[int], true_sequence: List[int], n: int = 4
) -> float:
    """
    Calculate the BLEU-n score for a generated sequence.

    Args:
        generated_sequence: Generated sequence of tokens
        true_sequence: Reference sequence of tokens
        n: Maximum n-gram size to consider

    Returns:
        BLEU-n score
    """
    gen_len = len(generated_sequence)
    true_len = len(true_sequence)

    # Handle empty sequences
    if gen_len == 0 or true_len == 0:
        return 0.0

    # Scores for each n-gram size
    scores = []

    # Calculate and store precision for 1-grams to n-grams
    for gram_size in range(1, n + 1):
        # Skip if sequences are too short for this n-gram size
        if gen_len < gram_size or true_len < gram_size:
            scores.append(0.0)
            continue

        # Calculate n-grams
        gen_ngrams = [
            tuple(generated_sequence[i : i + gram_size])
            for i in range(gen_len - gram_size + 1)
        ]
        true_ngrams = [
            tuple(true_sequence[i : i + gram_size])
            for i in range(true_len - gram_size + 1)
        ]

        # Count n-gram frequencies
        gen_grams_count = collections.Counter(gen_ngrams)
        true_grams_count = collections.Counter(true_ngrams)

        # Sum of how many n-grams appear in both sequences
        matching_grams_sum = sum(
            min(gen_grams_count[gram], true_grams_count[gram])
            for gram in gen_grams_count
        )

        # Precision = matching n-grams / total generated n-grams
        gram_score = 0.0
        if len(gen_grams_count) > 0:
            gram_score = matching_grams_sum / len(gen_ngrams)

        scores.append(gram_score)

    # Calculate geometric mean of scores
    # Skip if any score is zero (would make the geometric mean zero)
    for gram_score in scores:
        if gram_score == 0.0:
            return 0.0

    # Log-space calculation to avoid numerical issues
    geo_mean = 0.0
    for gram_score in scores:
        geo_mean += math.log(gram_score)
    geo_mean = math.exp(geo_mean / n)

    # Apply brevity penalty if generated sequence is shorter than reference
    if gen_len < true_len:
        # BP = exp(1 - true_len/gen_len)
        # Using try/except to handle potential division by zero
        try:
            brevity_penalty = math.exp(1.0 - true_len / gen_len)
            return brevity_penalty * geo_mean
        except ZeroDivisionError:
            return 0.0

    return geo_mean


def calculate_metrics(
    predictions: List[List[int]], targets: List[List[int]]
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for a batch of predictions.

    Args:
        predictions: List of predicted token sequences
        targets: List of target token sequences

    Returns:
        Dictionary of metrics
    """
    # Ensure predictions and targets are properly detached and on CPU
    predictions = _detach_to_cpu(predictions)
    targets = _detach_to_cpu(targets)

    # Check that predictions and targets have the same number of samples
    assert len(predictions) == len(targets), (
        "Predictions and targets must have the same length"
    )

    num_sequences = len(predictions)

    # Calculate BLEU score
    bleu_scores = [
        bleu_n_score(predictions[i], targets[i], 4) for i in range(num_sequences)
    ]
    mean_bleu = sum(bleu_scores) / num_sequences

    # Calculate Levenshtein similarity
    lev_similarities = [
        levenshtein_distance(predictions[i], targets[i]) for i in range(num_sequences)
    ]
    mean_lev = sum(lev_similarities) / num_sequences

    # Add batch size for logging
    result = {"bleu": mean_bleu, "levenshtein": mean_lev, "batch_size": num_sequences}

    return result


def masked_accuracy(
    predictions: torch.Tensor, targets: torch.Tensor, pad_token_id: int
) -> Tuple[float, int]:
    """
    Calculate masked accuracy for a batch.

    This function ignores padded tokens when calculating accuracy.

    Args:
        predictions: Prediction tensor of shape (batch_size, seq_length, vocab_size)
        targets: Target tensor of shape (batch_size, seq_length)
        pad_token_id: ID of the padding token to ignore

    Returns:
        Tuple of (accuracy, number of tokens)
    """
    # Ensure predictions and targets are properly detached and on CPU
    predictions = _detach_to_cpu(predictions)
    targets = _detach_to_cpu(targets)

    # Get the predicted tokens
    pred_tokens = torch.argmax(predictions, dim=-1)

    # Create a mask for non-padding tokens
    mask = targets != pad_token_id

    # Calculate accuracy only for non-padding tokens
    correct = torch.sum((pred_tokens == targets) * mask).item()
    total = torch.sum(mask).item()

    # Avoid division by zero
    if total == 0:
        return 0.0, 0

    accuracy = correct / total
    return accuracy, total


def analyze_token_distribution(
    predictions: List[List[int]],
    targets: List[List[int]],
    tokenizer: LaTeXTokenizer,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Analyze the distribution of tokens in predictions and targets.

    Args:
        predictions: List of predicted token sequences
        targets: List of target token sequences
        tokenizer: LaTeX tokenizer
        top_k: Number of top tokens to include

    Returns:
        Dictionary with token distribution analysis
    """
    # Ensure predictions and targets are properly detached and on CPU
    predictions = _detach_to_cpu(predictions)
    targets = _detach_to_cpu(targets)

    # Flatten token lists
    pred_tokens_flat = [token for seq in predictions for token in seq]
    target_tokens_flat = [token for seq in targets for token in seq]

    # Count token frequencies
    pred_counter = Counter(pred_tokens_flat)
    target_counter = Counter(target_tokens_flat)

    # Get top-k most common tokens
    pred_most_common = pred_counter.most_common(top_k)
    target_most_common = target_counter.most_common(top_k)

    # Calculate entropy of distributions
    pred_probs = np.array(
        [count / len(pred_tokens_flat) for _, count in pred_counter.items()]
    )
    target_probs = np.array(
        [count / len(target_tokens_flat) for _, count in target_counter.items()]
    )

    pred_entropy = entropy(pred_probs) if len(pred_probs) > 0 else 0
    target_entropy = entropy(target_probs) if len(target_probs) > 0 else 0

    # Convert token IDs to readable tokens
    pred_most_common_readable = [
        (tokenizer.id_to_token.get(token_id, "<UNK>"), count)
        for token_id, count in pred_most_common
    ]
    target_most_common_readable = [
        (tokenizer.id_to_token.get(token_id, "<UNK>"), count)
        for token_id, count in target_most_common
    ]

    # Check for repetition issues
    repetition_factor = (
        pred_most_common[0][1] / len(pred_tokens_flat) if pred_most_common else 0
    )

    # Calculate diversity ratio (unique tokens / total tokens)
    pred_diversity = (
        len(pred_counter) / len(pred_tokens_flat) if pred_tokens_flat else 0
    )
    target_diversity = (
        len(target_counter) / len(target_tokens_flat) if target_tokens_flat else 0
    )

    return {
        "predictions": {
            "top_tokens": pred_most_common_readable,
            "entropy": pred_entropy,
            "diversity": pred_diversity,
            "repetition_factor": repetition_factor,
        },
        "targets": {
            "top_tokens": target_most_common_readable,
            "entropy": target_entropy,
            "diversity": target_diversity,
        },
    }


def sample_predictions_and_targets(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    tokenizer: LaTeXTokenizer,
    num_samples: int = 2,
    confidence_threshold: float = 0.5,
) -> Dict[str, List]:
    """
    Sample and decode predictions and targets for visualization.

    Args:
        outputs: Model outputs tensor (batch_size, seq_length, vocab_size)
        targets: Target tensor (batch_size, seq_length)
        tokenizer: LaTeX tokenizer
        num_samples: Number of samples to include
        confidence_threshold: Threshold for highlighting low confidence predictions

    Returns:
        Dictionary with sampled predictions and targets
    """
    # Ensure outputs and targets are properly detached and on CPU
    outputs_cpu = _detach_to_cpu(outputs)
    targets_cpu = _detach_to_cpu(targets)

    batch_size, seq_length, vocab_size = outputs_cpu.shape

    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(outputs_cpu, dim=-1)

    # Get predicted tokens and their probabilities
    pred_tokens = torch.argmax(probs, dim=-1)
    pred_probs = torch.max(probs, dim=-1)[0]

    # Convert to numpy for easier handling
    pred_tokens_np = pred_tokens.numpy()
    pred_probs_np = pred_probs.numpy()
    targets_np = targets_cpu.numpy()

    # Clean up intermediate tensors
    del outputs_cpu, targets_cpu, probs, pred_tokens, pred_probs

    samples = []

    # Sample min(batch_size, num_samples) examples
    for i in range(min(batch_size, num_samples)):
        # Get masks for non-padding tokens
        pred_mask = pred_tokens_np[i] != tokenizer.pad_token_id
        target_mask = targets_np[i] != tokenizer.pad_token_id

        # Get non-padding tokens
        pred_seq = pred_tokens_np[i][pred_mask]
        target_seq = targets_np[i][target_mask]

        # Get corresponding probabilities
        pred_confidences = pred_probs_np[i][pred_mask]

        # Decode to readable LaTeX
        pred_latex = tokenizer.decode(pred_seq.tolist())
        target_latex = tokenizer.decode(target_seq.tolist())

        # Find low-confidence tokens
        low_confidence_indices = np.where(pred_confidences < confidence_threshold)[0]
        low_confidence_tokens = []

        for idx in low_confidence_indices:
            if idx < len(pred_seq):
                token_id = pred_seq[idx]
                token = tokenizer.id_to_token.get(token_id, "<UNK>")
                confidence = pred_confidences[idx]
                low_confidence_tokens.append((token, float(confidence)))

        # Add sample to list
        samples.append(
            {
                "prediction": pred_latex,
                "target": target_latex,
                "low_confidence_tokens": low_confidence_tokens,
                "token_by_token": [
                    {
                        "pred_token": tokenizer.id_to_token.get(t, "<UNK>"),
                        "confidence": float(c),
                        "is_correct": bool(t == target_seq[i])
                        if i < len(target_seq)
                        else None,
                    }
                    for i, (t, c) in enumerate(zip(pred_seq, pred_confidences))
                    if i < 20  # Limit to first 20 tokens for readability
                ],
            }
        )

    return {"samples": samples}


def save_enhanced_metrics(
    metrics: Dict[str, Any], experiment_name: str, metrics_dir: str, epoch: int
) -> None:
    """
    Save enhanced metrics to a JSON file.

    Args:
        metrics: Dictionary of metrics
        experiment_name: Name of the experiment
        metrics_dir: Directory to save metrics
        epoch: Current epoch
    """
    # Create metrics directory if it doesn't exist
    os.makedirs(metrics_dir, exist_ok=True)

    # Create filename
    filename = f"{experiment_name}_enhanced_metrics_epoch_{epoch}.json"
    filepath = os.path.join(metrics_dir, filename)

    # Convert numpy values to Python types for JSON serialization
    def convert_numpy(obj):
        if obj is None:
            return None
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, torch.Tensor):
            return convert_numpy(obj.detach().cpu().numpy())
        elif isinstance(obj, np.bool_) or isinstance(obj, bool):
            return bool(obj)
        else:
            return obj

    metrics_converted = convert_numpy(metrics)

    # Save to file
    with open(filepath, "w") as f:
        json.dump(metrics_converted, f, indent=2)

    logger.info(f"Enhanced metrics saved to {filepath}")


def log_enhanced_metrics_summary(metrics: Dict[str, Any]) -> None:
    """
    Log a summary of enhanced metrics.

    Args:
        metrics: Dictionary of metrics
    """
    # Extract token distribution metrics
    token_dist = metrics.get("token_distribution", {})
    pred_info = token_dist.get("predictions", {})

    # Log summary information
    logger.info("Enhanced Metrics Summary:")

    # Token distribution summary
    if "repetition_factor" in pred_info:
        repetition_factor = pred_info["repetition_factor"]
        if repetition_factor > 0.5:
            logger.warning(
                f"High token repetition detected (factor: {repetition_factor:.2f}). "
                "The model may be stuck predicting the same tokens."
            )

    if "diversity" in pred_info:
        diversity = pred_info["diversity"]
        logger.info(f"Prediction diversity: {diversity:.4f}")

    # Sample summary
    samples = metrics.get("samples", {}).get("samples", [])
    if samples:
        logger.info(f"Sample predictions analyzed: {len(samples)}")

        # Count samples with low confidence tokens
        low_conf_count = sum(1 for s in samples if s.get("low_confidence_tokens"))
        if low_conf_count > 0:
            logger.warning(
                f"{low_conf_count}/{len(samples)} samples have low confidence predictions."
            )


def compute_all_metrics(
    outputs: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
    all_predictions: List[List[int]],
    all_targets: List[List[int]],
    tokenizer: LaTeXTokenizer,
    num_samples: int = 2,
    confidence_threshold: float = 0.5,
    experiment_name: Optional[str] = None,
    metrics_dir: Optional[str] = None,
    save_to_file: bool = False,
    epoch: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute all metrics for model evaluation in a single call.

    Args:
        outputs: Model outputs tensor (batch_size, seq_length, vocab_size) or None
        targets: Target tensor (batch_size, seq_length) or None
        all_predictions: List of all predicted token sequences
        all_targets: List of all target token sequences
        tokenizer: LaTeX tokenizer
        num_samples: Number of samples to include in visualization
        confidence_threshold: Threshold for highlighting low confidence predictions
        experiment_name: Name of the experiment (required if save_to_file=True)
        metrics_dir: Directory to save metrics (required if save_to_file=True)
        save_to_file: Whether to save metrics to a file
        epoch: Current epoch (required if save_to_file=True)

    Returns:
        Dictionary of all metrics combined
    """
    # Initialize combined metrics dictionary
    combined_metrics = {}

    # 1. Detach tensors to CPU once to avoid redundant operations
    all_predictions_cpu = _detach_to_cpu(all_predictions)
    all_targets_cpu = _detach_to_cpu(all_targets)

    # 2. Calculate accuracy if outputs and targets are provided
    if outputs is not None and targets is not None:
        outputs_cpu = _detach_to_cpu(outputs)
        targets_cpu = _detach_to_cpu(targets)

        accuracy, num_tokens = masked_accuracy(
            outputs_cpu, targets_cpu, tokenizer.pad_token_id
        )
        combined_metrics["accuracy"] = accuracy
        combined_metrics["num_tokens"] = num_tokens

        # 5. Sample predictions and targets for visualization (only if outputs/targets provided)
        sample_data = sample_predictions_and_targets(
            outputs_cpu, targets_cpu, tokenizer, num_samples, confidence_threshold
        )
        combined_metrics["samples"] = sample_data["samples"]
    else:
        # Set default values when no tensor inputs provided
        combined_metrics["accuracy"] = 0.0
        combined_metrics["num_tokens"] = 0
        combined_metrics["samples"] = {"samples": []}

    # 3. Calculate BLEU and Levenshtein metrics (always calculate on list inputs)
    basic_metrics = calculate_metrics(all_predictions_cpu, all_targets_cpu)
    combined_metrics["bleu"] = basic_metrics["bleu"]
    combined_metrics["levenshtein"] = basic_metrics["levenshtein"]
    combined_metrics["batch_size"] = basic_metrics["batch_size"]

    # 4. Analyze token distribution (always calculate on list inputs)
    token_distribution = analyze_token_distribution(
        all_predictions_cpu, all_targets_cpu, tokenizer
    )
    combined_metrics["token_distribution"] = token_distribution

    # Add epoch if provided
    if epoch is not None:
        combined_metrics["epoch"] = epoch

    # 7. Save to file if requested
    if save_to_file:
        # Validate required parameters are present
        if not all([experiment_name, metrics_dir, epoch is not None]):
            logger.warning(
                "Cannot save metrics to file: missing required parameters "
                "(experiment_name, metrics_dir, or epoch)"
            )
        else:
            save_enhanced_metrics(combined_metrics, experiment_name, metrics_dir, epoch)

    # 8. Always log a summary
    log_enhanced_metrics_summary(combined_metrics)

    # 9. Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return combined_metrics

```

## File: training/__init__.py

- Extension: .py
- Language: python
- Size: 229 bytes
- Created: 2025-04-16 03:29:56
- Modified: 2025-04-16 03:29:56

### Code

```python
from img2latex.training.metrics import (
    bleu_n_score,
    calculate_metrics,
    levenshtein_distance,
    masked_accuracy,
)
from img2latex.training.predictor import Predictor
from img2latex.training.trainer import Trainer

```

## File: training/trainer.py

- Extension: .py
- Language: python
- Size: 32031 bytes
- Created: 2025-04-16 17:26:15
- Modified: 2025-04-16 17:26:15

### Code

```python
"""
Training and validation logic for the image-to-LaTeX model.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.data.utils import prepare_batch
from img2latex.training.metrics import compute_all_metrics, masked_accuracy
from img2latex.utils.logging import get_logger
from img2latex.utils.mps_utils import set_device
from img2latex.utils.path_utils import path_manager
from img2latex.utils.registry import experiment_registry

logger = get_logger(__name__)


class Trainer:
    """
    Trainer for the image-to-LaTeX model.

    This class handles the training and validation of the model,
    as well as checkpointing and metric logging.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: LaTeXTokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        experiment_name: str,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            tokenizer: Tokenizer for LaTeX formulas
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Configuration dictionary
            experiment_name: Name of the experiment
            device: Device to use for training (will be auto-detected if None)
        """
        # Set device
        if device is None:
            device_name = config.get("training", {}).get("device", None)
            self.device = set_device(device_name)
        else:
            self.device = device

        # Model and tokenizer
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training configuration
        self.config = config
        self.experiment_name = experiment_name

        # Get training parameters
        training_config = config.get("training", {})

        # Optimizer and loss
        self.learning_rate = training_config.get("learning_rate", 0.001)
        self.weight_decay = training_config.get("weight_decay", 0.0001)

        # Gradient accumulation steps
        self.accumulation_steps = training_config.get("accumulation_steps", 1)
        logger.info(f"Using gradient accumulation with {self.accumulation_steps} steps")

        # Create optimizer
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Create loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id, reduction="mean"
        )

        # Other training parameters
        self.max_epochs = training_config.get("epochs", 50)
        self.grad_clip_norm = training_config.get("clip_grad_norm", 5.0)
        self.early_stopping_patience = training_config.get(
            "early_stopping_patience", 10
        )

        # Use save_checkpoint_epochs if available, otherwise fall back to steps
        self.save_checkpoint_epochs = training_config.get("save_checkpoint_epochs", 10)
        self.save_checkpoint_steps = training_config.get("save_checkpoint_steps", 1000)
        self.use_epoch_checkpointing = "save_checkpoint_epochs" in training_config

        # Initialize step and epoch counters
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_metrics = {}
        self.patience_counter = 0

        # Register the experiment
        self._register_experiment()

        logger.info(
            f"Initialized trainer for experiment '{experiment_name}' "
            f"on device '{self.device}' with {self.max_epochs} epochs"
        )

    def _register_experiment(self) -> None:
        """Register the experiment with the experiment registry."""
        # Extract relevant config for logging
        training_config = self.config.get("training", {})
        model_config = self.config.get("model", {})

        # Build experiment description
        model_type = model_config.get("name", "cnn_lstm")
        model_desc = (
            f"{model_type} with {model_config['embedding_dim']} embedding dim, "
            f"{model_config['decoder']['hidden_dim']} hidden dim"
        )

        # Create description
        description = (
            f"Image-to-LaTeX model: {model_desc}. "
            f"Training with lr={training_config['learning_rate']}, "
            f"batch_size={training_config.get('batch_size', 128)}, "
            f"max_epochs={self.max_epochs}"
        )

        # Create tags
        tags = [model_type, f"lr_{training_config['learning_rate']}"]

        # Register with experiment registry
        experiment_registry.register_experiment(
            experiment_name=self.experiment_name,
            config=self.config,
            description=description,
            tags=tags,
        )

        # Update experiment status
        experiment_registry.update_experiment_status(
            self.experiment_name, "initialized"
        )

    def save_checkpoint(
        self, epoch: int, step: int, metrics: Dict[str, float], is_best: bool = False
    ) -> str:
        """
        Save a checkpoint of the model.

        Args:
            epoch: Current epoch
            step: Current step
            metrics: Dictionary of metrics
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to the saved checkpoint
        """
        # Get checkpoint directory
        checkpoint_dir = experiment_registry.path_manager.get_checkpoint_dir(
            self.experiment_name
        )

        # Create checkpoint name
        checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}.pt"
        if is_best:
            checkpoint_name = f"best_{checkpoint_name}"

        checkpoint_path = checkpoint_dir / checkpoint_name

        # Create a checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "tokenizer_config": {
                "token_to_id": self.tokenizer.token_to_id,
                "special_tokens": self.tokenizer.special_tokens,
                "max_sequence_length": self.tokenizer.max_sequence_length,
            },
        }

        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Also save best checkpoint separately
        if is_best:
            best_path = checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint

        Returns:
            Loaded checkpoint dictionary
        """
        # Check if the checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model and optimizer state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Update counters
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["step"]

        # Update best validation loss if available
        if "metrics" in checkpoint and "val_loss" in checkpoint["metrics"]:
            self.best_val_loss = checkpoint["metrics"]["val_loss"]

        logger.info(
            f"Loaded checkpoint from {checkpoint_path} "
            f"(epoch {self.current_epoch}, step {self.global_step})"
        )

        return checkpoint

    def train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_tokens = 0
        epoch_samples = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Training epoch {self.current_epoch + 1}/{self.max_epochs}",
            leave=False,
        )

        # Do a deep clean before starting the epoch
        if self.device.type == "mps":
            from img2latex.utils.mps_utils import deep_clean_memory

            deep_clean_memory()

        # Only zero gradients at the beginning if using accumulation
        if self.accumulation_steps > 1:
            self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch
            images, targets = prepare_batch(
                batch, self.device, model_type=self.config["model"]["name"]
            )

            # Zero the gradients if not using accumulation or at the start of accumulation cycle
            if self.accumulation_steps == 1:
                self.optimizer.zero_grad(
                    set_to_none=True
                )  # More efficient memory clearing

            # Forward pass
            outputs = self.model(images, targets)

            # Reshape for loss calculation
            batch_size, seq_length, vocab_size = outputs.shape
            targets_shifted = targets[:, 1:]  # Skip the first token (START token)

            # Calculate loss and normalize by accumulation steps
            loss = (
                self.criterion(
                    outputs.reshape(-1, vocab_size), targets_shifted.reshape(-1)
                )
                / self.accumulation_steps
            )  # Normalize loss

            # Backward pass
            loss.backward()

            # Get the loss value before optimization (multiply by accumulation steps to get the actual loss)
            loss_value = loss.item() * self.accumulation_steps
            # Use masked_accuracy for simple training metrics calculation
            acc_value, num_tokens_value = masked_accuracy(
                outputs, targets_shifted, self.tokenizer.pad_token_id
            )

            # Free memory by explicitly removing references to tensors
            del loss, outputs, images, targets, targets_shifted

            # Update counters
            epoch_loss += loss_value * batch_size
            epoch_acc += acc_value * num_tokens_value
            epoch_tokens += num_tokens_value
            epoch_samples += batch_size

            # Only update weights and reset gradients after accumulation steps or at the end of epoch
            if (batch_idx + 1) % self.accumulation_steps == 0 or batch_idx == len(
                self.train_loader
            ) - 1:
                # Gradient clipping
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )

                # Optimizer step
                self.optimizer.step()

                # Zero gradients after optimization
                self.optimizer.zero_grad(set_to_none=True)

                # Increment global step after accumulation cycle completes
                self.global_step += 1

                # Update progress bar
                progress_bar.set_postfix({"loss": loss_value, "acc": acc_value})

                # Save checkpoint if needed
                if (
                    self.use_epoch_checkpointing
                    and (self.current_epoch + 1) % self.save_checkpoint_epochs == 0
                    and batch_idx == len(self.train_loader) - 1
                ):
                    # Save at the end of epochs that are divisible by save_checkpoint_epochs
                    metrics = {
                        "train_loss": epoch_loss / epoch_samples,
                        "train_acc": epoch_acc / epoch_tokens,
                    }
                    self.save_checkpoint(
                        epoch=self.current_epoch, step=self.global_step, metrics=metrics
                    )
                elif (
                    not self.use_epoch_checkpointing
                    and self.global_step % self.save_checkpoint_steps == 0
                ):
                    # Traditional step-based saving if epoch-based saving is not enabled
                    metrics = {
                        "train_loss": epoch_loss / epoch_samples,
                        "train_acc": epoch_acc / epoch_tokens,
                    }
                    self.save_checkpoint(
                        epoch=self.current_epoch, step=self.global_step, metrics=metrics
                    )

                # Deep clean after saving checkpoint (for either type of checkpoint)
                if (
                    self.use_epoch_checkpointing
                    and (self.current_epoch + 1) % self.save_checkpoint_epochs == 0
                    and batch_idx == len(self.train_loader) - 1
                ) or (
                    not self.use_epoch_checkpointing
                    and self.global_step % self.save_checkpoint_steps == 0
                ):
                    if self.device.type == "mps":
                        from img2latex.utils.mps_utils import deep_clean_memory

                        deep_clean_memory()

            # Clean up GPU memory in MPS mode - do more frequently (every 5 batches)
            if self.device.type == "mps":
                from img2latex.utils.mps_utils import empty_cache

                if batch_idx % 5 == 0:
                    empty_cache(force_gc=True)
                # Perform a deeper clean periodically
                if batch_idx % 50 == 0:
                    from img2latex.utils.mps_utils import deep_clean_memory

                    deep_clean_memory()

        # Calculate epoch metrics
        metrics = {
            "train_loss": epoch_loss / epoch_samples,
            "train_acc": epoch_acc / epoch_tokens,
            "epoch": self.current_epoch + 1,
            "step": self.global_step,
        }

        # Log metrics
        experiment_registry.log_metrics(
            self.experiment_name, metrics, step=self.current_epoch + 1
        )

        logger.info(
            f"Epoch {self.current_epoch + 1}/{self.max_epochs}, "
            f"Train Loss: {metrics['train_loss']:.4f}, "
            f"Accuracy: {metrics['train_acc']:.4f}"
        )

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate the model on the validation set.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        # Do a deep clean before starting validation
        if self.device.type == "mps":
            from img2latex.utils.mps_utils import deep_clean_memory

            deep_clean_memory()

        val_loss = 0.0
        val_acc = 0.0
        val_tokens = 0
        val_samples = 0

        # Lists to store predictions and targets for metric calculation
        all_predictions = []
        all_targets = []

        # Store outputs and targets for enhanced metrics
        enhanced_metrics_batch = None
        enhanced_metrics_targets = None

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Validation epoch {self.current_epoch + 1}/{self.max_epochs}",
            leave=False,
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Prepare batch
                    images, targets = prepare_batch(
                        batch, self.device, model_type=self.config["model"]["name"]
                    )

                    # Forward pass
                    outputs = self.model(images, targets)

                    # Reshape for loss calculation
                    batch_size, seq_length, vocab_size = outputs.shape
                    targets_shifted = targets[
                        :, 1:
                    ]  # Skip the first token (START token)

                    # Calculate loss
                    loss = self.criterion(
                        outputs.reshape(-1, vocab_size), targets_shifted.reshape(-1)
                    )

                    # Store values before freeing memory
                    loss_value = loss.item()

                    # Calculate accuracy using only masked_accuracy
                    from img2latex.training.metrics import masked_accuracy

                    acc, num_tokens = masked_accuracy(
                        outputs, targets_shifted, self.tokenizer.pad_token_id
                    )
                    acc_value = acc

                    # Update counters
                    val_loss += loss_value * batch_size
                    val_acc += acc_value * num_tokens
                    val_tokens += num_tokens
                    val_samples += batch_size

                    # Update progress bar
                    progress_bar.set_postfix({"loss": loss_value, "acc": acc_value})

                    # Get predictions for metric calculation
                    # We'll use a subset of the validation set for BLEU and Levenshtein metrics
                    if batch_idx < 10:  # Limit to first 10 batches to save time
                        pred_ids = torch.argmax(outputs, dim=-1).cpu().numpy()
                        target_ids = targets_shifted.cpu().numpy()

                        # Convert to list of lists
                        for i in range(batch_size):
                            # Get masks for non-padding tokens
                            pred_mask = pred_ids[i] != self.tokenizer.pad_token_id
                            target_mask = target_ids[i] != self.tokenizer.pad_token_id

                            # Get non-padding tokens
                            pred_tokens = pred_ids[i][pred_mask].tolist()
                            target_tokens = target_ids[i][target_mask].tolist()

                            # Add to lists
                            all_predictions.append(pred_tokens)
                            all_targets.append(target_tokens)

                        # Store first batch for enhanced metrics (make copies to avoid memory issues)
                        if batch_idx == 0:
                            enhanced_metrics_batch = outputs.detach().cpu().clone()
                            enhanced_metrics_targets = (
                                targets_shifted.detach().cpu().clone()
                            )

                    # Free memory explicitly
                    del outputs, loss, images, targets, targets_shifted

                    # Clean up GPU memory in MPS mode more aggressively
                    if self.device.type == "mps":
                        # Clean up more frequently
                        if batch_idx % 5 == 0:
                            from img2latex.utils.mps_utils import empty_cache

                            empty_cache(force_gc=True)

                        # Perform deep clean periodically
                        if batch_idx % 25 == 0 and batch_idx > 0:
                            from img2latex.utils.mps_utils import deep_clean_memory

                            deep_clean_memory()

                except RuntimeError as e:
                    # Handle out of memory errors gracefully
                    if "out of memory" in str(e).lower():
                        logger.warning(
                            f"Out of memory during validation at batch {batch_idx}. Cleaning memory and continuing."
                        )

                        # Do an aggressive memory cleanup
                        if self.device.type == "mps":
                            from img2latex.utils.mps_utils import deep_clean_memory

                            deep_clean_memory()

                        # Skip this batch and continue
                        continue
                    else:
                        # Re-raise other errors
                        raise

        # Calculate validation metrics with safety checks to avoid division by zero
        if val_samples > 0 and val_tokens > 0:
            metrics = {
                "val_loss": val_loss / val_samples,
                "val_acc": val_acc / val_tokens,
                "epoch": self.current_epoch + 1,
                "step": self.global_step,
            }
        else:
            # Fallback if no valid samples were processed
            metrics = {
                "val_loss": float("inf"),
                "val_acc": 0.0,
                "epoch": self.current_epoch + 1,
                "step": self.global_step,
            }

        # Calculate additional metrics if we have predictions
        if all_predictions:
            try:
                # Get metrics directory
                metrics_dir = path_manager.get_metrics_dir(self.experiment_name)

                # Determine if we should generate enhanced metrics visualization
                should_generate_enhanced = (
                    enhanced_metrics_batch is not None
                    and enhanced_metrics_targets is not None
                    and (
                        self.current_epoch % 5 == 0
                        or val_loss / val_samples < self.best_val_loss
                    )
                )

                if should_generate_enhanced:
                    try:
                        # Do a memory cleanup before generating metrics
                        if self.device.type == "mps":
                            from img2latex.utils.mps_utils import empty_cache

                            empty_cache(force_gc=True)

                        # Compute all metrics in one call
                        enhanced_metrics = compute_all_metrics(
                            outputs=enhanced_metrics_batch,
                            targets=enhanced_metrics_targets,
                            all_predictions=all_predictions,
                            all_targets=all_targets,
                            tokenizer=self.tokenizer,
                            num_samples=min(2, len(all_predictions)),
                            experiment_name=self.experiment_name,
                            metrics_dir=metrics_dir,
                            epoch=self.current_epoch,
                            save_to_file=True,
                        )

                        # Update metrics with calculated values from the unified metrics
                        metrics.update(
                            {
                                "val_bleu": enhanced_metrics["bleu"],
                                "val_levenshtein": enhanced_metrics["levenshtein"],
                            }
                        )

                    except Exception as e:
                        logger.error(f"Error generating enhanced metrics: {e}")
                    finally:
                        # Clean up enhanced metrics tensors explicitly
                        del enhanced_metrics_batch
                        del enhanced_metrics_targets

                        if self.device.type == "mps":
                            from img2latex.utils.mps_utils import deep_clean_memory

                            deep_clean_memory()
                else:
                    # Just compute basic metrics without enhanced visualization
                    basic_metrics = compute_all_metrics(
                        outputs=None,  # Skip tensor-based metrics
                        targets=None,  # Skip tensor-based metrics
                        all_predictions=all_predictions,
                        all_targets=all_targets,
                        tokenizer=self.tokenizer,
                        save_to_file=False,  # Don't save when not doing enhanced metrics
                    )
                    metrics.update(
                        {
                            "val_bleu": basic_metrics["bleu"],
                            "val_levenshtein": basic_metrics["levenshtein"],
                        }
                    )

            except Exception as e:
                logger.error(f"Error calculating additional metrics: {e}")
                metrics.update(
                    {
                        "val_bleu": 0.0,
                        "val_levenshtein": 0.0,
                    }
                )

        # Log metrics
        experiment_registry.log_metrics(
            self.experiment_name, metrics, step=self.current_epoch + 1
        )

        logger.info(
            f"Epoch {self.current_epoch + 1}/{self.max_epochs}, "
            f"Validation Loss: {metrics['val_loss']:.4f}, "
            f"Accuracy: {metrics['val_acc']:.4f}"
            + (
                f", BLEU: {metrics.get('val_bleu', 0):.4f}"
                if "val_bleu" in metrics
                else ""
            )
            + (
                f", Levenshtein: {metrics.get('val_levenshtein', 0):.4f}"
                if "val_levenshtein" in metrics
                else ""
            )
        )

        return metrics

    def train(self) -> Dict[str, float]:
        """
        Train the model for the specified number of epochs.

        Returns:
            Dictionary of best validation metrics
        """
        # Update experiment status
        experiment_registry.update_experiment_status(self.experiment_name, "training")

        # Set smaller batch sizes for MPS if needed
        if self.device.type == "mps":
            # Check available MPS memory
            import os

            import torch

            try:
                rec_max = torch.mps.recommended_max_memory()
                # If we have less than 6GB of available memory, reduce batch sizes
                if rec_max < 6 * (1024**3):
                    train_batch_size = self.config.get("data", {}).get(
                        "batch_size", 128
                    )
                    if train_batch_size > 32:
                        new_batch_size = 32
                        logger.warning(
                            f"Limited MPS memory detected ({rec_max / (1024**3):.2f}GB). Reducing batch size from {train_batch_size} to {new_batch_size}"
                        )
                        # Set environment variable for batch size override
                        os.environ["MPS_BATCH_SIZE_OVERRIDE"] = str(new_batch_size)
            except Exception as e:
                logger.warning(f"Error checking MPS memory: {e}")

        # Training loop
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            # Clean up memory at the start of each epoch
            if self.device.type == "mps":
                from img2latex.utils.mps_utils import deep_clean_memory

                deep_clean_memory()

            try:
                # Train for one epoch
                train_metrics = self.train_epoch()

                # Clean up memory between training and validation
                if self.device.type == "mps":
                    from img2latex.utils.mps_utils import deep_clean_memory

                    deep_clean_memory()

                # Validate
                val_metrics = self.validate()

                # Explicit cache clearing after validation
                if self.device.type == "mps":
                    torch.mps.empty_cache()

                # Check for improvement
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.best_val_metrics = val_metrics
                    self.patience_counter = 0

                    # Save best checkpoint
                    self.save_checkpoint(
                        epoch=epoch,
                        step=self.global_step,
                        metrics=val_metrics,
                        is_best=True,
                    )
                else:
                    self.patience_counter += 1
                    logger.info(
                        f"No improvement for {self.patience_counter} epochs "
                        f"(best val_loss: {self.best_val_loss:.4f})"
                    )

                    # Save checkpoint
                    self.save_checkpoint(
                        epoch=epoch, step=self.global_step, metrics=val_metrics
                    )

                    # Clear cache after non-improvement checkpoint
                    if self.device.type == "mps":
                        torch.mps.empty_cache()

                    # Early stopping
                    if self.patience_counter >= self.early_stopping_patience:
                        logger.info(
                            f"Early stopping after {epoch + 1} epochs "
                            f"({self.patience_counter} epochs without improvement)"
                        )
                        break

                    # Ensure cache is cleared at end of epoch loop
                    if self.device.type == "mps":
                        torch.mps.empty_cache()

                    # Clean up memory at the end of each epoch
                    if self.device.type == "mps":
                        from img2latex.utils.mps_utils import deep_clean_memory

                        deep_clean_memory()

                        # Add diagnostics for MPS memory usage
                        print(
                            f"Epoch {epoch + 1} - MPS allocated: {torch.mps.current_allocated_memory() / (1024**2):.2f} MB"
                        )

            except RuntimeError as e:
                # Handle out of memory errors gracefully
                if "out of memory" in str(e).lower():
                    logger.error(f"Out of memory error during epoch {epoch + 1}: {e}")

                    # Clean memory and try to reduce batch size for next epoch
                    if self.device.type == "mps":
                        from img2latex.utils.mps_utils import deep_clean_memory

                        deep_clean_memory()

                        # Try to reduce batch size
                        if hasattr(self.train_loader, "batch_sampler") and hasattr(
                            self.train_loader.batch_sampler, "batch_size"
                        ):
                            current_size = self.train_loader.batch_sampler.batch_size
                            new_size = max(8, current_size // 2)  # Don't go below 8

                            if new_size < current_size:
                                logger.warning(
                                    f"Reducing batch size from {current_size} to {new_size} due to OOM error"
                                )
                                self.train_loader.batch_sampler.batch_size = new_size

                                # Also adjust val loader if possible
                                if hasattr(self.val_loader, "batch_sampler"):
                                    self.val_loader.batch_sampler.batch_size = new_size

                                # Continue to next epoch with reduced batch size
                                continue

                    # If we can't adjust, re-raise the error
                    raise
                else:
                    # Re-raise other errors
                    raise

        # Update experiment status
        experiment_registry.update_experiment_status(self.experiment_name, "completed")

        return self.best_val_metrics

```

## File: utils/logging.py

- Extension: .py
- Language: python
- Size: 10379 bytes
- Created: 2025-04-16 15:02:05
- Modified: 2025-04-16 15:02:05

### Code

```python
"""
Logging utilities for the img2latex project.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

# Dictionary to keep track of loggers that have been created
_LOGGERS: Dict[str, logging.Logger] = {}
_GLOBAL_FILE_HANDLER = None


class ImmediateFileHandler(logging.FileHandler):
    """Custom FileHandler that flushes after each emit to ensure logs are written immediately."""

    def __init__(self, filename, mode="a", encoding=None, delay=False):
        # Don't use the parent's initialization, manage our own file
        logging.Handler.__init__(self)
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.terminator = "\n"

        # Open the file immediately with line buffering (buffering=1)
        self._open()

    def _open(self):
        """Open the file with line buffering"""
        self.stream = open(
            self.baseFilename, self.mode, buffering=1, encoding=self.encoding
        )
        return self.stream

    def emit(self, record):
        """Override emit method to flush immediately after writing and ensure stream is open."""
        if self.stream is None:
            self._open()
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        """
        Close the file and clean up.
        """
        self.acquire()
        try:
            if self.stream:
                self.flush()
                self.stream.close()
                self.stream = None
        finally:
            self.release()


def get_logger(
    name: str,
    log_level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    timestamp: bool = False,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Create or retrieve a configured logger instance.

    Args:
        name: Name of the logger (typically __name__)
        log_level: Log level (e.g. INFO, DEBUG)
        log_file: Optional filename (e.g. 'train.log')
        log_dir: Directory to store log file
        timestamp: If True, append timestamp to log_file
        use_colors: Use colored log output for console

    Returns:
        logging.Logger instance
    """
    global _GLOBAL_FILE_HANDLER

    if name in _LOGGERS:
        return _LOGGERS[name]

    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Disable propagation for all loggers except the root logger
    # This is the key fix for double logging
    logger.propagate = False

    # Clear old handlers
    if logger.handlers:
        logger.handlers.clear()

    # Formatter
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, date_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    if use_colors and sys.stdout.isatty():
        console_handler.setFormatter(ColoredFormatter(log_format, date_format))
    else:
        console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler - Handle both global and per-logger file handlers
    if log_dir and log_file:
        os.makedirs(log_dir, exist_ok=True)
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{Path(log_file).stem}_{timestamp_str}{Path(log_file).suffix}"
        file_path = os.path.join(log_dir, log_file)

        # Use global file handler for main logger and its children
        handler_to_use = None
        if name == "img2latex" or name.startswith("img2latex."):
            # Create a new file handler if not already present
            if (
                _GLOBAL_FILE_HANDLER is None
                or _GLOBAL_FILE_HANDLER.baseFilename != file_path
            ):
                # Close previous handler if it exists
                if _GLOBAL_FILE_HANDLER is not None:
                    _GLOBAL_FILE_HANDLER.close()

                # Use our custom ImmediateFileHandler instead of regular FileHandler
                _GLOBAL_FILE_HANDLER = ImmediateFileHandler(file_path, mode="a")
                _GLOBAL_FILE_HANDLER.setFormatter(formatter)
                _GLOBAL_FILE_HANDLER.setLevel(log_level)
            handler_to_use = _GLOBAL_FILE_HANDLER
        # For non-img2latex loggers that need their own file handler
        else:
            # Create a new file handler for this logger using our custom handler
            file_handler = ImmediateFileHandler(file_path, mode="a")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            handler_to_use = file_handler

        # Ensure the handler is attached (it might have been removed)
        # Add handler_to_use only if it's valid and not already present
        if handler_to_use and handler_to_use not in logger.handlers:
            logger.addHandler(handler_to_use)
            # Avoid logging this message repeatedly if handler already exists
            # Initialize a flag on the handler itself the first time it's used for logging path
            if not getattr(handler_to_use, "_logging_path_logged", False):
                logger.info(f"Logging to file: {file_path}")
                handler_to_use._logging_path_logged = True  # Set flag after logging

    _LOGGERS[name] = logger
    return logger


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[37m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
        "RESET": "\033[0m",
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )
        return super().format(record)


def configure_logging(cfg) -> logging.Logger:
    """
    Global logging setup using config.

    Args:
        cfg: Config object from configuration

    Returns:
        Root logger
    """
    global _GLOBAL_FILE_HANDLER
    _GLOBAL_FILE_HANDLER = None

    # Force Python to use unbuffered mode - this affects stdout/stderr
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Pull settings from config
    experiment_name = (
        cfg.training.experiment_name
        if hasattr(cfg, "training") and hasattr(cfg.training, "experiment_name")
        else "default"
    )
    log_level = cfg.logging.level if hasattr(cfg, "logging") else "INFO"
    use_colors = cfg.logging.use_colors if hasattr(cfg, "logging") else True
    log_to_file = cfg.logging.log_to_file if hasattr(cfg, "logging") else True

    # Check if we have a path manager instance
    try:
        from img2latex.utils.path_utils import path_manager

        # Use the path manager to get the experiment directory
        log_dir = path_manager.get_log_dir(experiment_name)
        print(
            f"[DEBUG] configure_logging: experiment name '{experiment_name}', writing logs to {log_dir}"
        )
    except (ImportError, AttributeError):
        # Fallback if path_manager isn't available
        from pathlib import Path

        output_dir = (
            cfg.training.output_dir
            if hasattr(cfg, "training") and hasattr(cfg.training, "output_dir")
            else "outputs"
        )
        log_dir = Path(output_dir) / experiment_name / "logs"
        os.makedirs(log_dir, exist_ok=True)
        print(
            f"[DEBUG] configure_logging fallback: experiment name '{experiment_name}', writing logs to {log_dir}"
        )

    # Create log filename based on command
    log_file = "train.log"
    if (
        hasattr(cfg, "logging")
        and hasattr(cfg.logging, "log_file")
        and cfg.logging.log_file
    ):
        log_file = cfg.logging.log_file

    # Silence noisy libraries
    for noisy in ["matplotlib", "PIL"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Configure the root logger
    root_root_logger = logging.getLogger()
    root_root_logger.setLevel(
        logging.WARNING
    )  # Only show warnings and above for third-party libs
    # Remove all handlers from the root logger to avoid duplicated logs
    for handler in root_root_logger.handlers[:]:
        root_root_logger.removeHandler(handler)

    # Create main app logger
    if log_to_file:
        root_logger = get_logger(
            "img2latex",
            log_level=log_level,
            log_dir=str(log_dir),
            log_file=log_file,
            use_colors=use_colors,
        )
    else:
        root_logger = get_logger(
            "img2latex",
            log_level=log_level,
            use_colors=use_colors,
        )

    # Set up module loggers
    for module in [
        "data",
        "model",
        "training",
        "evaluation",
        "utils",
    ]:
        get_logger(f"img2latex.{module}", log_level=log_level)

    # Register cleanup
    import atexit

    def flush_all_loggers():
        """Flush all loggers at exit to ensure logs are written."""
        for _name, logger in _LOGGERS.items():
            for handler in logger.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

    atexit.register(flush_all_loggers)

    return root_logger


def log_execution_params(logger, cfg):
    """
    Log core execution metadata for reproducibility.

    Args:
        logger: Logger instance
        cfg: Config object from configuration
    """
    logger.info("------ Execution Context ------")
    logger.info(f"Command        : {cfg.get('command', 'N/A')}")
    logger.info(f"Model          : {cfg.model.get('name', 'N/A')}")
    logger.info(f"Dataset        : {cfg.data.get('dataset_name', 'N/A')}")
    logger.info(f"Experiment     : {cfg.get('experiment_name', 'N/A')}")
    logger.info(f"Time           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("-------------------------------")

```

## File: utils/visualize_metrics.py

- Extension: .py
- Language: python
- Size: 8719 bytes
- Created: 2025-04-16 16:05:58
- Modified: 2025-04-16 16:05:58

### Code

```python
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

```

## File: utils/registry.py

- Extension: .py
- Language: python
- Size: 18468 bytes
- Created: 2025-04-16 03:30:21
- Modified: 2025-04-16 03:30:21

### Code

```python
"""
Experiment registry for the img2latex project.

This module provides functionality for tracking, comparing, and managing experiments.
It builds on the PathManager's basic registry features to provide more advanced
experiment tracking capabilities.
"""

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from img2latex.utils.logging import get_logger
from img2latex.utils.path_utils import path_manager

logger = get_logger(__name__)


class ExperimentRegistry:
    """
    Advanced experiment registry for tracking and comparing experiments.

    This class extends the basic functionality in PathManager to provide
    comprehensive experiment tracking features.
    """

    def __init__(self):
        """Initialize the experiment registry."""
        self.path_manager = path_manager
        self.registry_dir = self.path_manager.registry_dir
        self.registry_file = self.path_manager.experiment_registry_file

        # Ensure the registry directory exists
        os.makedirs(self.registry_dir, exist_ok=True)

    def register_experiment(
        self,
        experiment_name: str,
        config: Optional[Dict] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Register a new experiment or update an existing one.

        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary used for the experiment
            description: Optional description of the experiment
            tags: Optional list of tags for categorizing experiments

        Returns:
            The experiment name with version (e.g., "experiment_v1")
        """
        # Get current registry data
        registry = self._load_registry()

        # Normalize the experiment name and get the versioned name
        if "_v" not in experiment_name:
            # Find the next version number if experiment exists
            base_name = experiment_name
            version = 1

            # Check if experiment name already exists with any version
            existing_versions = []
            for name in registry.keys():
                if name.startswith(f"{base_name}_v"):
                    try:
                        existing_versions.append(int(name.split("_v")[1]))
                    except ValueError:
                        continue

            if existing_versions:
                version = max(existing_versions) + 1

            experiment_name = f"{base_name}_v{version}"

        # Create experiment directory structure
        paths = self.path_manager.create_experiment_structure(experiment_name)

        # Prepare metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = {
            "creation_time": timestamp,
            "last_updated": timestamp,
            "path": str(paths["experiment_dir"]),
            "description": description,
            "tags": tags or [],
            "metrics": {},
            "status": "created",
        }

        # Save configuration if provided
        if config:
            config_path = paths["config_path"]
            self._save_config(config_path, config)
            metadata["config_path"] = str(config_path)

        # Update registry
        registry[experiment_name] = metadata
        self._save_registry(registry)

        logger.info(f"Registered experiment: {experiment_name}")
        return experiment_name

    def update_experiment_status(self, experiment_name: str, status: str) -> None:
        """
        Update the status of an experiment.

        Args:
            experiment_name: Name of the experiment
            status: New status (e.g., 'running', 'completed', 'failed')
        """
        registry = self._load_registry()
        if experiment_name in registry:
            registry[experiment_name]["status"] = status
            registry[experiment_name]["last_updated"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            self._save_registry(registry)
            logger.info(f"Updated status of experiment {experiment_name} to '{status}'")
        else:
            logger.warning(
                f"Cannot update status: Experiment {experiment_name} not found in registry"
            )

    def log_metrics(
        self, experiment_name: str, metrics: Dict[str, Any], step: Optional[int] = None
    ) -> None:
        """
        Log metrics for an experiment.

        Args:
            experiment_name: Name of the experiment
            metrics: Dictionary of metrics to log
            step: Optional step number (e.g., epoch or iteration)
        """
        registry = self._load_registry()
        if experiment_name not in registry:
            logger.warning(
                f"Cannot log metrics: Experiment {experiment_name} not found in registry"
            )
            return

        # Convert metrics to JSON serializable format
        metrics = self._convert_to_serializable(metrics)

        # Update registry with metrics
        if "metrics" not in registry[experiment_name]:
            registry[experiment_name]["metrics"] = {}

        # If step is provided, organize metrics by step
        if step is not None:
            if "steps" not in registry[experiment_name]["metrics"]:
                registry[experiment_name]["metrics"]["steps"] = {}

            step_key = str(step)  # Convert to string for JSON compatibility
            if step_key not in registry[experiment_name]["metrics"]["steps"]:
                registry[experiment_name]["metrics"]["steps"][step_key] = {}

            # Update with new metrics
            registry[experiment_name]["metrics"]["steps"][step_key].update(metrics)
        else:
            # Just update the latest metrics (overwrite previous values)
            registry[experiment_name]["metrics"].update(metrics)

        # Update timestamp
        registry[experiment_name]["last_updated"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Save registry
        self._save_registry(registry)

        # Also save metrics to a separate file for the experiment
        metrics_dir = self.path_manager.get_metrics_dir(experiment_name)
        metrics_file = metrics_dir / "metrics.json"

        # Load existing metrics if file exists
        existing_metrics = {}
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    existing_metrics = json.load(f)
            except Exception as e:
                logger.error(f"Error loading existing metrics: {e}")

        # Update with new metrics
        if step is not None:
            if "steps" not in existing_metrics:
                existing_metrics["steps"] = {}

            step_key = str(step)
            if step_key not in existing_metrics["steps"]:
                existing_metrics["steps"][step_key] = {}

            existing_metrics["steps"][step_key].update(metrics)
        else:
            existing_metrics.update(metrics)

        # Save updated metrics
        try:
            with open(metrics_file, "w") as f:
                json.dump(existing_metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def get_experiment(self, experiment_name: str) -> Optional[Dict]:
        """
        Get details for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary with experiment details or None if not found
        """
        registry = self._load_registry()
        if experiment_name in registry:
            return registry[experiment_name]

        # Check if the experiment name is missing version
        if "_v" not in experiment_name:
            # Find the latest version
            latest_version = self.get_latest_version(experiment_name)
            if latest_version:
                versioned_name = f"{experiment_name}_v{latest_version}"
                logger.info(f"Using latest version: {versioned_name}")
                return registry.get(versioned_name)

        logger.warning(f"Experiment {experiment_name} not found in registry")
        return None

    def get_latest_version(self, base_experiment_name: str) -> Optional[int]:
        """
        Get the latest version number for an experiment.

        Args:
            base_experiment_name: Base name of the experiment without version

        Returns:
            Latest version number or None if no versions exist
        """
        registry = self._load_registry()
        versions = []

        for name in registry.keys():
            if name.startswith(f"{base_experiment_name}_v"):
                try:
                    version = int(name.split("_v")[1])
                    versions.append(version)
                except ValueError:
                    continue

        if versions:
            return max(versions)
        return None

    def list_experiments(
        self,
        filter_tag: Optional[str] = None,
        filter_status: Optional[str] = None,
        sort_by: str = "creation_time",
    ) -> List[Dict]:
        """
        List all registered experiments, optionally filtered and sorted.

        Args:
            filter_tag: Optional tag to filter experiments
            filter_status: Optional status to filter experiments
            sort_by: Field to sort by (creation_time, last_updated, name)

        Returns:
            List of experiment records
        """
        registry = self._load_registry()
        experiments = []

        for name, data in registry.items():
            # Add the name to the experiment data
            exp_data = data.copy()
            exp_data["name"] = name

            # Apply tag filter if specified
            if filter_tag and "tags" in data:
                if filter_tag not in data["tags"]:
                    continue

            # Apply status filter if specified
            if filter_status and "status" in data:
                if data["status"] != filter_status:
                    continue

            experiments.append(exp_data)

        # Sort experiments
        if sort_by == "name":
            experiments.sort(key=lambda x: x["name"])
        elif sort_by == "last_updated" and all(
            "last_updated" in exp for exp in experiments
        ):
            experiments.sort(key=lambda x: x["last_updated"], reverse=True)
        else:  # Default to creation_time
            experiments.sort(key=lambda x: x.get("creation_time", ""), reverse=True)

        return experiments

    def delete_experiment(
        self, experiment_name: str, delete_files: bool = False
    ) -> bool:
        """
        Delete an experiment from the registry.

        Args:
            experiment_name: Name of the experiment
            delete_files: Whether to also delete associated files

        Returns:
            True if successful, False otherwise
        """
        registry = self._load_registry()
        if experiment_name not in registry:
            logger.warning(
                f"Cannot delete: Experiment {experiment_name} not found in registry"
            )
            return False

        # Remove from registry
        experiment_data = registry.pop(experiment_name)
        self._save_registry(registry)

        # Delete files if requested
        if delete_files and "path" in experiment_data:
            experiment_path = Path(experiment_data["path"])
            if experiment_path.exists():
                try:
                    import shutil

                    shutil.rmtree(experiment_path)
                    logger.info(f"Deleted experiment directory: {experiment_path}")
                except Exception as e:
                    logger.error(f"Error deleting experiment files: {e}")
                    return False

        logger.info(f"Deleted experiment: {experiment_name}")
        return True

    def get_experiment_comparison(
        self, experiment_names: List[str], metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare metrics between multiple experiments.

        Args:
            experiment_names: List of experiment names to compare
            metrics: Optional list of specific metrics to compare

        Returns:
            DataFrame with experiments as rows and metrics as columns
        """
        registry = self._load_registry()
        comparison_data = []

        for name in experiment_names:
            if name not in registry:
                logger.warning(f"Experiment {name} not found, skipping in comparison")
                continue

            experiment = registry[name]
            row_data = {"experiment": name}

            # Extract metrics
            if "metrics" in experiment:
                exp_metrics = experiment["metrics"]

                # If specific metrics requested
                if metrics:
                    for metric in metrics:
                        if metric in exp_metrics:
                            row_data[metric] = exp_metrics[metric]
                # Otherwise, get all top-level metrics (excluding 'steps')
                else:
                    for metric, value in exp_metrics.items():
                        if metric != "steps":
                            row_data[metric] = value

                # Add best value from steps if available
                if "steps" in exp_metrics:
                    steps_data = exp_metrics["steps"]
                    best_values = {}

                    # Find best values across steps (assuming higher is better)
                    for step, step_metrics in steps_data.items():
                        for metric, value in step_metrics.items():
                            if isinstance(value, (int, float)):
                                if (
                                    metric not in best_values
                                    or value > best_values[metric]["value"]
                                ):
                                    best_values[metric] = {"value": value, "step": step}

                    # Add to row data with "best_" prefix
                    for metric, data in best_values.items():
                        row_data[f"best_{metric}"] = data["value"]
                        row_data[f"best_{metric}_step"] = data["step"]

            comparison_data.append(row_data)

        # Convert to DataFrame
        if comparison_data:
            return pd.DataFrame(comparison_data)
        else:
            return pd.DataFrame()

    def _load_registry(self) -> Dict:
        """
        Load the experiment registry from disk.

        Returns:
            Dictionary containing the experiment registry
        """
        if not self.registry_file.exists():
            return {}

        try:
            with open(self.registry_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading experiment registry: {e}")
            return {}

    def _save_registry(self, registry: Dict) -> None:
        """
        Save the experiment registry to disk.

        Args:
            registry: Dictionary containing the experiment registry
        """
        # Convert registry to serializable format
        registry = self._convert_to_serializable(registry)

        try:
            with open(self.registry_file, "w") as f:
                json.dump(registry, f, indent=2, sort_keys=True)
        except Exception as e:
            logger.error(f"Error saving experiment registry: {e}")

    def _convert_to_serializable(self, obj):
        """
        Convert non-serializable objects to JSON serializable types.

        Args:
            obj: Object to convert

        Returns:
            Converted object that is JSON serializable
        """
        if obj is None:
            return None
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(i) for i in obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif (
            hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy")
        ):  # PyTorch tensor
            return self._convert_to_serializable(obj.detach().cpu().numpy())
        elif hasattr(obj, "tolist"):  # Handle other numpy-like types with tolist method
            return obj.tolist()
        elif hasattr(obj, "item"):  # Handle single-item numpy and torch types
            return obj.item()
        else:
            return obj

    def _save_config(self, config_path: Path, config: Dict) -> None:
        """
        Save experiment configuration to disk.

        Args:
            config_path: Path where the config should be saved
            config: Configuration dictionary
        """
        # Convert config to serializable format
        config = self._convert_to_serializable(config)

        try:
            # Determine format based on file extension
            if str(config_path).endswith(".json"):
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
            elif str(config_path).endswith(".yaml") or str(config_path).endswith(
                ".yml"
            ):
                import yaml

                with open(config_path, "w") as f:
                    yaml.dump(config, f)
            else:
                # Default to JSON if format not recognized
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")


# Create a singleton instance for easy import
experiment_registry = ExperimentRegistry()

```

## File: utils/__init__.py

- Extension: .py
- Language: python
- Size: 350 bytes
- Created: 2025-04-16 16:57:48
- Modified: 2025-04-16 16:57:48

### Code

```python
from img2latex.utils.logging import configure_logging, get_logger
from img2latex.utils.mps_utils import (
    empty_cache,
    is_mps_available,
    set_device,
    set_seed,
)
from img2latex.utils.path_utils import path_manager
from img2latex.utils.registry import experiment_registry
from img2latex.utils.visualize_metrics import visualize_metrics

```

## File: utils/mps_utils.py

- Extension: .py
- Language: python
- Size: 13275 bytes
- Created: 2025-04-16 12:40:11
- Modified: 2025-04-16 12:40:11

### Code

```python
"""
Utilities for working with Apple Metal Performance Shaders (MPS) on macOS.
"""

import platform
import time
from typing import Optional, Tuple, Union

import torch

from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


def is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available.

    Returns:
        bool: True if MPS is available, False otherwise
    """
    if platform.system() != "Darwin":
        return False

    try:
        if not torch.backends.mps.is_available():
            return False
        if not torch.backends.mps.is_built():
            return False
        # Actually try to create a tensor to confirm MPS works
        torch.zeros(1).to(torch.device("mps"))
        return True
    except (AttributeError, AssertionError, RuntimeError):
        return False


def get_mps_device() -> Optional[torch.device]:
    """
    Get MPS device if available.

    Returns:
        Optional[torch.device]: MPS device if available, None otherwise
    """
    if is_mps_available():
        return torch.device("mps")
    return None


def set_device(device_name: str = None) -> torch.device:
    """
    Set the device to use for training based on availability.

    Args:
        device_name: Device to use (mps, cuda, cpu). If None, it will try to use MPS if available,
                    or fall back to CPU.

    Returns:
        torch.device: The selected device
    """
    if device_name == "mps" and is_mps_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) device")
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        if device_name in ("mps", "cuda") and device_name != "cpu":
            logger.warning(
                f"Requested device '{device_name}' is not available, falling back to CPU"
            )
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


def get_device_info(device: torch.device) -> Tuple[str, Union[str, None]]:
    """
    Get information about the device.

    Args:
        device: PyTorch device

    Returns:
        Tuple[str, Union[str, None]]: Device type and name if available
    """
    device_name = None

    if device.type == "mps":
        device_type = "MPS (Metal Performance Shaders)"
        # No direct way to get the GPU name in PyTorch for MPS
        try:
            import subprocess

            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
            )
            output = result.stdout
            for line in output.split("\n"):
                if "Chipset Model" in line:
                    device_name = line.split(":")[1].strip()
                    break
        except Exception:
            pass
    elif device.type == "cuda":
        device_type = "CUDA"
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(
                device.index if device.index else 0
            )
    else:
        device_type = "CPU"
        try:
            import platform

            device_name = platform.processor()
        except Exception:
            pass

    return device_type, device_name


def synchronize() -> None:
    """
    Wait for all MPS or CUDA operations to complete.
    Useful before timing operations or when measuring memory usage.
    """
    if is_mps_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def empty_cache(force_gc: bool = False) -> None:
    """
    Empty the MPS cache to free up memory.
    This should be called between large operations or after validation epochs.

    Args:
        force_gc: Whether to also run Python's garbage collector
    """
    if is_mps_available():
        # Empty the MPS cache
        torch.mps.empty_cache()

        # Optionally run Python's garbage collector
        if force_gc:
            import gc

            gc.collect()
            # Empty cache again after GC
            torch.mps.empty_cache()


def deep_clean_memory() -> None:
    """
    Perform a deep cleaning of memory on MPS devices.
    This function is more aggressive than empty_cache and should be used
    before/after major operations that could cause memory pressure.
    """
    if not is_mps_available():
        return

    # First run Python's garbage collector
    import gc

    gc.collect()

    # Ensure all MPS operations are completed
    torch.mps.synchronize()

    # Empty MPS cache
    torch.mps.empty_cache()

    # Short sleep to let system process the cleanup
    import time

    time.sleep(0.1)

    # Create and immediately delete tensors to help force memory reclamation
    try:
        # Create a temporary tensor to help flush memory allocations
        temp_tensor = torch.ones((1, 1), device="mps")
        del temp_tensor
    except Exception:
        pass

    # Run GC and empty cache once more
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()

    # Final GC pass
    gc.collect()


def set_manual_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set the seed for MPS and other devices.

    This is a device-specific extension of set_seed that handles
    MPS-specific seeding when available.

    Args:
        seed: Seed number to set
        deterministic: Whether to set deterministic algorithms in torch
    """
    # Set random libraries seeds
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    # Set standard torch seed
    torch.manual_seed(seed)

    # Set MPS seed if available
    if is_mps_available():
        # MPS doesn't have a separate seed setting function,
        # but we can ensure tensors are created deterministically
        # by setting the global seed and creating a test tensor
        torch.mps.manual_seed(seed)
        # Create a test tensor to ensure proper seeding
        _ = torch.randn(1, device="mps")
        synchronize()
        logger.debug(f"Set MPS random seed to {seed}")

    # Set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Set deterministic behavior if requested
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Set deterministic algorithms for ops with non-deterministic implementations
            import os

            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=True)

        logger.debug(f"Set CUDA random seed to {seed} (deterministic={deterministic})")

    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")


def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Optimize model for inference on MPS."""
    model.eval()  # Set to evaluation mode

    # Find MultiheadAttention modules and optimize them
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            # On Apple Silicon, we can use a more efficient implementation
            module._qkv_same_embed_dim = True

    return model


def batch_size_finder(
    model, input_shape=(1, 64, 512), target_shape=128, start_batch=64, device=None
):
    """Find optimal batch size for given model on Apple Silicon hardware."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    batch_size = start_batch
    optimal_batch = 1

    while batch_size >= 1:
        try:
            # Test with increasingly large batch sizes
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            dummy_target = torch.ones(
                batch_size, target_shape, dtype=torch.long, device=device
            )

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    model(dummy_input, dummy_target)

                # Time the forward pass
                torch.mps.synchronize()
                start_time = time.time()
                for _ in range(5):
                    model(dummy_input, dummy_target)
                torch.mps.synchronize()
                end_time = time.time()

            duration = end_time - start_time
            throughput = 5 * batch_size / duration

            # Success - record and try larger
            optimal_batch = batch_size
            print(f"Batch size {batch_size}: {throughput:.2f} samples/sec")
            batch_size += 16

            # Clean up memory
            torch.mps.empty_cache()

        except RuntimeError:
            # Memory error, reduce batch size
            batch_size = max(1, batch_size // 2)
            torch.mps.empty_cache()

            # If we've tried this batch size before, we're done
            if batch_size <= optimal_batch:
                break

    # Return optimal batch size with a small safety margin
    return max(1, int(optimal_batch * 0.9))


def limit_mps_memory(fraction: float):
    """Set a memory fraction limit for the MPS device.

    Args:
        fraction (float): The fraction of recommended max memory to allow (0 to disable).
                          Values between 0 and 2 are valid.
    """
    if not is_mps_available():
        logger.warning("Attempted to limit MPS memory, but MPS is not available.")
        return

    if not (0.0 <= fraction <= 2.0):
        logger.error(
            f"Invalid MPS memory fraction: {fraction}. Must be between 0.0 and 2.0."
        )
        return

    if fraction == 0.0:
        logger.info("MPS memory limit disabled (fraction set to 0).")
        # PyTorch docs say 0 means unlimited, which is the default behavior
        # We don't need to explicitly call set_per_process_memory_fraction(0)
        # unless the default behavior changes in future PyTorch versions.
        return

    try:
        # Set environment variables for improved MPS memory behavior
        import os

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        os.environ["PYTORCH_PREFER_CHANNELS_LAST"] = "1"

        # Set memory fraction limit
        torch.mps.set_per_process_memory_fraction(fraction)
        recommended_max = torch.mps.recommended_max_memory()
        limit_bytes = int(recommended_max * fraction)
        logger.info(
            f"Set MPS process memory fraction limit to {fraction:.2f}. "
            f"Estimated limit: {limit_bytes / (1024**3):.2f} GB / "
            f"{recommended_max / (1024**3):.2f} GB (Recommended Max)"
        )

        # Perform initial clean to ensure we start with a clean slate
        deep_clean_memory()

    except Exception as e:
        logger.error(f"Failed to set MPS memory fraction limit: {e}", exc_info=True)


# --- Seeding ---
def set_seed(seed: int):
    """Set random seeds for reproducibility across libraries."""
    set_manual_seed(seed)


# --- Validation Optimization ---
def optimize_for_validation(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply optimizations specifically for validation on MPS devices.
    This function configures the model for maximum validation efficiency.

    Args:
        model: The model to optimize

    Returns:
        The optimized model
    """
    model.eval()  # Ensure in evaluation mode

    # Apply contiguous memory optimizations to MultiheadAttention modules
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            module._qkv_same_embed_dim = True
            # Ensure all parameters are contiguous for faster access
            for name, param in module.named_parameters():
                if param.requires_grad:
                    param.data = param.data.contiguous()

    return model


def temporarily_quantize_model(model, dtype=torch.float16):
    """
    Temporarily convert model parameters to a more memory-efficient dtype.
    This is useful during validation to reduce memory usage.

    Args:
        model: The model to quantize
        dtype: Data type to convert to (default: torch.float16)

    Returns:
        dict: Original dtypes to restore later
    """
    original_dtypes = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                original_dtypes[name] = param.data.dtype
                param.data = param.data.to(dtype)

    return original_dtypes


def restore_model_dtypes(model, original_dtypes):
    """
    Restore model parameters to their original data types after quantization.

    Args:
        model: The model to restore
        original_dtypes: Dictionary mapping parameter names to original dtypes
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_dtypes and param.requires_grad:
                param.data = param.data.to(original_dtypes[name])

```

## File: utils/path_utils.py

- Extension: .py
- Language: python
- Size: 13297 bytes
- Created: 2025-04-16 03:30:18
- Modified: 2025-04-16 03:30:18

### Code

```python
"""
Path handling utilities for the img2latex project.

This module provides a consistent way to access project paths across
different modules and environments.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


class PathManager:
    """
    Manages paths for the img2latex project.

    This class provides a centralized way to access project paths,
    ensuring consistency across different modules and environments.
    """

    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the PathManager with project root directory.

        Args:
            root_dir: Path to the project root directory. If None, attempts to
                     determine the root directory automatically.
        """
        if root_dir is None:
            # Try to find the project root directory automatically using multiple methods
            # Start from the current module's directory
            current_dir = Path(os.path.dirname(os.path.realpath(__file__)))

            # Try to find root by walking up until we find the img2latex package
            root_candidates = []

            # Method 1: Check parent directories for img2latex directory
            temp_dir = current_dir
            for _ in range(5):  # Limit search depth to avoid infinite loops
                if (temp_dir / "img2latex").exists() or (
                    temp_dir.parent / "img2latex"
                ).exists():
                    root_candidates.append(
                        temp_dir
                        if (temp_dir / "img2latex").exists()
                        else temp_dir.parent
                    )
                    break
                temp_dir = temp_dir.parent

            # Method 2: Try to use module's package structure (utils -> img2latex -> project root)
            if current_dir.name == "utils":
                package_root = current_dir.parent.parent
                if (package_root / "img2latex").exists():
                    root_candidates.append(package_root)

            # Method 3: Try using current working directory
            cwd = Path.cwd()
            if (cwd / "img2latex").exists():
                root_candidates.append(cwd)

            # Method 4: Look for specific project markers (Makefile, pyproject.toml)
            for marker in ["Makefile", "pyproject.toml"]:
                temp_dir = current_dir
                for _ in range(5):  # Limit search depth
                    if (temp_dir / marker).exists():
                        if (temp_dir / "img2latex").exists():
                            root_candidates.append(temp_dir)
                            break
                    temp_dir = temp_dir.parent

            # Choose the first valid candidate
            if root_candidates:
                self.root_dir = root_candidates[0]
                logger.info(f"Automatically determined project root: {self.root_dir}")
            else:
                # Fallback to current directory with a warning
                self.root_dir = cwd
                logger.warning(
                    "Could not automatically determine project root directory. "
                    f"Using current working directory: {self.root_dir}"
                )
        else:
            self.root_dir = Path(root_dir)

        # Validate that we have the correct root dir
        if not (self.root_dir / "img2latex").exists():
            logger.warning(
                f"The directory {self.root_dir} does not contain an 'img2latex' directory. "
                "This might not be the correct project root."
            )

        # Define standard paths
        self.img2latex_dir = self.root_dir / "img2latex"
        self.configs_dir = self.img2latex_dir / "configs"
        self.data_dir = self.root_dir / "data"
        self.outputs_dir = self.root_dir / "outputs"
        self.registry_dir = self.outputs_dir / "registry"

        # Create registry directory
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # Registry file path
        self.experiment_registry_file = self.registry_dir / "experiment_registry.json"

        # Initialize registry file if it doesn't exist or is corrupted
        if (
            not self.experiment_registry_file.exists()
            or self.experiment_registry_file.stat().st_size == 0
        ):
            # Create empty registry file
            self._save_registry({})

    def get_experiment_dir(self, experiment_name: str) -> Path:
        """
        Get the directory for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment directory
        """
        # Check if experiment name contains a version number
        if "_v" in experiment_name:
            # Extract model name and version (e.g., "model_v2" -> "model" and "2")
            dir_name = experiment_name
        else:
            # If no version in name, use the experiment name directly with v1
            dir_name = f"{experiment_name}_v1"

        experiment_dir = self.outputs_dir / dir_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir

    def get_checkpoint_dir(self, experiment_name: str) -> Path:
        """
        Get the checkpoints directory for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment's checkpoint directory
        """
        checkpoint_dir = self.get_experiment_dir(experiment_name) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    def get_log_dir(self, experiment_name: str) -> Path:
        """
        Get the logs directory for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment's log directory
        """
        log_dir = self.get_experiment_dir(experiment_name) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def get_metrics_dir(self, experiment_name: str) -> Path:
        """
        Get the metrics directory for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment's metrics directory
        """
        metrics_dir = self.get_experiment_dir(experiment_name) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        return metrics_dir

    def get_reports_dir(self, experiment_name: str) -> Path:
        """
        Get the reports directory for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment's reports directory
        """
        reports_dir = self.get_experiment_dir(experiment_name) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        return reports_dir

    def get_plots_dir(self, experiment_name: str) -> Path:
        """
        Get the plots directory for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment's plots directory
        """
        plots_dir = self.get_experiment_dir(experiment_name) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        return plots_dir

    def get_config_path(self, experiment_name: str) -> Path:
        """
        Get the path to the config.yaml file for an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment's config.yaml file
        """
        return self.get_experiment_dir(experiment_name) / "config.yaml"

    def register_experiment(self, experiment_name: str, metadata: Dict = None) -> None:
        """
        Register a new experiment in the experiment registry.

        Args:
            experiment_name: Name of the experiment
            metadata: Additional metadata to store for the experiment
        """
        registry = self._load_registry()

        # Add or update experiment entry
        if experiment_name not in registry:
            registry[experiment_name] = {
                "creation_time": self._get_timestamp(),
                "path": str(self.get_experiment_dir(experiment_name)),
                "metadata": metadata or {},
            }
        else:
            # Update existing entry with new metadata
            registry[experiment_name]["last_updated"] = self._get_timestamp()
            if metadata:
                registry[experiment_name]["metadata"].update(metadata)

        # Save updated registry
        self._save_registry(registry)
        logger.info(f"Registered experiment: {experiment_name}")

    def get_experiment_metadata(self, experiment_name: str) -> Dict:
        """
        Get metadata for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary of experiment metadata
        """
        registry = self._load_registry()
        if experiment_name in registry:
            return registry[experiment_name]
        return {}

    def list_experiments(self) -> List[str]:
        """
        Get a list of all registered experiments.

        Returns:
            List of experiment names
        """
        registry = self._load_registry()
        return list(registry.keys())

    def _load_registry(self) -> Dict:
        """
        Load the experiment registry from disk.

        Returns:
            Dictionary containing the experiment registry
        """
        if (
            not self.experiment_registry_file.exists()
            or self.experiment_registry_file.stat().st_size == 0
        ):
            # Initialize an empty registry
            empty_registry = {}
            self._save_registry(empty_registry)
            return empty_registry

        try:
            with open(self.experiment_registry_file) as f:
                content = f.read().strip()
                if not content:  # Empty file
                    return {}
                return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Error loading experiment registry: {e}")
            logger.info("Reinitializing registry file with empty registry")
            empty_registry = {}
            self._save_registry(empty_registry)
            return empty_registry
        except Exception as e:
            logger.error(f"Unexpected error loading experiment registry: {e}")
            return {}

    def _save_registry(self, registry: Dict) -> None:
        """
        Save the experiment registry to disk.

        Args:
            registry: Dictionary containing the experiment registry
        """
        try:
            with open(self.experiment_registry_file, "w") as f:
                json.dump(registry, f, indent=2, sort_keys=True)
        except Exception as e:
            logger.error(f"Error saving experiment registry: {e}")

    def _get_timestamp(self) -> str:
        """
        Get the current timestamp as a string.

        Returns:
            Current timestamp as a string
        """
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def create_experiment_structure(self, experiment_name: str) -> Dict[str, Path]:
        """
        Create the complete directory structure for an experiment.

        This method creates all the subdirectories for an experiment and
        returns a dictionary with the paths.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary with path names as keys and Path objects as values
        """
        # Create main experiment directory
        experiment_dir = self.get_experiment_dir(experiment_name)

        # Create all subdirectories
        paths = {
            "experiment_dir": experiment_dir,
            "checkpoints_dir": self.get_checkpoint_dir(experiment_name),
            "logs_dir": self.get_log_dir(experiment_name),
            "metrics_dir": self.get_metrics_dir(experiment_name),
            "reports_dir": self.get_reports_dir(experiment_name),
            "plots_dir": self.get_plots_dir(experiment_name),
            "config_path": self.get_config_path(experiment_name),
        }

        # Register the experiment
        self.register_experiment(experiment_name)

        logger.info(f"Created directory structure for experiment: {experiment_name}")
        return paths

    def as_dict(self) -> Dict[str, Path]:
        """
        Get all project paths as a dictionary.

        Returns:
            Dictionary with path names as keys and Path objects as values
        """
        return {
            "root_dir": self.root_dir,
            "img2latex_dir": self.img2latex_dir,
            "configs_dir": self.configs_dir,
            "data_dir": self.data_dir,
            "outputs_dir": self.outputs_dir,
            "registry_dir": self.registry_dir,
        }


# Create a singleton instance for easy import
path_manager = PathManager()

```

## File: configs/config.yaml

- Extension: .yaml
- Language: yaml
- Size: 1748 bytes
- Created: 2025-04-16 22:40:45
- Modified: 2025-04-16 22:40:45

### Code

```yaml
# Configuration for IM2LaTeX project

# Data settings
data:
  data_dir: "/Users/jeremy/hmer-im2latex/data"
  train_file: "im2latex_train_filter.lst"
  validate_file: "im2latex_validate_filter.lst"
  test_file: "im2latex_test_filter.lst"
  formulas_file: "im2latex_formulas.norm.lst"
  img_dir: "img"
  batch_size: 16
  num_workers: 0
  max_seq_length: 141  # Maximum formula length (95th percentile)

# Model settings
model:
  name: "resnet_lstm"  # Options: "cnn_lstm", "resnet_lstm"
  # Encoder settings
  encoder:
    # CNN encoder settings (used when model.name = "cnn_lstm")
    cnn:
      img_height: 128
      img_width: 800
      channels: 1
      conv_filters: [32, 64, 128]
      kernel_size: 3
      pool_size: 2
      padding: "same"
    # ResNet encoder settings (used when model.name = "resnet_lstm")
    resnet:
      img_height: 64
      img_width: 800
      channels: 3
      model_name: "resnet50"  # Options: "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
      freeze_backbone: true
  # Embedding and decoder settings
  embedding_dim: 512
  decoder:
    hidden_dim: 512
    lstm_layers: 4
    dropout: 0.1
    attention: false

# Training settings
training:
  optimizer: "adam"
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 30
  early_stopping_patience: 10
  clip_grad_norm: 5.0
  save_checkpoint_epochs: 5
  experiment_name: "img2latex_v1"
  device: "mps"  # Options: "mps", "cuda", "cpu"
  accumulation_steps: 4  # Added gradient accumulation to reduce memory pressure 

# Evaluation settings
evaluation:
  metrics: ["loss", "accuracy", "bleu", "levenshtein"]
  bleu_n: 4  # n for BLEU-n score

# Logging settings
logging:
  level: "INFO"
  log_to_file: true
  log_file: "train.log"
  use_colors: true

```

## File: configs/__init__.py

- Extension: .py
- Language: python
- Size: 33 bytes
- Created: 2025-04-16 03:29:56
- Modified: 2025-04-16 03:29:56

### Code

```python
# configs package initialization

```

## File: model/decoder.py

- Extension: .py
- Language: python
- Size: 10829 bytes
- Created: 2025-04-16 22:21:33
- Modified: 2025-04-16 22:21:33

### Code

```python
"""
LSTM-based decoder for the image-to-LaTeX model.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


class LSTMDecoder(nn.Module):
    """
    LSTM-based decoder for the image-to-LaTeX model.

    This decoder processes the encoder output and generates LaTeX tokens
    one at a time using an LSTM network.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        max_seq_length: int = 141,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        attention: bool = False,
    ):
        """
        Initialize the LSTM decoder.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the input and token embeddings
            hidden_dim: Dimension of the LSTM hidden state
            max_seq_length: Maximum length of generated sequences
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            attention: Whether to use attention mechanism
        """
        super(LSTMDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.use_attention = attention

        # Embedding layer for the input tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        # Input consists of: token embedding + encoder output
        # So total input size is embedding_dim + embedding_dim = 2 * embedding_dim
        self.lstm = nn.LSTM(
            input_size=2 * embedding_dim if not attention else embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Attention mechanism
        if attention:
            self.attention = Attention(hidden_dim, embedding_dim)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        logger.info(
            f"Initialized LSTM decoder with vocab size: {vocab_size}, hidden dim: {hidden_dim}"
        )

    def forward(
        self, encoder_output: torch.Tensor, target_sequence: torch.Tensor, hidden=None
    ) -> torch.Tensor:
        """
        Forward pass through the LSTM decoder (training mode).

        Args:
            encoder_output: Output from the encoder, shape (batch_size, embedding_dim)
            target_sequence: Input token sequence, shape (batch_size, seq_length)
            hidden: Initial hidden state for the LSTM

        Returns:
            Logits for each token in the output sequence, shape (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = target_sequence.shape

        # Get token embeddings
        embedded = self.embedding(
            target_sequence
        )  # (batch_size, seq_length, embedding_dim)

        if not self.use_attention:
            # Repeat encoder output for each time step
            # Shape: (batch_size, seq_length, embedding_dim)
            encoder_output_repeated = encoder_output.unsqueeze(1).repeat(
                1, seq_length, 1
            )

            # Concatenate token embeddings with encoder output
            # Shape: (batch_size, seq_length, 2*embedding_dim)
            lstm_input = torch.cat([embedded, encoder_output_repeated], dim=2)

            # Apply dropout to the input
            lstm_input = self.dropout_layer(lstm_input)

            # Pass through LSTM
            lstm_output, hidden = self.lstm(lstm_input, hidden)

            # Apply dropout to LSTM output
            lstm_output = self.dropout_layer(lstm_output)

            # Project to vocabulary size
            # Shape: (batch_size, seq_length, vocab_size)
            output = self.output_layer(lstm_output)
        else:
            # Initialize hidden state if not provided
            if hidden is None:
                h_0 = torch.zeros(
                    self.lstm_layers,
                    batch_size,
                    self.hidden_dim,
                    device=target_sequence.device,
                )
                c_0 = torch.zeros(
                    self.lstm_layers,
                    batch_size,
                    self.hidden_dim,
                    device=target_sequence.device,
                )
                hidden = (h_0, c_0)

            # Apply dropout to the embedded tokens
            embedded = self.dropout_layer(embedded)

            # Process one time step at a time to apply attention
            outputs = []
            h, c = hidden

            for t in range(seq_length):
                # Get the current token embedding
                current_input = embedded[:, t, :].unsqueeze(
                    1
                )  # (batch_size, 1, embedding_dim)

                # Apply attention
                context = self.attention(
                    h[-1].unsqueeze(1), encoder_output.unsqueeze(1)
                )

                # Concatenate with current token embedding
                lstm_input = torch.cat([current_input, context], dim=2)

                # Pass through LSTM
                lstm_output, (h, c) = self.lstm(lstm_input, (h, c))

                # Apply dropout
                lstm_output = self.dropout_layer(lstm_output)

                # Project to vocabulary size
                output_t = self.output_layer(lstm_output)
                outputs.append(output_t)

            # Stack outputs
            output = torch.cat(outputs, dim=1)  # (batch_size, seq_length, vocab_size)

        return output

    def decode_step(
        self, encoder_output: torch.Tensor, input_token: torch.Tensor, hidden=None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single decoding step for inference.

        Args:
            encoder_output: Output from the encoder, shape (batch_size, embedding_dim)
            input_token: Input token tensor, shape (batch_size, 1)
            hidden: Hidden state from previous step

        Returns:
            Tuple of (output_logits, new_hidden_state)
        """
        batch_size = input_token.shape[0]

        # Get token embedding
        embedded = self.embedding(input_token)  # (batch_size, 1, embedding_dim)

        if not self.use_attention:
            # No Attention
            encoder_output_repeated = encoder_output.unsqueeze(1)

            # --- Sanity Check Dimensions Before Concatenation ---
            if embedded.ndim != 3 or encoder_output_repeated.ndim != 3:
                raise RuntimeError(
                    f"Shape mismatch before cat! embedded: {embedded.shape}, "
                    f"encoder_output_repeated: {encoder_output_repeated.shape}"
                )
            # ----------------------------------------------------

            lstm_input = torch.cat([embedded, encoder_output_repeated], dim=2)

            # Initialize hidden state if not provided
            if hidden is None:
                h_0 = torch.zeros(
                    self.lstm_layers,
                    batch_size,
                    self.hidden_dim,
                    device=input_token.device,
                )
                c_0 = torch.zeros(
                    self.lstm_layers,
                    batch_size,
                    self.hidden_dim,
                    device=input_token.device,
                )
                hidden = (h_0, c_0)

            # Pass through LSTM
            lstm_output, hidden = self.lstm(lstm_input, hidden)

            # Project to vocabulary size
            output = self.output_layer(lstm_output)
        else:
            # Initialize hidden state if not provided
            if hidden is None:
                h_0 = torch.zeros(
                    self.lstm_layers,
                    batch_size,
                    self.hidden_dim,
                    device=input_token.device,
                )
                c_0 = torch.zeros(
                    self.lstm_layers,
                    batch_size,
                    self.hidden_dim,
                    device=input_token.device,
                )
                hidden = (h_0, c_0)

            h, c = hidden

            # Apply attention
            context = self.attention(h[-1].unsqueeze(1), encoder_output.unsqueeze(1))

            # Concatenate with current token embedding
            lstm_input = torch.cat([embedded, context], dim=2)

            # Pass through LSTM
            lstm_output, (h, c) = self.lstm(lstm_input, (h, c))

            # Project to vocabulary size
            output = self.output_layer(lstm_output)

            hidden = (h, c)

        return output, hidden


class Attention(nn.Module):
    """
    Attention mechanism for the decoder.

    This attention layer allows the decoder to focus on different parts
    of the encoder output at each decoding step.
    """

    def __init__(self, hidden_dim: int, encoder_dim: int):
        """
        Initialize the attention layer.

        Args:
            hidden_dim: Dimension of the decoder hidden state
            encoder_dim: Dimension of the encoder output
        """
        super(Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim

        # Attention layers
        self.attn = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the attention mechanism.

        Args:
            hidden: Decoder hidden state, shape (batch_size, 1, hidden_dim)
            encoder_outputs: Encoder outputs, shape (batch_size, seq_length, encoder_dim)

        Returns:
            Context vector, shape (batch_size, 1, encoder_dim)
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state
        hidden = hidden.repeat(1, src_len, 1)

        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # Calculate attention weights
        attention = self.v(energy).squeeze(2)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention, dim=1).unsqueeze(1)

        # Weighted sum of encoder outputs
        context = torch.bmm(attention_weights, encoder_outputs)

        return context

```

## File: model/__init__.py

- Extension: .py
- Language: python
- Size: 159 bytes
- Created: 2025-04-16 03:29:56
- Modified: 2025-04-16 03:29:56

### Code

```python
from img2latex.model.decoder import LSTMDecoder
from img2latex.model.encoder import CNNEncoder, ResNetEncoder
from img2latex.model.seq2seq import Seq2SeqModel

```

## File: model/encoder.py

- Extension: .py
- Language: python
- Size: 7248 bytes
- Created: 2025-04-16 03:29:56
- Modified: 2025-04-16 03:29:56

### Code

```python
"""
CNN and ResNet-based encoders for the image-to-LaTeX model.
"""

from typing import List

import torch
import torch.nn as nn
import torchvision.models as models

from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


class CNNEncoder(nn.Module):
    """
    CNN encoder for the image-to-LaTeX model.
    
    This encoder processes input images through a series of convolutional and pooling layers,
    followed by a dense layer to produce a fixed-size encoding.
    """

    def __init__(
        self,
        img_height: int = 64,
        img_width: int = 800,
        channels: int = 1,
        conv_filters: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        pool_size: int = 2,
        padding: str = "same",
        embedding_dim: int = 256
    ):
        """
        Initialize the CNN encoder.
        
        Args:
            img_height: Height of the input images
            img_width: Width of the input images
            channels: Number of channels in the input images (1 for grayscale)
            conv_filters: List of filter sizes for each convolutional layer
            kernel_size: Size of the convolutional kernels
            pool_size: Size of the pooling windows
            padding: Type of padding for convolutional layers
            embedding_dim: Dimension of the output embedding
        """
        super(CNNEncoder, self).__init__()

        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.embedding_dim = embedding_dim

        # Determine padding value based on string specification
        padding_val = kernel_size // 2 if padding == "same" else 0

        # Create the convolutional blocks
        layers = []
        in_channels = channels

        for filters in conv_filters:
            # Add convolutional layer with ReLU activation
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    padding=padding_val
                )
            )
            layers.append(nn.ReLU())

            # Add max pooling layer
            layers.append(nn.MaxPool2d(kernel_size=pool_size))

            in_channels = filters

        self.cnn_layers = nn.Sequential(*layers)

        # Calculate the flattened size after CNN layers
        # We need to do a forward pass with a dummy tensor to calculate this
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, img_height, img_width)
            dummy_output = self.cnn_layers(dummy_input)
            flattened_size = dummy_output.numel()

        # Add a dense layer to produce the embeddings
        self.flatten = nn.Flatten()
        self.embedding_layer = nn.Linear(flattened_size, embedding_dim)
        self.activation = nn.ReLU()

        logger.info(f"Initialized CNN encoder with output dimension: {embedding_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        # Apply CNN layers
        x = self.cnn_layers(x)

        # Flatten and pass through dense layer
        x = self.flatten(x)
        x = self.embedding_layer(x)
        x = self.activation(x)

        return x


class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder for the image-to-LaTeX model.
    
    This encoder uses a pre-trained ResNet model as a feature extractor,
    followed by a dense layer to produce a fixed-size encoding.
    """

    def __init__(
        self,
        img_height: int = 64,
        img_width: int = 800,
        channels: int = 3,
        model_name: str = "resnet50",
        embedding_dim: int = 256,
        freeze_backbone: bool = True
    ):
        """
        Initialize the ResNet encoder.
        
        Args:
            img_height: Height of the input images
            img_width: Width of the input images
            channels: Number of channels in the input images (should be 3 for ResNet)
            model_name: Name of the ResNet model to use
            embedding_dim: Dimension of the output embedding
            freeze_backbone: Whether to freeze the ResNet weights
        """
        super(ResNetEncoder, self).__init__()

        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.embedding_dim = embedding_dim

        # Check that channels is 3, as ResNet expects RGB images
        if channels != 3:
            logger.warning(
                f"ResNet expects 3-channel RGB images, but got {channels} channels. "
                "You'll need to convert your images to RGB format."
            )

        # Load the pre-trained ResNet model without the classification head
        if model_name == "resnet18":
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == "resnet34":
            self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif model_name == "resnet50":
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == "resnet101":
            self.resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        elif model_name == "resnet152":
            self.resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Invalid ResNet model name: {model_name}")

        # Remove the final fully connected layer (classification head)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Freeze the ResNet weights if requested
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Add a flatten layer
        self.flatten = nn.Flatten()

        # Add a dense layer to produce the embeddings
        # ResNet feature dimensions:
        # resnet18, resnet34: 512
        # resnet50, resnet101, resnet152: 2048
        if model_name in ["resnet18", "resnet34"]:
            resnet_out_features = 512
        else:  # resnet50, resnet101, resnet152
            resnet_out_features = 2048

        self.embedding_layer = nn.Linear(resnet_out_features, embedding_dim)
        self.activation = nn.ReLU()

        logger.info(f"Initialized ResNet encoder ({model_name}) with output dimension: {embedding_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        # Pass through ResNet backbone
        x = self.resnet(x)

        # Flatten and pass through dense layer
        x = self.flatten(x)
        x = self.embedding_layer(x)
        x = self.activation(x)

        return x

```

## File: model/seq2seq.py

- Extension: .py
- Language: python
- Size: 10903 bytes
- Created: 2025-04-16 22:22:35
- Modified: 2025-04-16 22:22:35

### Code

```python
"""
Sequence-to-sequence model for image-to-LaTeX conversion.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from img2latex.model.decoder import LSTMDecoder
from img2latex.model.encoder import CNNEncoder, ResNetEncoder
from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


class Seq2SeqModel(nn.Module):
    """
    Sequence-to-sequence model for image-to-LaTeX conversion.

    This model consists of an encoder (CNN or ResNet) that processes the input image
    and a decoder (LSTM) that generates the output LaTeX sequence.
    """

    def __init__(
        self,
        model_type: str = "cnn_lstm",
        vocab_size: int = 100,
        encoder_params: Dict = None,
        decoder_params: Dict = None,
    ):
        """
        Initialize the sequence-to-sequence model.

        Args:
            model_type: Type of model to use, either "cnn_lstm" or "resnet_lstm"
            vocab_size: Size of the vocabulary
            encoder_params: Parameters for the encoder
            decoder_params: Parameters for the decoder
        """
        super(Seq2SeqModel, self).__init__()

        # Set default parameters if not provided
        if encoder_params is None:
            encoder_params = {}
        if decoder_params is None:
            decoder_params = {}

        # Get embedding dimension from encoder params
        embedding_dim = encoder_params.get("embedding_dim", 256)

        # Initialize encoder
        if model_type == "cnn_lstm":
            self.encoder = CNNEncoder(
                img_height=encoder_params.get("img_height", 50),
                img_width=encoder_params.get("img_width", 200),
                channels=encoder_params.get("channels", 1),
                conv_filters=encoder_params.get("conv_filters", [32, 64, 128]),
                kernel_size=encoder_params.get("kernel_size", 3),
                pool_size=encoder_params.get("pool_size", 2),
                padding=encoder_params.get("padding", "same"),
                embedding_dim=embedding_dim,
            )
        elif model_type == "resnet_lstm":
            self.encoder = ResNetEncoder(
                img_height=encoder_params.get("img_height", 224),
                img_width=encoder_params.get("img_width", 224),
                channels=encoder_params.get("channels", 3),
                model_name=encoder_params.get("model_name", "resnet50"),
                embedding_dim=embedding_dim,
                freeze_backbone=encoder_params.get("freeze_backbone", True),
            )
        else:
            raise ValueError(
                f"Invalid model type: {model_type}. Expected 'cnn_lstm' or 'resnet_lstm'."
            )

        # Initialize decoder
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=decoder_params.get("hidden_dim", 256),
            max_seq_length=decoder_params.get("max_seq_length", 150),
            lstm_layers=decoder_params.get("lstm_layers", 1),
            dropout=decoder_params.get("dropout", 0.1),
            attention=decoder_params.get("attention", False),
        )

        self.model_type = model_type
        self.vocab_size = vocab_size

        logger.info(f"Initialized {model_type} model with vocab size: {vocab_size}")

    def forward(
        self, images: torch.Tensor, target_sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the sequence-to-sequence model (training mode).

        Args:
            images: Input images, shape (batch_size, channels, height, width)
            target_sequences: Target LaTeX sequences, shape (batch_size, seq_length)

        Returns:
            Output logits, shape (batch_size, seq_length, vocab_size)
        """
        # Encode the images
        encoder_output = self.encoder(images)

        # Decode the sequences
        decoder_output = self.decoder(
            encoder_output=encoder_output,
            target_sequence=target_sequences[
                :, :-1
            ],  # Exclude the last token (end token)
        )

        return decoder_output

    def inference(
        self,
        image: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int = 150,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        beam_size: int = 0,
    ) -> List[int]:
        """
        Generate a LaTeX sequence for an input image (inference mode).

        Args:
            image: Input image, shape (batch_size, channels, height, width)
            start_token_id: ID of the start token
            end_token_id: ID of the end token
            max_length: Maximum length of the generated sequence
            temperature: Softmax temperature (higher values produce more diverse outputs)
            top_k: If > 0, only sample from the top k most probable tokens
            top_p: If > 0.0, only sample from the top tokens with cumulative probability >= top_p
            beam_size: If > 0, use beam search with the specified beam size

        Returns:
            List of token IDs for the generated sequence
        """
        # Encode the image
        encoder_output = self.encoder(image)

        # Handle batch size 1 case for inference
        if encoder_output.dim() == 1:
            encoder_output = encoder_output.unsqueeze(0)

        batch_size = encoder_output.shape[0]
        device = encoder_output.device

        if beam_size > 0:
            return self._beam_search(
                encoder_output=encoder_output,
                start_token_id=start_token_id,
                end_token_id=end_token_id,
                max_length=max_length,
                beam_size=beam_size,
            )
        else:
            return self._greedy_search(
                encoder_output=encoder_output,
                start_token_id=start_token_id,
                end_token_id=end_token_id,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

    # --- This method seems unused / incorrect for batch processing ---
    # --- Reverting to a simple single-step call ---
    # def _greedy_search(...): ...

    def _beam_search(
        self,
        encoder_output: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int,
        beam_size: int,
    ) -> List[int]:
        """
        Perform beam search decoding.

        Args:
            encoder_output: Output from the encoder
            start_token_id: ID of the start token
            end_token_id: ID of the end token
            max_length: Maximum length of the generated sequence
            beam_size: Beam size

        Returns:
            List of token IDs for the best sequence found
        """
        device = encoder_output.device
        batch_size = encoder_output.shape[0]

        # Repeat encoder output for each beam
        encoder_output = encoder_output.repeat_interleave(beam_size, dim=0)

        # Start with the start token
        input_token = torch.tensor([[start_token_id]], device=device)

        # Initialize sequences, scores, and finished flags
        sequences = [[start_token_id]]
        sequence_scores = torch.zeros(1, device=device)
        finished_sequences = []
        finished_scores = []

        # Initialize the hidden state
        hidden = None

        # Generate tokens one by one
        for step in range(max_length):
            # Get the number of active sequences
            num_active = len(sequences)

            # Prepare the input for the decoder
            input_tokens = torch.tensor([[seq[-1]] for seq in sequences], device=device)

            # Get the next token probabilities
            output, hidden = self.decoder.decode_step(
                encoder_output[:num_active], input_tokens, hidden
            )
            logits = output.squeeze(1)

            # Convert to log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)

            # Add the log probabilities to the sequence scores
            if step == 0:
                # For the first step, we only have one sequence
                scores = log_probs[0]
            else:
                # For subsequent steps, we have multiple sequences
                scores = sequence_scores.unsqueeze(1) + log_probs
                scores = scores.view(-1)

            # Select the top-k sequences
            if step == 0:
                top_k_scores, top_k_tokens = torch.topk(scores, beam_size)
                beam_indices = torch.zeros(beam_size, device=device, dtype=torch.long)
            else:
                top_k_scores, top_k_indices = torch.topk(
                    scores, beam_size - len(finished_sequences)
                )
                top_k_tokens = top_k_indices % self.vocab_size
                beam_indices = top_k_indices // self.vocab_size

            # Create new sequences
            new_sequences = []
            new_scores = []
            new_hidden = (
                (hidden[0][:, beam_indices, :], hidden[1][:, beam_indices, :])
                if hidden is not None
                else None
            )

            for i, (token, beam_idx) in enumerate(zip(top_k_tokens, beam_indices)):
                token_item = token.item()
                score = top_k_scores[i].item()

                # Get the sequence corresponding to the beam index
                sequence = sequences[beam_idx]

                # Check if this sequence has finished
                if token_item == end_token_id:
                    finished_sequences.append(sequence + [token_item])
                    finished_scores.append(score)
                else:
                    new_sequences.append(sequence + [token_item])
                    new_scores.append(score)

            # Break if all sequences have finished or we've reached the maximum length
            if len(finished_sequences) >= beam_size or len(new_sequences) == 0:
                break

            # Update sequences, scores, and hidden state
            sequences = new_sequences
            sequence_scores = torch.tensor(new_scores, device=device)
            hidden = new_hidden

        # If we don't have any finished sequences, use the active ones
        if len(finished_sequences) == 0:
            finished_sequences = sequences
            finished_scores = sequence_scores.tolist()

        # Sort the finished sequences by score (higher is better)
        sorted_sequences = [
            seq
            for _, seq in sorted(
                zip(finished_scores, finished_sequences),
                key=lambda x: x[0],
                reverse=True,
            )
        ]

        # Return the best sequence
        return sorted_sequences[0]

```

## File: data/transforms.py

- Extension: .py
- Language: python
- Size: 3375 bytes
- Created: 2025-04-16 03:30:37
- Modified: 2025-04-16 03:30:37

### Code

```python
"""
Custom transforms for image preprocessing.
"""

import torchvision.transforms.functional as TF
from PIL import Image


class ResizeWithAspectRatio:
    """
    Resize image to target height while maintaining aspect ratio,
    then pad OR CROP to target width. Ensures exact output dimensions.

    Operates on PIL Images. Picklable for multiprocessing.
    """

    def __init__(self, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width
        # Determine resampling filter based on Pillow version
        try:
            self.resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            self.resample_filter = Image.LANCZOS  # Fallback for older Pillow

    def __call__(self, img):
        width, height = img.size
        if height == 0:  # Avoid division by zero
            return Image.new(img.mode, (self.target_width, self.target_height), 255)

        # Calculate width needed to preserve aspect ratio for target height
        aspect_ratio = width / height
        new_width = int(
            round(self.target_height * aspect_ratio)
        )  # Use round for possibly better results

        # Resize height first
        img_resized = img.resize((new_width, self.target_height), self.resample_filter)

        # Now ensure width is exactly target_width
        if new_width == self.target_width:
            return img_resized
        elif new_width < self.target_width:
            # Pad width (right padding with white)
            padded_img = Image.new(
                img.mode, (self.target_width, self.target_height), 255
            )  # White padding
            padded_img.paste(img_resized, (0, 0))  # Paste resized image at top-left
            return padded_img
        else:  # new_width > self.target_width
            # Crop width to target width (center crop is often better than left crop)
            left = (new_width - self.target_width) // 2
            right = left + self.target_width
            # Box is (left, upper, right, lower)
            img_cropped = img_resized.crop((left, 0, right, self.target_height))
            return img_cropped


class PadTensorTransform:
    """
    Pads the width of a Tensor [C, H, W] to target_width OR crops if wider.
    Ensures output tensor width is exactly target_width. Picklable.
    """

    def __init__(self, target_width, padding_value=0.0):
        self.target_width = target_width
        self.padding_value = padding_value

    def __call__(self, img_tensor):
        if img_tensor.dim() != 3:
            raise ValueError(
                f"PadTensorTransform expects 3D tensor [C,H,W], got {img_tensor.dim()}D"
            )

        C, H, current_width = img_tensor.shape
        padding_needed = self.target_width - current_width

        if padding_needed == 0:  # Already correct width
            return img_tensor
        elif padding_needed > 0:  # Needs padding
            # Padding format: (left, right, top, bottom) - Pad only right side
            padding = (0, padding_needed, 0, 0)
            return TF.pad(
                img_tensor, padding, fill=self.padding_value, padding_mode="constant"
            )
        else:  # Needs cropping (padding_needed < 0)
            # Crop from the right side to target_width
            return img_tensor[:, :, : self.target_width]

```

## File: data/__init__.py

- Extension: .py
- Language: python
- Size: 209 bytes
- Created: 2025-04-16 03:29:56
- Modified: 2025-04-16 03:29:56

### Code

```python
from img2latex.data.dataset import Im2LatexDataset, create_data_loaders
from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.data.utils import batch_convert_for_resnet, load_image, prepare_batch

```

## File: data/dataset.py

- Extension: .py
- Language: python
- Size: 22583 bytes
- Created: 2025-04-16 03:30:33
- Modified: 2025-04-16 03:30:33

### Code

```python
"""
Memory-optimized dataset classes and data loading utilities for the image-to-LaTeX model.
Handles loading images and formulas, applying necessary preprocessing,
and preparing batches for training and evaluation.

Features:
- Option to preload images into memory for faster access
- Optimized DataLoader configurations for different worker counts
- Memory-efficient processing options
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.data.transforms import (
    ResizeWithAspectRatio,
)  # Import custom transform classes
from img2latex.utils.logging import get_logger  # Relative import for logging

logger = get_logger(__name__)


# --- Define Collator Class at Top Level (for pickling) ---
class Im2LatexCollator:
    """
    Collator function for the DataLoader.

    Pads formula sequences to the maximum length in the batch and stacks images.
    """

    def __init__(self, pad_token_id: int):
        """
        Initialize the collator.

        Args:
            pad_token_id: The token ID used for padding sequences.
        """
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate a list of samples into a batch.

        Args:
            batch: A list of dictionaries, each from Im2LatexDataset.__getitem__.

        Returns:
            A dictionary containing batched data.
        """
        # Stack image tensors (they are already preprocessed to the same size)
        images = torch.stack([item["image"] for item in batch])
        formulas = [item["formula"] for item in batch]

        # Pad formula sequences to the maximum length within this batch
        max_len = max(len(formula) for formula in formulas)
        # Use torch.full for efficient padding
        padded_formulas = torch.full(
            (len(batch), max_len), self.pad_token_id, dtype=torch.long
        )
        for i, formula in enumerate(formulas):
            padded_formulas[i, : len(formula)] = formula

        # Return the collated batch
        return {
            "images": images,
            "formulas": padded_formulas,
            "raw_formulas": [item["raw_formula"] for item in batch],
            "image_paths": [item["image_path"] for item in batch],
            "formula_idxs": [item["formula_idx"] for item in batch],
        }


# --- End of Collator Class Definition ---


class Im2LatexDataset(Dataset):
    """
    PyTorch Dataset for the Im2LaTeX data.

    Loads images from the 'img' directory and corresponding
    formulas, applying preprocessing suitable for CNN-LSTM or ResNet-LSTM models.

    Features:
    - Optional in-memory caching of images for faster access
    - Efficient preprocessing pipeline
    - Robust error handling for missing files
    """

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        formulas_file: str,
        tokenizer: LaTeXTokenizer,
        img_dir: str = "img",  # Default to the processed images dir
        img_size: Tuple[int, int] = (64, 800),  # Default to final processed size
        channels: int = 1,  # Default based on CNN model
        transform=None,
        max_samples: Optional[int] = None,
        load_in_memory: bool = False,  # Whether to preload all images into memory
    ):
        """
        Initialize the Im2LatexDataset.

        Args:
            data_dir: Path to the main data directory (e.g., './data').
            split_file: Name of the file containing image names and formula indices
                        for this split (e.g., 'im2latex_train_filter.lst').
            formulas_file: Name of the file containing normalized formulas
                           (e.g., 'im2latex_formulas.norm.lst').
            tokenizer: Initialized LaTeXTokenizer instance.
            img_dir: Name of the directory containing the actual image files
                     (relative to data_dir). Should be 'img'.
            img_size: Target image size AFTER processing (height, width).
                      Should be (64, 800) based on analysis.
            channels: Number of channels for the model input (1 for CNN, 3 for ResNet).
            transform: Optional torchvision transforms pipeline. If None, a default
                       pipeline based on channels and img_size will be created.
            max_samples: Maximum number of samples to load (for debugging/testing).
            load_in_memory: If True, preload all images into memory for faster access.
                           Caution: This can use significant RAM for large datasets.
        """
        self.data_dir = Path(data_dir)
        self.img_base_dir = self.data_dir / img_dir  # Base directory for images
        self.split_file_path = self.data_dir / split_file
        self.formulas_file_path = self.data_dir / formulas_file
        self.tokenizer = tokenizer
        self.target_height = img_size[0]
        self.target_width = img_size[1]
        self.channels = channels
        self.transform = transform
        self.max_samples = max_samples
        self.load_in_memory = load_in_memory
        self.preloaded_images = {}  # Dict to store preloaded images

        # Validate paths
        if not self.split_file_path.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file_path}")
        if not self.formulas_file_path.exists():
            raise FileNotFoundError(
                f"Formulas file not found: {self.formulas_file_path}"
            )
        if not self.img_base_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_base_dir}")

        # Load formulas and samples
        self.formulas: List[str] = []  # Pre-load all formulas
        self.samples: List[Dict] = (
            self._load_data()
        )  # Load image paths and formula indices

        # Setup transforms if not provided
        if self.transform is None:
            self.transform = self._create_default_transforms()

        # Check available memory if loading into memory
        if self.load_in_memory:
            # Estimate memory usage: ~100KB per image on average
            estimated_memory_mb = len(self.samples) * 0.1  # MB
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)

            logger.info(
                f"Estimated memory for image preloading: {estimated_memory_mb:.2f} MB"
            )
            logger.info(f"Available system memory: {available_memory_mb:.2f} MB")

            if (
                estimated_memory_mb > available_memory_mb * 0.5
            ):  # If using more than 50% of available
                logger.warning(
                    f"Loading {len(self.samples)} images may use {estimated_memory_mb:.2f} MB "
                    f"of RAM. Consider using load_in_memory=False for the full dataset."
                )

                # Ask for confirmation if in interactive mode
                if hasattr(sys, "ps1"):  # Check if running interactively
                    confirmation = input(
                        f"Continue with loading {len(self.samples)} images into memory? (y/n): "
                    )
                    if confirmation.lower() != "y":
                        logger.info("Disabling in-memory loading based on user input.")
                        self.load_in_memory = False

        # Preload images if requested and not disabled by user
        if self.load_in_memory:
            logger.info(f"Preloading {len(self.samples)} images into memory...")
            for i, sample in enumerate(self.samples):
                if i % 1000 == 0 and i > 0:
                    logger.info(f"Preloaded {i}/{len(self.samples)} images...")
                image_filename = sample["image_filename"]
                # Only load the PIL image here, transformations will be applied at __getitem__ time
                self.preloaded_images[image_filename] = self._load_image(image_filename)
            logger.info(f"Finished preloading {len(self.preloaded_images)} images.")

        logger.info(
            f"Initialized Im2LatexDataset ({split_file}) with {len(self.samples)} samples. "
            f"Target image size: {self.target_height}x{self.target_width}, channels: {self.channels}, "
            f"load_in_memory: {self.load_in_memory}"
        )

    def _load_data(self) -> List[Dict]:
        """Loads formula strings and the list of samples for the current split."""
        # Load all formula strings into memory
        try:
            with open(self.formulas_file_path, "r", encoding="utf-8") as f:
                self.formulas = [line.strip() for line in f]
            logger.info(
                f"Loaded {len(self.formulas)} formulas from {self.formulas_file_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load formulas file: {e}")
            raise

        # Load image filenames and corresponding formula indices for this split
        samples = []
        try:
            with open(self.split_file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        image_filename = parts[0]
                        try:
                            formula_idx = int(parts[1])
                            if 0 <= formula_idx < len(self.formulas):
                                samples.append(
                                    {
                                        "image_filename": image_filename,
                                        "formula_idx": formula_idx,
                                    }
                                )
                            else:
                                logger.warning(
                                    f"Line {line_num + 1}: Formula index {formula_idx} out of range (0-{len(self.formulas) - 1}). Skipping."
                                )
                        except ValueError:
                            logger.warning(
                                f"Line {line_num + 1}: Invalid formula index '{parts[1]}'. Skipping."
                            )
                    else:
                        logger.warning(
                            f"Line {line_num + 1}: Invalid format '{line.strip()}'. Skipping."
                        )
        except Exception as e:
            logger.error(f"Failed to load split file {self.split_file_path}: {e}")
            raise

        # Apply max_samples limit if specified
        if self.max_samples is not None and self.max_samples > 0:
            samples = samples[: self.max_samples]
            logger.info(f"Limited dataset to {len(samples)} samples.")

        logger.info(
            f"Loaded {len(samples)} samples referencing images in {self.img_base_dir}"
        )
        return samples

    def _create_default_transforms(self) -> transforms.Compose:
        """Creates the default preprocessing pipeline based on target size and channels."""
        transform_list = []

        # 1. Resize height while maintaining aspect ratio (operates on PIL Image)
        transform_list.append(
            ResizeWithAspectRatio(  # Use the CLASS from transforms.py
                target_height=self.target_height, target_width=self.target_width
            )
        )

        # 2. Conditional Grayscale Conversion (operates on PIL Image)
        if self.channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))

        # 3. Convert PIL Image to PyTorch Tensor (scales pixels to [0, 1])
        transform_list.append(transforms.ToTensor())

        # 4. Apply Normalization (PadTensorTransform is NOT needed if ResizeWithAspectRatio works correctly)
        if self.channels == 1:
            # Normalize to [-1, 1] for grayscale
            normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        else:  # RGB
            # Apply ImageNet normalization for RGB
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        # 5. Apply Normalization
        transform_list.append(normalize)

        return transforms.Compose(transform_list)

    def _load_image(self, image_filename: str) -> Image.Image:
        """Loads a single image using PIL."""
        full_path = self.img_base_dir / image_filename
        try:
            img = Image.open(full_path)
            # Ensure image is loaded as RGB initially, grayscale conversion happens in transform if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except FileNotFoundError:
            logger.error(f"Image file not found: {full_path}")
        except Exception as e:
            logger.error(f"Error loading image {full_path}: {e}")

        # Return a blank white image on error
        logger.warning(f"Returning blank image for {image_filename}")
        return Image.new(
            "RGB", (100, self.target_height), (255, 255, 255)
        )  # Placeholder size

    def _load_formula(self, idx: int) -> str:
        """Retrieves a formula string from the pre-loaded list."""
        if 0 <= idx < len(self.formulas):
            return self.formulas[idx]
        else:
            logger.error(f"Invalid formula index requested: {idx}")
            return ""  # Return empty string on error

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieves the processed image and tokenized formula for a given index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary containing:
                'image': The preprocessed image tensor.
                'formula': The tokenized formula tensor (including START/END).
                'raw_formula': The original formula string.
                'image_path': The filename of the image.
                'formula_idx': The index of the formula.
        """
        if idx >= len(self.samples):
            raise IndexError("Index out of range")

        sample = self.samples[idx]
        image_filename = sample["image_filename"]
        formula_idx = sample["formula_idx"]

        # Load image (from memory if preloaded, or from disk)
        if self.load_in_memory and image_filename in self.preloaded_images:
            image = self.preloaded_images[image_filename]
        else:
            image = self._load_image(image_filename)

        # Apply transforms (resize, pad, grayscale (cond.), ToTensor, normalize)
        image_tensor = self.transform(image)

        # Load formula
        formula_str = self._load_formula(formula_idx)

        # Tokenize formula (add START/END tokens)
        formula_with_tokens = f"{self.tokenizer.special_tokens['START']} {formula_str} {self.tokenizer.special_tokens['END']}"
        formula_ids = self.tokenizer.encode(formula_with_tokens)
        formula_tensor = torch.tensor(formula_ids, dtype=torch.long)

        return {
            "image": image_tensor,
            "formula": formula_tensor,
            "raw_formula": formula_str,
            "image_path": image_filename,  # Store filename for reference
            "formula_idx": formula_idx,
        }

    def clear_memory_cache(self):
        """Clears the preloaded images from memory to free up RAM."""
        if hasattr(self, "preloaded_images") and self.preloaded_images:
            logger.info(
                f"Clearing {len(self.preloaded_images)} preloaded images from memory"
            )
            self.preloaded_images.clear()
            # Force garbage collection
            import gc

            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


def create_data_loaders(
    data_dir: str,
    tokenizer: LaTeXTokenizer,
    model_type: str = "cnn_lstm",
    batch_size: int = 128,
    num_workers: int = 0,  # Default to 0 based on performance testing
    cnn_img_size: Tuple[int, int] = (64, 800),  # Final processed size
    resnet_img_size: Tuple[int, int] = (64, 800),  # Final processed size
    max_samples: Optional[Dict[str, Optional[int]]] = None,  # Allow None per split
    load_in_memory: bool = False,  # Whether to preload images into memory
    prefetch_factor: int = 2,  # Number of batches to prefetch (default is 2)
    persistent_workers: Optional[
        bool
    ] = None,  # Whether to keep workers alive between iterations
    pin_memory: Optional[bool] = None,  # Whether to pin memory in CUDA
) -> Dict[str, DataLoader]:
    """
    Creates DataLoaders for train, validation, and test sets with memory optimization options.

    Args:
        data_dir: Path to the data directory.
        tokenizer: Initialized LaTeXTokenizer.
        model_type: 'cnn_lstm' or 'resnet_lstm'. Determines channels.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.
        cnn_img_size: Target (height, width) for CNN model inputs.
        resnet_img_size: Target (height, width) for ResNet model inputs.
        max_samples: Optional dictionary limiting samples per split (e.g., {"train": 1000}).
        load_in_memory: If True, preload all images into memory for faster access.
                        Useful for small datasets or when sufficient RAM is available.
        prefetch_factor: Number of batches loaded in advance by each worker.
                         Only used when num_workers > 0.
        persistent_workers: If True, keep worker processes alive between DataLoader iterations.
                           Default: True if num_workers > 0, else not applicable.
        pin_memory: If True, pin memory in CUDA.
                   Default: True if CUDA is available.

    Returns:
        A dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    if model_type == "cnn_lstm":
        img_size = cnn_img_size
        channels = 1
    elif model_type == "resnet_lstm":
        img_size = resnet_img_size
        channels = 3
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    if max_samples is None:
        max_samples = {"train": None, "val": None, "test": None}

    # Set defaults for optional parameters
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Instantiate the picklable collator class
    collator = Im2LatexCollator(pad_token_id=tokenizer.pad_token_id)

    loaders = {}
    splits = {
        "train": "im2latex_train_filter.lst",
        "val": "im2latex_validate_filter.lst",
        "test": "im2latex_test_filter.lst",
    }

    # Use the correct image directory name
    image_directory = "img"

    # Log memory optimization settings
    logger.info(
        f"Creating DataLoaders with: num_workers={num_workers}, load_in_memory={load_in_memory}, "
        f"prefetch_factor={prefetch_factor}, persistent_workers={persistent_workers}, "
        f"pin_memory={pin_memory}"
    )

    for split_name, split_filename in splits.items():
        # Determine whether to load this split in memory
        # For training splits with many samples, it may be impractical
        split_load_in_memory = load_in_memory
        if (
            split_name == "train"
            and max_samples.get(split_name) is None
            and load_in_memory
        ):
            logger.warning(
                "Loading full training set in memory may require substantial RAM. "
                "Consider using load_in_memory=False or setting max_samples for training."
            )

        dataset = Im2LatexDataset(
            data_dir=data_dir,
            split_file=split_filename,
            formulas_file="im2latex_formulas.norm.lst",
            tokenizer=tokenizer,
            img_dir=image_directory,  # Use the correct directory
            img_size=img_size,
            channels=channels,
            max_samples=max_samples.get(split_name),  # Get limit for this split
            load_in_memory=split_load_in_memory,  # Whether to preload images
            # transform=None will use the default created inside Im2LatexDataset
        )

        # Check if dataset is empty (e.g., due to file issues or max_samples=0)
        if len(dataset) == 0:
            logger.warning(
                f"Dataset for split '{split_name}' is empty. Skipping DataLoader creation."
            )
            loaders[split_name] = None  # Or an empty loader if preferred
            continue

        # Optimized DataLoader settings based on testing:
        # - For num_workers=0, no extra settings needed
        # - For num_workers>0, configure for best performance
        if num_workers > 0:
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == "train"),  # Shuffle only training data
                num_workers=num_workers,
                collate_fn=collator,  # Use the collator instance
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
        else:
            # For num_workers=0, simpler configuration
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == "train"),  # Shuffle only training data
                num_workers=0,
                collate_fn=collator,  # Use the collator instance
                pin_memory=pin_memory,
            )

        logger.info(
            f"Created DataLoader for '{split_name}' with {len(dataset)} samples."
        )

    # Log info about the data loaders
    train_count = len(loaders.get("train").dataset) if loaders.get("train") else 0
    val_count = len(loaders.get("val").dataset) if loaders.get("val") else 0
    test_count = len(loaders.get("test").dataset) if loaders.get("test") else 0

    logger.info(
        f"Created data loaders for model type {model_type}: "
        f"train={train_count} samples, val={val_count} samples, test={test_count} samples"
    )

    return loaders

```

## File: data/tokenizer.py

- Extension: .py
- Language: python
- Size: 11600 bytes
- Created: 2025-04-16 03:30:35
- Modified: 2025-04-16 03:30:35

### Code

```python
"""
Tokenizer for LaTeX formulas.
"""

import os
from collections import Counter
from typing import Dict, List

import torch

from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


class LaTeXTokenizer:
    """
    Tokenizer for LaTeX formulas.

    This tokenizer handles the conversion between LaTeX formula strings
    and token sequences, including handling of special tokens.
    """

    def __init__(
        self, special_tokens: Dict[str, str] = None, max_sequence_length: int = 150
    ):
        """
        Initialize the tokenizer.

        Args:
            special_tokens: Dictionary of special tokens
            max_sequence_length: Maximum sequence length
        """
        # Set default special tokens if not provided
        if special_tokens is None:
            self.special_tokens = {
                "PAD": "<PAD>",
                "START": "<START>",
                "END": "<END>",
                "UNK": "<UNK>",
            }
        else:
            self.special_tokens = special_tokens

        # Initialize vocabulary-related attributes
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.max_sequence_length = max_sequence_length

        # Initialize with special tokens
        self._init_special_tokens()

        logger.info(
            f"Initialized LaTeX tokenizer with max sequence length: {max_sequence_length}"
        )

    def _init_special_tokens(self) -> None:
        """Initialize the vocabulary with special tokens."""
        self.token_to_id = {}
        self.id_to_token = {}

        # Add special tokens to the vocabulary
        for idx, token in enumerate(self.special_tokens.values()):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        self.vocab_size = len(self.token_to_id)

        # Save the indices of special tokens for easy access
        self.pad_token_id = self.token_to_id[self.special_tokens["PAD"]]
        self.start_token_id = self.token_to_id[self.special_tokens["START"]]
        self.end_token_id = self.token_to_id[self.special_tokens["END"]]
        self.unk_token_id = self.token_to_id[self.special_tokens["UNK"]]

    def fit(self, texts: List[str]) -> None:
        """
        Fit the tokenizer on a list of LaTeX formula texts.

        Args:
            texts: List of LaTeX formula texts
        """
        # Reset vocabulary
        self._init_special_tokens()

        # Count token frequencies
        counter = Counter()
        for text in texts:
            tokens = text.split()
            counter.update(tokens)

        # Sort tokens by frequency (descending)
        sorted_tokens = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # Add tokens to vocabulary (skipping those already in vocabulary)
        for token, _ in sorted_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = self.vocab_size
                self.id_to_token[self.vocab_size] = token
                self.vocab_size += 1

        # Check if max sequence length is sufficient
        max_found_length = max(len(text.split()) for text in texts)
        if max_found_length > self.max_sequence_length:
            logger.warning(
                f"Found sequences of length {max_found_length}, "
                f"which is longer than max_sequence_length ({self.max_sequence_length}). "
                "Consider increasing max_sequence_length."
            )

        logger.info(
            f"Fitted tokenizer on {len(texts)} texts. Vocabulary size: {self.vocab_size}"
        )

    def fit_on_formulas_file(self, file_path: str) -> None:
        """
        Fit the tokenizer on a formulas file.

        Args:
            file_path: Path to the formulas file
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Formulas file not found: {file_path}")

        # Read formulas from file
        with open(file_path, "r", encoding="utf-8") as f:
            formulas = [line.strip() for line in f]

        # Add special tokens to formulas
        processed_formulas = [
            f"{self.special_tokens['START']} {formula} {self.special_tokens['END']}"
            for formula in formulas
        ]

        # Fit tokenizer on processed formulas
        self.fit(processed_formulas)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Convert a LaTeX formula text to a sequence of token IDs.

        Args:
            text: LaTeX formula text
            add_special_tokens: Whether to add START and END tokens

        Returns:
            List of token IDs
        """
        # Add special tokens if requested
        if add_special_tokens:
            text = f"{self.special_tokens['START']} {text} {self.special_tokens['END']}"

        # Split into tokens
        tokens = text.split()

        # Convert tokens to IDs
        ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert a sequence of token IDs to a LaTeX formula text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in the output

        Returns:
            LaTeX formula text
        """
        # Get special token IDs
        special_token_ids = (
            [self.token_to_id[token] for token in self.special_tokens.values()]
            if skip_special_tokens
            else []
        )

        # Convert IDs to tokens, skipping special tokens if requested
        tokens = [
            self.id_to_token.get(idx, self.special_tokens["UNK"])
            for idx in ids
            if idx not in special_token_ids or not skip_special_tokens
        ]

        # Join tokens into a string
        text = " ".join(tokens)

        return text

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = False,
        padding: bool = True,
        truncation: bool = True,
    ) -> torch.Tensor:
        """
        Convert a batch of LaTeX formula texts to a batch of token ID sequences.

        Args:
            texts: List of LaTeX formula texts
            add_special_tokens: Whether to add START and END tokens
            padding: Whether to pad sequences to max_sequence_length
            truncation: Whether to truncate sequences longer than max_sequence_length

        Returns:
            Tensor of token IDs of shape (batch_size, seq_length)
        """
        # Encode each text
        encoded_texts = [self.encode(text, add_special_tokens) for text in texts]

        # Truncate if needed
        if truncation:
            encoded_texts = [ids[: self.max_sequence_length] for ids in encoded_texts]

        # Pad if needed
        if padding:
            encoded_texts = [
                ids + [self.pad_token_id] * (self.max_sequence_length - len(ids))
                for ids in encoded_texts
            ]

        # Convert to tensor
        batch_tensor = torch.tensor(encoded_texts)

        return batch_tensor

    def decode_batch(
        self, batch_ids: torch.Tensor, skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Convert a batch of token ID sequences to a batch of LaTeX formula texts.

        Args:
            batch_ids: Tensor of token IDs of shape (batch_size, seq_length)
            skip_special_tokens: Whether to skip special tokens in the output

        Returns:
            List of LaTeX formula texts
        """
        # Convert tensor to list of lists
        batch_ids_list = batch_ids.tolist()

        # Decode each sequence
        decoded_texts = [
            self.decode(ids, skip_special_tokens) for ids in batch_ids_list
        ]

        return decoded_texts

    def save(self, file_path: str) -> None:
        """
        Save the tokenizer vocabulary to a file.

        Args:
            file_path: Path to save the vocabulary to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save vocabulary (token -> id)
        save_dict = {
            "token_to_id": self.token_to_id,
            "special_tokens": self.special_tokens,
            "max_sequence_length": self.max_sequence_length,
        }

        torch.save(save_dict, file_path)
        logger.info(f"Saved tokenizer to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "LaTeXTokenizer":
        """
        Load a tokenizer from a file.

        Args:
            file_path: Path to load the vocabulary from

        Returns:
            Loaded tokenizer
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tokenizer file not found: {file_path}")

        # Load vocabulary
        save_dict = torch.load(file_path)

        # Create tokenizer
        tokenizer = cls(
            special_tokens=save_dict["special_tokens"],
            max_sequence_length=save_dict["max_sequence_length"],
        )

        # Restore vocabulary
        tokenizer.token_to_id = save_dict["token_to_id"]
        tokenizer.vocab_size = len(tokenizer.token_to_id)

        # Rebuild id_to_token mapping
        tokenizer.id_to_token = {
            idx: token for token, idx in tokenizer.token_to_id.items()
        }

        # Set special token IDs
        tokenizer.pad_token_id = tokenizer.token_to_id[tokenizer.special_tokens["PAD"]]
        tokenizer.start_token_id = tokenizer.token_to_id[
            tokenizer.special_tokens["START"]
        ]
        tokenizer.end_token_id = tokenizer.token_to_id[tokenizer.special_tokens["END"]]
        tokenizer.unk_token_id = tokenizer.token_to_id[tokenizer.special_tokens["UNK"]]

        logger.info(
            f"Loaded tokenizer from {file_path} with vocabulary size: {tokenizer.vocab_size}"
        )
        return tokenizer

    def default_init(self):
        """
        Initialize the tokenizer with a minimal vocabulary for testing.
        """
        # Start with special tokens
        self._init_special_tokens()

        # Add some common LaTeX tokens
        common_tokens = [
            "+",
            "-",
            "=",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "\\frac",
            "\\sum",
            "\\int",
            "a",
            "b",
            "c",
            "x",
            "y",
            "z",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "\\alpha",
            "\\beta",
            "\\gamma",
            "\\delta",
            "\\theta",
            "\\pi",
            "\\sigma",
            "\\mathbf",
            "\\mathrm",
            "\\mathcal",
            "\\limits",
            "_",
            "^",
            "\\infty",
        ]

        # Add common tokens to vocabulary
        for token in common_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = self.vocab_size
                self.id_to_token[self.vocab_size] = token
                self.vocab_size += 1

        logger.info(
            f"Initialized tokenizer with default vocabulary. Size: {self.vocab_size}"
        )

```

## File: data/utils.py

- Extension: .py
- Language: python
- Size: 6972 bytes
- Created: 2025-04-16 03:30:39
- Modified: 2025-04-16 03:30:39

### Code

```python
"""
Utility functions for data processing.
"""

import os
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image

from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


import torchvision.transforms.functional as TF


def resize_maintain_aspect(img: Image.Image, target_height: int) -> Image.Image:
    """
    Resizes image to target_height while maintaining aspect ratio.

    Args:
        img: PIL Image
        target_height: Target height in pixels

    Returns:
        Resized PIL Image with exactly target_height height
    """
    width, height = img.size

    # Ensure we're resizing to the exact target height
    if height != target_height:
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)

        # Use LANCZOS for high quality resizing
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            # For older versions of PIL
            resample = Image.LANCZOS

        img = img.resize((new_width, target_height), resample)

    return img


def pad_image_width(
    img_tensor: torch.Tensor, target_width: int, padding_value: float = 0.0
) -> torch.Tensor:
    """
    Pads the width of an image tensor to target_width.

    Args:
        img_tensor: Image tensor of shape [C, H, W]
        target_width: Target width in pixels
        padding_value: Value to use for padding

    Returns:
        Padded tensor of shape [C, H, target_width]
    """
    # Check if tensor has expected dimensions
    if img_tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor [C,H,W], got {img_tensor.dim()}D")

    _, _, current_width = img_tensor.shape
    padding_needed = target_width - current_width

    # Return original tensor if no padding needed
    if padding_needed <= 0:
        # If image is wider than target, crop it to target_width
        if current_width > target_width:
            return img_tensor[:, :, :target_width]
        return img_tensor

    # Pad format for TF.pad is (left, right, top, bottom)
    padding = (0, padding_needed, 0, 0)
    return TF.pad(img_tensor, padding, fill=padding_value, padding_mode="constant")


def load_image(
    image_path: str,
    img_size: Tuple[int, int] = (64, 800),
    channels: int = 1,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Load and preprocess an image for inference.

    Args:
        image_path: Path to the image file
        img_size: Size to resize the image to (height, width)
        channels: Number of channels (1 for grayscale, 3 for RGB)
        normalize: Whether to normalize the image

    Returns:
        Preprocessed image tensor
    """
    try:
        # Check if the image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load the image
        img = Image.open(image_path)

        # Convert to the appropriate mode based on channels parameter
        if channels == 1 and img.mode != "L":
            img = img.convert("L")
        elif channels == 3 and img.mode != "RGB":
            img = img.convert("RGB")

        # 1. First resize the image to target height while maintaining aspect ratio
        img = resize_maintain_aspect(img, target_height=img_size[0])

        # Convert to tensor
        if channels == 1:
            # For grayscale, we need to add a channel dimension
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_tensor = torch.from_numpy(img_array).float()
        else:
            # For RGB images
            img_array = np.array(img)
            # Rearrange from HWC to CHW
            img_array = np.transpose(img_array, (2, 0, 1))
            img_tensor = torch.from_numpy(img_array).float()

        # Normalize to [0, 1]
        img_tensor = img_tensor / 255.0

        # 2. Ensure the tensor has the exact target dimensions (height, width)
        C, H, W = img_tensor.shape
        target_height, target_width = img_size

        padding_value = (
            1.0 if channels == 1 else 0.0
        )  # White for grayscale, black for RGB

        # Fix both height and width if needed
        if H != target_height or W != target_width:
            # Create a new tensor with the target dimensions
            padded_tensor = torch.ones((C, target_height, target_width)) * padding_value

            # Copy as much of the original tensor as fits
            h = min(H, target_height)
            w = min(W, target_width)
            padded_tensor[:, :h, :w] = img_tensor[:, :h, :w]

            img_tensor = padded_tensor
        # If width needs padding but height is correct
        elif W != target_width:
            img_tensor = pad_image_width(
                img_tensor, target_width=target_width, padding_value=padding_value
            )

        # Apply additional normalization
        if normalize:
            if channels == 1:
                # Normalize to [-1, 1] for grayscale
                img_tensor = img_tensor * 2.0 - 1.0
            else:
                # Apply ImageNet normalization for RGB
                mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
                img_tensor = (img_tensor - mean) / std

        return img_tensor

    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        # Return a blank image in case of error
        if channels == 1:
            return torch.zeros((1, img_size[0], img_size[1]))
        else:
            return torch.zeros((3, img_size[0], img_size[1]))


def batch_convert_for_resnet(batch_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of grayscale images to RGB for ResNet.

    Args:
        batch_tensor: Batch of grayscale images (batch_size, 1, height, width)

    Returns:
        Batch of RGB images (batch_size, 3, height, width)
    """
    # Check if the tensor is already RGB
    if batch_tensor.shape[1] == 3:
        return batch_tensor

    # Convert grayscale to RGB by repeating the channel
    rgb_tensor = batch_tensor.repeat(1, 3, 1, 1)

    return rgb_tensor


def prepare_batch(
    batch: Dict, device: torch.device, model_type: str = "cnn_lstm"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare a batch for training.

    Args:
        batch: Batch from the dataloader
        device: Device to move tensors to
        model_type: Type of model being used (cnn_lstm or resnet_lstm)

    Returns:
        Tuple of (images, target_sequences)
    """
    # Get images and formulas from batch
    images = batch["images"].to(device)
    formulas = batch["formulas"].to(device)

    # For ResNet model, convert grayscale to RGB if needed
    if model_type == "resnet_lstm" and images.shape[1] == 1:
        images = batch_convert_for_resnet(images)

    return images, formulas

```

