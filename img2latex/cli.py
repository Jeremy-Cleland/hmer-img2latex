"""
Command-line interface for the image-to-LaTeX model.
"""

import os
from typing import Dict, Optional

import torch
import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from img2latex.data.dataset import create_data_loaders
from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.model.seq2seq import Seq2SeqModel
from img2latex.training.predictor import Predictor
from img2latex.training.trainer import Trainer
from img2latex.utils.logging import configure_logging, get_logger
from img2latex.utils.mps_utils import set_device, set_seed
from img2latex.utils.path_utils import path_manager
from img2latex.utils.registry import experiment_registry

app = typer.Typer(name="img2latex", help="Image to LaTeX conversion tool")
console = Console()
logger = get_logger(__name__)


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

        # Configure logging
        configure_logging(config)

        # Set device
        device_obj = set_device(config["training"]["device"])

        # Create directories
        experiment_registry.path_manager.create_experiment_structure(experiment_name)

        # Save config to experiment directory
        config_save_path = path_manager.get_config_path(experiment_name)
        with open(config_save_path, "w") as f:
            yaml.dump(config, f)

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

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device_obj)
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

                # Filter out padding
                target_ids = [
                    idx for idx in target_ids if idx != predictor.tokenizer.pad_token_id
                ]

                # Convert prediction to token IDs
                pred_ids = predictor.tokenizer.encode(latex)

                # Add to lists
                all_predictions.append(pred_ids)
                all_targets.append(target_ids)

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


if __name__ == "__main__":
    app()
