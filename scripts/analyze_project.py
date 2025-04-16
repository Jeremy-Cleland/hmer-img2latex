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

import argparse
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
from utils import ensure_output_dir

# Suppress specific warnings if needed (e.g., from libraries)
# warnings.filterwarnings("ignore", category=SomeWarningCategory)
try:
    from img2latex.data.tokenizer import LaTeXTokenizer
    from img2latex.utils.logging import get_logger
except ImportError:
    print("Error: Could not import project modules.", file=sys.stderr)
    print(
        "Please ensure this script is run from the project root directory",
        file=sys.stderr,
    )
    print("or that the 'img2latex' package is in the Python path.", file=sys.stderr)
    sys.exit(1)

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


def analyze_project(
    config_path: str,
    base_dir: str = ".",
    output_dir: str = "outputs/project_analysis",
    detailed: bool = False,
) -> None:
    """Analyze the img2latex project configuration and status.

    Args:
        config_path: Path to the YAML config file
        base_dir: Base directory for the project
        output_dir: Directory to save analysis results
        detailed: Whether to perform detailed analysis
    """
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
        with open(output_path / "project_analysis.json", "w") as f:
            json.dump(results, f, indent=2)
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

    # Save analysis results
    with open(output_path / "project_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Project analysis complete. Results saved to {output_path / 'project_analysis.json'}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the img2latex project configuration and status"
    )
    parser.add_argument(
        "--config-path", required=True, help="Path to the YAML config file"
    )
    parser.add_argument(
        "--base-dir", default=".", help="Base directory for the project"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/project_analysis",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Perform detailed analysis (Git comparison, hyperparameter sweep)",
    )

    args = parser.parse_args()

    analyze_project(args.config_path, args.base_dir, args.output_dir, args.detailed)


if __name__ == "__main__":
    main()
