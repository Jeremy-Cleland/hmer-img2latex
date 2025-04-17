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
