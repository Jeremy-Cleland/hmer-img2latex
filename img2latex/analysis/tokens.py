#!/usr/bin/env python
"""
Script to analyze token distributions in predicted and ground truth sequences.

Features:
- Read decoded predictions and ground truths
- Compute token frequency histograms
- Calculate KL-divergence between distributions
- Highlight under/over-represented tokens
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from scipy.stats import entropy
from utils import ensure_output_dir, load_csv_file, load_json_file

# Create Typer app
app = typer.Typer(help="Analyze token distributions in predictions and ground truths")


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


def tokenize_sequences(sequences: List[str], delimiter: str = " ") -> List[List[str]]:
    """Tokenize sequences using the specified delimiter.

    Args:
        sequences: List of text sequences
        delimiter: Token delimiter (default: space)

    Returns:
        List of tokenized sequences
    """
    return [seq.split(delimiter) for seq in sequences]


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


@app.command()
def analyze_tokens(
    predictions_file: str = typer.Argument(
        ..., help="Path to file containing predictions and ground truths"
    ),
    output_dir: str = typer.Option(
        "outputs/token_analysis", help="Directory to save analysis results"
    ),
    token_delimiter: str = typer.Option(" ", help="Delimiter to use for tokenization"),
    top_k: int = typer.Option(20, help="Number of top divergent tokens to report"),
) -> None:
    """Analyze token distributions in predictions and ground truths."""
    # Ensure output directory exists
    output_path = ensure_output_dir(output_dir, "tokens")

    # Load predictions data
    print(f"Loading predictions from {predictions_file}...")
    try:
        predictions, ground_truths = load_predictions_data(predictions_file)
        print(f"Loaded {len(predictions)} prediction-reference pairs")

        # Tokenize sequences
        print("Tokenizing sequences...")
        tokenized_predictions = tokenize_sequences(predictions, token_delimiter)
        tokenized_ground_truths = tokenize_sequences(ground_truths, token_delimiter)

        # Compute token frequencies
        print("Computing token frequencies...")
        pred_freqs = compute_token_frequencies(tokenized_predictions)
        truth_freqs = compute_token_frequencies(tokenized_ground_truths)

        pred_unique_tokens = len(pred_freqs)
        truth_unique_tokens = len(truth_freqs)

        print(f"Predictions contain {pred_unique_tokens} unique tokens")
        print(f"Ground truths contain {truth_unique_tokens} unique tokens")

        # Calculate KL-divergence
        print("Calculating KL-divergence...")
        kl_divergence, token_kl = calculate_kl_divergence(pred_freqs, truth_freqs)

        # Find most divergent tokens
        print("Finding most divergent tokens...")
        under_represented, over_represented = find_divergent_tokens(token_kl, top_k)

        # Generate plots
        print("Generating plots...")
        plot_token_distributions(pred_freqs, truth_freqs, output_path)

        # Generate report
        print("Generating report...")
        print_divergence_report(
            under_represented, over_represented, kl_divergence, output_path
        )

        # Save token frequencies as CSV
        print("Saving token frequencies...")

        # Combine frequencies into a DataFrame
        all_tokens = sorted(set(list(pred_freqs.keys()) + list(truth_freqs.keys())))
        freq_data = {
            "token": all_tokens,
            "predictions_count": [pred_freqs.get(token, 0) for token in all_tokens],
            "ground_truth_count": [truth_freqs.get(token, 0) for token in all_tokens],
        }

        freq_df = pd.DataFrame(freq_data)
        freq_df.to_csv(output_path / "token_frequencies.csv", index=False)

        print(f"Token frequencies saved to {output_path / 'token_frequencies.csv'}")
        print(f"Token analysis complete. Results saved to {output_path}")

    except Exception as e:
        print(f"Error analyzing token distributions: {e}")


if __name__ == "__main__":
    app()
