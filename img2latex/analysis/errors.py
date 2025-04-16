#!/usr/bin/env python
"""
Script to analyze errors in model predictions.

Features:
- Load validation outputs with BLEU and Levenshtein scores
- Bucket examples by edit distance ranges
- Sample examples per bucket
- Generate Markdown report with error patterns
"""

import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import typer
from Levenshtein import distance
from utils import ensure_output_dir, load_csv_file, load_json_file, save_json_file

# Create Typer app
app = typer.Typer(help="Analyze errors in model predictions")


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
def analyze_errors(
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
) -> None:
    """Analyze errors in model predictions and generate a report."""
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Ensure output directory exists
    output_path = ensure_output_dir(output_dir, "errors")

    # Load predictions data
    print(f"Loading predictions from {predictions_file}...")
    try:
        records = load_predictions(predictions_file)
        print(f"Loaded {len(records)} prediction-reference pairs")

        # Group records by edit distance
        print("Grouping records by edit distance...")
        distance_ranges = [(0, 0), (1, 1), (2, 3), (4, float("inf"))]
        buckets = bucket_by_edit_distance(records, distance_ranges)

        # Print bucket statistics
        print("\nEdit distance buckets:")
        for bucket_name, bucket_records in buckets.items():
            print(f"  {bucket_name}: {len(bucket_records)} examples")

        # Identify error patterns
        print("\nIdentifying error patterns...")
        error_patterns = identify_error_patterns(records)

        # Print top error patterns
        if error_patterns:
            print("\nTop error patterns:")
            for pattern in error_patterns[:5]:  # Show top 5
                print(
                    f"  {pattern['pattern']}: {pattern['count']} occurrences - {pattern['description']}"
                )

        # Generate report
        print("\nGenerating error analysis report...")
        report_path = output_path / "error_analysis_report.md"
        report = generate_error_report(
            buckets, error_patterns, samples_per_bucket, report_path
        )

        # Save bucketed examples to JSON for further analysis
        bucketed_data = {}
        for bucket_name, bucket_records in buckets.items():
            bucketed_data[bucket_name] = [
                {
                    "id": record["id"],
                    "reference": record["reference"],
                    "hypothesis": record["hypothesis"],
                    "edit_distance": record.get(
                        "edit_distance", record.get("levenshtein", 0)
                    ),
                    "bleu": record.get("bleu", 0),
                }
                for record in bucket_records[:100]  # Limit to 100 examples per bucket
            ]

        save_json_file(bucketed_data, output_path / "error_buckets.json")

        print(f"Error analysis complete. Report saved to {report_path}")

    except Exception as e:
        print(f"Error during error analysis: {e}")


if __name__ == "__main__":
    app()
