#!/usr/bin/env python
"""
Script to analyze various aspects of the img2latex project setup,
including configuration, paths, data characteristics, and model parameters.
"""

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import typer
import yaml
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

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


def load_config(config_path: Path) -> Optional[Dict[str, Any]]:
    """Loads configuration from a YAML file."""
    if not config_path.is_file():
        logger.error(f"Config file not found: {config_path}")
        return None
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded config from: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file {config_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading config: {e}")
        return None


def check_paths(config: Dict[str, Any]) -> List[str]:
    """Verifies that essential data paths from config exist."""
    warnings_errors = []
    base_data_dir = Path(config.get("data", {}).get("data_dir", ""))
    if not base_data_dir or not base_data_dir.is_dir():
        warnings_errors.append(
            f"ERROR: Base data directory not found or not specified: '{base_data_dir}'"
        )
        # Stop further path checks if base dir is missing
        return warnings_errors

    # Check formula file
    formulas_file = config.get("data", {}).get("formulas_file")
    formulas_path = base_data_dir / (formulas_file if formulas_file else "")
    if not formulas_file or not formulas_path.is_file():
        warnings_errors.append(f"WARNING: Formulas file not found: '{formulas_path}'")

    # Check split files
    for split in ["train", "validate", "test"]:
        split_key = f"{split}_file"
        split_file = config.get("data", {}).get(split_key)
        split_path = base_data_dir / (split_file if split_file else "")
        if not split_file or not split_path.is_file():
            warnings_errors.append(
                f"WARNING: Split file '{split_key}' not found: '{split_path}'"
            )

    # Check image directory
    img_dir = config.get("data", {}).get("img_dir")
    expected_img_dir = "img"
    if not img_dir:
        warnings_errors.append("ERROR: 'data.img_dir' not specified in config.")
    elif img_dir != expected_img_dir:
        warnings_errors.append(
            f"WARNING: 'data.img_dir' is '{img_dir}', recommend setting to '{expected_img_dir}' based on dataset used."
        )
    else:
        img_path = base_data_dir / img_dir
        if not img_path.is_dir():
            warnings_errors.append(
                f"WARNING: Specified image directory not found: '{img_path}'"
            )

    return warnings_errors


def analyze_formula_lengths(
    formulas_path: Path,
    tokenizer: LaTeXTokenizer,
    config_max_len: int,
    percentile_cutoff: float = 95.0,
) -> Tuple[Dict[str, Any], List[str], np.ndarray]:
    """Analyzes tokenized lengths of formulas.

    Args:
        formulas_path: Path to the formulas file
        tokenizer: LaTeXTokenizer instance
        config_max_len: Maximum sequence length from config
        percentile_cutoff: Percentile to suggest as cutoff (default: 95.0)

    Returns:
        stats: Dictionary with length statistics
        warnings_errors: List of warning/error messages
        lengths_arr: Array of all formula lengths
    """
    warnings_errors = []
    stats = {
        "max_found_len": 0,
        "avg_len": 0,
        "median_len": 0,
        "num_exceeding": 0,
        "total_formulas": 0,
        "percentiles": {},
        "suggested_cutoff": 0,
    }
    if not formulas_path.is_file():
        warnings_errors.append(
            f"ERROR: Cannot analyze lengths, formulas file not found: {formulas_path}"
        )
        return stats, warnings_errors, np.array([])

    lengths = []
    try:
        with open(formulas_path, "r", encoding="utf-8") as f:
            logger.info("Analyzing formula lengths (this may take a moment)...")
            for line in tqdm(f, desc="Reading formulas"):
                formula = line.strip()
                # Tokenize *including* special tokens to get true sequence length
                tokens = tokenizer.encode(formula, add_special_tokens=True)
                lengths.append(len(tokens))

        if not lengths:
            warnings_errors.append(
                "WARNING: No formulas found or processed in the formulas file."
            )
            return stats, warnings_errors, np.array([])

        lengths_arr = np.array(lengths)

        # Basic statistics
        stats["max_found_len"] = int(np.max(lengths_arr))
        stats["avg_len"] = float(np.mean(lengths_arr))
        stats["median_len"] = int(np.median(lengths_arr))
        stats["std_len"] = float(np.std(lengths_arr))
        stats["num_exceeding"] = int(np.sum(lengths_arr > config_max_len))
        stats["total_formulas"] = len(lengths)

        # Calculate percentiles for distribution analysis
        percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
        for p in percentiles:
            stats["percentiles"][p] = int(np.percentile(lengths_arr, p))

        # Suggest cutoff based on provided percentile
        stats["suggested_cutoff"] = int(np.percentile(lengths_arr, percentile_cutoff))

        # Histogram data
        stats["histogram_bins"] = 50
        stats["histogram_counts"], stats["histogram_edges"] = np.histogram(
            lengths_arr, bins=stats["histogram_bins"]
        )

        if stats["max_found_len"] > config_max_len:
            warnings_errors.append(
                f"WARNING: Max formula length found ({stats['max_found_len']}) "
                f"exceeds config max_seq_length ({config_max_len}). "
                f"{stats['num_exceeding']} formulas will be affected (filtered/truncated)."
            )

        # Suggest filter strategy based on distribution
        if stats["suggested_cutoff"] < config_max_len:
            warnings_errors.append(
                f"INFO: Consider using {stats['suggested_cutoff']} as a length cutoff "
                f"({percentile_cutoff}th percentile), which is less than config max_seq_length ({config_max_len})."
            )
        else:
            warnings_errors.append(
                f"INFO: The {percentile_cutoff}th percentile length is {stats['suggested_cutoff']}, "
                f"which exceeds config max_seq_length ({config_max_len}). "
                f"Consider increasing max_seq_length or using a lower percentile cutoff."
            )

    except Exception as e:
        warnings_errors.append(f"ERROR: Failed during formula length analysis: {e}")
        return stats, warnings_errors, np.array([])

    return stats, warnings_errors, lengths_arr


def analyze_formulas(
    formulas_path: Path,
    tokenizer: LaTeXTokenizer,
    config_max_len: int,
    percentile_cutoff: float = 95.0,
) -> Tuple[Dict[str, Any], List[str], np.ndarray]:
    """Analyzes tokenized lengths and content of formulas."""
    warnings_errors = []
    stats = {
        "max_token_len": 0,
        "avg_token_len": 0,
        "median_token_len": 0,
        "std_token_len": 0,
        "num_exceeding_config": 0,
        "num_containing_unk": 0,
        "total_formulas": 0,
        "token_counts": Counter(),
        "longest_formula_idx": -1,
        "longest_formula_len": 0,
        "longest_formula_str": "",
        "percentiles": {},
        "suggested_cutoff": 0,
        "shortest_formulas": [],  # Will store (length, idx, formula_text) tuples
        "longest_formulas": [],  # Will store (length, idx, formula_text) tuples
    }
    if not formulas_path.is_file():
        warnings_errors.append(
            f"ERROR: Cannot analyze formulas, file not found: {formulas_path}"
        )
        return stats, warnings_errors, np.array([])

    token_lengths = []
    formulas_with_unk = 0
    token_counter = Counter()
    longest_len = 0
    longest_idx = -1
    longest_str = ""

    # Track shortest and longest formulas
    shortest_formulas = []  # Will contain (length, idx, formula) tuples
    longest_formulas = []  # Will contain (length, idx, formula) tuples

    # Number to track
    top_n = 20

    try:
        formulas_content = []
        with open(formulas_path, "r", encoding="utf-8") as f:
            formulas_content = [line.strip() for line in f]
        stats["total_formulas"] = len(formulas_content)

        logger.info("Analyzing formula lengths and tokens (this may take a moment)...")
        for idx, formula_str in enumerate(
            tqdm(formulas_content, desc="Analyzing formulas")
        ):
            # Tokenize including special tokens for length, but count base tokens
            formula_base_tokens = formula_str.split()
            token_counter.update(formula_base_tokens)

            # Check length against config (including START/END)
            encoded_ids_with_special = tokenizer.encode(
                formula_str, add_special_tokens=True
            )
            current_len = len(encoded_ids_with_special)
            token_lengths.append(current_len)

            # Update shortest/longest formula tracking
            if len(shortest_formulas) < top_n:
                shortest_formulas.append((current_len, idx, formula_str))
                shortest_formulas.sort()  # Sort by length (first element of tuple)
            elif current_len < shortest_formulas[-1][0]:
                shortest_formulas.pop()  # Remove the longest of the shortest
                shortest_formulas.append((current_len, idx, formula_str))
                shortest_formulas.sort()  # Sort by length

            if len(longest_formulas) < top_n:
                longest_formulas.append((current_len, idx, formula_str))
                longest_formulas.sort(reverse=True)  # Sort by length, descending
            elif current_len > longest_formulas[-1][0]:
                longest_formulas.pop()  # Remove the shortest of the longest
                longest_formulas.append((current_len, idx, formula_str))
                longest_formulas.sort(reverse=True)  # Sort by length, descending

            if current_len > longest_len:
                longest_len = current_len
                longest_idx = idx
                longest_str = formula_str

            # Check for UNK tokens (using base tokens before special tokens added)
            encoded_ids_no_special = tokenizer.encode(
                formula_str, add_special_tokens=False
            )
            if tokenizer.unk_token_id in encoded_ids_no_special:
                formulas_with_unk += 1

        if not token_lengths:
            warnings_errors.append(
                "WARNING: No formulas found or processed in the formulas file."
            )
            return stats, warnings_errors, np.array([])

        lengths_arr = np.array(token_lengths)
        stats["max_token_len"] = int(np.max(lengths_arr))
        stats["avg_token_len"] = float(np.mean(lengths_arr))
        stats["median_token_len"] = int(np.median(lengths_arr))
        stats["std_token_len"] = float(np.std(lengths_arr))
        stats["num_exceeding_config"] = int(np.sum(lengths_arr > config_max_len))
        stats["num_containing_unk"] = formulas_with_unk
        stats["token_counts"] = token_counter
        stats["longest_formula_idx"] = longest_idx
        stats["longest_formula_len"] = longest_len
        stats["longest_formula_str"] = longest_str
        stats["shortest_formulas"] = shortest_formulas
        stats["longest_formulas"] = longest_formulas

        # Calculate percentiles for distribution analysis
        percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
        for p in percentiles:
            stats["percentiles"][p] = int(np.percentile(lengths_arr, p))

        # Suggest cutoff based on provided percentile
        stats["suggested_cutoff"] = int(np.percentile(lengths_arr, percentile_cutoff))

        # Histogram data
        stats["histogram_bins"] = 50
        stats["histogram_counts"], stats["histogram_edges"] = np.histogram(
            lengths_arr, bins=stats["histogram_bins"]
        )

        if stats["max_token_len"] > config_max_len:
            warnings_errors.append(
                f"WARNING: Max formula token length ({stats['max_token_len']}) "
                f"exceeds config max_seq_length ({config_max_len}). "
                f"{stats['num_exceeding_config']} formulas will be affected (filtered/truncated)."
            )
        if stats["num_containing_unk"] > 0:
            warnings_errors.append(
                f"WARNING: Found {stats['num_containing_unk']} formulas containing <UNK> tokens. "
                "Consider adding frequent missing tokens to vocabulary or reviewing formulas."
            )

        # Suggest filter strategy based on distribution
        if stats["suggested_cutoff"] < config_max_len:
            warnings_errors.append(
                f"INFO: Consider using {stats['suggested_cutoff']} as a length cutoff "
                f"({percentile_cutoff}th percentile), which is less than config max_seq_length ({config_max_len})."
            )
        else:
            warnings_errors.append(
                f"INFO: The {percentile_cutoff}th percentile length is {stats['suggested_cutoff']}, "
                f"which exceeds config max_seq_length ({config_max_len}). "
                f"Consider increasing max_seq_length or using a lower percentile cutoff."
            )

    except Exception as e:
        warnings_errors.append(f"ERROR: Failed during formula analysis: {e}")
        return stats, warnings_errors, np.array([])

    return stats, warnings_errors, lengths_arr


def plot_length_distribution(
    lengths_arr: np.ndarray,
    output_dir: Path,
    cutoff: int = None,
    config_max: int = None,
):
    """Creates a histogram of formula lengths and saves it to a file."""
    if len(lengths_arr) == 0:
        logger.error("Cannot plot distribution: no length data available")
        return

    try:
        plt.figure(figsize=(10, 6))

        # Plot histogram with density=True for normalized view
        counts, bins, _ = plt.hist(lengths_arr, bins=50, alpha=0.7, density=False)

        # Add vertical lines for cutoffs if provided
        if cutoff is not None:
            plt.axvline(
                x=cutoff, color="r", linestyle="--", label=f"Suggested Cutoff: {cutoff}"
            )

        if config_max is not None:
            plt.axvline(
                x=config_max,
                color="g",
                linestyle="--",
                label=f"Config Max: {config_max}",
            )

        plt.title("Distribution of Formula Sequence Lengths")
        plt.xlabel("Sequence Length")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "formula_length_distribution.png"

        plt.savefig(output_file)
        plt.close()

        logger.info(f"Length distribution plot saved to: {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Failed to create length distribution plot: {e}")
        return None


def verify_model_config(config: Dict[str, Any]) -> List[str]:
    """Checks consistency within the model configuration."""
    warnings_errors = []
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name")
    encoder_cfg = model_cfg.get("encoder", {})
    decoder_cfg = model_cfg.get("decoder", {})
    embedding_dim = model_cfg.get("embedding_dim")

    # --- General Checks ---
    if not model_name:
        warnings_errors.append("ERROR: 'model.name' is not specified.")
    if not encoder_cfg:
        warnings_errors.append("WARNING: 'model.encoder' section missing.")
    if not decoder_cfg:
        warnings_errors.append("WARNING: 'model.decoder' section missing.")
    if not embedding_dim:
        warnings_errors.append("WARNING: 'model.embedding_dim' is not specified.")

    # --- Encoder Specific Checks ---
    expected_height, expected_width = 64, 800
    if model_name == "cnn_lstm":
        cnn_params = encoder_cfg.get("cnn", {})
        if not cnn_params:
            warnings_errors.append(
                "ERROR: Model type is 'cnn_lstm' but 'model.encoder.cnn' section is missing."
            )
        else:
            channels = cnn_params.get("channels")
            height = cnn_params.get("img_height")
            width = cnn_params.get("img_width")
            if channels != 1:
                warnings_errors.append(
                    f"WARNING: CNN channels set to {channels}, expected 1 (Grayscale)."
                )
            if (height, width) != (expected_height, expected_width):
                warnings_errors.append(
                    f"WARNING: CNN input size is ({height}, {width}), expected ({expected_height}, {expected_width})."
                )
            if not cnn_params.get("conv_filters"):
                warnings_errors.append("WARNING: CNN 'conv_filters' not specified.")
            if not cnn_params.get("kernel_size"):
                warnings_errors.append("WARNING: CNN 'kernel_size' not specified.")
            if not cnn_params.get("pool_size"):
                warnings_errors.append("WARNING: CNN 'pool_size' not specified.")

    elif model_name == "resnet_lstm":
        res_params = encoder_cfg.get("resnet", {})
        if not res_params:
            warnings_errors.append(
                "ERROR: Model type is 'resnet_lstm' but 'model.encoder.resnet' section is missing."
            )
        else:
            channels = res_params.get("channels")
            height = res_params.get("img_height")
            width = res_params.get("img_width")
            if channels != 3:
                warnings_errors.append(
                    f"WARNING: ResNet channels set to {channels}, expected 3 (RGB)."
                )
            if (height, width) != (expected_height, expected_width):
                warnings_errors.append(
                    f"WARNING: ResNet input size is ({height}, {width}), expected ({expected_height}, {expected_width})."
                )
            if not res_params.get("model_name"):
                warnings_errors.append("WARNING: ResNet 'model_name' not specified.")

    else:
        warnings_errors.append(f"ERROR: Unknown 'model.name': {model_name}")

    # --- Decoder Specific Checks ---
    if not decoder_cfg.get("hidden_dim"):
        warnings_errors.append("WARNING: Decoder 'hidden_dim' not specified.")
    if decoder_cfg.get("hidden_dim") != embedding_dim and not decoder_cfg.get(
        "attention"
    ):
        warnings_errors.append(
            f"WARNING: Decoder 'hidden_dim' ({decoder_cfg.get('hidden_dim')}) typically matches 'embedding_dim' ({embedding_dim}) when attention is not used, check decoder input concatenation logic."
        )
    if not decoder_cfg.get("lstm_layers"):
        warnings_errors.append(
            "INFO: Decoder 'lstm_layers' not specified (defaults likely used)."
        )
    if decoder_cfg.get("attention") is None:
        warnings_errors.append(
            "INFO: Decoder 'attention' flag not specified (defaults likely used)."
        )

    return warnings_errors


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
    logger.info(f"Output directory set to: {output_dir}")

    return output_dir


# --- Main Analysis Function ---


def analyze_project(
    config_path: str,
    percentile_cutoff: float = 95.0,
    plot_dir: str = None,
    detailed: bool = False,
):
    """Runs the full project analysis."""
    logger.info(f"--- Starting Project Analysis ({config_path}) ---")
    all_warnings_errors: List[str] = []

    # 1. Load Config
    config = load_config(Path(config_path))
    if config is None:
        logger.critical("Analysis cannot proceed without a valid config file.")
        sys.exit(1)

    # Setup output directory
    output_dir = ensure_output_dir(plot_dir or "outputs", "formula_analysis")

    # 2. Check Paths
    logger.info("--- Verifying Paths ---")
    path_warnings = check_paths(config)
    all_warnings_errors.extend(path_warnings)
    for msg in path_warnings:
        logger.warning(msg) if "WARNING" in msg else logger.error(msg)
    if any("ERROR" in msg for msg in path_warnings):
        logger.critical("Aborting due to critical path errors.")
        sys.exit(1)
    logger.info("Path verification complete.")

    # 3. Initialize & Analyze Tokenizer/Formulas
    logger.info("--- Analyzing Tokenizer & Formula Lengths ---")
    formulas_path = Path(config["data"]["data_dir"]) / config["data"]["formulas_file"]
    config_max_seq_len = config.get("data", {}).get(
        "max_seq_length", 150
    )  # Default if missing
    formula_stats = {}
    vocab_size = 0
    lengths_arr = np.array([])

    if formulas_path.is_file():
        try:
            # Init tokenizer (will fit inside)
            tokenizer = LaTeXTokenizer(max_sequence_length=config_max_seq_len)
            tokenizer.fit_on_formulas_file(str(formulas_path))  # fit needs string path
            vocab_size = tokenizer.vocab_size
            logger.info(f"Tokenizer Initialized. Vocabulary Size: {vocab_size}")

            # Analyze lengths
            if detailed:
                formula_stats, formula_warnings, lengths_arr = analyze_formulas(
                    formulas_path, tokenizer, config_max_seq_len, percentile_cutoff
                )
            else:
                formula_stats, formula_warnings, lengths_arr = analyze_formula_lengths(
                    formulas_path, tokenizer, config_max_seq_len, percentile_cutoff
                )

            all_warnings_errors.extend(formula_warnings)
            for msg in formula_warnings:
                logger.warning(msg) if "WARNING" in msg else logger.error(msg)

            # Generate distribution plot
            if len(lengths_arr) > 0:
                plot_file = plot_length_distribution(
                    lengths_arr,
                    output_dir,
                    cutoff=formula_stats.get("suggested_cutoff"),
                    config_max=config_max_seq_len,
                )
                if plot_file:
                    all_warnings_errors.append(
                        f"INFO: Length distribution plot saved to: {plot_file}"
                    )

            # Save histogram data as CSV for further analysis
            if (
                "histogram_counts" in formula_stats
                and "histogram_edges" in formula_stats
            ):
                try:
                    hist_file = output_dir / "length_histogram_data.csv"
                    with open(hist_file, "w") as f:
                        f.write("bin_start,bin_end,count\n")
                        for i, count in enumerate(formula_stats["histogram_counts"]):
                            bin_start = formula_stats["histogram_edges"][i]
                            bin_end = formula_stats["histogram_edges"][i + 1]
                            f.write(f"{bin_start:.1f},{bin_end:.1f},{count}\n")
                    logger.info(f"Histogram data saved to: {hist_file}")
                    all_warnings_errors.append(
                        f"INFO: Histogram data saved to: {hist_file}"
                    )
                except Exception as e:
                    logger.error(f"Failed to save histogram data: {e}")

        except Exception as e:
            logger.error(f"Error during tokenizer/formula analysis: {e}")
            all_warnings_errors.append(f"ERROR: Tokenizer/formula analysis failed: {e}")
    else:
        logger.error(
            "Skipping tokenizer/formula analysis as formulas file was not found."
        )
        all_warnings_errors.append(
            "ERROR: Formulas file missing, cannot analyze lengths or vocab."
        )

    # 4. Verify Model Config Consistency
    logger.info("--- Verifying Model Configuration ---")
    model_warnings = verify_model_config(config)
    all_warnings_errors.extend(model_warnings)
    for msg in model_warnings:
        logger.warning(msg) if "WARNING" in msg else logger.error(msg)

    # 5. Print Summary Report
    print("\n" + "=" * 30 + " Analysis Summary " + "=" * 30)

    print("\n[Configuration]")
    print(f"  Config File:          {config_path}")
    print(f"  Model Type:           {config.get('model', {}).get('name', 'N/A')}")
    print(f"  Target Device:        {config.get('training', {}).get('device', 'N/A')}")
    print(f"  Base Data Directory:  {config.get('data', {}).get('data_dir', 'N/A')}")
    img_dir = config.get("data", {}).get("img_dir", "N/A")
    print(f"  Image Directory:      {img_dir} (Expected: img)")
    print(f"  Num Workers:          {config.get('data', {}).get('num_workers', 'N/A')}")

    print("\n[Tokenizer & Formulas]")
    print(f"  Config max_seq_len:   {config_max_seq_len}")
    if formula_stats and tokenizer:
        # Create Rich table for formula statistics
        table = Table(title="Formula Length Statistics")
        table.add_column("Statistic", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Vocabulary Size", str(vocab_size))

        if detailed:
            table.add_row(
                "Special Tokens", str(list(tokenizer.special_tokens.values()))
            )
            table.add_row(
                "Total Formulas", str(formula_stats.get("total_formulas", "N/A"))
            )
            table.add_row(
                "Max Token Length", str(formula_stats.get("max_token_len", "N/A"))
            )
            table.add_row(
                "Average Token Length", f"{formula_stats.get('avg_token_len', 0):.1f}"
            )
            table.add_row(
                "Median Token Length", str(formula_stats.get("median_token_len", "N/A"))
            )
            table.add_row(
                "Std Dev Token Length", f"{formula_stats.get('std_token_len', 0):.2f}"
            )
            table.add_row(
                "Num > config_max_len",
                str(formula_stats.get("num_exceeding_config", "N/A")),
            )
            table.add_row(
                "Num w/ <UNK> Tokens",
                str(formula_stats.get("num_containing_unk", "N/A")),
            )
        else:
            table.add_row(
                "Total Formulas", str(formula_stats.get("total_formulas", "N/A"))
            )
            table.add_row(
                "Max Length Found", str(formula_stats.get("max_found_len", "N/A"))
            )
            table.add_row("Average Length", f"{formula_stats.get('avg_len', 0):.1f}")
            table.add_row("Median Length", str(formula_stats.get("median_len", "N/A")))
            table.add_row("Std Dev Length", f"{formula_stats.get('std_len', 0):.2f}")
            table.add_row(
                "Num > config_max_len", str(formula_stats.get("num_exceeding", "N/A"))
            )

        # Add percentile information
        table.add_section()
        for p, val in formula_stats.get("percentiles", {}).items():
            table.add_row(f"{p}th Percentile", str(val))

        table.add_section()
        table.add_row(
            "Suggested Cutoff",
            f"{formula_stats.get('suggested_cutoff', 'N/A')} (at {percentile_cutoff}th percentile)",
        )

        console.print(table)

        if detailed and formula_stats.get("token_counts"):
            most_common = formula_stats["token_counts"].most_common(10)
            print(
                f"  Most Common Tokens:   {', '.join([f'{t}({c})' for t, c in most_common])}"
            )

        if detailed and formula_stats.get("longest_formula_idx", -1) >= 0:
            print(
                f"  Longest Formula (idx {formula_stats['longest_formula_idx']}, len {formula_stats['longest_formula_len']}):"
            )
            print(
                f"    '{formula_stats.get('longest_formula_str', '')[:100]}...'"
            )  # Print start of longest

        if "histogram_counts" in formula_stats:
            print("\nLength Distribution Summary:")
            max_bin_index = np.argmax(formula_stats["histogram_counts"])
            lower_edge = int(formula_stats["histogram_edges"][max_bin_index])
            upper_edge = int(formula_stats["histogram_edges"][max_bin_index + 1])
            print(f"  Most common length range: {lower_edge} - {upper_edge} tokens")

        # Add extreme formulas output
        if "shortest_formulas" in formula_stats and "longest_formulas" in formula_stats:
            # Save extreme formulas to file
            extremes_file = output_dir / "extreme_formulas.txt"
            try:
                with open(extremes_file, "w") as f:
                    f.write("=== TOP 20 SHORTEST FORMULAS ===\n\n")
                    for i, (length, idx, formula) in enumerate(
                        formula_stats["shortest_formulas"]
                    ):
                        f.write(f"#{i + 1}: Length={length}, Index={idx}\n")
                        f.write(f"{formula}\n\n")

                    f.write("\n\n=== TOP 20 LONGEST FORMULAS ===\n\n")
                    for i, (length, idx, formula) in enumerate(
                        formula_stats["longest_formulas"]
                    ):
                        f.write(f"#{i + 1}: Length={length}, Index={idx}\n")
                        f.write(f"{formula}\n\n")

                logger.info(f"Extreme formulas saved to: {extremes_file}")
                all_warnings_errors.append(
                    f"INFO: Extreme formulas saved to: {extremes_file}"
                )
            except Exception as e:
                logger.error(f"Failed to save extreme formulas: {e}")

            # Print extreme formulas summary to console
            print("\n[Top 20 Shortest Formulas]")
            table_short = Table(title="Shortest Formulas by Token Length")
            table_short.add_column("Rank", style="cyan")
            table_short.add_column("Length", style="green")
            table_short.add_column("Index", style="blue")
            table_short.add_column("Formula", style="magenta")

            for i, (length, idx, formula) in enumerate(
                formula_stats["shortest_formulas"]
            ):
                # Truncate formula if too long
                display_formula = formula if len(formula) < 50 else formula[:47] + "..."
                table_short.add_row(f"#{i + 1}", str(length), str(idx), display_formula)

            console.print(table_short)

            print("\n[Top 20 Longest Formulas]")
            table_long = Table(title="Longest Formulas by Token Length")
            table_long.add_column("Rank", style="cyan")
            table_long.add_column("Length", style="green")
            table_long.add_column("Index", style="blue")
            table_long.add_column("Formula (truncated)", style="magenta")

            for i, (length, idx, formula) in enumerate(
                formula_stats["longest_formulas"]
            ):
                # Always truncate formula for display
                display_formula = formula[:47] + "..."
                table_long.add_row(f"#{i + 1}", str(length), str(idx), display_formula)

            console.print(table_long)
    else:
        print("  (Analysis skipped due to missing formulas file)")

    print("\n[Model Parameters (from Config)]")
    model_name = config.get("model", {}).get("name", "N/A")
    if model_name == "cnn_lstm":
        cnn_params = config.get("model", {}).get("encoder", {}).get("cnn", {})
        print("  Encoder (CNN):")
        print(
            f"    Input Size (HxWxC): {cnn_params.get('img_height', 'N/A')}x{cnn_params.get('img_width', 'N/A')}x{cnn_params.get('channels', 'N/A')} (Expected 64x800x1)"
        )
        print(f"    Conv Filters:       {cnn_params.get('conv_filters', 'N/A')}")
        print(f"    Kernel Size:        {cnn_params.get('kernel_size', 'N/A')}")
        print(f"    Pool Size:          {cnn_params.get('pool_size', 'N/A')}")
        print(f"    Padding:            {cnn_params.get('padding', 'N/A')}")
    elif model_name == "resnet_lstm":
        res_params = config.get("model", {}).get("encoder", {}).get("resnet", {})
        print("  Encoder (ResNet):")
        print(
            f"    Input Size (HxWxC): {res_params.get('img_height', 'N/A')}x{res_params.get('img_width', 'N/A')}x{res_params.get('channels', 'N/A')} (Expected 64x800x3)"
        )
        print(f"    ResNet Model:       {res_params.get('model_name', 'N/A')}")
        print(f"    Freeze Backbone:    {res_params.get('freeze_backbone', 'N/A')}")
    else:
        print(f"  Encoder:              Unknown type '{model_name}'")

    dec_params = config.get("model", {}).get("decoder", {})
    print("  Decoder (LSTM):")
    print(
        f"    Embedding Dim:      {config.get('model', {}).get('embedding_dim', 'N/A')}"
    )
    print(f"    Hidden Dim:         {dec_params.get('hidden_dim', 'N/A')}")
    print(f"    LSTM Layers:        {dec_params.get('lstm_layers', 'N/A')}")
    print(f"    Dropout:            {dec_params.get('dropout', 'N/A')}")
    print(f"    Attention:          {dec_params.get('attention', 'N/A')}")

    print("\n[Training Parameters (from Config)]")
    train_params = config.get("training", {})
    print(f"  Optimizer:            {train_params.get('optimizer', 'N/A')}")
    print(f"  Learning Rate:        {train_params.get('learning_rate', 'N/A')}")
    print(f"  Weight Decay:         {train_params.get('weight_decay', 'N/A')}")
    print(f"  Epochs:               {train_params.get('epochs', 'N/A')}")
    print(
        f"  Early Stopping:       {train_params.get('early_stopping_patience', 'N/A')}"
    )
    print(f"  Gradient Clipping:    {train_params.get('clip_grad_norm', 'N/A')}")

    print("\n[Analysis Issues Found]")
    if not all_warnings_errors:
        print("  ✅ No major issues detected.")
    else:
        print(f"  Detected {len(all_warnings_errors)} potential issue(s):")
        for i, msg in enumerate(all_warnings_errors):
            level = (
                msg.split(":")[0] if ":" in msg else "INFO"
            )  # ERROR or WARNING or INFO
            icon = "❌" if level == "ERROR" else "⚠️" if level == "WARNING" else "ℹ️"
            print(f"    {icon} {i + 1}. {msg}")

    print("=" * 78)
    logger.info("--- Project Analysis Complete ---")


@app.command()
def main(
    config: str = typer.Option(
        "img2latex/configs/config.yaml",
        "--config",
        "-c",
        help="Path to the configuration YAML file.",
    ),
    percentile_cutoff: float = typer.Option(
        95.0,
        "--cutoff",
        "-p",
        help="Percentile to suggest as length cutoff (default: 95.0)",
        min=1.0,
        max=100.0,
    ),
    plot_dir: str = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Base directory for saving analysis outputs",
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Enable detailed analysis (more comprehensive but slower)",
    ),
):
    """Analyze img2latex project configuration and data."""
    config_file_path = Path(config)

    # Attempt to make path relative to CWD if it doesn't exist as absolute/relative
    if not config_file_path.exists():
        logger.warning(
            f"Config path '{config_file_path}' not found directly. Trying relative to current directory..."
        )
        config_file_path = Path.cwd() / config
        if not config_file_path.exists():
            logger.error(f"Config file '{config}' not found relative to CWD either.")
            sys.exit(1)
        else:
            logger.info(f"Found config file relative to CWD: {config_file_path}")

    analyze_project(str(config_file_path), percentile_cutoff, plot_dir, detailed)


if __name__ == "__main__":
    app()
