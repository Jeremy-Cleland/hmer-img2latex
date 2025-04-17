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

logger = get_logger(__name__, log_level="INFO")


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
    generated_sequence: List[int], true_sequence: List[int], n: int = None
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
    # Set default value if not provided
    if n is None:
        n = 4  # Fallback only if config value not passed

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


def masked_accuracy(logits, targets, pad_token_id):
    # 1. get CPU tensors
    logits = logits.detach().cpu()
    targets = targets.detach().cpu()
    # 2. predict
    pred = torch.argmax(logits, dim=-1)
    # 3. mask
    mask = targets.ne(pad_token_id)
    # 4. count
    correct = torch.logical_and(pred.eq(targets), mask).sum().item()
    # 5. safe div
    total = mask.sum().item()
    return correct, total


def token_list_accuracy(
    predictions: List[List[int]], targets: List[List[int]], pad_token_id: int
) -> Tuple[float, int]:
    """
    Calculate token-level accuracy for lists of token sequences.

    Args:
        predictions: List of predicted token sequences
        targets: List of target token sequences
        pad_token_id: ID of the padding token to ignore

    Returns:
        Tuple of (number of correct tokens, total tokens)
    """
    total_correct = 0
    total_tokens = 0

    # Ensure we compare sequences of same length
    for pred_seq, target_seq in zip(predictions, targets):
        # Only compare up to the shorter sequence length
        min_len = min(len(pred_seq), len(target_seq))

        # Count correct tokens
        correct = sum(
            1
            for i in range(min_len)
            if pred_seq[i] == target_seq[i] and target_seq[i] != pad_token_id
        )

        # Count non-padding tokens
        non_pad = sum(1 for t in target_seq[:min_len] if t != pad_token_id)

        total_correct += correct
        total_tokens += non_pad

    # Return raw counts, not ratio
    return total_correct, total_tokens


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

    # 2. Calculate accuracy
    if outputs is not None and targets is not None:
        # If we have model outputs, use them for more accurate token probability-based metrics
        outputs_cpu = _detach_to_cpu(outputs)
        targets_cpu = _detach_to_cpu(targets)

        correct_count, num_tokens = masked_accuracy(
            outputs_cpu, targets_cpu, tokenizer.pad_token_id
        )

        # Calculate accuracy ratio from counts
        accuracy = correct_count / num_tokens if num_tokens > 0 else 0.0

        # 5. Sample predictions and targets for visualization (only if outputs/targets provided)
        sample_data = sample_predictions_and_targets(
            outputs_cpu, targets_cpu, tokenizer, num_samples, confidence_threshold
        )
        combined_metrics["samples"] = sample_data
    else:
        # Calculate accuracy directly from token lists if tensors aren't available
        logger.info(
            "Computing accuracy from token lists (outputs/targets tensors not provided)"
        )
        correct_count, num_tokens = token_list_accuracy(
            all_predictions_cpu, all_targets_cpu, tokenizer.pad_token_id
        )

        # Calculate accuracy ratio from counts
        accuracy = correct_count / num_tokens if num_tokens > 0 else 0.0

        combined_metrics["samples"] = {"samples": []}

    combined_metrics["accuracy"] = accuracy
    combined_metrics["num_tokens"] = num_tokens

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
