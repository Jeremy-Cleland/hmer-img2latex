"""
Enhanced metrics and visualizations for the image-to-LaTeX model.

This module provides additional metrics and visualizations to help with
debugging and understanding model behavior during training.
"""

import json
import os
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import torch
from scipy.stats import entropy

from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


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
    num_samples: int = 5,
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
    batch_size, seq_length, vocab_size = outputs.shape

    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(outputs, dim=-1)

    # Get predicted tokens and their probabilities
    pred_tokens = torch.argmax(probs, dim=-1)
    pred_probs = torch.max(probs, dim=-1)[0]

    # Convert to numpy for easier handling
    pred_tokens_np = pred_tokens.cpu().numpy()
    pred_probs_np = pred_probs.cpu().numpy()
    targets_np = targets.cpu().numpy()

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


def generate_enhanced_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    all_predictions: List[List[int]],
    all_targets: List[List[int]],
    tokenizer: LaTeXTokenizer,
    num_samples: int = 5,
    experiment_name: str = "",
    metrics_dir: str = "",
    epoch: int = 0,
    save_to_file: bool = True,
) -> Dict[str, Any]:
    """
    Generate enhanced metrics for model evaluation.

    Args:
        outputs: Model outputs tensor (batch_size, seq_length, vocab_size)
        targets: Target tensor (batch_size, seq_length)
        all_predictions: List of all predicted token sequences
        all_targets: List of all target token sequences
        tokenizer: LaTeX tokenizer
        num_samples: Number of samples to include
        experiment_name: Name of the experiment
        metrics_dir: Directory to save metrics
        epoch: Current epoch
        save_to_file: Whether to save metrics to a file

    Returns:
        Dictionary of enhanced metrics
    """
    # Sample predictions and targets
    sample_data = sample_predictions_and_targets(
        outputs, targets, tokenizer, num_samples
    )

    # Analyze token distribution
    token_dist = analyze_token_distribution(all_predictions, all_targets, tokenizer)

    # Combine metrics
    enhanced_metrics = {
        "samples": sample_data,
        "token_distribution": token_dist,
        "epoch": epoch,
    }

    # Save to file if requested
    if save_to_file and metrics_dir and experiment_name:
        save_enhanced_metrics(enhanced_metrics, experiment_name, metrics_dir, epoch)

    # Log summary
    log_enhanced_metrics_summary(enhanced_metrics)

    return enhanced_metrics
