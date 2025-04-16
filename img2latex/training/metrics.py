"""
Evaluation metrics for the image-to-LaTeX model.
"""

import collections
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


def levenshtein_distance(
    sequence_one: List[int], 
    sequence_two: List[int]
) -> float:
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
                    dist_tab[r - 1][c],
                    dist_tab[r][c - 1],
                    dist_tab[r - 1][c - 1]
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
    generated_sequence: List[int], 
    true_sequence: List[int], 
    n: int = 4
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
            tuple(generated_sequence[i:i + gram_size]) 
            for i in range(gen_len - gram_size + 1)
        ]
        true_ngrams = [
            tuple(true_sequence[i:i + gram_size]) 
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
    predictions: List[List[int]], 
    targets: List[List[int]]
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for a batch of predictions.
    
    Args:
        predictions: List of predicted token sequences
        targets: List of target token sequences
        
    Returns:
        Dictionary of metrics
    """
    # Check that predictions and targets have the same number of samples
    assert len(predictions) == len(targets), "Predictions and targets must have the same length"
    
    num_sequences = len(predictions)
    
    # Calculate BLEU score
    bleu_scores = [
        bleu_n_score(predictions[i], targets[i], 4) 
        for i in range(num_sequences)
    ]
    mean_bleu = sum(bleu_scores) / num_sequences
    
    # Calculate Levenshtein similarity
    lev_similarities = [
        levenshtein_distance(predictions[i], targets[i]) 
        for i in range(num_sequences)
    ]
    mean_lev = sum(lev_similarities) / num_sequences
    
    # Add batch size for logging
    result = {
        "bleu": mean_bleu,
        "levenshtein": mean_lev,
        "batch_size": num_sequences
    }
    
    return result


def masked_accuracy(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    pad_token_id: int
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
    # Get the predicted tokens
    pred_tokens = torch.argmax(predictions, dim=-1)
    
    # Create a mask for non-padding tokens
    mask = (targets != pad_token_id)
    
    # Calculate accuracy only for non-padding tokens
    correct = torch.sum((pred_tokens == targets) * mask).item()
    total = torch.sum(mask).item()
    
    # Avoid division by zero
    if total == 0:
        return 0.0, 0
    
    accuracy = correct / total
    return accuracy, total