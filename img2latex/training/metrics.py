"""
Evaluation metrics for the image-to-LaTeX model.

BLEU is corpus-level with Chen-Cherry method3 smoothing so short formulas
do not collapse to zero. Exact match and normalized edit distance are the
metrics typically reported alongside BLEU for im2latex.
"""

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.utils.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

_SMOOTHING = SmoothingFunction().method3


def _detach_to_cpu(tensor_or_list: Union[torch.Tensor, List, Any]) -> Union[torch.Tensor, List, Any]:
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list.detach().cpu()
    if isinstance(tensor_or_list, list):
        return [
            t.detach().cpu().tolist()
            if isinstance(t, torch.Tensor)
            else _detach_to_cpu(t)
            if isinstance(t, list)
            else t
            for t in tensor_or_list
        ]
    return tensor_or_list


def _clean_sequence(
    sequence: Sequence[int],
    pad_token_id: Optional[int] = None,
    start_token_id: Optional[int] = None,
    end_token_id: Optional[int] = None,
) -> List[int]:
    cleaned: List[int] = []
    skip = {idx for idx in (pad_token_id, start_token_id) if idx is not None}
    for token in sequence:
        token = int(token)
        if token in skip:
            continue
        if end_token_id is not None and token == end_token_id:
            break
        cleaned.append(token)
    return cleaned


def levenshtein_distance(sequence_one: List[int], sequence_two: List[int]) -> float:
    """Normalized Levenshtein similarity in [0, 1] (higher is better)."""
    rows = len(sequence_one)
    cols = len(sequence_two)
    dist_tab = np.zeros((rows + 1, cols + 1), dtype=int)
    for i in range(1, rows + 1):
        dist_tab[i][0] = i
    for i in range(1, cols + 1):
        dist_tab[0][i] = i
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if sequence_one[r - 1] == sequence_two[c - 1]:
                dist_tab[r][c] = dist_tab[r - 1][c - 1]
            else:
                dist_tab[r][c] = 1 + min(dist_tab[r - 1][c], dist_tab[r][c - 1], dist_tab[r - 1][c - 1])
    raw_distance = int(dist_tab[rows][cols])
    max_length = max(rows, cols)
    if max_length == 0:
        return 1.0
    return 1.0 - (raw_distance / max_length)


def raw_levenshtein(sequence_one: List[int], sequence_two: List[int]) -> int:
    rows = len(sequence_one)
    cols = len(sequence_two)
    prev = list(range(cols + 1))
    for r, a in enumerate(sequence_one, start=1):
        curr = [r]
        for c, b in enumerate(sequence_two, start=1):
            if a == b:
                curr.append(prev[c - 1])
            else:
                curr.append(1 + min(prev[c], curr[-1], prev[c - 1]))
        prev = curr
    return prev[-1]


def bleu_n_score(generated_sequence: List[int], true_sequence: List[int], n: int = 4) -> float:
    """Sentence BLEU kept for compatibility; prefer corpus_bleu_score for reporting."""
    if not generated_sequence or not true_sequence:
        return 0.0
    return float(
        corpus_bleu(
            [[true_sequence]],
            [generated_sequence],
            weights=tuple(1.0 / n for _ in range(n)),
            smoothing_function=_SMOOTHING,
        )
    )


def corpus_bleu_score(
    predictions: List[List[int]],
    targets: List[List[int]],
    n: int = 4,
) -> float:
    """Smoothed corpus BLEU-n over token-id sequences."""
    if not predictions:
        return 0.0
    list_of_references = [[ref] for ref in targets]
    weights = tuple(1.0 / n for _ in range(n))
    return float(corpus_bleu(list_of_references, predictions, weights=weights, smoothing_function=_SMOOTHING))


def exact_match_rate(predictions: List[List[int]], targets: List[List[int]]) -> float:
    if not predictions:
        return 0.0
    hits = sum(pred == tgt for pred, tgt in zip(predictions, targets))
    return hits / len(predictions)


def mean_normalized_edit_distance(predictions: List[List[int]], targets: List[List[int]]) -> float:
    """Mean Levenshtein distance / max(len). Lower is better."""
    if not predictions:
        return 1.0
    total = 0.0
    for pred, tgt in zip(predictions, targets):
        denom = max(len(pred), len(tgt), 1)
        total += raw_levenshtein(pred, tgt) / denom
    return total / len(predictions)


def calculate_metrics(
    predictions: List[List[int]],
    targets: List[List[int]],
    pad_token_id: Optional[int] = None,
    start_token_id: Optional[int] = None,
    end_token_id: Optional[int] = None,
) -> Dict[str, float]:
    predictions = [_clean_sequence(p, pad_token_id, start_token_id, end_token_id) for p in _detach_to_cpu(predictions)]
    targets = [_clean_sequence(t, pad_token_id, start_token_id, end_token_id) for t in _detach_to_cpu(targets)]
    assert len(predictions) == len(targets), "Predictions and targets must have the same length"

    bleu = corpus_bleu_score(predictions, targets, n=4)
    lev_sims = [levenshtein_distance(p, t) for p, t in zip(predictions, targets)]
    return {
        "bleu": bleu,
        "levenshtein": sum(lev_sims) / len(lev_sims) if lev_sims else 0.0,
        "exact_match": exact_match_rate(predictions, targets),
        "edit_distance": mean_normalized_edit_distance(predictions, targets),
        "batch_size": len(predictions),
    }


def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_token_id: int) -> Tuple[float, int]:
    """Token accuracy computed on-device. Returns (correct_count, total_count)."""
    pred = logits.argmax(dim=-1)
    mask = targets.ne(pad_token_id)
    correct = torch.logical_and(pred.eq(targets), mask).sum().item()
    total = mask.sum().item()
    return correct, total


def token_list_accuracy(
    predictions: List[List[int]], targets: List[List[int]], pad_token_id: int
) -> Tuple[float, int]:
    total_correct = 0
    total_tokens = 0
    for pred_seq, target_seq in zip(predictions, targets):
        min_len = min(len(pred_seq), len(target_seq))
        correct = sum(
            1
            for i in range(min_len)
            if pred_seq[i] == target_seq[i] and target_seq[i] != pad_token_id
        )
        non_pad = sum(1 for t in target_seq[:min_len] if t != pad_token_id)
        total_correct += correct
        total_tokens += non_pad
    return total_correct, total_tokens


def analyze_token_distribution(
    predictions: List[List[int]],
    targets: List[List[int]],
    tokenizer: LaTeXTokenizer,
    top_k: int = 10,
) -> Dict[str, Any]:
    predictions = _detach_to_cpu(predictions)
    targets = _detach_to_cpu(targets)
    pred_tokens_flat = [token for seq in predictions for token in seq]
    target_tokens_flat = [token for seq in targets for token in seq]
    pred_counter = Counter(pred_tokens_flat)
    target_counter = Counter(target_tokens_flat)
    pred_most_common = pred_counter.most_common(top_k)
    target_most_common = target_counter.most_common(top_k)
    pred_probs = np.array([count / len(pred_tokens_flat) for _, count in pred_counter.items()]) if pred_tokens_flat else np.array([])
    target_probs = np.array([count / len(target_tokens_flat) for _, count in target_counter.items()]) if target_tokens_flat else np.array([])
    def _entropy(probs: np.ndarray) -> float:
        probs = probs[probs > 0]
        if probs.size == 0:
            return 0.0
        return float(-(probs * np.log(probs)).sum())

    pred_entropy = _entropy(pred_probs) if len(pred_probs) > 0 else 0
    target_entropy = _entropy(target_probs) if len(target_probs) > 0 else 0
    pred_most_common_readable = [
        (tokenizer.id_to_token.get(token_id, "<UNK>"), count) for token_id, count in pred_most_common
    ]
    target_most_common_readable = [
        (tokenizer.id_to_token.get(token_id, "<UNK>"), count) for token_id, count in target_most_common
    ]
    repetition_factor = pred_most_common[0][1] / len(pred_tokens_flat) if pred_most_common else 0
    pred_diversity = len(pred_counter) / len(pred_tokens_flat) if pred_tokens_flat else 0
    target_diversity = len(target_counter) / len(target_tokens_flat) if target_tokens_flat else 0
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
    outputs_cpu = _detach_to_cpu(outputs)
    targets_cpu = _detach_to_cpu(targets)
    batch_size = outputs_cpu.shape[0]
    probs = torch.nn.functional.softmax(outputs_cpu, dim=-1)
    pred_tokens = torch.argmax(probs, dim=-1)
    pred_probs = torch.max(probs, dim=-1)[0]
    pred_tokens_np = pred_tokens.numpy()
    pred_probs_np = pred_probs.numpy()
    targets_np = targets_cpu.numpy()
    samples = []
    for i in range(min(batch_size, num_samples)):
        pred_mask = pred_tokens_np[i] != tokenizer.pad_token_id
        target_mask = targets_np[i] != tokenizer.pad_token_id
        pred_seq = pred_tokens_np[i][pred_mask]
        target_seq = targets_np[i][target_mask]
        pred_confidences = pred_probs_np[i][pred_mask]
        pred_latex = tokenizer.decode(pred_seq.tolist())
        target_latex = tokenizer.decode(target_seq.tolist())
        low_confidence_indices = np.where(pred_confidences < confidence_threshold)[0]
        low_confidence_tokens = []
        for idx in low_confidence_indices:
            if idx < len(pred_seq):
                token_id = pred_seq[idx]
                token = tokenizer.id_to_token.get(token_id, "<UNK>")
                low_confidence_tokens.append((token, float(pred_confidences[idx])))
        samples.append(
            {
                "prediction": pred_latex,
                "target": target_latex,
                "low_confidence_tokens": low_confidence_tokens,
                "token_by_token": [
                    {
                        "pred_token": tokenizer.id_to_token.get(t, "<UNK>"),
                        "confidence": float(c),
                        "is_correct": bool(t == target_seq[i]) if i < len(target_seq) else None,
                    }
                    for i, (t, c) in enumerate(zip(pred_seq, pred_confidences))
                    if i < 20
                ],
            }
        )
    return {"samples": samples}


def save_enhanced_metrics(metrics: Dict[str, Any], experiment_name: str, metrics_dir: str, epoch: int) -> None:
    os.makedirs(metrics_dir, exist_ok=True)
    filename = f"{experiment_name}_enhanced_metrics_epoch_{epoch}.json"
    filepath = os.path.join(metrics_dir, filename)

    def convert_numpy(obj):
        if obj is None:
            return None
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        if isinstance(obj, torch.Tensor):
            return convert_numpy(obj.detach().cpu().numpy())
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    with open(filepath, "w") as f:
        json.dump(convert_numpy(metrics), f, indent=2)
    logger.info("Enhanced metrics saved to %s", filepath)


def log_enhanced_metrics_summary(metrics: Dict[str, Any]) -> None:
    token_dist = metrics.get("token_distribution", {})
    pred_info = token_dist.get("predictions", {})
    logger.info("Enhanced Metrics Summary:")
    if "repetition_factor" in pred_info and pred_info["repetition_factor"] > 0.5:
        logger.warning(
            "High token repetition detected (factor: %.2f).",
            pred_info["repetition_factor"],
        )
    if "diversity" in pred_info:
        logger.info("Prediction diversity: %.4f", pred_info["diversity"])
    samples = metrics.get("samples", {}).get("samples", [])
    if samples:
        logger.info("Sample predictions analyzed: %s", len(samples))


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
    combined_metrics: Dict[str, Any] = {}
    all_predictions_cpu = _detach_to_cpu(all_predictions)
    all_targets_cpu = _detach_to_cpu(all_targets)

    if outputs is not None and targets is not None:
        outputs_cpu = _detach_to_cpu(outputs)
        targets_cpu = _detach_to_cpu(targets)
        correct_count, num_tokens = masked_accuracy(outputs_cpu, targets_cpu, tokenizer.pad_token_id)
        sample_data = sample_predictions_and_targets(
            outputs_cpu, targets_cpu, tokenizer, num_samples, confidence_threshold
        )
        combined_metrics["samples"] = sample_data
    else:
        correct_count, num_tokens = token_list_accuracy(
            all_predictions_cpu, all_targets_cpu, tokenizer.pad_token_id
        )
        combined_metrics["samples"] = {"samples": []}

    accuracy = correct_count / num_tokens if num_tokens > 0 else 0.0
    combined_metrics["accuracy"] = accuracy
    combined_metrics["num_tokens"] = num_tokens

    basic_metrics = calculate_metrics(
        all_predictions_cpu,
        all_targets_cpu,
        pad_token_id=tokenizer.pad_token_id,
        start_token_id=tokenizer.start_token_id,
        end_token_id=tokenizer.end_token_id,
    )
    combined_metrics.update(basic_metrics)
    combined_metrics["token_distribution"] = analyze_token_distribution(
        all_predictions_cpu, all_targets_cpu, tokenizer
    )
    if epoch is not None:
        combined_metrics["epoch"] = epoch
    if save_to_file:
        if not all([experiment_name, metrics_dir, epoch is not None]):
            logger.warning("Cannot save metrics to file: missing required parameters")
        else:
            save_enhanced_metrics(combined_metrics, experiment_name, metrics_dir, epoch)
    log_enhanced_metrics_summary(combined_metrics)
    return combined_metrics
