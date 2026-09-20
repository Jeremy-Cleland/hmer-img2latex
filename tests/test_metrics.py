"""Tests for smoothed corpus BLEU, exact match, and edit distance."""

from img2latex.training.metrics import calculate_metrics, corpus_bleu_score, exact_match_rate


def test_corpus_bleu_does_not_collapse_on_short_mismatch():
    pred = [[1, 2, 3]]
    target = [[1, 2, 4]]
    score = corpus_bleu_score(pred, target, n=4)
    assert score > 0.0


def test_exact_match_and_edit_distance():
    predictions = [[1, 2, 3], [4, 5]]
    targets = [[1, 2, 3], [4, 6]]
    metrics = calculate_metrics(predictions, targets)
    assert metrics["exact_match"] == 0.5
    assert 0.0 < metrics["edit_distance"] < 1.0
    assert metrics["bleu"] > 0.0
    assert exact_match_rate(predictions, predictions) == 1.0


def test_special_token_stripping():
    pad, start, end = 0, 1, 2
    predictions = [[start, 5, 6, end, pad, pad]]
    targets = [[start, 5, 6, end]]
    metrics = calculate_metrics(
        predictions, targets, pad_token_id=pad, start_token_id=start, end_token_id=end
    )
    assert metrics["exact_match"] == 1.0
    assert metrics["edit_distance"] == 0.0
