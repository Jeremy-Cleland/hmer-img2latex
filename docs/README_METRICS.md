# Metrics System for img2latex

This document explains how to use the unified metrics system in the img2latex project.

## Overview

The metrics system provides a comprehensive suite of measurements to evaluate the performance of your image-to-LaTeX models. It combines basic metrics (like BLEU, Levenshtein, accuracy) with enhanced metrics (token distribution analysis, visualization of low-confidence predictions).

Our unified metrics module merges what was previously separated into basic and enhanced metrics, creating a single cohesive system that handles all metric computation, visualization, and analysis.

## Automatic Metrics Collection

The system automatically collects metrics during training and validation. These metrics are saved as JSON files in:

```
outputs/<experiment_name>/metrics/
```

Each validation cycle generates a metrics file named `<experiment_name>_enhanced_metrics_epoch_<N>.json` with comprehensive metrics for that epoch.

## Using the Makefile

The Makefile provides convenient commands for working with metrics:

```bash
# Visualize metrics for an experiment with plots and samples
make metrics-visualize EXPERIMENT=your_experiment_name

# Show just the latest metrics in a concise format
make metrics-latest EXPERIMENT=your_experiment_name

# Compare metrics across different experiments
make metrics-compare

# Export metrics to CSV or JSON for further analysis
make metrics-export EXPERIMENT=your_experiment_name [FORMAT=csv|json]

# Clean metrics files if needed
make clean-metrics
```

## Using the CLI Directly

You can also use the CLI commands directly for more control:

```bash
# Visualize metrics for a specific experiment
python -m img2latex.cli analyze metrics visualize --experiment your_experiment_name [--epoch N] [--no-show-samples] [--no-show-token-dist] [--no-plot-history]

# Show latest metrics
python -m img2latex.cli analyze metrics latest --experiment your_experiment_name

# Compare experiments 
python -m img2latex.cli analyze metrics compare [--experiments exp1 exp2 ...] [--metric bleu|levenshtein|accuracy]

# Export metrics to a file
python -m img2latex.cli analyze metrics export --experiment your_experiment_name [--format csv|json] [--output path/to/output]
```

## Available Metrics

The unified metrics system tracks:

- **BLEU Score**: Measure of prediction quality using n-gram precision
  - Calculated with n-grams up to n=4
  - Includes brevity penalty for short predictions
  - Range: 0.0 (worst) to 1.0 (best)

- **Levenshtein Similarity**: Normalized edit distance between predictions and targets
  - Computed as 1 - (edit_distance / max_length)
  - Range: 0.0 (completely different) to 1.0 (identical)

- **Accuracy**: Token-level prediction accuracy
  - Correctly predicted tokens / total tokens
  - Ignores padding tokens
  - Range: 0.0 (no correct tokens) to 1.0 (all correct)

- **Token Distribution Analysis**:
  - Top tokens and their frequencies
  - Entropy of token distribution
  - Diversity (unique tokens / total tokens)
  - Repetition factor (how often the most common token appears)

- **Prediction Confidence**:
  - Highlights low-confidence tokens (below threshold)
  - Per-token confidence scores
  - Helps identify where the model is uncertain

## Interpreting Metrics Files

The JSON metrics files contain:

```json
{
  "accuracy": 0.85,
  "num_tokens": 2500,
  "bleu": 0.78,
  "levenshtein": 0.82,
  "batch_size": 32,
  "token_distribution": {
    "predictions": {
      "top_tokens": [["\\frac", 423], ["x", 312], ...],
      "entropy": 4.2,
      "diversity": 0.75,
      "repetition_factor": 0.12
    },
    "targets": {
      "top_tokens": [["\\frac", 412], ["x", 301], ...],
      "entropy": 4.5,
      "diversity": 0.78
    }
  },
  "samples": [
    {
      "prediction": "\\frac{x}{y}",
      "target": "\\frac{x}{y}",
      "low_confidence_tokens": [["y", 0.42]],
      "token_by_token": [
        {"pred_token": "\\frac", "confidence": 0.98, "is_correct": true},
        {"pred_token": "x", "confidence": 0.92, "is_correct": true},
        {"pred_token": "y", "confidence": 0.42, "is_correct": true}
      ]
    },
    ...
  ],
  "epoch": 10
}
```

## Common Metrics Commands and Examples

Here are some common tasks you might want to perform:

### View Latest Results

```bash
make metrics-latest EXPERIMENT=img2latex_v3
```

This shows a quick summary of the most recent metrics for a specific experiment.

### Compare Best Models

```bash
make metrics-compare
```

This displays a table comparing the best metrics across all your experiments.

### Generate CSV for Further Analysis

```bash
make metrics-export EXPERIMENT=img2latex_v3 FORMAT=csv
```

This creates a CSV file with metrics from all epochs, which you can import into Excel, pandas, or other analysis tools.

### Full Visualization with Samples

```bash
python -m img2latex.cli analyze metrics visualize --experiment img2latex_v3 --epoch 20
```

This shows comprehensive metrics visualization, including token distribution, prediction samples, and confidence scores for a specific epoch.

## Metrics Implementation Details

Our metrics system is composed of several key components:

1. **Core Metrics Calculation**
   - `levenshtein_distance()`: Calculates normalized Levenshtein similarity
   - `bleu_n_score()`: Calculates BLEU score with n-grams up to n=4
   - `masked_accuracy()`: Calculates token-level accuracy ignoring padding

2. **Enhanced Analysis**
   - `analyze_token_distribution()`: Analyzes token frequency and diversity
   - `sample_predictions_and_targets()`: Provides detailed prediction samples

3. **Unified Interface**
   - `compute_all_metrics()`: High-level function that computes all metrics at once
   - Handles tensor detachment, CPU conversion, cache clearing, and more

4. **Visualization and Export**
   - `print_prediction_samples()`: Renders prediction samples with confidence
   - `print_token_distribution()`: Displays token distribution analysis
   - `plot_metrics_over_time()`: Plots metrics trends across epochs

## Troubleshooting

If you encounter issues:

1. Make sure the experiment has completed at least one validation cycle
2. Check if the metrics directory exists: `outputs/<experiment_name>/metrics/`
3. Verify the JSON files are properly formatted
4. If using MPS or other accelerators, memory issues might prevent metrics generation - try using a smaller batch size
5. For "No metrics found" errors, ensure the experiment name matches exactly (case-sensitive)
6. If metrics files exist but can't be loaded, check for JSON formatting errors

## Additional Resources

For more details on the implementation, see:

- `img2latex/training/metrics.py`: Core metrics implementation
- `img2latex/analysis/metrics.py`: CLI commands for analyzing metrics
- `img2latex/training/trainer.py`: How metrics are collected during training
- `img2latex/utils/visualize_metrics.py`: Visualization utilities 