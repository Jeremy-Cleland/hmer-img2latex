# Image to LaTeX Converter (img2latex)

## Overview

This project implements a deep learning-based tool for converting images of mathematical expressions into LaTeX code. It uses a sequence-to-sequence architecture with either a CNN or ResNet encoder combined with an LSTM decoder to generate LaTeX output from input images.

## Features

- Two model types:
  - `cnn_lstm`: A CNN-based encoder with an LSTM decoder
  - `resnet_lstm`: A ResNet-based encoder (pre-trained) with an LSTM decoder
- Command-line interface for training, prediction, and evaluation
- Support for Apple Silicon (M-series) through MPS acceleration
- Optional beam search decoding for improved prediction quality
- Evaluation using BLEU and Levenshtein distance metrics
- Comprehensive metrics and analysis tools

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/yourusername/hmer-im2latex.git
cd hmer-im2latex
pip install -e .
```

## Data Preparation

The system uses the IM2LaTeX-100k dataset with the following structure:

```
data/
├── img/                  # Directory with formula images
│   ├── *.png             # Images of math formulas
├── im2latex_formulas.norm.lst   # Formulas in normalized format
├── im2latex_train_filter.lst    # Training split
├── im2latex_validate_filter.lst # Validation split
└── im2latex_test_filter.lst     # Test split
```

Each line in the split files has the format: `<image_file> <formula_index>`, where `<formula_index>` is the line number in the formulas file.

## Commands

### Training and Inference

```bash
# Train a new model
make train EXPERIMENT=experiment_name CONFIG=path/to/config.yaml

# Resume training from a checkpoint
make train-resume MODEL=path/to/checkpoint.pt EXPERIMENT=experiment_name

# Run prediction on an image
make predict MODEL=path/to/checkpoint.pt IMAGE=path/to/image.png

# Evaluate model on test set
make evaluate MODEL=path/to/checkpoint.pt
```

### CLI Commands

You can also use the CLI directly for more options:

```bash
# Training with custom parameters
python -m img2latex.cli train --config-path path/to/config.yaml --experiment-name experiment_name --device cuda

# Prediction with beam search
python -m img2latex.cli predict checkpoint.pt image.png --beam-size 5 --max-length 150

# Evaluation with custom batch size
python -m img2latex.cli evaluate checkpoint.pt data_dir --split test --batch-size 64 --beam-size 3
```

### Metrics and Analysis

```bash
# Visualize metrics for an experiment
make metrics-visualize EXPERIMENT=experiment_name

# Show latest metrics in a concise format
make metrics-latest EXPERIMENT=experiment_name

# Compare metrics across different experiments
make metrics-compare

# Export metrics to CSV or JSON
make metrics-export EXPERIMENT=experiment_name

# Run specific analysis tools
make analyze-images    # Analyze dataset images
make analyze-curves    # Plot learning curves
make analyze-tokens    # Analyze token distributions
make analyze-errors    # Analyze prediction errors
make analyze-preprocess # Visualize preprocessing steps

# Run all analysis tools
make analyze-all
```

### Development and Maintenance

```bash
# Initialize required directories
make dirs

# Lint code
make lint

# Format code
make format

# Run type checking
make typecheck

# Run all code quality checks
make check-all

# Clean Python artifacts
make clean-pyc

# Clean all outputs
make clean-outputs

# Clean only metrics files
make clean-metrics

# Clean everything
make clean-all

# Show help message with all commands
make help
```

For detailed information about the metrics system, see [README_METRICS.md](README_METRICS.md).

## Dataset Analysis

We conducted a comprehensive analysis of the 103,536 images in the dataset with the following findings:

#### Image Size Statistics

- **Width**: Range from 128 to 800 pixels (mean: 319.2px)
- **Height**: Range from 32 to 800 pixels (mean: 61.2px)
- **Aspect Ratio**: Range from 1.00 to 15.00 (mean: 5.79)
- **Most Common Size**: 320x64 pixels (11,821 images)

#### Color and Pixel Analysis

- **Mode**: All images are RGB (3 channels)
- **Data Type**: uint8 (0-255 range)
- **Pixel Statistics**: Mean value = 241.51, Standard deviation = 46.84
- **Background**: Predominantly white backgrounds

#### Implementation Strategy

Based on this analysis, our implementation:

1. Resizes all images to a fixed height of 64px (maintaining aspect ratio)
2. Pads width to 800px to accommodate all formulas without information loss
3. Converts RGB to grayscale for CNN models (keeping RGB for ResNet models)
4. Normalizes pixel values from [0-255] to [0-1] range

## Configuration

The `config.yaml` file contains settings for the model, training, and data. Key parameters include:

- Model type (`cnn_lstm` or `resnet_lstm`)
- Image dimensions:
  - Height: 64 pixels (fixed)
  - Width: 800 pixels (padded)
  - Channels: 1 for CNN, 3 for ResNet
- Maximum sequence length: 141 tokens (95th percentile of dataset formula lengths)
- Embedding and hidden dimensions
- Training parameters (learning rate, batch size, etc.)
- Device selection (MPS, CUDA, or CPU)

## Model Architecture

The model consists of two main components:

1. **Encoder**:
   - CNN: Processes grayscale images with 3 convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool)
   - ResNet: Uses pre-trained ResNet50 (or variants) that processes RGB images

2. **Decoder**:
   - LSTM-based decoder with attention mechanism
   - Teacher forcing during training
   - Greedy or beam search decoding during inference

## Preprocessing Pipeline

Our image preprocessing pipeline includes:

1. **Resizing**: Height fixed at 64px while maintaining the aspect ratio
2. **Padding**: Width padded to 800px (right padding with white)
3. **Channel Conversion**:
   - CNN: RGB to grayscale (1 channel)
   - ResNet: Keep as RGB (3 channels)
4. **Normalization**:
   - ToTensor(): Converts uint8 [0,255] to float [0,1]
   - Normalize(): Standardizes using appropriate means and standard deviations
5. **Data Augmentation** (training only):
   - Small rotations
   - Slight scaling
   - Random crops with padding

## Performance Optimizations

The implementation includes several optimizations:

1. **Batch Processing**: Efficient batch prediction for evaluation
2. **Formula Caching**: Formulas are preloaded to eliminate repeated disk reads
3. **Root Path Detection**: Robust directory structure detection for reliable execution

## License

MIT License
