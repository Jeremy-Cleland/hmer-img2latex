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

## Usage

### Training

```bash
img2latex train --config-path img2latex/configs/config.yaml --experiment-name my_experiment
```

### Prediction

```bash
img2latex predict path/to/checkpoint.pt path/to/image.png
```

### Evaluation

```bash
img2latex evaluate path/to/checkpoint.pt path/to/data/directory --split test
```

## Configuration

The `config.yaml` file contains settings for the model, training, and data. Key parameters include:

- Model type (`cnn_lstm` or `resnet_lstm`)
- Image dimensions
- Embedding and hidden dimensions
- Training parameters (learning rate, batch size, etc.)
- Device selection (MPS, CUDA, or CPU)

## Model Architecture

The model consists of two main components:

1. **Encoder**:
   - CNN: 3 convolutional blocks (Conv2D -> MaxPool)
   - ResNet: Pre-trained ResNet50 (or other variants)

2. **Decoder**:
   - LSTM-based decoder with optional attention
   - Teacher forcing during training
   - Greedy or beam search decoding during inference

## License

MIT License