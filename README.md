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

### Dataset Analysis

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
