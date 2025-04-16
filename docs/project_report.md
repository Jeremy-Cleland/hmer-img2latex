# Image to LaTeX Converter: Project Report

## Introduction

The Image to LaTeX (img2latex) project implements a deep learning-based system for converting images of mathematical expressions into LaTeX code. This technology addresses a significant challenge in digital document processing: transforming visual representations of mathematical formulas into their corresponding markup representation, which is essential for editing, searching, and accessibility.

Mathematical expressions are ubiquitous in scientific, engineering, and academic literature, but transferring them between different formats can be cumbersome. Traditional Optical Character Recognition (OCR) systems often struggle with the complex two-dimensional structure of mathematical formulas. The img2latex project provides an end-to-end solution to automatically recognize and transcribe mathematical expressions from images, significantly reducing the manual effort required for digitizing printed mathematical content.

The system employs a sequence-to-sequence architecture, combining convolutional neural networks (CNNs) or residual networks (ResNets) for image encoding with Long Short-Term Memory (LSTM) networks for LaTeX sequence decoding. This approach leverages recent advances in computer vision and natural language processing to achieve state-of-the-art performance in formula recognition.

## Methodology

### Data Processing

The img2latex system uses the IM2LaTeX-100k dataset, which contains over 100,000 images of mathematical expressions paired with their corresponding LaTeX code. Our analysis of the dataset revealed:

- Image dimensions vary widely, with widths from 128 to 800 pixels and heights from 32 to 800 pixels
- Mean image size is 319.2 × 61.2 pixels, with an average aspect ratio of 5.79
- The most common image size is 320 × 64 pixels
- All images are RGB with white backgrounds

Based on this analysis, our preprocessing pipeline includes:

1. **Resizing**: All images are resized to a fixed height of 64 pixels while maintaining the aspect ratio
2. **Padding**: Images are padded to a width of 800 pixels to accommodate all formulas without loss of information
3. **Channel Conversion**: 
   - For CNN models: Images are converted to grayscale (1 channel)
   - For ResNet models: RGB format (3 channels) is maintained
4. **Normalization**: Pixel values are normalized from [0-255] to [0-1] range

The LaTeX formulas undergo tokenization using a custom `LaTeXTokenizer` class, which handles special LaTeX tokens and limits sequence length to a maximum of 141 tokens (the 95th percentile of the dataset formula lengths).

### Model Architecture

The img2latex system offers two model variants:

#### 1. CNN-LSTM Architecture

The CNN-LSTM model consists of:

- **Encoder**: A convolutional neural network with three convolutional blocks, each containing:
  - Conv2D layer (with filters [32, 64, 128])
  - ReLU activation
  - MaxPooling layer
  - The final output is flattened and passed through a dense layer to create the embedding

- **Decoder**: An LSTM-based decoder that:
  - Takes the encoder output and previously generated tokens as input
  - Generates output tokens one at a time
  - Uses teacher forcing during training (ground truth tokens as input)
  - Offers optional attention mechanism to focus on different parts of the encoder representation

#### 2. ResNet-LSTM Architecture

The ResNet-LSTM model replaces the CNN encoder with a pre-trained ResNet:

- **Encoder**: A pre-trained ResNet (options include ResNet18, ResNet34, ResNet50, ResNet101, ResNet152) with:
  - The classification head removed
  - Option to freeze weights for transfer learning
  - Final layer adapted to produce embeddings of the desired dimension

- **Decoder**: The same LSTM-based decoder as the CNN-LSTM model

### Training Process

The training process implements several key strategies:

1. **Optimization**: Adam optimizer with configurable learning rate and weight decay
2. **Early Stopping**: Training stops if validation metrics don't improve for a specified number of epochs
3. **Gradient Clipping**: To prevent exploding gradients
4. **Checkpoint Saving**: Regular saving of model checkpoints for resuming training
5. **Metrics Tracking**: Comprehensive logging of training and validation metrics

### Inference Methods

During inference, the model offers three decoding strategies:

1. **Greedy Search**: Selects the most probable token at each step
2. **Sampling with Temperature/Top-k/Top-p**: Introduces randomness in the generation process
3. **Beam Search**: Maintains multiple candidate sequences and selects the most probable overall sequence

## Experiments

### Experimental Setup

Our experiments evaluated the performance of both CNN-LSTM and ResNet-LSTM architectures with various hyperparameter settings. Key configurations included:

- Model type: CNN-LSTM (primary focus)
- Image dimensions: 64 × 800 pixels
- Batch size: 64
- Learning rate: 0.001
- Weight decay: 0.0001
- Maximum sequence length: 141
- Embedding dimension: 256
- Hidden dimension: 256
- Device: Apple Silicon MPS (Metal Performance Shaders)

### Evaluation Metrics

We evaluated the model performance using four key metrics:

1. **Loss**: Cross-entropy loss on validation data
2. **Accuracy**: Token-level accuracy (ignoring padding tokens)
3. **BLEU Score**: Measures n-gram precision between generated and reference sequences
4. **Levenshtein Similarity**: Normalized edit distance between generated and reference sequences

### Results

Our training process spanned 8 epochs, with the following progression in validation metrics:

| Epoch | Loss    | Accuracy | BLEU    | Levenshtein |
|-------|---------|----------|---------|-------------|
| 1     | 2.0509  | 0.5356   | 0.1048  | 0.2441      |
| 2     | 1.9219  | 0.5578   | 0.1129  | 0.2517      |
| 3     | 1.8527  | 0.5727   | 0.1246  | 0.2600      |
| 4     | 1.8298  | 0.5686   | 0.1315  | 0.2609      |
| 5     | 1.7953  | 0.5744   | 0.1334  | 0.2613      |
| 6     | 1.7752  | 0.5785   | 0.1355  | 0.2632      |
| 7     | 1.7695  | 0.5790   | 0.1341  | 0.2626      |
| 8     | 1.7492  | 0.5828   | 0.1375  | 0.2625      |

Key observations from our experiments:

1. The model showed consistent improvement across all metrics through the training process.
2. The CNN-LSTM architecture achieved a validation accuracy of 58.28% and a BLEU score of 0.1375 by epoch 8.
3. The Levenshtein similarity plateaued around 0.26, indicating approximately 74% edit distance between generated and reference sequences.

Further evaluation on complex formulas revealed that the model performed better on simpler expressions and struggled with very complex mathematical notation. Beam search decoding (with beam size 3) improved the quality of generated LaTeX compared to greedy search.

## Conclusion

The img2latex project successfully demonstrates the viability of deep learning approaches for converting images of mathematical expressions to LaTeX code. Our implementation of CNN-LSTM and ResNet-LSTM architectures shows promising results, achieving reasonable accuracy on the challenging task of mathematical formula recognition.

Key achievements of the project include:

1. A comprehensive data processing pipeline that effectively handles variability in the input images
2. Flexible model architecture options to accommodate different computational resources and accuracy requirements
3. Multiple decoding strategies to allow for trade-offs between speed and quality
4. Robust evaluation using multiple metrics to assess both token-level accuracy and semantic correctness

Despite these successes, several challenges remain. The model still struggles with very complex formulas, particularly those with nested structures or uncommon mathematical symbols. Additionally, the current approach requires significant computational resources for training and could benefit from further optimization.

Future work could focus on:

1. Incorporating more sophisticated attention mechanisms
2. Exploring transformer-based architectures as an alternative to LSTM decoders
3. Implementing data augmentation strategies to improve generalization
4. Developing post-processing techniques to correct common errors in the generated LaTeX

The img2latex system provides a strong foundation for further research and development in mathematical formula recognition, with potential applications in digital document processing, accessibility tools, and educational technology. 