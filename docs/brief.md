

## Introduction

Good afternoon, everyone. Thank you for joining my presentation on the **Image to LaTeX Converter** project. My name is Jeremy Cleland, and I’m a graduate student in Advanced AI at the University of Michigan–Dearborn. Today, I’ll walk you through the motivation behind this work, our overall approach, the experiments we conducted, and our concluding insights.

- **Motivation**  
  Mathematical expressions appear everywhere in scientific, engineering, and academic documents. Yet converting those visual formulas into editable LaTeX code remains a laborious, error‑prone process. Traditional OCR systems excel at plain text but falter on the two‑dimensional structure of math. Our goal was to build an end‑to‑end, deep learning–based system that automatically transcribes math images into LaTeX markup, streamlining document digitization and improving accessibility.

- **Contributions**  
  We designed two model variants—CNN‑LSTM and ResNet‑LSTM—that leverage modern computer vision and sequence modeling techniques. We developed a robust preprocessing pipeline, implemented advanced training strategies, and conducted a thorough evaluation on the IM2LaTeX‑100k dataset, achieving promising accuracy and BLEU scores.

---

## Methodology

I’ll now describe the key components of our system: data preparation, model architectures, and training setup.

1. **Data Preparation**  
   - **Dataset**: IM2LaTeX‑100k, containing 103,536 images paired with LaTeX formulas.  
   - **Preprocessing**:  
     - **Resizing** images to a fixed height of 64 pixels while preserving aspect ratio.  
     - **Padding** to uniform width (800 px) with white background.  
     - **Channel conversion**: grayscale for CNN, RGB for ResNet.  
     - **Normalization** to [0,1] range.  
     - **Data augmentation** (rotations ±5°, scaling ±10%, random crops) to improve robustness.  
   - **Tokenization**: Custom `LaTeXTokenizer` that handles commands, limits sequence length to 141 tokens, and adds special `<sos>`, `<eos>`, and `<pad>` tokens.

2. **Model Architectures**  
   - **CNN‑LSTM**  
     - **Encoder**: Three Conv2D blocks (filters 32→64→128), each with ReLU, batch norm, and 2×2 max pooling; flattened embedding via a dense layer.  
     - **Decoder**: Two‑layer LSTM (hidden size 256) with Bahdanau attention, teacher forcing during training, and dropout (0.2).  
   - **ResNet‑LSTM**  
     - **Encoder**: Pretrained ResNet (e.g., ResNet50) with classification head removed; optional freezing of early layers; adapter layer for embedding dimension.  
     - **Decoder**: Identical to the CNN‑LSTM decoder, ensuring fair comparison.

3. **Training Setup**  
   - **Optimizer**: Adam (LR = 0.001, weight decay = 1e‑4).  
   - **LR Scheduler**: ReduceLROnPlateau (patience = 3, factor = 0.5).  
   - **Loss**: Cross‑entropy with label smoothing (0.1), ignoring `<pad>` tokens.  
   - **Techniques**: Scheduled sampling (teacher forcing → predictions), gradient clipping (norm ≤ 5.0), early stopping (no improvement after 5 epochs), and checkpointing on best BLEU.  
   - **Hardware**: Mixed‑precision training on Apple M4 Max via MPS, with CPU fallback.

---

## Experiments

Next, I’ll present our evaluation protocol and key results.

1. **Experimental Configuration**  
   - **Image size**: 64 × 800 px  
   - **Batch size**: 64  
   - **Max sequence length**: 141 tokens  
   - **Embedding & hidden dim**: 256  

2. **Metrics**  
   - **Loss**: Validation cross‑entropy  
   - **Accuracy**: Token‑level accuracy (excluding padding)  
   - **BLEU Score**: N‑gram precision between generated and reference sequences  
   - **Levenshtein Similarity**: 1 – (edit distance / reference length)

3. **Training Progression (Best Model: CNN‑LSTM)**  
   | Epoch | Loss   | Accuracy | BLEU   | Lev. Sim. |
   |-------|--------|----------|--------|-----------|
   | 1     | 2.2778 | 49.86%   | 0.0827 | 0.2311    |
   | 5     | 1.8408 | 57.60%   | 0.1241 | 0.2609    |
   | 10    | 1.6909 | 60.22%   | 0.1377 | 0.2716    |
   | 15    | 1.6338 | 61.16%   | 0.1464 | 0.2781    |
   | 20    | 1.6030 | 61.80%   | 0.1502 | 0.2799    |
   | 25    | 1.5663 | 62.56%   | 0.1539 | 0.2829    |

4. **Architecture Comparison**  
   - **CNN‑LSTM**: 62.56% accuracy, BLEU = 0.1539  
   - **ResNet50‑LSTM**: 59.42% accuracy, BLEU = 0.1487 (fewer epochs)  
   - **Observation**: CNN‑LSTM outperforms with lower compute requirements.

5. **Error Analysis & Decoding Strategies**  
   - **Common errors**: Misplaced brackets, symbol confusions (1/l vs. 0/O), complex nested structures.  
   - **Beam search (size 3)**: +7.2% BLEU over greedy search, validating the benefit of exploring multiple hypotheses.

---

## Conclusion

To conclude:

- **Achievements**  
  1. Developed a robust preprocessing pipeline for diverse formula images.  
  2. Designed and compared two seq2seq architectures (CNN‑LSTM vs. ResNet‑LSTM).  
  3. Achieved token‑level accuracy of 62.56% and BLEU score of 0.1539 on a challenging dataset.  
  4. Demonstrated the efficacy of attention, scheduled sampling, and beam search in formula transcription.

- **Limitations**  
  - Struggles with highly complex or nested formulas.  
  - Significant computational resources required for longer sequences.

- **Future Directions**  
  1. **Architectural enhancements**: Multi‑head attention or transformer decoders.  
  2. **Advanced data augmentation** and synthetic data for rare formula types.  
  3. **Curriculum learning** by formula complexity.  
  4. **User‑facing tools**: Interactive UI for real‑time transcription and error correction.

Thank you for your attention. I welcome any questions or feedback.