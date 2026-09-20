# Image to LaTeX Converter (img2latex)

[![Deploy to GitHub Pages](https://github.com/Jeremy-Cleland/hmer-img2latex/actions/workflows/deploy.yml/badge.svg)](https://github.com/Jeremy-Cleland/hmer-img2latex/actions/workflows/deploy.yml)

**View Project Website:** [https://jeremy-cleland.github.io/hmer-img2latex/](https://jeremy-cleland.github.io/hmer-img2latex/)

## Overview

The Image to LaTeX (`img2latex`) project converts images of printed mathematical expressions into LaTeX. It is trained on the IM2LaTeX-100k dataset.

The original CNN-LSTM / ResNet-LSTM models reported 62.56% token accuracy and 0.15 BLEU after 25 epochs. Those numbers were **teacher-forced** (ground-truth tokens fed at every step) and the encoder collapsed each image to a single vector, so the attention layer was a no-op. This revision replaces that stack with a spatial CNN encoder and a Transformer decoder, and reports BLEU, exact match, and edit distance from **autoregressive** decoding.

## Architecture

```
Image 64x512 grayscale
  -> Conv-BN-ReLU-MaxPool x3  (downsample /8)
  -> 1x1 projection to d_model=256
  -> 2D sinusoidal positional encoding
  -> feature grid 8x64 = 512 memory tokens + padding mask
  -> Transformer decoder (4 layers, 8 heads, pre-norm, tied embeddings)
  -> token logits
```

The encoder keeps the spatial layout of the formula. White right-padding is masked out of cross-attention so the decoder cannot attend to empty canvas.

Optional `resnet_transformer` config uses ResNet-18 with the classification head and avg-pool removed and `layer4` stride set to 1 (grid /16).

## Why the old score was stuck at 0.15 BLEU

1. **No spatial features.** Both encoders reduced the image to one vector (`Flatten + Linear` on the CNN, `avgpool` kept on ResNet). Attention over a sequence of length 1 is always 1.0.
2. **Teacher-forced metrics.** Validation ran `argmax` on teacher-forced logits, not generated text.
3. **Broken inference.** Beam search was hard-disabled and predict resized to 64x800 while training used 128x800.

## Metrics

Reported scores are computed on generated token sequences:

| Metric | Meaning |
|--------|---------|
| BLEU-4 | Smoothed corpus BLEU (`nltk` method3), not a per-sentence geometric mean that zeros out short formulas |
| Exact match | Generated tokens equal the reference after stripping START/END/PAD |
| Normalized edit distance | Mean Levenshtein / max(len); lower is better |
| Token accuracy | Teacher-forced, logged only as a training diagnostic |

Best checkpoints are selected on validation BLEU, not validation loss.

Legacy teacher-forced numbers (for comparison only): accuracy 62.56%, BLEU 0.1539, Levenshtein similarity 0.2829. No old checkpoint was in the repo, so those runs could not be re-scored honestly. See `outputs/baseline_honest.json`.

Latest Transformer results are written to `outputs/img2latex_transformer_v1/metrics/metrics.json` during training.

## Training setup

- **Optimizer:** AdamW, lr 3e-4, weight decay 1e-4
- **Schedule:** linear warmup (5% of steps) then cosine decay
- **Loss:** cross-entropy with label smoothing 0.1, pad ignored
- **Grad clip:** 1.0
- **Precision:** fp32 on Apple Silicon MPS (no GradScaler)
- **Data:** 64x512 scale-to-fit (never cropped), sequence truncation at 141, tokenizer fit on the **train split only** with `min_freq=5`
- **Loading:** uint8 memmap cache, length-bucketed batches (DataLoader workers stay at 0 on macOS because libomp + multiprocessing segfaults)
- **Decode:** batched greedy during validation; beam search (size 5, length penalty 0.7) for test/predict

## Dataset

IM2LaTeX-100k processed images (hashed PNG names) plus the filtered splits:

```
data/
├── img/                             # processed formula PNGs
├── cache/                           # 64x512 uint8 memmap (built once)
├── im2latex_formulas.norm.lst
├── im2latex_train_filter.lst
├── im2latex_validate_filter.lst
└── im2latex_test_filter.lst
```

## Installation (Apple Silicon / MPS)

Miniforge is required. From the repo root:

```bash
# If conda is not installed:
curl -fsSL -o /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"

conda env create -f environment.yml
conda activate hmer-im2latex
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1
conda activate hmer-im2latex   # reload vars
pip install -e .
```

The env installs the official macOS ARM PyTorch wheels, which include MPS. Confirm with:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

## Data preparation

```bash
make download-data    # ~588MB processed images
make build-cache      # ~3GB uint8 memmap at 64x512
```

## Commands

```bash
# Train (writes outputs/<name>_vN/)
make train EXPERIMENT=img2latex_transformer CONFIG=img2latex/configs/config.yaml

# Resume
make train-resume MODEL=outputs/img2latex_transformer_v1/checkpoints/best_checkpoint.pt EXPERIMENT=img2latex_transformer

# Predict with beam search
python -m img2latex.cli predict path/to/best_checkpoint.pt path/to/image.png --beam-size 5

# Evaluate on the test split
python -m img2latex.cli evaluate path/to/best_checkpoint.pt data --split test --beam-size 5 --device mps
```

```bash
make lint
make format
python -m pytest tests -q
```

Set `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` if pytest hits the duplicate OpenMP runtime on macOS.

## License

MIT License
