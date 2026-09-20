"""Tiny forward and generate tests for the Transformer seq2seq model."""

import torch

from img2latex.model.seq2seq import Seq2SeqModel


def test_seq2seq_forward_and_greedy():
    vocab = 32
    model = Seq2SeqModel(
        model_type="cnn_transformer",
        vocab_size=vocab,
        encoder_params={
            "img_height": 64,
            "img_width": 512,
            "channels": 1,
            "embedding_dim": 64,
            "conv_filters": [16, 32, 32],
        },
        decoder_params={
            "hidden_dim": 64,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 128,
            "dropout": 0.0,
            "max_seq_length": 20,
        },
        pad_token_id=0,
    )
    images = torch.randn(2, 1, 64, 512)
    widths = torch.tensor([300, 512])
    formulas = torch.randint(3, vocab, (2, 12))
    formulas[:, 0] = 1
    formulas[:, -1] = 2
    logits = model(images, formulas, valid_widths=widths)
    assert logits.shape == (2, 11, vocab)
    sequences = model.generate(images, start_token_id=1, end_token_id=2, max_length=8, beam_size=1)
    assert len(sequences) == 2
    beams = model.generate(images, start_token_id=1, end_token_id=2, max_length=6, beam_size=3)
    assert len(beams) == 2
