#!/usr/bin/env python
"""Encoder unit tests for the spatial-grid CNN/ResNet backends."""

import torch

from img2latex.model.encoder import CNNEncoder, ResNetEncoder


def test_cnn_encoder_grid_shape():
    img_height, img_width, channels = 64, 512, 1
    batch_size = 2
    embedding_dim = 256
    encoder = CNNEncoder(
        img_height=img_height,
        img_width=img_width,
        channels=channels,
        embedding_dim=embedding_dim,
    )
    test_input = torch.randn(batch_size, channels, img_height, img_width)
    valid_widths = torch.tensor([400, 512])
    memory, mask = encoder(test_input, valid_widths=valid_widths)
    grid_h, grid_w = img_height // 8, img_width // 8
    assert memory.shape == (batch_size, grid_h * grid_w, embedding_dim)
    assert mask is not None
    assert mask.shape == (batch_size, grid_h * grid_w)
    assert mask.dtype == torch.bool
    assert mask[1].sum() == 0
    assert mask[0].any()


def test_resnet_encoder_grid_shape():
    img_height, img_width, channels = 64, 512, 3
    batch_size = 1
    embedding_dim = 256
    encoder = ResNetEncoder(
        img_height=img_height,
        img_width=img_width,
        channels=channels,
        model_name="resnet18",
        embedding_dim=embedding_dim,
        freeze_backbone=True,
    )
    test_input = torch.randn(batch_size, channels, img_height, img_width)
    memory, mask = encoder(test_input)
    assert memory.dim() == 3
    assert memory.shape[0] == batch_size
    assert memory.shape[2] == embedding_dim
    assert mask is None
