#!/usr/bin/env python
"""
Test script for checking encoder compatibility with new image dimensions.
"""

import torch

from img2latex.model.encoder import CNNEncoder, ResNetEncoder


def test_cnn_encoder():
    """Test CNN encoder with the new image dimensions."""
    # Setup
    img_height = 64
    img_width = 800
    channels = 1
    batch_size = 4
    embedding_dim = 256

    # Create test input
    test_input = torch.randn(batch_size, channels, img_height, img_width)

    # Create CNN encoder
    encoder = CNNEncoder(
        img_height=img_height,
        img_width=img_width,
        channels=channels,
        embedding_dim=embedding_dim,
    )

    # Test forward pass
    output = encoder(test_input)

    # Check output shape
    expected_shape = (batch_size, embedding_dim)
    actual_shape = output.shape

    print("CNN Encoder Test:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {actual_shape}")
    print(f"  Expected shape: {expected_shape}")
    print(f"  Test {'passed' if actual_shape == expected_shape else 'failed'}")


def test_resnet_encoder():
    """Test ResNet encoder with the new image dimensions."""
    # Setup
    img_height = 64
    img_width = 800
    channels = 3
    batch_size = 4
    embedding_dim = 256

    # Create test input
    test_input = torch.randn(batch_size, channels, img_height, img_width)

    # Create ResNet encoder
    encoder = ResNetEncoder(
        img_height=img_height,
        img_width=img_width,
        channels=channels,
        embedding_dim=embedding_dim,
    )

    # Test forward pass
    output = encoder(test_input)

    # Check output shape
    expected_shape = (batch_size, embedding_dim)
    actual_shape = output.shape

    print("ResNet Encoder Test:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {actual_shape}")
    print(f"  Expected shape: {expected_shape}")
    print(f"  Test {'passed' if actual_shape == expected_shape else 'failed'}")


if __name__ == "__main__":
    print("Testing encoder compatibility with new image dimensions (64x800)...")
    test_cnn_encoder()
    test_resnet_encoder()
    print("Tests completed.")
