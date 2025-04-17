"""
CNN and ResNet-based encoders for the image-to-LaTeX model.
"""

from typing import List

import torch
import torch.nn as nn
import torchvision.models as models

from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


class CNNEncoder(nn.Module):
    """
    CNN encoder for the image-to-LaTeX model.

    This encoder processes input images through a series of convolutional and pooling layers,
    followed by a dense layer to produce a fixed-size encoding.
    """

    def __init__(
        self,
        img_height: int = 64,
        img_width: int = 800,
        channels: int = 1,
        conv_filters: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        pool_size: int = 2,
        padding: str = "same",
        embedding_dim: int = 256,
    ):
        """
        Initialize the CNN encoder.

        Args:
            img_height: Height of the input images
            img_width: Width of the input images
            channels: Number of channels in the input images (1 for grayscale)
            conv_filters: List of filter sizes for each convolutional layer
            kernel_size: Size of the convolutional kernels
            pool_size: Size of the pooling windows
            padding: Type of padding for convolutional layers
            embedding_dim: Dimension of the output embedding
        """
        super(CNNEncoder, self).__init__()

        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.embedding_dim = embedding_dim

        # Determine padding value based on string specification
        padding_val = kernel_size // 2 if padding == "same" else 0

        # Create the convolutional blocks
        layers = []
        in_channels = channels

        for filters in conv_filters:
            # Add convolutional layer with ReLU activation
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    padding=padding_val,
                )
            )
            layers.append(nn.ReLU())

            # Add max pooling layer
            layers.append(nn.MaxPool2d(kernel_size=pool_size))

            in_channels = filters

        self.cnn_layers = nn.Sequential(*layers)

        # Calculate the flattened size after CNN layers
        # We need to do a forward pass with a dummy tensor to calculate this
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, img_height, img_width)
            dummy_output = self.cnn_layers(dummy_input)
            flattened_size = dummy_output.numel()

        # Add a dense layer to produce the embeddings
        self.flatten = nn.Flatten()
        self.embedding_layer = nn.Linear(flattened_size, embedding_dim)
        self.activation = nn.ReLU()

        logger.info(f"Initialized CNN encoder with output dimension: {embedding_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN encoder.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        # Apply CNN layers
        x = self.cnn_layers(x)

        # Flatten and pass through dense layer
        x = self.flatten(x)
        x = self.embedding_layer(x)
        x = self.activation(x)

        return x


class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder for the image-to-LaTeX model.

    This encoder uses a pre-trained ResNet model as a feature extractor,
    followed by a dense layer to produce a fixed-size encoding.
    """

    def __init__(
        self,
        img_height: int = 64,
        img_width: int = 800,
        channels: int = 3,
        model_name: str = "resnet50",
        embedding_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        """
        Initialize the ResNet encoder.

        Args:
            img_height: Height of the input images
            img_width: Width of the input images
            channels: Number of channels in the input images (should be 3 for ResNet)
            model_name: Name of the ResNet model to use
            embedding_dim: Dimension of the output embedding
            freeze_backbone: Whether to freeze the ResNet weights
        """
        super(ResNetEncoder, self).__init__()

        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.embedding_dim = embedding_dim

        # Check that channels is 3, as ResNet expects RGB images
        if channels != 3:
            logger.warning(
                f"ResNet expects 3-channel RGB images, but got {channels} channels. "
                "You'll need to convert your images to RGB format."
            )

        # Load the pre-trained ResNet model without the classification head
        if model_name == "resnet18":
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == "resnet34":
            self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif model_name == "resnet50":
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == "resnet101":
            self.resnet = models.resnet101(
                weights=models.ResNet101_Weights.IMAGENET1K_V1
            )
        elif model_name == "resnet152":
            self.resnet = models.resnet152(
                weights=models.ResNet152_Weights.IMAGENET1K_V1
            )
        else:
            raise ValueError(f"Invalid ResNet model name: {model_name}")

        # Remove the final fully connected layer (classification head)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Freeze the ResNet weights if requested
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Add a flatten layer
        self.flatten = nn.Flatten()

        # Add a dense layer to produce the embeddings
        # ResNet feature dimensions:
        # resnet18, resnet34: 512
        # resnet50, resnet101, resnet152: 2048
        if model_name in ["resnet18", "resnet34"]:
            resnet_out_features = 512
        else:  # resnet50, resnet101, resnet152
            resnet_out_features = 2048

        self.embedding_layer = nn.Linear(resnet_out_features, embedding_dim)
        self.activation = nn.ReLU()

        logger.info(
            f"Initialized ResNet encoder ({model_name}) with output dimension: {embedding_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet encoder.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        # Pass through ResNet backbone
        x = self.resnet(x)

        # Flatten and pass through dense layer
        x = self.flatten(x)
        x = self.embedding_layer(x)
        x = self.activation(x)

        return x
