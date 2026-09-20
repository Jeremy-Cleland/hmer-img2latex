"""
CNN and ResNet encoders that emit a spatial feature grid plus a padding mask.

Both encoders return:
    memory: (batch, H' * W', d_model)
    memory_key_padding_mask: (batch, H' * W') with True = padded / ignore
"""

from typing import List, Optional, Tuple

import math

import torch
import torch.nn as nn
import torchvision.models as models

from img2latex.utils.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


class PositionalEncoding2D(nn.Module):
    """Sinusoidal 2D positional encoding added to a (B, C, H, W) feature map."""

    def __init__(self, d_model: int, max_h: int = 32, max_w: int = 128):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D positional encoding")

        d_half = d_model // 2
        y = torch.arange(max_h, dtype=torch.float32).unsqueeze(1)
        x = torch.arange(max_w, dtype=torch.float32).unsqueeze(1)
        div_y = torch.exp(
            torch.arange(0, d_half, 2, dtype=torch.float32) * (-math.log(10000.0) / d_half)
        )
        div_x = torch.exp(
            torch.arange(0, d_half, 2, dtype=torch.float32) * (-math.log(10000.0) / d_half)
        )
        sin_y = torch.sin(y * div_y)  # (H, d_half/2)
        cos_y = torch.cos(y * div_y)
        sin_x = torch.sin(x * div_x)  # (W, d_half/2)
        cos_x = torch.cos(x * div_x)

        pe_y = torch.zeros(max_h, d_half)
        pe_y[:, 0::2] = sin_y
        pe_y[:, 1::2] = cos_y
        pe_y = pe_y.permute(1, 0).unsqueeze(-1).repeat(1, 1, max_w)

        pe_x = torch.zeros(max_w, d_half)
        pe_x[:, 0::2] = sin_x
        pe_x[:, 1::2] = cos_x
        pe_x = pe_x.permute(1, 0).unsqueeze(-2).repeat(1, max_h, 1)

        pe = torch.cat([pe_y, pe_x], dim=0).unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        return x + self.pe[:, :, :h, :w]


def _spatial_padding_mask(
    batch_size: int,
    grid_h: int,
    grid_w: int,
    downsample: int,
    valid_widths: Optional[torch.Tensor],
    valid_heights: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build a (B, H'*W') key-padding mask. True means ignore."""
    if valid_widths is None and valid_heights is None:
        return None

    col_idx = torch.arange(grid_w, device=device).view(1, 1, grid_w)
    row_idx = torch.arange(grid_h, device=device).view(1, grid_h, 1)
    mask = torch.zeros(batch_size, grid_h, grid_w, dtype=torch.bool, device=device)

    if valid_widths is not None:
        valid_cols = torch.ceil(valid_widths.float() / downsample).long().clamp(min=1, max=grid_w)
        mask |= col_idx.expand(batch_size, grid_h, grid_w) >= valid_cols.view(batch_size, 1, 1)

    if valid_heights is not None:
        valid_rows = torch.ceil(valid_heights.float() / downsample).long().clamp(min=1, max=grid_h)
        mask |= row_idx.expand(batch_size, grid_h, grid_w) >= valid_rows.view(batch_size, 1, 1)

    return mask.flatten(1)


class CNNEncoder(nn.Module):
    """
    Convolutional encoder that keeps the spatial grid.

    Three Conv-BN-ReLU-MaxPool blocks downsample by 8, then a 1x1 projection
    maps channels to d_model. A 2D positional encoding is added before flattening.
    """

    def __init__(
        self,
        img_height: int = 64,
        img_width: int = 512,
        channels: int = 1,
        conv_filters: Optional[List[int]] = None,
        kernel_size: int = 3,
        pool_size: int = 2,
        padding: str = "same",
        embedding_dim: int = 256,
    ):
        super().__init__()
        if conv_filters is None:
            conv_filters = [32, 64, 128]

        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.downsample = pool_size ** len(conv_filters)

        padding_val = kernel_size // 2 if padding == "same" else 0
        layers = []
        in_channels = channels
        for filters in conv_filters:
            layers.extend(
                [
                    nn.Conv2d(in_channels, filters, kernel_size, padding=padding_val),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=pool_size),
                ]
            )
            in_channels = filters

        self.cnn_layers = nn.Sequential(*layers)
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=1)
        grid_h = img_height // self.downsample
        grid_w = img_width // self.downsample
        self.pos2d = PositionalEncoding2D(embedding_dim, max_h=max(grid_h, 16), max_w=max(grid_w, 128))

        logger.info(
            "Initialized CNN encoder: %sx%s -> grid %sx%s, d_model=%s",
            img_height,
            img_width,
            grid_h,
            grid_w,
            embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_widths: Optional[torch.Tensor] = None,
        valid_heights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        features = self.proj(self.cnn_layers(x))
        features = self.pos2d(features)
        batch, dim, grid_h, grid_w = features.shape
        memory = features.flatten(2).transpose(1, 2).contiguous()
        mask = _spatial_padding_mask(
            batch, grid_h, grid_w, self.downsample, valid_widths, valid_heights, x.device
        )
        if mask is not None:
            fully_masked = mask.all(dim=1)
            if fully_masked.any():
                mask = mask.clone()
                mask[fully_masked, 0] = False
        return memory, mask


def _remove_layer_stride(layer: nn.Module) -> None:
    """Set residual-block stride to 1 so the last stage does not downsample."""
    for block in layer:
        if hasattr(block, "conv1") and block.conv1.kernel_size == (3, 3):
            block.conv1.stride = (1, 1)
        if hasattr(block, "conv2") and getattr(block.conv2, "kernel_size", None) == (3, 3):
            if block.conv2.stride != (1, 1):
                block.conv2.stride = (1, 1)
        if getattr(block, "downsample", None) is not None:
            down = block.downsample[0]
            if isinstance(down, nn.Conv2d):
                down.stride = (1, 1)


class ResNetEncoder(nn.Module):
    """
    ResNet backbone without avgpool/fc, last stage stride removed.

    Output grid is /16 of the input (conv1 + maxpool + layer2 + layer3).
    """

    def __init__(
        self,
        img_height: int = 64,
        img_width: int = 512,
        channels: int = 3,
        model_name: str = "resnet18",
        embedding_dim: int = 256,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.downsample = 16

        constructors = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
            "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
            "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, 2048),
            "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1, 2048),
            "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1, 2048),
        }
        if model_name not in constructors:
            raise ValueError(f"Invalid ResNet model name: {model_name}")

        ctor, weights, out_channels = constructors[model_name]
        backbone = ctor(weights=weights)
        _remove_layer_stride(backbone.layer4)
        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone[-1].parameters():
                param.requires_grad = True

        self.proj = nn.Conv2d(out_channels, embedding_dim, kernel_size=1)
        grid_h = img_height // self.downsample
        grid_w = img_width // self.downsample
        self.pos2d = PositionalEncoding2D(embedding_dim, max_h=max(grid_h, 16), max_w=max(grid_w, 128))

        logger.info(
            "Initialized ResNet encoder (%s): %sx%s -> grid ~%sx%s, d_model=%s",
            model_name,
            img_height,
            img_width,
            grid_h,
            grid_w,
            embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_widths: Optional[torch.Tensor] = None,
        valid_heights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.proj(self.backbone(x))
        features = self.pos2d(features)
        batch, dim, grid_h, grid_w = features.shape
        memory = features.flatten(2).transpose(1, 2).contiguous()
        mask = _spatial_padding_mask(
            batch, grid_h, grid_w, self.downsample, valid_widths, valid_heights, x.device
        )
        if mask is not None:
            fully_masked = mask.all(dim=1)
            if fully_masked.any():
                mask = mask.clone()
                mask[fully_masked, 0] = False
        return memory, mask
