"""
Utility functions for data processing.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from img2latex.data.transforms import ResizeWithAspectRatio
from img2latex.utils.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


def pil_to_tensor(
    img: Image.Image,
    channels: int,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert a resized/padded PIL image to a float tensor."""
    if channels == 1:
        if img.mode != "L":
            img = img.convert("L")
        img_array = np.expand_dims(np.array(img), axis=0)
    else:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.transpose(np.array(img), (2, 0, 1))
    img_tensor = torch.from_numpy(img_array).float() / 255.0
    if normalize:
        if channels == 1:
            img_tensor = img_tensor * 2.0 - 1.0
        else:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            img_tensor = (img_tensor - mean) / std
    return img_tensor


def load_image_with_size(
    image_path: str,
    img_size: Tuple[int, int] = (64, 512),
    channels: int = 1,
    normalize: bool = True,
) -> Tuple[torch.Tensor, int, int]:
    """Load an image and return (tensor, valid_width, valid_height)."""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = Image.open(image_path)
        if channels == 1 and img.mode != "L":
            img = img.convert("L")
        elif channels == 3 and img.mode != "RGB":
            img = img.convert("RGB")
        target_height, target_width = img_size
        resized, valid_w, valid_h = ResizeWithAspectRatio(target_height, target_width)(img)
        tensor = pil_to_tensor(resized, channels=channels, normalize=normalize)
        return tensor, int(valid_w), int(valid_h)
    except Exception as e:
        logger.error("Error loading image %s: %s", image_path, e)
        zeros = torch.zeros((channels, img_size[0], img_size[1]))
        return zeros, img_size[1], img_size[0]


def load_image(
    image_path: str,
    img_size: Tuple[int, int] = (64, 512),
    channels: int = 1,
    normalize: bool = True,
) -> torch.Tensor:
    tensor, _, _ = load_image_with_size(image_path, img_size, channels, normalize)
    return tensor


def array_to_tensor(
    array: np.ndarray,
    channels: int,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert a cached HxW (or HxWxC) uint8 array to a model tensor."""
    if array.ndim == 2:
        if channels == 1:
            img_array = np.expand_dims(array, axis=0)
        else:
            img_array = np.repeat(array[None, ...], 3, axis=0)
    else:
        img_array = np.transpose(array, (2, 0, 1))
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_array)).float() / 255.0
    if normalize:
        if channels == 1:
            img_tensor = img_tensor * 2.0 - 1.0
        else:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            img_tensor = (img_tensor - mean) / std
    return img_tensor


def batch_convert_for_resnet(batch_tensor: torch.Tensor) -> torch.Tensor:
    if batch_tensor.shape[1] == 3:
        return batch_tensor
    return batch_tensor.repeat(1, 3, 1, 1)


def prepare_batch(
    batch: Dict, device: torch.device, model_type: str = "cnn_transformer"
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    images = batch["images"].to(device)
    formulas = batch["formulas"].to(device)
    widths = batch["widths"].to(device) if "widths" in batch else None
    heights = batch["heights"].to(device) if "heights" in batch else None
    if model_type.startswith("resnet") and images.shape[1] == 1:
        images = batch_convert_for_resnet(images)
    return images, formulas, widths, heights
