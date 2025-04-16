"""
Utility functions for data processing.
"""

import os
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image

from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


import torchvision.transforms.functional as TF


def resize_maintain_aspect(img: Image.Image, target_height: int) -> Image.Image:
    """
    Resizes image to target_height while maintaining aspect ratio.

    Args:
        img: PIL Image
        target_height: Target height in pixels

    Returns:
        Resized PIL Image with exactly target_height height
    """
    width, height = img.size
    
    # Ensure we're resizing to the exact target height
    if height != target_height:
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        
        # Use LANCZOS for high quality resizing
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            # For older versions of PIL
            resample = Image.LANCZOS
        
        img = img.resize((new_width, target_height), resample)
    
    return img


def pad_image_width(
    img_tensor: torch.Tensor, target_width: int, padding_value: float = 0.0
) -> torch.Tensor:
    """
    Pads the width of an image tensor to target_width.

    Args:
        img_tensor: Image tensor of shape [C, H, W]
        target_width: Target width in pixels
        padding_value: Value to use for padding

    Returns:
        Padded tensor of shape [C, H, target_width]
    """
    # Check if tensor has expected dimensions
    if img_tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor [C,H,W], got {img_tensor.dim()}D")

    _, _, current_width = img_tensor.shape
    padding_needed = target_width - current_width

    # Return original tensor if no padding needed
    if padding_needed <= 0:
        # If image is wider than target, crop it to target_width
        if current_width > target_width:
            return img_tensor[:, :, :target_width]
        return img_tensor

    # Pad format for TF.pad is (left, right, top, bottom)
    padding = (0, padding_needed, 0, 0)
    return TF.pad(img_tensor, padding, fill=padding_value, padding_mode="constant")


def load_image(
    image_path: str,
    img_size: Tuple[int, int] = (64, 800),
    channels: int = 1,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Load and preprocess an image for inference.

    Args:
        image_path: Path to the image file
        img_size: Size to resize the image to (height, width)
        channels: Number of channels (1 for grayscale, 3 for RGB)
        normalize: Whether to normalize the image

    Returns:
        Preprocessed image tensor
    """
    try:
        # Check if the image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load the image
        img = Image.open(image_path)

        # Convert to the appropriate mode based on channels parameter
        if channels == 1 and img.mode != "L":
            img = img.convert("L")
        elif channels == 3 and img.mode != "RGB":
            img = img.convert("RGB")

        # 1. First resize the image to target height while maintaining aspect ratio
        img = resize_maintain_aspect(img, target_height=img_size[0])

        # Convert to tensor
        if channels == 1:
            # For grayscale, we need to add a channel dimension
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_tensor = torch.from_numpy(img_array).float()
        else:
            # For RGB images
            img_array = np.array(img)
            # Rearrange from HWC to CHW
            img_array = np.transpose(img_array, (2, 0, 1))
            img_tensor = torch.from_numpy(img_array).float()

        # Normalize to [0, 1]
        img_tensor = img_tensor / 255.0
        
        # 2. Ensure the tensor has the exact target dimensions (height, width)
        C, H, W = img_tensor.shape
        target_height, target_width = img_size
        
        padding_value = 1.0 if channels == 1 else 0.0  # White for grayscale, black for RGB
        
        # Fix both height and width if needed
        if H != target_height or W != target_width:
            # Create a new tensor with the target dimensions
            padded_tensor = torch.ones((C, target_height, target_width)) * padding_value
            
            # Copy as much of the original tensor as fits
            h = min(H, target_height)
            w = min(W, target_width)
            padded_tensor[:, :h, :w] = img_tensor[:, :h, :w]
            
            img_tensor = padded_tensor
        # If width needs padding but height is correct
        elif W != target_width:
            img_tensor = pad_image_width(img_tensor, target_width=target_width, padding_value=padding_value)

        # Apply additional normalization
        if normalize:
            if channels == 1:
                # Normalize to [-1, 1] for grayscale
                img_tensor = img_tensor * 2.0 - 1.0
            else:
                # Apply ImageNet normalization for RGB
                mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
                img_tensor = (img_tensor - mean) / std

        return img_tensor

    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        # Return a blank image in case of error
        if channels == 1:
            return torch.zeros((1, img_size[0], img_size[1]))
        else:
            return torch.zeros((3, img_size[0], img_size[1]))


def batch_convert_for_resnet(batch_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of grayscale images to RGB for ResNet.

    Args:
        batch_tensor: Batch of grayscale images (batch_size, 1, height, width)

    Returns:
        Batch of RGB images (batch_size, 3, height, width)
    """
    # Check if the tensor is already RGB
    if batch_tensor.shape[1] == 3:
        return batch_tensor

    # Convert grayscale to RGB by repeating the channel
    rgb_tensor = batch_tensor.repeat(1, 3, 1, 1)

    return rgb_tensor


def prepare_batch(
    batch: Dict, device: torch.device, model_type: str = "cnn_lstm"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare a batch for training.

    Args:
        batch: Batch from the dataloader
        device: Device to move tensors to
        model_type: Type of model being used (cnn_lstm or resnet_lstm)

    Returns:
        Tuple of (images, target_sequences)
    """
    # Get images and formulas from batch
    images = batch["images"].to(device)
    formulas = batch["formulas"].to(device)

    # For ResNet model, convert grayscale to RGB if needed
    if model_type == "resnet_lstm" and images.shape[1] == 1:
        images = batch_convert_for_resnet(images)

    return images, formulas
