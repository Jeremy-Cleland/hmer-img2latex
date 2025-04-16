"""
Utility functions for data processing.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


def load_image(
    image_path: str,
    img_size: Tuple[int, int] = (50, 200),
    channels: int = 1,
    normalize: bool = True
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
        
        # Convert to the appropriate mode
        if channels == 1 and img.mode != "L":
            img = img.convert("L")
        elif channels == 3 and img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize the image
        img = img.resize(img_size[::-1])  # PIL uses (width, height)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Add channel dimension for grayscale images
        if channels == 1:
            img_array = np.expand_dims(img_array, axis=0)
        else:
            # Rearrange from HWC to CHW
            img_array = np.transpose(img_array, (2, 0, 1))
        
        # Convert to torch tensor
        img_tensor = torch.from_numpy(img_array).float()
        
        # Normalize to [0, 1]
        if normalize:
            img_tensor = img_tensor / 255.0
            
            # Normalize to [-1, 1]
            img_tensor = img_tensor * 2.0 - 1.0
        
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
    batch: Dict, 
    device: torch.device,
    model_type: str = "cnn_lstm"
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