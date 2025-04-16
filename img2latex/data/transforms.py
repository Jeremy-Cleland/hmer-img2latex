"""
Custom transforms for image preprocessing.
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


class ResizeWithAspectRatio:
    """
    Resize image to target height while maintaining aspect ratio,
    then pad to target width.
    
    This is a callable class designed to be picklable for multiprocessing.
    """
    
    def __init__(self, target_height, target_width):
        """
        Initialize with target dimensions.
        
        Args:
            target_height: Target height in pixels
            target_width: Target width in pixels
        """
        self.target_height = target_height
        self.target_width = target_width
    
    def __call__(self, img):
        """
        Apply the transform to the image.
        
        Args:
            img: PIL Image
            
        Returns:
            Resized and padded PIL Image
        """
        # Calculate new height and width while maintaining aspect ratio
        width, height = img.size
        new_width = int(width * (self.target_height / height))
        
        # Resize to target height while maintaining aspect ratio
        img = img.resize((new_width, self.target_height), Image.BILINEAR)
        
        # Pad width to target width (right padding)
        if new_width < self.target_width:
            # Create new image with white background
            padded_img = Image.new(img.mode, (self.target_width, self.target_height), 255)
            # Paste original image on left side
            padded_img.paste(img, (0, 0))
            return padded_img
        return img


class PadTensorWidth:
    """
    Pad a tensor along the width dimension to a target width.
    
    This works on tensors after ToTensor() has been applied.
    """
    
    def __init__(self, target_width, padding_value=0.0):
        """
        Initialize with target width and padding value.
        
        Args:
            target_width: Target width in pixels
            padding_value: Value to use for padding (default: 0.0 for white)
        """
        self.target_width = target_width
        self.padding_value = padding_value
    
    def __call__(self, img_tensor):
        """
        Pad the tensor along the width dimension.
        
        Args:
            img_tensor: Image tensor of shape [C, H, W]
            
        Returns:
            Padded tensor of shape [C, H, target_width]
        """
        # Check tensor dimensions
        if img_tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor [C,H,W], got {img_tensor.dim()}D")
        
        _, _, current_width = img_tensor.shape
        padding_needed = self.target_width - current_width
        
        # Return original tensor if no padding needed
        if padding_needed <= 0:
            return img_tensor
        
        # Define padding as (left, right, top, bottom)
        padding = (0, padding_needed, 0, 0)
        return TF.pad(img_tensor, padding, fill=self.padding_value, padding_mode='constant')