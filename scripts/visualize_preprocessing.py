#!/usr/bin/env python
"""
Script to visualize the image preprocessing pipeline used in the img2latex project.
Shows each step of the transformation process.
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from matplotlib.gridspec import GridSpec
import glob

def pad_image(img, target_width=800, pad_value=255):
    """Pad an image to the target width with white pixels."""
    width, height = img.size
    padding = (0, 0, target_width - width, 0)  # left, top, right, bottom
    return F.pad(img, padding, pad_value)

def show_image_tensor(ax, tensor, title=None, cmap=None):
    """Display a tensor as an image."""
    if tensor.dim() == 4:  # batch dimension
        tensor = tensor[0]
    
    if tensor.dim() == 3:
        if tensor.size(0) == 1:  # grayscale
            img = tensor.squeeze(0).cpu().numpy()
            cmap = 'gray'
        else:  # RGB
            img = tensor.permute(1, 2, 0).cpu().numpy()
            # If normalized to [-1, 1], convert back to [0, 1]
            if img.min() < 0:
                img = (img + 1) / 2
    else:
        img = tensor.cpu().numpy()
    
    # Clip to valid range
    img = np.clip(img, 0, 1)
    
    ax.imshow(img, cmap=cmap)
    ax.axis('off')
    if title:
        ax.set_title(title)

def create_preprocessing_visualization(image_path, output_path, bg_color="#121212", cnn_mode=True):
    """
    Create a visualization of the preprocessing pipeline for a single image.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the visualization
        bg_color: Background color for the plot
        cnn_mode: Whether to show CNN (grayscale) or ResNet (RGB) preprocessing
    """
    # Set style for plots
    plt.rcParams['axes.facecolor'] = bg_color
    plt.rcParams['figure.facecolor'] = bg_color
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    
    # Create figure with dark background
    fig = plt.figure(figsize=(15, 10), facecolor=bg_color)
    gs = GridSpec(2, 3, figure=fig)
    
    # Load original image
    image = Image.open(image_path)
    original_width, original_height = image.size
    
    # Step 1: Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.array(image))
    ax1.set_title(f"Original Image\n({original_width}x{original_height}px, RGB)", color='white')
    ax1.axis('off')
    
    # Step 2: Resize to fixed height (64px)
    new_height = 64
    new_width = int(original_width * (new_height / original_height))
    resized_image = image.resize((new_width, new_height), Image.BILINEAR)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(np.array(resized_image))
    ax2.set_title(f"Fixed Height\n({new_width}x{new_height}px, RGB)", color='white')
    ax2.axis('off')
    
    # Step 3: Pad to fixed width (800px)
    padded_image = pad_image(resized_image, target_width=800)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(np.array(padded_image))
    ax3.set_title(f"Fixed Width (Padded)\n(800x{new_height}px, RGB)", color='white')
    ax3.axis('off')
    
    # Step 4: Color conversion (for CNN) or keep RGB (for ResNet)
    if cnn_mode:
        # Convert to grayscale
        converted_image = padded_image.convert('L')
        color_mode_text = "Grayscale"
        channels = 1
    else:
        # Keep as RGB
        converted_image = padded_image
        color_mode_text = "RGB"
        channels = 3
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(np.array(converted_image), cmap='gray' if cnn_mode else None)
    ax4.set_title(f"Color Conversion\n(800x{new_height}px, {color_mode_text})", color='white')
    ax4.axis('off')
    
    # Step 5: Convert to tensor and normalize to [0, 1]
    tensor_transform = transforms.ToTensor()
    tensor_image = tensor_transform(converted_image)
    
    ax5 = fig.add_subplot(gs[1, 1])
    show_image_tensor(ax5, tensor_image, 
                     title=f"ToTensor Normalization\n[0, 1] range, {channels} channel{'s' if channels > 1 else ''}",
                     cmap='gray' if cnn_mode else None)
    
    # Step 6: Apply channel-specific normalization
    if cnn_mode:
        # For grayscale (CNN)
        norm_transform = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        # For RGB (ResNet)
        norm_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    
    normalized_image = norm_transform(tensor_image)
    
    ax6 = fig.add_subplot(gs[1, 2])
    # For visualization, we need to denormalize
    if cnn_mode:
        # Denormalize from [-1, 1] back to [0, 1] for display
        denorm_img = normalized_image * 0.5 + 0.5
    else:
        # Reverse ImageNet normalization (more complex)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        denorm_img = normalized_image * std + mean
    
    show_image_tensor(ax6, denorm_img, 
                     title=f"Final Normalized Image\nModel input format",
                     cmap='gray' if cnn_mode else None)
    
    # Add dataset statistics and configuration text
    title_text = "CNN Preprocessing Pipeline" if cnn_mode else "ResNet Preprocessing Pipeline"
    text = (
        f"Image processing for {title_text}\n"
        f"Dataset statistics (103,536 images):\n"
        f"- Most common size: 320x64 pixels (11,821 images)\n"
        f"- Mean aspect ratio: 5.79 (width/height)\n"
        f"- All images: RGB mode, uint8 type [0-255]\n"
        f"- Mean pixel value: 241.51, Std: 46.84"
    )
    
    plt.suptitle(title_text, fontsize=16, color='white', y=0.98)
    fig.text(0.5, 0.01, text, fontsize=12, color='white', 
             horizontalalignment='center', verticalalignment='bottom')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    plt.savefig(output_path, facecolor=bg_color, bbox_inches='tight', dpi=300)
    print(f"Preprocessing visualization saved to {output_path}")
    
    return fig

def main():
    # Set paths
    if len(sys.argv) > 1:
        # Use image path provided as argument
        image_path = sys.argv[1]
    else:
        # Pick a random image from the dataset
        image_folder = os.path.join(os.getcwd(), "data", "img")
        image_files = glob.glob(os.path.join(image_folder, "*.png"))
        image_path = random.choice(image_files)
    
    output_folder = os.path.join(os.getcwd(), "outputs", "analysis")
    os.makedirs(output_folder, exist_ok=True)
    
    # Create CNN preprocessing visualization
    cnn_output_path = os.path.join(output_folder, "cnn_preprocessing.png")
    create_preprocessing_visualization(
        image_path=image_path, 
        output_path=cnn_output_path,
        bg_color="#121212",
        cnn_mode=True
    )
    
    # Create ResNet preprocessing visualization
    resnet_output_path = os.path.join(output_folder, "resnet_preprocessing.png")
    create_preprocessing_visualization(
        image_path=image_path, 
        output_path=resnet_output_path,
        bg_color="#121212",
        cnn_mode=False
    )
    
    print(f"Visualizations completed for image: {image_path}")

if __name__ == "__main__":
    main()