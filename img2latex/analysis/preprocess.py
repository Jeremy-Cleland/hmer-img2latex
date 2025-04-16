#!/usr/bin/env python
"""
Script to visualize the preprocessing steps for images in the latex recognition pipeline.

Features:
- Pad image to fixed width
- Display tensor visualization
- Show preprocessing steps with annotations
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from PIL import Image
from utils import ensure_output_dir

# Create Typer app
app = typer.Typer(help="Visualize preprocessing steps for images")


def pad_image(img: np.ndarray, target_width: int, pad_value: int = 255) -> np.ndarray:
    """Pad image to target width.

    Args:
        img: Input image array
        target_width: Desired width after padding
        pad_value: Value to use for padding (default: 255 for white)

    Returns:
        Padded image array
    """
    height, width = img.shape[:2]

    # No padding needed if image is already wider than target
    if width >= target_width:
        return img

    # Calculate padding
    pad_width = target_width - width

    # Create padded image
    if len(img.shape) == 3:  # Color image
        padded_img = (
            np.ones((height, target_width, img.shape[2]), dtype=img.dtype) * pad_value
        )
        padded_img[:, :width, :] = img
    else:  # Grayscale image
        padded_img = np.ones((height, target_width), dtype=img.dtype) * pad_value
        padded_img[:, :width] = img

    return padded_img


def show_image_tensor(
    ax: plt.Axes,
    tensor: Union[torch.Tensor, np.ndarray],
    title: str = "",
    cmap: Optional[str] = None,
) -> None:
    """Display image tensor on a matplotlib axes.

    Args:
        ax: Matplotlib axes to plot on
        tensor: Image tensor or array to display
        title: Title for the plot
        cmap: Colormap to use (default: None, which uses grayscale for 1-channel, RGB for 3-channel)
    """
    # Convert tensor to numpy if needed
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if tensor.requires_grad:
            tensor = tensor.detach()
        img_array = tensor.numpy()
    else:
        img_array = tensor

    # Squeeze singleton dimensions (e.g., batch size of 1)
    img_array = np.squeeze(img_array)

    # Handle channel dimension for plotting
    if len(img_array.shape) == 3:
        # Check if channels-first format (C,H,W)
        if img_array.shape[0] in [1, 3]:
            # Convert from (C,H,W) to (H,W,C)
            img_array = np.transpose(img_array, (1, 2, 0))

        # If single channel in last dimension, squeeze it
        if img_array.shape[2] == 1:
            img_array = np.squeeze(img_array, axis=2)

    # Display image
    ax.imshow(img_array, cmap=cmap)

    # Add title if provided
    if title:
        ax.set_title(title)

    # Remove axes
    ax.axis("off")


def get_image_stats(
    image_folder: Union[str, Path], num_samples: int = 1000
) -> Tuple[float, float, float, float]:
    """Get average statistics for images in a folder.

    Args:
        image_folder: Path to folder with images
        num_samples: Maximum number of images to sample

    Returns:
        Tuple of (mean_width, mean_height, mean_aspect_ratio, std_aspect_ratio)
    """
    image_folder = Path(image_folder)
    image_files = list(image_folder.glob("*.png"))

    # Sample image files if more than num_samples
    if len(image_files) > num_samples:
        import random

        random.shuffle(image_files)
        image_files = image_files[:num_samples]

    widths = []
    heights = []
    aspect_ratios = []

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if not widths:
        return 0, 0, 0, 0

    mean_width = sum(widths) / len(widths)
    mean_height = sum(heights) / len(heights)
    mean_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    std_aspect_ratio = (
        sum((r - mean_aspect_ratio) ** 2 for r in aspect_ratios) / len(aspect_ratios)
    ) ** 0.5

    return mean_width, mean_height, mean_aspect_ratio, std_aspect_ratio


def create_preprocessing_visualization(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    image_folder: Optional[Union[str, Path]] = None,
    bg_color: str = "white",
    cnn_mode: bool = True,
) -> None:
    """Create visualization of preprocessing steps.

    Args:
        image_path: Path to input image
        output_path: Path to save visualization
        image_folder: Path to folder with similar images (for stats)
        bg_color: Background color for the plot
        cnn_mode: Whether to use CNN mode (grayscale) or ResNet mode (RGB)
    """
    # Load the image
    img = Image.open(image_path)

    # Get dataset stats if folder provided
    stats_text = ""
    if image_folder:
        mean_width, mean_height, mean_aspect_ratio, std_aspect_ratio = get_image_stats(
            image_folder
        )
        stats_text = (
            f"Dataset stats: Mean size: {mean_width:.1f}x{mean_height:.1f}, "
            f"Mean aspect ratio: {mean_aspect_ratio:.2f}±{std_aspect_ratio:.2f}"
        )

    # Create a figure with 2 rows: CNN and ResNet preprocessing
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=bg_color)

    # Add a subtitle for each row
    axes[0, 0].text(
        -0.1,
        0.5,
        "CNN Pipeline",
        va="center",
        ha="right",
        transform=axes[0, 0].transAxes,
        fontsize=14,
        fontweight="bold",
    )

    axes[1, 0].text(
        -0.1,
        0.5,
        "ResNet Pipeline",
        va="center",
        ha="right",
        transform=axes[1, 0].transAxes,
        fontsize=14,
        fontweight="bold",
    )

    # Add overall title
    plt.suptitle(
        f"Image Preprocessing Visualization: {Path(image_path).name}",
        fontsize=16,
        fontweight="bold",
    )

    # Process image for each pipeline (CNN and ResNet)
    for row, mode in enumerate(["cnn", "resnet"]):
        # Step 1: Original image
        orig_img = np.array(img)
        show_image_tensor(axes[row, 0], orig_img, f"1. Original: {orig_img.shape}")

        # Step 2: Resize to fixed height (64 px)
        target_height = 64
        width, height = img.size
        aspect_ratio = width / height
        target_width = int(aspect_ratio * target_height)

        resized_img = img.resize((target_width, target_height), Image.LANCZOS)
        resized_arr = np.array(resized_img)
        show_image_tensor(axes[row, 1], resized_arr, f"2. Resize: {resized_arr.shape}")

        # Step 3: Pad to fixed width (800 px)
        target_width = 800
        if mode == "cnn":
            # For CNN, convert to grayscale first
            if len(resized_arr.shape) == 3:
                grayscale_img = resized_img.convert("L")
                resized_arr = np.array(grayscale_img)

            padded_arr = pad_image(resized_arr, target_width)
            show_image_tensor(
                axes[row, 2],
                padded_arr,
                f"3. Pad+Gray: {padded_arr.shape}",
                cmap="gray",
            )
        else:
            # For ResNet, keep RGB
            if len(resized_arr.shape) == 2:
                # If already grayscale, convert to RGB
                rgb_img = Image.fromarray(resized_arr).convert("RGB")
                resized_arr = np.array(rgb_img)

            padded_arr = pad_image(resized_arr, target_width)
            show_image_tensor(
                axes[row, 2], padded_arr, f"3. Pad+RGB: {padded_arr.shape}"
            )

        # Step 4: Convert to tensor and normalize
        if mode == "cnn":
            # Convert to tensor with shape [1, H, W]
            tensor = torch.from_numpy(padded_arr).float()
            tensor = tensor.unsqueeze(0) if len(tensor.shape) == 2 else tensor

            # Normalize to [0, 1]
            tensor = tensor / 255.0

            # Display normalized tensor
            show_image_tensor(
                axes[row, 3],
                tensor,
                f"4. Normalize: {tuple(tensor.shape)}",
                cmap="gray",
            )
        else:
            # Convert to tensor with shape [3, H, W]
            tensor = torch.from_numpy(padded_arr).float()
            tensor = tensor.permute(2, 0, 1)  # From HWC to CHW

            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            normalized = (tensor / 255.0 - mean) / std

            # For display, denormalize
            display_tensor = normalized * std + mean

            # Display normalized tensor
            show_image_tensor(
                axes[row, 3], display_tensor, f"4. Normalize: {tuple(normalized.shape)}"
            )

    # Add dataset stats if available
    if stats_text:
        plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=12)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close()

    print(f"Preprocessing visualization saved to {output_path}")


@app.command()
def visualize_preprocessing(
    image_path: str = typer.Argument(..., help="Path to the input image"),
    output_dir: str = typer.Option(
        "outputs/preprocessing", help="Directory to save visualization"
    ),
    image_folder: Optional[str] = typer.Option(
        None, help="Path to folder with similar images (for stats)"
    ),
    bg_color: str = typer.Option("white", help="Background color for the plot"),
    cnn_mode: bool = typer.Option(
        True, help="Visualize CNN preprocessing (will show both pipelines anyway)"
    ),
) -> None:
    """Visualize preprocessing steps for an image in the latex recognition pipeline."""
    # Ensure output directory exists
    output_path = ensure_output_dir(output_dir, "preprocessing")

    # Generate output file path
    image_name = Path(image_path).stem
    output_file = output_path / f"{image_name}_preprocessing.png"

    # Create visualization
    create_preprocessing_visualization(
        image_path=image_path,
        output_path=output_file,
        image_folder=image_folder,
        bg_color=bg_color,
        cnn_mode=cnn_mode,
    )

    print(f"Preprocessing visualization created for {image_path}")
    print(f"Output saved to {output_file}")


if __name__ == "__main__":
    app()
