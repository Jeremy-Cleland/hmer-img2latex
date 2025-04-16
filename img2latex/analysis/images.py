#!/usr/bin/env python
"""
Module to analyze image characteristics including sizes, color channels,
and pixel value distributions of LaTeX formula images.
"""

import glob
import json
import os
import random
from collections import Counter, defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer
from matplotlib.gridspec import GridSpec
from PIL import Image

from img2latex.utils import ensure_output_dir

# Create Typer app
app = typer.Typer(help="Analyze raw images in an image directory")


def analyze_images(
    image_folder: str,
    output_dir: str = "outputs/image_analysis",
    max_images: Optional[int] = None,
    sample_grid_rows: int = 5,
    sample_grid_cols: int = 6,
    bg_color: str = "#FFFFFF",
):
    """
    Analyze images to determine size distribution, color channels, and pixel values.

    Args:
        image_folder: Path to folder containing images
        output_dir: Directory to save analysis results
        max_images: Maximum number of images to analyze (None for all)
        sample_grid_rows: Number of rows in the sample image grid
        sample_grid_cols: Number of columns in the sample image grid
        bg_color: Background color for visualizations (hex code)
    """
    # Ensure output directory exists
    output_path = ensure_output_dir(output_dir, "images")

    print(f"Analyzing images in {image_folder}...")

    # Get all image files
    image_files = glob.glob(os.path.join(image_folder, "*.png"))

    if not image_files:
        print(f"No PNG images found in {image_folder}")
        return

    # Determine number of images to analyze
    num_samples = len(image_files)
    if max_images is not None and num_samples > max_images:
        num_samples = max_images
        print(f"Sampling {num_samples} images out of {len(image_files)}")
        image_files = random.sample(image_files, num_samples)
    else:
        print(f"Analyzing all {num_samples} images")

    # Determine number of images for detailed pixel analysis (max 100)
    detailed_samples = min(100, num_samples)
    detailed_files = image_files[:detailed_samples]

    # Collect information
    widths = []
    heights = []
    aspect_ratios = []
    size_counts = defaultdict(int)

    # Color and pixel value analysis
    color_modes = []
    channel_counts = []
    dtypes = []
    min_values = []
    max_values = []
    mean_values = []
    std_values = []

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                # Size analysis
                width, height = img.size
                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height)
                size_counts[(width, height)] += 1

                # Color analysis
                color_modes.append(img.mode)

                # Convert to numpy for detailed analysis
                if img_path in detailed_files:
                    img_array = np.array(img)

                    # Determine number of channels
                    if len(img_array.shape) == 2:
                        channels = 1
                    else:
                        channels = img_array.shape[2]
                    channel_counts.append(channels)

                    # Pixel value analysis
                    dtypes.append(img_array.dtype)
                    min_values.append(float(img_array.min()))
                    max_values.append(float(img_array.max()))
                    mean_values.append(float(np.mean(img_array)))
                    std_values.append(float(np.std(img_array)))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Calculate size statistics
    stats = {
        "num_images": len(widths),
        "width_min": min(widths),
        "width_max": max(widths),
        "width_mean": np.mean(widths),
        "width_median": np.median(widths),
        "height_min": min(heights),
        "height_max": max(heights),
        "height_mean": np.mean(heights),
        "height_median": np.median(heights),
        "aspect_ratio_min": min(aspect_ratios),
        "aspect_ratio_max": max(aspect_ratios),
        "aspect_ratio_mean": np.mean(aspect_ratios),
        "aspect_ratio_median": np.median(aspect_ratios),
        "most_common_sizes": sorted(
            size_counts.items(), key=lambda x: x[1], reverse=True
        )[:10],
        "widths": widths,
        "heights": heights,
        "aspect_ratios": aspect_ratios,
        # Color and pixel statistics
        "color_modes": dict(Counter(color_modes)),
        "channel_counts": dict(Counter(channel_counts)) if channel_counts else {},
        "dtypes": dict(Counter(str(dt) for dt in dtypes)) if dtypes else {},
        "min_pixel_value": min(min_values) if min_values else None,
        "max_pixel_value": max(max_values) if max_values else None,
        "mean_pixel_value": np.mean(mean_values) if mean_values else None,
        "std_pixel_value": np.mean(std_values) if std_values else None,
        # Normalization check
        "is_normalized_0_1": (
            min(min_values) >= 0 and max(max_values) <= 1
            if min_values and max_values
            else None
        ),
        "is_normalized_neg1_1": (
            min(min_values) >= -1 and max(max_values) <= 1 and min(min_values) < 0
            if min_values and max_values
            else None
        ),
        "is_uint8": (
            min(min_values) >= 0
            and max(max_values) <= 255
            and any(str(dt) == "uint8" for dt in dtypes)
            if min_values and max_values and dtypes
            else None
        ),
    }

    # Create visualizations
    print("Creating image grid visualization...")
    create_image_grid(
        image_folder,
        output_path / "sample_images.png",
        rows=sample_grid_rows,
        cols=sample_grid_cols,
        bg_color=bg_color,
    )

    print("Creating size distribution visualization...")
    visualize_size_distribution(
        stats, output_path / "size_distribution.png", bg_color=bg_color
    )

    print("Creating pixel value visualization...")
    visualize_pixel_values(stats, output_path / "pixel_values.png", bg_color=bg_color)

    # Save statistics to JSON
    with open(output_path / "image_stats.json", "w") as f:
        # Convert numpy types to Python native types for JSON serialization
        serializable_stats = {
            k: v if not isinstance(v, (np.ndarray, np.generic)) else v.tolist()
            for k, v in stats.items()
        }
        # Convert tuples to lists for JSON serialization
        serializable_stats["most_common_sizes"] = [
            [(w, h), count]
            for ((w, h), count) in serializable_stats["most_common_sizes"]
        ]
        json.dump(serializable_stats, f, indent=2)

    print(f"Analysis complete. Results saved to {output_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of images analyzed: {stats['num_images']}")
    print(
        f"Image dimensions (width x height): {stats['width_mean']:.1f} x {stats['height_mean']:.1f} (mean)"
    )
    print(f"Aspect ratio: {stats['aspect_ratio_mean']:.2f} (mean)")
    print(
        f"Most common size: {stats['most_common_sizes'][0][0][0]} x {stats['most_common_sizes'][0][0][1]} ({stats['most_common_sizes'][0][1]} images)"
    )
    print(
        f"Color modes: {', '.join(f'{mode}: {count}' for mode, count in stats['color_modes'].items())}"
    )

    if stats["is_normalized_0_1"]:
        print("Images are normalized to range [0, 1]")
    elif stats["is_normalized_neg1_1"]:
        print("Images are normalized to range [-1, 1]")
    elif stats["is_uint8"]:
        print("Images are in uint8 format (0-255)")


def create_image_grid(
    image_folder, output_path, rows=5, cols=6, figsize=(15, 12), bg_color="#121212"
):
    """
    Create a grid of images with formulas.

    Args:
        image_folder: Path to folder containing images
        output_path: Path to save the output visualization
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        figsize: Figure size (width, height) in inches
        bg_color: Background color for the plot
    """
    # Get all image files
    image_files = glob.glob(os.path.join(image_folder, "*.png"))

    # Choose a random sample
    num_images = rows * cols
    if len(image_files) > num_images:
        selected_images = random.sample(image_files, num_images)
    else:
        selected_images = image_files[:num_images]

    # Create figure with dark background
    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    gs = GridSpec(rows, cols, figure=fig)

    # Load and display images
    for i, img_path in enumerate(selected_images):
        if i >= rows * cols:
            break

        row = i // cols
        col = i % cols

        # Create subplot
        ax = fig.add_subplot(gs[row, col])

        # Load image
        img = Image.open(img_path)
        img_array = np.array(img)

        # Invert colors for dark background if image is grayscale
        if len(img_array.shape) == 2 or (
            len(img_array.shape) == 3 and img_array.shape[2] == 1
        ):
            # For grayscale images, invert so formula appears white on dark background
            if np.mean(img_array) < 128:  # If image is mostly dark
                img_array = 255 - img_array  # Invert

        # Display image
        cmap = "gray" if len(img_array.shape) == 2 or img_array.shape[-1] == 1 else None
        ax.imshow(img_array, cmap=cmap, vmin=0, vmax=255)

        # Remove axes and set background color
        ax.axis("off")
        ax.set_facecolor(bg_color)

        # Add image filename as title
        filename = os.path.basename(img_path)
        ax.set_title(filename, color="white", fontsize=8)

    # Adjust layout and set overall background color
    plt.tight_layout()
    fig.patch.set_facecolor(bg_color)

    # Save the figure
    plt.savefig(output_path, facecolor=bg_color, bbox_inches="tight", dpi=300)
    print(f"Image grid saved to {output_path}")

    plt.close(fig)
    return output_path


def visualize_size_distribution(stats, output_path, bg_color="#121212"):
    """
    Create visualizations of image size distributions.

    Args:
        stats: Dictionary containing image statistics
        output_path: Path to save the output visualization
        bg_color: Background color for the plot
    """
    # Create figure with dark background
    fig = plt.figure(figsize=(15, 12), facecolor=bg_color)
    gs = GridSpec(2, 2, figure=fig)

    # Set text color
    text_color = "white"

    # 1. Width distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(stats["widths"], kde=True, ax=ax1, color="skyblue")
    ax1.set_title("Width Distribution", color=text_color)
    ax1.set_xlabel("Width (pixels)", color=text_color)
    ax1.set_ylabel("Count", color=text_color)
    ax1.tick_params(colors=text_color)
    ax1.set_facecolor(bg_color)
    for spine in ax1.spines.values():
        spine.set_color(text_color)

    # 2. Height distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(stats["heights"], kde=True, ax=ax2, color="lightgreen")
    ax2.set_title("Height Distribution", color=text_color)
    ax2.set_xlabel("Height (pixels)", color=text_color)
    ax2.set_ylabel("Count", color=text_color)
    ax2.tick_params(colors=text_color)
    ax2.set_facecolor(bg_color)
    for spine in ax2.spines.values():
        spine.set_color(text_color)

    # 3. Aspect ratio distribution
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(stats["aspect_ratios"], kde=True, ax=ax3, color="coral")
    ax3.set_title("Aspect Ratio Distribution", color=text_color)
    ax3.set_xlabel("Aspect Ratio (width/height)", color=text_color)
    ax3.set_ylabel("Count", color=text_color)
    ax3.tick_params(colors=text_color)
    ax3.set_facecolor(bg_color)
    for spine in ax3.spines.values():
        spine.set_color(text_color)

    # 4. Scatter plot of width vs. height
    ax4 = fig.add_subplot(gs[1, 1])

    # Create counter for point density
    from collections import Counter

    size_counter = Counter(zip(stats["widths"], stats["heights"]))
    x, y, counts = zip(*[(w, h, c) for (w, h), c in size_counter.items()])

    scatter = ax4.scatter(
        x, y, c=counts, cmap="viridis", alpha=0.8, s=50, edgecolors="white"
    )
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label("Count", color=text_color)
    cbar.ax.tick_params(colors=text_color)
    cbar.outline.set_edgecolor(text_color)

    ax4.set_title("Width vs. Height Scatter Plot", color=text_color)
    ax4.set_xlabel("Width (pixels)", color=text_color)
    ax4.set_ylabel("Height (pixels)", color=text_color)
    ax4.tick_params(colors=text_color)
    ax4.set_facecolor(bg_color)
    for spine in ax4.spines.values():
        spine.set_color(text_color)

    # Add a text box with summary statistics
    stats_text = (
        f"Total Images: {stats['num_images']}\n"
        f"Width: {stats['width_min']}-{stats['width_max']} px"
        f" (mean: {stats['width_mean']:.1f}, median: {stats['width_median']:.1f})\n"
        f"Height: {stats['height_min']}-{stats['height_max']} px"
        f" (mean: {stats['height_mean']:.1f}, median: {stats['height_median']:.1f})\n"
        f"Aspect Ratio: {stats['aspect_ratio_min']:.2f}-{stats['aspect_ratio_max']:.2f}"
        f" (mean: {stats['aspect_ratio_mean']:.2f})"
    )

    plt.figtext(
        0.5,
        0.01,
        stats_text,
        ha="center",
        fontsize=12,
        color=text_color,
        bbox=dict(facecolor=bg_color, alpha=0.5, edgecolor=text_color),
    )

    # Adjust layout and set overall background color
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.patch.set_facecolor(bg_color)

    # Save the figure
    plt.savefig(output_path, facecolor=bg_color, bbox_inches="tight", dpi=300)
    print(f"Size distribution visualization saved to {output_path}")

    plt.close(fig)
    return output_path


def visualize_pixel_values(stats, output_path, bg_color="#121212"):
    """
    Create visualizations of pixel value distributions.

    Args:
        stats: Dictionary containing image statistics
        output_path: Path to save the output visualization
        bg_color: Background color for the plot
    """
    # Create figure with dark background
    fig = plt.figure(figsize=(15, 10), facecolor=bg_color)
    gs = GridSpec(2, 2, figure=fig)

    # Set text color
    text_color = "white"

    # 1. Color mode distribution
    ax1 = fig.add_subplot(gs[0, 0])
    color_modes = stats.get("color_modes", {})
    if color_modes:
        modes = list(color_modes.keys())
        counts = list(color_modes.values())
        colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))

        ax1.bar(modes, counts, color=colors)
        ax1.set_title("Color Mode Distribution", color=text_color)
        ax1.set_xlabel("Color Mode", color=text_color)
        ax1.set_ylabel("Count", color=text_color)
        ax1.tick_params(colors=text_color)
        ax1.set_facecolor(bg_color)
        for spine in ax1.spines.values():
            spine.set_color(text_color)

        # Add count labels on top of bars
        for i, count in enumerate(counts):
            ax1.text(
                i,
                count + 0.1,
                str(count),
                ha="center",
                color=text_color,
                fontweight="bold",
            )

    # 2. Channel count distribution
    ax2 = fig.add_subplot(gs[0, 1])
    channel_counts = stats.get("channel_counts", {})
    if channel_counts:
        channels = list(map(str, channel_counts.keys()))
        counts = list(channel_counts.values())
        colors = plt.cm.tab10(np.linspace(0, 1, len(channels)))

        ax2.bar(channels, counts, color=colors)
        ax2.set_title("Channel Count Distribution", color=text_color)
        ax2.set_xlabel("Number of Channels", color=text_color)
        ax2.set_ylabel("Count", color=text_color)
        ax2.tick_params(colors=text_color)
        ax2.set_facecolor(bg_color)
        for spine in ax2.spines.values():
            spine.set_color(text_color)

        # Add count labels on top of bars
        for i, count in enumerate(counts):
            ax2.text(
                i,
                count + 0.1,
                str(count),
                ha="center",
                color=text_color,
                fontweight="bold",
            )

    # 3. Data type distribution
    ax3 = fig.add_subplot(gs[1, 0])
    dtypes = stats.get("dtypes", {})
    if dtypes:
        dtype_names = list(dtypes.keys())
        dtype_counts = list(dtypes.values())
        colors = plt.cm.tab10(np.linspace(0, 1, len(dtype_names)))

        ax3.bar(dtype_names, dtype_counts, color=colors)
        ax3.set_title("Data Type Distribution", color=text_color)
        ax3.set_xlabel("Data Type", color=text_color)
        ax3.set_ylabel("Count", color=text_color)
        ax3.tick_params(colors=text_color)
        ax3.set_facecolor(bg_color)
        for spine in ax3.spines.values():
            spine.set_color(text_color)

        # Add count labels on top of bars
        for i, count in enumerate(dtype_counts):
            ax3.text(
                i,
                count + 0.1,
                str(count),
                ha="center",
                color=text_color,
                fontweight="bold",
            )

    # 4. Text box with pixel value statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.set_facecolor(bg_color)

    # Prepare statistics text
    pixel_stats = [
        "Pixel Value Statistics:",
        "----------------------",
        f"Min value: {stats.get('min_pixel_value', 'N/A')}",
        f"Max value: {stats.get('max_pixel_value', 'N/A')}",
        f"Mean value: {stats.get('mean_pixel_value', 'N/A'):.2f}",
        f"Std. deviation: {stats.get('std_pixel_value', 'N/A'):.2f}",
        "",
        "Normalization Status:",
        "----------------------",
    ]

    # Add normalization status
    if stats.get("is_normalized_0_1"):
        pixel_stats.append("✓ Images are normalized to range [0, 1]")
    else:
        pixel_stats.append("✗ Images are NOT normalized to range [0, 1]")

    if stats.get("is_normalized_neg1_1"):
        pixel_stats.append("✓ Images are normalized to range [-1, 1]")
    else:
        pixel_stats.append("✗ Images are NOT normalized to range [-1, 1]")

    if stats.get("is_uint8"):
        pixel_stats.append("✓ Images are 8-bit (values: 0-255)")
    else:
        pixel_stats.append("✗ Images are NOT 8-bit")

    # Display the statistics
    ax4.text(
        0.5,
        0.5,
        "\n".join(pixel_stats),
        ha="center",
        va="center",
        color=text_color,
        fontsize=12,
        fontfamily="monospace",
        transform=ax4.transAxes,
    )

    # Adjust layout and set overall background color
    plt.tight_layout()
    fig.patch.set_facecolor(bg_color)

    # Save the figure
    plt.savefig(output_path, facecolor=bg_color, bbox_inches="tight", dpi=300)
    print(f"Pixel value visualization saved to {output_path}")

    plt.close(fig)
    return output_path


@app.command()
def analyze(
    image_folder: str = typer.Argument(..., help="Path to folder containing images"),
    output_dir: str = typer.Option(
        "outputs/image_analysis", help="Directory to save analysis results"
    ),
    max_images: Optional[int] = typer.Option(
        None, help="Maximum number of images to analyze (None for all)"
    ),
    rows: int = typer.Option(5, help="Number of rows in the sample image grid"),
    cols: int = typer.Option(6, help="Number of columns in the sample image grid"),
    bg_color: str = typer.Option(
        "#FFFFFF", help="Background color for visualizations (hex code)"
    ),
):
    """Analyze image characteristics in a dataset."""
    analyze_images(
        image_folder=image_folder,
        output_dir=output_dir,
        max_images=max_images,
        sample_grid_rows=rows,
        sample_grid_cols=cols,
        bg_color=bg_color,
    )


if __name__ == "__main__":
    app()
