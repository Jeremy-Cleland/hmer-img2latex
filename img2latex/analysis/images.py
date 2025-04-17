#!/usr/bin/env python
"""
Script to analyze image characteristics including sizes, color channels,
and pixel value distributions of LaTeX formula images.
"""

import glob
import os
import random
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer
import yaml
from matplotlib.gridspec import GridSpec
from PIL import Image

from img2latex.analysis.utils import ensure_output_dir, save_json_file

# Create Typer app
app = typer.Typer(help="Analyze image characteristics for the img2latex model")


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def analyze_images(image_folder, num_samples=500, detailed_samples=100):
    """
    Analyze images to determine size distribution, color channels, and pixel values.

    Args:
        image_folder: Path to folder containing images
        num_samples: Number of images to sample for basic analysis
        detailed_samples: Number of images to use for detailed pixel analysis

    Returns:
        Dictionary with image statistics
    """
    # Get all image files
    image_files = glob.glob(os.path.join(image_folder, "*.png"))

    # Sample if there are more images than num_samples
    if len(image_files) > num_samples:
        image_files = random.sample(image_files, num_samples)

    # Select a smaller subset for detailed pixel analysis
    detailed_files = image_files[: min(detailed_samples, len(image_files))]

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

    return stats


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

    return fig


def visualize_size_distribution(stats, output_path, bg_color="#121212"):
    """
    Create visualizations of image size distributions.

    Args:
        stats: Dictionary with image statistics
        output_path: Path to save the output visualization
        bg_color: Background color for the plot
    """
    # Set style for plots
    sns.set(style="darkgrid")
    plt.rcParams["axes.facecolor"] = bg_color
    plt.rcParams["figure.facecolor"] = bg_color
    plt.rcParams["text.color"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["ytick.color"] = "white"

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 12), facecolor=bg_color)
    gs = GridSpec(2, 2, figure=fig)

    # Width distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(stats["widths"], ax=ax1, color="skyblue", kde=True)
    ax1.set_title("Width Distribution", color="white")
    ax1.set_xlabel("Width (pixels)")
    ax1.set_ylabel("Count")

    # Height distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(stats["heights"], ax=ax2, color="salmon", kde=True)
    ax2.set_title("Height Distribution", color="white")
    ax2.set_xlabel("Height (pixels)")
    ax2.set_ylabel("Count")

    # Aspect ratio distribution
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(stats["aspect_ratios"], ax=ax3, color="lightgreen", kde=True)
    ax3.set_title("Aspect Ratio Distribution", color="white")
    ax3.set_xlabel("Aspect Ratio (width/height)")
    ax3.set_ylabel("Count")

    # Width vs Height scatter plot
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(stats["widths"], stats["heights"], alpha=0.5, color="orchid")
    ax4.set_title("Width vs Height", color="white")
    ax4.set_xlabel("Width (pixels)")
    ax4.set_ylabel("Height (pixels)")

    # Add table with statistics
    table_text = (
        f"Number of images: {stats['num_images']}\n"
        f"Width range: {stats['width_min']} - {stats['width_max']} px\n"
        f"Width mean: {stats['width_mean']:.1f} px\n"
        f"Height range: {stats['height_min']} - {stats['height_max']} px\n"
        f"Height mean: {stats['height_mean']:.1f} px\n"
        f"Aspect ratio range: {stats['aspect_ratio_min']:.2f} - {stats['aspect_ratio_max']:.2f}\n"
        f"Aspect ratio mean: {stats['aspect_ratio_mean']:.2f}\n"
        f"Most common size: {stats['most_common_sizes'][0][0]} px ({stats['most_common_sizes'][0][1]} images)"
    )

    fig.text(
        0.5,
        0.01,
        table_text,
        fontsize=12,
        color="white",
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle("Image Size Analysis", fontsize=16, color="white")

    # Save the figure
    plt.savefig(output_path, facecolor=bg_color, bbox_inches="tight", dpi=300)
    print(f"Size distribution visualization saved to {output_path}")

    return fig


def visualize_pixel_values(stats, output_path, bg_color="#121212"):
    """
    Create visualizations of pixel value distributions and color modes.

    Args:
        stats: Dictionary with image statistics
        output_path: Path to save the output visualization
        bg_color: Background color for the plot
    """
    # Set style for plots
    sns.set(style="darkgrid")
    plt.rcParams["axes.facecolor"] = bg_color
    plt.rcParams["figure.facecolor"] = bg_color
    plt.rcParams["text.color"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["ytick.color"] = "white"

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 12), facecolor=bg_color)
    gs = GridSpec(2, 2, figure=fig)

    # Color modes pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    color_modes = stats["color_modes"]
    labels = color_modes.keys()
    sizes = color_modes.values()
    explode = [0.1] * len(labels)  # explode all slices
    ax1.pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
        textprops={"color": "white"},
    )
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.set_title("Image Color Modes", color="white")

    # Channel counts pie chart
    ax2 = fig.add_subplot(gs[0, 1])
    if stats["channel_counts"]:
        channel_counts = stats["channel_counts"]
        labels = [f"{k} channel{'s' if k > 1 else ''}" for k in channel_counts.keys()]
        sizes = channel_counts.values()
        explode = [0.1] * len(labels)  # explode all slices
        ax2.pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
            textprops={"color": "white"},
        )
        ax2.axis("equal")
    ax2.set_title("Number of Channels", color="white")

    # Pixel value range
    ax3 = fig.add_subplot(gs[1, 0])
    if stats["min_pixel_value"] is not None and stats["max_pixel_value"] is not None:
        pixel_range = [stats["min_pixel_value"], stats["max_pixel_value"]]
        bar_labels = ["Min", "Max"]
        ax3.bar(bar_labels, pixel_range, color=["blue", "red"])
        ax3.set_title("Pixel Value Range", color="white")
        ax3.set_ylabel("Pixel Value")

        # Annotate the exact values
        for i, v in enumerate(pixel_range):
            ax3.text(i, v / 2, f"{v:.2f}", ha="center", color="white")

    # Data types
    ax4 = fig.add_subplot(gs[1, 1])
    if stats["dtypes"]:
        dtypes = stats["dtypes"]
        labels = list(dtypes.keys())
        sizes = list(dtypes.values())
        ax4.bar(labels, sizes, color="purple")
        ax4.set_title("Image Data Types", color="white")
        ax4.set_ylabel("Count")
        ax4.tick_params(axis="x", rotation=45)

        # Annotate the bars with counts
        for i, v in enumerate(sizes):
            ax4.text(i, v / 2, str(v), ha="center", color="white")

    # Add normalization information
    normalization_text = "Pixel Value Analysis:\n"

    if stats["is_normalized_0_1"]:
        normalization_text += "✓ Images appear to be normalized to [0, 1] range\n"
    elif stats["is_normalized_neg1_1"]:
        normalization_text += "✓ Images appear to be normalized to [-1, 1] range\n"
    elif stats["is_uint8"]:
        normalization_text += "✓ Images appear to be in standard uint8 format (0-255)\n"
    else:
        normalization_text += "? Images have non-standard normalization\n"

    normalization_text += (
        f"Min: {stats['min_pixel_value']:.2f}, Max: {stats['max_pixel_value']:.2f}\n"
    )
    normalization_text += (
        f"Mean: {stats['mean_pixel_value']:.2f}, Std: {stats['std_pixel_value']:.2f}\n"
    )

    most_common_mode = max(stats["color_modes"].items(), key=lambda x: x[1])[0]
    normalization_text += (
        f"Most common color mode: {most_common_mode} "
        + f"({stats['color_modes'][most_common_mode]} images, "
        + f"{stats['color_modes'][most_common_mode] / stats['num_images'] * 100:.1f}%)"
    )

    fig.text(
        0.5,
        0.01,
        normalization_text,
        fontsize=12,
        color="white",
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle("Color and Pixel Value Analysis", fontsize=16, color="white")

    # Save the figure
    plt.savefig(output_path, facecolor=bg_color, bbox_inches="tight", dpi=300)
    print(f"Pixel value distribution visualization saved to {output_path}")

    return fig


@app.command()
def analyze(
    config_path: str = typer.Option(
        "img2latex/configs/config.yaml", help="Path to the configuration file"
    ),
    image_folder: str = typer.Option(
        None,
        help="Path to folder containing images (defaults to data_dir/img_dir from config)",
    ),
    output_dir: str = typer.Option(
        "outputs/image_analysis", help="Directory to save the output analysis"
    ),
    num_samples: int = typer.Option(
        500, help="Number of images to sample for basic analysis"
    ),
    detailed_samples: int = typer.Option(
        100, help="Number of images to use for detailed pixel analysis"
    ),
) -> None:
    """Analyze images to determine size distribution, color channels, and pixel values."""
    # Load configuration
    cfg = load_config(config_path)

    # Use config for image_folder if not specified

    data_dir = cfg["data"]["data_dir"]
    img_dir = cfg["data"]["img_dir"]
    image_folder = os.path.join(data_dir, img_dir)

    # Setup output directory
    output_path = ensure_output_dir(output_dir, "images")

    # Analyze images
    print(f"Analyzing images from {image_folder}...")
    stats = analyze_images(
        image_folder, num_samples=num_samples, detailed_samples=detailed_samples
    )

    # Save stats to JSON
    stats_path = output_path / "image_stats.json"
    # Use save_json_file that handles NumPy types
    save_json_file(stats, stats_path)

    # Print summary statistics
    print("\nImage Size Statistics:")
    print(f"Number of images analyzed: {stats['num_images']}")
    print(f"Width range: {stats['width_min']} - {stats['width_max']} pixels")
    print(f"Width mean: {stats['width_mean']:.1f} pixels")
    print(f"Height range: {stats['height_min']} - {stats['height_max']} pixels")
    print(f"Height mean: {stats['height_mean']:.1f} pixels")
    print(
        f"Aspect ratio range: {stats['aspect_ratio_min']:.2f} - {stats['aspect_ratio_max']:.2f}"
    )
    print(f"Aspect ratio mean: {stats['aspect_ratio_mean']:.2f}")

    print("\nMost common image sizes:")
    for (width, height), count in stats["most_common_sizes"]:
        print(f"  {width}x{height} pixels: {count} images")

    print("\nColor and Pixel Value Analysis:")
    print(f"Color modes: {stats['color_modes']}")
    print(f"Channel counts: {stats['channel_counts']}")
    print(f"Data types: {stats['dtypes']}")
    print(f"Pixel value range: {stats['min_pixel_value']} - {stats['max_pixel_value']}")
    print(f"Mean pixel value: {stats['mean_pixel_value']:.2f}")
    print(f"Std dev of pixel values: {stats['std_pixel_value']:.2f}")

    # Determine normalization
    if stats["is_normalized_0_1"]:
        print("Images appear to be normalized to [0, 1] range")
    elif stats["is_normalized_neg1_1"]:
        print("Images appear to be normalized to [-1, 1] range")
    elif stats["is_uint8"]:
        print("Images appear to be in standard uint8 format (0-255)")
    else:
        print("Images have non-standard normalization")

    # Create image grid
    print("\nCreating image grid...")
    grid_output_path = output_path / "formula_image_grid.png"
    create_image_grid(
        image_folder, grid_output_path, rows=5, cols=6, bg_color="#121212"
    )

    # Create size distribution visualization
    print("\nCreating size distribution visualization...")
    dist_output_path = output_path / "size_distribution.png"
    visualize_size_distribution(stats, dist_output_path, bg_color="#121212")

    # Create pixel value distribution visualization
    print("\nCreating pixel value distribution visualization...")
    pixel_output_path = output_path / "pixel_distribution.png"
    visualize_pixel_values(stats, pixel_output_path, bg_color="#121212")

    print(f"\nAnalysis complete. Results saved to {output_path}")


if __name__ == "__main__":
    app()
