#!/usr/bin/env python
"""
Script to analyze image sizes and create a visualization grid of LaTeX formula images.
"""

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

def analyze_image_sizes(image_folder, num_samples=500):
    """
    Analyze a sample of images to determine size distribution.
    
    Args:
        image_folder: Path to folder containing images
        num_samples: Number of images to sample for analysis
        
    Returns:
        Dictionary with size statistics
    """
    # Get all image files
    image_files = glob.glob(os.path.join(image_folder, "*.png"))
    
    # Sample if there are more images than num_samples
    if len(image_files) > num_samples:
        image_files = random.sample(image_files, num_samples)
    
    # Collect width and height information
    widths = []
    heights = []
    aspect_ratios = []
    size_counts = defaultdict(int)
    
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height)
                size_counts[(width, height)] += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Calculate statistics
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
        "most_common_sizes": sorted(size_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        "widths": widths,
        "heights": heights,
        "aspect_ratios": aspect_ratios
    }
    
    return stats

def create_image_grid(image_folder, output_path, rows=5, cols=6, figsize=(15, 12), bg_color="#121212"):
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
        if len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1):
            # For grayscale images, invert so formula appears white on dark background
            if np.mean(img_array) < 128:  # If image is mostly dark
                img_array = 255 - img_array  # Invert
        
        # Display image
        ax.imshow(img_array, cmap='gray', vmin=0, vmax=255)
        
        # Remove axes and set background color
        ax.axis('off')
        ax.set_facecolor(bg_color)
        
        # Add image filename as title
        filename = os.path.basename(img_path)
        ax.set_title(filename, color='white', fontsize=8)
    
    # Adjust layout and set overall background color
    plt.tight_layout()
    fig.patch.set_facecolor(bg_color)
    
    # Save the figure
    plt.savefig(output_path, facecolor=bg_color, bbox_inches='tight', dpi=300)
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
    plt.rcParams['axes.facecolor'] = bg_color
    plt.rcParams['figure.facecolor'] = bg_color
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 12), facecolor=bg_color)
    gs = GridSpec(2, 2, figure=fig)
    
    # Width distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(stats['widths'], ax=ax1, color='skyblue', kde=True)
    ax1.set_title('Width Distribution', color='white')
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Count')
    
    # Height distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(stats['heights'], ax=ax2, color='salmon', kde=True)
    ax2.set_title('Height Distribution', color='white')
    ax2.set_xlabel('Height (pixels)')
    ax2.set_ylabel('Count')
    
    # Aspect ratio distribution
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(stats['aspect_ratios'], ax=ax3, color='lightgreen', kde=True)
    ax3.set_title('Aspect Ratio Distribution', color='white')
    ax3.set_xlabel('Aspect Ratio (width/height)')
    ax3.set_ylabel('Count')
    
    # Width vs Height scatter plot
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(stats['widths'], stats['heights'], alpha=0.5, color='orchid')
    ax4.set_title('Width vs Height', color='white')
    ax4.set_xlabel('Width (pixels)')
    ax4.set_ylabel('Height (pixels)')
    
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
    
    fig.text(0.5, 0.01, table_text, fontsize=12, color='white', 
             horizontalalignment='center', verticalalignment='bottom')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle('Image Size Analysis', fontsize=16, color='white')
    
    # Save the figure
    plt.savefig(output_path, facecolor=bg_color, bbox_inches='tight', dpi=300)
    print(f"Size distribution visualization saved to {output_path}")
    
    return fig

def main():
    # Set paths
    image_folder = os.path.join(os.getcwd(), "data", "img")
    output_folder = os.path.join(os.getcwd(), "outputs", "analysis")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Analyze image sizes
    print("Analyzing image sizes...")
    stats = analyze_image_sizes(image_folder, num_samples=1000)
    
    # Print summary statistics
    print("\nImage Size Statistics:")
    print(f"Number of images analyzed: {stats['num_images']}")
    print(f"Width range: {stats['width_min']} - {stats['width_max']} pixels")
    print(f"Width mean: {stats['width_mean']:.1f} pixels")
    print(f"Height range: {stats['height_min']} - {stats['height_max']} pixels")
    print(f"Height mean: {stats['height_mean']:.1f} pixels")
    print(f"Aspect ratio range: {stats['aspect_ratio_min']:.2f} - {stats['aspect_ratio_max']:.2f}")
    print(f"Aspect ratio mean: {stats['aspect_ratio_mean']:.2f}")
    
    print("\nMost common image sizes:")
    for (width, height), count in stats['most_common_sizes']:
        print(f"  {width}x{height} pixels: {count} images")
    
    # Create image grid
    print("\nCreating image grid...")
    grid_output_path = os.path.join(output_folder, "formula_image_grid.png")
    create_image_grid(
        image_folder, 
        grid_output_path, 
        rows=5, 
        cols=6,
        bg_color="#121212"
    )
    
    # Create size distribution visualization
    print("\nCreating size distribution visualization...")
    dist_output_path = os.path.join(output_folder, "size_distribution.png")
    visualize_size_distribution(
        stats, 
        dist_output_path,
        bg_color="#121212"
    )
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()