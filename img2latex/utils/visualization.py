"""
Base visualization utilities for model analysis and reporting.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns

# Default dark theme settings
DEFAULT_THEME = {
    "background_color": "#121212",
    "text_color": "#f5f5f5",
    "grid_color": "#404040",
    "main_color": "#34d399",
    "bar_colors": ["#a78bfa", "#22d3ee", "#34d399", "#d62728", "#e27c7c"],
    "cmap": "YlOrRd",
}


def apply_dark_theme(theme: Optional[Dict] = None) -> None:
    """
    Apply dark theme to matplotlib plots.

    Args:
        theme: Theme settings dictionary (uses DEFAULT_THEME if None)
    """
    theme = theme or DEFAULT_THEME

    # Set the style
    plt.style.use("dark_background")

    # Configure plot settings
    plt.rcParams["figure.facecolor"] = theme["background_color"]
    plt.rcParams["axes.facecolor"] = theme["background_color"]
    plt.rcParams["text.color"] = theme["text_color"]
    plt.rcParams["axes.labelcolor"] = theme["text_color"]
    plt.rcParams["xtick.color"] = theme["text_color"]
    plt.rcParams["ytick.color"] = theme["text_color"]
    plt.rcParams["grid.color"] = theme["grid_color"]
    plt.rcParams["axes.edgecolor"] = theme["grid_color"]
    plt.rcParams["savefig.facecolor"] = theme["background_color"]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", theme["bar_colors"])

    # Configure seaborn
    sns.set_palette(theme["bar_colors"])
    sns.set_style(
        "darkgrid",
        {
            "axes.facecolor": theme["background_color"],
            "grid.color": theme["grid_color"],
        },
    )
    sns.set_context("paper", font_scale=1.5)


def ensure_plots_dir(output_dir: Path) -> Path:
    """Ensure plots directory exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
