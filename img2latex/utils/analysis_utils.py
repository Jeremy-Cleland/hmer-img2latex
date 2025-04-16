"""
Common utility functions for analysis scripts in the img2latex project.
"""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def ensure_output_dir(base_dir: str, analysis_type: str) -> Path:
    """Ensure output directory exists, creating it if necessary.

    Args:
        base_dir: Base directory path (can be relative or absolute)
        analysis_type: Type of analysis (subdirectory name)

    Returns:
        Path object to the full output directory
    """
    if os.path.isabs(base_dir):
        # If absolute path is provided, use it directly
        output_dir = Path(base_dir)
    else:
        # If relative path, create it under the project root
        output_dir = Path(os.getcwd()) / base_dir

    # Add analysis type subdirectory
    if analysis_type:
        output_dir = output_dir / analysis_type

    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory set to: {output_dir}")

    return output_dir


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file and return its contents as a dictionary.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing the JSON data

    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file cannot be decoded as JSON
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save dictionary to a JSON file.

    Args:
        data: Dictionary to save
        file_path: Path where to save the JSON file
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv_file(
    data: List[Dict[str, Any]],
    file_path: Union[str, Path],
    fieldnames: Optional[List[str]] = None,
) -> None:
    """Save list of dictionaries to a CSV file.

    Args:
        data: List of dictionaries to save
        file_path: Path where to save the CSV file
        fieldnames: List of field names for the CSV header. If None, uses keys from the first dictionary.
    """
    if not data:
        print(f"Warning: No data to save to {file_path}")
        return

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def load_csv_file(file_path: Union[str, Path]) -> List[Dict[str, str]]:
    """Load CSV file and return its contents as a list of dictionaries.

    Args:
        file_path: Path to the CSV file

    Returns:
        List of dictionaries, where each dictionary represents a row

    Raises:
        FileNotFoundError: If the file does not exist
    """
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Returns:
        Path object pointing to the project root
    """
    # This needs to be adjusted since we're now in the img2latex package
    return Path(os.getcwd())
