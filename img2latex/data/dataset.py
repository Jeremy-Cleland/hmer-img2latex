"""
Memory-optimized dataset classes and data loading utilities for the image-to-LaTeX model.
Handles loading images and formulas, applying necessary preprocessing,
and preparing batches for training and evaluation.

Features:
- Option to preload images into memory for faster access
- Optimized DataLoader configurations for different worker counts
- Memory-efficient processing options
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.data.utils import load_image
from img2latex.utils.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


# --- Define Collator Class at Top Level (for pickling) ---
class Im2LatexCollator:
    """
    Collator function for the DataLoader.

    Pads formula sequences to the maximum length in the batch and stacks images.
    """

    def __init__(self, pad_token_id: int):
        """
        Initialize the collator.

        Args:
            pad_token_id: The token ID used for padding sequences.
        """
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate a list of samples into a batch.

        Args:
            batch: A list of dictionaries, each from Im2LatexDataset.__getitem__.

        Returns:
            A dictionary containing batched data.
        """
        # Stack image tensors (they are already preprocessed to the same size)
        images = torch.stack([item["image"] for item in batch])
        formulas = [item["formula"] for item in batch]

        # Pad formula sequences to the maximum length within this batch
        max_len = max(len(formula) for formula in formulas)
        # Use torch.full for efficient padding
        padded_formulas = torch.full(
            (len(batch), max_len), self.pad_token_id, dtype=torch.long
        )
        for i, formula in enumerate(formulas):
            padded_formulas[i, : len(formula)] = formula

        # Return the collated batch
        return {
            "images": images,
            "formulas": padded_formulas,
            "raw_formulas": [item["raw_formula"] for item in batch],
            "image_paths": [item["image_path"] for item in batch],
            "formula_idxs": [item["formula_idx"] for item in batch],
        }


# --- End of Collator Class Definition ---


class Im2LatexDataset(Dataset):
    """
    PyTorch Dataset for the Im2LaTeX data.

    Loads images from the 'img' directory and corresponding
    formulas, applying preprocessing suitable for CNN-LSTM or ResNet-LSTM models.

    Features:
    - Optional in-memory caching of images for faster access
    - Efficient preprocessing pipeline
    - Robust error handling for missing files
    """

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        formulas_file: str,
        tokenizer: LaTeXTokenizer,
        img_dir: str = None,
        img_size: Tuple[int, int] = None,
        channels: int = None,
        transform=None,
        max_samples: Optional[int] = None,
        load_in_memory: bool = False,
        log_frequency: int = 1000,
    ):
        """
        Initialize the Im2LatexDataset.

        Args:
            data_dir: Path to the main data directory (e.g., './data').
            split_file: Name of the file containing image names and formula indices
                        for this split (e.g., 'im2latex_train_filter.lst').
            formulas_file: Name of the file containing normalized formulas
                           (e.g., 'im2latex_formulas.norm.lst').
            tokenizer: Initialized LaTeXTokenizer instance.
            img_dir: Name of the directory containing the actual image files
                     (relative to data_dir). Default is 'img'.
            img_size: Target image size AFTER processing (height, width).
                      Default is (64, 800) based on analysis.
            channels: Number of channels for the model input (1 for CNN, 3 for ResNet).
                      Default is 1.
            transform: Optional torchvision transforms pipeline. If None, a default
                       pipeline based on channels and img_size will be created.
            max_samples: Maximum number of samples to load (for debugging/testing).
            load_in_memory: If True, preload all images into memory for faster access.
                           Caution: This can use significant RAM for large datasets.
            log_frequency: How often to log progress during preloading (in # of samples).
        """
        self.data_dir = Path(data_dir)
        self.img_base_dir = self.data_dir / (
            img_dir or "img"
        )  # Base directory for images
        self.split_file_path = self.data_dir / split_file
        self.formulas_file_path = self.data_dir / formulas_file
        self.tokenizer = tokenizer
        self.log_frequency = log_frequency

        # Set default values if not provided
        self.target_height = img_size[0] if img_size else 64
        self.target_width = img_size[1] if img_size else 800
        self.channels = channels if channels is not None else 1
        self.img_size = (self.target_height, self.target_width)
        self.transform = transform
        self.max_samples = max_samples
        self.load_in_memory = load_in_memory
        self.preloaded_images = {}  # Dict to store preloaded images

        # Validate paths
        if not self.split_file_path.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file_path}")
        if not self.formulas_file_path.exists():
            raise FileNotFoundError(
                f"Formulas file not found: {self.formulas_file_path}"
            )
        if not self.img_base_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_base_dir}")

        # Load formulas and samples
        self.formulas: List[str] = []  # Pre-load all formulas
        self.samples: List[Dict] = (
            self._load_data()
        )  # Load image paths and formula indices

        # Check available memory if loading into memory
        if self.load_in_memory:
            # Estimate memory usage: ~100KB per image on average
            estimated_memory_mb = len(self.samples) * 0.1  # MB
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)

            logger.info(
                f"Estimated memory for image preloading: {estimated_memory_mb:.2f} MB"
            )
            logger.info(f"Available system memory: {available_memory_mb:.2f} MB")

            if (
                estimated_memory_mb > available_memory_mb * 0.5
            ):  # If using more than 50% of available
                logger.warning(
                    f"Loading {len(self.samples)} images may use {estimated_memory_mb:.2f} MB "
                    f"of RAM. Consider using load_in_memory=False for the full dataset."
                )

                # Ask for confirmation if in interactive mode
                if hasattr(sys, "ps1"):  # Check if running interactively
                    confirmation = input(
                        f"Continue with loading {len(self.samples)} images into memory? (y/n): "
                    )
                    if confirmation.lower() != "y":
                        logger.info("Disabling in-memory loading based on user input.")
                        self.load_in_memory = False

        # Preload images if requested and not disabled by user
        if self.load_in_memory:
            logger.info(f"Preloading {len(self.samples)} images into memory...")
            for i, sample in enumerate(self.samples):
                # Log progress at specified frequency
                if i % self.log_frequency == 0 and i > 0:
                    logger.info(f"Preloaded {i}/{len(self.samples)} images...")
                image_filename = sample["image_filename"]
                # Load the raw PIL image (no preprocessing) for memory efficiency
                image_path = str(self.img_base_dir / image_filename)
                try:
                    img = Image.open(image_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    self.preloaded_images[image_filename] = img
                except Exception as e:
                    logger.error(f"Error preloading image {image_path}: {e}")
                    # Skip failed images rather than storing placeholders
            logger.info(f"Finished preloading {len(self.preloaded_images)} images.")

        logger.info(
            f"Initialized Im2LatexDataset ({split_file}) with {len(self.samples)} samples. "
            f"Target image size: {self.target_height}x{self.target_width}, channels: {self.channels}, "
            f"load_in_memory: {self.load_in_memory}"
        )

    def _load_data(self) -> List[Dict]:
        """Loads formula strings and the list of samples for the current split."""
        # Load all formula strings into memory
        try:
            with open(self.formulas_file_path, "r", encoding="utf-8") as f:
                self.formulas = [line.strip() for line in f]
            logger.info(
                f"Loaded {len(self.formulas)} formulas from {self.formulas_file_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load formulas file: {e}")
            raise

        # Load image filenames and corresponding formula indices for this split
        samples = []
        try:
            with open(self.split_file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        image_filename = parts[0]
                        try:
                            formula_idx = int(parts[1])
                            if 0 <= formula_idx < len(self.formulas):
                                samples.append(
                                    {
                                        "image_filename": image_filename,
                                        "formula_idx": formula_idx,
                                    }
                                )
                            else:
                                logger.warning(
                                    f"Line {line_num + 1}: Formula index {formula_idx} out of range (0-{len(self.formulas) - 1}). Skipping."
                                )
                        except ValueError:
                            logger.warning(
                                f"Line {line_num + 1}: Invalid formula index '{parts[1]}'. Skipping."
                            )
                    else:
                        logger.warning(
                            f"Line {line_num + 1}: Invalid format '{line.strip()}'. Skipping."
                        )
        except Exception as e:
            logger.error(f"Failed to load split file {self.split_file_path}: {e}")
            raise

        # Apply max_samples limit if specified
        if self.max_samples is not None and self.max_samples > 0:
            samples = samples[: self.max_samples]
            logger.info(f"Limited dataset to {len(samples)} samples.")

        logger.info(
            f"Loaded {len(samples)} samples referencing images in {self.img_base_dir}"
        )
        return samples

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieves the processed image and tokenized formula for a given index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary containing:
                'image': The preprocessed image tensor.
                'formula': The tokenized formula tensor (including START/END).
                'raw_formula': The original formula string.
                'image_path': The filename of the image.
                'formula_idx': The index of the formula.
        """
        if idx >= len(self.samples):
            raise IndexError("Index out of range")

        sample = self.samples[idx]
        image_filename = sample["image_filename"]
        formula_idx = sample["formula_idx"]

        # Process image - either from preloaded cache or from disk
        if self.load_in_memory and image_filename in self.preloaded_images:
            # For preloaded images, we have the PIL image - apply transforms
            pil_image = self.preloaded_images[image_filename]
            # Process the PIL image using load_image
            image_path = str(self.img_base_dir / image_filename)
            if self.transform:
                # Use custom transform if provided
                image_tensor = self.transform(pil_image)
            else:
                # Otherwise use load_image with the PIL image
                image_tensor = load_image(
                    image_path,
                    img_size=self.img_size,
                    channels=self.channels,
                    normalize=True,
                )
        else:
            # For images not in memory, load and process directly
            image_path = str(self.img_base_dir / image_filename)
            image_tensor = load_image(
                image_path,
                img_size=self.img_size,
                channels=self.channels,
                normalize=True,
            )

        # Load formula
        formula_str = self._load_formula(formula_idx)

        # Tokenize formula (add START/END tokens)
        formula_with_tokens = f"{self.tokenizer.special_tokens['START']} {formula_str} {self.tokenizer.special_tokens['END']}"
        formula_ids = self.tokenizer.encode(formula_with_tokens)
        formula_tensor = torch.tensor(formula_ids, dtype=torch.long)

        return {
            "image": image_tensor,
            "formula": formula_tensor,
            "raw_formula": formula_str,
            "image_path": image_filename,  # Store filename for reference
            "formula_idx": formula_idx,
        }

    def _load_formula(self, idx: int) -> str:
        """Retrieves a formula string from the pre-loaded list."""
        if 0 <= idx < len(self.formulas):
            return self.formulas[idx]
        else:
            logger.error(f"Invalid formula index requested: {idx}")
            return ""  # Return empty string on error

    def clear_memory_cache(self):
        """Clears the preloaded images from memory to free up RAM."""
        if hasattr(self, "preloaded_images") and self.preloaded_images:
            logger.info(
                f"Clearing {len(self.preloaded_images)} preloaded images from memory"
            )
            self.preloaded_images.clear()
            # Force garbage collection
            import gc

            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


def create_data_loaders(
    config: dict,
    tokenizer: LaTeXTokenizer,
    max_samples: Optional[
        Dict[str, Optional[int]]
    ] = None,  # Optional override for samples per split
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        config: Configuration dictionary
        tokenizer: LaTeXTokenizer instance
        max_samples: Dict with keys 'train', 'val', 'test' and values as max samples to use

    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    # Get values from config
    data_dir = config["data"]["data_dir"]
    data_config = config["data"]
    model_type = config["model"]["name"]
    encoder_config = config["model"]["encoder"]

    # Extract required parameters with fallbacks
    batch_size = data_config.get("batch_size", 128)
    num_workers = data_config.get("num_workers", 0)
    prefetch_factor = data_config.get("prefetch_factor", 2)
    log_frequency = data_config.get("log_frequency", 1000)
    load_in_memory = data_config.get("load_in_memory", False)
    persistent_workers = data_config.get("persistent_workers", num_workers > 0)
    pin_memory = data_config.get("pin_memory", torch.cuda.is_available())

    # Prepare default max_samples dict if not provided
    if max_samples is None:
        max_samples = {"train": None, "val": None, "test": None}

    # Set image size and channels based on model type
    if model_type == "cnn_lstm" or model_type.startswith("cnn"):
        channels = encoder_config["cnn"].get("channels", 1)
        img_height = encoder_config["cnn"].get("img_height", 64)
        img_width = encoder_config["cnn"].get("img_width", 800)
        img_size = (img_height, img_width)
    else:  # resnet_lstm or anything with resnet
        channels = encoder_config["resnet"].get("channels", 3)
        img_height = encoder_config["resnet"].get("img_height", 64)
        img_width = encoder_config["resnet"].get("img_width", 800)
        img_size = (img_height, img_width)

    # Validate parameters
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"Number of workers must be non-negative, got {num_workers}")

    # Configure DataLoader parameters
    dataloader_kwargs = {}

    # For system with multiple CPUs/workers
    if num_workers > 0:
        # Configure for best performance
        dataloader_kwargs.update(
            {
                "num_workers": num_workers,
                "prefetch_factor": prefetch_factor,
                "persistent_workers": persistent_workers,
            }
        )

    # Set pin_memory
    dataloader_kwargs["pin_memory"] = pin_memory

    # Create an instance of the collator
    collator = Im2LatexCollator(pad_token_id=tokenizer.pad_token_id)

    # Get filenames from config
    train_file = data_config.get("train_file", "im2latex_train_filter.lst")
    validate_file = data_config.get("validate_file", "im2latex_validate_filter.lst")
    test_file = data_config.get("test_file", "im2latex_test_filter.lst")
    formulas_file = data_config.get("formulas_file", "im2latex_formulas.norm.lst")
    img_dir = data_config.get("img_dir", "img")

    # Map split names to file names
    split_files = {
        "train": train_file,
        "val": validate_file,
        "test": test_file,
    }

    # Create datasets for train/val/test
    datasets = {}
    for split in ["train", "val", "test"]:
        # Configure dataset with preprocessing config for normalization
        datasets[split] = Im2LatexDataset(
            data_dir=data_dir,
            split_file=split_files[split],
            formulas_file=formulas_file,
            tokenizer=tokenizer,
            img_dir=img_dir,
            img_size=img_size,
            channels=channels,
            max_samples=max_samples.get(split),
            load_in_memory=load_in_memory,
            log_frequency=log_frequency,
        )
        # Log dataset initialization with configured frequency
        logger.info(
            f"Initialized {split} dataset with {len(datasets[split])} samples. "
            f"Images: {img_size[0]}x{img_size[1]}, channels: {channels}"
        )

    # Check if all datasets are empty
    if not datasets or all(len(ds) == 0 for ds in datasets.values() if ds is not None):
        logger.warning(
            "All datasets are empty! Check file paths and max_samples settings."
        )
        return {}

    # Get eval batch size parameters
    eval_batch_multiplier = data_config.get("eval_batch_size_multiplier", 2)
    max_eval_batch_size = data_config.get("max_eval_batch_size", 256)

    # Create and return data loaders
    loaders = {}
    for split in ["train", "val", "test"]:
        if split in datasets and datasets[split] is not None:
            # Use a slightly larger batch size for validation and testing if memory allows
            current_batch_size = (
                batch_size
                if split == "train"
                else min(batch_size * eval_batch_multiplier, max_eval_batch_size)
            )

            # Create DataLoader with optimal settings
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=current_batch_size,
                shuffle=(split == "train"),  # Only shuffle training data
                collate_fn=collator,
                drop_last=(split == "train"),  # Only drop last batch in training
                **dataloader_kwargs,
            )

            logger.info(
                f"{split.capitalize()} DataLoader: {len(datasets[split])} samples, "
                f"batch_size={current_batch_size}, num_workers={dataloader_kwargs.get('num_workers', 0)}"
            )

    return loaders
