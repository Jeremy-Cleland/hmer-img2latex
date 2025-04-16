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
from torchvision import transforms

from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.data.transforms import (
    ResizeWithAspectRatio,
)  # Import custom transform classes
from img2latex.utils.logging import get_logger  # Relative import for logging

logger = get_logger(__name__)


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
        img_dir: str = "img",  # Default to the processed images dir
        img_size: Tuple[int, int] = (64, 800),  # Default to final processed size
        channels: int = 1,  # Default based on CNN model
        transform=None,
        max_samples: Optional[int] = None,
        load_in_memory: bool = False,  # Whether to preload all images into memory
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
                     (relative to data_dir). Should be 'img'.
            img_size: Target image size AFTER processing (height, width).
                      Should be (64, 800) based on analysis.
            channels: Number of channels for the model input (1 for CNN, 3 for ResNet).
            transform: Optional torchvision transforms pipeline. If None, a default
                       pipeline based on channels and img_size will be created.
            max_samples: Maximum number of samples to load (for debugging/testing).
            load_in_memory: If True, preload all images into memory for faster access.
                           Caution: This can use significant RAM for large datasets.
        """
        self.data_dir = Path(data_dir)
        self.img_base_dir = self.data_dir / img_dir  # Base directory for images
        self.split_file_path = self.data_dir / split_file
        self.formulas_file_path = self.data_dir / formulas_file
        self.tokenizer = tokenizer
        self.target_height = img_size[0]
        self.target_width = img_size[1]
        self.channels = channels
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

        # Setup transforms if not provided
        if self.transform is None:
            self.transform = self._create_default_transforms()

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
                if i % 1000 == 0 and i > 0:
                    logger.info(f"Preloaded {i}/{len(self.samples)} images...")
                image_filename = sample["image_filename"]
                # Only load the PIL image here, transformations will be applied at __getitem__ time
                self.preloaded_images[image_filename] = self._load_image(image_filename)
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

    def _create_default_transforms(self) -> transforms.Compose:
        """Creates the default preprocessing pipeline based on target size and channels."""
        transform_list = []

        # 1. Resize height while maintaining aspect ratio (operates on PIL Image)
        transform_list.append(
            ResizeWithAspectRatio(  # Use the CLASS from transforms.py
                target_height=self.target_height, target_width=self.target_width
            )
        )

        # 2. Conditional Grayscale Conversion (operates on PIL Image)
        if self.channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))

        # 3. Convert PIL Image to PyTorch Tensor (scales pixels to [0, 1])
        transform_list.append(transforms.ToTensor())

        # 4. Apply Normalization (PadTensorTransform is NOT needed if ResizeWithAspectRatio works correctly)
        if self.channels == 1:
            # Normalize to [-1, 1] for grayscale
            normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        else:  # RGB
            # Apply ImageNet normalization for RGB
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        # 5. Apply Normalization
        transform_list.append(normalize)

        return transforms.Compose(transform_list)

    def _load_image(self, image_filename: str) -> Image.Image:
        """Loads a single image using PIL."""
        full_path = self.img_base_dir / image_filename
        try:
            img = Image.open(full_path)
            # Ensure image is loaded as RGB initially, grayscale conversion happens in transform if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except FileNotFoundError:
            logger.error(f"Image file not found: {full_path}")
        except Exception as e:
            logger.error(f"Error loading image {full_path}: {e}")

        # Return a blank white image on error
        logger.warning(f"Returning blank image for {image_filename}")
        return Image.new(
            "RGB", (100, self.target_height), (255, 255, 255)
        )  # Placeholder size

    def _load_formula(self, idx: int) -> str:
        """Retrieves a formula string from the pre-loaded list."""
        if 0 <= idx < len(self.formulas):
            return self.formulas[idx]
        else:
            logger.error(f"Invalid formula index requested: {idx}")
            return ""  # Return empty string on error

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

        # Load image (from memory if preloaded, or from disk)
        if self.load_in_memory and image_filename in self.preloaded_images:
            image = self.preloaded_images[image_filename]
        else:
            image = self._load_image(image_filename)

        # Apply transforms (resize, pad, grayscale (cond.), ToTensor, normalize)
        image_tensor = self.transform(image)

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
    data_dir: str,
    tokenizer: LaTeXTokenizer,
    model_type: str = "cnn_lstm",
    batch_size: int = 128,
    num_workers: int = 0,  # Default to 0 based on performance testing
    cnn_img_size: Tuple[int, int] = (64, 800),  # Final processed size
    resnet_img_size: Tuple[int, int] = (64, 800),  # Final processed size
    max_samples: Optional[Dict[str, Optional[int]]] = None,  # Allow None per split
    load_in_memory: bool = False,  # Whether to preload images into memory
    prefetch_factor: int = 2,  # Number of batches to prefetch (default is 2)
    persistent_workers: Optional[
        bool
    ] = None,  # Whether to keep workers alive between iterations
    pin_memory: Optional[bool] = None,  # Whether to pin memory in CUDA
) -> Dict[str, DataLoader]:
    """
    Creates DataLoaders for train, validation, and test sets with memory optimization options.

    Args:
        data_dir: Path to the data directory.
        tokenizer: Initialized LaTeXTokenizer.
        model_type: 'cnn_lstm' or 'resnet_lstm'. Determines channels.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.
        cnn_img_size: Target (height, width) for CNN model inputs.
        resnet_img_size: Target (height, width) for ResNet model inputs.
        max_samples: Optional dictionary limiting samples per split (e.g., {"train": 1000}).
        load_in_memory: If True, preload all images into memory for faster access.
                        Useful for small datasets or when sufficient RAM is available.
        prefetch_factor: Number of batches loaded in advance by each worker.
                         Only used when num_workers > 0.
        persistent_workers: If True, keep worker processes alive between DataLoader iterations.
                           Default: True if num_workers > 0, else not applicable.
        pin_memory: If True, pin memory in CUDA.
                   Default: True if CUDA is available.

    Returns:
        A dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    if model_type == "cnn_lstm":
        img_size = cnn_img_size
        channels = 1
    elif model_type == "resnet_lstm":
        img_size = resnet_img_size
        channels = 3
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    if max_samples is None:
        max_samples = {"train": None, "val": None, "test": None}

    # Set defaults for optional parameters
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Instantiate the picklable collator class
    collator = Im2LatexCollator(pad_token_id=tokenizer.pad_token_id)

    loaders = {}
    splits = {
        "train": "im2latex_train_filter.lst",
        "val": "im2latex_validate_filter.lst",
        "test": "im2latex_test_filter.lst",
    }

    # Use the correct image directory name
    image_directory = "img"

    # Log memory optimization settings
    logger.info(
        f"Creating DataLoaders with: num_workers={num_workers}, load_in_memory={load_in_memory}, "
        f"prefetch_factor={prefetch_factor}, persistent_workers={persistent_workers}, "
        f"pin_memory={pin_memory}"
    )

    for split_name, split_filename in splits.items():
        # Determine whether to load this split in memory
        # For training splits with many samples, it may be impractical
        split_load_in_memory = load_in_memory
        if (
            split_name == "train"
            and max_samples.get(split_name) is None
            and load_in_memory
        ):
            logger.warning(
                "Loading full training set in memory may require substantial RAM. "
                "Consider using load_in_memory=False or setting max_samples for training."
            )

        dataset = Im2LatexDataset(
            data_dir=data_dir,
            split_file=split_filename,
            formulas_file="im2latex_formulas.norm.lst",
            tokenizer=tokenizer,
            img_dir=image_directory,  # Use the correct directory
            img_size=img_size,
            channels=channels,
            max_samples=max_samples.get(split_name),  # Get limit for this split
            load_in_memory=split_load_in_memory,  # Whether to preload images
            # transform=None will use the default created inside Im2LatexDataset
        )

        # Check if dataset is empty (e.g., due to file issues or max_samples=0)
        if len(dataset) == 0:
            logger.warning(
                f"Dataset for split '{split_name}' is empty. Skipping DataLoader creation."
            )
            loaders[split_name] = None  # Or an empty loader if preferred
            continue

        # Optimized DataLoader settings based on testing:
        # - For num_workers=0, no extra settings needed
        # - For num_workers>0, configure for best performance
        if num_workers > 0:
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == "train"),  # Shuffle only training data
                num_workers=num_workers,
                collate_fn=collator,  # Use the collator instance
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
        else:
            # For num_workers=0, simpler configuration
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == "train"),  # Shuffle only training data
                num_workers=0,
                collate_fn=collator,  # Use the collator instance
                pin_memory=pin_memory,
            )

        logger.info(
            f"Created DataLoader for '{split_name}' with {len(dataset)} samples."
        )

    # Log info about the data loaders
    train_count = len(loaders.get("train").dataset) if loaders.get("train") else 0
    val_count = len(loaders.get("val").dataset) if loaders.get("val") else 0
    test_count = len(loaders.get("test").dataset) if loaders.get("test") else 0

    logger.info(
        f"Created data loaders for model type {model_type}: "
        f"train={train_count} samples, val={val_count} samples, test={test_count} samples"
    )

    return loaders
