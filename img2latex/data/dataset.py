"""
Dataset classes for the image-to-LaTeX model.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


class Im2LatexDataset(Dataset):
    """
    Dataset for image-to-LaTeX conversion.

    This dataset loads and preprocesses images of mathematical expressions
    and their corresponding LaTeX formulas.
    """

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        formulas_file: str,
        tokenizer: LaTeXTokenizer,
        img_dir: str = "img",
        img_size: Tuple[int, int] = (50, 200),
        channels: int = 1,
        transform=None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the Im2LatexDataset.

        Args:
            data_dir: Path to the data directory
            split_file: Name of the split file (train, val, test)
            formulas_file: Name of the formulas file
            tokenizer: Tokenizer for LaTeX formulas
            img_dir: Name of the directory containing images
            img_size: Size to resize images to (height, width)
            channels: Number of image channels (1 for grayscale, 3 for RGB)
            transform: Optional transforms to apply to images
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / img_dir
        self.split_file = self.data_dir / split_file
        self.formulas_file = self.data_dir / formulas_file
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.channels = channels
        self.transform = transform
        self.max_samples = max_samples

        # Check if files exist
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        if not self.formulas_file.exists():
            raise FileNotFoundError(f"Formulas file not found: {self.formulas_file}")
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        # Load formulas and samples
        self.formulas = []  # Will be populated in _load_data
        self.samples = self._load_data()

        # Create default transforms if none provided
        if self.transform is None:
            self.transform = self._create_default_transforms()

        logger.info(
            f"Initialized Im2LatexDataset with {len(self.samples)} samples, "
            f"img_size={(img_size[0])}x{img_size[1]}, channels={channels}"
        )

    def _load_data(self) -> List[Dict]:
        """
        Load data from split file and formulas file.

        Returns:
            List of dictionaries with image paths and formula indices
        """
        # Load formulas
        with open(self.formulas_file, "r", encoding="utf-8") as f:
            self.formulas = [line.strip() for line in f]

        # Load split
        with open(self.split_file, "r", encoding="utf-8") as f:
            split_lines = [line.strip() for line in f]

        # Parse split lines
        samples = []
        for line in split_lines:
            parts = line.split()
            if len(parts) == 2:
                image_path = parts[0]
                formula_idx = int(parts[1])

                # Skip if formula index is out of range
                if formula_idx >= len(self.formulas):
                    logger.warning(
                        f"Formula index {formula_idx} out of range. "
                        f"There are only {len(self.formulas)} formulas."
                    )
                    continue

                samples.append({"image_path": image_path, "formula_idx": formula_idx})

        # Limit samples if specified
        if self.max_samples is not None:
            samples = samples[: self.max_samples]

        # Log statistics
        logger.info(f"Loaded {len(samples)} samples from {self.split_file}")
        return samples


# Import the custom transform from transforms.py
from img2latex.data.transforms import ResizeWithAspectRatio


class Im2LatexDataset(Dataset):
    """
    Dataset for image-to-LaTeX conversion.

    This dataset loads and preprocesses images of mathematical expressions
    and their corresponding LaTeX formulas.
    """

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        formulas_file: str,
        tokenizer: LaTeXTokenizer,
        img_dir: str = "img",
        img_size: Tuple[int, int] = (50, 200),
        channels: int = 1,
        transform=None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the Im2LatexDataset.

        Args:
            data_dir: Path to the data directory
            split_file: Name of the split file (train, val, test)
            formulas_file: Name of the formulas file
            tokenizer: Tokenizer for LaTeX formulas
            img_dir: Name of the directory containing images
            img_size: Size to resize images to (height, width)
            channels: Number of image channels (1 for grayscale, 3 for RGB)
            transform: Optional transforms to apply to images
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / img_dir
        self.split_file = self.data_dir / split_file
        self.formulas_file = self.data_dir / formulas_file
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.channels = channels
        self.transform = transform
        self.max_samples = max_samples

        # Check if files exist
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        if not self.formulas_file.exists():
            raise FileNotFoundError(f"Formulas file not found: {self.formulas_file}")
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        # Load formulas and samples
        self.formulas = []  # Will be populated in _load_data
        self.samples = self._load_data()

        # Create default transforms if none provided
        if self.transform is None:
            self.transform = self._create_default_transforms()

        logger.info(
            f"Initialized Im2LatexDataset with {len(self.samples)} samples, "
            f"img_size={(img_size[0])}x{img_size[1]}, channels={channels}"
        )

    def _load_data(self) -> List[Dict]:
        """
        Load data from split file and formulas file.

        Returns:
            List of dictionaries with image paths and formula indices
        """
        # Load formulas
        with open(self.formulas_file, "r", encoding="utf-8") as f:
            self.formulas = [line.strip() for line in f]

        # Load split
        with open(self.split_file, "r", encoding="utf-8") as f:
            split_lines = [line.strip() for line in f]

        # Parse split lines
        samples = []
        for line in split_lines:
            parts = line.split()
            if len(parts) == 2:
                image_path = parts[0]
                formula_idx = int(parts[1])

                # Skip if formula index is out of range
                if formula_idx >= len(self.formulas):
                    logger.warning(
                        f"Formula index {formula_idx} out of range. "
                        f"There are only {len(self.formulas)} formulas."
                    )
                    continue

                samples.append({"image_path": image_path, "formula_idx": formula_idx})

        # Limit samples if specified
        if self.max_samples is not None:
            samples = samples[: self.max_samples]

        # Log statistics
        logger.info(f"Loaded {len(samples)} samples from {self.split_file}")
        return samples

    def _create_default_transforms(self) -> transforms.Compose:
        """
        Create default transforms for image preprocessing.

        Returns:
            Composed transforms
        """
        # Use custom transform class instead of lambda
        resize_transform = ResizeWithAspectRatio(
            target_height=self.img_size[0], target_width=self.img_size[1]
        )

        # Define transforms
        if self.channels == 1:
            # Grayscale with custom resizing and padding for CNN
            return transforms.Compose(
                [
                    resize_transform,
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )
        else:
            # RGB with custom resizing and padding for ResNet
            # Use ImageNet normalization for pretrained models
            return transforms.Compose(
                [
                    resize_transform,
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def _load_formula(self, idx: int) -> str:
        """
        Get a formula by index from the preloaded formulas list.

        Args:
            idx: Formula index

        Returns:
            Formula string
        """
        # Simply return the formula from our preloaded list
        if 0 <= idx < len(self.formulas):
            return self.formulas[idx]

        # If we get here, the formula index is out of range
        raise ValueError(
            f"Formula index {idx} out of range. There are only {len(self.formulas)} formulas."
        )

    def _load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from disk.

        Args:
            image_path: Path to the image (relative to img_dir)

        Returns:
            PIL Image
        """
        full_path = self.img_dir / image_path

        # Load the image
        try:
            # Use PIL for best compatibility with torchvision transforms
            img = Image.open(full_path)

            # Convert to grayscale or RGB as needed
            if self.channels == 1 and img.mode != "L":
                img = img.convert("L")
            elif self.channels == 3 and img.mode != "RGB":
                img = img.convert("RGB")

            return img
        except Exception as e:
            logger.error(f"Error loading image {full_path}: {e}")
            # Create a dummy image on error
            if self.channels == 1:
                return Image.new("L", self.img_size, 0)
            else:
                return Image.new("RGB", self.img_size, (0, 0, 0))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with image tensor, formula tokens, etc.
        """
        sample = self.samples[idx]
        image_path = sample["image_path"]
        formula_idx = sample["formula_idx"]

        # Load and process the image
        image = self._load_image(image_path)
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            # Convert to tensor without transforms
            if self.channels == 1:
                image_tensor = (
                    torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255.0
                )
            else:
                image_tensor = (
                    torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
                )

        # Load and process the formula
        formula = self._load_formula(formula_idx)
        # Add special tokens to the formula
        formula_with_tokens = f"{self.tokenizer.special_tokens['START']} {formula} {self.tokenizer.special_tokens['END']}"
        # Tokenize the formula
        formula_tokens = self.tokenizer.encode(formula_with_tokens)
        # Convert to tensor
        formula_tensor = torch.tensor(formula_tokens)

        return {
            "image": image_tensor,
            "formula": formula_tensor,
            "raw_formula": formula,
            "image_path": image_path,
            "formula_idx": formula_idx,
        }


def create_data_loaders(
    data_dir: str,
    tokenizer: LaTeXTokenizer,
    model_type: str = "cnn_lstm",
    batch_size: int = 128,
    num_workers: int = 4,
    cnn_img_size: Tuple[int, int] = (64, 800),
    resnet_img_size: Tuple[int, int] = (64, 800),
    max_samples: Optional[Dict[str, int]] = None,
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        data_dir: Path to the data directory
        tokenizer: Tokenizer for LaTeX formulas
        model_type: Type of model to use (cnn_lstm or resnet_lstm)
        batch_size: Batch size
        num_workers: Number of workers for data loading
        cnn_img_size: Image size for CNN encoder (height, width)
        resnet_img_size: Image size for ResNet encoder (height, width)
        max_samples: Maximum number of samples to load for each split (for debugging)

    Returns:
        Dictionary with data loaders for train, val, and test splits
    """
    # Determine image size and channels based on model type
    if model_type == "cnn_lstm":
        img_size = cnn_img_size
        channels = 1  # Grayscale
    elif model_type == "resnet_lstm":
        img_size = resnet_img_size
        channels = 3  # RGB
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # Define default max_samples
    if max_samples is None:
        max_samples = {"train": None, "val": None, "test": None}

    # Create collate function for batching
    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        formulas = [item["formula"] for item in batch]

        # Pad formulas to the same length
        max_len = max(len(formula) for formula in formulas)
        padded_formulas = torch.zeros(len(formulas), max_len, dtype=torch.long)
        for i, formula in enumerate(formulas):
            padded_formulas[i, : len(formula)] = formula

        # Create dictionary with batch items
        return {
            "images": images,
            "formulas": padded_formulas,
            "raw_formulas": [item["raw_formula"] for item in batch],
            "image_paths": [item["image_path"] for item in batch],
            "formula_idxs": [item["formula_idx"] for item in batch],
        }

    # Create datasets
    train_dataset = Im2LatexDataset(
        data_dir=data_dir,
        split_file="im2latex_train_filter.lst",
        formulas_file="im2latex_formulas.norm.lst",
        tokenizer=tokenizer,
        img_dir="img",
        img_size=img_size,
        channels=channels,
        max_samples=max_samples["train"],
    )

    val_dataset = Im2LatexDataset(
        data_dir=data_dir,
        split_file="im2latex_validate_filter.lst",
        formulas_file="im2latex_formulas.norm.lst",
        tokenizer=tokenizer,
        img_dir="img",
        img_size=img_size,
        channels=channels,
        max_samples=max_samples["val"],
    )

    test_dataset = Im2LatexDataset(
        data_dir=data_dir,
        split_file="im2latex_test_filter.lst",
        formulas_file="im2latex_formulas.norm.lst",
        tokenizer=tokenizer,
        img_dir="img",
        img_size=img_size,
        channels=channels,
        max_samples=max_samples["test"],
    )

    # Create data loaders
    # With our picklable transform classes, we can use multiprocessing
    effective_workers = num_workers  # Use the provided num_workers parameter

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    logger.info(
        f"Created data loaders for model type {model_type}: "
        f"train={len(train_dataset)} samples, "
        f"val={len(val_dataset)} samples, "
        f"test={len(test_dataset)} samples"
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
