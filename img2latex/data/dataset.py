"""
Dataset and DataLoader construction for image-to-LaTeX.

Supports a uint8 memmap cache, scale-to-fit padding, length truncation,
training augmentation, and length-bucketed batches.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from img2latex.data.cache import build_image_cache, cache_is_complete, load_image_cache
from img2latex.data.sampler import BucketBatchSampler
from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.data.transforms import ResizeWithAspectRatio
from img2latex.data.utils import array_to_tensor, load_image_with_size, pil_to_tensor
from img2latex.utils.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


class Im2LatexCollator:
    """Pad formula sequences to the batch max length and stack images."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict:
        images = torch.stack([item["image"] for item in batch])
        formulas = [item["formula"] for item in batch]
        max_len = max(len(formula) for formula in formulas)
        padded_formulas = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        for i, formula in enumerate(formulas):
            padded_formulas[i, : len(formula)] = formula
        return {
            "images": images,
            "formulas": padded_formulas,
            "widths": torch.tensor([item["width"] for item in batch], dtype=torch.long),
            "heights": torch.tensor([item["height"] for item in batch], dtype=torch.long),
            "raw_formulas": [item["raw_formula"] for item in batch],
            "image_paths": [item["image_path"] for item in batch],
            "formula_idxs": [item["formula_idx"] for item in batch],
        }


class Im2LatexDataset(Dataset):
    """Loads formula images and tokenized LaTeX, optionally from a memmap cache."""

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
        augment: bool = False,
        cache_images=None,
        cache_widths=None,
        cache_heights=None,
        cache_index: Optional[Dict[str, int]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.img_base_dir = self.data_dir / (img_dir or "img")
        self.split_file_path = self.data_dir / split_file
        self.formulas_file_path = self.data_dir / formulas_file
        self.tokenizer = tokenizer
        self.log_frequency = log_frequency
        self.target_height = img_size[0] if img_size else 64
        self.target_width = img_size[1] if img_size else 512
        self.channels = channels if channels is not None else 1
        self.img_size = (self.target_height, self.target_width)
        self.transform = transform
        self.max_samples = max_samples
        self.augment = augment
        self.resize = ResizeWithAspectRatio(self.target_height, self.target_width)
        self.cache_images = cache_images
        self.cache_widths = cache_widths
        self.cache_heights = cache_heights
        self.cache_index = cache_index or {}

        if not self.split_file_path.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file_path}")
        if not self.formulas_file_path.exists():
            raise FileNotFoundError(f"Formulas file not found: {self.formulas_file_path}")

        self.formulas: List[str] = []
        self.samples: List[Dict] = self._load_data()
        self.lengths = [sample["length"] for sample in self.samples]
        logger.info(
            "Initialized Im2LatexDataset (%s) with %s samples. Target %sx%s, channels=%s, augment=%s, cache=%s",
            split_file,
            len(self.samples),
            self.target_height,
            self.target_width,
            self.channels,
            self.augment,
            bool(self.cache_index),
        )

    def _load_data(self) -> List[Dict]:
        with open(self.formulas_file_path, "r", encoding="utf-8") as handle:
            self.formulas = [line.strip() for line in handle]
        samples = []
        max_len = self.tokenizer.max_sequence_length
        with open(self.split_file_path, "r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle):
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                image_filename = parts[0]
                try:
                    formula_idx = int(parts[1])
                except ValueError:
                    continue
                if not (0 <= formula_idx < len(self.formulas)):
                    continue
                token_len = len(self.formulas[formula_idx].split()) + 2  # START + END
                samples.append(
                    {
                        "image_filename": image_filename,
                        "formula_idx": formula_idx,
                        "length": min(token_len, max_len),
                    }
                )
        if self.max_samples is not None and self.max_samples > 0:
            samples = samples[: self.max_samples]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_pil(self, img: Image.Image) -> Image.Image:
        import random

        if img.mode == "L":
            fill = 255
        else:
            fill = (255, 255, 255)
        angle = random.uniform(-3.0, 3.0)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=fill)
        max_dx = max(1, int(0.02 * img.size[0]))
        max_dy = max(1, int(0.02 * img.size[1]))
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=fill)
        return img

    def _load_from_cache(self, image_filename: str):
        cache_idx = self.cache_index[image_filename]
        array = np.array(self.cache_images[cache_idx])
        width = int(self.cache_widths[cache_idx])
        height = int(self.cache_heights[cache_idx])
        if self.augment:
            img = Image.fromarray(array)
            img = self._augment_pil(img)
            tensor = pil_to_tensor(img, channels=self.channels, normalize=True)
        else:
            tensor = array_to_tensor(array, channels=self.channels, normalize=True)
        return tensor, width, height

    def _load_from_disk(self, image_filename: str):
        image_path = str(self.img_base_dir / image_filename)
        if self.augment:
            img = Image.open(image_path)
            if self.channels == 1 and img.mode != "L":
                img = img.convert("L")
            elif self.channels == 3 and img.mode != "RGB":
                img = img.convert("RGB")
            img = self._augment_pil(img)
            padded, width, height = self.resize(img)
            tensor = pil_to_tensor(padded, channels=self.channels, normalize=True)
            return tensor, width, height
        return load_image_with_size(image_path, img_size=self.img_size, channels=self.channels)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image_filename = sample["image_filename"]
        formula_idx = sample["formula_idx"]

        if image_filename in self.cache_index and self.cache_images is not None:
            image_tensor, width, height = self._load_from_cache(image_filename)
        else:
            image_tensor, width, height = self._load_from_disk(image_filename)

        formula_str = self.formulas[formula_idx] if 0 <= formula_idx < len(self.formulas) else ""
        formula_with_tokens = (
            f"{self.tokenizer.special_tokens['START']} {formula_str} {self.tokenizer.special_tokens['END']}"
        )
        formula_ids = self.tokenizer.encode(formula_with_tokens)
        max_len = self.tokenizer.max_sequence_length
        if len(formula_ids) > max_len:
            formula_ids = formula_ids[:max_len]
            formula_ids[-1] = self.tokenizer.end_token_id

        return {
            "image": image_tensor,
            "formula": torch.tensor(formula_ids, dtype=torch.long),
            "width": width,
            "height": height,
            "raw_formula": formula_str,
            "image_path": image_filename,
            "formula_idx": formula_idx,
        }


def create_data_loaders(
    config: Optional[dict] = None,
    tokenizer: Optional[LaTeXTokenizer] = None,
    max_samples: Optional[Dict[str, Optional[int]]] = None,
    **kwargs,
) -> Dict[str, DataLoader]:
    if (config is None or not isinstance(config, dict)) or "data_dir" in kwargs:
        data_dir_val = kwargs.get("data_dir")
        if not data_dir_val or not tokenizer:
            raise ValueError("When config is not provided, 'data_dir' and 'tokenizer' must be supplied")
        from pathlib import Path as _Path

        _p = _Path(data_dir_val)
        if not _p.exists():
            _alt = _Path.cwd() / _p.name
            if _alt.exists():
                data_dir_val = str(_alt)
        data_cfg: Dict[str, Any] = {"data_dir": data_dir_val}
        for key in (
            "batch_size",
            "num_workers",
            "prefetch_factor",
            "log_frequency",
            "load_in_memory",
            "pin_memory",
            "persistent_workers",
            "eval_batch_size_multiplier",
            "max_eval_batch_size",
            "img_dir",
            "cache_dir",
            "bucket_batching",
            "augment",
        ):
            if key in kwargs and kwargs[key] is not None:
                data_cfg[key] = kwargs[key]
        model_cfg: Dict[str, Any] = {"name": kwargs.get("model_name", "cnn_transformer")}
        model_cfg["encoder"] = {"cnn": {"channels": kwargs.get("channels", 1)}}
        config = {"data": data_cfg, "model": model_cfg}

    data_config = config.get("data", {})
    model_conf = config.get("model", {})
    data_dir = data_config.get("data_dir")
    model_type = model_conf.get("name", "cnn_transformer")
    encoder_config = model_conf.get("encoder", {})

    batch_size = data_config.get("batch_size", 32)
    num_workers = data_config.get("num_workers", 0)
    prefetch_factor = data_config.get("prefetch_factor", 2)
    log_frequency = data_config.get("log_frequency", 1000)
    load_in_memory = data_config.get("load_in_memory", False)
    persistent_workers = data_config.get("persistent_workers", num_workers > 0)
    pin_memory = data_config.get("pin_memory", torch.cuda.is_available())
    bucket_batching = data_config.get("bucket_batching", True)
    use_augment = data_config.get("augment", True)

    if max_samples is None:
        max_samples = {"train": None, "val": None, "test": None}

    if model_type.startswith("resnet"):
        enc = encoder_config.get("resnet", {})
        channels = enc.get("channels", 3)
        img_size = (enc.get("img_height", 64), enc.get("img_width", 512))
    else:
        enc = encoder_config.get("cnn", encoder_config)
        channels = enc.get("channels", 1)
        img_size = (enc.get("img_height", 64), enc.get("img_width", 512))

    dataloader_kwargs = {}
    if num_workers > 0:
        def _worker_init(_worker_id: int) -> None:
            import os

            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            os.environ["OMP_NUM_THREADS"] = "1"

        dataloader_kwargs.update(
            {
                "num_workers": num_workers,
                "prefetch_factor": prefetch_factor,
                "persistent_workers": persistent_workers,
                "worker_init_fn": _worker_init,
            }
        )
    dataloader_kwargs["pin_memory"] = pin_memory

    collator = Im2LatexCollator(pad_token_id=tokenizer.pad_token_id)
    train_file = data_config.get("train_file", "im2latex_train_filter.lst")
    validate_file = data_config.get("validate_file", "im2latex_validate_filter.lst")
    test_file = data_config.get("test_file", "im2latex_test_filter.lst")
    formulas_file = data_config.get("formulas_file", "im2latex_formulas.norm.lst")
    img_dir = data_config.get("img_dir", "img")
    split_files = {"train": train_file, "val": validate_file, "test": test_file}

    cache_dir = Path(data_config.get("cache_dir", str(Path(data_dir) / "cache")))
    cache_images = cache_widths = cache_heights = cache_index = None
    img_root = Path(data_dir) / img_dir
    if img_root.exists():
        try:
            if not cache_is_complete(cache_dir, img_size):
                build_image_cache(
                    data_dir=data_dir,
                    img_dir=img_dir,
                    split_files=split_files.values(),
                    img_size=img_size,
                    cache_dir=str(cache_dir),
                )
            if cache_is_complete(cache_dir, img_size):
                cache_images, cache_widths, cache_heights, cache_index = load_image_cache(
                    cache_dir, img_size
                )
                logger.info("Using image cache at %s (%s images)", cache_dir, len(cache_index))
        except Exception as exc:
            logger.warning("Image cache unavailable (%s); falling back to PNG loads", exc)

    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = Im2LatexDataset(
            data_dir=data_dir,
            split_file=split_files[split],
            formulas_file=formulas_file,
            tokenizer=tokenizer,
            img_dir=img_dir,
            img_size=img_size,
            channels=channels,
            transform=None,
            max_samples=max_samples.get(split),
            load_in_memory=load_in_memory,
            log_frequency=log_frequency,
            augment=(use_augment and split == "train"),
            cache_images=cache_images,
            cache_widths=cache_widths,
            cache_heights=cache_heights,
            cache_index=cache_index,
        )

    if not datasets or all(len(ds) == 0 for ds in datasets.values() if ds is not None):
        logger.warning("All datasets are empty! Check file paths and max_samples settings.")
        return {}

    eval_batch_multiplier = data_config.get("eval_batch_size_multiplier", 1)
    max_eval_batch_size = data_config.get("max_eval_batch_size", 64)

    loaders = {}
    for split in ["train", "val", "test"]:
        if split not in datasets or datasets[split] is None:
            continue
        current_batch_size = (
            batch_size
            if split == "train"
            else min(batch_size * eval_batch_multiplier, max_eval_batch_size)
        )
        extra = dict(dataloader_kwargs)
        if split == "train" and bucket_batching and len(datasets[split]) >= current_batch_size:
            sampler = BucketBatchSampler(
                datasets[split].lengths, batch_size=current_batch_size, drop_last=True
            )
            loaders[split] = DataLoader(
                datasets[split],
                batch_sampler=sampler,
                collate_fn=collator,
                **extra,
            )
        else:
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=current_batch_size,
                shuffle=(split == "train"),
                collate_fn=collator,
                drop_last=(split == "train"),
                **extra,
            )
        logger.info(
            "%s DataLoader: %s samples, batch_size=%s, num_workers=%s",
            split.capitalize(),
            len(datasets[split]),
            current_batch_size,
            extra.get("num_workers", 0),
        )
    return loaders
