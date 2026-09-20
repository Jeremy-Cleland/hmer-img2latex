"""One-time image cache: uint8 memmap of resized/padded formula images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from img2latex.data.transforms import ResizeWithAspectRatio
from img2latex.utils.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


def _unique_filenames(data_dir: Path, split_files: Iterable[str]) -> List[str]:
    seen = set()
    names: List[str] = []
    for split in split_files:
        path = data_dir / split
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue
                name = parts[0]
                if name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


def cache_paths(cache_dir: Path, img_size: Tuple[int, int]) -> Dict[str, Path]:
    height, width = img_size
    tag = f"{height}x{width}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "images": cache_dir / f"images_{tag}.npy",
        "widths": cache_dir / f"widths_{tag}.npy",
        "heights": cache_dir / f"heights_{tag}.npy",
        "index": cache_dir / f"index_{tag}.json",
    }


def cache_is_complete(cache_dir: Path, img_size: Tuple[int, int]) -> bool:
    paths = cache_paths(cache_dir, img_size)
    return all(path.exists() for path in paths.values())


def build_image_cache(
    data_dir: str,
    img_dir: str,
    split_files: Iterable[str],
    img_size: Tuple[int, int] = (64, 512),
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Path]:
    """Write a memmapped uint8 cache of every unique split image."""
    data_path = Path(data_dir)
    image_root = data_path / img_dir
    cache_path = Path(cache_dir) if cache_dir else data_path / "cache"
    paths = cache_paths(cache_path, img_size)
    if cache_is_complete(cache_path, img_size) and not force:
        logger.info("Image cache already exists at %s", cache_path)
        return paths

    filenames = _unique_filenames(data_path, split_files)
    if not filenames:
        raise FileNotFoundError(f"No image filenames found under {data_path}")

    height, width = img_size
    n_images = len(filenames)
    logger.info("Building image cache for %s images at %sx%s", n_images, height, width)

    images = np.lib.format.open_memmap(
        paths["images"], mode="w+", dtype=np.uint8, shape=(n_images, height, width)
    )
    widths = np.zeros(n_images, dtype=np.int16)
    heights = np.zeros(n_images, dtype=np.int16)
    resize = ResizeWithAspectRatio(height, width)
    index: Dict[str, int] = {}

    for i, name in enumerate(tqdm(filenames, desc="Caching images")):
        image_file = image_root / name
        try:
            img = Image.open(image_file).convert("L")
            padded, valid_w, valid_h = resize(img)
            images[i] = np.array(padded, dtype=np.uint8)
            widths[i] = valid_w
            heights[i] = valid_h
        except Exception as exc:
            logger.error("Failed to cache %s: %s", image_file, exc)
            images[i] = 255
            widths[i] = width
            heights[i] = height
        index[name] = i
        if i % 5000 == 0:
            images.flush()

    images.flush()
    np.save(paths["widths"], widths)
    np.save(paths["heights"], heights)
    with open(paths["index"], "w", encoding="utf-8") as handle:
        json.dump(index, handle)
    logger.info("Wrote image cache to %s", cache_path)
    return paths


def load_image_cache(cache_dir: Path, img_size: Tuple[int, int]):
    paths = cache_paths(cache_dir, img_size)
    images = np.load(paths["images"], mmap_mode="r")
    widths = np.load(paths["widths"])
    heights = np.load(paths["heights"])
    with open(paths["index"], "r", encoding="utf-8") as handle:
        index = json.load(handle)
    return images, widths, heights, index
