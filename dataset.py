"""MFNet RGB + thermal segmentation dataset with synchronized, always-on augmentation.

Expected data layout:
  MFNet_dataset/
    images/<name>.png   # RGBA PNG: RGB channels 0:3, thermal stored in alpha
    labels/<name>.png   # single-channel class IDs 0..8 (255 may be ignored)
    train.txt
    val.txt
    test.txt (optional for training; required only for a test-set audit)

Images returned as float32 [4,H,W] in [0,1]. RGB normalization for the pretrained
encoder and thermal normalization are applied inside my_model.py. Label IDs are
never linearly interpolated.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import DataLoader, Dataset

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
NUM_CLASSES = 9
IN_CHANNELS = 4
IGNORE_INDEX = 255
NUM_WORKERS = 4

CLASS_NAMES = [
    "Unlabeled", "Car", "Person", "Bike", "Curve",
    "Car Stop", "Guardrail", "Color Cone", "Bump",
]
KAGGLE_DATASET_ID = "danialqashqai/mfnet-dataset"

# Also supports masks stored as RGB visualizations (most datasets should be indexed/grayscale).
MFNET_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [220, 20, 60],
        [0, 200, 0],
        [30, 144, 255],
        [255, 215, 0],
        [255, 20, 147],
        [0, 255, 255],
        [255, 140, 0],
        [138, 43, 226],
    ],
    dtype=np.uint8,
)


def _dataset_is_ready(data_root: str) -> bool:
    root = Path(data_root)
    return (
        (root / "images").is_dir()
        and (root / "labels").is_dir()
        and (root / "train.txt").is_file()
        and (root / "val.txt").is_file()
        and any(p.suffix.lower() == ".png" for p in (root / "images").iterdir())
        and any(p.suffix.lower() == ".png" for p in (root / "labels").iterdir())
    )


def _try_kaggle_download(data_root: str) -> bool:
    """Download the source dataset with the Kaggle CLI if it is not present."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[dataset.py] Kaggle CLI package is missing; install it with `pip install kaggle`.")
        return False

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"[dataset.py] Downloading {KAGGLE_DATASET_ID} to '{root}'...")
    command = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET_ID, "--unzip", "-p", str(root),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[dataset.py] Kaggle download failed:\n{result.stderr or result.stdout}")
        return False

    for nested in (root / "MFNet-dataset" / "ir_seg_dataset",
                   root / "MFNet-dataset", root / "ir_seg_dataset"):
        if nested.is_dir() and _dataset_is_ready(str(nested)):
            for child in nested.iterdir():
                destination = root / child.name
                if not destination.exists():
                    shutil.move(str(child), str(destination))
            # Remove only the expected Kaggle wrapper folder if empty.
            for wrapper in (root / "MFNet-dataset",):
                if wrapper.exists():
                    try:
                        shutil.rmtree(wrapper)
                    except OSError:
                        pass
            break
    return _dataset_is_ready(str(root))


def ensure_dataset(data_root: str) -> bool:
    if _dataset_is_ready(data_root):
        print(f"[dataset.py] Dataset found at '{data_root}'.")
        return True
    print(f"[dataset.py] Dataset not found or incomplete at '{data_root}'.")
    if _try_kaggle_download(data_root):
        print(f"[dataset.py] Dataset is ready at '{data_root}'.")
        return True
    return False


def _canonical_name(name: str) -> str:
    name = Path(name).stem
    return name[:-5] if name.endswith("_flip") else name


def _read_split_canonical_names(data_root: str, split: str) -> set[str]:
    split_path = Path(data_root) / f"{split}.txt"
    with split_path.open("r", encoding="utf-8-sig") as file:
        return {_canonical_name(line.strip()) for line in file if line.strip()}


class MFNetDataset(Dataset):
    """MFNet dataset returning 4-channel float images and integer segmentation masks."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: Tuple[int, int] = IMAGE_SIZE,
        augment: bool = True,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split {split!r}; expected train, val, or test.")
        self.data_root = str(data_root)
        self.split = split
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.augment = bool(augment and split == "train")
        self.ignore_index = int(ignore_index)
        self.num_classes = NUM_CLASSES
        self.images_dir = os.path.join(data_root, "images")
        self.labels_dir = os.path.join(data_root, "labels")

        split_file = os.path.join(data_root, f"{split}.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")
        with open(split_file, "r", encoding="utf-8-sig") as file:
            raw_names = [line.strip() for line in file if line.strip()]
        self.names = [Path(name).stem for name in raw_names]

        self.samples: list[tuple[str, str, bool]] = []
        missing: list[str] = []
        for name in self.names:
            img_path = os.path.join(self.images_dir, f"{name}.png")
            mask_path = os.path.join(self.labels_dir, f"{name}.png")
            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                self.samples.append((img_path, mask_path, False))
                continue

            if name.endswith("_flip"):
                base = name[:-5]
                img_path = os.path.join(self.images_dir, f"{base}.png")
                mask_path = os.path.join(self.labels_dir, f"{base}.png")
                if os.path.isfile(img_path) and os.path.isfile(mask_path):
                    self.samples.append((img_path, mask_path, True))
                    continue
            missing.append(name)

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} files listed in {split}.txt are missing. "
                f"Examples: {missing[:8]}"
            )
        if not self.samples:
            raise RuntimeError(f"No usable samples found for split='{split}' at {data_root}.")

        print(f"[dataset.py] Split='{split}' | {len(self.samples)} samples | augment={self.augment}")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_rgba(image_path: str) -> np.ndarray:
        with Image.open(image_path) as image:
            image_array = np.array(image, copy=True)
            source_mode = image.mode

        # This project expects thermal intensity in the PNG alpha channel.
        if image_array.ndim != 3 or image_array.shape[2] != 4:
            raise ValueError(
                f"Expected a 4-channel RGB+thermal PNG (RGBA alpha=thermal), got "
                f"mode={source_mode!r}, shape={image_array.shape} at {image_path}. "
                "Refusing to silently replace missing thermal data with zeros."
            )
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        return image_array

    @staticmethod
    def _read_mask(mask_path: str) -> np.ndarray:
        with Image.open(mask_path) as mask:
            array = np.array(mask, copy=True)
            mode = mask.mode

        if array.ndim == 2:
            return array.astype(np.int64, copy=False)

        if array.ndim == 3:
            # RGBA/ RGB mask with equal channels is still a grayscale ID mask.
            rgb = array[..., :3]
            if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
                return rgb[..., 0].astype(np.int64, copy=False)

            # If the source accidentally stores a colorized mask, map known MFNet colors.
            out = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.int64)
            for class_id, color in enumerate(MFNET_PALETTE):
                out[np.all(rgb == color, axis=-1)] = class_id
            unknown = np.all(out == IGNORE_INDEX)
            if unknown:
                raise ValueError(
                    f"Mask {mask_path} is RGB mode {mode!r} but is not grayscale or the "
                    "known MFNet visualization palette. Inspect the mask encoding before training."
                )
            return out

        raise ValueError(f"Unsupported mask shape {array.shape} in {mask_path}")

    def _resize(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = self.img_size
        image_pil = Image.fromarray(image, mode="RGBA")
        mask_pil = Image.fromarray(mask.astype(np.uint8), mode="L")
        image_pil = image_pil.resize((width, height), resample=Image.Resampling.BILINEAR)
        mask_pil = mask_pil.resize((width, height), resample=Image.Resampling.NEAREST)
        return np.array(image_pil, copy=True), np.array(mask_pil, copy=True).astype(np.int64)

    def _spatial_augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = self.img_size

        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # Mild paired scale jitter. Image uses bilinear interpolation; masks use nearest.
        if random.random() < 0.80:
            scale = random.uniform(0.90, 1.10)
            scaled_h = max(2, int(round(height * scale)))
            scaled_w = max(2, int(round(width * scale)))
            image_pil = Image.fromarray(image, mode="RGBA").resize(
                (scaled_w, scaled_h), resample=Image.Resampling.BILINEAR
            )
            mask_pil = Image.fromarray(mask.astype(np.uint8), mode="L").resize(
                (scaled_w, scaled_h), resample=Image.Resampling.NEAREST
            )
            image = np.array(image_pil, copy=True)
            mask = np.array(mask_pil, copy=True).astype(np.int64)

            # Random crop when zoomed in; reflected image padding and ignored target
            # pixels when zoomed out. The same crop/pad geometry is used for all modalities.
            if scaled_h > height:
                top = random.randint(0, scaled_h - height)
                image = image[top:top + height]
                mask = mask[top:top + height]
            elif scaled_h < height:
                total = height - scaled_h
                top = random.randint(0, total)
                bottom = total - top
                image = np.pad(image, ((top, bottom), (0, 0), (0, 0)), mode="reflect")
                mask = np.pad(mask, ((top, bottom), (0, 0)), mode="constant",
                              constant_values=self.ignore_index)

            if scaled_w > width:
                left = random.randint(0, scaled_w - width)
                image = image[:, left:left + width]
                mask = mask[:, left:left + width]
            elif scaled_w < width:
                total = width - scaled_w
                left = random.randint(0, total)
                right = total - left
                image = np.pad(image, ((0, 0), (left, right), (0, 0)), mode="reflect")
                mask = np.pad(mask, ((0, 0), (left, right)), mode="constant",
                              constant_values=self.ignore_index)

        return image, mask

    @staticmethod
    def _photometric_augment(image: np.ndarray) -> np.ndarray:
        # RGB color jitter is deliberately isolated from the thermal channel.
        rgb = Image.fromarray(image[..., :3], mode="RGB")
        if random.random() < 0.75:
            rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.85, 1.15))
            rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(0.85, 1.15))
            rgb = ImageEnhance.Color(rgb).enhance(random.uniform(0.90, 1.10))
        image[..., :3] = np.array(rgb, copy=True)

        # Use smaller, independent thermal gain/bias jitter to preserve heat contrast.
        if random.random() < 0.30:
            thermal = image[..., 3].astype(np.float32) / 255.0
            gain = random.uniform(0.90, 1.10)
            bias = random.uniform(-0.025, 0.025)
            thermal = np.clip((thermal - 0.5) * gain + 0.5 + bias, 0.0, 1.0)
            image[..., 3] = np.round(thermal * 255.0).astype(np.uint8)
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path, need_manifest_flip = self.samples[index]
        image = self._read_rgba(image_path)
        mask = self._read_mask(mask_path)

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"RGB+thermal/mask size mismatch for {image_path} and {mask_path}: "
                f"{image.shape[:2]} vs {mask.shape[:2]}"
            )

        if need_manifest_flip:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        image, mask = self._resize(image, mask)
        if self.augment:
            image, mask = self._spatial_augment(image, mask)
            image = self._photometric_augment(image)

        invalid = (mask != self.ignore_index) & ((mask < 0) | (mask >= NUM_CLASSES))
        mask[invalid] = self.ignore_index
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)
        mask_tensor = torch.from_numpy(mask.astype(np.int64, copy=True)).long()
        return image_tensor, mask_tensor


def get_dataloaders(
    data_root: str = "../MFNet_dataset",
    batch_size: int = 4,
    img_size: Tuple[int, int] = IMAGE_SIZE,
    num_workers: int = NUM_WORKERS,
    auto_download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    root = os.path.abspath(data_root)
    if not _dataset_is_ready(root):
        nested = os.path.join(root, "ir_seg_dataset")
        if _dataset_is_ready(nested):
            root = nested
        elif auto_download:
            ensure_dataset(root)
            if not _dataset_is_ready(root) and _dataset_is_ready(nested):
                root = nested

    if not _dataset_is_ready(root):
        raise FileNotFoundError(
            f"MFNet dataset not found or incomplete at {root}. Expected images/, labels/, "
            "train.txt and val.txt. Run audit_mfnet.py to inspect the data."
        )

    train_names = _read_split_canonical_names(root, "train")
    val_names = _read_split_canonical_names(root, "val")
    overlap = train_names & val_names
    if overlap:
        example = sorted(overlap)[:10]
        raise RuntimeError(
            f"Train/validation data leakage detected: {len(overlap)} canonical filenames overlap. "
            f"Examples: {example}"
        )

    print(f"[dataset.py] Loading MFNet dataset from: {root}")
    train_ds = MFNetDataset(root, split="train", img_size=img_size, augment=True)
    val_ds = MFNetDataset(root, split="val", img_size=img_size, augment=False)

    workers = max(0, int(num_workers))
    loader_options = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **loader_options
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **loader_options
    )
    return train_loader, val_loader
