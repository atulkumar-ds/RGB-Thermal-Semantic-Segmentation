"""Deep dataset audit for the MFNet RGB+thermal project.

Run from the repository root:
    python audit_mfnet.py --data-root ../MFNet_dataset

This script is read-only and never downloads, renames, or edits dataset files.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = [
    "Unlabeled", "Car", "Person", "Bike", "Curve",
    "Car Stop", "Guardrail", "Color Cone", "Bump",
]
NUM_CLASSES = 9
IGNORE_INDEX = 255


def canonical(name: str) -> str:
    name = Path(name).stem
    return name[:-5] if name.endswith("_flip") else name


def read_names(root: Path, split: str) -> list[str]:
    path = root / f"{split}.txt"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig") as f:
        return [Path(line.strip()).stem for line in f if line.strip()]


def find_pair(root: Path, name: str):
    image = root / "images" / f"{name}.png"
    mask = root / "labels" / f"{name}.png"
    flipped = False
    if (not image.is_file() or not mask.is_file()) and name.endswith("_flip"):
        base = name[:-5]
        image = root / "images" / f"{base}.png"
        mask = root / "labels" / f"{base}.png"
        flipped = True
    return image, mask, flipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../MFNet_dataset")
    parser.add_argument("--image-check-limit", type=int, default=0,
                        help="Inspect up to this many image files for RGBA/thermal statistics; 0 means all (default).")
    args = parser.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    print(f"MFNet audit root: {root}")
    for required in ("images", "labels", "train.txt", "val.txt"):
        exists = (root / required).exists()
        print(f"  {'[OK]' if exists else '[MISSING]'} {required}")
    if not all((root / p).exists() for p in ("images", "labels", "train.txt", "val.txt")):
        print("\nDataset layout is incomplete; no files were changed.")
        return 2

    split_names = {split: read_names(root, split) for split in ("train", "val", "test")}
    canonical_sets = {}
    for split, names in split_names.items():
        if not names:
            continue
        duplicates = [n for n, count in Counter(names).items() if count > 1]
        canonical_sets[split] = {canonical(n) for n in names}
        missing = []
        for name in names:
            image, mask, _ = find_pair(root, name)
            if not image.is_file() or not mask.is_file():
                missing.append(name)
        print(f"\n[{split}] entries={len(names)}, unique canonical files={len(canonical_sets[split])}, "
              f"duplicate lines={len(duplicates)}, missing pairs={len(missing)}")
        if duplicates:
            print(f"  Duplicate entries (first 10): {duplicates[:10]}")
        if missing:
            print(f"  Missing pairs (first 10): {missing[:10]}")

    print("\nChecking pairwise split overlap by canonical filename...")
    pairwise_overlap_found = False
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if left not in canonical_sets or right not in canonical_sets:
            continue
        overlap = sorted(canonical_sets[left] & canonical_sets[right])
        print(f"  {left}/{right} overlap: {len(overlap)}")
        if overlap:
            pairwise_overlap_found = True
            print(f"    WARNING: possible data leakage: {overlap[:20]}")
    if not pairwise_overlap_found:
        print("  [OK] No canonical filename overlap across available splits.")

    image_dir = root / "images"
    physical_image_names = [p.stem for p in image_dir.glob("*.png")]
    canonical_image_names = {canonical(name) for name in physical_image_names}
    listed_canonical_names = set().union(*canonical_sets.values()) if canonical_sets else set()
    unlisted_images = sorted(canonical_image_names - listed_canonical_names)
    duplicate_physical_variants = len(physical_image_names) - len(canonical_image_names)
    print(f"\nImage inventory: {len(physical_image_names)} PNG files, "
          f"{len(canonical_image_names)} canonical stems, "
          f"{duplicate_physical_variants} extra physical variants (e.g. *_flip).")
    print(f"  Canonical image stems absent from all split manifests: {len(unlisted_images)}")
    if unlisted_images:
        print(f"  INFO: first unlisted stems: {unlisted_images[:20]}")

    print("\nScanning masks for class IDs and pixel imbalance...")
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    ignored_pixels = 0
    unexpected_counts = Counter()
    mask_modes = Counter()
    size_mismatches = []
    total_masks = 0
    for split in ("train", "val"):
        for name in split_names[split]:
            _, mask_path, _ = find_pair(root, name)
            if not mask_path.is_file():
                continue
            with Image.open(mask_path) as im:
                array = np.asarray(im)
                mask_modes[im.mode] += 1
            if array.ndim == 3:
                if np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(array[..., 0], array[..., 2]):
                    array = array[..., 0]
                else:
                    print(f"  WARNING: color mask encoding needs inspection: {mask_path.name}, mode={im.mode}")
                    array = array[..., 0]
            values, counts = np.unique(array, return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                if 0 <= value < NUM_CLASSES:
                    pixel_counts[int(value)] += int(count)
                elif value == IGNORE_INDEX:
                    ignored_pixels += int(count)
                else:
                    unexpected_counts[int(value)] += int(count)
            image_path, _, _ = find_pair(root, name)
            if image_path.is_file():
                with Image.open(image_path) as im_img:
                    im_size = im_img.size
                if im_size != (array.shape[1], array.shape[0]):
                    size_mismatches.append((name, im_size, (array.shape[1], array.shape[0])))
            total_masks += 1

    all_valid = pixel_counts.sum()
    print(f"  Masks scanned: {total_masks}")
    print(f"  Mask modes: {dict(mask_modes)}")
    print(f"  Ignored pixels (255): {ignored_pixels:,}")
    print(f"  Unexpected label IDs: {dict(unexpected_counts) if unexpected_counts else 'none'}")
    for class_id, (class_name, count) in enumerate(zip(CLASS_NAMES, pixel_counts.tolist())):
        pct = 100.0 * count / max(1, all_valid)
        print(f"  class {class_id:>2} {class_name:<12}: {count:>14,} pixels ({pct:6.3f}%)")
    if size_mismatches:
        print(f"  WARNING: {len(size_mismatches)} image/mask size mismatches. Examples: {size_mismatches[:10]}")
    if unexpected_counts:
        print("  WARNING: remapping/ignoring labels should be reviewed before training.")

    print("\nInspecting RGB+thermal image encoding...")
    image_files = sorted((root / "images").glob("*.png"))
    if args.image_check_limit > 0:
        image_files = image_files[:args.image_check_limit]
    modes = Counter()
    channel_shapes = Counter()
    thermal_image_stats = []
    thermal_pixel_count = 0
    thermal_pixel_sum = 0.0
    thermal_pixel_sq_sum = 0.0
    thermal_global_min = 255
    thermal_global_max = 0
    all_image_files = sorted((root / "images").glob("*.png"))
    for image_path in image_files:
        with Image.open(image_path) as im:
            modes[im.mode] += 1
            arr = np.asarray(im)
        channel_shapes[str(arr.shape)] += 1
        if arr.ndim == 3 and arr.shape[-1] == 4:
            thermal = arr[..., 3].astype(np.float64)
            thermal_pixel_count += int(thermal.size)
            thermal_pixel_sum += float(thermal.sum())
            thermal_pixel_sq_sum += float(np.square(thermal).sum())
            thermal_global_min = min(thermal_global_min, int(thermal.min()))
            thermal_global_max = max(thermal_global_max, int(thermal.max()))
            thermal_image_stats.append((float(thermal.mean()), float(thermal.std())))
    print(f"  Images inspected: {len(image_files)} / {len(all_image_files)}")
    print(f"  Modes: {dict(modes)}")
    print(f"  Shapes: {dict(channel_shapes)}")
    if thermal_image_stats and thermal_pixel_count:
        global_mean = thermal_pixel_sum / thermal_pixel_count
        global_var = max(0.0, thermal_pixel_sq_sum / thermal_pixel_count - global_mean ** 2)
        global_std = global_var ** 0.5
        per_image_means = np.asarray([x[0] for x in thermal_image_stats], dtype=np.float64)
        per_image_stds = np.asarray([x[1] for x in thermal_image_stats], dtype=np.float64)
        print(f"  Thermal alpha PIXEL stats: min={thermal_global_min}, "
              f"mean={global_mean:.2f}, max={thermal_global_max}, global_std={global_std:.2f}")
        print(f"  Per-image alpha mean: min={per_image_means.min():.2f}, "
              f"median={np.median(per_image_means):.2f}, max={per_image_means.max():.2f}")
        constant_thermal = int(np.sum(per_image_stds < 1.0))
        print(f"  Near-constant thermal alpha images (per-image std<1): {constant_thermal}")
        if constant_thermal:
            print("  WARNING: inspect whether alpha really stores thermal for these images.")
    else:
        print("  ERROR: no 4-channel images found in the inspected sample.")

    print("\nAudit complete. This script did not change any dataset files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
