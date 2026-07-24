"""Dataset loading, preprocessing, and stratified splitting for EL images.

This module reads a ZAE-Bayern-format CSV of EL solar cell images, applies
denoising (5×5 median filter) and pseudo-RGB conversion, normalises with
ImageNet statistics, and produces 70/15/15 train/val/test DataLoaders with
bucket-stratified splitting and inverse-frequency weighted sampling.

Typical usage::

    train_loader, val_loader, test_loader, counts = create_dataloaders(
        csv_path="labels.csv",
        data_root=".",
        batch_size=32,
        seed=42,
    )
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
"""ImageNet channel-wise means (RGB)."""

IMAGENET_STD = [0.229, 0.224, 0.225]
"""ImageNet channel-wise standard deviations (RGB)."""

REQUIRED_CSV_COLUMNS = {"image_path", "defect_probability", "cell_type"}
"""Expected columns in the ZAE-Bayern-format CSV."""

MEDIAN_FILTER_KERNEL = 5
"""Kernel size for the median-blur denoising step (must be odd)."""


# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------


def defect_probability_to_bucket(defect_probability: float) -> int:
    """Map a continuous defect probability to a discrete stratification bucket.

    Buckets are defined as:
        - Bucket 0: [0.0000, 0.1667)
        - Bucket 1: [0.1667, 0.5000)
        - Bucket 2: [0.5000, 0.8334)
        - Bucket 3: [0.8334, 1.0000]

    Args:
        defect_probability: A float in ``[0.0, 1.0]``.

    Returns:
        Integer bucket index (0--3).
    """
    if defect_probability < 0.1667:
        return 0
    if defect_probability < 0.5:
        return 1
    if defect_probability < 0.8334:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SplitItems:
    """Container for the three-way dataset split.

    Attributes:
        train: Training samples as ``(image_path, defect_probability)`` tuples.
        val: Validation samples.
        test: Held-out test samples.
    """

    train: List[Tuple[Path, float]]
    val: List[Tuple[Path, float]]
    test: List[Tuple[Path, float]]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all frameworks.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_csv_samples(
    csv_path: Path, data_root: Path
) -> List[Tuple[Path, float]]:
    """Load and validate image samples from a ZAE-Bayern-format CSV.

    Expects columns ``image_path``, ``defect_probability``, and
    ``cell_type``.  Validates that every referenced image exists on disk
    and that defect probabilities fall within ``[0.0, 1.0]``.

    Args:
        csv_path: Path to the CSV file.
        data_root: Base directory for resolving relative image paths.

    Returns:
        Sorted list of ``(absolute_image_path, defect_probability)`` pairs.

    Raises:
        FileNotFoundError: If the CSV or any referenced image is missing.
        ValueError: If required columns are absent or probabilities are
            out of range.
        RuntimeError: If no valid samples remain after validation.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV dataset file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing_columns = REQUIRED_CSV_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"CSV missing required columns: {sorted(missing_columns)}"
        )

    samples: List[Tuple[Path, float]] = []
    missing_files: List[str] = []

    for row in frame.itertuples(index=False):
        raw_image_path = str(row.image_path)
        image_path = Path(raw_image_path)
        if not image_path.is_absolute():
            image_path = data_root / image_path

        defect_probability = float(row.defect_probability)
        if defect_probability < 0.0 or defect_probability > 1.0:
            raise ValueError(
                f"Invalid defect_probability {defect_probability} for "
                f"image_path={raw_image_path}. Expected values in [0.0, 1.0]."
            )

        if image_path.exists() and image_path.is_file():
            samples.append((image_path, defect_probability))
        else:
            missing_files.append(str(image_path))

    if missing_files:
        preview = "\n".join(missing_files[:10])
        suffix = (
            ""
            if len(missing_files) <= 10
            else f"\n... and {len(missing_files) - 10} more"
        )
        raise FileNotFoundError(
            f"CSV references missing image files:\n{preview}{suffix}"
        )

    if not samples:
        raise RuntimeError("No valid image samples were found from CSV rows.")

    return sorted(samples, key=lambda item: str(item[0]))


# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------


def stratified_split(
    samples: Sequence[Tuple[Path, float]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitItems:
    """Split samples into train/val/test sets with per-bucket stratification.

    Samples are first grouped into severity buckets (via
    :func:`defect_probability_to_bucket`), then split independently per
    bucket to preserve the defect-severity distribution across all three
    sets.  After splitting, items are interleaved and shuffled.

    Args:
        samples: ``(image_path, defect_probability)`` pairs.
        train_ratio: Fraction of samples for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`SplitItems` object with the three splits.

    Raises:
        ValueError: If the ratios do not sum to 1.0.
        RuntimeError: If any bucket has too few samples for a 70/15/15
            split (minimum 3).
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    rng = random.Random(seed)
    by_bucket: Dict[int, List[Tuple[Path, float]]] = {
        0: [], 1: [], 2: [], 3: []
    }
    for image_path, defect_probability in samples:
        bucket = defect_probability_to_bucket(defect_probability)
        by_bucket[bucket].append((image_path, defect_probability))

    train_items: List[Tuple[Path, float]] = []
    val_items: List[Tuple[Path, float]] = []
    test_items: List[Tuple[Path, float]] = []

    for bucket, bucket_items in by_bucket.items():
        if not bucket_items:
            continue

        if len(bucket_items) < 3:
            raise RuntimeError(
                f"Bucket {bucket} has too few samples "
                f"({len(bucket_items)}) for 70/15/15 split."
            )

        shuffled = bucket_items[:]
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        n_test = n_total - n_train - n_val

        if n_test < 1:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1

        train_items.extend(shuffled[:n_train])
        val_items.extend(shuffled[n_train : n_train + n_val])
        test_items.extend(shuffled[n_train + n_val :])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    rng.shuffle(test_items)

    return SplitItems(train=train_items, val=val_items, test=test_items)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def build_transforms(
    image_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """Create train and evaluation image transformation pipelines.

    Both pipelines apply:
        1. Pseudo-RGB conversion via ``Grayscale(3)`` (replicates the
           single EL channel across R, G, and B for compatibility with
           ImageNet-pretrained backbones).
        2. Resize to ``(image_size, image_size)``.
        3. ``ToTensor()``.
        4. Normalisation with ImageNet mean and std.

    The training pipeline additionally applies random horizontal and
    vertical flips for data augmentation.

    Args:
        image_size: Target square dimension in pixels.

    Returns:
        A 3-tuple of ``(train_transform, val_transform, test_transform)``.
        Validation and test transforms are identical (no augmentation).
    """
    train_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_tfms, eval_tfms, eval_tfms


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class ELDataset(Dataset):
    """PyTorch Dataset for electroluminescence solar cell images.

    Each item is loaded as a single-channel grayscale image, denoised with
    a 5×5 median filter, converted to a PIL image, and passed through the
    provided transformation pipeline.

    Args:
        items: ``(image_path, defect_probability)`` pairs.
        transform: A ``torchvision`` transform composition.
    """

    def __init__(
        self, items: Sequence[Tuple[Path, float]], transform: transforms.Compose
    ) -> None:
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load, denoise, and transform the image at ``idx``.

        Args:
            idx: Sample index.

        Returns:
            ``(image_tensor, target)`` where ``image_tensor`` has shape
            ``(3, H, W)`` and ``target`` is a scalar float tensor.

        Raises:
            RuntimeError: If the image file cannot be read.
        """
        image_path, defect_probability = self.items[idx]

        image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        image_denoised = cv2.medianBlur(image_gray, MEDIAN_FILTER_KERNEL)
        image_pil = Image.fromarray(image_denoised)
        image_tensor = self.transform(image_pil)
        target = torch.tensor(defect_probability, dtype=torch.float32)

        return image_tensor, target


# ---------------------------------------------------------------------------
# Weighted sampling
# ---------------------------------------------------------------------------


def build_weighted_sampler(
    items: Sequence[Tuple[Path, float]],
) -> WeightedRandomSampler:
    """Create an inverse-frequency weighted sampler for balanced training.

    Weights each sample by ``1 / bucket_count`` so that underrepresented
    severity buckets are sampled more frequently during training.

    Args:
        items: ``(image_path, defect_probability)`` pairs for the training set.

    Returns:
        A ``WeightedRandomSampler`` with replacement.
    """
    bucket_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    for _, defect_probability in items:
        bucket = defect_probability_to_bucket(defect_probability)
        bucket_counts[bucket] += 1

    sample_weights: List[float] = []
    for _, defect_probability in items:
        bucket = defect_probability_to_bucket(defect_probability)
        count = bucket_counts[bucket]
        weight = 1.0 / float(count) if count > 0 else 0.0
        sample_weights.append(weight)

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------


def create_dataloaders(
    csv_path: str,
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """End-to-end pipeline: CSV → stratified split → DataLoaders.

    This is the primary entry point for training scripts.  It chains
    together CSV loading, stratified 70/15/15 splitting, transform
    construction, dataset creation, weighted-sampler setup, and
    DataLoader instantiation.

    Args:
        csv_path: Path to the ZAE-Bayern-format labels CSV.
        data_root: Root directory for resolving relative image paths.
        batch_size: Mini-batch size.
        num_workers: Number of ``DataLoader`` worker processes.
        image_size: Input image dimension (square).
        seed: Random seed for reproducibility.

    Returns:
        A 4-tuple of ``(train_loader, val_loader, test_loader,
        split_counts)`` where ``split_counts`` is a dict with keys
        ``"train"``, ``"val"``, ``"test"``, and ``"total"``.
    """
    set_seed(seed)

    all_samples = load_csv_samples(Path(csv_path), Path(data_root))
    split_items = stratified_split(all_samples, 0.70, 0.15, 0.15, seed)

    train_tfms, val_tfms, test_tfms = build_transforms(image_size=image_size)

    train_ds = ELDataset(split_items.train, train_tfms)
    val_ds = ELDataset(split_items.val, val_tfms)
    test_ds = ELDataset(split_items.test, test_tfms)
    train_sampler = build_weighted_sampler(split_items.train)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    split_counts = {
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
        "total": len(all_samples),
    }

    return train_loader, val_loader, test_loader, split_counts


# ---------------------------------------------------------------------------
# Standalone dataset inspection
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: validate the dataset pipeline and print split info."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare CSV-based EL dataset and create DataLoaders "
            "(inspection mode)."
        )
    )
    parser.add_argument("--csv_path", type=str, default="labels.csv")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, split_counts = create_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed,
    )

    sample_images, sample_labels = next(iter(train_loader))
    print("Split counts:", split_counts)
    print("Sample batch image tensor shape:", tuple(sample_images.shape))
    print("Sample batch labels shape:", tuple(sample_labels.shape))


if __name__ == "__main__":
    main()
