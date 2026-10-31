"""
dataset.py — Data Loading, Cleaning, Splitting & Augmentation
==============================================================
Handles:
  - Downloading datasets from Kaggle/Google Drive
  - Cleaning: remove corrupt/duplicate images
  - Splitting into train/val/test with stratification
  - Custom PyTorch Dataset with augmentation transforms
"""

import os
import sys
import hashlib
import shutil
import argparse
from pathlib import Path
from collections import Counter

import yaml
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ============================================================
# Configuration Loader
# ============================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# ============================================================
# Data Cleaning Utilities
# ============================================================

def is_valid_image(filepath: str) -> bool:
    """Check if a file is a valid, non-corrupt image."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def compute_image_hash(filepath: str) -> str:
    """Compute MD5 hash for duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_dataset(raw_dir: str, processed_dir: str, image_size: int = 224) -> dict:
    """
    Clean raw images:
    1. Remove corrupt files
    2. Remove exact duplicates (by MD5 hash)
    3. Resize to uniform dimensions
    4. Save to processed directory

    Returns:
        dict with cleaning statistics
    """
    stats = {
        "total_found": 0,
        "corrupt_removed": 0,
        "duplicates_removed": 0,
        "valid_saved": 0,
        "class_distribution": {}
    }

    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)

    if not raw_path.exists():
        print(f"[ERROR] Raw data directory not found: {raw_dir}")
        print("Please download data first: python src/dataset.py --download")
        return stats

    seen_hashes = set()

    for class_dir in sorted(raw_path.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue

        class_name = class_dir.name.lower()
        output_class_dir = proc_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        class_count = 0
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() not in image_extensions:
                continue

            stats["total_found"] += 1

            # Check for corrupt images
            if not is_valid_image(str(img_file)):
                stats["corrupt_removed"] += 1
                print(f"  [CORRUPT] Removed: {img_file.name}")
                continue

            # Check for duplicates
            img_hash = compute_image_hash(str(img_file))
            if img_hash in seen_hashes:
                stats["duplicates_removed"] += 1
                print(f"  [DUPLICATE] Skipped: {img_file.name}")
                continue
            seen_hashes.add(img_hash)

            # Resize and save
            try:
                with Image.open(img_file) as img:
                    img = img.convert("RGB")
                    img = img.resize((image_size, image_size), Image.LANCZOS)
                    output_path = output_class_dir / f"{class_name}_{class_count:05d}.jpg"
                    img.save(output_path, "JPEG", quality=95)
                    class_count += 1
                    stats["valid_saved"] += 1
            except Exception as e:
                print(f"  [ERROR] Failed to process {img_file.name}: {e}")
                stats["corrupt_removed"] += 1

        stats["class_distribution"][class_name] = class_count
        print(f"  ✓ {class_name}: {class_count} images saved")

    return stats


# ============================================================
# Data Splitting
# ============================================================

def create_splits(
    processed_dir: str,
    splits_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> dict:
    """
    Create stratified train/val/test splits and save as CSV manifests.

    Returns:
        dict with split counts per class
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    np.random.seed(seed)
    proc_path = Path(processed_dir)
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)

    all_records = []

    for class_dir in sorted(proc_path.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue

        class_name = class_dir.name
        images = sorted([
            str(img) for img in class_dir.iterdir()
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        np.random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for i, img_path in enumerate(images):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            all_records.append({
                "filepath": img_path,
                "class_name": class_name,
                "split": split
            })

    df = pd.DataFrame(all_records)

    # Save split manifests
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        split_df.to_csv(splits_path / f"{split_name}.csv", index=False)
        print(f"  ✓ {split_name}: {len(split_df)} images")

    # Save complete manifest
    df.to_csv(splits_path / "all_splits.csv", index=False)

    # Print distribution summary
    print("\n  Split distribution per class:")
    print(df.groupby(["class_name", "split"]).size().unstack(fill_value=0))

    return {
        "train": len(df[df["split"] == "train"]),
        "val": len(df[df["split"] == "val"]),
        "test": len(df[df["split"] == "test"]),
    }


# ============================================================
# PyTorch Dataset
# ============================================================

class WasteDataset(Dataset):
    """
    PyTorch Dataset for Vietnamese waste classification.

    Args:
        csv_path: Path to split CSV manifest (train.csv, val.csv, test.csv)
        class_names: List of class names in order
        transform: torchvision transforms to apply
    """

    def __init__(self, csv_path: str, class_names: list, transform=None):
        self.df = pd.read_csv(csv_path)
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        class_name = row["class_name"]
        label = self.class_to_idx[class_name]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for handling imbalanced data.
        Useful for weighted CrossEntropyLoss.
        """
        counts = Counter(self.df["class_name"])
        total = sum(counts.values())
        weights = []
        for class_name in self.class_names:
            count = counts.get(class_name, 1)
            weights.append(total / (len(self.class_names) * count))
        return torch.FloatTensor(weights)


# ============================================================
# Transform Builders
# ============================================================

def get_train_transforms(config: dict) -> transforms.Compose:
    """Build training data augmentation pipeline from config."""
    aug_cfg = config["augmentation"]
    img_size = config["data"]["image_size"]

    transform_list = [
        transforms.Resize((img_size, img_size)),
    ]

    if aug_cfg.get("horizontal_flip"):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if aug_cfg.get("vertical_flip"):
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))

    if aug_cfg.get("rotation_degrees"):
        transform_list.append(
            transforms.RandomRotation(degrees=aug_cfg["rotation_degrees"])
        )

    if aug_cfg.get("color_jitter"):
        cj = aug_cfg["color_jitter"]
        transform_list.append(
            transforms.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
                saturation=cj.get("saturation", 0),
                hue=cj.get("hue", 0),
            )
        )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug_cfg["normalize"]["mean"],
            std=aug_cfg["normalize"]["std"],
        ),
    ])

    return transforms.Compose(transform_list)


def get_eval_transforms(config: dict) -> transforms.Compose:
    """Build evaluation/inference transforms (no augmentation)."""
    aug_cfg = config["augmentation"]
    img_size = config["data"]["image_size"]

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug_cfg["normalize"]["mean"],
            std=aug_cfg["normalize"]["std"],
        ),
    ])


# ============================================================
# DataLoader Builder
# ============================================================

def get_dataloaders(config: dict) -> dict:
    """
    Build train, val, test DataLoaders from split CSV manifests.

    Returns:
        dict with keys 'train', 'val', 'test' -> DataLoader instances
    """
    splits_dir = config["data"]["splits_dir"]
    class_names = config["data"]["class_names"]
    batch_size = config.get("baseline", {}).get("batch_size", 32)

    train_transform = get_train_transforms(config)
    eval_transform = get_eval_transforms(config)

    loaders = {}

    for split_name, transform in [
        ("train", train_transform),
        ("val", eval_transform),
        ("test", eval_transform),
    ]:
        csv_path = os.path.join(splits_dir, f"{split_name}.csv")
        if not os.path.exists(csv_path):
            print(f"  [WARN] Split file not found: {csv_path}")
            continue

        dataset = WasteDataset(csv_path, class_names, transform)
        shuffle = (split_name == "train")

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
            drop_last=(split_name == "train"),
        )

        print(f"  ✓ {split_name}: {len(dataset)} samples, "
              f"{len(loaders[split_name])} batches")

    return loaders


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="VN Waste Classification — Dataset Management"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download raw datasets (TrashNet + Garbage Classification)"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Clean raw images: remove corrupt/duplicates, resize"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Create train/val/test splits from processed data"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.download:
        print("\n" + "=" * 60)
        print("📥 DATA DOWNLOAD")
        print("=" * 60)
        print("\nPlease download datasets manually:")
        print("  1. TrashNet: https://github.com/garythung/trashnet")
        print("  2. Garbage Classification: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification")
        print(f"\nPlace images into: {config['data']['raw_dir']}/<class_name>/")
        print("  Example: data/raw/plastic/, data/raw/paper/, etc.")
        print("\nExpected class folders:", config["data"]["class_names"])

    if args.clean:
        print("\n" + "=" * 60)
        print("🧹 DATA CLEANING")
        print("=" * 60)
        stats = clean_dataset(
            raw_dir=config["data"]["raw_dir"],
            processed_dir=config["data"]["processed_dir"],
            image_size=config["data"]["image_size"],
        )
        print(f"\n📊 Cleaning Summary:")
        print(f"  Total found:       {stats['total_found']}")
        print(f"  Corrupt removed:   {stats['corrupt_removed']}")
        print(f"  Duplicates removed:{stats['duplicates_removed']}")
        print(f"  Valid saved:       {stats['valid_saved']}")
        print(f"  Class distribution:{stats['class_distribution']}")

    if args.split:
        print("\n" + "=" * 60)
        print("✂️  DATA SPLITTING")
        print("=" * 60)
        split_stats = create_splits(
            processed_dir=config["data"]["processed_dir"],
            splits_dir=config["data"]["splits_dir"],
            train_ratio=config["split"]["train"],
            val_ratio=config["split"]["val"],
            test_ratio=config["split"]["test"],
            seed=config["seed"],
        )
        print(f"\n📊 Split Summary: {split_stats}")

    if not any([args.download, args.clean, args.split]):
        parser.print_help()


if __name__ == "__main__":
    main()
