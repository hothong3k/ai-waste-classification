"""
download_data.py — Automated Dataset Download & Setup
=======================================================
Downloads the TrashNet dataset from GitHub and organizes it
into the project's expected directory structure.
"""

import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

RAW_DIR = Path("data/raw")
TEMP_DIR = Path("data/_temp_download")

# TrashNet dataset (hosted on GitHub)
TRASHNET_URL = "https://github.com/garythung/trashnet/archive/refs/heads/master.zip"
TRASHNET_ZIP = TEMP_DIR / "trashnet.zip"

# Mapping from TrashNet classes to our project classes
# TrashNet has: cardboard, glass, metal, paper, plastic, trash
# Our project:  plastic, paper, metal, glass, organic, styrofoam
CLASS_MAPPING = {
    "plastic": "plastic",
    "paper": "paper",
    "metal": "metal",
    "glass": "glass",
    "cardboard": "paper",       # Merge cardboard into paper
    "trash": "organic",         # Map 'trash' to organic (closest match)
}


# ============================================================
# Download Utilities
# ============================================================

def download_with_progress(url, dest):
    """Download a file with progress bar."""
    print(f"  Downloading from: {url}")
    print(f"  Saving to: {dest}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"\r  [{bar}] {pct:5.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    print()  # newline after progress bar


# ============================================================
# Main Download & Setup Pipeline
# ============================================================

def main():
    print("=" * 60)
    print("VN WASTE CLASSIFICATION - DATA DOWNLOAD & SETUP")
    print("=" * 60)

    # Step 1: Create directories
    print("\n[Step 1] Creating directories...")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for class_name in ["plastic", "paper", "metal", "glass", "organic", "styrofoam"]:
        (RAW_DIR / class_name).mkdir(parents=True, exist_ok=True)
    print("  DONE: All directories created")

    # Step 2: Download TrashNet
    print("\n[Step 2] Downloading TrashNet dataset from GitHub...")
    if TRASHNET_ZIP.exists():
        print("  SKIP: ZIP already exists")
    else:
        try:
            download_with_progress(TRASHNET_URL, str(TRASHNET_ZIP))
            print("  DONE: Download complete!")
        except Exception as e:
            print(f"  FAIL: Download failed: {e}")
            sys.exit(1)

    # Step 3: Extract outer ZIP (GitHub repo)
    print("\n[Step 3] Extracting GitHub repo ZIP...")
    extract_dir = TEMP_DIR / "trashnet-master"
    if extract_dir.exists():
        print("  SKIP: Already extracted")
    else:
        with zipfile.ZipFile(TRASHNET_ZIP, "r") as zf:
            zf.extractall(TEMP_DIR)
        print("  DONE: Repo extracted!")

    # Step 4: Extract inner dataset ZIP (dataset-resized.zip inside the repo)
    print("\n[Step 4] Extracting dataset-resized.zip...")
    inner_zip = extract_dir / "data" / "dataset-resized.zip"
    dataset_dir = TEMP_DIR / "dataset-resized"

    if not inner_zip.exists():
        print(f"  ERROR: Could not find {inner_zip}")
        # List what's available
        data_dir = extract_dir / "data"
        if data_dir.exists():
            print(f"  Contents of {data_dir}:")
            for item in data_dir.iterdir():
                print(f"    {item.name} ({'DIR' if item.is_dir() else f'FILE {item.stat().st_size / 1024 / 1024:.1f}MB'})")
        sys.exit(1)

    if dataset_dir.exists():
        print("  SKIP: Already extracted")
    else:
        print(f"  Extracting {inner_zip.name} ({inner_zip.stat().st_size / 1024 / 1024:.1f} MB)...")
        with zipfile.ZipFile(inner_zip, "r") as zf:
            zf.extractall(TEMP_DIR)
        print("  DONE: Dataset extracted!")

    # Step 5: Find actual image directories
    print("\n[Step 5] Locating image class directories...")
    # The extracted structure might be dataset-resized/<class>/ or dataset-resized/dataset-resized/<class>/
    candidate_dirs = [
        dataset_dir,
        TEMP_DIR / "dataset-resized" / "dataset-resized",
    ]
    
    actual_dir = None
    for candidate in candidate_dirs:
        if candidate.exists():
            # Check if it has subdirectories with images
            subdirs = [d for d in candidate.iterdir() if d.is_dir()]
            if subdirs:
                actual_dir = candidate
                break

    if actual_dir is None:
        print("  ERROR: Could not locate image directories!")
        print(f"  Searching in {TEMP_DIR}...")
        for root, dirs, files in os.walk(str(TEMP_DIR)):
            for d in dirs:
                full = Path(root) / d
                jpgs = list(full.glob("*.jpg"))
                if jpgs:
                    print(f"    Found {len(jpgs)} JPGs in: {full}")
        sys.exit(1)

    print(f"  FOUND: {actual_dir}")
    for d in sorted(actual_dir.iterdir()):
        if d.is_dir():
            count = len(list(d.glob("*.*")))
            print(f"    {d.name}: {count} files")

    # Step 6: Copy and organize images
    print("\n[Step 6] Organizing images into project structure...")
    total_copied = 0
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for src_class_dir in sorted(actual_dir.iterdir()):
        if not src_class_dir.is_dir():
            continue

        src_class_name = src_class_dir.name.lower()
        if src_class_name not in CLASS_MAPPING:
            print(f"  WARNING: Unknown class '{src_class_name}', skipping")
            continue

        target_class = CLASS_MAPPING[src_class_name]
        target_dir = RAW_DIR / target_class

        count = 0
        for img_file in sorted(src_class_dir.iterdir()):
            if img_file.suffix.lower() not in image_extensions:
                continue

            # Generate unique filename to avoid conflicts when merging classes
            dst_name = f"{src_class_name}_{img_file.name}"
            dst_path = target_dir / dst_name

            if not dst_path.exists():
                shutil.copy2(img_file, dst_path)
                count += 1
                total_copied += 1

        print(f"  {src_class_name:>12s} -> {target_class:<12s}: {count} images")

    print(f"\n  Total images copied: {total_copied}")

    # Step 7: Print final class distribution
    print("\n[Step 7] Final class distribution in data/raw/:")
    print("-" * 35)
    for class_dir in sorted(RAW_DIR.iterdir()):
        if class_dir.is_dir() and not class_dir.name.startswith((".", "_")):
            img_count = sum(
                1 for f in class_dir.iterdir()
                if f.suffix.lower() in image_extensions
            )
            bar = "#" * (img_count // 20) if img_count > 0 else ""
            print(f"  {class_dir.name:<12s}: {img_count:>5d} images  {bar}")
    print("-" * 35)

    # Step 8: Cleanup temp files
    print("\n[Step 8] Cleaning up temporary files...")
    try:
        shutil.rmtree(TEMP_DIR)
        print("  DONE: Temporary files removed")
    except Exception as e:
        print(f"  WARNING: Could not remove temp dir: {e}")
        print(f"  You can manually delete: {TEMP_DIR}")

    print("\n" + "=" * 60)
    print("DATA SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Clean data:   python src/dataset.py --clean")
    print("  2. Split data:   python src/dataset.py --split")
    print("  3. Train model:  python src/train.py --model baseline")
    print()
    print("NOTE: 'styrofoam' class has 0 images from TrashNet.")
    print("  You will need to collect local photos for this class.")
    print("  (This is planned for Week 2 per your project document)")


if __name__ == "__main__":
    main()
