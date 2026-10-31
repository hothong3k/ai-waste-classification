# Data Directory

## Overview
This directory contains all data assets for the VN Smart Waste Classification project.

## Structure

```
data/
├── raw/           # Original, unmodified images
├── processed/     # Cleaned, resized, and augmented images
├── splits/        # Train/val/test split manifests (CSV files)
└── README.md      # This file
```

## Waste Categories (6 Classes)

| Label | Category | Description |
|---|---|---|
| 0 | Plastic | Bottles, bags, rigid containers |
| 1 | Paper | Cardboard, newspapers, paper bags |
| 2 | Metal | Aluminum cans, tin cans, metal lids |
| 3 | Glass | Glass bottles, jars |
| 4 | Organic | Food scraps, fruit peels, leaves |
| 5 | Styrofoam | Lunch boxes, packaging foam |

## Data Sources

### Primary Sources (~80% of data volume)
- **TrashNet** (Stanford): ~2,527 images across 6 categories — [GitHub](https://github.com/garythung/trashnet)
- **Garbage Classification (Kaggle)**: ~15,000+ images — [Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

### Local Supplement (~20% of data volume)
- ~200–300 photographs of Vietnamese domestic waste captured in real-world conditions
- Categories: soiled styrofoam boxes, tied plastic bags, crushed cans on pavement, etc.

## Download Instructions

```bash
# Automated download (coming in Week 2)
python src/dataset.py --download

# Manual download:
# 1. Download TrashNet from the GitHub link above
# 2. Download Garbage Classification from Kaggle
# 3. Place all images into data/raw/<category_name>/
```

## Important Notes
- **DO NOT commit raw images to Git** — use Google Drive or Kaggle for hosting
- All processed images should be resized to 224×224 pixels
- Random seed for splits is locked in `config.yaml` for reproducibility
