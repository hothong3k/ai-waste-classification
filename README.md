# VN Smart Waste Classification

🗑️ **Smart Waste Classification System in Vietnam** — An end-to-end deep learning pipeline for automatically classifying common municipal waste categories from images.

## 🎯 Project Overview

This project develops a complete AI pipeline (`Data → Model → Output`) capable of classifying Vietnamese domestic waste into 6 categories:

| Category | Examples |
|---|---|
| **Plastic** | Bottles, bags, containers |
| **Paper** | Cardboard, newspapers, paper bags |
| **Metal** | Aluminum cans, tin cans |
| **Glass** | Bottles, jars |
| **Organic** | Food scraps, fruit peels |
| **Styrofoam** | Lunch boxes, packaging |

## 🏗️ Repository Structure

```
vn-smart-waste-classification/
├── data/                    # Data management
│   ├── raw/                 # Original images
│   ├── processed/           # Cleaned images
│   └── splits/              # Train/val/test manifests
├── notebooks/               # Jupyter notebooks
├── src/                     # Source code modules
├── app/                     # Demo web interface
├── models/                  # Model checkpoints
├── reports/                 # Slides & final report
├── config.yaml              # Hyperparameters & seeds
├── requirements.txt         # Dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repo-url>
cd vn-smart-waste-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data
```bash
python src/dataset.py --download
```

### 3. Train Baseline Model
```bash
python src/train.py --config config.yaml --model baseline
```

### 4. Train Transfer Learning Model
```bash
python src/train.py --config config.yaml --model mobilenetv2
```

### 5. Evaluate
```bash
python src/evaluate.py --model-path models/best_model.pth
```

### 6. Run Demo
```bash
python app/demo.py
```

## 📊 Model Performance

| Model | Accuracy | F1-Score | AUC |
|---|---|---|---|
| Baseline CNN | TBD | TBD | TBD |
| MobileNetV2 (Transfer Learning) | TBD | TBD | TBD |

## 📅 Project Timeline

| Week | Milestone |
|---|---|
| Week 1 | Foundation & Scope Definition |
| Week 2 | Problem & Data Exploration |
| Week 3 | Preprocessing & Baseline Model |
| Week 4 | Model Development & Evaluation |
| Week 5 | Integration & Final Delivery |

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Deep Learning:** PyTorch
- **Data Processing:** torchvision, Pillow, albumentations
- **Visualization:** matplotlib, seaborn
- **Metrics:** scikit-learn
- **Demo UI:** Gradio
- **Environment:** Google Colab / Kaggle (free GPU)

## 📝 License

This project is for educational purposes as part of the AI Course Project.
