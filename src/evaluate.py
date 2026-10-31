"""
evaluate.py — Model Evaluation & Comparison
=============================================
Handles:
  - Loading trained checkpoints
  - Computing metrics: Accuracy, F1 (macro/per-class), AUC
  - Generating confusion matrix visualization
  - Comparing Baseline vs Transfer Learning models
  - Exporting evaluation report
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from dataset import load_config, get_dataloaders, get_eval_transforms
from models import create_model


# ============================================================
# Model Loading
# ============================================================

def load_checkpoint(model_path: str, device: torch.device) -> dict:
    """
    Load a saved model checkpoint.

    Returns:
        dict with 'model', 'config', 'class_names', 'metadata'
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_type = checkpoint.get("model_type", "baseline")
    config = checkpoint.get("config", {})
    class_names = checkpoint.get("class_names", [])
    num_classes = len(class_names) if class_names else 6

    model = create_model(
        model_name=model_type,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return {
        "model": model,
        "config": config,
        "class_names": class_names,
        "metadata": {
            "epoch": checkpoint.get("epoch", "?"),
            "val_f1": checkpoint.get("val_f1", "?"),
            "val_acc": checkpoint.get("val_acc", "?"),
            "model_type": model_type,
        }
    }


# ============================================================
# Evaluation on Test Set
# ============================================================

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    class_names: list,
    device: torch.device,
) -> dict:
    """
    Run full evaluation on a dataset split.

    Returns:
        dict with metrics: accuracy, f1_macro, f1_per_class, auc, 
        confusion_matrix, classification_report, predictions, labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)

    # F1 Score
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # AUC (One-vs-Rest)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0  # Not enough classes in split

    # Classification Report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_per_class": {name: float(f1) for name, f1 in zip(class_names, f1_per_class)},
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist(),
        "probabilities": all_probs.tolist(),
    }


# ============================================================
# Visualization
# ============================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str,
    title: str = "Confusion Matrix",
):
    """Generate and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Count"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Confusion matrix saved → {save_path}")


def plot_training_history(history: dict, save_path: str):
    """Plot training/validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    axes[0].set_title("Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=3)
    axes[1].set_title("Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 Score
    axes[2].plot(epochs, history["val_f1"], "g-o", label="Val F1 (Macro)", markersize=3)
    axes[2].set_title("Validation F1 Score", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 Training history plot saved → {save_path}")


def plot_model_comparison(results: dict, save_path: str):
    """
    Bar chart comparing metrics across multiple models.
    
    Args:
        results: dict mapping model_name -> evaluation metrics dict
    """
    model_names = list(results.keys())
    metrics = ["accuracy", "f1_macro", "auc"]
    metric_labels = ["Accuracy", "F1 (Macro)", "AUC"]

    x = np.arange(len(metrics))
    width = 0.35
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_name in enumerate(model_names):
        values = [results[model_name].get(m, 0) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Model comparison saved → {save_path}")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="VN Waste Classification — Model Evaluation"
    )
    parser.add_argument(
        "--model-path", type=str, default="models/best_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["val", "test"],
        help="Which data split to evaluate on"
    )
    parser.add_argument(
        "--plot-history", action="store_true",
        help="Also plot training history curves"
    )

    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("📊 MODEL EVALUATION")
    print("=" * 60)

    # Load model
    checkpoint_data = load_checkpoint(args.model_path, device)
    model = checkpoint_data["model"]
    class_names = checkpoint_data["class_names"] or config["data"]["class_names"]
    metadata = checkpoint_data["metadata"]

    print(f"\n  Model: {metadata['model_type']}")
    print(f"  Trained epoch: {metadata['epoch']}")
    print(f"  Val F1 (at save): {metadata['val_f1']}")

    # Load test data
    print(f"\n  Loading {args.split} data...")
    loaders = get_dataloaders(config)

    if args.split not in loaders:
        print(f"  [ERROR] Split '{args.split}' not found. Run data splitting first.")
        return

    # Evaluate
    print(f"\n  Evaluating on {args.split} set...")
    results = evaluate_model(model, loaders[args.split], class_names, device)

    # Print results
    print(f"\n  {'=' * 40}")
    print(f"  📊 Results on {args.split.upper()} set:")
    print(f"  {'=' * 40}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  F1 Macro:  {results['f1_macro']:.4f}")
    print(f"  AUC:       {results['auc']:.4f}")
    print(f"\n  Per-class F1:")
    for cls, f1 in results["f1_per_class"].items():
        print(f"    {cls:<12s}: {f1:.4f}")

    # Save plots
    save_dir = Path(config["evaluation"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    cm = np.array(results["confusion_matrix"])
    plot_confusion_matrix(
        cm, class_names,
        save_path=str(save_dir / f"confusion_matrix_{args.split}.png"),
        title=f"Confusion Matrix ({metadata['model_type']} — {args.split})",
    )

    # Plot training history if requested
    if args.plot_history:
        history_path = Path(config["checkpoint"]["save_dir"]) / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            plot_training_history(
                history,
                save_path=str(save_dir / "training_history.png"),
            )
        else:
            print(f"  [WARN] History file not found: {history_path}")

    # Save results to JSON
    report_path = save_dir / f"eval_results_{args.split}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  📄 Results saved → {report_path}")


if __name__ == "__main__":
    main()
