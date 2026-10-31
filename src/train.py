"""
train.py — Training Pipeline
==============================
Handles:
  - Training loop with validation
  - Learning rate scheduling
  - Early stopping & checkpoint saving
  - Backbone unfreezing for transfer learning
  - Logging metrics per epoch
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm

from dataset import load_config, get_dataloaders, WasteDataset
from models import create_model


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"  ✓ Random seed set to {seed}")


# ============================================================
# Training Engine
# ============================================================

class Trainer:
    """
    Complete training engine with validation, scheduling, and checkpointing.
    """

    def __init__(self, config: dict, model_type: str = "baseline"):
        self.config = config
        self.model_type = model_type
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"\n🖥️  Device: {self.device}")

        # Get model-specific config
        if model_type in ("baseline", "baselinecnn"):
            self.model_config = config["baseline"]
        else:
            self.model_config = config["transfer"]

        # Build model
        self.model = create_model(
            model_name=model_type,
            num_classes=config["data"]["num_classes"],
            pretrained=self.model_config.get("pretrained", False),
            freeze_backbone=self.model_config.get("freeze_backbone", False),
        ).to(self.device)

        # Build dataloaders
        print("\n📂 Loading data...")
        self.loaders = get_dataloaders(config)

        # Loss function (with optional class weights for imbalance)
        train_dataset = self.loaders["train"].dataset
        class_weights = train_dataset.get_class_weights().to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  ✓ Class weights: {class_weights.cpu().numpy().round(3)}")

        # Optimizer
        lr = self.model_config["learning_rate"]
        wd = self.model_config["weight_decay"]
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=wd,
        )
        print(f"  ✓ Optimizer: Adam (lr={lr}, wd={wd})")

        # Scheduler
        scheduler_type = self.model_config.get("scheduler", "step_lr")
        epochs = self.model_config["epochs"]
        if scheduler_type == "step_lr":
            step = self.model_config.get("scheduler_step_size", 10)
            gamma = self.model_config.get("scheduler_gamma", 0.1)
            self.scheduler = StepLR(self.optimizer, step_size=step, gamma=gamma)
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        else:
            self.scheduler = None
        print(f"  ✓ Scheduler: {scheduler_type}")

        # Training state
        self.epochs = epochs
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_f1": [],
        }

        # Checkpoint config
        self.save_dir = Path(config["checkpoint"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Transfer learning: unfreeze schedule
        self.unfreeze_after = self.model_config.get("unfreeze_after_epochs", None)

    def train_one_epoch(self, epoch: int) -> dict:
        """Train for one epoch. Returns dict with loss and accuracy."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.loaders["train"],
            desc=f"  Epoch {epoch:>3d} [Train]",
            leave=False
        )

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Stats
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.4f}"
            )

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return {"loss": epoch_loss, "acc": epoch_acc}

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate on validation set. Returns dict with loss, accuracy, F1."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in self.loaders["val"]:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        total = len(all_labels)
        epoch_loss = running_loss / total

        # Accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        epoch_acc = (all_preds == all_labels).mean()

        # F1 Score (macro)
        from sklearn.metrics import f1_score
        epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return {"loss": epoch_loss, "acc": epoch_acc, "f1": epoch_f1}

    def save_checkpoint(self, epoch: int, val_metrics: dict):
        """Save model checkpoint if it's the best so far."""
        if val_metrics["f1"] > self.best_val_f1:
            self.best_val_f1 = val_metrics["f1"]
            self.best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_f1": val_metrics["f1"],
                "val_acc": val_metrics["acc"],
                "config": self.config,
                "model_type": self.model_type,
                "class_names": self.config["data"]["class_names"],
            }

            save_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, save_path)
            print(f"  💾 Best model saved (F1={val_metrics['f1']:.4f}) → {save_path}")

    def train(self):
        """Full training loop."""
        print("\n" + "=" * 60)
        print(f"🚀 TRAINING: {self.model_type.upper()}")
        print(f"   Epochs: {self.epochs} | Device: {self.device}")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            # Check if we need to unfreeze backbone (Transfer Learning)
            if (self.unfreeze_after and epoch == self.unfreeze_after + 1
                    and hasattr(self.model, "unfreeze_backbone")):
                print(f"\n  🔓 Unfreezing backbone at epoch {epoch}...")
                self.model.unfreeze_backbone()
                # Reset optimizer to include backbone params
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.model_config["learning_rate"] * 0.1,  # Lower LR for fine-tuning
                    weight_decay=self.model_config["weight_decay"],
                )

            # Train & Validate
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["acc"])
            self.history["val_f1"].append(val_metrics["f1"])

            # Print epoch summary
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:>3d}/{self.epochs} │ "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.4f} │ "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} "
                f"F1: {val_metrics['f1']:.4f} │ LR: {lr:.6f}"
            )

            # Save best checkpoint
            self.save_checkpoint(epoch, val_metrics)

        elapsed = time.time() - start_time
        print(f"\n✅ Training complete in {elapsed / 60:.1f} minutes")
        print(f"   Best Val F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")

        # Save training history
        import json
        history_path = self.save_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"   History saved → {history_path}")

        return self.history


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="VN Waste Classification — Model Training"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model", type=str, default="baseline",
        choices=["baseline", "mobilenetv2", "resnet50"],
        help="Model architecture to train"
    )

    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    trainer = Trainer(config, model_type=args.model)
    history = trainer.train()


if __name__ == "__main__":
    main()
