"""
models.py — Neural Network Architecture Definitions
=====================================================
Contains:
  - BaselineCNN: Simple custom CNN for initial benchmarking (Week 3)
  - MobileNetV2Transfer: MobileNetV2 with Transfer Learning (Week 4)
  - ResNet50Transfer: ResNet50 with Transfer Learning (Week 4)
  - Model factory function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# Baseline CNN (Week 3)
# ============================================================

class BaselineCNN(nn.Module):
    """
    A simple 4-layer CNN for baseline benchmarking.

    Architecture:
        Conv2D(3→32) → BN → ReLU → MaxPool →
        Conv2D(32→64) → BN → ReLU → MaxPool →
        Conv2D(64→128) → BN → ReLU → MaxPool →
        Conv2D(128→256) → BN → ReLU → AdaptiveAvgPool →
        Flatten → Dropout(0.5) → FC(256→128) → ReLU → Dropout(0.3) → FC(128→num_classes)

    Note: This architecture matches the Colab training notebook to ensure
    checkpoint compatibility when loading models trained on Colab.
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224x224x3 → 112x112x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 112x112x32 → 56x56x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 56x56x64 → 28x28x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 28x28x128 → 1x1x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
        )

        # Classifier: matches Colab notebook layer ordering
        self.classifier = nn.Sequential(
            nn.Flatten(),       # idx 0
            nn.Dropout(0.5),    # idx 1
            nn.Linear(256, 128),# idx 2
            nn.ReLU(inplace=True),  # idx 3
            nn.Dropout(0.3),    # idx 4
            nn.Linear(128, num_classes),  # idx 5
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# Transfer Learning: MobileNetV2 (Week 4)
# ============================================================

class MobileNetV2Transfer(nn.Module):
    """
    MobileNetV2 with Transfer Learning.
    
    - Loads ImageNet pre-trained weights
    - Replaces final classifier head for waste classification
    - Supports freezing/unfreezing backbone for fine-tuning
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()

        # Load pre-trained MobileNetV2
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all feature extraction layers — only train classifier."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("  ✓ Backbone frozen (only classifier is trainable)")

    def unfreeze_backbone(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        print("  ✓ Backbone unfrozen (all layers trainable)")

    def get_trainable_params(self) -> int:
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Transfer Learning: ResNet50 (Week 4 — alternative)
# ============================================================

class ResNet50Transfer(nn.Module):
    """
    ResNet50 with Transfer Learning.

    - Loads ImageNet pre-trained weights
    - Replaces final FC layer for waste classification
    - Supports freezing/unfreezing backbone
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all layers except the final FC."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        print("  ✓ Backbone frozen (only FC is trainable)")

    def unfreeze_backbone(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("  ✓ Backbone unfrozen (all layers trainable)")

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Model Factory
# ============================================================

MODEL_REGISTRY = {
    "baseline": BaselineCNN,
    "baselinecnn": BaselineCNN,
    "mobilenetv2": MobileNetV2Transfer,
    "mobilenet_v2": MobileNetV2Transfer,
    "resnet50": ResNet50Transfer,
    "resnet_50": ResNet50Transfer,
}


def create_model(
    model_name: str,
    num_classes: int = 6,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Factory function to create a model by name.

    Args:
        model_name: One of 'baseline', 'mobilenetv2', 'resnet50'
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pre-trained weights (transfer learning only)
        freeze_backbone: Whether to freeze feature extraction layers

    Returns:
        nn.Module: The constructed model
    """
    key = model_name.lower().replace(" ", "")
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    ModelClass = MODEL_REGISTRY[key]

    if key in ("baseline", "baselinecnn"):
        model = ModelClass(num_classes=num_classes)
    else:
        model = ModelClass(num_classes=num_classes, pretrained=pretrained)
        if freeze_backbone:
            model.freeze_backbone()

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📐 Model: {model_name}")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters:    {total_params - trainable_params:,}")

    return model


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing model architectures...")
    print("=" * 60)

    dummy_input = torch.randn(2, 3, 224, 224)

    for name in ["baseline", "mobilenetv2", "resnet50"]:
        model = create_model(name, num_classes=6, pretrained=False)
        output = model(dummy_input)
        print(f"   Output shape: {output.shape}")
        assert output.shape == (2, 6), f"Unexpected output shape for {name}"
        print(f"   ✓ {name} — PASSED\n")

    print("All models passed! ✓")
