"""
inference.py — End-to-End Inference Pipeline
==============================================
Handles:
  - Single image prediction (Data → Model → Output)
  - Batch prediction on a directory
  - Returns class label + confidence score + sorting guide
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# Ensure src/ is importable regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).parent))

from dataset import load_config, get_eval_transforms
from models import create_model


# ============================================================
# Vietnamese Waste Sorting Guide
# ============================================================

SORTING_GUIDE = {
    "plastic": {
        "bin": "♻️ Thùng TÁI CHẾ (Recyclable)",
        "color": "Xanh lá / Vàng",
        "instructions": "Rửa sạch, bỏ nắp riêng. Chai nhựa PET, hộp nhựa cứng có thể tái chế.",
        "tips": "Túi ni-lông mỏng thường KHÔNG tái chế được — bỏ vào thùng rác thường.",
    },
    "paper": {
        "bin": "♻️ Thùng TÁI CHẾ (Recyclable)",
        "color": "Xanh lá / Vàng",
        "instructions": "Giấy, bìa carton, báo cũ. Gấp gọn trước khi bỏ vào thùng.",
        "tips": "Giấy ướt, dính dầu mỡ KHÔNG tái chế được.",
    },
    "metal": {
        "bin": "♻️ Thùng TÁI CHẾ (Recyclable)",
        "color": "Xanh lá / Vàng",
        "instructions": "Lon nhôm, lon thiếc. Rửa sạch và ép dẹp nếu có thể.",
        "tips": "Lon nước ngọt, lon bia là phế liệu có giá trị cao.",
    },
    "glass": {
        "bin": "♻️ Thùng TÁI CHẾ (Recyclable)",
        "color": "Xanh lá / Vàng",
        "instructions": "Chai lọ thủy tinh. Rửa sạch, bỏ riêng để tránh vỡ.",
        "tips": "Gương, kính cửa sổ, bóng đèn KHÔNG bỏ chung với thủy tinh tái chế.",
    },
    "organic": {
        "bin": "🍂 Thùng HỮU CƠ (Organic/Compost)",
        "color": "Xanh lá đậm",
        "instructions": "Thức ăn thừa, vỏ trái cây, rau củ. Có thể ủ phân compost.",
        "tips": "Không bỏ lẫn túi ni-lông vào rác hữu cơ.",
    },
    "styrofoam": {
        "bin": "🗑️ Thùng RÁC THƯỜNG (General Waste)",
        "color": "Xám / Đen",
        "instructions": "Hộp xốp, khay xốp đựng thức ăn. Hiện chưa có hệ thống tái chế xốp phổ biến tại VN.",
        "tips": "Hạn chế sử dụng hộp xốp — thay bằng hộp giấy hoặc hộp nhựa tái sử dụng.",
    },
}


# ============================================================
# Inference Engine
# ============================================================

class WasteClassifier:
    """
    End-to-end waste classification inference engine.
    
    Usage:
        classifier = WasteClassifier("models/best_model.pth")
        result = classifier.predict("path/to/waste_image.jpg")
        print(result)
    """

    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device(
            self.config.get("app", {}).get("device", "cpu")
        )

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.class_names = checkpoint.get(
            "class_names", self.config["data"]["class_names"]
        )
        model_type = checkpoint.get("model_type", "baseline")

        # Rebuild model
        self.model = create_model(
            model_name=model_type,
            num_classes=len(self.class_names),
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Transforms
        self.transform = get_eval_transforms(self.config)

        print(f"✓ Loaded {model_type} model ({len(self.class_names)} classes)")
        print(f"  Classes: {self.class_names}")
        print(f"  Device: {self.device}")

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        Classify a single waste image.

        Args:
            image_path: Path to the image file

        Returns:
            dict with:
                - class_name: predicted waste category
                - confidence: probability score (%)
                - all_scores: dict of all class probabilities
                - sorting_guide: Vietnamese sorting instructions
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Forward pass
        outputs = self.model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]

        # Get prediction
        confidence, pred_idx = torch.max(probs, 0)
        pred_class = self.class_names[pred_idx.item()]
        confidence_pct = confidence.item() * 100

        # All class scores
        all_scores = {
            name: round(probs[i].item() * 100, 2)
            for i, name in enumerate(self.class_names)
        }

        # Sorting guide
        guide = SORTING_GUIDE.get(pred_class, {})

        return {
            "class_name": pred_class,
            "confidence": round(confidence_pct, 2),
            "all_scores": all_scores,
            "sorting_guide": guide,
        }

    def predict_batch(self, image_dir: str) -> list:
        """Classify all images in a directory."""
        results = []
        img_dir = Path(image_dir)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() in extensions:
                result = self.predict(str(img_path))
                result["file"] = img_path.name
                results.append(result)
                print(
                    f"  {img_path.name}: {result['class_name']} "
                    f"({result['confidence']:.1f}%)"
                )

        return results


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="VN Waste Classification — Inference"
    )
    parser.add_argument(
        "image", type=str,
        help="Path to image file or directory of images"
    )
    parser.add_argument(
        "--model", type=str, default="models/best_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()

    classifier = WasteClassifier(args.model, args.config)

    if os.path.isdir(args.image):
        # Batch prediction
        print(f"\n📂 Batch prediction on: {args.image}")
        results = classifier.predict_batch(args.image)
        print(f"\n✅ Processed {len(results)} images")
    else:
        # Single prediction
        print(f"\n🖼️  Predicting: {args.image}")
        result = classifier.predict(args.image)

        print(f"\n{'=' * 50}")
        print(f"  🏷️  Category:    {result['class_name'].upper()}")
        print(f"  📊 Confidence:  {result['confidence']:.1f}%")
        print(f"{'=' * 50}")

        guide = result["sorting_guide"]
        if guide:
            print(f"\n  🗑️  {guide['bin']}")
            print(f"  🎨 Màu thùng: {guide['color']}")
            print(f"  📋 {guide['instructions']}")
            print(f"  💡 {guide['tips']}")

        print(f"\n  All scores:")
        for cls, score in sorted(
            result["all_scores"].items(), key=lambda x: -x[1]
        ):
            bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
            print(f"    {cls:<12s} {bar} {score:>6.2f}%")


if __name__ == "__main__":
    main()
