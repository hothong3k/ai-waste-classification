"""
demo.py — Gradio Web Interface for Waste Classification
=========================================================
A beautiful, user-friendly demo for classifying waste images.
Supports:
  - Upload or capture from camera
  - Real-time classification with confidence scores
  - Vietnamese sorting guide
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
from src.inference import WasteClassifier, SORTING_GUIDE


# ============================================================
# Initialize Classifier
# ============================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pth")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")

try:
    classifier = WasteClassifier(MODEL_PATH, CONFIG_PATH)
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️  Could not load model: {e}")
    print("   Demo will run in placeholder mode.")
    MODEL_LOADED = False


# ============================================================
# Prediction Function
# ============================================================

def classify_waste(image):
    """
    Classify an uploaded waste image.

    Returns:
        - label_scores: dict for Gradio Label component
        - sorting_info: formatted text with Vietnamese sorting guide
    """
    if image is None:
        return {}, "⚠️ Vui lòng tải lên một hình ảnh rác."

    if not MODEL_LOADED:
        return (
            {"plastic": 0.35, "paper": 0.25, "metal": 0.15,
             "glass": 0.10, "organic": 0.10, "styrofoam": 0.05},
            "⚠️ **Chế độ demo** — Model chưa được tải. "
            "Hãy huấn luyện model trước (`python src/train.py`)."
        )

    # Save temp image for inference
    temp_path = "temp_upload.jpg"
    image.save(temp_path)

    try:
        result = classifier.predict(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Format scores for Label component
    label_scores = {k: v / 100 for k, v in result["all_scores"].items()}

    # Format sorting guide
    guide = result["sorting_guide"]
    pred = result["class_name"]
    conf = result["confidence"]

    sorting_text = f"""
## 🏷️ Kết quả phân loại: **{pred.upper()}**
### 📊 Độ tin cậy: **{conf:.1f}%**

---

### 🗑️ Hướng dẫn phân loại rác

| Mục | Chi tiết |
|---|---|
| **Thùng rác** | {guide.get('bin', 'N/A')} |
| **Màu thùng** | {guide.get('color', 'N/A')} |
| **Hướng dẫn** | {guide.get('instructions', 'N/A')} |
| **Mẹo** | {guide.get('tips', 'N/A')} |
"""

    return label_scores, sorting_text


# ============================================================
# Gradio Interface
# ============================================================

# Custom CSS for premium look
custom_css = """
.gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #00b894, #00cec9) !important;
    border: none !important;
}
.gr-button-primary:hover {
    background: linear-gradient(135deg, #00cec9, #0984e3) !important;
}
"""

# Build the interface
with gr.Blocks(
    title="🗑️ VN Smart Waste Classifier",
) as demo:

    gr.Markdown("""
    # 🗑️ Phân Loại Rác Thông Minh — Việt Nam
    ### Smart Waste Classification System

    Tải lên hình ảnh rác thải và hệ thống AI sẽ tự động phân loại 
    thành **6 loại**: Nhựa, Giấy, Kim Loại, Thủy Tinh, Hữu Cơ, Xốp.
    
    Kèm theo hướng dẫn phân loại rác phù hợp với quy định tại Việt Nam.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="📸 Tải ảnh rác lên",
                sources=["upload", "clipboard"],
            )
            classify_btn = gr.Button(
                "🔍 Phân Loại",
                variant="primary",
                size="lg",
            )

            gr.Markdown("""
            ### 📋 Danh mục rác hỗ trợ:
            | # | Loại | Ví dụ |
            |---|---|---|
            | 1 | ♻️ Nhựa (Plastic) | Chai, túi, hộp nhựa |
            | 2 | 📄 Giấy (Paper) | Bìa carton, báo |
            | 3 | 🥫 Kim loại (Metal) | Lon nhôm, lon thiếc |
            | 4 | 🫙 Thủy tinh (Glass) | Chai, lọ thủy tinh |
            | 5 | 🍂 Hữu cơ (Organic) | Thức ăn thừa, vỏ trái cây |
            | 6 | 📦 Xốp (Styrofoam) | Hộp xốp, khay xốp |
            """)

        with gr.Column(scale=1):
            output_label = gr.Label(
                label="📊 Xác suất từng loại",
                num_top_classes=6,
            )
            output_guide = gr.Markdown(
                label="🗑️ Hướng dẫn phân loại",
                value="*Tải ảnh lên để nhận kết quả phân loại.*",
            )

    classify_btn.click(
        fn=classify_waste,
        inputs=[input_image],
        outputs=[output_label, output_guide],
    )

    gr.Markdown("""
    ---
    **Dự án AI Course** | Smart Waste Classification in Vietnam | 
    Sử dụng CNN + Transfer Learning (MobileNetV2)
    """)


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue"),
        css=custom_css,
    )
