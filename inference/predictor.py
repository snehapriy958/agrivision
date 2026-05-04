"""
inference/predictor.py — Single-Image Inference for CNN Image Classifier
Dependencies : torch, torchvision, Pillow
Usage        : from inference.predictor import predict
               results = predict("path/to/image.jpg")
"""

import os

import urllib.request

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from models.cnn_baseline import CNNModel
from inference.preprocess import PlantDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "checkpoint_path" : "models/best_model.pth",
    "data_root"       : "data/raw",
    "num_classes"     : 9,
    "top_k"           : 3,
    "image_size"      : 224,
    "mean"            : [0.485, 0.456, 0.406],
    "std"             : [0.229, 0.224, 0.225],
}

MODEL, CLASS_NAMES = None, None

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(cfg: dict = CONFIG):
    global MODEL, CLASS_NAMES

    if MODEL is None:
        import gdown

        # Absolute path (robust for Streamlit Cloud)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH = os.path.join(BASE_DIR, cfg["checkpoint_path"])

        # Google Drive file ID (NOT full link)
        MODEL_URL = "https://drive.google.com/uc?id=1OJ_0ckqRI4xGp-MpYa3FgjhEntL_pHe4"

        # Download if not exists
        if not os.path.exists(MODEL_PATH):
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            print("Downloading model...")
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

        # Load model (PyTorch 2.6 fix)
        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )

        # Initialize model
        MODEL = CNNModel(num_classes=cfg["num_classes"])

        # Flexible loading
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            MODEL.load_state_dict(checkpoint["state_dict"])
        else:
            MODEL.load_state_dict(checkpoint)

        MODEL.eval()

        # Load class names once
        if CLASS_NAMES is None:
            CLASS_NAMES = PlantDataset(root_dir=cfg["data_root"]).class_names

    return MODEL, CLASS_NAMES

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str, cfg: dict = CONFIG) -> torch.Tensor:
    """Open, transform, and batch a single image."""
    transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)          # (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(image_path: str, cfg: dict = CONFIG):
    model, class_names = load_model(cfg)
    tensor = preprocess_image(image_path, cfg)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)

    top_probs, top_indices = torch.topk(probs, k=cfg["top_k"], dim=1)

    return [
        {
            "label": class_names[idx.item()],
            "confidence": round(prob.item(), 4),
        }
        for prob, idx in zip(top_probs[0], top_indices[0])
    ]

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python inference/predictor.py <image_path>")
        sys.exit(1)

    results = predict(sys.argv[1])

    print("\nTop-3 Predictions")
    print("-" * 35)
    for rank, result in enumerate(results, start=1):
        confidence_pct = result["confidence"] * 100
        print(f"  {rank}. {result['label']:<20} {confidence_pct:>6.2f}%")
    print()