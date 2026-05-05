import os
import sys
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import Tuple, List, Dict, Union, Optional

import gdown

# ---------------------------------------------------------------------
# MODEL DOWNLOAD CONFIG
# ---------------------------------------------------------------------

MODEL_PATH = "models/best_model.pth"
MODEL_URL = "https://drive.google.com/uc?id=1JKlHmyCNOrVSIlfyHtSADj2u2BnotDFv"

# Download model if missing
if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    print("Downloading model...")
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    except Exception as e:
        raise RuntimeError("Model download failed") from e

# Final check
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file missing after download")

# ---------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_baseline import CNNModel
from inference.preprocess import PlantDataset
from utils.gradcam import GradCAM

# ---------------------------------------------------------------------

CONFIG = {
    "checkpoint_path": MODEL_PATH,
    "data_root": "data/raw",
    "num_classes": 9,
    "top_k": 3,
    "image_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

_MODEL: Optional[CNNModel] = None
_CLASS_NAMES: Optional[List[str]] = None

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------

def load_model(cfg: dict = CONFIG) -> Tuple[CNNModel, List[str]]:
    global _MODEL, _CLASS_NAMES

    if _MODEL is not None:
        return _MODEL, _CLASS_NAMES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(cfg["checkpoint_path"], map_location=device)

    model = CNNModel(num_classes=cfg["num_classes"])

    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # ✅ CLEAN LOAD (NO FILTERING)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    _MODEL = model
    _CLASS_NAMES = [
        "Corn Cercospora Leaf Spot",
        "Corn Common Rust",
        "Corn Healthy",
        "Potato Early Blight",
        "Potato Healthy",
        "Potato Late Blight",
        "Tomato Early Blight",
        "Tomato Healthy",
        "Tomato Late Blight",
    ]
    return _MODEL, _CLASS_NAMES


# ---------------------------------------------------------------------
# PREPROCESS
# ---------------------------------------------------------------------

def preprocess_image(image_path: str, cfg: dict = CONFIG) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image: {image_path}") from e

    return transform(image).unsqueeze(0)


# ---------------------------------------------------------------------
# TOP-K PREDICTIONS
# ---------------------------------------------------------------------

def _top_k_predictions(
    probs: torch.Tensor,
    class_names: List[str],
    top_k: int,
) -> List[Dict]:
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

    return [
        {
            "label": class_names[idx.item()],
            "confidence": round(prob.item(), 4),
        }
        for prob, idx in zip(top_probs[0], top_indices[0])
    ]


# ---------------------------------------------------------------------
# MAIN PREDICT FUNCTION
# ---------------------------------------------------------------------

def predict(
    image_path: str,
    cfg: dict = CONFIG,
    return_heatmap: bool = False,
) -> Union[List[Dict[str, Union[str, float]]], Dict[str, object]]:

    model, class_names = load_model(cfg)
    device = next(model.parameters()).device
    tensor = preprocess_image(image_path, cfg).to(device)

    # ---------------- NORMAL INFERENCE ----------------
    if not return_heatmap:
        with torch.inference_mode():
            probs = F.softmax(model(tensor), dim=1)
        return _top_k_predictions(probs, class_names, cfg["top_k"])

    # ---------------- GRAD-CAM ----------------
    heatmap = None
    probs = None

    gradcam = GradCAM(model, target_layer=model.features[-1])

    try:
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)

        top_idx = int(probs.argmax(dim=1).item())
        heatmap = gradcam.generate(tensor, class_idx=top_idx)

    except Exception as exc:
        logging.warning(f"[GradCAM] {exc}")
        heatmap = None

        if probs is None:
            with torch.inference_mode():
                probs = F.softmax(model(tensor), dim=1)

    finally:
        gradcam.remove_hooks()

    return {
        "predictions": _top_k_predictions(probs, class_names, cfg["top_k"]),
        "heatmap": heatmap,
    }


# ---------------------------------------------------------------------
# CLI (TEST)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python inference/predictor.py <image_path> [--heatmap]")
        sys.exit(1)

    image_path = sys.argv[1]
    want_heatmap = len(sys.argv) == 3 and sys.argv[2] == "--heatmap"

    results = predict(image_path, return_heatmap=want_heatmap)

    predictions = results["predictions"] if want_heatmap else results
    heatmap = results.get("heatmap") if want_heatmap else None

    print("\nTop-3 Predictions")
    print("-" * 35)

    for i, r in enumerate(predictions, 1):
        print(f"{i}. {r['label']} ({r['confidence']*100:.2f}%)")

    if want_heatmap:
        if heatmap is not None:
            print(f"\nHeatmap: shape={heatmap.shape}")
        else:
            print("\nHeatmap unavailable")