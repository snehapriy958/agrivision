"""
inference/predictor.py — Single-image inference for PlantDiseaseResNet
"""

import os
import sys
import logging

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_baseline import PlantDiseaseResNet
from utils.gradcam import GradCAM

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Model download config
# ---------------------------------------------------------------------------

MODEL_PATH = "models/best_model.pth"
MODEL_URL  = "https://drive.google.com/uc?id=1JKlHmyCNOrVSIlfyHtSADj2u2BnotDFv"


def _ensure_model_downloaded() -> None:
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        print("Downloading model weights…")
        try:
            import gdown
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        except Exception as exc:
            raise RuntimeError("Model download failed.") from exc

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'.")


_ensure_model_downloaded()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "checkpoint_path" : MODEL_PATH,
    "num_classes"     : 9,
    "top_k"           : 3,
    "image_size"      : 224,
    "mean"            : [0.485, 0.456, 0.406],
    "std"             : [0.229, 0.224, 0.225],
}

# Fixed class names — no dataset dependency
CLASS_NAMES: list[str] = [
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

# ---------------------------------------------------------------------------
# Deterministic inference transforms — NO random ops
# ---------------------------------------------------------------------------

INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(CONFIG["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIG["mean"], std=CONFIG["std"]),
])

# ---------------------------------------------------------------------------
# Global model cache
# ---------------------------------------------------------------------------

_MODEL       : Optional[PlantDiseaseResNet] = None
_DEVICE      : Optional[torch.device]       = None


def load_model(cfg: dict = CONFIG) -> tuple[PlantDiseaseResNet, torch.device]:
    global _MODEL, _DEVICE

    if _MODEL is not None:
        return _MODEL, _DEVICE

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(cfg["checkpoint_path"], map_location=device)

    model = PlantDiseaseResNet(num_classes=cfg["num_classes"])

    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    dummy = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        model(dummy)

    _MODEL  = model
    _DEVICE = device
    return _MODEL, _DEVICE


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str) -> torch.Tensor:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Invalid or corrupted image : '{image_path}'") from exc

    return INFERENCE_TRANSFORMS(image).unsqueeze(0)   # (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _top_k_predictions(
    probs       : torch.Tensor,
    class_names : list[str],
    top_k       : int,
) -> list[dict]:
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)
    return [
        {
            "label"      : class_names[idx.item()],
            "confidence" : round(prob.item(), 4),
        }
        for prob, idx in zip(top_probs[0], top_indices[0])
    ]


def _compute_entropy(probs: torch.Tensor) -> float:
    """Shannon entropy (nats) over the full probability distribution."""
    p = probs.squeeze(0).clamp(min=1e-9)
    return round(float(-torch.sum(p * torch.log(p)).item()), 6)


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------

def predict(
    image_path    : str,
    cfg           : dict = CONFIG,
    return_heatmap: bool = False,
) -> dict:
    """
    Run inference on a single image.

    Args:
        image_path     : Path to the input image.
        cfg            : Configuration dict.
        return_heatmap : When True, run Grad-CAM and include heatmap in output.

    Returns:
        {
            "predictions" : [{"label": str, "confidence": float}, ...],
            "entropy"     : float,              # uncertainty measure
            "heatmap"     : object | None,  # only when return_heatmap=True
        }
    """
    model, device = load_model(cfg)
    tensor        = preprocess_image(image_path).to(device)

    # ── Standard inference path ────────────────────────────────────────
    if not return_heatmap:
        with torch.inference_mode():
            probs = F.softmax(model(tensor), dim=1)

        return {
            "predictions" : _top_k_predictions(probs, CLASS_NAMES, cfg["top_k"]),
            "entropy"     : _compute_entropy(probs),
            "heatmap"     : None,
        }

    # ── Grad-CAM path — gradients required ────────────────────────────
    heatmap : Optional[object] = None
    probs   : Optional[torch.Tensor] = None

    # Target layer: last ResNet block via helper method (last ResNet block before avgpool)
    gradcam = GradCAM(model, target_layer=model.get_gradcam_target_layer())

    try:
        with torch.enable_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)


            top_idx = int(probs.argmax(dim=1).item())
            heatmap = gradcam.generate(tensor, class_idx=top_idx)

    except Exception as exc:
        logging.warning(f"[GradCAM] Heatmap generation failed — {exc}")
        heatmap = None

        if probs is None:
            with torch.inference_mode():
                probs = F.softmax(model(tensor), dim=1)

    finally:
        gradcam.remove_hooks()

    return {
        "predictions" : _top_k_predictions(probs, CLASS_NAMES, cfg["top_k"]),
        "entropy"     : _compute_entropy(probs),
        "heatmap"     : heatmap,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python inference/predictor.py <image_path> [--heatmap]")
        sys.exit(1)

    _path         = sys.argv[1]
    _want_heatmap = len(sys.argv) == 3 and sys.argv[2] == "--heatmap"
    _result       = predict(_path, return_heatmap=_want_heatmap)

    print("\nTop-3 Predictions")
    print("-" * 35)
    for i, r in enumerate(_result["predictions"], 1):
        print(f"  {i}. {r['label']:<30} {r['confidence'] * 100:>6.2f}%")
    print(f"\nEntropy : {_result['entropy']}")

    if _want_heatmap:
        hm = _result["heatmap"]
        print(f"Heatmap : {f'shape={hm.shape}  min={hm.min():.4f}  max={hm.max():.4f}' if hm is not None else 'unavailable'}")
    print()