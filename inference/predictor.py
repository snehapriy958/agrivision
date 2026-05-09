"""
inference/predictor.py — Production inference engine for PlantDiseaseEfficientNet.

"""

import logging
import threading
from pathlib import Path
from typing import TypedDict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.gradcam import GradCAM
from inference.preprocess import ImageInput, preprocess_image
from models.efficientnet_model import PlantDiseaseEfficientNet, build_model
from utils.config import (
    CLASS_NAMES,
    DEVICE,
    IDX_TO_CLASS,
    MODEL_PATH,
    NUM_CLASSES,
)

log = logging.getLogger(__name__)  


class ClassPrediction(TypedDict):
    rank:        int
    label:       str
    class_index: int
    confidence:  float   # softmax probability, rounded to 4 d.p.


class PredictionResult(TypedDict):
    top_prediction: ClassPrediction          
    top_k:          List[ClassPrediction]     
    entropy:        float                     
    heatmap:        Optional[np.ndarray]        

_CACHE_LOCK: threading.Lock                      = threading.Lock()
_MODEL:      Optional[PlantDiseaseEfficientNet] = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Checkpoint loading & validation
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_checkpoint(path: Path) -> PlantDiseaseEfficientNet:
   
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: '{path}'.\n"
            "Run train.py first, or point MODEL_PATH in config.py to the correct file."
        )

    log.info("Loading checkpoint: %s", path)
    ckpt: dict = torch.load(path, map_location=DEVICE, weights_only=False)

    required = {"model_state_dict", "class_names", "num_classes"}
    missing  = required - set(ckpt.keys())
    if missing:
        raise KeyError(
            f"Checkpoint is missing required keys: {missing}.\n"
            "This checkpoint was not saved by the corrected train.py.\n"
            f"Found keys: {set(ckpt.keys())}"
        )

    ckpt_classes: List[str] = ckpt["class_names"]
    if ckpt_classes != CLASS_NAMES:
        diff = [
            f"  [{i}] ckpt='{c}'  config='{k}'"
            for i, (c, k) in enumerate(zip(ckpt_classes, CLASS_NAMES))
            if c != k
        ]
        raise RuntimeError(
            "Checkpoint class_names ≠ config.CLASS_NAMES.\n"
            "Every prediction would map to the wrong label.\n"
            "Mismatched positions:\n" + "\n".join(diff)
        )

    if ckpt["num_classes"] != NUM_CLASSES:
        raise RuntimeError(
            f"Checkpoint num_classes={ckpt['num_classes']} ≠ "
            f"config.NUM_CLASSES={NUM_CLASSES}.\n"
            "The final Linear layer shape would mismatch. Retrain with current config."
        )

 
    model  = build_model(freeze_backbone=False, device=DEVICE)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=True)

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"state_dict shape mismatch:\n"
            f"  Missing keys    : {result.missing_keys}\n"
            f"  Unexpected keys : {result.unexpected_keys}\n"
            "Do not change strict=False to suppress this — fix the source mismatch."
        )

    model.eval()
    log.info(
        "Checkpoint loaded | epoch=%s | val_f1=%s | val_acc=%s",
        ckpt.get("epoch",        "?"),
        f"{ckpt.get('val_f1',        float('nan')):.4f}",
        f"{ckpt.get('val_accuracy',  float('nan')):.2%}",
    )
    return model


def load_model(path: Path = MODEL_PATH) -> PlantDiseaseEfficientNet:
    
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    with _CACHE_LOCK:
        if _MODEL is not None:         
            return _MODEL
        _MODEL = _load_from_checkpoint(path)

    return _MODEL


# ─────────────────────────────────────────────────────────────────────────────
# 2. Inference utilities
# ─────────────────────────────────────────────────────────────────────────────

def _build_top_k(probs_1d: torch.Tensor, top_k: int) -> List[ClassPrediction]:
    
    k       = min(top_k, NUM_CLASSES)

    if k <= 0:
        raise ValueError("top_k must be greater than 0.")

    vals, idxs = torch.topk(probs_1d, k=k)

    return [
        ClassPrediction(
            rank        = rank + 1,
            label       = IDX_TO_CLASS[idx.item()],
            class_index = idx.item(),
            confidence  = round(val.item(), 4),
        )
        for rank, (val, idx) in enumerate(zip(vals, idxs))
    ]


def _compute_entropy(probs_1d: torch.Tensor) -> float:
   
    p = probs_1d.clamp(min=1e-9)
    return round(float(-(p * p.log()).sum().item()), 6)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main predict entry point
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    source:        ImageInput,
    *,
    top_k:         int  = 3,
    return_gradcam: bool = False,
    ckpt_path:     Path = MODEL_PATH,
) -> PredictionResult:
    
    model = load_model(ckpt_path)

    tensor: torch.Tensor
    pil_image: Image.Image
    tensor, pil_image = preprocess_image(source, return_pil=True)   # type: ignore[misc]
    tensor = tensor.to(DEVICE)
    if torch.isnan(tensor).any():
        raise RuntimeError("Input tensor contains NaN values.")

    if not return_gradcam:
        with torch.inference_mode():
            logits:  torch.Tensor = model(tensor)    
            if torch.isnan(logits).any():
                raise RuntimeError("NaN logits detected.")

            probs:   torch.Tensor = F.softmax(logits, dim=1)[0]     

        top_k_preds = _build_top_k(probs, top_k)
        return PredictionResult(
            top_prediction = top_k_preds[0],
            top_k          = top_k_preds,
            entropy        = _compute_entropy(probs),
            heatmap        = None,
        )

   
    heatmap: Optional[np.ndarray] = None

    target_layer = model.get_gradcam_target_layer()
    gradcam      = GradCAM(model=model, target_layer=target_layer)

    try:
        with torch.enable_grad():
            logits = model(tensor)     
            if torch.isnan(logits).any():
                raise RuntimeError("NaN logits detected.")

            probs  = F.softmax(logits, dim=1)[0]                   

        top_idx: int = int(logits.argmax(dim=1).item())
        heatmap = gradcam.generate(
            tensor    = tensor,
            pil_image = pil_image,
            class_idx = top_idx,
        )

    except Exception as exc:
        log.warning("GradCAM generation failed: %s — returning prediction without heatmap.", exc)
        heatmap = None

        with torch.inference_mode():
            logits = model(tensor)

            if torch.isnan(logits).any():
                raise RuntimeError("NaN logits detected.")
            
            probs = F.softmax(logits, dim=1)[0]

    finally:
        gradcam.remove_hooks()

    top_k_preds = _build_top_k(probs, top_k)
    return PredictionResult(
        top_prediction = top_k_preds[0],
        top_k          = top_k_preds,
        entropy        = _compute_entropy(probs),
        heatmap        = heatmap,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Plant disease single-image inference.")
    parser.add_argument("image",     type=Path, help="Path to input image.")
    parser.add_argument("--top-k",   type=int,  default=3)
    parser.add_argument("--heatmap", action="store_true", help="Generate GradCAM overlay.")
    parser.add_argument("--ckpt",    type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    result = predict(args.image, top_k=args.top_k, return_gradcam=args.heatmap, ckpt_path=args.ckpt)

    print(f"\n{'─'*45}")
    print(f"  Top-{args.top_k} Predictions")
    print(f"{'─'*45}")
    for p in result["top_k"]:
        bar = "█" * int(p["confidence"] * 30)
        print(f"  {p['rank']}. {p['label']:<32} {p['confidence']*100:>6.2f}%  {bar}")
    print(f"\n  Entropy : {result['entropy']:.6f}  ({'uncertain' if result['entropy'] > 1.5 else 'confident'})")

    hm = result["heatmap"]
    if args.heatmap:
        print(f"  Heatmap : "
              f"{'shape=' + str(hm.shape) + f'  min={hm.min()}'+ f'  max={hm.max()}' if hm is not None else 'unavailable'}")
    print()


if __name__ == "__main__":
    _cli()