"""
models/evaluate.py — Deterministic post-training evaluation for PlantDiseaseEfficientNet.

"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from typing import Tuple, Dict, List, Any
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

from inference.preprocess import build_inference_transform
from models.efficientnet_model import PlantDiseaseEfficientNet, build_model
from utils.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATA_DIR,
    DEVICE,
    LOG_DIR,
    MODEL_PATH,
    NUM_CLASSES,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED,
    VAL_SPLIT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "evaluate.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Validation split  — must be byte-for-byte identical to train.py's split
# ─────────────────────────────────────────────────────────────────────────────

def _get_val_indices(total: int) -> List[int]:
    
    generator = torch.Generator().manual_seed(SEED)
    perm      = torch.randperm(total, generator=generator).tolist()
    val_n     = int(total * VAL_SPLIT)
    train_n   = total - val_n
    return perm[train_n:]


def build_val_loader(data_dir: Path = DATA_DIR) -> Tuple[DataLoader, List[str]]:
   
    probe = datasets.ImageFolder(str(data_dir))

    if len(probe) == 0:
        raise RuntimeError("Dataset is empty.")

    _validate_class_ordering(probe.classes)

    val_indices = _get_val_indices(len(probe))

    val_ds = datasets.ImageFolder(
        str(data_dir),
        transform=build_inference_transform(),
    )

    loader = DataLoader(
        Subset(val_ds, val_indices),
        batch_size  = BATCH_SIZE,
        shuffle     = False,       
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
    )

    log.info(
        "Val split: %d samples (%.0f%% of %d total) | %d classes",
        len(val_indices), VAL_SPLIT * 100, len(probe), NUM_CLASSES,
    )
    return loader, probe.classes


def _validate_class_ordering(dataset_classes: List[str]) -> None:
    """Raise if filesystem class order diverges from config.CLASS_NAMES."""
    if dataset_classes != CLASS_NAMES:
        pairs = "\n".join(
            f"  [{i}] fs='{fs}'  cfg='{cfg}'"
            for i, (fs, cfg) in enumerate(zip(dataset_classes, CLASS_NAMES))
            if fs != cfg
        )
        raise RuntimeError(
            "Class ordering mismatch between filesystem and config.CLASS_NAMES.\n"
            f"Mismatched indices:\n{pairs}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> Tuple[PlantDiseaseEfficientNet, Dict[str, Any]]:
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: '{path}'")

    log.info("Loading checkpoint: %s", path)
    ckpt: dict = torch.load(path, map_location=DEVICE, weights_only=False)

    required = {"model_state_dict", "class_names", "num_classes", "epoch", "val_f1"}
    missing  = required - ckpt.keys()
    if missing:
        raise KeyError(
            f"Checkpoint is missing required keys: {missing}.\n"
            "Was this checkpoint saved by the corrected train.py?"
        )

    ckpt_classes = ckpt["class_names"]
    if ckpt_classes != CLASS_NAMES:
        raise RuntimeError(
            "Checkpoint class_names do not match config.CLASS_NAMES.\n"
            f"  Checkpoint : {ckpt_classes}\n"
            f"  Config     : {CLASS_NAMES}\n"
            "All predictions would be wrong. Retrain or fix config."
        )

    if ckpt["num_classes"] != NUM_CLASSES:
        raise RuntimeError(
            f"Checkpoint num_classes={ckpt['num_classes']} "
            f"!= config.NUM_CLASSES={NUM_CLASSES}."
        )

    model = build_model(freeze_backbone=False)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=True)

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"state_dict mismatch after load:\n"
            f"  Missing keys    : {result.missing_keys}\n"
            f"  Unexpected keys : {result.unexpected_keys}"
        )

    model.eval()
    log.info(
        "Checkpoint loaded  epoch=%d | val_acc=%.2f%% | val_f1=%.4f",
        ckpt["epoch"], ckpt.get("val_accuracy", float("nan")) * 100, ckpt["val_f1"],
    )
    return model, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# 3. Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:  PlantDiseaseEfficientNet,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    all_labels: List[int]        = []
    all_preds:  List[int]        = []
    all_probs:  List[torch.Tensor] = []

    for images, labels in tqdm(loader, desc="Evaluating", unit="batch"):
        images = images.to(DEVICE, non_blocking=True)

        logits: torch.Tensor = model(images)  
        if torch.isnan(logits).any():
            raise RuntimeError("NaN logits detected.")
                         # (N, C) — raw logits
        probs:  torch.Tensor = F.softmax(logits, dim=1)   
        preds:  torch.Tensor = logits.argmax(dim=1)       
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.append(probs.cpu())

    y_true  = np.array(all_labels,  dtype=np.int64)
    y_pred  = np.array(all_preds,   dtype=np.int64)

    if not all_probs:
        raise RuntimeError("No predictions generated.")
    y_probs = torch.cat(all_probs, dim=0).numpy()         

    return y_true, y_pred, y_probs


# ─────────────────────────────────────────────────────────────────────────────
# 4. Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _top_k_accuracy(y_true: np.ndarray, y_probs: np.ndarray, k: int) -> float:
    """Fraction of samples where the true label is in the top-k predicted classes."""
    top_k = np.argsort(y_probs, axis=1)[:, -k:]           # (N, k) highest prob indices
    return float(np.any(top_k == y_true[:, None], axis=1).mean())


def compute_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_probs: np.ndarray,
) -> Dict[str, Any]:
   
    cm            = confusion_matrix(y_true, y_pred)
    cm_norm       = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    max_probs     = y_probs.max(axis=1)                                
    log_probs     = np.log(y_probs.clip(min=1e-9))
    entropy       = -(y_probs * log_probs).sum(axis=1)               

    return {
        # ── Accuracy ─────────────────────────────────────────────────
        "top1_accuracy":         float((y_true == y_pred).mean()),
        "top3_accuracy":         _top_k_accuracy(y_true, y_probs, k=3),

        # ── Precision / Recall / F1 ──────────────────────────────────
        "weighted_precision":    float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision":       float(precision_score(y_true, y_pred, average="macro",    zero_division=0)),
        "weighted_recall":       float(recall_score(y_true, y_pred,    average="weighted", zero_division=0)),
        "macro_recall":          float(recall_score(y_true, y_pred,    average="macro",    zero_division=0)),
        "weighted_f1":           float(f1_score(y_true, y_pred,        average="weighted", zero_division=0)),
        "macro_f1":              float(f1_score(y_true, y_pred,        average="macro",    zero_division=0)),

        # ── Per-class text report ─────────────────────────────────────
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, zero_division=0, digits=4,
        ),

        # ── Confusion matrices ────────────────────────────────────────
        "confusion_matrix":            cm,
        "confusion_matrix_normalized": cm_norm,

        # ── Confidence / calibration ─────────────────────────────────
        "mean_confidence":   float(max_probs.mean()),
        "std_confidence":    float(max_probs.std()),
        "mean_entropy":      float(entropy.mean()),
        "std_entropy":       float(entropy.std()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def _format_confusion_matrix(cm: np.ndarray, cm_norm: np.ndarray) -> str:
   
    n   = len(CLASS_NAMES)
    col = 7                      # column width for counts
    sep = "─" * (col * n + 6)

    lines = [
        "",
        "╔══════════════════════════════╗",
        "║       CONFUSION MATRIX       ║",
        "╚══════════════════════════════╝",
        "",
        "  (rows = true class  |  cols = predicted class)",
        "",
        "        " + "".join(f"{i:>{col}}" for i in range(n)),
        "        " + sep,
    ]
    for i, row in enumerate(cm):
        lines.append(f"  [{i:>2}]  " + "".join(f"{v:>{col}}" for v in row))

    lines += [
        "",
        "  Normalised (recall per true class):",
        "        " + "".join(f"{i:>{col}}" for i in range(n)),
        "        " + sep,
    ]
    for i, row in enumerate(cm_norm):
        lines.append(f"  [{i:>2}]  " + "".join(f"{v:>{col}.2f}" for v in row))

    lines += ["", "  Legend:"]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"    [{i}] {name}")

    return "\n".join(lines)


def _format_summary(m: Dict[str, Any], ckpt_meta: Dict[str, Any]) -> str:
    """Headline metric block — one screen, immediately scannable."""
    return "\n".join([
        "",
        "╔══════════════════════════════════════════════╗",
        "║          EVALUATION SUMMARY                  ║",
        "╚══════════════════════════════════════════════╝",
        f"  Checkpoint epoch     : {ckpt_meta.get('epoch', '?')}",
        f"  Train val_f1         : {ckpt_meta.get('val_f1', float('nan')):.4f}",
        "  ─────────────────────────────────────────────",
        f"  Top-1 Accuracy       : {m['top1_accuracy']:.4f}  ({m['top1_accuracy']*100:.2f}%)",
        f"  Top-3 Accuracy       : {m['top3_accuracy']:.4f}  ({m['top3_accuracy']*100:.2f}%)",
        "  ─────────────────────────────────────────────",
        f"  Weighted Precision   : {m['weighted_precision']:.4f}",
        f"  Macro    Precision   : {m['macro_precision']:.4f}",
        f"  Weighted Recall      : {m['weighted_recall']:.4f}",
        f"  Macro    Recall      : {m['macro_recall']:.4f}",
        f"  Weighted F1          : {m['weighted_f1']:.4f}",
        f"  Macro    F1          : {m['macro_f1']:.4f}",
        "  ─────────────────────────────────────────────",
        f"  Mean Confidence      : {m['mean_confidence']:.4f} ± {m['std_confidence']:.4f}",
        f"  Mean Entropy         : {m['mean_entropy']:.4f} ± {m['std_entropy']:.4f}",
        "",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 6. Persist results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    metrics: Dict[str, Any],
    ckpt_meta: Dict[str, Any],
    out_path: Path,
) -> None:
    
    serialisable = {
        "checkpoint_epoch":        ckpt_meta.get("epoch"),
        "checkpoint_val_f1":       ckpt_meta.get("val_f1"),
        "checkpoint_val_accuracy": ckpt_meta.get("val_accuracy"),
        "class_names":             CLASS_NAMES,
        "top1_accuracy":           metrics["top1_accuracy"],
        "top3_accuracy":           metrics["top3_accuracy"],
        "weighted_precision":      metrics["weighted_precision"],
        "macro_precision":         metrics["macro_precision"],
        "weighted_recall":         metrics["weighted_recall"],
        "macro_recall":            metrics["macro_recall"],
        "weighted_f1":             metrics["weighted_f1"],
        "macro_f1":                metrics["macro_f1"],
        "mean_confidence":         metrics["mean_confidence"],
        "std_confidence":          metrics["std_confidence"],
        "mean_entropy":            metrics["mean_entropy"],
        "std_entropy":             metrics["std_entropy"],
        "confusion_matrix":        metrics["confusion_matrix"].tolist(),
        "confusion_matrix_normalized": metrics["confusion_matrix_normalized"].tolist(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(serialisable, indent=2))
    log.info("Results saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(ckpt_path: Path = MODEL_PATH) -> Dict[str, Any]:
    
    log.info("Device: %s", DEVICE)

    val_loader, _ = build_val_loader()
    model, ckpt   = load_checkpoint(ckpt_path)

    y_true, y_pred, y_probs = run_inference(model, val_loader)
    metrics                 = compute_metrics(y_true, y_pred, y_probs)

    log.info(_format_summary(metrics, ckpt))
    log.info("\n%s", _format_confusion_matrix(
        metrics["confusion_matrix"],
        metrics["confusion_matrix_normalized"],
    ))
    log.info("\n%s", metrics["classification_report"])

    results_path = LOG_DIR / "eval_results.json"
    save_results(metrics, ckpt, results_path)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PlantDiseaseEfficientNet.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=MODEL_PATH,
        help=f"Path to checkpoint file (default: {MODEL_PATH})",
    )
    args = parser.parse_args()
    evaluate(ckpt_path=args.ckpt)