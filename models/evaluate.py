"""
models/evaluate.py — Evaluation Script for CNN Image Classifier
Dependencies : torch, numpy, scikit-learn
Usage        : python models/evaluate.py
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report

from models.cnn_baseline import CNNModel
from inference.preprocess import PlantDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "data_root"       : "data/raw",
    "num_classes"     : 9,
    "val_split"       : 0.2,
    "seed"            : 42,
    "batch_size"      : 32,
    "checkpoint_path" : "best_model.pth",
    "device"          : "cpu",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(cfg: dict) -> tuple[DataLoader, list[str]]:
    dataset    = PlantDataset(root_dir=cfg["data_root"])
    total      = len(dataset)
    val_size   = int(total * cfg["val_split"])
    train_size = total - val_size

    _, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    class_names = dataset.class_names
    return val_loader, class_names


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(cfg: dict) -> torch.nn.Module:
    device     = torch.device(cfg["device"])
    model      = CNNModel(num_classes=cfg["num_classes"])
    checkpoint = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(cfg: dict) -> None:
    device                 = torch.device(cfg["device"])
    val_loader, class_names = load_data(cfg)
    model                  = load_model(cfg)

    all_preds : list[int] = []
    all_labels: list[int] = []

    for images, labels in val_loader:
        images = images.to(device)
        preds  = model(images).argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

    _print_results(all_labels, all_preds, class_names)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_results(
    all_labels : list[int],
    all_preds  : list[int],
    class_names: list[str],
) -> None:
    cm     = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)

    col_width  = 8
    label_width = max(len(n) for n in class_names) + 2

    print("\n" + "=" * 60)
    print(" CONFUSION MATRIX")
    print("=" * 60)

    header = " " * label_width + "".join(
        f"{name:>{col_width}}" for name in class_names
    )
    print(header)
    print(" " * label_width + "-" * (col_width * len(class_names)))

    for i, row in enumerate(cm):
        row_label = f"{class_names[i]:<{label_width}}"
        row_vals  = "".join(f"{val:>{col_width}}" for val in row)
        print(row_label + row_vals)

    print("\n" + "=" * 60)
    print(" CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate(CONFIG)