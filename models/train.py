# import torch
# from models.cnn_baseline import CNNModel 

# model = CNNModel(num_classes=9)
# model.eval()

# x = torch.randn(1, 3, 224, 224)

# with torch.no_grad():
#     y = model(x)

# print("Output shape:", y.shape)

"""
Training Script — Image Classification
Assumes:
    - PlantDataset  : custom Dataset returning (image, label) pairs
    - CNNModel : model defined in cnn_baseline.py
Usage:
    python train.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.cnn_baseline import CNNModel
from inference.preprocess import PlantDataset          # adjust import path as needed




# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "data_root"        : "data/raw",          # root folder passed to PlantDataset
    "num_classes"      : 9,
    "image_size"       : 224,
    "batch_size"       : 32,
    "num_epochs"       : 30,
    "learning_rate"    : 1e-3,
    "weight_decay"     : 1e-4,
    "dropout"          : 0.5,
    "val_split"        : 0.2,              # 20 % held out for validation
    "num_workers"      : 0,
    "checkpoint_path"  : "best_model.pth",
    "seed"             : 42,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Split dataset and return (train_loader, val_loader)."""
    dataset    = PlantDataset(root_dir=cfg["data_root"])
    print("Classes:", dataset.class_names)
    total      = len(dataset)
    val_size   = int(total * cfg["val_split"])
    train_size = total - val_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=False,
    )

    print(f"Dataset      : {total:,} samples  "
          f"(train {train_size:,} | val {val_size:,})")
    return train_loader, val_loader

    

def save_checkpoint(model: nn.Module, path: str, epoch: int, val_acc: float) -> None:
    torch.save(
        {
            "epoch"    : epoch,
            "val_acc"  : val_acc,
            "state_dict": model.state_dict(),
        },
        path,
    )


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------



def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer,
    device    : torch.device,
) -> float:
    """Returns average training loss for the epoch."""
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model    : nn.Module,
    loader   : DataLoader,
    device   : torch.device,
) -> float:
    """Returns validation accuracy (0–100)."""
    model.eval()
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds   = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()

    return 100.0 * correct / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------



def train(cfg: dict) -> None:
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device       : {device}")

    # Data
    train_loader, val_loader = build_dataloaders(cfg)

    # Model
    model = CNNModel(
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
    ).to(device)
    print(f"Parameters   : {model.count_parameters():,}\n")

    # Loss, optimiser, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_acc = 0.0
    header = f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Acc (%)':>11}  {'LR':>10}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, cfg["num_epochs"] + 1):

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc    = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6}  {train_loss:>11.4f}  {val_acc:>10.2f}%  {current_lr:>10.2e}")

        # Checkpoint — save whenever validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, cfg["checkpoint_path"], epoch, val_acc)
            print(f"         ↳ best model saved  (val_acc={val_acc:.2f}%)")

    print(f"\nTraining complete. Best val accuracy : {best_val_acc:.2f}%")
    print(f"Checkpoint saved to                  : {cfg['checkpoint_path']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(CONFIG)