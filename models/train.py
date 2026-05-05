import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from typing import Tuple

from models.cnn_baseline import PlantDiseaseResNet
from inference.preprocess import PlantDataset


CONFIG = {
    "data_root": "data/raw",
    "num_classes": 9,
    "batch_size": 32,
    "num_epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "val_split": 0.2,
    "num_workers": 0,
    "checkpoint_path": "models/best_model.pth",
    "seed": 42,
    "step_size": 5,
    "gamma": 0.3,
    "patience": 5,
}


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    base_dataset = PlantDataset(root_dir=cfg["data_root"])

    total = len(base_dataset)
    val_size = int(total * cfg["val_split"])
    train_size = total - val_size

    generator = torch.Generator().manual_seed(cfg["seed"])

    train_split, val_split = random_split(
        range(total), [train_size, val_size], generator=generator
    )

    train_dataset = PlantDataset(cfg["data_root"], transform=TRAIN_TRANSFORMS)
    val_dataset = PlantDataset(cfg["data_root"], transform=VAL_TRANSFORMS)

    # ✅ Consistent indices (fixes IDE + edge-case issues)
    train_idx = list(train_split.indices)
    val_idx = list(val_split.indices)

    # ✅ Respect config but optimize GPU
    num_workers = cfg["num_workers"]
    if num_workers == 0 and torch.cuda.is_available():
        num_workers = 2
    
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Classes      : {base_dataset.class_names}")
    print(f"Dataset      : {total:,} samples (train {train_size:,} | val {val_size:,})")

    return train_loader, val_loader


# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------

def save_checkpoint(model, optimizer, path, epoch, val_acc):
    torch.save({
        "epoch": epoch,
        "val_acc": val_acc,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


# ---------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # ✅ Stability (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()

    return 100.0 * correct / len(loader.dataset)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def train(cfg: dict) -> None:
    set_seed(cfg["seed"])
    device = get_device()

    # ✅ Safe directory creation
    checkpoint_dir = os.path.dirname(cfg["checkpoint_path"])
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device : {device} ({device_name})")

    train_loader, val_loader = build_dataloaders(cfg)

    model = PlantDiseaseResNet(cfg["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = StepLR(optimizer, cfg["step_size"], cfg["gamma"])
    scaler = GradScaler(enabled=torch.cuda.is_available())

    print(f"Trainable params : {model.count_parameters():,}\n")

    best_val_acc = 0.0
    no_improve_epochs = 0

    for epoch in range(1, cfg["num_epochs"] + 1):

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Val: {val_acc:.2f}% | LR: {lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0

            save_checkpoint(model, optimizer, cfg["checkpoint_path"], epoch, val_acc)
            print(f"         ↳ best model saved (val_acc={val_acc:.2f}%)")

        else:
            no_improve_epochs += 1

            if no_improve_epochs >= cfg["patience"]:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoint saved to: {cfg['checkpoint_path']}")


if __name__ == "__main__":
    train(CONFIG)