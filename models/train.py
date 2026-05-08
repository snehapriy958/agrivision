import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from models.efficientnet_model import PlantDiseaseResNet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "data_dir"        : "data/raw",
    "num_classes"     : 9,
    "batch_size"      : 32,
    "epochs"          : 30,
    "lr"              : 1e-4,
    "val_split"       : 0.2,
    "seed"            : 42,
    "patience"        : 3,
    "checkpoint_path" : "models/best_model.pth",
    "num_workers"     : 0,
}

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Data — split indices first, then apply different transforms via Subset
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Splits dataset into train / val using index-based Subset so that
    train and val images never share the same augmented copy.
    Each subset gets its own transform — no data leakage.
    """
    # Load once just to get total length and class names
    full_dataset = datasets.ImageFolder(cfg["data_dir"])
    total        = len(full_dataset)
    val_size     = int(total * cfg["val_split"])
    train_size   = total - val_size

    generator = torch.Generator().manual_seed(cfg["seed"])
    indices = list(range(total))
    train_subset, val_subset = random_split(
        indices,
        [train_size, val_size],
        generator=generator,
    )
    train_indices = train_subset.indices
    val_indices   = val_subset.indices

    # Re-instantiate with correct transforms per split
    train_dataset = datasets.ImageFolder(cfg["data_dir"], transform=TRAIN_TRANSFORMS)
    val_dataset   = datasets.ImageFolder(cfg["data_dir"], transform=VAL_TRANSFORMS)
    
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
    )

    class_names = full_dataset.classes
    print(f"Classes          : {class_names}")
    print(f"Dataset          : {total:,} samples  "
          f"(train {train_size:,} | val {val_size:,})")
    return train_loader, val_loader, class_names


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer,
    device    : torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model  : nn.Module,
    loader : DataLoader,
    device : torch.device,
) -> float:
    model.eval()
    correct = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correct += (model(images).argmax(dim=1) == labels).sum().item()

    return correct / len(loader.dataset)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    model    : nn.Module,
    path     : str,
    epoch    : int,
    val_acc  : float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict = CONFIG) -> None:
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device           : {device}")

    train_loader, val_loader, _ = build_dataloaders(cfg)

    model = PlantDiseaseResNet(num_classes=cfg["num_classes"]).to(device)
    print(f"Trainable params : {model.count_parameters():,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max", 
        patience=2,
        factor=0.3,
    )

    best_val_acc    = 0.0
    epochs_no_improve = 0

    header = f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Acc':>9}  {'Status'}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc    = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        improved = val_acc > best_val_acc
        status   = "✓ saved" if improved else ""
        status = "✓ saved" if improved else f"no improve ({epochs_no_improve})"
        print(f"{epoch:>6}  {train_loss:>11.4f}  {val_acc:>8.2%}  {status}")

        if improved:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(model, cfg["checkpoint_path"], epoch, val_acc)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                print(f"\nEarly stopping triggered (no improvement for {cfg['patience']} epochs).")
                break

    print(f"\nTraining complete.  Best val accuracy : {best_val_acc:.2%}")
    print(f"Checkpoint saved to                   : {cfg['checkpoint_path']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()