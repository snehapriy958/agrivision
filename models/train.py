import os
import sys

# ---------------- FIX IMPORT PATH ----------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from models.cnn_baseline import CNNModel


# ---------------- CONFIG ----------------
CONFIG = {
    "data_dir": "data/raw",
    "batch_size": 32,
    "epochs": 10,
    "lr": 1e-3,
    "image_size": 224,
    "num_classes": 9,
    "checkpoint_path": "models/best_model.pth",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- TRAIN FUNCTION ----------------
def train():

    # ---------------- TRANSFORMS ----------------
    train_transforms = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.RandomResizedCrop(CONFIG["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ---------------- DATA ----------------
    full_dataset = datasets.ImageFolder(CONFIG["data_dir"])

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Assign transforms
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0  # ✅ FIX for Windows
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0
    )

    # ---------------- MODEL ----------------
    model = CNNModel(num_classes=CONFIG["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # ---------------- TRAIN LOOP ----------------
    best_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # ---------------- VALIDATION ----------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total

        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

        # ---------------- SAVE BEST MODEL ----------------
        if acc > best_acc:
            best_acc = acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), CONFIG["checkpoint_path"])
            print("✅ Best model saved")

    print(f"\n🎯 Final Best Accuracy: {best_acc:.4f}")


# ---------------- MAIN ENTRY ----------------
if __name__ == "__main__":
    train()