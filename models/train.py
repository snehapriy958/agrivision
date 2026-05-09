"""
train.py — Two-stage EfficientNet-B0 training pipeline for plant disease classification.

"""

import logging
import random
from collections import Counter
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from models.efficientnet_model import PlantDiseaseEfficientNet, build_model
from utils.config import (
    VAL_SPLIT,
    BATCH_SIZE,
    CLASS_NAMES,
    DATA_DIR,
    DEVICE,
    EPOCHS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    LEARNING_RATE,
    LOG_DIR,
    MODEL_PATH,
    NUM_CLASSES,
    NUM_WORKERS,
    PIN_MEMORY,
    RESIZE_SIZE,
    SEED,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "train.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Training-specific constants  (not hyper-parameters → not in config.py)
# ─────────────────────────────────────────────────────────────────────────────

PATIENCE:        int   = 5        
GRAD_CLIP:       float = 1.0      
UNFREEZE_EPOCH:  int   = 10       
STAGE2_LR_SCALE: float = 0.1     
WEIGHT_DECAY:    float = 1e-4    

_USE_AMP: bool = DEVICE == "cuda"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = SEED) -> None:
   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _worker_init_fn(worker_id: int) -> None:
    """Per-worker seed so augmentations are reproducible across runs."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transforms
# ─────────────────────────────────────────────────────────────────────────────

_TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
])

_VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(RESIZE_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
])


# ─────────────────────────────────────────────────────────────────────────────
# 3. DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

def _validate_class_ordering(dataset_classes: List[str]) -> None:
    
    if dataset_classes != CLASS_NAMES:
        mismatch = [
            f"  [{i}] dataset='{d}'  config='{c}'"
            for i, (d, c) in enumerate(
                zip(dataset_classes, CLASS_NAMES + ["<missing>"] * max(0, len(dataset_classes) - len(CLASS_NAMES)))
            )
            if d != c
        ]
        raise RuntimeError(
            "Class ordering mismatch between ImageFolder and config.CLASS_NAMES.\n"
            "This would corrupt every label assignment.\n"
            "Mismatched indices:\n" + "\n".join(mismatch)
        )


def build_dataloaders() -> Tuple[DataLoader, DataLoader]:
   
    probe = datasets.ImageFolder(str(DATA_DIR))
    if len(probe) == 0:
        raise RuntimeError("Dataset is empty.")
    
    _validate_class_ordering(probe.classes)

    n       = len(probe)
    val_n   = int(n * VAL_SPLIT)
    train_n = n - val_n

    generator   = torch.Generator().manual_seed(SEED)
    perm        = torch.randperm(n, generator=generator).tolist()
    train_idx   = perm[:train_n]
    val_idx     = perm[train_n:]

    train_ds = datasets.ImageFolder(str(DATA_DIR), transform=_TRAIN_TRANSFORMS)
    val_ds   = datasets.ImageFolder(str(DATA_DIR), transform=_VAL_TRANSFORMS)

    # ── WeightedRandomSampler ────────────────────────────────────────
    train_targets  = [probe.targets[i] for i in train_idx]
    class_counts   = Counter(train_targets)
    sample_weights = [1.0 / class_counts[t] for t in train_targets]
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
        generator   = generator,
    )

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size      = BATCH_SIZE,
        sampler         = sampler,       
        num_workers     = NUM_WORKERS,
        pin_memory      = PIN_MEMORY,
        worker_init_fn  = _worker_init_fn,
        persistent_workers = NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size      = BATCH_SIZE,
        shuffle         = False,
        num_workers     = NUM_WORKERS,
        pin_memory      = PIN_MEMORY,
        worker_init_fn  = _worker_init_fn,
        persistent_workers = NUM_WORKERS > 0,
    )

    log.info(
        "Dataset: %d samples (train %d | val %d) across %d classes.",
        n, train_n, val_n, NUM_CLASSES,
    )
    log.info("Class distribution (train): %s", dict(sorted(class_counts.items())))
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# 4. Optimizer and Scheduler factories
# ─────────────────────────────────────────────────────────────────────────────

def _build_optimizer(model: PlantDiseaseEfficientNet, stage: int) -> optim.Optimizer:
    
    if stage == 1:
        trainable = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    return optim.AdamW(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr":     LEARNING_RATE * STAGE2_LR_SCALE,
            },
            {
                "params": model.classifier.parameters(),
                "lr":     LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )


def _build_scheduler(optimizer: optim.Optimizer) -> optim.lr_scheduler.ReduceLROnPlateau:
    
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "max",   # maximise F1
        patience = 2,
        factor   = 0.3,
        min_lr   = 1e-7,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Train / Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     PlantDiseaseEfficientNet,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler:    torch.cuda.amp.GradScaler,
) -> float:
    """Return mean cross-entropy loss over the epoch."""
    model.train()
    running_loss = 0.0

    bar = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for images, labels in bar:
        if images.size(0) == 0:
            continue
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  

        with torch.autocast(device_type="cuda", enabled=_USE_AMP):
            loss = criterion(model(images), labels)

            if torch.isnan(loss).any():
                raise RuntimeError("NaN loss detected.")


        scaler.scale(loss).backward()
        # Unscale before clip so the norm is on the real gradient magnitudes.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model:  PlantDiseaseEfficientNet,
    loader: DataLoader,
) -> Dict[str, object]:
    
    model.eval()
    all_preds:  List[int] = []
    all_labels: List[int] = []

    for images, labels in tqdm(loader, desc="    Val", leave=False, unit="batch"):
        images = images.to(DEVICE, non_blocking=True)
        preds  = model(images).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "f1":       float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "report":   classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0),
        "cm":       confusion_matrix(y_true, y_pred),
        "preds":    y_pred,
        "labels":   y_true,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:      PlantDiseaseEfficientNet,
    optimizer:  optim.Optimizer,
    scheduler:  optim.lr_scheduler.ReduceLROnPlateau,
    epoch:      int,
    val_acc:    float,
    val_f1:     float,
) -> None:
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_accuracy":         val_acc,
            "val_f1":               val_f1,
            "class_names":          CLASS_NAMES,
            "num_classes":          NUM_CLASSES,
        },
        MODEL_PATH,

    )

    if not MODEL_PATH.exists():
        raise RuntimeError("Checkpoint save failed.")


    log.info("Checkpoint saved → %s  (epoch %d | acc %.2f%% | F1 %.4f)",
             MODEL_PATH, epoch, val_acc * 100, val_f1)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train() -> None:
    seed_everything(SEED)
    log.info("Device: %s  |  AMP: %s  |  Seed: %d", DEVICE, _USE_AMP, SEED)

    train_loader, val_loader = build_dataloaders()

    # ── Stage 1: head-only training ───────────────────────────────────
    model     = build_model(freeze_backbone=True)
    optimizer = _build_optimizer(model, stage=1)
    scheduler = _build_scheduler(optimizer)
    scaler    = torch.cuda.amp.GradScaler(enabled=_USE_AMP)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    counts = model.count_parameters()
    log.info(
        "Model: PlantDiseaseEfficientNet  |  "
        "trainable %s  |  frozen %s  |  total %s",
        f"{counts['trainable']:,}", f"{counts['frozen']:,}", f"{counts['total']:,}",
    )

    best_val_f1       = 0.0
    best_val_acc      = 0.0
    epochs_no_improve = 0
    stage             = 1

    header = (
        f"\n{'Epoch':>6} {'Stage':>6} {'Loss':>10} "
        f"{'Val Acc':>9} {'Val F1':>8} {'LR (head)':>12} {'Status'}"
    )
    log.info(header)
    log.info("─" * len(header))

    for epoch in range(1, EPOCHS + 1):

        # ── Stage transition ─────────────────────────────────────────
        if epoch == UNFREEZE_EPOCH + 1 and stage == 1:
            stage = 2
            model.unfreeze_backbone(num_blocks=3)
            optimizer = _build_optimizer(model, stage=2)
            scheduler = _build_scheduler(optimizer)
            counts    = model.count_parameters()
            log.info(
                "\n⟶  Stage 2: unfroze last 3 backbone blocks.  "
                "Trainable params now: %s", f"{counts['trainable']:,}"
            )

        # ── Forward / backward ───────────────────────────────────────
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        metrics    = evaluate(model, val_loader)
        val_acc    = metrics["accuracy"]
        val_f1     = metrics["f1"]

        head_lr = optimizer.param_groups[-1]["lr"]
        scheduler.step(val_f1)

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1  = val_f1
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc, val_f1)
            status = "✓ saved"
        else:
            epochs_no_improve += 1
            status = f"–{epochs_no_improve}/{PATIENCE}"

        log.info(
            "%6d %6d %10.4f %8.2f%% %8.4f %12.2e  %s",
            epoch, stage, train_loss, val_acc * 100, val_f1, head_lr, status,
        )

        if epoch % 5 == 0 or improved:
            log.info("\n%s", metrics["report"])

        if epochs_no_improve >= PATIENCE:
            log.info(
                "\nEarly stopping: no F1 improvement for %d consecutive epochs.", PATIENCE
            )
            break

    log.info(
        "\nTraining complete.  "
        "Best val accuracy: %.2f%%  |  Best val F1: %.4f",
        best_val_acc * 100, best_val_f1,
    )
    log.info("Checkpoint: %s", MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()